"""algorithms/genetic_algorithm.py — DEAP-based Genetic Algorithm solver.

Chromosome encoding:
    A list of M integers, one per match.
    gene[m] = d * V + v   where d ∈ [0, D), v ∈ [0, V)
    → decode: matchday   d = gene[m] // V
              venue-slot v = gene[m]  % V

Fitness = total_revenue − PENALTY × (venue_conflicts + team_conflicts)
    venue_conflicts: pairs of matches sharing the same (matchday, venue-slot)
    team_conflicts:  per matchday, any team appearing in >1 match

PENALTY >> max possible revenue to guarantee feasibility dominates.

Evolution:
    Selection : Tournament (k=3)
    Crossover : Two-point   (prob=0.7)
    Mutation  : Uniform int (prob=0.2 per individual, indpb=2/M per gene)
    Hall of Fame : size 2  (yields best + second-best)
"""

from __future__ import annotations

import logging
import random
import time as _time
from collections import defaultdict
from typing import Any

from ..model import (
    ScheduledMatch,
    SolverResult,
    TournamentProblem,
)

logger = logging.getLogger(__name__)

# ── GA hyper-parameters (overridable via solve()) ────────────────────────────
_DEFAULT_POP_SIZE = 300
_DEFAULT_N_GEN = 600
_DEFAULT_CX_PB = 0.7
_DEFAULT_MUT_PB = 0.2
_DEFAULT_TOURNAMENT_K = 7
_PENALTY_MULTIPLIER = 100_000   # >> max possible revenue per match


# ── DEAP setup (module-level, lazy) ──────────────────────────────────────────

def _init_deap() -> tuple[Any, Any, Any]:
    """Import DEAP and register creator types; idempotent."""
    try:
        from deap import algorithms, base, creator, tools  # noqa: F401
    except ImportError as exc:
        raise ImportError(
            "DEAP is required for the genetic algorithm solver. "
            "Install it with: uv pip install 'deap>=1.4.4'"
        ) from exc

    if not hasattr(creator, "FitnessMax"):
        creator.create("FitnessMax", base.Fitness, weights=(1.0,))
    if not hasattr(creator, "Individual"):
        creator.create("Individual", list, fitness=creator.FitnessMax)

    return creator, base, tools


# ── Fitness function ──────────────────────────────────────────────────────────

def _evaluate(
    individual: list[int],
    *,
    problem: TournamentProblem,
) -> tuple[float]:
    """Decode chromosome and compute penalised revenue fitness."""
    M = len(problem.matches)
    V = len(problem.venue_slots)
    revenues = [vs.revenue for vs in problem.venue_slots]

    total_revenue = 0
    venue_usage: dict[tuple[int, int], int] = defaultdict(int)   # (d, v) → count
    team_day: dict[tuple[int, int], int] = defaultdict(int)       # (team_id, d) → count

    for m_idx, gene in enumerate(individual):
        d = gene // V
        v = gene % V
        total_revenue += revenues[v]
        venue_usage[(d, v)] += 1
        match = problem.matches[m_idx]
        team_day[(match.home.id, d)] += 1
        team_day[(match.away.id, d)] += 1

    # Venue conflicts: each (d,v) used more than once
    venue_violations = sum(
        max(0, cnt - 1) for cnt in venue_usage.values()
    )
    # Team conflicts: each team plays more than once on a matchday
    team_violations = sum(
        max(0, cnt - 1) for cnt in team_day.values()
    )

    penalty = _PENALTY_MULTIPLIER * (venue_violations + team_violations)
    return (float(total_revenue - penalty),)


# ── Chromosome decoder ────────────────────────────────────────────────────────

def _decode(
    individual: list[int],
    problem: TournamentProblem,
) -> list[ScheduledMatch]:
    """Convert a chromosome to a list of :class:`ScheduledMatch` objects."""
    V = len(problem.venue_slots)
    schedule: list[ScheduledMatch] = []
    for m_idx, gene in enumerate(individual):
        d = gene // V
        v = gene % V
        schedule.append(
            ScheduledMatch(
                match=problem.matches[m_idx],
                matchday=d,
                matchday_date=problem.matchday_dates[d],
                venue_slot=problem.venue_slots[v],
            )
        )
    return schedule


def _revenue_of(individual: list[int], problem: TournamentProblem) -> int:
    V = len(problem.venue_slots)
    revenues = [vs.revenue for vs in problem.venue_slots]
    return sum(revenues[gene % V] for gene in individual)


def _is_feasible(individual: list[int], problem: TournamentProblem) -> bool:
    """True iff the chromosome encodes a constraint-satisfying schedule."""
    V = len(problem.venue_slots)
    venue_usage: dict[tuple[int, int], int] = defaultdict(int)
    team_day: dict[tuple[int, int], int] = defaultdict(int)
    for m_idx, gene in enumerate(individual):
        d, v = gene // V, gene % V
        venue_usage[(d, v)] += 1
        match = problem.matches[m_idx]
        team_day[(match.home.id, d)] += 1
        team_day[(match.away.id, d)] += 1
    return (
        all(cnt <= 1 for cnt in venue_usage.values())
        and all(cnt <= 1 for cnt in team_day.values())
    )


# ── Public solver entry-point ─────────────────────────────────────────────────

def solve(
    problem: TournamentProblem,
    *,
    pop_size: int = _DEFAULT_POP_SIZE,
    n_gen: int = _DEFAULT_N_GEN,
    cx_pb: float = _DEFAULT_CX_PB,
    mut_pb: float = _DEFAULT_MUT_PB,
    tournament_k: int = _DEFAULT_TOURNAMENT_K,
    seed: int = 42,
) -> SolverResult:
    """Run the GA on *problem* and return the best + second-best solutions.

    Args:
        problem:  Validated :class:`~model.TournamentProblem`.
        pop_size: Population size.
        n_gen:    Number of generations.
        cx_pb:    Crossover probability per pair.
        mut_pb:   Mutation probability per individual.
        tournament_k: Selection pressure (tournament size).
        seed:     Random seed for reproducibility.

    Returns:
        :class:`~model.SolverResult`.
    """
    creator, base_module, tools = _init_deap()
    from deap import algorithms

    M = len(problem.matches)
    D = len(problem.matchday_dates)
    V = len(problem.venue_slots)

    if D * V == 0:
        return SolverResult(
            status="INFEASIBLE",
            infeasibility_reason=(
                "Zero feasible (matchday × venue-slot) combinations. "
                "Check date range and field configuration."
            ),
        )

    logger.info(
        "GA | M=%d matches  D=%d matchdays  V=%d venue-slots  "
        "pop=%d  gen=%d  seed=%d",
        M, D, V, pop_size, n_gen, seed,
    )

    random.seed(seed)

    toolbox = base_module.Toolbox()
    toolbox.register("gene", random.randint, 0, D * V - 1)
    toolbox.register(
        "individual",
        tools.initRepeat,
        creator.Individual,
        toolbox.gene,
        n=M,
    )
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    toolbox.register("evaluate", _evaluate, problem=problem)
    toolbox.register("mate", tools.cxTwoPoint)
    toolbox.register(
        "mutate",
        tools.mutUniformInt,
        low=0,
        up=D * V - 1,
        indpb=max(0.05, 2.0 / M),
    )
    toolbox.register("select", tools.selTournament, tournsize=tournament_k)

    hof = tools.HallOfFame(2)
    stats = tools.Statistics(key=lambda ind: ind.fitness.values[0])
    stats.register("max", max)
    stats.register("avg", lambda vals: sum(vals) / len(vals))

    population = toolbox.population(n=pop_size)

    t0 = _time.monotonic()
    population, logbook = algorithms.eaSimple(
        population,
        toolbox,
        cxpb=cx_pb,
        mutpb=mut_pb,
        ngen=n_gen,
        stats=stats,
        halloffame=hof,
        verbose=False,
    )
    elapsed = _time.monotonic() - t0

    logger.info(
        "GA finished in %.2fs | gen=%d | best_fitness=%.0f",
        elapsed,
        n_gen,
        hof[0].fitness.values[0] if hof else float("-inf"),
    )

    if not hof:
        return SolverResult(
            status="NOT_CONVERGED",
            infeasibility_reason="Hall of Fame is empty — population may have collapsed.",
            solve_time_s=elapsed,
        )

    best_ind = hof[0]
    best_feasible = _is_feasible(best_ind, problem)
    best_rev = _revenue_of(best_ind, problem)
    best_sched = _decode(best_ind, problem)

    if not best_feasible:
        reason = _diagnose_convergence(best_ind, problem, n_gen)
        logger.error("GA best solution is infeasible: %s", reason)
        return SolverResult(
            status="NOT_CONVERGED",
            infeasibility_reason=reason,
            best_solution=best_sched,   # Return anyway for inspection
            best_revenue=best_rev,
            solve_time_s=elapsed,
        )

    # Second-best
    second_sched: list[ScheduledMatch] | None = None
    second_rev = 0
    if len(hof) >= 2:
        second_ind = hof[1]
        if _is_feasible(second_ind, problem):
            second_sched = _decode(second_ind, problem)
            second_rev = _revenue_of(second_ind, problem)
        else:
            logger.info("Second HoF individual is infeasible — no second-best returned.")

    logger.info(
        "GA result | best_revenue=$%d | second_best_revenue=$%s | feasible=%s",
        best_rev,
        second_rev if second_sched else "N/A",
        best_feasible,
    )
    return SolverResult(
        status="OPTIMAL" if best_feasible else "NOT_CONVERGED",
        best_solution=best_sched,
        second_best_solution=second_sched,
        best_revenue=best_rev,
        second_best_revenue=second_rev,
        solve_time_s=elapsed,
    )


def _diagnose_convergence(
    individual: list[int],
    problem: TournamentProblem,
    n_gen: int,
) -> str:
    V = len(problem.venue_slots)
    venue_usage: dict[tuple[int, int], int] = defaultdict(int)
    team_day: dict[tuple[int, int], int] = defaultdict(int)
    for m_idx, gene in enumerate(individual):
        d, v = gene // V, gene % V
        venue_usage[(d, v)] += 1
        match = problem.matches[m_idx]
        team_day[(match.home.id, d)] += 1
        team_day[(match.away.id, d)] += 1

    venue_viol = sum(max(0, c - 1) for c in venue_usage.values())
    team_viol = sum(max(0, c - 1) for c in team_day.values())
    parts = [f"GA did not converge after {n_gen} generations."]
    if venue_viol:
        parts.append(f"Venue-slot conflicts remaining: {venue_viol}.")
    if team_viol:
        parts.append(f"Team double-booking conflicts remaining: {team_viol}.")
    parts.append(
        "Try increasing pop_size / n_gen, or check that enough venue-slots "
        "exist for simultaneous matches."
    )
    return " ".join(parts)
