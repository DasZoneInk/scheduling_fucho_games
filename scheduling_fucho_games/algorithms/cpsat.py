"""algorithms/cpsat.py — CP-SAT (OR-Tools) solver for the scheduling ILP.

Decision variable:
    x[m, d, v] ∈ {0,1}  —  match m assigned to matchday d using venue-slot v

Objective:
    maximize  Σ_{m,d,v}  revenue[v] · x[m,d,v]

Constraints:
    C1  Each match scheduled exactly once:
        Σ_{d,v}  x[m,d,v] = 1   ∀m
    C2  Each team plays exactly once per matchday:
        Σ_{m∋t, v}  x[m,d,v] = 1   ∀t, d
    C3  No venue-slot double-booked per matchday:
        Σ_m  x[m,d,v] ≤ 1   ∀d, v
    C4  Correct number of matches per matchday:
        Σ_{m,v}  x[m,d,v] = K/2   ∀d
    (C5) Matchday date separation ≥ gap_days — enforced by pre-assigned dates.
"""

from __future__ import annotations

import logging
import time as _time
from typing import Any

from ortools.sat.python import cp_model

from ..model import (
    InfeasibilityError,
    ScheduledMatch,
    SolverResult,
    TournamentProblem,
)

logger = logging.getLogger(__name__)


# ── Solution callback ─────────────────────────────────────────────────────────

class _SolutionRecorder(cp_model.CpSolverSolutionCallback):
    """Records up to *keep* improving solutions during search.

    OR-Tools calls ``on_solution_callback`` for each improving incumbent when
    maximising, so solutions arrive in strictly increasing order of objective.
    We keep a rolling window of the last *keep* solutions.
    """

    def __init__(
        self,
        x: dict[tuple[int, int, int], Any],
        problem: TournamentProblem,
        keep: int = 2,
    ) -> None:
        super().__init__()
        self._x = x
        self._p = problem
        self._keep = keep
        # List of (revenue, schedule) in ascending order of objective
        self.solutions: list[tuple[int, list[ScheduledMatch]]] = []

    def on_solution_callback(self) -> None:
        p = self._p
        schedule: list[ScheduledMatch] = []
        revenue = 0
        for m_idx, match in enumerate(p.matches):
            for d_idx, md_date in enumerate(p.matchday_dates):
                for v_idx, vs in enumerate(p.venue_slots):
                    if self.Value(self._x[m_idx, d_idx, v_idx]):
                        schedule.append(
                            ScheduledMatch(
                                match=match,
                                matchday=d_idx,
                                matchday_date=md_date,
                                venue_slot=vs,
                            )
                        )
                        revenue += vs.revenue
                        break  # match found for this (d, v) — skip rest

        self.solutions.append((revenue, schedule))
        if len(self.solutions) > self._keep:
            self.solutions.pop(0)   # evict oldest (lowest revenue)


# ── Public solver entry-point ─────────────────────────────────────────────────

def solve(problem: TournamentProblem, timeout_s: float = 60.0) -> SolverResult:
    """Solve *problem* with CP-SAT and return optimal + second-best solutions.

    Args:
        problem:   A validated :class:`~model.TournamentProblem`.
        timeout_s: Wall-clock time limit for the solver.

    Returns:
        :class:`~model.SolverResult` with status, schedules, and revenues.
    """
    M = len(problem.matches)
    D = len(problem.matchday_dates)
    V = len(problem.venue_slots)
    mpm = problem.matches_per_matchday

    logger.info(
        "CP-SAT | variables: %d × %d × %d = %d | timeout: %.0fs",
        M, D, V, M * D * V, timeout_s,
    )

    model = cp_model.CpModel()

    # ── Decision variables ────────────────────────────────────────────────────
    x: dict[tuple[int, int, int], Any] = {
        (m, d, v): model.new_bool_var(f"x_m{m}_d{d}_v{v}")
        for m in range(M)
        for d in range(D)
        for v in range(V)
    }

    # ── Pre-compute knockout match-to-matchday mapping ─────────────────────────
    # For knockout: matches are pre-assigned to specific matchdays by round.
    # Build a set of allowed matchdays per match for knockout; None = any day.
    ko_match_days: dict[int, set[int]] | None = None
    if problem.format == "knockout" and problem.knockout_rounds:
        ko_match_days = {}
        for rnd in problem.knockout_rounds:
            for mid in rnd.match_ids:
                ko_match_days[mid] = set(rnd.matchday_indices)

    # Build team → set of matchdays where team is expected to play
    team_expected_days: dict[int, set[int]] = {}
    if problem.format == "knockout" and problem.knockout_rounds:
        for rnd in problem.knockout_rounds:
            for mid in rnd.match_ids:
                match_obj = problem.matches[mid]
                for t in (match_obj.home, match_obj.away):
                    if t.id not in team_expected_days:
                        team_expected_days[t.id] = set()
                    team_expected_days[t.id].update(rnd.matchday_indices)

    # C1 — each match scheduled exactly once
    for m in range(M):
        model.add(sum(x[m, d, v] for d in range(D) for v in range(V)) == 1)

    # C1b — Knockout round ordering: restrict matches to their assigned matchdays
    if ko_match_days:
        for m_idx in range(M):
            allowed = ko_match_days.get(m_idx)
            if allowed is not None:
                for d in range(D):
                    if d not in allowed:
                        for v in range(V):
                            model.add(x[m_idx, d, v] == 0)

    # C2 — team plays exactly once per matchday (round_robin) or at most once (knockout)
    for d in range(D):
        for team in problem.teams:
            team_match_idxs = [
                m for m, match in enumerate(problem.matches) if match.involves(team)
            ]
            if not team_match_idxs:
                continue

            if problem.format == "knockout":
                expected = team_expected_days.get(team.id, set())
                if d in expected:
                    model.add(
                        sum(x[m, d, v] for m in team_match_idxs for v in range(V)) <= 1
                    )
                else:
                    # Team has no matches on this day
                    model.add(
                        sum(x[m, d, v] for m in team_match_idxs for v in range(V)) == 0
                    )
            else:
                model.add(
                    sum(x[m, d, v] for m in team_match_idxs for v in range(V)) == 1
                )

    # C3 — no venue-slot double-booked on the same matchday
    for d in range(D):
        for v in range(V):
            model.add(sum(x[m, d, v] for m in range(M)) <= 1)

    # C4 — correct number of matches per matchday
    mpm_map = problem.matches_per_matchday_map
    for d in range(D):
        day_mpm = mpm_map[d] if mpm_map else mpm
        model.add(sum(x[m, d, v] for m in range(M) for v in range(V)) == day_mpm)

    # ── Objective ─────────────────────────────────────────────────────────────
    revenues = [vs.revenue for vs in problem.venue_slots]
    model.maximize(
        sum(
            revenues[v] * x[m, d, v]
            for m in range(M)
            for d in range(D)
            for v in range(V)
        )
    )

    # ── Solve ─────────────────────────────────────────────────────────────────
    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = timeout_s
    solver.parameters.log_search_progress = False

    callback = _SolutionRecorder(x, problem, keep=2)
    t0 = _time.monotonic()
    status_code = solver.solve(model, callback)
    elapsed = _time.monotonic() - t0

    status_name = solver.status_name(status_code)
    logger.info("CP-SAT status: %s | elapsed: %.2fs", status_name, elapsed)

    if status_code in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        solutions = callback.solutions
        if not solutions:
            # Unlikely: solver reported feasible but callback was never called
            logger.warning("Solver returned %s but no solution was recorded.", status_name)
            return SolverResult(
                status=status_name,
                infeasibility_reason="Solver reported feasible but no solution captured.",
                solve_time_s=elapsed,
            )

        best_rev, best_sched = solutions[-1]
        if len(solutions) >= 2:
            second_rev, second_sched = solutions[-2]
        else:
            second_rev, second_sched = 0, None
            logger.info("Only one feasible solution found; no second-best available.")

        logger.info(
            "Best revenue: $%d | Second-best: $%s",
            best_rev,
            second_rev if second_sched else "N/A",
        )
        return SolverResult(
            status=status_name,
            best_solution=best_sched,
            second_best_solution=second_sched,
            best_revenue=best_rev,
            second_best_revenue=second_rev,
            solve_time_s=elapsed,
        )

    # ── Infeasible / timeout ──────────────────────────────────────────────────
    reason = _diagnose_infeasibility(problem, status_name, timeout_s, elapsed)
    logger.error("CP-SAT: %s — %s", status_name, reason)
    return SolverResult(
        status=status_name,
        infeasibility_reason=reason,
        solve_time_s=elapsed,
    )


def _diagnose_infeasibility(
    problem: TournamentProblem,
    status: str,
    timeout_s: float,
    elapsed: float,
) -> str:
    K = len(problem.teams)
    D = K - 1
    mpm = K // 2
    V = len(problem.venue_slots)
    n_dates = len(problem.matchday_dates)
    reasons: list[str] = []

    if status == "UNKNOWN" and elapsed >= timeout_s * 0.99:
        reasons.append(
            f"Search timed out after {elapsed:.1f}s without proving "
            "optimality or infeasibility. Increase --timeout."
        )
    if n_dates < D:
        reasons.append(
            f"Only {n_dates} feasible calendar date(s) for {D} matchday(s) "
            f"(gap ≥ {problem.matchday_gap_days} days)."
        )
    if V < mpm:
        reasons.append(
            f"Only {V} venue-slot(s) but {mpm} simultaneous match(es) required "
            f"per matchday ({K} teams ÷ 2)."
        )

    if not reasons:
        reasons.append(
            f"No feasible assignment found for K={K} teams, {D} matchdays, "
            f"{V} venue-slots. Verify that a valid 1-factorisation exists "
            "and that venue capacity is sufficient."
        )
    return " | ".join(reasons)
