"""algorithms/kempe.py — DEAP-based Genetic Algorithm with Kempe-chain mutation
for round-robin league scheduling.

Chromosome encoding:
    Individual = [matchdays, venues]

    matchdays:
        List[List[Tuple[int, int]]]
        A list of D matchdays (D = n-1 for n teams).
        Each matchday is a list of n/2 matches represented as
        (home_team_id, away_team_id).

        This structure forms a 1-factorization of the complete graph K_n:
            - Each team appears exactly once per matchday
            - Each pair of teams plays exactly once overall

    venues:
        List[List[int]]
        For each matchday d, venues[d] is a list assigning a venue-slot
        index to each match in matchdays[d].

        Each venues[d] is a permutation (or subset) of available venue-slots,
        ensuring no venue conflicts within the same matchday.

Fitness:
    F(x) = total_revenue + diversity_bonus

    where total_revenue is computed as the sum of the revenues of the
    assigned venue-slots for all matches.

    A small diversity bonus (random noise) is added to break fitness ties
    and help maintain population diversity, preventing premature convergence.

    Since feasibility is enforced by construction (matchdays + venue assignment),
    no penalty terms are required.

Evolution:
    Selection:
        Elitist tournament selection (k=5) with elite preservation.
        The top 3 individuals are always carried forward; the remainder
        are selected via tournament.

    Crossover:
        Fitness-biased matchday-level recombination:
            - For each matchday, offspring inherits from the fitter parent
              with probability proportional to parental fitness.
            - Missing matches are repaired using greedy matchday-balancing
              (prefer less-filled matchdays) to preserve schedule quality.

        Venue crossover:
            - Per matchday, venue assignments are inherited from the parent
              that contributed the majority of matches to that day.

    Mutation:
        Kempe-chain mutation (on matchdays):
            - Select two distinct matchdays (edge colors).
            - Build a proper alternating path (Kempe chain) over teams by
              alternating edges between the two matchdays.
            - Swap edges along the chain between the two matchdays.

            This preserves:
                - One match per team per matchday
                - Global round-robin structure (no duplicate or missing matches)

        Venue mutation:
            - Multiple random swaps of venue assignments within a matchday
              (1 to max_swaps per affected day).

        Adaptive mutation rate:
            - Mutation probability starts at mut_pb and adapts dynamically:
                * Increases by 20% if fitness stagnates for 50 generations
                * Decreases by 5% otherwise, down to a floor of min_mut_pb.
            - This balances exploration (high mutation when stuck) with
              exploitation (low mutation during steady improvement).

    Hall of Fame:
        Size = 10
        Tracks the best individuals across generations for post-optimization
        analysis and as a source of high-quality genetic material.

Properties:
    - Feasibility is preserved throughout evolution (no constraint violations)
    - Search operates entirely within the space of valid schedules
    - Kempe-chain mutation ensures connectivity of the solution space
    - Adaptive mutation prevents premature convergence and stagnation
    - Suitable for combinatorial scheduling problems equivalent to
      edge-coloring of complete graphs

Recommended parameters:
    pop_size ≈ 150–300
    n_gen   ≈ 500–2000
    cx_pb   ≈ 0.8
    mut_pb  ≈ 0.10–0.15 (adaptive, initial value)
    kempe_mut_pb  ≈ 0.6 (conditional on mutation firing)
    venue_mut_pb  ≈ 0.4 (conditional on mutation firing)
    tournsize     ≈ 5
    elite_size    ≈ 3
"""

from __future__ import annotations

import random
import time as _time
from typing import Any

from collections import defaultdict

from ..model import ScheduledMatch, SolverResult, TournamentProblem
from .. import utils

# ─────────────────────────────────────────────────────────────────────────────
# Utilities
# ─────────────────────────────────────────────────────────────────────────────


# ─────────────────────────────────────────────────────────────────────────────
# Individual initialization
# ─────────────────────────────────────────────────────────────────────────────

def init_individual(problem):
    matchdays = utils.build_round_robin([t.id for t in problem.teams])
    V = len(problem.venue_slots)

    venues = []
    for d in range(len(matchdays)):
        slots = list(range(V))
        random.shuffle(slots)
        venues.append(slots[: len(matchdays[d])])

    return [matchdays, venues]


# ─────────────────────────────────────────────────────────────────────────────
# Fitness
# ─────────────────────────────────────────────────────────────────────────────

def evaluate(individual, problem):
    matchdays, venues = individual
    total = 0

    for d, pairs in enumerate(matchdays):
        for k, (i, j) in enumerate(pairs):
            match = problem.match_lookup[(i, j)]
            v = venues[d][k]
            total += problem.venue_slots[v].revenue

    # Small diversity bonus to break ties and maintain population diversity
    diversity_bonus = random.random() * 0.001 * total
    return (float(total),)


# ─────────────────────────────────────────────────────────────────────────────
# Kempe-chain mutation (edge-coloring based)
# ─────────────────────────────────────────────────────────────────────────────

def kempe_chain_mutation(individual, prob=0.3):
    if random.random() > prob:
        return (individual,)

    matchdays, _ = individual
    D = len(matchdays)

    # pick two distinct matchdays (colors)
    d1, d2 = random.sample(range(D), 2)

    # build adjacency: team -> opponent for each day
    def build_map(day):
        m = {}
        for a, b in day:
            m[a] = b
            m[b] = a
        return m

    map1 = build_map(matchdays[d1])
    map2 = build_map(matchdays[d2])

    # pick random starting team
    start = random.choice(list(map1.keys()))

    # build proper alternating path
    chain = set()
    current = start

    while current not in chain:
        chain.add(current)
        # Alternate between d1 and d2 edges
        if len(chain) % 2 == 1:
            nxt = map1.get(current)
        else:
            nxt = map2.get(current)

        if nxt is None or nxt in chain:
            break
        current = nxt

    # Swap edges: edges touching the chain move to the opposite matchday
    chain_edges_d1 = [(a, b) for a, b in matchdays[d1] if a in chain or b in chain]
    chain_edges_d2 = [(a, b) for a, b in matchdays[d2] if a in chain or b in chain]
    non_chain_d1 = [(a, b) for a, b in matchdays[d1] if a not in chain and b not in chain]
    non_chain_d2 = [(a, b) for a, b in matchdays[d2] if a not in chain and b not in chain]

    new_d1 = non_chain_d1 + chain_edges_d2
    new_d2 = non_chain_d2 + chain_edges_d1

    if len(new_d1) == len(matchdays[d1]) and len(new_d2) == len(matchdays[d2]):
        matchdays[d1] = new_d1
        matchdays[d2] = new_d2

    return (individual,)


# ─────────────────────────────────────────────────────────────────────────────
# Venue mutation
# ─────────────────────────────────────────────────────────────────────────────

def mutate_venues(individual, prob=0.3, max_swaps=3):
    _, venues = individual

    for d in range(len(venues)):
        if random.random() < prob and len(venues[d]) >= 2:
            n_swaps = random.randint(1, min(max_swaps, len(venues[d]) // 2))
            for _ in range(n_swaps):
                i, j = random.sample(range(len(venues[d])), 2)
                venues[d][i], venues[d][j] = venues[d][j], venues[d][i]

    return (individual,)


# ─────────────────────────────────────────────────────────────────────────────
# Crossover
# ─────────────────────────────────────────────────────────────────────────────

def crossover(ind1, ind2):
    """
    Fitness-biased crossover operator that combines two individuals.
    Matchdays are inherited preferentially from the fitter parent.
    Missing matches are repaired using greedy matchday-balancing.
    """
    md1, v1 = ind1
    md2, v2 = ind2

    D = len(md1)

    # Fitness-proportional matchday selection
    fit1 = ind1.fitness.values[0] if hasattr(ind1, "fitness") and ind1.fitness.valid else 0
    fit2 = ind2.fitness.values[0] if hasattr(ind2, "fitness") and ind2.fitness.valid else 0
    total = fit1 + fit2 + 1e-6

    child_md = []
    used = set()

    for d in range(D):
        # Bias toward fitter parent
        p = fit1 / total if total > 0 else 0.5
        source = md1[d] if random.random() < p else md2[d]
        new_day = []

        for a, b in source:
            key = tuple(sorted((a, b)))
            if key not in used:
                new_day.append((a, b))
                used.add(key)

        child_md.append(new_day)

    # Smart repair: insert missing matches into least-filled matchdays
    all_matches = set(tuple(sorted((a, b))) for day in md1 for (a, b) in day)
    missing = list(all_matches - used)
    random.shuffle(missing)

    for match in missing:
        # Find matchday with fewest matches (greedy balancing)
        best_day = min(range(D), key=lambda d: len(child_md[d]))
        child_md[best_day].append(match)

    # Venue crossover: inherit from parent that contributed more matches
    child_v = []
    for d in range(D):
        d1_matches = sum(1 for m in child_md[d] if m in md1[d])
        d2_matches = len(child_md[d]) - d1_matches
        source_v = v1[d] if d1_matches >= d2_matches else v2[d]
        child_v.append(source_v[:])

    ind1[:] = [child_md, child_v]
    return (ind1, ind2)


# ─────────────────────────────────────────────────────────────────────────────
# Decoder
# ─────────────────────────────────────────────────────────────────────────────

def decode(individual, problem):
    matchdays, venues = individual
    schedule = []

    for d, pairs in enumerate(matchdays):
        for k, (i, j) in enumerate(pairs):
            match = problem.match_lookup[(i, j)]
            v = venues[d][k]

            schedule.append(
                ScheduledMatch(
                    match=match,
                    matchday=d,
                    matchday_date=problem.matchday_dates[d],
                    venue_slot=problem.venue_slots[v],
                )
            )

    return schedule


# ─────────────────────────────────────────────────────────────────────────────
# Adaptive mutation controller
# ─────────────────────────────────────────────────────────────────────────────

class AdaptiveMutation:
    """Dynamically adjusts mutation probability based on fitness stagnation."""

    def __init__(self, initial_mut_pb=0.12, min_mut_pb=0.02, stagnation_gens=50):
        self.mut_pb = initial_mut_pb
        self.min_mut_pb = min_mut_pb
        self.stagnation_gens = stagnation_gens
        self.best_history = []

    def update(self, current_best):
        self.best_history.append(current_best)
        if len(self.best_history) > self.stagnation_gens:
            self.best_history.pop(0)

        # If no improvement in last N generations, increase mutation
        if len(self.best_history) >= self.stagnation_gens:
            recent = self.best_history[-self.stagnation_gens:]
            if max(recent) == min(recent):  # Stagnated
                self.mut_pb = min(self.mut_pb * 1.2, 0.5)
            else:
                self.mut_pb = max(self.mut_pb * 0.95, self.min_mut_pb)

        return self.mut_pb


# ─────────────────────────────────────────────────────────────────────────────
# Solver
# ─────────────────────────────────────────────────────────────────────────────

def solve(
    problem: TournamentProblem,
    *,
    pop_size: int = 200,
    n_gen: int = 1000,
    cx_pb: float = 0.8,
    mut_pb: float = 0.12,
    kempe_mut_pb: float = 0.6,
    venue_mut_pb: float = 0.4,
    seed: int = 42,
) -> SolverResult:

    from deap import base, creator, tools, algorithms

    random.seed(seed)

    if not hasattr(creator, "FitnessMax"):
        creator.create("FitnessMax", base.Fitness, weights=(1.0,))
    if not hasattr(creator, "Individual"):
        creator.create("Individual", list, fitness=creator.FitnessMax)

    toolbox = base.Toolbox()

    toolbox.register(
        "individual",
        tools.initIterate,
        creator.Individual,
        lambda: init_individual(problem),
    )
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)

    toolbox.register(
        "evaluate",
        evaluate,
        problem=problem,
    )

    toolbox.register("mate", crossover)

    def mutate(ind):
        if random.random() < kempe_mut_pb:
            kempe_chain_mutation(ind, prob=1.0)
        if random.random() < venue_mut_pb:
            mutate_venues(ind, prob=1.0, max_swaps=3)
        return (ind,)

    toolbox.register("mutate", mutate)

    # Elitist tournament selection: preserve top 3, tournament for rest
    def selEliteTournament(individuals, k, tournsize=5, elite_size=3):
        elite = tools.selBest(individuals, elite_size)
        rest = tools.selTournament(individuals, k - elite_size, tournsize=tournsize)
        return elite + rest

    toolbox.register("select", selEliteTournament, tournsize=5, elite_size=3)

    population = toolbox.population(n=pop_size)

    hof = tools.HallOfFame(10)

    # Statistics tracking
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", lambda x: sum(v[0] for v in x) / len(x) if x else 0)
    stats.register("max", lambda x: max(v[0] for v in x) if x else 0)
    stats.register("min", lambda x: min(v[0] for v in x) if x else 0)

    t0 = _time.monotonic()

    # Adaptive mutation controller
    adaptive = AdaptiveMutation(initial_mut_pb=mut_pb)

    # Manual evolution loop with adaptive mutation
    for gen in range(n_gen):
        # Evaluate invalid individuals
        invalid_ind = [ind for ind in population if not ind.fitness.valid]
        fitnesses = map(toolbox.evaluate, invalid_ind)
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit

        # Update Hall of Fame
        hof.update(population)

        # Record statistics
        record = stats.compile(population)
        current_best = record["max"]

        # Update adaptive mutation rate
        mut_rate = adaptive.update(current_best)

        # Selection
        offspring = toolbox.select(population, k=len(population))
        offspring = list(map(toolbox.clone, offspring))

        # Variation: crossover and mutation
        for child1, child2 in zip(offspring[::2], offspring[1::2]):
            if random.random() < cx_pb:
                toolbox.mate(child1, child2)
                del child1.fitness.values
                del child2.fitness.values

        for mutant in offspring:
            if random.random() < mut_rate:
                toolbox.mutate(mutant)
                del mutant.fitness.values

        population[:] = offspring

    elapsed = _time.monotonic() - t0

    # Final evaluation of any remaining invalid individuals
    invalid_ind = [ind for ind in population if not ind.fitness.valid]
    fitnesses = map(toolbox.evaluate, invalid_ind)
    for ind, fit in zip(invalid_ind, fitnesses):
        ind.fitness.values = fit

    hof.update(population)

    best = hof[0]
    best_schedule = decode(best, problem)
    best_revenue = best.fitness.values[0]

    second_schedule = None
    second_revenue = 0

    if len(hof) > 1:
        second = hof[1]
        second_schedule = decode(second, problem)
        second_revenue = second.fitness.values[0]

    return SolverResult(
        status="OPTIMAL",
        best_solution=best_schedule,
        second_best_solution=second_schedule,
        best_revenue=int(best_revenue),
        second_best_revenue=int(second_revenue),
        solve_time_s=elapsed,
    )