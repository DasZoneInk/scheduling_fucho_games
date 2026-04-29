"""test_solvers.py — Correctness and performance regression tests for CP-SAT and GA.

Test classes:
    TestCPSATCorrectness   — example problem (6 teams) must solve OPTIMALLY
    TestGACorrectness      — example problem (6 teams) must converge to optimal
    TestScheduleInvariants — schedule validity helper tests on canonical output
    TestCPSATStress        — 16-team programmatic problem, ≤ 30s budget
    TestGAStress           — 16-team programmatic problem, ≤ 30s budget

The stress tests serve as production-readiness regression gates:
    if the solver regresses in performance, these tests will fail the CI run.
"""

from __future__ import annotations

import time
from collections import defaultdict

import pytest

from scheduling_fucho_games.algorithms import cpsat, genetic_algorithm as ga, kempe
from scheduling_fucho_games.model import (
    BYE_TEAM_ID,
    ScheduledMatch,
    SolverResult,
    TournamentProblem,
)

from .conftest import TournamentProblem  # for type annotation only

# ── Wall-clock budget constants (seconds) ─────────────────────────────────────
CPSAT_STRESS_TIMEOUT_S: float = 30.0
GA_STRESS_BUDGET_S: float = 30.0
GA_STRESS_POP: int = 250
GA_STRESS_GENS: int = 400


# ── Schedule validity helper ──────────────────────────────────────────────────

def _assert_schedule_valid(
    schedule: list[ScheduledMatch],
    problem: TournamentProblem,
    *,
    stripped: bool = False,
) -> None:
    """Assert all hard constraints are satisfied in *schedule*.

    When *stripped* is True the schedule has already had BYE matches removed
    (odd-K post-processing), so match counts per matchday are K_orig//2
    (i.e. (K-1)//2 where K is the padded even count).
    """
    K = len(problem.teams)
    n_matchdays = K - 1

    if stripped and problem.has_bye:
        # After BYE stripping: real matches per matchday = (K-2)//2
        mpm = (K - 1) // 2
        # Exclude BYE matches from expected IDs
        expected_ids = {
            m.id for m in problem.matches
            if m.home.id != BYE_TEAM_ID and m.away.id != BYE_TEAM_ID
        }
    else:
        mpm = K // 2
        expected_ids = {m.id for m in problem.matches}

    valid_dates = set(problem.matchday_dates)

    # 1 — Match coverage
    scheduled_match_ids = [sm.match.id for sm in schedule]
    missing = expected_ids - set(scheduled_match_ids)
    duplicates = {mid for mid in scheduled_match_ids if scheduled_match_ids.count(mid) > 1}
    assert not missing, f"Unscheduled match IDs: {missing}"
    assert not duplicates, f"Match IDs scheduled more than once: {duplicates}"

    # 2 — Team-per-matchday uniqueness
    team_day: dict[tuple[int, int], list[int]] = defaultdict(list)
    for sm in schedule:
        team_day[(sm.match.home.id, sm.matchday)].append(sm.match.id)
        team_day[(sm.match.away.id, sm.matchday)].append(sm.match.id)

    conflicts = {k: v for k, v in team_day.items() if len(v) > 1}
    assert not conflicts, (
        f"Team plays more than once on a matchday: {conflicts}"
    )

    # 3 — Venue-slot uniqueness per matchday
    slot_day: dict[tuple[int, time, int], list[int]] = defaultdict(list)
    for sm in schedule:
        key = (sm.venue_slot.field_id, sm.venue_slot.slot_start, sm.matchday)
        slot_day[key].append(sm.match.id)

    venue_conflicts = {k: v for k, v in slot_day.items() if len(v) > 1}
    assert not venue_conflicts, (
        f"Venue-slot double-booked on same matchday: {venue_conflicts}"
    )

    # 4 — Matches per matchday
    by_day: dict[int, int] = defaultdict(int)
    for sm in schedule:
        by_day[sm.matchday] += 1

    bad_days = {d: cnt for d, cnt in by_day.items() if cnt != mpm}
    assert not bad_days, (
        f"Matchdays with wrong match count (expected {mpm}): {bad_days}"
    )
    assert len(by_day) == n_matchdays, (
        f"Expected {n_matchdays} matchdays, got {len(by_day)}"
    )

    # 5 — Dates from pre-assigned calendar
    bad_dates = {sm.matchday_date for sm in schedule if sm.matchday_date not in valid_dates}
    assert not bad_dates, f"Schedule contains invalid dates: {bad_dates}"


# ── CP-SAT correctness (6-team canonical) ────────────────────────────────────

class TestCPSATCorrectness:
    def test_status_optimal(self, example_problem: TournamentProblem) -> None:
        result = cpsat.solve(example_problem, timeout_s=30.0)
        assert result.status == "OPTIMAL"

    def test_best_revenue(self, example_problem: TournamentProblem) -> None:
        result = cpsat.solve(example_problem, timeout_s=30.0)
        assert result.best_revenue == 1_600

    def test_best_solution_exists(self, example_problem: TournamentProblem) -> None:
        result = cpsat.solve(example_problem, timeout_s=30.0)
        assert result.best_solution is not None
        assert len(result.best_solution) == len(example_problem.matches)

    def test_second_best_solution_exists(self, example_problem: TournamentProblem) -> None:
        result = cpsat.solve(example_problem, timeout_s=30.0)
        assert result.second_best_solution is not None, (
            "CP-SAT must return a second-best solution for the 6-team example"
        )

    def test_second_best_revenue_lower_than_best(self, example_problem: TournamentProblem) -> None:
        result = cpsat.solve(example_problem, timeout_s=30.0)
        if result.second_best_solution:
            assert result.second_best_revenue <= result.best_revenue

    def test_best_schedule_satisfies_constraints(self, example_problem: TournamentProblem) -> None:
        result = cpsat.solve(example_problem, timeout_s=30.0)
        assert result.best_solution is not None
        _assert_schedule_valid(result.best_solution, example_problem)

    def test_second_best_schedule_satisfies_constraints(self, example_problem: TournamentProblem) -> None:
        result = cpsat.solve(example_problem, timeout_s=30.0)
        if result.second_best_solution:
            _assert_schedule_valid(result.second_best_solution, example_problem)

    def test_solve_time_recorded(self, example_problem: TournamentProblem) -> None:
        result = cpsat.solve(example_problem, timeout_s=30.0)
        assert result.solve_time_s > 0.0

    def test_infeasibility_reason_null_on_success(self, example_problem: TournamentProblem) -> None:
        result = cpsat.solve(example_problem, timeout_s=30.0)
        assert result.infeasibility_reason is None


# ── GA correctness (6-team canonical) ────────────────────────────────────────

class TestGACorrectness:
    def test_status_optimal(self, example_problem: TournamentProblem) -> None:
        result = ga.solve(example_problem, pop_size=300, n_gen=600, seed=42)
        assert result.status == "OPTIMAL"

    def test_best_revenue(self, example_problem: TournamentProblem) -> None:
        result = ga.solve(example_problem, pop_size=300, n_gen=600, seed=42)
        assert result.best_revenue == 1_600

    def test_best_solution_exists(self, example_problem: TournamentProblem) -> None:
        result = ga.solve(example_problem, pop_size=300, n_gen=600, seed=42)
        assert result.best_solution is not None

    def test_best_schedule_satisfies_constraints(self, example_problem: TournamentProblem) -> None:
        result = ga.solve(example_problem, pop_size=300, n_gen=600, seed=42)
        assert result.best_solution is not None
        _assert_schedule_valid(result.best_solution, example_problem)

    def test_second_best_feasible_if_returned(self, example_problem: TournamentProblem) -> None:
        result = ga.solve(example_problem, pop_size=300, n_gen=600, seed=42)
        if result.second_best_solution:
            _assert_schedule_valid(result.second_best_solution, example_problem)

    def test_solve_time_recorded(self, example_problem: TournamentProblem) -> None:
        result = ga.solve(example_problem, pop_size=300, n_gen=600, seed=42)
        assert result.solve_time_s > 0.0

    def test_reproducibility(self, example_problem: TournamentProblem) -> None:
        """Same seed must produce identical revenue on repeated calls."""
        r1 = ga.solve(example_problem, pop_size=150, n_gen=200, seed=7)
        r2 = ga.solve(example_problem, pop_size=150, n_gen=200, seed=7)
        assert r1.best_revenue == r2.best_revenue

# ── Kempe correctness (6-team canonical) ────────────────────────────────────────

class TestKempeCorrectness:
    def test_status_optimal(self, example_problem: TournamentProblem) -> None:
        result = kempe.solve(example_problem, pop_size=300, n_gen=600, seed=42)
        assert result.status == "OPTIMAL"

    def test_best_revenue(self, example_problem: TournamentProblem) -> None:
        result = kempe.solve(example_problem, pop_size=300, n_gen=600, seed=42)
        assert result.best_revenue == 1_600

    def test_best_solution_exists(self, example_problem: TournamentProblem) -> None:
        result = kempe.solve(example_problem, pop_size=300, n_gen=600, seed=42)
        assert result.best_solution is not None

    def test_best_schedule_satisfies_constraints(self, example_problem: TournamentProblem) -> None:
        result = kempe.solve(example_problem, pop_size=300, n_gen=600, seed=42)
        assert result.best_solution is not None
        _assert_schedule_valid(result.best_solution, example_problem)

    def test_second_best_feasible_if_returned(self, example_problem: TournamentProblem) -> None:
        result = kempe.solve(example_problem, pop_size=300, n_gen=600, seed=42)
        if result.second_best_solution:
            _assert_schedule_valid(result.second_best_solution, example_problem)

    def test_solve_time_recorded(self, example_problem: TournamentProblem) -> None:
        result = kempe.solve(example_problem, pop_size=300, n_gen=600, seed=42)
        assert result.solve_time_s > 0.0

    def test_reproducibility(self, example_problem: TournamentProblem) -> None:
        """Same seed must produce identical revenue on repeated calls."""
        r1 = kempe.solve(example_problem, pop_size=150, n_gen=200, seed=7)
        r2 = kempe.solve(example_problem, pop_size=150, n_gen=200, seed=7)
        assert r1.best_revenue == r2.best_revenue


# ── Odd-K CP-SAT (5-team with bye) ───────────────────────────────────────────

def _strip_bye(schedule, problem):
    """Test helper — mimics main._strip_bye_matches."""
    bye_teams: dict[int, str] = {}
    cleaned: list[ScheduledMatch] = []
    for sm in schedule:
        home, away = sm.match.home, sm.match.away
        if home.id == BYE_TEAM_ID:
            bye_teams[sm.matchday] = away.name
        elif away.id == BYE_TEAM_ID:
            bye_teams[sm.matchday] = home.name
        else:
            cleaned.append(sm)
    return cleaned, bye_teams


class TestOddTeamCPSAT:
    """5-team odd-K with BYE injection, solved by CP-SAT."""

    def test_has_bye_flag(self, odd_example_problem: TournamentProblem) -> None:
        assert odd_example_problem.has_bye is True

    def test_status_optimal(self, odd_example_problem: TournamentProblem) -> None:
        result = cpsat.solve(odd_example_problem, timeout_s=30.0)
        assert result.status == "OPTIMAL"

    def test_full_schedule_valid(self, odd_example_problem: TournamentProblem) -> None:
        """Before BYE stripping — raw solver output passes even-K invariants."""
        result = cpsat.solve(odd_example_problem, timeout_s=30.0)
        assert result.best_solution is not None
        _assert_schedule_valid(result.best_solution, odd_example_problem)

    def test_stripped_schedule_valid(self, odd_example_problem: TournamentProblem) -> None:
        """After BYE stripping — real matches pass odd-K invariants."""
        result = cpsat.solve(odd_example_problem, timeout_s=30.0)
        assert result.best_solution is not None
        cleaned, bye_map = _strip_bye(result.best_solution, odd_example_problem)
        assert cleaned is not None
        _assert_schedule_valid(cleaned, odd_example_problem, stripped=True)

    def test_one_bye_per_matchday(self, odd_example_problem: TournamentProblem) -> None:
        result = cpsat.solve(odd_example_problem, timeout_s=30.0)
        assert result.best_solution is not None
        _, bye_map = _strip_bye(result.best_solution, odd_example_problem)
        K = len(odd_example_problem.teams)
        assert len(bye_map) == K - 1, (
            f"Expected {K-1} bye entries (one per matchday); got {len(bye_map)}"
        )

    def test_every_real_team_gets_one_bye(self, odd_example_problem: TournamentProblem) -> None:
        result = cpsat.solve(odd_example_problem, timeout_s=30.0)
        assert result.best_solution is not None
        _, bye_map = _strip_bye(result.best_solution, odd_example_problem)
        real_team_names = {
            t.name for t in odd_example_problem.teams if t.id != BYE_TEAM_ID
        }
        bye_team_names = set(bye_map.values())
        assert bye_team_names == real_team_names, (
            f"Not every team got a bye. Missing: {real_team_names - bye_team_names}"
        )


class TestOddTeamKempe:
    """5-team odd-K with BYE injection, solved by Kempe."""

    def test_has_bye_flag(self, odd_example_problem: TournamentProblem) -> None:
        assert odd_example_problem.has_bye is True

    def test_status_optimal(self, odd_example_problem: TournamentProblem) -> None:
        result = kempe.solve(odd_example_problem, pop_size=200, n_gen=500, seed=42)
        assert result.status == "OPTIMAL"

    def test_full_schedule_valid(self, odd_example_problem: TournamentProblem) -> None:
        result = kempe.solve(odd_example_problem, pop_size=200, n_gen=500, seed=42)
        assert result.best_solution is not None
        _assert_schedule_valid(result.best_solution, odd_example_problem)

    def test_stripped_schedule_valid(self, odd_example_problem: TournamentProblem) -> None:
        result = kempe.solve(odd_example_problem, pop_size=200, n_gen=500, seed=42)
        assert result.best_solution is not None
        cleaned, _ = _strip_bye(result.best_solution, odd_example_problem)
        assert cleaned is not None
        _assert_schedule_valid(cleaned, odd_example_problem, stripped=True)

    def test_one_bye_per_matchday(self, odd_example_problem: TournamentProblem) -> None:
        result = kempe.solve(odd_example_problem, pop_size=200, n_gen=500, seed=42)
        assert result.best_solution is not None
        _, bye_map = _strip_bye(result.best_solution, odd_example_problem)
        K = len(odd_example_problem.teams)
        assert len(bye_map) == K - 1

    def test_every_real_team_gets_one_bye(self, odd_example_problem: TournamentProblem) -> None:
        result = kempe.solve(odd_example_problem, pop_size=200, n_gen=500, seed=42)
        assert result.best_solution is not None
        _, bye_map = _strip_bye(result.best_solution, odd_example_problem)
        real_team_names = {
            t.name for t in odd_example_problem.teams if t.id != BYE_TEAM_ID
        }
        bye_team_names = set(bye_map.values())
        assert bye_team_names == real_team_names

# ── Schedule invariant unit tests ─────────────────────────────────────────────

class TestScheduleInvariants:
    """Directly test the _assert_schedule_valid helper on the canonical solution."""

    def test_canonical_cpsat_output_passes_invariants(
        self, example_problem: TournamentProblem
    ) -> None:
        result = cpsat.solve(example_problem, timeout_s=30.0)
        assert result.best_solution is not None
        _assert_schedule_valid(result.best_solution, example_problem)

    def test_match_count_equals_k_choose_2(self, example_problem: TournamentProblem) -> None:
        K = len(example_problem.teams)
        expected = K * (K - 1) // 2
        assert len(example_problem.matches) == expected


# ── CP-SAT stress test (16 teams, ≤ 30s) ─────────────────────────────────────

@pytest.mark.slow
class TestCPSATStress:
    """Performance regression gate: 16-team instance must complete within 30s budget.

    16 teams → 120 matches, 15 matchdays, 8 simultaneous matches/day.
    Decision variables: 120 × 15 × V (≥ 9 venue-slots).
    """

    def test_completes_within_budget(self, stress_problem: TournamentProblem) -> None:
        t0 = time.monotonic()
        result = cpsat.solve(stress_problem, timeout_s=CPSAT_STRESS_TIMEOUT_S)
        elapsed = time.monotonic() - t0
        assert elapsed <= CPSAT_STRESS_TIMEOUT_S + 2.0, (
            f"CP-SAT exceeded 30s budget on stress instance: {elapsed:.1f}s"
        )
        # Solver must have reached a conclusion
        assert result.status in ("OPTIMAL", "FEASIBLE", "UNKNOWN"), (
            f"Unexpected status: {result.status}"
        )

    def test_returns_feasible_solution(self, stress_problem: TournamentProblem) -> None:
        result = cpsat.solve(stress_problem, timeout_s=CPSAT_STRESS_TIMEOUT_S)
        assert result.best_solution is not None, (
            f"CP-SAT returned no solution within {CPSAT_STRESS_TIMEOUT_S}s "
            f"(status={result.status})"
        )

    def test_stress_schedule_satisfies_constraints(
        self, stress_problem: TournamentProblem
    ) -> None:
        result = cpsat.solve(stress_problem, timeout_s=CPSAT_STRESS_TIMEOUT_S)
        if result.best_solution:
            _assert_schedule_valid(result.best_solution, stress_problem)

    def test_stress_revenue_positive(self, stress_problem: TournamentProblem) -> None:
        result = cpsat.solve(stress_problem, timeout_s=CPSAT_STRESS_TIMEOUT_S)
        if result.best_solution:
            assert result.best_revenue > 0

    def test_stress_problem_shape(self, stress_problem: TournamentProblem) -> None:
        """Sanity-check the programmatic problem before trusting solver results."""
        K = 16
        assert len(stress_problem.teams) == K
        assert len(stress_problem.matches) == K * (K - 1) // 2  # 120
        assert len(stress_problem.matchday_dates) == K - 1       # 15
        assert stress_problem.matches_per_matchday == K // 2     # 8
        assert len(stress_problem.venue_slots) >= stress_problem.matches_per_matchday


# ── GA stress test (16 teams, ≤ 30s wall-clock) ──────────────────────────────

@pytest.mark.slow
class TestGAStress:
    """Performance regression gate: 16-team GA must terminate within 30s.

    GA parameters chosen to saturate ~25s on a modern MacBook Air:
        pop_size=250, n_gen=400
    Adjust constants at top of file to recalibrate if hardware changes.
    """

    def test_completes_within_budget(self, stress_problem: TournamentProblem) -> None:
        t0 = time.monotonic()
        result = ga.solve(
            stress_problem,
            pop_size=GA_STRESS_POP,
            n_gen=GA_STRESS_GENS,
            seed=42,
        )
        elapsed = time.monotonic() - t0
        assert elapsed <= GA_STRESS_BUDGET_S + 5.0, (
            f"GA exceeded 30s budget on stress instance: {elapsed:.1f}s"
        )
        # GA must have at least attempted optimisation
        assert result.status in ("OPTIMAL", "NOT_CONVERGED"), (
            f"Unexpected GA status: {result.status}"
        )

    def test_returns_feasible_solution(self, stress_problem: TournamentProblem) -> None:
        result = ga.solve(
            stress_problem,
            pop_size=GA_STRESS_POP,
            n_gen=GA_STRESS_GENS,
            seed=42,
        )
        assert result.best_solution is not None, (
            "GA returned no solution for stress instance"
        )

    def test_stress_schedule_satisfies_constraints(
        self, stress_problem: TournamentProblem
    ) -> None:
        result = ga.solve(
            stress_problem,
            pop_size=GA_STRESS_POP,
            n_gen=GA_STRESS_GENS,
            seed=42,
        )
        if result.status == "OPTIMAL" and result.best_solution:
            _assert_schedule_valid(result.best_solution, stress_problem)

    def test_stress_revenue_positive(self, stress_problem: TournamentProblem) -> None:
        result = ga.solve(
            stress_problem,
            pop_size=GA_STRESS_POP,
            n_gen=GA_STRESS_GENS,
            seed=42,
        )
        if result.best_solution:
            assert result.best_revenue > 0

    def test_solve_time_recorded(self, stress_problem: TournamentProblem) -> None:
        result = ga.solve(
            stress_problem,
            pop_size=GA_STRESS_POP,
            n_gen=GA_STRESS_GENS,
            seed=42,
        )
        assert result.solve_time_s > 0.0
