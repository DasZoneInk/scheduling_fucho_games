"""test_loader.py — Validates yml_loader error handling and schema enforcement.

Covers:
    - Non-canonical structure → ValueError before any solver invocation.
    - Odd team count → ValueError (round-robin parity constraint).
    - Insufficient feasible dates → ValueError / InfeasibilityError.
    - Happy-path canonical input → TournamentProblem with correct shape.
"""

from __future__ import annotations

import pytest

from scheduling_fucho_games.model import InfeasibilityError, TournamentProblem
from scheduling_fucho_games.yml_loader import load_problem

from .conftest import TEST_ASSETS, PKG_ASSETS


# ── Non-canonical structure ───────────────────────────────────────────────────

class TestNonCanonical:
    """YAML files that violate the expected schema must raise ValueError."""

    def test_non_canonical_raises_value_error(self) -> None:
        """constraints as YAML list + missing team name + bad time range → ValueError."""
        with pytest.raises(ValueError):
            load_problem(TEST_ASSETS / "non_canonical.yml")

    def test_file_not_found_raises(self) -> None:
        with pytest.raises(FileNotFoundError):
            load_problem(TEST_ASSETS / "does_not_exist.yml")

    def test_non_canonical_not_infeasibility_error(self) -> None:
        """Schema errors must be ValueError, not InfeasibilityError.
        InfeasibilityError is reserved for structurally valid but unsolvable problems.
        """
        with pytest.raises(ValueError) as exc_info:
            load_problem(TEST_ASSETS / "non_canonical.yml")
        assert not isinstance(exc_info.value, InfeasibilityError), (
            "Schema violation should not be InfeasibilityError"
        )


# ── Odd team count ────────────────────────────────────────────────────────────

class TestOddTeam:
    """5-team configs must fail before date expansion or solver invocation."""

    def test_odd_team_raises(self) -> None:
        with pytest.raises(ValueError):
            load_problem(TEST_ASSETS / "odd_team.yml")

    def test_odd_team_error_mentions_even(self) -> None:
        """Error message must guide the user toward the parity requirement."""
        with pytest.raises(ValueError, match="(?i)even"):
            load_problem(TEST_ASSETS / "odd_team.yml")

    def test_odd_team_raises_before_date_expansion(self) -> None:
        """Parity check happens in validate_team_count, before get_feasible_dates.
        Confirm the error message is NOT about dates or matchdays.
        """
        with pytest.raises(ValueError) as exc_info:
            load_problem(TEST_ASSETS / "odd_team.yml")
        msg = str(exc_info.value).lower()
        assert "date" not in msg or "even" in msg, (
            "Expected parity error, not a date-range error"
        )


# ── Infeasible date window ────────────────────────────────────────────────────

class TestNonFeasibleDates:
    """6 teams need 5 matchdays but only 4 Mondays fit in the window → infeasible."""

    def test_non_feasible_raises(self) -> None:
        with pytest.raises((ValueError, InfeasibilityError)):
            load_problem(TEST_ASSETS / "non_feasible_1.yml")

    def test_non_feasible_error_mentions_matchday_or_date(self) -> None:
        """Error must be diagnostic — mention dates or matchdays."""
        with pytest.raises((ValueError, InfeasibilityError)) as exc_info:
            load_problem(TEST_ASSETS / "non_feasible_1.yml")
        msg = str(exc_info.value).lower()
        assert any(kw in msg for kw in ("date", "matchday", "feasible", "gap")), (
            f"Expected diagnostic date-shortage message; got: {exc_info.value}"
        )

    def test_non_feasible_is_not_schema_error(self) -> None:
        """File is structurally valid YAML — must fail at the date-assignment
        stage, not at schema parsing.  Therefore the error type must NOT arise
        from team-count or time-range parsing.
        """
        with pytest.raises((ValueError, InfeasibilityError)) as exc_info:
            load_problem(TEST_ASSETS / "non_feasible_1.yml")
        msg = str(exc_info.value).lower()
        assert "even" not in msg, "Should be a date-infeasibility error, not parity"
        assert "time" not in msg or "date" in msg


# ── Happy-path (canonical example) ───────────────────────────────────────────

class TestCanonicalLoad:
    """Canonical example_input.yml must parse cleanly into TournamentProblem."""

    def test_returns_tournament_problem(self, example_problem: TournamentProblem) -> None:
        assert isinstance(example_problem, TournamentProblem)

    def test_team_count(self, example_problem: TournamentProblem) -> None:
        assert len(example_problem.teams) == 6

    def test_match_count(self, example_problem: TournamentProblem) -> None:
        # K*(K-1)/2 = 6*5/2 = 15
        assert len(example_problem.matches) == 15

    def test_matchday_count(self, example_problem: TournamentProblem) -> None:
        # K-1 = 5
        assert len(example_problem.matchday_dates) == 5

    def test_matches_per_matchday(self, example_problem: TournamentProblem) -> None:
        # K/2 = 3
        assert example_problem.matches_per_matchday == 3

    def test_venue_slots_positive(self, example_problem: TournamentProblem) -> None:
        assert len(example_problem.venue_slots) >= example_problem.matches_per_matchday

    def test_matchday_dates_sorted_and_gapped(self, example_problem: TournamentProblem) -> None:
        dates = example_problem.matchday_dates
        for prev, nxt in zip(dates, dates[1:]):
            assert (nxt - prev).days >= example_problem.matchday_gap_days, (
                f"Gap too short: {prev} → {nxt}"
            )

    def test_all_team_ids_unique(self, example_problem: TournamentProblem) -> None:
        ids = [t.id for t in example_problem.teams]
        assert len(ids) == len(set(ids))

    def test_all_match_pairs_unique(self, example_problem: TournamentProblem) -> None:
        pairs = {(m.home.id, m.away.id) for m in example_problem.matches}
        assert len(pairs) == len(example_problem.matches)
