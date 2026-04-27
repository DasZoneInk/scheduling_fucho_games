"""model.py — Canonical domain model for the scheduling ILP.

TournamentProblem is the single intermediate representation produced by
yml_loader and consumed by all solver implementations.  Solvers must not
modify the problem; they return SolverResult instances.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import date, time
from typing import Final

logger = logging.getLogger(__name__)

# ── Supported tournament formats ──────────────────────────────────────────────
SUPPORTED_FORMATS: Final[frozenset[str]] = frozenset({"round_robin"})
BYE_TEAM_ID: Final[int] = -1


# ── Custom exception ──────────────────────────────────────────────────────────

class InfeasibilityError(ValueError):
    """Raised when the problem instance has no feasible solution."""


# ── Domain dataclasses ────────────────────────────────────────────────────────

@dataclass(frozen=True)
class Team:
    id: int
    name: str

    def __repr__(self) -> str:
        return self.name


@dataclass(frozen=True)
class Match:
    id: int
    home: Team
    away: Team

    def involves(self, team: Team) -> bool:
        return self.home == team or self.away == team

    def label(self) -> str:
        return f"{self.home.name} vs {self.away.name}"

    def __repr__(self) -> str:
        return self.label()


@dataclass(frozen=True)
class VenueSlot:
    """A (field, time-slot, revenue) triple — the atomic scheduling unit."""

    field_id: int
    field_name: str
    slot_start: time
    slot_end: time
    revenue: int

    def label(self) -> str:
        return (
            f"{self.field_name} "
            f"[{self.slot_start.strftime('%H:%M')}-{self.slot_end.strftime('%H:%M')}]"
        )

    def __repr__(self) -> str:
        return f"{self.label()} (${self.revenue})"


@dataclass(frozen=True)
class ScheduledMatch:
    """One match assigned to a specific matchday, date, and venue-slot."""

    match: Match
    matchday: int           # 0-indexed
    matchday_date: date
    venue_slot: VenueSlot


# ── Solver output ─────────────────────────────────────────────────────────────

@dataclass
class SolverResult:
    status: str                                     # OPTIMAL | FEASIBLE | INFEASIBLE | NOT_CONVERGED
    best_solution: list[ScheduledMatch] | None = None
    second_best_solution: list[ScheduledMatch] | None = None
    best_revenue: int = 0
    second_best_revenue: int = 0
    solve_time_s: float = 0.0
    infeasibility_reason: str | None = None
    bye_teams: dict[int, str] | None = None         # matchday → bye team name (odd-K only)


# ── Problem definition ────────────────────────────────────────────────────────

@dataclass(frozen=True)
class TournamentProblem:
    """Immutable, fully-validated scheduling problem instance."""

    format: str
    teams: list[Team]
    matches: list[Match]
    venue_slots: list[VenueSlot]
    matchday_dates: list[date]      # length = K-1, pre-assigned calendar dates
    matchday_gap_days: int
    matches_per_matchday: int       # = K // 2
    has_bye: bool = False           # True when odd-K was padded with BYE team

    @property
    def match_lookup(self) -> dict[tuple[int, int], Match]:
        lookup = {}
        for m in self.matches:
            i, j = m.home.id, m.away.id
            lookup[(i, j)] = m
            lookup[(j, i)] = m
        return lookup

    def validate(self) -> None:
        """Raises InfeasibilityError if the problem is structurally infeasible."""
        K = len(self.teams)
        n_matchdays = K - 1
        mpm = K // 2

        if len(self.matchday_dates) < n_matchdays:
            raise InfeasibilityError(
                f"Structural infeasibility: only {len(self.matchday_dates)} feasible "
                f"date(s) available for {n_matchdays} matchdays "
                f"(gap ≥ {self.matchday_gap_days} days). "
                "Expand the date range, reduce min_matchday_gap_days, or add "
                "more days_available."
            )
        if len(self.venue_slots) < mpm:
            raise InfeasibilityError(
                f"Structural infeasibility: only {len(self.venue_slots)} venue-slot(s), "
                f"but {mpm} simultaneous matches required per matchday "
                f"({K} teams ÷ 2). "
                "Add more fields or time-slots."
            )
        expected_matches = K * (K - 1) // 2
        if len(self.matches) != expected_matches:
            raise InfeasibilityError(
                f"Match count mismatch: expected {expected_matches} for {K} teams, "
                f"got {len(self.matches)}."
            )
        if self.has_bye:
            logger.info(
                "Problem validated ✓  format=%s  teams=%d (incl. BYE)  "
                "matches=%d  matchdays=%d  venue_slots=%d  matches/matchday=%d  "
                "bye_mode=ON",
                self.format, K, len(self.matches), n_matchdays,
                len(self.venue_slots), mpm,
            )
        else:
            logger.info(
                "Problem validated ✓  format=%s  teams=%d  matches=%d  "
                "matchdays=%d  venue_slots=%d  matches/matchday=%d",
                self.format, K, len(self.matches), n_matchdays,
                len(self.venue_slots), mpm,
            )
