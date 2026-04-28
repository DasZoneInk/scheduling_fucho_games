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
SUPPORTED_FORMATS: Final[frozenset[str]] = frozenset({"round_robin", "knockout"})
BYE_TEAM_ID: Final[int] = -1
PLACEHOLDER_ID_BASE: Final[int] = -100  # placeholder IDs: -100, -101, …


# ── Custom exception ──────────────────────────────────────────────────────────

class InfeasibilityError(ValueError):
    """Raised when the problem instance has no feasible solution."""


# ── Domain dataclasses ────────────────────────────────────────────────────────

@dataclass(frozen=True)
class Team:
    id: int
    name: str
    seed: int | None = None      # required for knockout, optional for round_robin

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


# ── Knockout configuration ────────────────────────────────────────────────────

@dataclass(frozen=True)
class KnockoutConfig:
    """Parsed knockout YAML block."""
    default_legs: int                                  # 1 or 2
    final_legs: int                                    # 1 or 2
    third_place: str                                   # "none" | "single" | "double"
    brackets: bool                                     # True → 2-bracket split
    bracket_assignment: dict[str, list[int]] | None    # {"A": [seeds], "B": [seeds]}
    seeding: str                                       # "top_vs_bottom" | "sequential"


@dataclass(frozen=True)
class KnockoutRound:
    """One round of a knockout tournament."""
    name: str                    # "QF", "SF", "3rd", "F"
    round_index: int             # sequential ordering
    legs: int                    # 1 or 2
    match_ids: list[int]         # IDs of matches in this round (per leg)
    matchday_indices: list[int]  # which matchday(s) this round maps to


# ── Problem definition ────────────────────────────────────────────────────────

@dataclass(frozen=True)
class TournamentProblem:
    """Immutable, fully-validated scheduling problem instance."""

    format: str
    teams: list[Team]
    matches: list[Match]
    venue_slots: list[VenueSlot]
    matchday_dates: list[date]      # pre-assigned calendar dates
    matchday_gap_days: int
    matches_per_matchday: int       # = K // 2 for round_robin; max for knockout
    has_bye: bool = False           # True when odd-K was padded with BYE team

    # ── Knockout-specific fields ──────────────────────────────────────────────
    knockout_config: KnockoutConfig | None = None
    knockout_rounds: list[KnockoutRound] | None = None
    matches_per_matchday_map: dict[int, int] | None = None  # overrides flat mpm

    @property
    def match_lookup(self) -> dict[tuple[int, int], Match]:
        lookup = {}
        for m in self.matches:
            i, j = m.home.id, m.away.id
            lookup[(i, j)] = m
            lookup[(j, i)] = m
        return lookup

    def validate(self) -> None:
        """Dispatch to format-specific validation."""
        if self.format == "round_robin":
            self._validate_round_robin()
        elif self.format == "knockout":
            self._validate_knockout()
        else:
            raise InfeasibilityError(f"Unknown format '{self.format}'.")

    # ── Round-robin validation ────────────────────────────────────────────────

    def _validate_round_robin(self) -> None:
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

    # ── Knockout validation ───────────────────────────────────────────────────

    def _validate_knockout(self) -> None:
        if self.knockout_config is None or self.knockout_rounds is None:
            raise InfeasibilityError(
                "Knockout format requires knockout_config and knockout_rounds."
            )
        if self.matches_per_matchday_map is None:
            raise InfeasibilityError(
                "Knockout format requires matches_per_matchday_map."
            )

        n_matchdays = len(self.matchday_dates)
        max_mpm = max(self.matches_per_matchday_map.values())

        if len(self.venue_slots) < max_mpm:
            raise InfeasibilityError(
                f"Structural infeasibility: only {len(self.venue_slots)} venue-slot(s), "
                f"but {max_mpm} simultaneous matches required on peak matchday. "
                "Add more fields or time-slots."
            )

        # Verify all matchday indices in rounds are within range
        for rnd in self.knockout_rounds:
            for md_idx in rnd.matchday_indices:
                if md_idx >= n_matchdays:
                    raise InfeasibilityError(
                        f"Round '{rnd.name}' references matchday {md_idx} "
                        f"but only {n_matchdays} matchday dates available."
                    )

        logger.info(
            "Problem validated ✓  format=knockout  teams=%d  matches=%d  "
            "matchdays=%d  rounds=%d  venue_slots=%d",
            len(self.teams), len(self.matches), n_matchdays,
            len(self.knockout_rounds), len(self.venue_slots),
        )
