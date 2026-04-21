"""conftest.py — shared pytest fixtures for scheduling_fucho_games tests."""

from __future__ import annotations

import random
from datetime import date, time, timedelta
from pathlib import Path

import pytest

from scheduling_fucho_games.model import (
    Match,
    Team,
    TournamentProblem,
    VenueSlot,
)
from scheduling_fucho_games.utils import generate_pairs_round_robin
from scheduling_fucho_games.yml_loader import load_problem

# ── Path helpers ──────────────────────────────────────────────────────────────
TESTS_DIR = Path(__file__).parent
TEST_ASSETS = TESTS_DIR / "assets"
PKG_ASSETS = TESTS_DIR.parent / "assets"


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture(scope="session")
def example_problem() -> TournamentProblem:
    """Canonical 6-team example problem loaded from the package assets."""
    return load_problem(PKG_ASSETS / "example_input.yml")


def build_stress_problem(n_teams: int = 16, seed: int = 42) -> TournamentProblem:
    """Construct a large, feasible TournamentProblem programmatically.

    No YAML file is involved — the object is built directly from model
    dataclasses so that the test stresses solver logic independently of
    the loader.

    Parameters
    ----------
    n_teams:
        Number of teams (must be even).  Produces:
        - M = n_teams * (n_teams-1) / 2  matches
        - D = n_teams - 1               matchdays
        - mpm = n_teams // 2            simultaneous matches / matchday
    seed:
        RNG seed for reproducible random revenues.

    Venue layout (always keeps V > mpm):
        n_fields = ceil(mpm / 3) + 1      (each field has 3 slots)
    """
    assert n_teams % 2 == 0, "n_teams must be even"
    rng = random.Random(seed)

    teams = [Team(id=i + 1, name=f"T{i+1:02d}") for i in range(n_teams)]
    pairs = generate_pairs_round_robin(teams)
    matches = [Match(id=i, home=h, away=a) for i, (h, a) in enumerate(pairs)]

    mpm = n_teams // 2
    slots_per_field = 3
    n_fields = (mpm + slots_per_field - 1) // slots_per_field + 1  # ≥ mpm capacity

    venue_slots: list[VenueSlot] = []
    for f in range(1, n_fields + 1):
        for s in range(slots_per_field):
            h_start = 8 + s * 2
            venue_slots.append(
                VenueSlot(
                    field_id=f,
                    field_name=f"Field {f}",
                    slot_start=time(h_start, 0),
                    slot_end=time(h_start + 1, 59),
                    revenue=rng.randint(80, 200),
                )
            )

    # Pre-assign matchday dates: every 7 days from 2023-01-02 (Monday)
    n_matchdays = n_teams - 1
    base = date(2023, 1, 2)
    matchday_dates = [base + timedelta(days=7 * i) for i in range(n_matchdays)]

    problem = TournamentProblem(
        format="round_robin",
        teams=teams,
        matches=matches,
        venue_slots=venue_slots,
        matchday_dates=matchday_dates,
        matchday_gap_days=7,
        matches_per_matchday=mpm,
    )
    problem.validate()
    return problem


@pytest.fixture(scope="session")
def stress_problem() -> TournamentProblem:
    """16-team stress-test problem (120 matches, 15 matchdays, 8 simultaneous)."""
    return build_stress_problem(n_teams=16, seed=42)
