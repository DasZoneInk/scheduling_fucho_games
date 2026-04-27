"""utils.py — Pure utility functions for date, slot, and match generation.

All functions are side-effect-free.  Domain objects are not imported here to
avoid circular dependencies; callers are responsible for constructing them.
"""

from __future__ import annotations

import logging
from datetime import date, time, timedelta
from itertools import combinations
from typing import Any

logger = logging.getLogger(__name__)

# Weekday name → Python weekday() integer (Monday = 0, Sunday = 6)
DAY_NAME_MAP: dict[str, int] = {
    "Monday": 0,
    "Tuesday": 1,
    "Wednesday": 2,
    "Thursday": 3,
    "Friday": 4,
    "Saturday": 5,
    "Sunday": 6,
}


# ── Time parsing ──────────────────────────────────────────────────────────────

def parse_time_range(range_str: str) -> tuple[time, time]:
    """Parse ``"HH:MM-HH:MM"`` into ``(start_time, end_time)``.

    Raises:
        ValueError: If the format is invalid.
    """
    try:
        raw = range_str.strip().split("-")
        if len(raw) != 2:
            raise ValueError
        sh, sm = map(int, raw[0].strip().split(":"))
        eh, em = map(int, raw[1].strip().split(":"))
        return time(sh, sm), time(eh, em)
    except (ValueError, AttributeError) as exc:
        raise ValueError(
            f"Invalid time range '{range_str}'. Expected format 'HH:MM-HH:MM'."
        ) from exc


# ── Holiday resolution ────────────────────────────────────────────────────────

def get_holidays(country: str, years: list[int]) -> set[date]:
    """Return official holiday dates for *country* across *years*.

    Falls back to empty set if the ``holidays`` library is unavailable or the
    country code is not recognised.
    """
    try:
        import holidays as hol_lib

        result: set[date] = set()
        for yr in years:
            result |= set(hol_lib.country_holidays(country, years=yr).keys())
        logger.info(
            "Loaded %d holiday(s) for '%s' years=%s", len(result), country, years
        )
        return result
    except ImportError:
        logger.warning(
            "'holidays' package not installed — skipping holiday exclusion."
        )
        return set()
    except Exception as exc:  # noqa: BLE001
        logger.warning("Could not load holidays for '%s': %s", country, exc)
        return set()


# ── Calendar expansion ────────────────────────────────────────────────────────

def get_feasible_dates(
    start: date,
    end: date,
    days_available: list[str],
    excluded_dates: set[date],
) -> list[date]:
    """Expand ``[start, end]`` to feasible calendar dates.

    Filters by:
    - day-of-week in *days_available*
    - not in *excluded_dates*

    Returns a sorted list.

    Raises:
        ValueError: If *days_available* contains an unrecognised name.
    """
    allowed: set[int] = set()
    for d in days_available:
        if d not in DAY_NAME_MAP:
            raise ValueError(
                f"Unrecognised day name '{d}'. "
                f"Valid names: {sorted(DAY_NAME_MAP)}."
            )
        allowed.add(DAY_NAME_MAP[d])

    result: list[date] = []
    current = start
    while current <= end:
        if current.weekday() in allowed and current not in excluded_dates:
            result.append(current)
        current += timedelta(days=1)

    logger.info(
        "Feasible dates: %d candidate(s) in [%s, %s]  "
        "days=%s  excluded=%d",
        len(result),
        start,
        end,
        days_available,
        len(excluded_dates),
    )
    return result


def assign_matchday_dates(
    feasible_dates: list[date],
    n_matchdays: int,
    gap_days: int,
) -> list[date]:
    """Greedily select *n_matchdays* dates, each ≥ *gap_days* apart.

    Raises:
        ValueError: With an explicit infeasibility reason if selection fails.
    """
    if not feasible_dates:
        raise ValueError(
            f"No feasible dates available for {n_matchdays} matchday(s). "
            "Expand the date range or relax availability constraints."
        )

    selected: list[date] = []
    last: date | None = None
    for d in feasible_dates:
        if last is None or (d - last).days >= gap_days:
            selected.append(d)
            last = d
            if len(selected) == n_matchdays:
                break

    if len(selected) < n_matchdays:
        raise ValueError(
            f"Cannot select {n_matchdays} matchday date(s) with gap ≥ {gap_days} day(s) "
            f"from {len(feasible_dates)} feasible date(s) "
            f"(range {feasible_dates[0]} – {feasible_dates[-1]}). "
            "Widen the date range, reduce min_matchday_gap_days, or add more "
            "days_available."
        )

    logger.info(
        "Matchday dates assigned: %s … %s  (gap ≥ %d days)",
        selected[0],
        selected[-1],
        gap_days,
    )
    return selected


# ── Match generation ──────────────────────────────────────────────────────────

def generate_pairs_round_robin(teams: list[Any]) -> list[tuple[Any, Any]]:
    """Return all unordered pairs for a single round-robin tournament.

    Total pairs = K*(K-1)/2 where K = len(teams).
    """
    return list(combinations(teams, 2))


def build_round_robin(teams: list[Any]) -> list[list[tuple[Any, Any]]]:
    """Build a 1-factorization (list of matchdays) for round-robin.

    Requires even cardinality; caller must pad with BYE if odd.
    """
    teams = list(teams)
    n = len(teams)
    if n % 2 != 0:
        raise ValueError(
            f"build_round_robin requires even team count; got {n}. "
            "Pad with BYE team before calling."
        )

    fixed = teams[-1]
    others = teams[:-1]

    D = n - 1
    matchdays = []

    for _ in range(D):
        pairs = []
        left = [fixed] + others[: (n // 2 - 1)]
        right = others[(n // 2 - 1):][::-1]

        for a, b in zip(left, right):
            pairs.append((a, b))

        matchdays.append(pairs)
        others = [others[-1]] + others[:-1]

    return matchdays


def validate_team_count(teams: list[Any], fmt: str) -> None:
    """Raise ValueError if *teams* is incompatible with *fmt*.

    Odd cardinality for round_robin logs a warning; the caller is
    responsible for injecting a BYE team to pad to even.
    """
    if len(teams) < 2:
        raise ValueError("At least 2 teams are required.")
    if fmt == "round_robin" and len(teams) % 2 != 0:
        logger.warning(
            "Odd team count (%d) for round_robin — a BYE team will be "
            "injected automatically.",
            len(teams),
        )
