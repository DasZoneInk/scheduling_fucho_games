"""yml_loader.py — YAML input parser → validated TournamentProblem.

Schema (canonical):

    format: "round_robin"
    teams:
      - {id: 1, name: "Team 1"}
      ...
    fields:
      - id: 1
        name: "Field 1"
        time_slots:
          - range: "20:00-21:00"
            revenue: 100
          ...
    constraints:
      min_matchday_gap_days: 7
      inclusion:
        start_date: "2022-01-01"
        end_date: "2022-03-01"
        days_available: ["Friday", "Saturday", "Sunday"]
      exclusion:
        holidays: true
        holiday_country: "MX"
        specific_dates: []
"""

from __future__ import annotations

import logging
from datetime import date
from pathlib import Path
from typing import Any

import yaml

from .model import (
    SUPPORTED_FORMATS,
    InfeasibilityError,
    Match,
    Team,
    TournamentProblem,
    VenueSlot,
)
from .utils import (
    assign_matchday_dates,
    generate_pairs_round_robin,
    get_feasible_dates,
    get_holidays,
    parse_time_range,
    validate_team_count,
)

logger = logging.getLogger(__name__)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _require(mapping: dict, key: str, context: str) -> Any:
    """Return mapping[key] or raise ValueError with a clear diagnostic."""
    if key not in mapping:
        raise ValueError(f"Missing required key '{key}' in {context}.")
    return mapping[key]


def _parse_date(value: Any, field_name: str) -> date:
    try:
        return date.fromisoformat(str(value))
    except (ValueError, TypeError) as exc:
        raise ValueError(
            f"Invalid date '{value}' for '{field_name}'. Expected YYYY-MM-DD."
        ) from exc


# ── Public API ────────────────────────────────────────────────────────────────

def load_problem(config_path: str | Path) -> TournamentProblem:
    """Parse *config_path* (YAML) and return a validated :class:`TournamentProblem`.

    Raises:
        FileNotFoundError: Config file does not exist.
        ValueError: Any schema or semantic validation failure.
        InfeasibilityError: The problem is structurally infeasible.
    """
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path.resolve()}")

    logger.info("Loading config: %s", path.resolve())

    with path.open(encoding="utf-8") as fh:
        try:
            raw: dict[str, Any] = yaml.safe_load(fh)
        except yaml.YAMLError as exc:
            raise ValueError(f"YAML parse error in '{path}': {exc}") from exc

    if not isinstance(raw, dict):
        raise ValueError(
            f"Config root must be a YAML mapping; got {type(raw).__name__}."
        )

    # ── Tournament format ─────────────────────────────────────────────────────
    fmt = str(_require(raw, "format", "root")).strip().lower()
    if fmt not in SUPPORTED_FORMATS:
        raise ValueError(
            f"Unsupported format '{fmt}'. "
            f"Supported formats: {sorted(SUPPORTED_FORMATS)}."
        )

    # ── Teams ─────────────────────────────────────────────────────────────────
    raw_teams = _require(raw, "teams", "root")
    if not isinstance(raw_teams, list) or not raw_teams:
        raise ValueError("'teams' must be a non-empty list.")

    teams: list[Team] = []
    seen_ids: set[int] = set()
    for i, t in enumerate(raw_teams):
        if not isinstance(t, dict):
            raise ValueError(
                f"teams[{i}] must be a mapping; got {type(t).__name__}."
            )
        try:
            tid = int(_require(t, "id", f"teams[{i}]"))
            tname = str(_require(t, "name", f"teams[{i}]")).strip()
        except (TypeError, ValueError) as exc:
            raise ValueError(f"teams[{i}] has invalid fields: {exc}") from exc
        if tid in seen_ids:
            raise ValueError(f"Duplicate team id {tid} at teams[{i}].")
        seen_ids.add(tid)
        teams.append(Team(id=tid, name=tname))

    validate_team_count(teams, fmt)

    # ── Fields / VenueSlots ───────────────────────────────────────────────────
    raw_fields = _require(raw, "fields", "root")
    if not isinstance(raw_fields, list) or not raw_fields:
        raise ValueError("'fields' must be a non-empty list.")

    venue_slots: list[VenueSlot] = []
    seen_field_ids: set[int] = set()
    for fi, fld in enumerate(raw_fields):
        if not isinstance(fld, dict):
            raise ValueError(
                f"fields[{fi}] must be a mapping; got {type(fld).__name__}."
            )
        fid = int(_require(fld, "id", f"fields[{fi}]"))
        fname = str(_require(fld, "name", f"fields[{fi}]")).strip()
        if fid in seen_field_ids:
            raise ValueError(f"Duplicate field id {fid} at fields[{fi}].")
        seen_field_ids.add(fid)

        raw_slots = fld.get("time_slots", [])
        if not isinstance(raw_slots, list):
            raise ValueError(f"fields[{fi}].time_slots must be a list.")
        if not raw_slots:
            logger.warning("fields[%d] ('%s') has no time_slots — skipping.", fi, fname)
            continue

        for si, slot in enumerate(raw_slots):
            if not isinstance(slot, dict):
                raise ValueError(
                    f"fields[{fi}].time_slots[{si}] must be a mapping."
                )
            rng = str(_require(slot, "range", f"fields[{fi}].time_slots[{si}]"))
            try:
                rev = int(_require(slot, "revenue", f"fields[{fi}].time_slots[{si}]"))
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    f"fields[{fi}].time_slots[{si}].revenue must be an integer: {exc}"
                ) from exc
            if rev < 0:
                raise ValueError(
                    f"fields[{fi}].time_slots[{si}].revenue must be ≥ 0; got {rev}."
                )
            start_t, end_t = parse_time_range(rng)
            venue_slots.append(
                VenueSlot(
                    field_id=fid,
                    field_name=fname,
                    slot_start=start_t,
                    slot_end=end_t,
                    revenue=rev,
                )
            )

    if not venue_slots:
        raise ValueError("No venue-slots found across all fields.")

    # ── Constraints ───────────────────────────────────────────────────────────
    raw_constraints = raw.get("constraints", {})
    if not isinstance(raw_constraints, dict):
        raise ValueError("'constraints' must be a mapping.")

    gap_days = int(raw_constraints.get("min_matchday_gap_days", 7))
    if gap_days < 0:
        raise ValueError("min_matchday_gap_days must be ≥ 0.")

    # -- Inclusion block -------------------------------------------------------
    inc = raw_constraints.get("inclusion", {})
    if not isinstance(inc, dict):
        raise ValueError(
            "'constraints.inclusion' must be a mapping with keys: "
            "start_date, end_date, days_available. "
            "(Hint: flatten the list-of-dicts format to a plain mapping.)"
        )
    start_date = _parse_date(
        _require(inc, "start_date", "constraints.inclusion"), "inclusion.start_date"
    )
    end_date = _parse_date(
        _require(inc, "end_date", "constraints.inclusion"), "inclusion.end_date"
    )
    if end_date < start_date:
        raise ValueError(
            f"inclusion.end_date ({end_date}) must be ≥ start_date ({start_date})."
        )

    raw_days = _require(inc, "days_available", "constraints.inclusion")
    if not isinstance(raw_days, list) or not raw_days:
        raise ValueError("'inclusion.days_available' must be a non-empty list.")

    # -- Exclusion block -------------------------------------------------------
    exc_block = raw_constraints.get("exclusion", {})
    if not isinstance(exc_block, dict):
        exc_block = {}

    use_holidays: bool = bool(exc_block.get("holidays", False))
    holiday_country: str = str(exc_block.get("holiday_country", "MX"))
    specific_raw = exc_block.get("specific_dates") or []
    if not isinstance(specific_raw, list):
        raise ValueError("'exclusion.specific_dates' must be a list or null.")

    excluded: set[date] = set()
    if use_holidays:
        years = list(range(start_date.year, end_date.year + 1))
        excluded |= get_holidays(holiday_country, years)
    for idx, sd in enumerate(specific_raw):
        excluded.add(_parse_date(sd, f"exclusion.specific_dates[{idx}]"))

    # ── Date expansion ────────────────────────────────────────────────────────
    feasible_dates = get_feasible_dates(start_date, end_date, raw_days, excluded)
    K = len(teams)
    n_matchdays = K - 1
    matchday_dates = assign_matchday_dates(feasible_dates, n_matchdays, gap_days)

    # ── Match generation ──────────────────────────────────────────────────────
    pairs = generate_pairs_round_robin(teams)
    matches = [Match(id=i, home=h, away=a) for i, (h, a) in enumerate(pairs)]

    # ── Assemble and validate ─────────────────────────────────────────────────
    problem = TournamentProblem(
        format=fmt,
        teams=teams,
        matches=matches,
        venue_slots=venue_slots,
        matchday_dates=matchday_dates,
        matchday_gap_days=gap_days,
        matches_per_matchday=K // 2,
    )
    problem.validate()
    return problem
