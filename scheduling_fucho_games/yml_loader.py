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
    BYE_TEAM_ID,
    SUPPORTED_FORMATS,
    InfeasibilityError,
    KnockoutConfig,
    Match,
    Team,
    TournamentProblem,
    VenueSlot,
)
from .utils import (
    assign_matchday_dates,
    generate_knockout_bracket,
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

    return load_problem_from_dict(raw)

def load_problem_from_dict(raw: dict[str, Any]) -> TournamentProblem:
    """Parse dictionary input and return a validated :class:`TournamentProblem`."""
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

        # Seed: required for knockout, optional for round_robin
        raw_seed = t.get("seed")
        seed = int(raw_seed) if raw_seed is not None else None
        teams.append(Team(id=tid, name=tname, seed=seed))

    validate_team_count(teams, fmt)

    # ── Bye-round injection (odd K) ───────────────────────────────────────────
    has_bye = False
    if fmt == "round_robin" and len(teams) % 2 != 0:
        logger.info(
            "Odd team count (%d) → adding BYE team for bye-round scheduling.",
            len(teams),
        )
        teams.append(Team(id=BYE_TEAM_ID, name="BYE"))
        has_bye = True

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

    # ── Format-specific match generation & assembly ──────────────────────────

    if fmt == "round_robin":
        K = len(teams)
        n_matchdays = K - 1
        matchday_dates = assign_matchday_dates(feasible_dates, n_matchdays, gap_days)

        pairs = generate_pairs_round_robin(teams)
        matches = [Match(id=i, home=h, away=a) for i, (h, a) in enumerate(pairs)]

        problem = TournamentProblem(
            format=fmt,
            teams=teams,
            matches=matches,
            venue_slots=venue_slots,
            matchday_dates=matchday_dates,
            matchday_gap_days=gap_days,
            matches_per_matchday=K // 2,
            has_bye=has_bye,
        )

    elif fmt == "knockout":
        # ── Validate seeds ───────────────────────────────────────────────────
        for t in teams:
            if t.seed is None:
                raise ValueError(
                    f"Team '{t.name}' (id={t.id}) missing 'seed' — "
                    "required for knockout format."
                )
        seeds = [t.seed for t in teams]
        if len(set(seeds)) != len(seeds):
            raise ValueError("Duplicate seed values found in teams.")

        # ── Parse knockout block ────────────────────────────────────────────
        raw_ko = _require(raw, "knockout", "root")
        if not isinstance(raw_ko, dict):
            raise ValueError("'knockout' must be a mapping.")

        default_legs = int(raw_ko.get("default_legs", 2))
        final_legs = int(raw_ko.get("final_legs", 1))
        third_place = str(raw_ko.get("third_place", "none")).strip().lower()
        brackets_enabled = bool(raw_ko.get("brackets", False))
        seeding = str(raw_ko.get("seeding", "top_vs_bottom")).strip().lower()

        if default_legs not in (1, 2):
            raise ValueError(f"default_legs must be 1 or 2; got {default_legs}.")
        if final_legs not in (1, 2):
            raise ValueError(f"final_legs must be 1 or 2; got {final_legs}.")
        if third_place not in ("none", "single", "double"):
            raise ValueError(
                f"third_place must be 'none', 'single', or 'double'; got '{third_place}'."
            )
        if seeding not in ("top_vs_bottom", "sequential"):
            raise ValueError(
                f"seeding must be 'top_vs_bottom' or 'sequential'; got '{seeding}'."
            )

        bracket_assignment: dict[str, list[int]] | None = None
        if brackets_enabled:
            raw_ba = _require(raw_ko, "bracket_assignment", "knockout")
            if not isinstance(raw_ba, dict):
                raise ValueError("'bracket_assignment' must be a mapping.")
            ba_a = [int(s) for s in _require(raw_ba, "A", "bracket_assignment")]
            ba_b = [int(s) for s in _require(raw_ba, "B", "bracket_assignment")]
            if len(ba_a) != len(ba_b):
                raise ValueError(
                    f"Bracket sizes must be equal; A={len(ba_a)}, B={len(ba_b)}."
                )
            all_seeds = set(ba_a) | set(ba_b)
            if all_seeds != set(seeds):
                raise ValueError(
                    "bracket_assignment must cover all team seeds exactly once."
                )
            bracket_assignment = {"A": ba_a, "B": ba_b}

        ko_config = KnockoutConfig(
            default_legs=default_legs,
            final_legs=final_legs,
            third_place=third_place,
            brackets=brackets_enabled,
            bracket_assignment=bracket_assignment,
            seeding=seeding,
        )

        # ── Generate bracket ───────────────────────────────────────────────
        all_teams, matches, ko_rounds, mpm_map = generate_knockout_bracket(
            teams, ko_config
        )

        n_matchdays = max(mpm_map.keys()) + 1
        matchday_dates = assign_matchday_dates(feasible_dates, n_matchdays, gap_days)
        max_mpm = max(mpm_map.values())

        problem = TournamentProblem(
            format=fmt,
            teams=all_teams,
            matches=matches,
            venue_slots=venue_slots,
            matchday_dates=matchday_dates,
            matchday_gap_days=gap_days,
            matches_per_matchday=max_mpm,
            knockout_config=ko_config,
            knockout_rounds=ko_rounds,
            matches_per_matchday_map=mpm_map,
        )
    else:
        raise ValueError(f"Unsupported format '{fmt}'.")

    problem.validate()
    return problem
