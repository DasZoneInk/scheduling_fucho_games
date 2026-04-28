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


def _is_power_of_2(n: int) -> bool:
    return n >= 2 and (n & (n - 1)) == 0


def validate_team_count(teams: list[Any], fmt: str) -> None:
    """Raise ValueError if *teams* is incompatible with *fmt*.

    Odd cardinality for round_robin logs a warning; the caller is
    responsible for injecting a BYE team to pad to even.
    Knockout requires a power-of-2 count (4, 8, 16, 32).
    """
    if len(teams) < 2:
        raise ValueError("At least 2 teams are required.")
    if fmt == "round_robin" and len(teams) % 2 != 0:
        logger.warning(
            "Odd team count (%d) for round_robin — a BYE team will be "
            "injected automatically.",
            len(teams),
        )
    if fmt == "knockout":
        if not _is_power_of_2(len(teams)):
            raise ValueError(
                f"Knockout requires a power-of-2 team count (4, 8, 16, 32); "
                f"got {len(teams)}."
            )


# ── Knockout bracket generation ───────────────────────────────────────────────

def _round_name(n_teams: int, teams_in_round: int) -> str:
    """Return human-readable round name."""
    if teams_in_round == 2:
        return "Final"
    if teams_in_round == 4:
        return "Semi-Final"
    if teams_in_round == 8:
        return "Quarter-Final"
    if teams_in_round == 16:
        return "Round of 16"
    return f"Round of {teams_in_round}"


def _round_short(teams_in_round: int) -> str:
    if teams_in_round == 2:
        return "F"
    if teams_in_round == 4:
        return "SF"
    if teams_in_round == 8:
        return "QF"
    return f"R{teams_in_round}"


def generate_knockout_bracket(
    teams: list[Any],
    config: Any,
) -> tuple[list[Any], list[Any], list[Any], dict[int, int]]:
    """Generate full knockout bracket with placeholder teams for later rounds.

    Args:
        teams: List of Team objects with ``id``, ``name``, ``seed`` attributes.
        config: KnockoutConfig instance.

    Returns:
        (all_teams, all_matches, knockout_rounds, mpm_map)
        - all_teams: original + placeholder Team objects
        - all_matches: all Match objects across all rounds
        - knockout_rounds: list of KnockoutRound
        - mpm_map: {matchday_index: match_count}
    """
    from .model import (
        KnockoutRound,
        Match,
        PLACEHOLDER_ID_BASE,
        Team as TeamCls,
    )

    N = len(teams)
    sorted_teams = sorted(teams, key=lambda t: t.seed)

    all_teams = list(teams)
    all_matches: list[Match] = []
    rounds: list[KnockoutRound] = []
    mpm_map: dict[int, int] = {}

    match_id_counter = 0
    placeholder_id_counter = PLACEHOLDER_ID_BASE
    matchday_idx = 0

    # ── Seeded first-round pairings ──────────────────────────────────────────
    if config.brackets and config.bracket_assignment:
        seeds_a = config.bracket_assignment["A"]
        seeds_b = config.bracket_assignment["B"]
        team_by_seed = {t.seed: t for t in sorted_teams}
        bracket_a = [team_by_seed[s] for s in sorted(seeds_a)]
        bracket_b = [team_by_seed[s] for s in sorted(seeds_b)]
        pool_groups = [bracket_a, bracket_b]
    else:
        pool_groups = [sorted_teams]

    # ── Build rounds ─────────────────────────────────────────────────────────
    # Each pool is processed independently until they merge at the final.
    # For non-bracket mode, there's a single pool.

    # Track per-pool winners for next round
    pool_winners: list[list[Any]] = []

    for pool_idx, pool in enumerate(pool_groups):
        pool_size = len(pool)
        current_contenders = list(pool)
        pool_round_idx = 0

        while len(current_contenders) > 1:
            teams_in_round = len(current_contenders)
            is_final_within_pool = (teams_in_round == 2 and len(pool_groups) == 1)

            # Determine legs for this round
            if is_final_within_pool:
                legs = config.final_legs
            else:
                legs = config.default_legs

            # Generate pairings
            pairings: list[tuple[Any, Any]] = []
            if config.seeding == "top_vs_bottom":
                half = teams_in_round // 2
                for i in range(half):
                    pairings.append((current_contenders[i], current_contenders[-(i + 1)]))
            else:  # sequential
                for i in range(0, teams_in_round, 2):
                    pairings.append((current_contenders[i], current_contenders[i + 1]))

            # Determine round name
            if is_final_within_pool:
                rnd_name = "Final"
            elif len(pool_groups) > 1 and teams_in_round == 2:
                bracket_label = "A" if pool_idx == 0 else "B"
                rnd_name = f"Semi-Final (Bracket {bracket_label})"
            else:
                base_name = _round_name(N, teams_in_round * len(pool_groups))
                if len(pool_groups) > 1:
                    bracket_label = "A" if pool_idx == 0 else "B"
                    rnd_name = f"{base_name} (Bracket {bracket_label})"
                else:
                    rnd_name = base_name

            # Create matches
            round_match_ids: list[int] = []
            next_round_contenders: list[Any] = []
            matchday_indices: list[int] = []

            for leg in range(legs):
                leg_matchday = matchday_idx
                matchday_indices.append(leg_matchday)
                leg_match_count = 0

                for pair_idx, (t_a, t_b) in enumerate(pairings):
                    if leg == 0:
                        home, away = t_a, t_b
                    else:
                        home, away = t_b, t_a  # swap home/away for leg 2

                    m = Match(id=match_id_counter, home=home, away=away)
                    all_matches.append(m)
                    round_match_ids.append(match_id_counter)
                    match_id_counter += 1
                    leg_match_count += 1

                    # Create placeholder winner for next round (only on leg 1)
                    if leg == 0:
                        rnd_short = _round_short(teams_in_round * len(pool_groups))
                        placeholder_name = f"W-{rnd_short}{pair_idx + 1}"
                        if len(pool_groups) > 1:
                            bracket_label = "A" if pool_idx == 0 else "B"
                            placeholder_name = f"W-{rnd_short}{bracket_label}{pair_idx + 1}"
                        placeholder = TeamCls(
                            id=placeholder_id_counter,
                            name=placeholder_name,
                        )
                        placeholder_id_counter -= 1
                        all_teams.append(placeholder)
                        next_round_contenders.append(placeholder)

                mpm_map[leg_matchday] = leg_match_count
                matchday_idx += 1

            rnd_label = rnd_name
            if legs > 1:
                rnd_label = f"{rnd_name} (Legs 1-{legs})"

            rounds.append(KnockoutRound(
                name=rnd_label,
                round_index=len(rounds),
                legs=legs,
                match_ids=round_match_ids,
                matchday_indices=matchday_indices,
            ))

            current_contenders = next_round_contenders
            pool_round_idx += 1

        pool_winners.append(current_contenders)

    # ── Cross-bracket Final + Third Place ─────────────────────────────────────
    if len(pool_groups) == 2:
        finalist_a = pool_winners[0][0]
        finalist_b = pool_winners[1][0]

        # Third-place match (losers of bracket semi-finals)
        if config.third_place != "none":
            # Create placeholder losers
            loser_a = TeamCls(id=placeholder_id_counter, name="L-SFA")
            placeholder_id_counter -= 1
            loser_b = TeamCls(id=placeholder_id_counter, name="L-SFB")
            placeholder_id_counter -= 1
            all_teams.extend([loser_a, loser_b])

            tp_legs = 2 if config.third_place == "double" else 1
            tp_match_ids: list[int] = []
            tp_matchday_indices: list[int] = []

            for leg in range(tp_legs):
                home = loser_a if leg == 0 else loser_b
                away = loser_b if leg == 0 else loser_a
                m = Match(id=match_id_counter, home=home, away=away)
                all_matches.append(m)
                tp_match_ids.append(match_id_counter)
                match_id_counter += 1
                tp_matchday_indices.append(matchday_idx)
                mpm_map[matchday_idx] = 1
                matchday_idx += 1

            rounds.append(KnockoutRound(
                name="Third Place",
                round_index=len(rounds),
                legs=tp_legs,
                match_ids=tp_match_ids,
                matchday_indices=tp_matchday_indices,
            ))

        # Final
        final_legs = config.final_legs
        final_match_ids: list[int] = []
        final_matchday_indices: list[int] = []

        for leg in range(final_legs):
            home = finalist_a if leg == 0 else finalist_b
            away = finalist_b if leg == 0 else finalist_a
            m = Match(id=match_id_counter, home=home, away=away)
            all_matches.append(m)
            final_match_ids.append(match_id_counter)
            match_id_counter += 1
            final_matchday_indices.append(matchday_idx)
            mpm_map[matchday_idx] = 1
            matchday_idx += 1

        rounds.append(KnockoutRound(
            name="Final" + (f" (Legs 1-{final_legs})" if final_legs > 1 else ""),
            round_index=len(rounds),
            legs=final_legs,
            match_ids=final_match_ids,
            matchday_indices=final_matchday_indices,
        ))

    # ── Single-pool third-place ──────────────────────────────────────────────
    elif config.third_place != "none" and N > 2:
        loser_a = TeamCls(id=placeholder_id_counter, name="L-SF1")
        placeholder_id_counter -= 1
        loser_b = TeamCls(id=placeholder_id_counter, name="L-SF2")
        placeholder_id_counter -= 1
        all_teams.extend([loser_a, loser_b])

        tp_legs = 2 if config.third_place == "double" else 1
        tp_match_ids_sp: list[int] = []
        tp_md_sp: list[int] = []

        for leg in range(tp_legs):
            home = loser_a if leg == 0 else loser_b
            away = loser_b if leg == 0 else loser_a
            m = Match(id=match_id_counter, home=home, away=away)
            all_matches.append(m)
            tp_match_ids_sp.append(match_id_counter)
            match_id_counter += 1

            # Third-place and final share the last matchday in single-pool
            # but we give third-place its own day before the final
            tp_md_sp.append(matchday_idx)
            mpm_map[matchday_idx] = 1
            matchday_idx += 1

        rounds.insert(-1, KnockoutRound(
            name="Third Place",
            round_index=len(rounds) - 1,
            legs=tp_legs,
            match_ids=tp_match_ids_sp,
            matchday_indices=tp_md_sp,
        ))
        # Re-index final round
        final_rnd = rounds[-1]
        rounds[-1] = KnockoutRound(
            name=final_rnd.name,
            round_index=len(rounds) - 1,
            legs=final_rnd.legs,
            match_ids=final_rnd.match_ids,
            matchday_indices=final_rnd.matchday_indices,
        )

    return all_teams, all_matches, rounds, mpm_map
