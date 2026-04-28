"""main.py — CLI entrypoint for scheduling_fucho_games.

Usage (from project root with venv active):
    python -m scheduling_fucho_games --config assets/example_input.yml \\
        --algorithm cpsat --output assets/example_output.yml

    python -m scheduling_fucho_games --config assets/example_input.yml \\
        --algorithm genetic --output assets/example_output.yml
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Any

import yaml

from .model import BYE_TEAM_ID, InfeasibilityError, ScheduledMatch, SolverResult
from .yml_loader import load_problem

logger = logging.getLogger(__name__)

SUPPORTED_ALGORITHMS = ("cpsat", "genetic", "kempe")


# ── Output serialisation ──────────────────────────────────────────────────────

def _serialise_schedule(
    schedule: list[ScheduledMatch] | None,
    bye_teams: dict[int, str] | None = None,
    round_name_map: dict[int, str] | None = None,
) -> Any:
    """Convert a schedule to a YAML-serialisable dict."""
    if not schedule:
        return None

    # Group by matchday
    by_day: dict[int, list[ScheduledMatch]] = {}
    for sm in schedule:
        by_day.setdefault(sm.matchday, []).append(sm)

    matchdays_out = []
    for d_idx in sorted(by_day):
        matches_in_day = sorted(
            by_day[d_idx], key=lambda s: (s.venue_slot.field_id, s.venue_slot.slot_start)
        )
        day_dict: dict[str, Any] = {
            "matchday": d_idx + 1,
            "date": str(matches_in_day[0].matchday_date),
        }
        if round_name_map and d_idx in round_name_map:
            day_dict["round"] = round_name_map[d_idx]
        day_dict["matches"] = [
            {
                "match": sm.match.label(),
                "field": sm.venue_slot.field_name,
                "time_slot": (
                    f"{sm.venue_slot.slot_start.strftime('%H:%M')}"
                    f"-{sm.venue_slot.slot_end.strftime('%H:%M')}"
                ),
                "revenue": sm.venue_slot.revenue,
            }
            for sm in matches_in_day
        ]
        if bye_teams and d_idx in bye_teams:
            day_dict["bye_team"] = bye_teams[d_idx]
        matchdays_out.append(day_dict)
    return matchdays_out


def _strip_bye_matches(
    schedule: list[ScheduledMatch] | None,
) -> tuple[list[ScheduledMatch] | None, dict[int, str]]:
    """Remove BYE-involving matches, returning cleaned schedule and bye map."""
    if not schedule:
        return None, {}

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


def _build_output(
    algorithm: str,
    result: SolverResult,
    round_name_map: dict[int, str] | None = None,
) -> dict[str, Any]:
    out: dict[str, Any] = {
        "algorithm": algorithm,
        "status": result.status,
        "solve_time_s": round(result.solve_time_s, 4),
        "infeasibility_reason": result.infeasibility_reason,
        "best_solution": {
            "total_revenue": result.best_revenue,
            "schedule": _serialise_schedule(
                result.best_solution, result.bye_teams, round_name_map,
            ),
        }
        if result.best_solution
        else None,
        "second_best_solution": {
            "total_revenue": result.second_best_revenue,
            "schedule": _serialise_schedule(
                result.second_best_solution, result.bye_teams, round_name_map,
            ),
        }
        if result.second_best_solution
        else None,
    }
    return out


# ── CLI ───────────────────────────────────────────────────────────────────────

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="scheduling_fucho_games",
        description="Football match scheduler — CP-SAT and Genetic Algorithm solvers.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python -m scheduling_fucho_games "
            "--config assets/example_input.yml --algorithm cpsat\n"
            "  python -m scheduling_fucho_games "
            "--config assets/example_input.yml --algorithm genetic\n"
        ),
    )
    p.add_argument(
        "--config",
        required=True,
        metavar="PATH",
        help="Path to the YAML input file (relative to CWD or absolute).",
    )
    p.add_argument(
        "--algorithm",
        required=True,
        choices=SUPPORTED_ALGORITHMS,
        metavar=f"{{{','.join(SUPPORTED_ALGORITHMS)}}}",
        help="Solver to use.",
    )
    p.add_argument(
        "--output",
        default="assets/example_output.yml",
        metavar="PATH",
        help="Output YAML path (default: assets/example_output.yml).",
    )
    p.add_argument(
        "--timeout",
        type=float,
        default=60.0,
        metavar="SECONDS",
        help="CP-SAT wall-clock time limit (default: 60s, ignored for genetic).",
    )
    p.add_argument(
        "--pop-size",
        type=int,
        default=200,
        dest="pop_size",
        help="GA population size (default: 300, ignored for cpsat).",
    )
    p.add_argument(
        "--generations",
        type=int,
        default=1500,
        help="GA number of generations (default: 600, ignored for cpsat).",
    )
    p.add_argument(
        "--seed",
        type=int,
        default=42,
        help="GA random seed (default: 42, ignored for cpsat).",
    )
    p.add_argument(
        "--cx-pb",
        type=float,
        default=0.8,
        help="GA crossover probability (default: 0.7, range [0,1]).",
    )
    p.add_argument(
        "--mut-pb",
        type=float,
        default=0.2,
        help="GA individual mutation probability (default: 0.2, range [0,1]).",
    )
    p.add_argument(
        "--kempe-mut-pb",
        type=float,
        default=0.15,
        help="Kempe chain mutation probability (default: 0.3, kempe only).",
    )
    p.add_argument(
        "--venue-mut-pb",
        type=float,
        default=0.1,
        help="Venue mutation probability (default: 0.3, kempe only).",
    )
    p.add_argument(
        "--tournament-k",
        type=int,
        default=7,
        help="GA selection pressure / tournament size (default: 7).",
    )
    p.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        dest="log_level",
        help="Logging verbosity (default: INFO).",
    )
    return p


def cli_main(argv: list[str] | None = None) -> int:
    """Parse arguments, solve, write output.  Returns exit code."""
    parser = _build_parser()
    args = parser.parse_args(argv)

    # Configure logging
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    # ── Load problem ──────────────────────────────────────────────────────────
    try:
        problem = load_problem(args.config)
    except FileNotFoundError as exc:
        logger.error("Config not found: %s", exc)
        return 1
    except (ValueError, InfeasibilityError) as exc:
        logger.error("Problem configuration error: %s", exc)
        return 1
    except Exception as exc:  # noqa: BLE001
        logger.exception("Unexpected error loading config: %s", exc)
        return 1

    # ── Dispatch solver ───────────────────────────────────────────────────────
    # Knockout only supports CP-SAT
    if problem.format == "knockout" and args.algorithm != "cpsat":
        logger.error(
            "Knockout format only supports the 'cpsat' algorithm; got '%s'.",
            args.algorithm,
        )
        return 1

    result: SolverResult
    try:
        if args.algorithm == "cpsat":
            from .algorithms import cpsat
            result = cpsat.solve(problem, timeout_s=args.timeout)
        elif args.algorithm == "genetic":
            from .algorithms import genetic_algorithm as ga
            result = ga.solve(
                problem,
                pop_size=args.pop_size,
                n_gen=args.generations,
                cx_pb=args.cx_pb,
                mut_pb=args.mut_pb,
                tournament_k=args.tournament_k,
                seed=args.seed,
            )
        elif args.algorithm == "kempe":
            from .algorithms import kempe
            result = kempe.solve(
                problem,
                pop_size=args.pop_size,
                n_gen=args.generations,
                cx_pb=args.cx_pb,
                mut_pb=args.mut_pb,
                kempe_mut_pb=args.kempe_mut_pb,
                venue_mut_pb=args.venue_mut_pb,
                seed=args.seed,
            )
        else:
            logger.error(
                "Unknown algorithm '%s'. Supported: %s",
                args.algorithm,
                SUPPORTED_ALGORITHMS,
            )
            return 1
    except ImportError as exc:
        logger.error("Solver import failed: %s", exc)
        return 1
    except Exception as exc:  # noqa: BLE001
        logger.exception("Solver raised an unexpected error: %s", exc)
        return 1

    # ── Post-process: strip BYE matches (odd-K) ────────────────────────────
    if problem.has_bye:
        best_clean, bye_map = _strip_bye_matches(result.best_solution)
        result.best_solution = best_clean
        result.bye_teams = bye_map
        # Recompute revenue excluding BYE matches
        if best_clean:
            result.best_revenue = sum(sm.venue_slot.revenue for sm in best_clean)

        second_clean, _ = _strip_bye_matches(result.second_best_solution)
        result.second_best_solution = second_clean
        if second_clean:
            result.second_best_revenue = sum(sm.venue_slot.revenue for sm in second_clean)

    # ── Build round labels (knockout) ───────────────────────────────────────────
    round_name_map: dict[int, str] | None = None
    if problem.format == "knockout" and problem.knockout_rounds:
        round_name_map = {}
        for rnd in problem.knockout_rounds:
            for md_idx in rnd.matchday_indices:
                round_name_map[md_idx] = rnd.name

    # ── Write output ──────────────────────────────────────────────────────────
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_data = _build_output(args.algorithm, result, round_name_map)

    try:
        with output_path.open("w", encoding="utf-8") as fh:
            yaml.dump(output_data, fh, default_flow_style=False, allow_unicode=True, sort_keys=False)
        logger.info("Output written to: %s", output_path.resolve())
    except OSError as exc:
        logger.error("Failed to write output: %s", exc)
        return 1

    # ── Summary ───────────────────────────────────────────────────────────────
    if result.status in ("OPTIMAL", "FEASIBLE"):
        print(
            f"\n✓  Status        : {result.status}\n"
            f"   Algorithm     : {args.algorithm.upper()}\n"
            f"   Best revenue  : ${result.best_revenue:,}\n"
            f"   Second-best   : ${result.second_best_revenue:,}\n"
            f"   Solve time    : {result.solve_time_s:.2f}s\n"
            f"   Output        : {output_path.resolve()}\n"
        )
        return 0
    else:
        print(
            f"\n✗  Status        : {result.status}\n"
            f"   Reason        : {result.infeasibility_reason}\n",
            file=sys.stderr,
        )
        return 2
