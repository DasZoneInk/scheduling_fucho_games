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

from .model import InfeasibilityError, ScheduledMatch, SolverResult
from .yml_loader import load_problem

logger = logging.getLogger(__name__)

SUPPORTED_ALGORITHMS = ("cpsat", "genetic")


# ── Output serialisation ──────────────────────────────────────────────────────

def _serialise_schedule(schedule: list[ScheduledMatch] | None) -> Any:
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
        matchdays_out.append(
            {
                "matchday": d_idx + 1,
                "date": str(matches_in_day[0].matchday_date),
                "matches": [
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
                ],
            }
        )
    return matchdays_out


def _build_output(
    algorithm: str,
    result: SolverResult,
) -> dict[str, Any]:
    return {
        "algorithm": algorithm,
        "status": result.status,
        "solve_time_s": round(result.solve_time_s, 4),
        "infeasibility_reason": result.infeasibility_reason,
        "best_solution": {
            "total_revenue": result.best_revenue,
            "schedule": _serialise_schedule(result.best_solution),
        }
        if result.best_solution
        else None,
        "second_best_solution": {
            "total_revenue": result.second_best_revenue,
            "schedule": _serialise_schedule(result.second_best_solution),
        }
        if result.second_best_solution
        else None,
    }


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
        default=300,
        dest="pop_size",
        help="GA population size (default: 300, ignored for cpsat).",
    )
    p.add_argument(
        "--generations",
        type=int,
        default=600,
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
        default=0.7,
        help="GA crossover probability (default: 0.7, range [0,1]).",
    )
    p.add_argument(
        "--mut-pb",
        type=float,
        default=0.2,
        help="GA individual mutation probability (default: 0.2, range [0,1]).",
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

    # ── Write output ──────────────────────────────────────────────────────────
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_data = _build_output(args.algorithm, result)

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
