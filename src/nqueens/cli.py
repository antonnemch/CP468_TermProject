"""
nqueens.cli
Purpose: Command-line interface to run solvers and print JSON summaries.

Commands:
- solve    : Min-Conflicts
             flags: --n --seed --max-steps --k-sample --nbhd-width --restart-limit --json-out
- solve-bt : Backtracking
             flags: --n --time-limit --mrv/--no-mrv --lcv/--no-lcv

Responsibilities:
- Parse args, call solver, time execution, validate result, emit structured JSON.
"""
from __future__ import annotations

import argparse
from typing import List, Dict, Any

import numpy as np

from nqueens.config import AppConfig, build_config_from_args
from nqueens.solver_minconflicts import solve_minconflicts
from nqueens.solver_backtracking import solve_backtracking
from nqueens.instrumentation import Timer, summarize_run
from nqueens.validation import is_valid_solution, count_total_conflicts


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="nqueens",
        description="N-Queens solver CLI (Min-Conflicts + Backtracking).",
    )

    # ---- Required problem size ----
    parser.add_argument(
        "-n",
        "--n",
        type=int,
        required=True,
        help="Board size (number of queens / columns).",
    )

    # ---- Solver choice ----
    parser.add_argument(
        "--solver",
        choices=["minconflicts", "backtracking"],
        default="minconflicts",
        help="Which solver to use.",
    )

    # ---- Min-Conflicts options ----
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Base random seed for the Min-Conflicts solver.",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=100_000_000,
        help="Maximum number of steps per restart for Min-Conflicts.",
    )
    parser.add_argument(
        "--k-sample",
        type=int,
        default=512,
        help="Number of candidate rows for k-sampling (Min-Conflicts). "
             "0 or >= n means full-domain search.",
    )
    parser.add_argument(
        "--restart-limit",
        type=int,
        default=10,
        help="Maximum number of restarts for Min-Conflicts.",
    )
    parser.add_argument(
        "--nbhd-width",
        type=int,
        default=0,
        help="Neighborhood half-width for local search (Min-Conflicts). "
             "0 disables neighborhood search and uses k-sampling instead.",
    )
    parser.add_argument(
        "--random-init",
        action="store_true",
        help="Use random initialization instead of structured initialization "
             "for Min-Conflicts.",
    )

    # ---- Backtracking options ----
    parser.add_argument(
        "--time-limit",
        type=float,
        default=None,
        help="Time limit in seconds for backtracking (None = no limit).",
    )

    # ---- Meta options ----
    parser.add_argument(
        "-r",
        "--runs",
        type=int,
        default=1,
        help="Number of repeated runs (useful for timing/averages).",
    )
    parser.add_argument(
        "--validate",
        action="store_true",
        help="Run an extra solver-agnostic validity check on the final board(s).",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Print detailed run information.",
    )

    return parser


def _run_minconflicts(config: AppConfig, run_idx: int) -> Dict[str, Any]:
    """
    Run a single Min-Conflicts solve with instrumentation and return a summary dict.
    """
    # Adjust seed per run so repeated runs are not identical
    seed = config.seed + run_idx

    params = {
        "solver": "minconflicts",
        "n": config.n,
        "seed": seed,
        "max_steps": config.max_steps,
        "k_sample": config.k_sample,
        "restart_limit": config.restart_limit,
        "nbhd_width": config.nbhd_width,
        "structured_init": config.structured_init,
    }

    with Timer() as t:
        pos, stats = solve_minconflicts(
            n=config.n,
            seed=seed,
            max_steps=config.max_steps,
            k_sample=config.k_sample,
            restart_limit=config.restart_limit,
            nbhd_width=config.nbhd_width,
            structured_init=config.structured_init,
        )

    solved = bool(stats.get("solved", False))
    steps = int(stats.get("steps", 0))
    restarts = int(stats.get("restarts", 0))

    # Extra validation if requested
    if config.validate:
        valid = bool(is_valid_solution(pos))
    else:
        valid = solved

    # Attach solver stats to params for richer summarize_run output
    params.update(stats)

    summary = summarize_run(
        steps=steps,
        restarts=restarts,
        time=t,
        params=params,
        valid=valid,
    )

    summary["pos"] = pos  # keep the solution array if caller wants it
    return summary


def _run_backtracking(config: AppConfig, run_idx: int) -> Dict[str, Any]:
    """
    Run a single Backtracking solve with instrumentation and return a summary dict.
    """
    params = {
        "solver": "backtracking",
        "n": config.n,
        "time_limit": config.time_limit,
    }

    with Timer() as t:
        pos = solve_backtracking(config.n, time_limit=config.time_limit)

    solved = pos is not None

    if pos is not None and config.validate:
        valid = bool(is_valid_solution(pos))
    else:
        valid = solved

    # Backtracking implementation does not currently track steps/restarts.
    steps = None
    restarts = None

    summary = summarize_run(
        steps=steps,
        restarts=restarts,
        time=t,
        params=params,
        valid=valid,
    )

    summary["pos"] = pos
    return summary


def _pretty_print_board(pos: np.ndarray) -> None:
    """
    Print a small textual board for visualization when n is modest.
    """
    n = pos.size
    for r in range(n):
        row_str = "".join("Q" if pos[c] == r else "." for c in range(n))
        print(row_str)


def main(argv: List[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    config = build_config_from_args(args)

    if config.verbose:
        print(">> Configuration:")
        print(config)
        print()

    summaries: List[dict] = []

    for run_idx in range(config.runs):
        if config.verbose and config.runs > 1:
            print(f"=== Run {run_idx + 1}/{config.runs} ===")

        if config.solver == "minconflicts":
            summary = _run_minconflicts(config, run_idx)
        elif config.solver == "backtracking":
            summary = _run_backtracking(config, run_idx)
        else:
            raise ValueError(f"Unknown solver: {config.solver!r}")

        summaries.append(summary)

        if config.verbose:
            print(f"Run {run_idx + 1} summary (without pos):")
            shallow = {k: v for k, v in summary.items() if k != "pos"}
            print(shallow)
            print()

    # ---- Aggregate + print high-level summary ----
    solved_count = sum(1 for s in summaries if s.get("valid"))
    print(f"Ran {config.runs} run(s) for n={config.n} using solver='{config.solver}'.")
    print(f"Valid (solved) runs: {solved_count}/{config.runs}")

    times = [s["time"] for s in summaries if isinstance(s.get("time"), (int, float))]
    if times:
        avg_time = sum(times) / len(times)
        print(f"Average time over valid entries: {avg_time:.6f} s")

    # For single-run, optionally show board + conflict count if unsolved
    if config.runs == 1:
        summary = summaries[0]
        pos = summary.get("pos")
        if isinstance(pos, np.ndarray):
            if config.verbose and config.n <= 20:
                print("\nSolution board (Q = queen, . = empty):")
                _pretty_print_board(pos)

            if not summary.get("valid", False):
                # For debugging: show remaining conflicts if any
                conflicts = count_total_conflicts(pos)
                print(f"\nWARNING: board is not valid; remaining conflicts = {conflicts}")
        else:
            print("\nNo solution found (pos is None).")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
