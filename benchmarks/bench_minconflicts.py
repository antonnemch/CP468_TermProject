"""
benchmarks.bench_minconflicts
Purpose:
    Benchmark the N-Queens Min-Conflicts solver over a range of board sizes.

This script:
- runs solve_minconflicts(n, ...) for several n values and seeds
- records:
    n, trial, seed, solved, steps, restarts, total_time_sec,
    k_sample, nbhd_width, max_steps, structured_init
- writes results to a CSV file and optionally prints a summary.

Usage (from project root, with venv active):
    python -m benchmarks.bench_minconflicts \
        --ns 1000 5000 10000 \
        --trials 5 \
        --k-sample 64 \
        --restart-limit 5 \
        --max-steps 1000000 \
        --output benchmarks/minconflicts_results.csv
"""

import argparse
import csv
import os
import time
from typing import List

import numpy as np

from nqueens.solver_minconflicts import solve_minconflicts


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark the N-Queens Min-Conflicts solver."
    )
    parser.add_argument(
        "--ns",
        type=int,
        nargs="+",
        default=[100, 500, 1000, 5000, 10000],
        help="List of board sizes n to benchmark (space-separated).",
    )
    parser.add_argument(
        "--trials",
        type=int,
        default=5,
        help="Number of trials per board size.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=123,
        help="Base random seed (trial index is added on top).",
    )
    parser.add_argument(
        "--k-sample",
        type=int,
        default=64,
        help="Number of candidate rows for k-sampling.",
    )
    parser.add_argument(
        "--nbhd-width",
        type=int,
        default=0,
        help="Neighborhood half-width. 0 = pure k-sampling.",
    )
    parser.add_argument(
        "--restart-limit",
        type=int,
        default=10,
        help="Maximum number of restarts per trial.",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=1_000_000,
        help="Maximum number of steps per restart.",
    )
    parser.add_argument(
        "--structured-init",
        action="store_true",
        help="Use structured initial positions instead of random.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="benchmarks/minconflicts_results.csv",
        help="Path to CSV file for results.",
    )
    parser.add_argument(
        "--no-header",
        action="store_true",
        help="Do not write a header row to the CSV (append-only mode).",
    )
    return parser.parse_args()


def run_single_benchmark(
    n: int,
    trial_idx: int,
    base_seed: int,
    k_sample: int,
    nbhd_width: int,
    restart_limit: int,
    max_steps: int,
    structured_init: bool,
):
    """
    Run a single trial of solve_minconflicts on board size n and return
    (row_dict, solved_flag) where row_dict is ready to be written as CSV.
    """
    seed = base_seed + trial_idx

    t0 = time.perf_counter()
    pos, stats = solve_minconflicts(
        n=n,
        seed=seed,
        max_steps=max_steps,
        k_sample=k_sample,
        restart_limit=restart_limit,
        nbhd_width=nbhd_width,
        structured_init=structured_init,
    )
    t1 = time.perf_counter()

    total_time = t1 - t0
    solved = bool(stats.get("solved", False))
    steps = int(stats.get("steps", -1))
    restarts = int(stats.get("restarts", -1))

    row = {
        "n": int(n),
        "trial": int(trial_idx),
        "seed": int(seed),
        "solved": int(solved),
        "steps": steps,
        "restarts": restarts,
        "total_time_sec": float(total_time),
        "k_sample": int(k_sample),
        "nbhd_width": int(nbhd_width),
        "max_steps": int(max_steps),
        "restart_limit": int(restart_limit),
        "structured_init": int(bool(structured_init)),
    }
    return row, solved


def ensure_parent_dir(path: str) -> None:
    parent = os.path.dirname(path)
    if parent and not os.path.exists(parent):
        os.makedirs(parent, exist_ok=True)


def main() -> None:
    args = parse_args()

    ns: List[int] = sorted(set(args.ns))
    trials = args.trials

    ensure_parent_dir(args.output)

    fieldnames = [
        "n",
        "trial",
        "seed",
        "solved",
        "steps",
        "restarts",
        "total_time_sec",
        "k_sample",
        "nbhd_width",
        "max_steps",
        "restart_limit",
        "structured_init",
    ]

    write_header = not args.no_header or not os.path.exists(args.output)

    print(f"Writing results to: {args.output}")
    print(f"Board sizes: {ns}")
    print(f"Trials per n: {trials}")
    print(f"k_sample={args.k_sample}, nbhd_width={args.nbhd_width}, "
          f"restart_limit={args.restart_limit}, max_steps={args.max_steps}, "
          f"structured_init={args.structured_init}")

    with open(args.output, mode="a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()

        for n in ns:
            print(f"\n=== n = {n} ===")
            solved_count = 0
            times: List[float] = []
            steps_list: List[int] = []

            for trial in range(trials):
                row, solved = run_single_benchmark(
                    n=n,
                    trial_idx=trial,
                    base_seed=args.seed,
                    k_sample=args.k_sample,
                    nbhd_width=args.nbhd_width,
                    restart_limit=args.restart_limit,
                    max_steps=args.max_steps,
                    structured_init=args.structured_init,
                )
                writer.writerow(row)

                solved_count += int(solved)
                times.append(row["total_time_sec"])
                steps_list.append(row["steps"])

                status = "OK" if solved else "FAIL"
                print(
                    f"  trial {trial:02d}: {status}, "
                    f"time={row['total_time_sec']:.4f}s, "
                    f"steps={row['steps']}, restarts={row['restarts']}"
                )

            # Simple summary per n
            times_arr = np.array(times, dtype=np.float64)
            steps_arr = np.array(steps_list, dtype=np.int64)

            print(
                f"Summary for n={n}: "
                f"solved {solved_count}/{trials}, "
                f"avg_time={times_arr.mean():.4f}s, "
                f"median_time={np.median(times_arr):.4f}s, "
                f"avg_steps={steps_arr.mean():.1f}"
            )

    print("\nBenchmarking complete.")


if __name__ == "__main__":
    main()
