"""
benchmarks.bench_minconflicts

Purpose:
    Run a systematic grid search over Min-Conflicts hyperparameters and log
    detailed results to a CSV file. Designed to be interrupt-safe and
    resume-friendly.

Usage (from project root, venv active):

    python -m benchmarks.bench_minconflicts

Notes:
    - Hyperparameter grid is defined in GRID_CONFIG in __main__.
    - One warmup run is performed before timing (to trigger Numba compilation).
    - max_steps is always proportional to n via max_step_ratio.
    - Each configuration is defined by:
        (n, candidate_selector, candidate_count, structured_init, seed)
      and is run exactly ONCE.
    - Results are appended to a CSV; previously completed configs are skipped
      based on that 5-tuple key.
"""

from __future__ import annotations

import csv
import os
import time
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Set, Tuple

import numpy as np

from nqueens.solver_minconflicts import solve_minconflicts
from nqueens.validation import count_total_conflicts


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ExperimentConfig:
    """
    One logical configuration of hyperparameters.

    Each config is uniquely identified by:
        (n, candidate_selector, candidate_count, structured_init, seed)

    We also store a monotonically increasing config_id for convenience
    in logging and downstream analysis.
    """
    config_id: int
    n: int
    candidate_selector: str          # "k_sample" or "nbhd"
    candidate_count: int             # 32, 128, ...
    structured_init: int             # 0 or 1 (kept as int for CSV)
    seed: int                        # exact seed used by the solver


@dataclass(frozen=True)
class RunKey:
    """
    Unique identifier for a single run in the grid.

    Used for resume-safety: if a row with this key already exists in the CSV,
    we skip re-running it.
    """
    n: int
    candidate_selector: str
    candidate_count: int
    structured_init: int
    seed: int


# ---------------------------------------------------------------------------
# CSV helpers
# ---------------------------------------------------------------------------

FIELDNAMES: List[str] = [
    # Identifiers
    "config_id",
    "n",
    "candidate_selector",
    "candidate_count",
    "structured_init",
    "seed",

    # Solver hyperparameters (derived)
    "k_sample",
    "nbhd_width",
    "max_restart",
    "max_step_ratio",
    "max_steps",
    "time_limit_sec",

    # Outcomes from solver
    "solved",
    "steps",
    "restarts",
    "total_time_sec",
    "final_conflicts",

    # Stats returned by solver (redundant but useful)
    "stats_n",
    "stats_solved",
    "stats_steps",
    "stats_restarts",
    "stats_k_sample",
    "stats_nbhd_width",
    "stats_seed",
    "stats_max_steps",

    # Diagnostics
    "timeout_hit",        # 1 if total_time_sec >= time_limit_sec
    "exception_occurred",
    "exception_message",
]


def ensure_parent_dir(path: str) -> None:
    parent = os.path.dirname(path)
    if parent and not os.path.exists(parent):
        os.makedirs(parent, exist_ok=True)


def read_existing_runs(csv_path: str) -> Set[RunKey]:
    """
    Read an existing CSV (if any) and collect RunKey objects for runs
    that have already been completed.
    """
    done: Set[RunKey] = set()
    if not os.path.exists(csv_path):
        return done

    with open(csv_path, mode="r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                rk = RunKey(
                    n=int(row["n"]),
                    candidate_selector=row["candidate_selector"],
                    candidate_count=int(row["candidate_count"]),
                    structured_init=int(row["structured_init"]),
                    seed=int(row["seed"]),
                )
                done.add(rk)
            except Exception:
                # If a row is malformed, skip it rather than crashing
                continue
    return done


def open_csv_for_append(csv_path: str) -> Tuple[csv.DictWriter, Any]:
    """
    Open the CSV for appending. If the file is new or empty, write the header.
    Returns (writer, file_handle).
    """
    ensure_parent_dir(csv_path)
    file_exists = os.path.exists(csv_path)
    needs_header = not file_exists or os.path.getsize(csv_path) == 0

    f = open(csv_path, mode="a", newline="")
    writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
    if needs_header:
        writer.writeheader()
        f.flush()
    return writer, f


# ---------------------------------------------------------------------------
# Grid generation and mapping to solver parameters
# ---------------------------------------------------------------------------

def generate_config_grid(grid_config: Dict[str, Any]) -> List[ExperimentConfig]:
    """
    Enumerate all ExperimentConfig objects from GRID_CONFIG.

    Each combination of (n, candidate_selector, candidate_count,
    structured_init, seed) is treated as a separate config.
    """
    n_values: List[int] = grid_config["n_values"]
    candidate_selectors: List[str] = grid_config["candidate_selectors"]
    candidate_counts: List[int] = grid_config["candidate_counts"]
    structured_inits: List[int] = grid_config["structured_init_values"]
    seeds: List[int] = grid_config["seeds"]

    configs: List[ExperimentConfig] = []
    config_id = 0
    for n in n_values:
        for selector in candidate_selectors:
            for cand_count in candidate_counts:
                for struct in structured_inits:
                    for seed in seeds:
                        configs.append(
                            ExperimentConfig(
                                config_id=config_id,
                                n=n,
                                candidate_selector=selector,
                                candidate_count=cand_count,
                                structured_init=struct,
                                seed=seed,
                            )
                        )
                        config_id += 1
    return configs


def derive_solver_params_for_config(
    cfg: ExperimentConfig,
    max_restart: int,
    max_step_ratio: float,
) -> Dict[str, Any]:
    """
    Map an ExperimentConfig to concrete solver parameters.

    - max_steps is proportional to n via max_step_ratio.
    - candidate_selector determines whether we use k-sampling or neighborhood.
    """
    n = cfg.n
    candidate_selector = cfg.candidate_selector
    candidate_count = cfg.candidate_count
    structured_init = bool(cfg.structured_init)

    # max_steps proportional to n
    max_steps = int(max_step_ratio * float(n))
    if max_steps <= 0:
        max_steps = n  # minimal safeguard

    # Determine k_sample and nbhd_width
    if candidate_selector == "k_sample":
        k_sample = int(min(candidate_count, n))
        nbhd_width = 0
    elif candidate_selector == "nbhd":
        # Choose nbhd_width so that roughly 2*width + 1 ≈ candidate_count
        nbhd_width = candidate_count // 2
        if nbhd_width < 0:
            nbhd_width = 0
        if 2 * nbhd_width + 1 > n:
            nbhd_width = max((n - 1) // 2, 0)
        k_sample = 0  # selection entirely via neighborhood in this mode
    else:
        raise ValueError(f"Unknown candidate_selector: {candidate_selector}")

    return {
        "n": n,
        "seed": int(cfg.seed),
        "k_sample": int(k_sample),
        "nbhd_width": int(nbhd_width),
        "max_steps": int(max_steps),
        "max_restart": int(max_restart),
        "structured_init": structured_init,
    }


# ---------------------------------------------------------------------------
# Warmup
# ---------------------------------------------------------------------------

def perform_warmup() -> None:
    """
    Run a single small solve to trigger Numba compilation.
    This run is not written to the CSV.
    """
    print("[warmup] Running small warmup solve for Numba compilation...")
    _pos, _stats = solve_minconflicts(
        n=1000,
        seed=0,
        max_steps=10_000,
        k_sample=32,
        restart_limit=1,
        nbhd_width=0,
        structured_init=True,
    )
    print("[warmup] Done.")


# ---------------------------------------------------------------------------
# Main grid execution
# ---------------------------------------------------------------------------

def run_grid_search(grid_config: Dict[str, Any]) -> None:
    """
    Main driver: perform warmup, then run the grid search over all configurations.
    """
    output_csv: str = grid_config["output_csv"]
    max_restart: int = int(grid_config["max_restart"])
    max_step_ratio: float = float(grid_config["max_step_ratio"])
    time_limit_sec: float = float(grid_config["timeout_per_config_sec"])

    perform_warmup()

    # Load already-completed runs (for resume-safety)
    done_keys: Set[RunKey] = read_existing_runs(output_csv)
    print(f"[resume] Loaded {len(done_keys)} existing run records.")

    configs = generate_config_grid(grid_config)
    print(f"[grid] Total ExperimentConfig objects: {len(configs)}")

    writer, fh = open_csv_for_append(output_csv)
    try:
        for cfg in configs:
            run_key = RunKey(
                n=cfg.n,
                candidate_selector=cfg.candidate_selector,
                candidate_count=cfg.candidate_count,
                structured_init=cfg.structured_init,
                seed=cfg.seed,
            )

            if run_key in done_keys:
                print(
                    f"\n[config] id={cfg.config_id}, n={cfg.n}, "
                    f"selector={cfg.candidate_selector}, cand_count={cfg.candidate_count}, "
                    f"structured_init={cfg.structured_init}, seed={cfg.seed}"
                )
                print("  [skip] Already done.")
                continue

            print(
                f"\n[config] id={cfg.config_id}, n={cfg.n}, "
                f"selector={cfg.candidate_selector}, cand_count={cfg.candidate_count}, "
                f"structured_init={cfg.structured_init}, seed={cfg.seed}"
            )

            solver_params = derive_solver_params_for_config(
                cfg=cfg,
                max_restart=max_restart,
                max_step_ratio=max_step_ratio,
            )

            print(
                f"  [run] seed={solver_params['seed']}, "
                f"k_sample={solver_params['k_sample']}, "
                f"nbhd_width={solver_params['nbhd_width']}, "
                f"max_steps={solver_params['max_steps']}"
            )

            # Execute solver and measure time
            run_start = time.perf_counter()
            solved = 0
            steps = -1
            restarts = -1
            total_time_sec = 0.0
            final_conflicts = -1
            exception_occurred = 0
            exception_message = ""
            stats: Dict[str, Any] = {}

            try:
                pos, stats = solve_minconflicts(
                    n=solver_params["n"],
                    seed=solver_params["seed"],
                    max_steps=solver_params["max_steps"],
                    k_sample=solver_params["k_sample"],
                    restart_limit=solver_params["max_restart"],
                    nbhd_width=solver_params["nbhd_width"],
                    structured_init=solver_params["structured_init"],
                )
                run_end = time.perf_counter()
                total_time_sec = float(run_end - run_start)

                solved = int(bool(stats.get("solved", False)))
                steps = int(stats.get("steps", -1))
                restarts = int(stats.get("restarts", -1))

                # Independent conflict count (for safety)
                final_conflicts = int(count_total_conflicts(pos))

            except Exception as e:
                run_end = time.perf_counter()
                total_time_sec = float(run_end - run_start)
                solved = 0
                steps = -1
                restarts = -1
                final_conflicts = -1
                exception_occurred = 1
                exception_message = repr(e)
                stats = {
                    "n": cfg.n,
                    "solved": False,
                    "steps": -1,
                    "restarts": -1,
                    "k_sample": solver_params["k_sample"],
                    "nbhd_width": solver_params["nbhd_width"],
                    "seed": solver_params["seed"],
                    "max_steps": solver_params["max_steps"],
                }
                print(f"    [error] Exception in run: {exception_message}")

            timeout_hit_flag = 1 if total_time_sec >= time_limit_sec else 0

            # Build CSV row
            row = {
                "config_id": cfg.config_id,
                "n": cfg.n,
                "candidate_selector": cfg.candidate_selector,
                "candidate_count": cfg.candidate_count,
                "structured_init": cfg.structured_init,
                "seed": cfg.seed,

                "k_sample": solver_params["k_sample"],
                "nbhd_width": solver_params["nbhd_width"],
                "max_restart": solver_params["max_restart"],
                "max_step_ratio": max_step_ratio,
                "max_steps": solver_params["max_steps"],
                "time_limit_sec": time_limit_sec,

                "solved": solved,
                "steps": steps,
                "restarts": restarts,
                "total_time_sec": total_time_sec,
                "final_conflicts": final_conflicts,

                "stats_n": int(stats.get("n", -1)),
                "stats_solved": int(bool(stats.get("solved", False))),
                "stats_steps": int(stats.get("steps", -1)),
                "stats_restarts": int(stats.get("restarts", -1)),
                "stats_k_sample": int(stats.get("k_sample", solver_params["k_sample"])),
                "stats_nbhd_width": int(stats.get("nbhd_width", solver_params["nbhd_width"])),
                "stats_seed": int(stats.get("seed", solver_params["seed"])),
                "stats_max_steps": int(stats.get("max_steps", solver_params["max_steps"])),

                "timeout_hit": timeout_hit_flag,
                "exception_occurred": exception_occurred,
                "exception_message": exception_message,
            }

            writer.writerow(row)
            fh.flush()
            done_keys.add(run_key)

            print(
                f"    [done] solved={solved}, steps={steps}, "
                f"restarts={restarts}, time={total_time_sec:.4f}s, "
                f"conflicts={final_conflicts}, timeout_hit={timeout_hit_flag}"
            )

    finally:
        fh.close()
        print("\n[grid] Finished (or interrupted); CSV closed.")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # Hyperparameter grid configuration
    GRID_CONFIG: Dict[str, Any] = {
        "n_values": [10_000_000],  # problem sizes 100, 1000, 10_000, 1_000_000, 5_000_000
        "candidate_selectors": ["k_sample"],      # two modes
        "candidate_counts": [2048],         # used for both K-sampling and nbhd
        "structured_init_values": [0],                 # 0=False, 1=True
        "seeds": [123],                           # each seed = its own config #42, 123, 67
        "max_restart": 3,
        "max_step_ratio": 20,                           # max_steps = ratio * n
        "timeout_per_config_sec": 30 * 60.0,            # wall-time budget per config (used for diagnostics)
        "output_csv": "benchmarks/results/minconflicts_grid_improved.csv",
    }

    run_grid_search(GRID_CONFIG)
