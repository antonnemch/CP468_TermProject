"""
nqueens.config
Purpose: Centralized defaults and toggles.

Defaults:
- DEFAULT_MAX_STEPS_PER_RESTART
- DEFAULT_K_SAMPLE
- DEFAULT_NBHD_WIDTH
- DEFAULT_RESTART_LIMIT
- DEFAULT_SEED

Notes:
- Use for consistent experiments and CLI defaults.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass
class AppConfig:
    """
    Central configuration for running the N-Queens solvers from the CLI.

    This is intentionally simple: it collects all parameters that the CLI
    parses and passes them on to the appropriate solver(s).

    Fields are chosen to match:
    - nqueens.solver_minconflicts.solve_minconflicts(...)
    - nqueens.solver_backtracking.solve_backtracking(...)
    """

    # Problem size
    n: int

    # Solver selection
    #   "minconflicts" -> nqueens.solver_minconflicts.solve_minconflicts
    #   "backtracking" -> nqueens.solver_backtracking.solve_backtracking
    solver: str = "minconflicts"

    # ---- Min-Conflicts parameters ----
    seed: int = 42
    max_steps: int = 100_000_000
    k_sample: int = 512
    restart_limit: int = 10
    nbhd_width: int = 0
    structured_init: bool = True  # False means random initialization

    # ---- Backtracking parameters ----
    # Time limit in seconds; None means no explicit limit.
    time_limit: Optional[float] = None

    # ---- Meta / CLI-only options ----
    runs: int = 1              # number of repeated runs
    validate: bool = False     # run extra validation after solving
    verbose: bool = False      # print detailed info


def build_config_from_args(args) -> AppConfig:
    """
    Build an AppConfig instance from an argparse.Namespace.
    """

    return AppConfig(
        n=args.n,
        solver=args.solver,
        seed=args.seed,
        max_steps=args.max_steps,
        k_sample=args.k_sample,
        restart_limit=args.restart_limit,
        nbhd_width=args.nbhd_width,
        structured_init=not args.random_init,
        time_limit=args.time_limit,
        runs=args.runs,
        validate=args.validate,
        verbose=args.verbose,
    )
