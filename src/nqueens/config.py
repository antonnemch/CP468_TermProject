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

    
    n: int

    
    solver: str = "minconflicts"

    # minconflicts
    seed: int = 42
    max_steps: int = 100_000_000
    k_sample: int = 512
    restart_limit: int = 10
    nbhd_width: int = 0
    structured_init: bool = True 

    #backtracking param
    time_limit: Optional[float] = None

    
    runs: int = 1              # num of exp runs
    validate: bool = False     # run extra validation after solving
    verbose: bool = False      # printing


def build_config_from_args(args) -> AppConfig:
    

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
