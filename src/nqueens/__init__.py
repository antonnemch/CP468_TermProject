"""
nqueens.__init__
Purpose: Define the public API of the package.

Exports:
- solve_minconflicts(...): High-performance Min-Conflicts solver (Numba-backed).
- solve_backtracking(...): Complete CSP solver baseline for small/medium n.
"""
from __future__ import annotations

from nqueens.config import AppConfig
from nqueens.solver_minconflicts import solve_minconflicts
from nqueens.solver_backtracking import solve_backtracking

__all__ = [
    "AppConfig",
    "solve_minconflicts",
    "solve_backtracking",
]

__version__ = "0.1.0"


def main() -> None:
    """
    Entry point for `python -m nqueens`.
    """
    from nqueens.cli import main as cli_main

    cli_main()
