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
