"""
nqueens.validation
Purpose: Solver-agnostic correctness checks and metrics.

Functions:
- rebuild_counters(pos) -> (row, d1, d2): O(n) recomputation from positions.
- is_valid_solution(pos) -> bool        : Ensures no row/diagonal conflicts remain.
- count_total_conflicts(pos) -> int     : Returns total attacking pairs for diagnostics.

Notes:
- Uses integer NumPy arrays; independent of Min-Conflicts counters to avoid bias.
"""
