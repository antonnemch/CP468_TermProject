"""
nqueens.common_arrays
Purpose: Typed array allocation and initialization (Numba-friendly).

Functions:
- alloc_state(n, seed, structured) -> (pos, row, d1, d2):
    * Allocates np.int32 arrays (C-contiguous).
    * Initializes pos (random or structured one-per-column).
    * Fills row/diag counters in O(n).
- init_counters_from_pos(pos) -> (row, d1, d2): Counter rebuild helper.

Indexing:
- diag1 index: r - c + (n-1)
- diag2 index: r + c

Notes:
- Keep dtypes as np.int32 for Numba performance and compact memory.
"""
