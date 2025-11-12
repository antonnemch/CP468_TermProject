"""
nqueens.sampling
Purpose: JIT-safe row candidate generation for Min-Conflicts.

Functions (all @njit-safe):
- k_sample_rows(n: int, k: int, rng_state) -> np.ndarray:
    Returns k unique (or nearly unique) candidate rows uniformly.
- neighborhood_rows(r0: int, width: int, n: int, out: np.ndarray) -> int:
    Writes circular window [r0-width .. r0+width] into 'out'; returns count.

Notes:
- Avoid Python sets/lists; operate on preallocated arrays for JIT compatibility.
"""
