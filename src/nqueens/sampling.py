"""
nqueens.sampling
Purpose:
    JIT-safe helper functions for generating candidate rows and random
    integers inside the Min-Conflicts solver.

These functions are designed to be called from Numba-jitted code:
- no Python objects (lists, sets, dicts)
- only fixed-dtype NumPy arrays and scalars.

Contributors:
- Anton Nemchinski

Functions (all @njit-safe):
- k_sample_rows(n: int, k: int, rng_state) -> np.ndarray:
    Returns k unique (or nearly unique) candidate rows uniformly.
- neighborhood_rows(r0: int, width: int, n: int, out: np.ndarray) -> int:
    Writes circular window [r0-width .. r0+width] into 'out'; returns count.
- def make_rng_state(seed: int) -> np.ndarray:
    Create a length-1 uint64 array suitable for use with rng_next/rng_randint.

Notes:
- Avoids Python sets/lists; operates on preallocated numpy arrays for JIT compatibility.
"""

# Imports

from typing import Tuple
import numpy as np
from numba import njit

# Internal Helper Functions

@njit(cache=True)
def rng_next(state: np.ndarray) -> np.uint64:
    """
    Advance the RNG state and return a new 64-bit random value.

    Params:
    state : np.ndarray[uint64]; Mutable RNG state.

    Returns:
    value : np.uint64; Pseudorandom 64-bit integer.
    """
    # Xorshift* algorithm for Pseudorandom Number Generation
    x = state[0]
    x ^= (x << np.uint64(13)) & np.uint64(0xFFFFFFFFFFFFFFFF)
    x ^= (x >> np.uint64(7))
    x ^= (x << np.uint64(17)) & np.uint64(0xFFFFFFFFFFFFFFFF)
    state[0] = x
    return x


@njit(cache=True)
def rng_randint(state: np.ndarray, hi: int) -> int:
    """
    Params:
    state : np.ndarray[uint64]; RNG state.
    hi : int; Exclusive upper bound (must be > 0).

    Returns:
    r : int; Random integer in [0, hi).
    """
    if hi <= 0:
        return 0
    return int(rng_next(state) % np.uint64(hi))


# Outward-facing Functions (API)

@njit(cache=True)
def k_sample_rows(
    n: int,
    k: int,
    rng_state: np.ndarray,
    out_rows: np.ndarray,
) -> int:
    """
    Fill out_rows with up to k randomly sampled row indices in [0, n).

    Params:
    n : int; Number of rows (board size).
    k : int; Number of samples requested.
    rng_state : np.ndarray[uint64]; RNG state used for sampling.
    out_rows : np.ndarray[int32]; Preallocated array to write into. 

    Returns:
    count : int
        Number of candidates written to out_rows (<= k).

    Notes
    -----
    - This function does NOT guarantee uniqueness; duplicates are OK.
    """
    if k <= 0:
        return 0
    if k > out_rows.shape[0]:
        k = out_rows.shape[0]

    count = 0
    for _ in range(k):
        r = rng_randint(rng_state, n)
        out_rows[count] = r
        count += 1
    return count


@njit(cache=True)
def neighborhood_rows(
    center_row: int,
    width: int,
    n: int,
    out_rows: np.ndarray,
) -> int:
    """
    Fill out_rows with a circular window of rows around center_row.

    Params:
    center_row : int; The row around which to build the neighborhood (e.g., current row).
    width : int; Half-width of the window. The neighborhood spans
    n : int; Number of rows (board size).
    out_rows : np.ndarray[int32]; Preallocated array to write candidates into. Must have length >= 2*width+1.

    Returns:
    count : int; Number of candidates written to out_rows.
    """
    if width < 0:
        return 0

    count = 0
    start = center_row - width
    end = center_row + width

    for r in range(start, end + 1):
        rr = r
        if rr < 0:
            rr += n
            if rr < 0:
                # in case width > n, wrap multiple times
                rr %= n
        elif rr >= n:
            rr -= n
            if rr >= n:
                rr %= n
        out_rows[count] = rr
        count += 1

    return count


def make_rng_state(seed: int) -> np.ndarray:
    """
    Create a length-1 uint64 array suitable for use with rng_next/rng_randint.
    Params:
    seed : int; will be mixed into a non-zero 64-bit state.
    Returns:
    state : np.ndarray[uint64]; RNG state array.
        
    Notes
    - Intended to be called from Python code, not from jitted code.
    """
    # Mix the seed and force non-zero.
    x = np.uint64(seed) | np.uint64(1)
    return np.array([x], dtype=np.uint64)


# Short local testing

if __name__ == "__main__":

    # Test RNG and k_sample_rows
    n = 10
    k = 5
    rng_state = make_rng_state(123)
    out_rows = np.empty(k, dtype=np.int32)

    count = k_sample_rows(n, k, rng_state, out_rows)
    print(f"k_sample_rows: count = {count}")
    print("sampled rows:", out_rows[:count])

    assert count == k, "k_sample_rows should fill exactly k entries when k <= len(out_rows)"
    assert np.all((out_rows[:count] >= 0) & (out_rows[:count] < n)), "rows must be in [0, n)"

    # Test neighborhood_rows
    center_row = 3
    width = 2
    nbhd_buffer = np.empty(2 * width + 1, dtype=np.int32)

    nbhd_count = neighborhood_rows(center_row, width, n, nbhd_buffer)
    print(f"neighborhood_rows: count = {nbhd_count}")
    print("neighborhood rows:", nbhd_buffer[:nbhd_count])

    expected_count = 2 * width + 1
    assert nbhd_count == expected_count, "neighborhood_rows should return 2*width+1 rows"
    assert np.all((nbhd_buffer[:nbhd_count] >= 0) & (nbhd_buffer[:nbhd_count] < n)), "rows must wrap into [0, n)"

    print("sampling.py self-test passed.")