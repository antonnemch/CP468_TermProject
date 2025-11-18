"""
nqueens.common_arrays
Purpose: Typed array allocation and initialization (Numba-friendly) for the N-Queens Min-Conflicts solver.
These functions are used to initialize the data structure that represent the board state in an efficient manner.

Contributors: 
- Anton Nemchinski

Stores values:
- pos[c] = row index of the queen in column c
- row[r] = number of queens in row r
- diag1[r - c + B] = number of queens on negatively sloped diagonal
    - where B = n - 1 and both diag arrays have length 2*n - 1.
- diag2[r + c] = number of queens on positively sloped diagonal
    - where B = n - 1 and both diag arrays have length 2*n - 1.

Functions:
- init_counters_from_pos(pos) -> (row, d1, d2)
    Rebuild counters from a position array in O(n).

- alloc_state(n, seed, structured) -> (pos, row, d1, d2)
    Allocate np.int32 arrays and initialize them with either a
    structured or random one-queen-per-column placement.

Notes:
- Keep dtypes as np.int32 for Numba performance and compact memory.
"""


from typing import Tuple
import numpy as np


def initialize_counters_from_positions(pos: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    build row and diagonal counters from a given position array.
    pos : np.ndarray
        Shape (n,), integer dtype. pos[c] = row index of the queen in column c.
    
    Returns:
    row: np.ndarray[int32]     shape (n)
    diag1: np.ndarray[int32]   shape (2*n - 1) index = r - c + (n-1)
    diag2: np.ndarray[int32]   shape (2*n - 1) index = r + c
    """
    n = int(pos.shape[0])

    # Ensure int32 and C-contiguous for Numba
    pos = np.ascontiguousarray(pos, dtype=np.int32)

    # Allocate counters
    row = np.zeros(n, dtype=np.int32)
    diag1 = np.zeros(2 * n - 1, dtype=np.int32)
    diag2 = np.zeros(2 * n - 1, dtype=np.int32)

    # Fill counters
    base = n - 1
    for c in range(n):
        r = int(pos[c])
        row[r] += 1
        diag1[r - c + base] += 1
        diag2[r + c] += 1

    return row, diag1, diag2


def _structured_initial_position(n: int, shift: int = 0) -> np.ndarray:
    """
    Construct a low-conflict initial placement: pos[c] = (c + shift) % n
    - Exactly one queen per column.
    - Exactly one queen per row.
    - Only diagonal conflicts are possible.
    """
    pos = np.empty(n, dtype=np.int32)
    for c in range(n):
        pos[c] = (c + shift) % n
    return pos


def _random_initial_position(n: int, rng: np.random.Generator) -> np.ndarray:
    """
    Construct an initial placement with one queen per column and
    a random row in [0, n) for each column.
    """
    # independent random row for each column
    pos = rng.integers(low=0, high=n, size=n, dtype=np.int32)
    return pos


def alloc_state(
    n: int,
    seed: int = 42,
    structured: bool = True,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Allocate and initialize the state arrays for N-Queens.

    Params:
    n : int  = Board size (n x n) and number of queens.
    seed : int, default 42
    structured : bool, default True
        If True use structured initial positions: [c] = (c + shift) % n.
        If False use random rows per column.

    Returns:
    - pos[c] = row index of the queen in column c
    - row[r] = number of queens in row r
    - diag1[r - c + B] = number of queens on negatively sloped diagonal
        - where B = n - 1 and both diag arrays have length 2*n - 1.
    - diag2[r + c] = number of queens on positively sloped diagonal
        - where B = n - 1 and both diag arrays have length 2*n - 1.

    Note: All arrays are np.int32 and C-contiguous, which is important for Numba.
    """
    n = int(n)
    if n <= 0:
        raise ValueError("n must be positive")

    rng = np.random.default_rng(seed)

    if structured:
        # Simple way to vary the permutation slightly with the seed.
        shift = seed % max(n, 1)
        pos = _structured_initial_position(n, shift=shift)
    else:
        pos = _random_initial_position(n, rng)

    row, d1, d2 = initialize_counters_from_positions(pos)
    return pos, row, d1, d2


if __name__ == "__main__":
    # check
    n = 8
    pos, row, d1, d2 = alloc_state(n, seed=1, structured=True)
    print("pos:", pos)
    print("row:", row)
    print("d1 :", d1)
    print("d2 :", d2)

    # Rebuild counters from pos and compare
    row2, d1_2, d2_2 = initialize_counters_from_positions(pos)
    assert np.array_equal(row, row2)
    assert np.array_equal(d1, d1_2)
    assert np.array_equal(d2, d2_2)
    print("Self-test passed.")
