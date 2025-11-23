"""
nqueens.solver_minconflicts
Purpose: High-performance Min-Conflicts solver for the N-Queens problem, with a Numba-accelerated inner loop.

API:
- solve_minconflicts(n, seed=42, max_steps=300000, k_sample=64, restart_limit=10, nbhd_width=0)

Returns:
- (pos, stats)
      pos   : np.ndarray[int32]; column -> row positions
      stats : dict with keys:
         "n", "solved", "steps", "restarts",
         "k_sample", "nbhd_width", "seed", "max_steps"

Core representation (from common_arrays.py):
    pos[c]          = row index of queen in column c
    row[r]          = number of queens in row r
    diag1[r-c+base] = number of queens on negatively sloped diagonal (↘)
    diag2[r+c]      = number of queens on positively sloped diagonal (↙)
        where base = n - 1 and diag arrays have length 2*n - 1.

Contributors:
- Anton Nemchinski
"""

# External Imports
from typing import Dict, Tuple

import numpy as np
from numba import njit

# Internal Imports
from nqueens.common_arrays import alloc_state
from nqueens.sampling import rng_randint, k_sample_rows, neighborhood_rows
from nqueens.validation import rebuild_counters, is_valid_solution


# Internal helper utilities (Numba-jitted)

@njit(cache=True)
def _column_conflicts(
   c: int,
   pos: np.ndarray,
   row: np.ndarray,
   diag1: np.ndarray,
   diag2: np.ndarray,
   base: int,
) -> int:
   """
   Return the number of conflicts involving the queen in column c.

   Conflicts arise from sharing:
   - the same row
   - the same negatively sloped diagonal (r - c)
   - the same positively sloped diagonal (r + c)

   Since this queen is counted once in each counter, subtract 1 from each.
   """
   c_i = int(c)
   r = int(pos[c_i])

   total = (int(row[r]) - 1) \
         + (int(diag1[r - c_i + base]) - 1) \
         + (int(diag2[r + c_i]) - 1)

   return int(total)


@njit(cache=True)
def _conflicts_if_move(
   c: int,
   r: int,
   pos: np.ndarray,
   row: np.ndarray,
   diag1: np.ndarray,
   diag2: np.ndarray,
   base: int,
) -> int:
   """
   Return the number of conflicts if the queen in column c were placed at row r.
   If r is the queen's current row subtract this queen's own contribution.
   """
   c_i = int(c)
   r_i = int(r)

   current_r = int(pos[c_i])
   cost = int(row[r_i]) \
        + int(diag1[r_i - c_i + base]) \
        + int(diag2[r_i + c_i])

   # If we "move" to the same square, subtract this queen's own 3 counts
   if r_i == current_r:
      cost -= 3

   return int(cost)


@njit(cache=True)
def _pick_conflicting_column(
   pos: np.ndarray,
   row: np.ndarray,
   diag1: np.ndarray,
   diag2: np.ndarray,
   n: int,
   base: int,
   rng_state: np.ndarray,
) -> int:
   """
   Pick a column whose queen is currently in conflict, if any.
   """
   # Try a few random probes first
   for _ in range(8):
      c = int(rng_randint(rng_state, n))
      if _column_conflicts(c, pos, row, diag1, diag2, base) > 0:
         return c

   # Deterministic full scan with stride 1, starting at a random column.
   c0 = int(rng_randint(rng_state, n))
   c = c0
   for _ in range(n):
      if _column_conflicts(c, pos, row, diag1, diag2, base) > 0:
         return c
      c += 1
      if c == n:
         c = 0

   # No conflicts anywhere according to our counters
   return -1


# Internal Min-Conflicts kernel (Numba @njit) KEY FUNCTION

@njit(cache=True, fastmath=True)
def _minconflicts_kernel(
   pos: np.ndarray,
   row: np.ndarray,
   diag1: np.ndarray,
   diag2: np.ndarray,
   n: int,
   max_steps: int,
   k_sample: int,
   nbhd_width: int,
   seed: int,
) -> Tuple[int, int]:
   """
   Run Min-Conflicts on the given state for at most max_steps iterations
   Params:
   - pos, row, diag1, diag2 : int32 arrays; Board state; updated in-place!!!!!!!
   - n : int; Board size
   - max_steps : int; Maximum number of single-queen moves allowed
   - k_sample : int; Number of random candidate rows when nbhd_width == 0
   - nbhd_width : int; If > 0, use a neighborhood [r0-width to r0+width] around
      the current row; otherwise use k-sampling (or full domain if k_sample>=n).
   - seed : int; Seed for the internal RNG (each restart uses a different seed)

   Returns:
   - solved_flag : int  (1 if solved, 0 otherwise)
   - steps_used  : int  (number of steps performed)
   """
   base = n - 1  # Shift for diagonal indexing (prevents negative indices)

   # RNG state: single uint64 value in a 1-element array (Numba compatibility)
   rng_state = np.empty(1, dtype=np.uint64)
   rng_state[0] = np.uint64(seed) | np.uint64(1)  # ensure non-zero

   # Candidate buffer for row choices
   max_candidates = k_sample
   local_count = 0
   if nbhd_width > 0:
      local_count = 2 * nbhd_width + 1
   if local_count > max_candidates:
      max_candidates = local_count
   if max_candidates < n:   # allow full-domain search if needed
      max_candidates = n
   if max_candidates < 1:
      max_candidates = 1

   candidates = np.empty(max_candidates, dtype=np.int32)

   for step in range(max_steps):
      # Pick a conflicting column
      c = _pick_conflicting_column(pos, row, diag1, diag2, n, base, rng_state)
      if c == -1:
         # Defensive: double-check counters to avoid false positives
         conflict_found = False
         for r_idx in range(n):
            if row[r_idx] > 1:
               conflict_found = True
               break
         if not conflict_found:
            for i in range(diag1.shape[0]):
               if diag1[i] > 1 or diag2[i] > 1:
                  conflict_found = True
                  break

         if not conflict_found:
               # No conflicts anywhere, then solution is found
               return 1, step

      current_r = int(pos[c])

      # Build candidate row set
      if nbhd_width > 0:
         # Use local neighborhood around current row
         k = neighborhood_rows(current_r, nbhd_width, n, candidates)
      else:
         if k_sample > 0 and k_sample < n:
            # Use k-sampling of rows
            k = k_sample_rows(n, k_sample, rng_state, candidates)
         else:
            # Fall back to full domain search over all n rows
            k = n
            for idx in range(n):
               candidates[idx] = idx

      if k <= 0:
         # Should not happen
         raise RuntimeError("No candidate rows generated")

      # Choose best row among candidates, with random tie-breaking
      best_r = int(candidates[0])
      best_cost = _conflicts_if_move(c, best_r, pos, row, diag1, diag2, base)

      for i in range(1, k):
         r = int(candidates[i])
         cost = _conflicts_if_move(c, r, pos, row, diag1, diag2, base)
         if cost < best_cost:
            best_cost = cost
            best_r = r
         elif cost == best_cost:
            # Randomly break ties to avoid cycles
            if rng_randint(rng_state, 2) == 0:
               best_r = r

      # Apply move (if it actually changes row) in O(1)
      if best_r != current_r:
         # Remove old queen
         row[current_r] -= 1
         diag1[current_r - c + base] -= 1
         diag2[current_r + c] -= 1

         # Place at new row
         pos[c] = best_r
         row[best_r] += 1
         diag1[best_r - c + base] += 1
         diag2[best_r + c] += 1
   # Not solved within budget
   return 0, max_steps


# Public API Python wrapper

def solve_minconflicts(
   n: int,
   seed: int = 42,
   max_steps: int = 100000000,
   k_sample: int = 512,
   restart_limit: int = 10,
   nbhd_width: int = 0,
   structured_init: bool = True,
) -> Tuple[np.ndarray, Dict]:
   """
   Public entry point for the Min-Conflicts solver.
   - allocates an initial board via alloc_state(...)
   - runs the Numba kernel with restarts (different seeds)
   - performs a final correctness check
   - returns (pos, stats)

   Parameters
   - n : int; Board size.
   - seed : int; Base seed; each restart uses seed + restart_idx.
   - max_steps : int; Maximum number of steps per restart.
   - k_sample : int; Number of candidate rows for k-sampling.
   - restart_limit : int; Maximum number of restarts.
   - nbhd_width : int; If > 0, use neighborhood search; else k-sampling.
   - structured_init : bool;
      If True, start from a low-conflict structured position;
      if False, from a random initial position.
    
   Returns:
   pos : np.ndarray[int32]; Final column->row assignment.
   stats : dict; Descriptive statistics
   """
   if n <= 0:
      raise ValueError("n must be positive")

   total_steps = 0
   restarts = 0
   solved = False

   # Initial state (allocated as mutable numpy arrays for numba compatibility)
   pos, row, diag1, diag2 = alloc_state(n, seed=seed, structured=structured_init)

   while restarts <= restart_limit:
      kernel_seed = seed + restarts

      # Run Min-Conflicts kernel
      solved_flag, steps_used = _minconflicts_kernel(
         pos,
         row,
         diag1,
         diag2,
         int(n),
         int(max_steps),
         int(k_sample),
         int(nbhd_width),
         int(kernel_seed),
         )
      total_steps += int(steps_used)

      if solved_flag == 1 and is_valid_solution(pos):
         solved = True
         break

      # Restart if allowed: reinitialize board state
      restarts += 1
      if restarts <= restart_limit:
         pos, row, diag1, diag2 = alloc_state(
             n, seed=seed + restarts, structured=structured_init
         )

   # Final correctness validation (for stats only)
   is_valid = bool(is_valid_solution(pos))

   stats: Dict[str, object] = {
      "n": int(n),
      "solved": bool(solved and is_valid),
      "steps": int(total_steps),
      "restarts": int(restarts),
      "k_sample": int(k_sample),
      "nbhd_width": int(nbhd_width),
      "seed": int(seed),
      "max_steps": int(max_steps),
   }
   return pos, stats


# Minimal self-test (run this file directly)


if __name__ == "__main__":
   # Quick sanity check for small/medium n
   for n_test in (8, 20, 50, 100, 10000, 100000, 1000000):
      print(f"\nSolving n={n_test} with Min-Conflicts.")
      pos, stats = solve_minconflicts(
         n_test,
         seed=123,
         max_steps=100000000,
         k_sample=512,       # use k-sampling; set to 0 for full-domain
         restart_limit=5,
         nbhd_width=0,
         structured_init=True,
      )
      print("  stats:", stats)
      if stats["solved"]:
         row2, d1_2, d2_2 = rebuild_counters(pos)
         assert np.all(row2 <= 1)
         assert np.all(d1_2 <= 1)
         assert np.all(d2_2 <= 1)
         print("  solution is valid.")
      else:
         # For debugging, you can also print total conflicts:
         from nqueens.validation import count_total_conflicts
         print("  solver did NOT find a solution within limits.")
         print("  remaining conflicts:", count_total_conflicts(pos))
