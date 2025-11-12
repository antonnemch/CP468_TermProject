"""
nqueens.solver_minconflicts
Purpose: High-performance Min-Conflicts solver with a single Numba JIT kernel.

Public API:
- solve_minconflicts(
    n: int,
    seed: int,
    max_steps: int,
    k_sample: int,
    restart_limit: int,
    nbhd_width: int | None = None,
  ) -> tuple[np.ndarray, dict]
  Orchestrates allocation, restart loop, kernel call, validation, and metrics.

Core JIT kernel (specification):
- @njit(cache=True, nopython=True, fastmath=True)
  def minconflicts_kernel(pos, row, d1, d2, n, max_steps, k_sample, nbhd_width, seed)
    Main loop up to max_steps:
      1) Pick a conflicting column c:
         - Strategy A: sample random columns until one is conflicting (few retries)
         - Strategy B: stride-scan columns; pick first with conflicts
      2) Choose best row r* for c (never scan all rows):
         - If nbhd_width > 0: scan window [r0-w .. r0+w] (wrap)
         - Else: K-sample rows (k_sample ~ 32–64)
         - Conflict score for placing (c, r):
             cost = row[r] + d1[r - c + base] + d2[r + c]
             (subtract self-count if r == pos[c])
         - Break ties randomly (prevents cycles)
      3) Move queen in O(1):
         rold = pos[c]; rnew = r*
         row[rold]--; d1[rold - c + base]--; d2[rold + c]--
         pos[c] = rnew
         row[rnew]++; d1[rnew - c + base]++; d2[rnew + c]++
      4) Periodic zero-conflict check → early exit solved

RNG in JIT:
- Implement a tiny LCG/xorshift (uint64 state) for reproducible fast ints.
- Do NOT use np.random inside @njit loops.

NUMBA PERFORMANCE RULES (critical):
- No Python objects in-kernel; single tight kernel; minimal branches
- All arrays np.int32, C-contiguous; precompute base = n-1; avoid modulo/div
- Candidate selection via K-sampling or neighborhood window
- Parallelize restarts at Python level if needed (not inner loop)
- Validate result by rebuilding counters (validation.is_valid_solution)

Returns:
- Kernel: (solved_flag, steps_used); Wrapper: (pos, metrics dict).
"""
