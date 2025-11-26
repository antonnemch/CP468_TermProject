"""
nqueens.solver_backtracking
Purpose: Complete CSP solver baseline using MRV/LCV and Forward Checking.

Function:
- solve_backtracking(n: int, time_limit: float | None = None) -> np.ndarray | None

Model:
- Variables: columns; Domain: rows [0..n-1]
- Constraints: unique rows; unique diagonals (r-c, r+c)

Heuristics:
- MRV (fewest legal values), Degree (most-constraining tie-break),
  LCV (least-constraining value), Forward Checking (prune downstream)

Implementation notes:
- Use bitmasks (ints) for rows and diagonals to prune quickly.
- Iterative DFS optional to avoid recursion depth issues.
- Target: correctness/insight for small/medium n (not for million-scale).
"""

import time
import numpy as np
from .heuristics import (
    mrv_select_var,
    degree_tiebreak,
    lcv_order_values,
    forward_check,
)

def constraints(v1, r1, v2, r2):
    """True if (v1=r1) and (v2=r2) is legal and false otherwise"""
  
    if r1 is None or r2 is None:
      return True
    if r1 == r2:
      return False
    if abs(r1-r2) == abs(v1-v2):
      return False
      
    return True

def solve_backtracking(n, time_limit = None):
    """Return solution to N-queens as array or None if not solveable"""
  
    start = time.time()
    # Setup domains and assignments for each n
    domains = [set(range(n)) for _ in range(n)]
    assignment = [None]*n

  # helper func to undo values in a specific variables domain that were previously pruned
    def undo_pruned(pruned):
        for var, vals in pruned: 
            domains[var].update(vals)
          
    # core algorithm
    def backtrack():
      
      if time_limit and time.time() - start > time_limit:
        return False

      var = mrv_select_var(domains)
      if var is None:
        return False

        # if all variables are assigned --> solved
        if all(len(dom) == 1 for dom in domains):
          
            for i, dom in enumerate(domains):
                assignment[i] =  next(iter(dom))
            return True

        return False

    if backtrack():
      # Explicitly convert to int type
      return np.array(assignment,dtype=int)
    return None
