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

def solve_backtracking(n, time_limit=None):
    start = time.time()
    domains = [set(range(n)) for _ in range(n)]
    assignment = [None] * n

    def undo_pruned(pruned):
        for var, vals in pruned:
            domains[var].update(vals)

    def backtrack():
        if time_limit and time.time() - start > time_limit:
            return False

        # Check if solved (all domains size 1)
        if all(len(dom) == 1 for dom in domains):
            for i, dom in enumerate(domains):
                assignment[i] = next(iter(dom))
            return True

        # MRV
        var = mrv_select_var(domains)
        if var is None:
            return False

        # Degree tiebreak: gather ties
        min_size = min(len(dom) for dom in domains)
        ties = []
        for i, d in enumerate(domains):
            if len(d) == min_size:
                ties.append(i)

        if len(ties) > 1:
            var = degree_tiebreak(ties, constraints)

        # LCV ordering
        for value in lcv_order_values(var, domains, constraints):

            # save what we remove for undo
            removed_from_var = []
            for x in domains[var]:
                if x != value:
                    removed_from_var.append(x)

            # assign value
            domains[var] = {value}

            pruned = forward_check(var, value, domains, constraints)
            if pruned is not None:
                if backtrack():
                    return True

                # undo forward check
                undo_pruned(pruned)

            # undo assignment
            domains[var].update(removed_from_var)

        return False

    if backtrack():
        return np.array(assignment, dtype=int)

    return None



if __name__ == "__main__":
    import sys

    # simple argument handling
    n = 8
    if len(sys.argv) > 1:
        n = int(sys.argv[1])

    sol = solve_backtracking(n)

    if sol is None:
        print("No solution found.")
    else:
        print("Solution:", sol)

