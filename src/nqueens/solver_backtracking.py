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
    if abs(r1 - r2) == abs(v1 - v2):
        return False
    return True


def solve_backtracking(n, time_limit=None):
    start = time.time()
    domains = [set(range(n)) for _ in range(n)]
    assignment = [None] * n
    
    def backtrack(depth):
        if time_limit and time.time() - start > time_limit:
            return False
        
        # Check if already complete
        if depth == n:
            return True
        
        var = None
        min_size = float('inf')
        for i in range(n):
            if assignment[i] is None and len(domains[i]) < min_size:
                min_size = len(domains[i])
                var = i
        
        if var is None or min_size == 0:
            return False
        
        for value in lcv_order_values(var, domains, constraints):
            if value not in domains[var]:
                continue
            
            # Assign
            assignment[var] = value
            old_domain = domains[var]
            domains[var] = {value}
            
            # Forward check
            pruned = forward_check(var, value, domains, constraints)
            
            if pruned is not None:
                if backtrack(depth + 1):
                    return True
                
                # Undo forward checking
                for other_var, removed_vals in pruned:
                    domains[other_var].update(removed_vals)
            
            # Undo assignment
            assignment[var] = None
            domains[var] = old_domain
        
        return False
    
    if backtrack(0):
        return np.array(assignment, dtype=int)
    return None


if __name__ == "__main__":
    import sys
    n = 8
    if len(sys.argv) > 1:
        n = int(sys.argv[1])
    sol = solve_backtracking(n)
    if sol is None:
        print("No solution found.")
    else:
        print("Solution:", sol)

