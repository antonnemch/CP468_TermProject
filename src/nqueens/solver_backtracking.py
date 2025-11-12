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
