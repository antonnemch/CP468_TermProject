"""
nqueens.heuristics
Purpose: Backtracking heuristics used by the CSP solver.

Functions:
- mrv_select_var(domains)                       : index of variable with fewest legal values
- degree_tiebreak(vars, constraints)           : prefer most-constraining variable
- lcv_order_values(var, domains, constraints)  : iterate least-constraining first
- forward_check(var, value, domains, constraints): prune downstream domains

Notes:
- Keep pure-Python; performance needs are modest (small/medium n).
"""
