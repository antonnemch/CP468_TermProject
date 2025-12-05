"""
nqueens.heuristics
Purpose: Backtracking heuristics used by the CSP solver.

Functions:
- mrv_select_var(domains)                       : index of variable with fewest legal values
- degree_tiebreak(vars, constraints)           : prefer most-constraining variable
- lcv_order_values(var, domains, constraints)  : iterate least-constraining first
- forward_check(var, value, domains, constraints): prune downstream domains
"""

def mrv_select_var(domains):
    """Return the index odf the variable with the smallest non-empty domain."""
    best = None
    best_size = float('inf')

    for i, dom in enumerate(domains):
        size = len(dom)

       
        if size == 0:
            return None
        if size < best_size:
            best = i
            best_size = size

    return best


def degree_tiebreak(varss, constraints):
    """
    For tied MRV vars pick the one with highest degree (most constsraints).
    """
    best = None
    best_degree = -1

    for v in varss:
        deg = 0
        for u in varss:
            if u != v:
                deg += 1   

        if deg > best_degree:   
            best = v
            best_degree = deg   

    return best


def lcv_order_values(var, domains, constraints):
    """
    Order var's domain values by how manyy values they eliminate from neighbors.
    """
    pairs = []  # (val, count of values eliminated)
    for value in domains[var]:
        count = 0
        for other_var in range(len(domains)):
            if other_var == var:

                continue
            for v2 in list(domains[other_var]):  # iterate oveer copy
                if not constraints(var, value, other_var, v2):

                    count += 1
        pairs.append((value, count))
    
    # Sort by fewest eliminationss
    pairs.sort(key=lambda p: p[1])
    return [val for val, _ in pairs]


def forward_check(var, value, domains, constraints):
    
    pruned = []
    for other_var in range(len(domains)):
        if other_var == var:
            continue

        to_remove = []
        for v2 in list(domains[other_var]):
            if not constraints(var, value, other_var, v2):
                to_remove.append(v2)
        if to_remove:
            
            for val in to_remove:
                domains[other_var].discard(val)
            pruned.append((other_var, to_remove))
            if not domains[other_var]:
                return None
    return pruned
