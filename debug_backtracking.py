import sys
from src.nqueens.heuristics import lcv_order_values, forward_check
from src.nqueens.solver_backtracking import constraints

def debug_solve(n):
    domains = [set(range(n)) for _ in range(n)]
    assignment = [None] * n
    call_count = [0]  # Use list to modify in nested function
    
    def backtrack(depth, indent=0):
        call_count[0] += 1
        prefix = "  " * indent
        print(f"{prefix}[Call {call_count[0]}] backtrack(depth={depth})")
        print(f"{prefix}  assignment: {assignment}")
        print(f"{prefix}  domains: {[len(d) for d in domains]}")
        
        if call_count[0] > 50:  # Stop after 50 calls to avoid spam
            print(f"{prefix}  [STOPPED - too many calls]")
            return False
        
        # Check if complete
        if depth == n:
            print(f"{prefix}  ✓ SOLUTION FOUND!")
            return True
        
        # Find unassigned var with smallest domain
        var = None
        min_size = float('inf')
        for i in range(n):
            if assignment[i] is None and len(domains[i]) < min_size:
                min_size = len(domains[i])
                var = i
        
        print(f"{prefix}  Selected var={var}, domain_size={min_size}")
        
        if var is None or min_size == 0:
            print(f"{prefix}  ✗ Dead end (var={var}, min_size={min_size})")
            return False
        
        # Get LCV ordered values
        values = lcv_order_values(var, domains, constraints)
        print(f"{prefix}  LCV ordered values for var {var}: {values}")
        
        for value in values:
            if value not in domains[var]:
                print(f"{prefix}    Skipping {value} (not in domain)")
                continue
            
            print(f"{prefix}    Trying var={var}, value={value}")
            
            # Assign
            assignment[var] = value
            old_domain = domains[var].copy()  # Make a copy
            domains[var] = {value}
            
            print(f"{prefix}      Before forward_check: {[len(d) for d in domains]}")
            
            # Forward check
            pruned = forward_check(var, value, domains, constraints)
            
            print(f"{prefix}      After forward_check: {[len(d) for d in domains]}")
            print(f"{prefix}      Pruned: {pruned}")
            
            if pruned is not None:
                result = backtrack(depth + 1, indent + 1)
                if result:
                    return True
                
                print(f"{prefix}      Backtracking... restoring domains")
                # Undo forward checking
                for other_var, removed_vals in pruned:
                    domains[other_var].update(removed_vals)
                print(f"{prefix}      After restore: {[len(d) for d in domains]}")
            else:
                print(f"{prefix}      Forward check returned None (domain wipeout)")
            
            # Undo assignment
            assignment[var] = None
            domains[var] = old_domain
            print(f"{prefix}      Undid assignment, domains: {[len(d) for d in domains]}")
        
        print(f"{prefix}  ✗ All values exhausted for var={var}")
        return False
    
    success = backtrack(0)
    print(f"\n{'='*60}")
    print(f"Total calls: {call_count[0]}")
    print(f"Result: {'SOLUTION FOUND' if success else 'NO SOLUTION'}")
    print(f"Final assignment: {assignment}")
    return assignment if success else None

if __name__ == "__main__":
    n = 4
    if len(sys.argv) > 1:
        n = int(sys.argv[1])
    debug_solve(n)
