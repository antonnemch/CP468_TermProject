"""
nqueens.validation
Purpose: Solver-agnostic correctness checks and metrics.

Functions:
- rebuild_counters(pos) -> (row, d1, d2): O(n) recomputation from positions.
- is_valid_solution(pos) -> bool        : Ensures no row/diagonal conflicts remain.
- count_total_conflicts(pos) -> int     : Returns total attacking pairs for diagnostics.

Notes:
- Uses integer NumPy arrays; independent of Min-Conflicts counters to avoid bias.
"""
import numpy as np

def rebuild_counters(pos):
    '''
    Recomputation of counters for row and diagonals

    Parameter -> np array of length n. pos[c] = row index of queen in that column
    Return -> row (np.ndarray): # of queens in row
              diagonal1 (np.ndarray): # of queens in diagonal (top right to bottom left)
              diagonal2 (np.ndarray): # of queens in diagonal (top left to bottom right)

    '''
    n = pos.size
    
    row = np.zeros(n, dtype=np.int64)
    diagonal1 = np.zeros(2*n-1, dtype=np.int64)
    diagonal2 = np.zeros(2*n-1, dtype=np.int64)

    for c in range(n):
        r = pos[c]
        row[r] += 1
        diagonal1[r - c + (n - 1)] += 1
        diagonal2[r+c] += 1
    
    return row, diagonal1, diagonal2



def is_valid_solution(pos):
    '''
    Return True if valid board. Return False if row/diagonal conflicts.

    Parameter-> pos: np.ndarray representing queen rows by column
    '''
    row,diagonal1, diagonal2 = rebuild_counters(pos)

    if np.any(row >1):
        return False
    if np.any(diagonal1 > 1):
        return False
    if np.any(diagonal2 > 1):
        return False

    return True


def count_total_conflicts(pos):
    '''
    Counts the total conflicts of queen pairs

    Parameter -> pos: np.ndarray
    Returns -> int: total conflict pairs
    '''

    row,diagonal1, diagonal2 = rebuild_counters(pos)

    def combination2(arr):
        return int(np.sum(arr * (arr -1) // 2)) #Combination formula to count pairs (queens creating conflicts). arr is # of queens
    
    return combination2(row) + combination2(diagonal1) + combination2(diagonal2)

