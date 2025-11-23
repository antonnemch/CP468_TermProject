"""
tests.test_validation
Purpose: Unit tests for validation utilities.

Covers:
- rebuild_counters vs hand-computed small boards
- is_valid_solution on known valid/invalid states
- count_total_conflicts sanity checks
"""

import numpy as np
import pytest
from src.nqueens.validation import rebuild_counters, is_valid_solution, count_total_conflicts

def test_rebuild_counters():
    '''
    testing rebuild_counters for 4x4 board
    '''
    pos = np.array([1, 3, 0, 2])

    row, diagonal1, diagonal2 = rebuild_counters(pos)

    # Check if each row has exactly 1 queen
    assert np.array_equal(row, np.ones(4, dtype=np.int64))

    # Check diagonal 1 counts (top right to bottom left)
    # [1,3,0,2] -> [1-0+(4-1)= 4, 3-1+3= 5, 0-2+3= 1, 2-3+3= 2] -> [4,5,1,2]

    expected_diagonal1 = np.zeros(7, dtype=np.int64) # n*2-1 -> 7
    expected_diagonal1[[4,5,1,2]] = 1 # Check each diagonal has exactly 1 queen

    assert np.array_equal(diagonal1, expected_diagonal1)

    # Check diagonal 2 counts (top left to bottom right)
    # [1,3,0,2] -> [1+0= 1, 3+1= 4, 0+2 = 2, 2+3= 5] -> [1,4,2,5]

    expected_diagonal2 = np.zeros(7, dtype=np.int64)
    expected_diagonal2[[1,4,2,5]] = 1 # Check each diagonal has exactly 1 queen

    assert np.array_equal(diagonal2, expected_diagonal2)



def test_is_valid_solution():
    '''
    test is_valid_solution for known valid/invalid states
    '''

    valid = np.array([1,3,0,2])
    invalid_row= np.array([1,1,1,1]) # Row 1 has 4 queens
    invalid_diag= np.array([0,1,2,3]) # Each queen on main diagonal (top left to bottom right)


    assert is_valid_solution(valid) is True
    assert is_valid_solution(invalid_row) is False
    assert is_valid_solution(invalid_diag) is False

def test_count_total_conflicts():
    '''
    Test count_total_conflicts for number of conflicts in a board
    '''
    valid = np.array([1,3,0,2])
    invalid_row = np.array([1,1,1,1])

    assert count_total_conflicts(valid) == 0

    # Test invalid -> should give 6 as rows (0,1), (0,2), (0,3), (1,2), (1,3), (2,3)
    assert count_total_conflicts(invalid_row) == 6


