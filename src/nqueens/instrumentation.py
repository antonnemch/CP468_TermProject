"""
nqueens.instrumentation
Purpose: Lightweight timing and run metadata aggregation.

Components:
- Timer context manager: measure wall clock durations.
- summarize_run(...): Build a dict of steps, restarts, time, params, and validity.
"""
import time

class Timer:
    '''
    timer context manager to measure wall-clock duration
    '''

    def __enter__(self):
        self.start = time.time()
        self.elapsed = None
        return self

    def __exit__(self, type, value, traceback):
        self.elapsed = time.time() - self.start

def summarize_run(steps, restarts, time, params, valid):
    '''
    Dictionary to summarize steps, restarts, timer, params, valid
    Parameters -> steps: # of steps taken
                  restarts: # of restarts peformed (can be 0)
                  time: time duration
                  params: dictionary of parameters
                  valid: check if resulting board is valid

    Return -> dictionary: run metadata aggregation
    '''
    return{
        "steps": steps,
        "restarts": restarts,
        "time": time.elapsed,
        "params": params,
        "valid": valid
    }
