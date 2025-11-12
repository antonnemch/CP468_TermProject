"""
benchmarks.bench_minconflicts
Purpose: Benchmark Min-Conflicts across n, k-sample, nbhd-width; emit CSV/plots.

CLI flags:
- --n-list ...        : list of problem sizes
- --k-sample 64       : candidate rows per decision
- --nbhd-width 0      : neighborhood half-width (0 disables)
- --repeats 3         : independent runs per n

Outputs:
- CSV with wall time, steps, restarts, solved rate
- Optional matplotlib plots under benchmarks/
"""
