# N-Queens at Scale — Min-Conflicts (Numba) + Backtracking Baseline

High-performance N-Queens with two complementary solvers:
- **Min-Conflicts + Numba JIT** for very large `n`
- **Backtracking (MRV/LCV + Forward Checking)** as a complete baseline for small/medium `n`

No C/C++ required. The hot path is compiled with **Numba**. Uses **venv** (not conda).

---

## Quick start (venv)

```bash
# create and activate venv (Linux/macOS)
python3 -m venv .venv
source .venv/bin/activate

# Windows PowerShell
python -m venv .venv
.\\.venv\\Scripts\\Activate.ps1

# install deps
pip install -U pip
pip install -r requirements.txt

# run a large instance with Min-Conflicts
python -m nqueens.cli solve --n 100000 --seed 42 --max-steps 300000 --k-sample 64

# backtracking baseline (small n)
python -m nqueens.cli solve-bt --n 200

```

## Directory structure (with one-line purpose per file)
```
nqueens-minconflicts/
├─ README.md                         # Project overview, usage, structure
├─ pyproject.toml                    # Packaging metadata (PEP 621)
├─ requirements.txt                  # Pip dependencies for venv
├─ .gitignore                        # Git ignores for Python/cache/artifacts
├─ src/
│  └─ nqueens/
│     ├─ __init__.py                 # Public API exports
│     ├─ cli.py                      # CLI entrypoints for both solvers
│     ├─ config.py                   # Default hyperparameters & flags
│     ├─ validation.py               # Solution/counter validation utilities
│     ├─ instrumentation.py          # Timing/metrics helpers
│     ├─ common_arrays.py            # Typed array allocation & initialization
│     ├─ solver_backtracking.py      # Complete CSP solver (MRV/LCV/FC)
│     ├─ solver_minconflicts.py      # Numba-compiled Min-Conflicts kernel
│     ├─ heuristics.py               # Backtracking heuristics implementations
│     └─ sampling.py                 # JIT-safe row sampling utilities
├─ benchmarks/
|  ├─ results/                       # Folder for csv benchmark results
│  ├─ bench_minconflicts.py          # Sweeps & timing for Min-Conflicts
│  └─ bench_backtracking.py          # Sweeps & timing for Backtracking
├─ tests/
│  ├─ test_validation.py             # Unit tests for validators/counters
│  ├─ test_minconflicts_small.py     # Small-n Min-Conflicts correctness
│  └─ test_backtracking_small.py     # Small-n Backtracking correctness
├─ scripts/
│  ├─ run_small.sh                   # Handy small-n smoke tests (bash)
│  ├─ run_large.sh                   # Example large-n runs (bash)
│  ├─ run_required_sizes.sh          # Runs for values of n specified in assignment description
│  ├─ make_poster.py                 # Generates poster PNG for submission
│  └─ profile_numba.py               # Microbenchmark of the JIT kernel
└─ notebooks/
   └─ exploration.ipynb              # Small-n convergence/visualization
```

## Usage examples
```
# solve and emit JSON summary
python -m nqueens.cli solve --n 50000 --k-sample 64 --nbhd-width 0 --json-out run.json

# compare strategies (plots saved to benchmarks/)
python benchmarks/bench_minconflicts.py --n-list 10000 20000 50000 100000
python benchmarks/bench_backtracking.py --n-list 50 100 200 400
```

## Architectural overview

This repo separates **interfaces**, **algorithms**, and **utilities** so you can swap parts without touching the rest.

- **CLI (`src/nqueens/cli.py`)** is the front door. It parses flags, seeds RNG, times runs, and calls a solver.
- **Solvers**
  - `solver_minconflicts.py` holds the **Numba-JIT kernel** (hot loop) and a thin Python wrapper that allocates arrays, runs restarts, and validates the result.
  - `solver_backtracking.py` is the **complete CSP baseline** (MRV/LCV + Forward Checking) for small/medium `n`.
- **State & validation**
  - `common_arrays.py` creates typed, contiguous `np.int32` arrays (`pos`, `row`, `diag1`, `diag2`) and initializes counters from an assignment.
  - `validation.py` independently rebuilds counters and checks correctness (no conflicts) so solvers can’t “cheat.”
- **Heuristics & sampling**
  - `heuristics.py` contains backtracking heuristics (MRV, Degree, LCV, forward checking).
  - `sampling.py` provides **JIT-safe** row candidate generators (K-sampling / neighborhood windows) for Min-Conflicts.
- **Observability**
  - `instrumentation.py` records timings and run metadata; benchmarks read these to produce CSVs/plots.
- **Benchmarks & tests**
  - `benchmarks/*.py` run parameter sweeps and save metrics.
  - `tests/*.py` assert correctness of validators and both solvers on small `n`.
- **Packaging**
  - `pyproject.toml` defines project metadata and install/runtime dependencies.
  - `requirements.txt` pins Python packages for a `venv`.

Control flow:
`cli.py` → choose solver → `common_arrays.alloc_state` → solver (Numba kernel or backtracking) → `validation.is_valid_solution` → `instrumentation.summarize_run` → optional `benchmarks/` plotting.


## Numba quick checklist (details in solver_minconflicts.py)
- Single @njit(nopython=True, fastmath=True, cache=True) hot loop
- Arrays np.int32, C-contiguous: pos, row_count, diag1, diag2
- O(1) counter updates per move; no Python objects/allocations in kernel
- Best-row via K-sampling or ±neighborhood scan (never scan all rows)
- Simple in-kernel RNG (LCG/xorshift) for reproducible ints
- Validate by rebuilding counters in O(n)

