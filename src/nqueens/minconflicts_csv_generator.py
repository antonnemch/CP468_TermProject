from pathlib import Path
from nqueens.solver_minconflicts import _minconflicts_kernel, alloc_state
from nqueens.validation import is_valid_solution
import numpy as np
import csv

def run_minconflicts_and_export_solution(
    n: int,
    out_csv: str | Path,
    *,
    seed: int = 123,
    k_sample: int = 512,
    nbhd_width: int = 0,
    max_steps: int = 100_000_000,
    restart_limit: int = 10,
    structured_init: bool = True,
) -> dict:
    """
    Run the existing Min-Conflicts solver once for the given hyperparameters,
    and export a per-column view of the solution to a CSV file.

    This function is both:
      - a "wrapper" around the Numba kernel (it allocates state, handles
        restarts, aggregates steps); AND
      - a "dump script" (it writes a CSV with everything needed for plotting).

    It does NOT modify the existing _minconflicts_kernel or solve_minconflicts.

    Parameters
    ----------
    n : int
        Board size.
    out_csv : str or pathlib.Path
        Path to the CSV file to write.
    seed : int
        Base RNG seed used for the first restart.
    k_sample : int
        Number of candidate rows to sample when candidate_selector = "k_sample".
        (This function doesn’t care which selector you used in the benchmarks;
         it just passes k_sample and nbhd_width to the kernel.)
    nbhd_width : int
        Half-width of the neighborhood when using a neighborhood-based
        candidate selector. If 0, only k_sample is used.
    max_steps : int
        Maximum number of steps PER RESTART for the kernel.
    restart_limit : int
        Maximum number of restarts (inclusive). Total steps is the sum over
        all restarts.
    structured_init : bool
        Whether to use the "structured" initial placement (same flag as in
        your existing allocate/init code).

    Returns
    -------
    stats : dict
        Dictionary with aggregate run statistics:
            - n, solved, steps, restarts, k_sample, nbhd_width,
              seed, max_steps
        This mirrors the existing solve_minconflicts stats so it can be
        printed/logged if needed.

    CSV schema
    ----------
    The output CSV has one row per column c:

        n              : board size
        c              : column index
        row_initial    : pos[c] before the first kernel call
        row_final      : pos[c] after the solver finishes (solution or last state)
        move_distance  : |row_final - row_initial|
        d1             : row_final - c
        d2             : row_final + c

    This matches what the analysis notebook expects for all visualizations
    (small n board, large-n scatter/heatmaps, etc.).
    """
    out_path = Path(out_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if n <= 0:
        raise ValueError("n must be positive")

    # --- 1. allocate initial state and snapshot pos befoire any moves ---
    pos, row, diag1, diag2 = alloc_state(
        n,
        seed=seed,
        structured=structured_init,
    )
    pos_init = pos.copy()  # initial placement for move_distance, etc.

    total_steps = 0
    restarts = 0
    solved = False

    # --- 2. restart loop: identical structure to solve_minconflicts ---
    while restarts <= restart_limit:
        kernel_seed = seed + restarts

        solved_flag, steps_used = _minconflicts_kernel(
            pos,
            row,
            diag1,
            diag2,
            int(n),
            int(max_steps),
            int(k_sample),
            int(nbhd_width),
            int(kernel_seed),
        )
        total_steps += int(steps_used)

        
        if solved_flag == 1 and is_valid_solution(pos):
            solved = True
            break

        
        restarts += 1
        if restarts <= restart_limit:
            pos, row, diag1, diag2 = alloc_state(
                n,
                seed=seed + restarts,
                structured=structured_init,
            )

    pos_final = pos.copy()
    is_valid = bool(is_valid_solution(pos_final))

    # --- 3. compute per-column metrics from initial and final pos ---
    n_val = int(n)
    c = np.arange(n_val, dtype=np.int64)
    row_initial = pos_init.astype(np.int64)
    row_final = pos_final.astype(np.int64)

    move_distance = np.abs(row_final - row_initial)
    d1 = row_final - c
    d2 = row_final + c

    # --- 4. write CSV (one row per column) ---
    fieldnames = [
        "n",
        "c",
        "row_initial",
        "row_final",
        "move_distance",
        "d1",
        "d2",
    ]

    with out_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for col in range(n_val):
            writer.writerow(
                {
                    "n": n_val,
                    "c": int(col),
                    "row_initial": int(row_initial[col]),
                    "row_final": int(row_final[col]),
                    "move_distance": int(move_distance[col]),
                    "d1": int(d1[col]),
                    "d2": int(d2[col]),
                }
            )

    # --- 5. return stats dict for logging / sanity checks ---
    stats: dict = {
        "n": n_val,
        "solved": bool(solved and is_valid),
        "steps": int(total_steps),
        "restarts": int(restarts),
        "k_sample": int(k_sample),
        "nbhd_width": int(nbhd_width),
        "seed": int(seed),
        "max_steps": int(max_steps),
    }
    return stats




stats_50 = run_minconflicts_and_export_solution(
    n=10,
    out_csv=Path("benchmarks/results/solution_n10.csv"),
    seed=123,
    k_sample=3,
    nbhd_width=0,
    max_steps=1_000_000,
    restart_limit=3,
    structured_init=True,
)
print("n=50 stats:", stats_50)

"""
stats_1m = run_minconflicts_and_export_solution(
    n=1_000_000,
    out_csv=Path("benchmarks/results/solution_n1e6.csv"),
    seed=123,
    k_sample=2048,
    nbhd_width=0,
    max_steps=100_000_000,
    restart_limit=5,
    structured_init=False,
)
print("n=1_000_000 stats:", stats_1m)
"""
