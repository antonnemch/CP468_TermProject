"""
scripts.make_poster
Purpose:
    Generate a high-resolution visualization of a solved N-Queens configuration
    for inclusion in the final poster submission.

Usage:
    python scripts/make_poster.py --solution-path path/to/solution.npy \
                                  --output docs/poster/board_large.png \
                                  --sample 500

Arguments:
- --solution-path : Path to a NumPy .npy file or JSON file containing the solution array (1D list of row indices).
- --output        : Path to save the output PNG file.
- --sample        : Optional stride for sampling (default=0, plot all queens).

Behavior:
    Loads the saved configuration (e.g., from a previous run of the solver),
    calls `nqueens.visualize.save_board_png`, and saves a publication-quality image.

Example:
    >>> python scripts/make_poster.py --solution-path results/solution_100000.npy \
                                      --output docs/poster/board_100000.png \
                                      --sample 1000

Notes:
- Produces a 300-DPI PNG suitable for inclusion in the final PDF poster.
- Handles large n by sampling queens for visual clarity.
- Run this after generating your largest successful solution (100k–1M queens).
"""
