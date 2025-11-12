"""
nqueens.visualize
Purpose:
    Generate graphical representations of N-Queens board configurations
    for inclusion in the final poster or reports. Intended for both small
    (n ≤ 100) and large (n ≥ 10⁴) boards.

Functions:
- save_board_png(pos: np.ndarray, path: str, sample: int = 0):
    Saves an image of the board with queens marked as black dots.
    * pos: 1D np.ndarray of length n (pos[c] = row of queen in column c)
    * path: output filename (.png)
    * sample: optional stride for subsampling large boards (e.g., 1000 -> plot every 1000th queen)

Behavior:
- For small n (≤ 1000): plots all queens with a true chessboard grid.
- For large n: plots only sampled queens (to prevent dense overdraw),
  labeling axes with scaled indices.
- Uses matplotlib only; no external GUI dependencies.

Example:
    >>> from nqueens.visualize import save_board_png
    >>> save_board_png(pos, "board_100000.png", sample=500)
    # Produces a scatter plot of queens for use in final poster.

Notes:
- Designed for offline rendering in scripts/make_poster.py.
- Execution time grows linearly with n/sampling rate.
"""
