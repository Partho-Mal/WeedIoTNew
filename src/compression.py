# src/compression.py
import numpy as np

def mask_to_heatmap(mask_np, grid_size=20):
    """
    mask_np: 2D array with 0 background and 255 weeds.
    Returns heatmap grid_size x grid_size with values 0..1 (weed density).
    """
    if mask_np.ndim != 2:
        raise ValueError("mask must be 2D grayscale")
    H, W = mask_np.shape
    cell_h = H // grid_size
    cell_w = W // grid_size

    # if image smaller than grid_size, upsample mask to at least grid_size
    if cell_h == 0 or cell_w == 0:
        mask_resized = np.resize(mask_np, (grid_size, grid_size))
        return (mask_resized / 255.0).astype(float)

    heatmap = np.zeros((grid_size, grid_size), dtype=float)
    for i in range(grid_size):
        for j in range(grid_size):
            y0, y1 = i*cell_h, (i+1)*cell_h
            x0, x1 = j*cell_w, (j+1)*cell_w
            patch = mask_np[y0:y1, x0:x1]
            if patch.size == 0:
                heatmap[i, j] = 0.0
            else:
                heatmap[i, j] = float(patch.mean()) / 255.0
    return heatmap
