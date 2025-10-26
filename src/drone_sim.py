# src/drone_sim.py
import os
import random
import numpy as np
from src.compression import mask_to_heatmap
from src.models.segment_stub import segment

def simulate_drone_from_image(image_bgr, num_drones=4, drop_prob=0.2, seg_method='ndvi', threshold=0.12):
    """
    Simulate multiple drones scanning the same image.
    Each drone runs segmentation -> compress -> possibly drop packet.
    Returns a list of heatmaps (or None where dropped).
    """
    heatmaps = []
    for i in range(num_drones):
        # Optionally add small random perturbation to simulate different view/noise
        img = image_bgr.copy().astype('uint8')
        # small brightness jitter
        alpha = 1.0 + random.uniform(-0.05, 0.05)
        beta = random.uniform(-10, 10)
        img = np.clip(img * alpha + beta, 0, 255).astype('uint8')

        mask, _aux = segment(img, method=seg_method, threshold=threshold)
        heat = mask_to_heatmap(mask, grid_size=20)
        if random.random() > drop_prob:
            heatmaps.append(heat)
        else:
            # dropped packet
            heatmaps.append(None)
            print(f"[DroneSim] Packet dropped (drone #{i+1})")
    return heatmaps
