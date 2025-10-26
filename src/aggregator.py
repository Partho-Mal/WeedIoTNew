# src/aggregator.py
import numpy as np
import json
from pathlib import Path

AGG_FILE = Path("results/aggregated.json")

def aggregate_list_of_heatmaps(list_of_heatmaps):
    """
    list_of_heatmaps: list containing numpy arrays (grid) or None
    returns averaged heatmap (numpy array) over non-None entries or None if none valid
    """
    valid = [h for h in list_of_heatmaps if h is not None]
    if not valid:
        return None
    arr = np.stack(valid, axis=0)
    return np.mean(arr, axis=0)

def persist_aggregate(new_agg):
    """
    Save the farm-wide aggregate to results/aggregated.json
    Format: {"agg": [[...],[...]], "count": N}
    """
    if new_agg is None:
        return
    AGG_FILE.parent.mkdir(parents=True, exist_ok=True)
    payload = {"agg": new_agg.tolist()}
    AGG_FILE.write_text(json.dumps(payload))
