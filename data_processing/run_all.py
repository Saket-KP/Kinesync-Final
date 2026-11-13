import json
import os
from pathlib import Path
import numpy as np
import pandas as pd

# Prevent CPU thread thrashing when running parallel worker processes
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

from kinesync.extract_audio import extract_audio_features, _compute_clap_1hz, filter_lyrics
from kinesync.extract_video import detect_cut_times, compute_video_features
from kinesync.label_shots import label_single_shot

def compute_norm_stats(data_dir: Path, splits_df: pd.DataFrame):
    """Compute feature mean and std across TRAIN videos only (prevents data leakage)."""
    train_ids = splits_df[splits_df["split"] == "train"]["video_id"].tolist()
    channels = ["onset", "flux", "chroma", "flow", "color_drift", "brightness", "saturation"]
    stats = {}
    
    for ch in channels:
        vals = [np.load(data_dir / "features" / f"{v}.npz")[ch].astype(np.float64) for v in train_ids]
        cat = np.concatenate(vals, axis=0)
        stats[ch] = {
            "mean": np.mean(cat, axis=0).tolist(),
            "std": np.maximum(np.std(cat, axis=0), 1e-6).tolist(),
        }
        
    (data_dir / "stats" / "norm_train.json").write_text(json.dumps(stats, indent=2))