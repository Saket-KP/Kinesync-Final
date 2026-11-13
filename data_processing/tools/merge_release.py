import numpy as np

CHANNELS = ["onset", "flux", "chroma", "flow", "color_drift", "brightness", "saturation"]

def assign_artist_splits(ok_df, val_frac=0.10, test_frac=0.10, seed=42):
    """Stratified group split: keeps all videos of an artist in the same split per genre."""
    rng = np.random.default_rng(seed)
    art_split = {}
    for genre in sorted(ok_df.genre.unique()):
        g = ok_df[ok_df.genre == genre]
        counts = g.groupby("artist").size()
        artists = sorted(counts.index)
        rng.shuffle(artists)
        
        total, tv, vv = len(g), 0, 0
        for a in artists:
            n = int(counts[a])
            if tv < test_frac * total:
                art_split[a] = "test"; tv += n
            elif vv < val_frac * total:
                art_split[a] = "val"; vv += n
            else:
                art_split[a] = "train"
    return art_split

def norm_stats(train_npz_paths: list) -> dict:
    """Compute global mean and std on TRAIN videos only using E[X^2] - (E[X])^2."""
    n = {c: 0 for c in CHANNELS}
    s = {c: 0.0 for c in CHANNELS}
    ss = {c: 0.0 for c in CHANNELS}
    
    for path in train_npz_paths:
        z = np.load(path)
        for c in CHANNELS:
            a = z[c].astype(np.float64)
            n[c] += a.shape[0]
            s[c] += a.sum(axis=0)
            ss[c] += (a ** 2).sum(axis=0)
            
    stats = {}
    for c in CHANNELS:
        mean = s[c] / max(1, n[c])
        std = np.sqrt(np.maximum(ss[c] / max(1, n[c]) - mean ** 2, 0.0))
        stats[c] = {"mean": np.asarray(mean).tolist(), "std": np.maximum(std, 1e-6).tolist()}
    return stats