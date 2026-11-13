import numpy as np

# Max allowed tags per semantic category
LIMITS = {"Setting": 1, "Lighting": 1, "Subject": 2, "Action": 2, "Style": 2}

def label_from_sims(calib, groups, scale_sim, tag_sim, cam_motion, dur_s):
    """Convert raw CLIP similarities into calibrated scale and z-scored tags."""
    # 1. Prior correction for shot scale: subtract alpha * mean similarity
    alpha = calib.get("scale_alpha", 0.5)
    c = np.asarray(scale_sim) - alpha * np.asarray(calib["scale_mean"])
    
    # Softmax with temperature=100 for confidence score
    e = np.exp((c - c.max()) * 100.0)
    scale = int(c.argmax())
    conf = round(float(e.max() / e.sum()), 4)
    
    # 2. Binary camera motion flag
    moving = int(cam_motion > calib.get("moving_threshold", 4.0))
    
    # 3. Z-score standardization for tags (only for shots >= 0.5s)
    tags, scores = [], []
    if dur_s >= 0.5:
        z = (np.asarray(tag_sim) - np.asarray(calib["tag_mean"])) / np.asarray(calib["tag_std"])
        z_tag = calib.get("z_tag", 1.0)
        
        for g, k in LIMITS.items():
            idx = [i for i, gg in enumerate(groups) if gg == g]
            # Pick top-k tags in this group that exceed z-score >= 1.0
            for i in sorted(idx, key=lambda i: -z[i])[:k]:
                if z[i] >= z_tag:
                    tags.append(int(i))
                    scores.append(round(float(z[i]), 3))
                    
    return scale, conf, moving, tags, scores