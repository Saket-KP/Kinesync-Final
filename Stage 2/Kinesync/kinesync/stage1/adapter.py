"""The only place downstream code reads Stage 1 output.

Contract (handoff section 2): stage1/{vid}_s{k}.npz with cut, flow, color_drift, brightness,
saturation (T,) in raw units on a 25 fps grid, and stage1/{vid}_s{k}_shots.json.
When the real Stage 1 lands and writes something else, change only this file.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from kinesync.common import KIN_KEYS, data_root

DEFAULT_FPS = 25


def decode_cuts(cut: np.ndarray, fps: float, thresh: float = 0.5, min_shot_s: float = 0.25):
    """Peak-pick the cut channel (fallback when no shots file exists): local maxima above
    `thresh`, at least `min_shot_s` apart. Returns cut times in seconds."""
    c = np.asarray(cut, dtype=np.float64)
    peaks = [i for i in range(1, len(c) - 1)
             if c[i] >= thresh and c[i] >= c[i - 1] and c[i] > c[i + 1]]
    if len(c) > 1 and c[-1] >= thresh and c[-1] > c[-2]:
        peaks.append(len(c) - 1)
    out = []
    for i in peaks:
        t = i / fps
        if t >= min_shot_s and (not out or t - out[-1] >= min_shot_s):
            out.append(t)
    return np.array(out)


def load_stage1(vid: str, k: int, root=None) -> dict:
    """Returns {"fps", "T", "duration", "flow", ..., "cut", "shots": [(start, end)], "mock"}."""
    d = Path(root or data_root()) / "stage1"
    with np.load(d / f"{vid}_s{k}.npz") as z:
        out = {key: z[key].astype(np.float32) for key in ("cut",) + KIN_KEYS}
        fps = float(z["fps"]) if "fps" in z.files else DEFAULT_FPS
        mock = bool(z["mock"]) if "mock" in z.files else False
    T = len(out["cut"])
    dur = T / fps
    js = d / f"{vid}_s{k}_shots.json"
    if js.exists():
        with open(js, encoding="utf-8") as f:
            meta = json.load(f)
        shots = [(float(s["start"]), float(s["end"])) for s in meta["shots"]]
        mock = mock or bool(meta.get("mock", False))
    else:
        edges = np.concatenate([[0.0], decode_cuts(out["cut"], fps), [dur]])
        shots = list(zip(edges[:-1].tolist(), edges[1:].tolist()))
    shots = sorted(s for s in shots if s[1] > s[0])
    out.update(fps=fps, T=T, duration=dur, shots=shots, mock=mock)
    return out
