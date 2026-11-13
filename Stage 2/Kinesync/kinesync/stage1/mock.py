"""MOCK Stage 1. The real EDGE-style Stage 1 is not in this repo, so this writes files in the
Stage 1 output contract (handoff section 2) for Stage 2, rendering and the pipeline to consume:

    stage1/{video_id}_s{k}.npz        cut, flow, color_drift, brightness, saturation  (T,) raw units
                                      + fps (25) and mock=1
    stage1/{video_id}_s{k}_shots.json {"shots": [{"start", "end"}], "mock": true, "mode": ...}

Two modes:
  perturb  (songs in the release) start from the REAL cuts and kinetics, then drop / add / jitter
           cuts and add smooth noise. Stands in for a good Stage 1. Uses the real video, which a
           real Stage 1 never sees, so numbers from it are NOT Stage 1 results.
  audio    (any song with audio features) cuts on beats, kinetics from the onset envelope and
           train-set statistics. A crude audio-only baseline.

Every output says "mock" so it can never be mistaken for real Stage 1 output.

    python -m kinesync.stage1.mock --split test --n_samples 3
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from kinesync.common import (AUDIO_FPS, KIN_KEYS, VIDEO_FPS, data_root, kin_mean_std,
                             load_features, videos_in)

FPS = 25                 # Stage 1 contract grid
MIN_SHOT_S = 0.25


def stage1_dir(root=None) -> Path:
    return Path(root or data_root()) / "stage1"


def _rng(vid: str, k: int, seed: int) -> np.random.Generator:
    return np.random.default_rng([seed, k, int(vid.lstrip("v") or 0) if vid.lstrip("v").isdigit()
                                  else sum(map(ord, vid))])


def _smooth_noise(rng, n: int, sigma: float, width: int) -> np.ndarray:
    """Gaussian noise smoothed by a moving average of `width` frames, rescaled to std sigma."""
    x = rng.normal(size=n + width)
    x = np.convolve(x, np.ones(width) / width, mode="valid")[:n]
    return x / (x.std() + 1e-9) * sigma


def _clean_cuts(cuts: np.ndarray, dur: float) -> np.ndarray:
    cuts = np.sort(cuts[(cuts > MIN_SHOT_S) & (cuts < dur - MIN_SHOT_S)])
    out = []
    for c in cuts:
        if not out or c - out[-1] >= MIN_SHOT_S:
            out.append(c)
    return np.array(out)


def mock_perturb(feat: dict, rng, drop=0.15, jitter_frames=2.0, kin_sigma=0.3):
    dur = len(feat["onset"]) / AUDIO_FPS
    real = feat["cut_times"].astype(np.float64)
    keep = real[rng.random(len(real)) > drop]
    beats = feat["beat_times"].astype(np.float64)
    n_add = len(real) - len(keep)
    added = rng.choice(beats, size=min(n_add, len(beats)), replace=False) if len(beats) else np.array([])
    cuts = np.concatenate([keep, added])
    cuts = cuts + np.clip(rng.normal(0, jitter_frames, len(cuts)), -3, 3) / FPS
    cuts = _clean_cuts(cuts, dur)

    T = int(np.floor(dur * FPS))
    t = (np.arange(T) + 0.5) / FPS
    t2 = (np.arange(len(feat["flow"])) + 0.5) / VIDEO_FPS
    mean, std = kin_mean_std()
    kin = {}
    for j, k in enumerate(KIN_KEYS):
        z = (np.interp(t, t2, feat[k]) - mean[j]) / std[j]
        z = z + _smooth_noise(rng, T, kin_sigma, FPS)
        kin[k] = z * std[j] + mean[j]
    return dur, cuts, kin


def mock_audio(feat: dict, rng, genre_rate=None):
    dur = len(feat["onset"]) / AUDIO_FPS
    beats = feat["beat_times"].astype(np.float64)
    if len(beats) < 4:
        beats = np.arange(0.5, dur, 0.5)
    cuts, i = [], 0
    while i < len(beats):
        i += int(rng.choice([1, 2, 4], p=[0.3, 0.4, 0.3]))
        if i < len(beats):
            cuts.append(beats[i])
    cuts = _clean_cuts(np.array(cuts), dur)

    T = int(np.floor(dur * FPS))
    t = (np.arange(T) + 0.5) / FPS
    t10 = (np.arange(len(feat["onset"])) + 0.5) / AUDIO_FPS
    on = np.convolve(feat["onset"], np.ones(5) / 5, mode="same")
    on_z = (np.interp(t, t10, on) - on.mean()) / (on.std() + 1e-9)
    mean, std = kin_mean_std()
    kin = {}
    for j, k in enumerate(KIN_KEYS):
        base = rng.normal(0, 0.5)                       # per-song level, normalised units
        z = base + _smooth_noise(rng, T, 0.4, 2 * FPS)
        if k in ("flow", "color_drift"):
            z = z + 0.5 * on_z
        kin[k] = z * std[j] + mean[j]
    return dur, cuts, kin


def write_mock(vid: str, k: int, mode: str = "perturb", seed: int = 0, root=None,
               feat: dict | None = None) -> tuple[Path, Path]:
    feat = feat if feat is not None else load_features(vid, root)
    rng = _rng(vid, k, seed)
    if mode == "perturb" and "cut_times" in feat and "flow" in feat:
        dur, cuts, kin = mock_perturb(feat, rng)
    else:
        mode = "audio"
        dur, cuts, kin = mock_audio(feat, rng)
    T = len(next(iter(kin.values())))
    cut = np.zeros(T, np.float32)
    cut[np.clip((cuts * FPS).astype(int), 0, T - 1)] = 1.0
    arrays = {key: np.clip(v, 0, 1 if key in ("brightness", "saturation") else None).astype(np.float32)
              for key, v in kin.items()}
    out = stage1_dir(root)
    out.mkdir(parents=True, exist_ok=True)
    npz = out / f"{vid}_s{k}.npz"
    np.savez_compressed(npz, cut=cut, fps=np.int32(FPS), mock=np.int8(1), **arrays)
    edges = np.concatenate([[0.0], cuts, [dur]])
    shots = [{"start": round(float(a), 4), "end": round(float(b), 4)} for a, b in zip(edges, edges[1:])]
    js = out / f"{vid}_s{k}_shots.json"
    with open(js, "w", encoding="utf-8") as f:
        json.dump({"shots": shots, "mock": True, "mode": mode, "seed": seed}, f)
    return npz, js


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--split", default="test")
    ap.add_argument("--video_id", nargs="*")
    ap.add_argument("--n_samples", type=int, default=3)
    ap.add_argument("--mode", choices=["perturb", "audio"], default="perturb")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    vids = a.video_id or videos_in(a.split)
    for vid in vids:
        for k in range(a.n_samples):
            write_mock(vid, k, a.mode, a.seed)
    print(f"MOCK Stage 1 ({a.mode}): {len(vids)} songs x {a.n_samples} samples -> {stage1_dir()}")


if __name__ == "__main__":
    main()
