"""Turn a generated cut channel into shot spans, plus the sanity metrics.

The model emits a smooth cut-likelihood curve (see common/features.soft_cuts); the
shot boundaries are its peaks. Threshold and minimum separation are the two knobs
that set cut density, so `calibrate_threshold` sweeps the threshold on a split until
the generated cut rate matches the real one -- without that, comparing shot-length
distributions mostly measures how the threshold was picked.

Metric note: Beat Alignment Score needs a sigma well below the beat period. The
median beat period in this corpus is 511 ms, so at the common sigma=0.1 s a random
cut already scores ~0.5 and the metric is saturated and uninformative. Default here
is 0.05 s; see DATA_AUDIT.md Q4.
"""
from __future__ import annotations

import numpy as np
from scipy.signal import find_peaks
from scipy.stats import wasserstein_distance


# ----------------------------------------------------------------- peak picking
def pick_cuts(
    cut_channel: np.ndarray,
    fps: int,
    threshold: float = 0.25,
    min_distance_s: float = 0.25,
) -> np.ndarray:
    """Peak times (seconds) of the cut-likelihood curve."""
    dist = max(1, int(round(min_distance_s * fps)))
    pk, _ = find_peaks(np.asarray(cut_channel, dtype=np.float64),
                       height=threshold, distance=dist)
    return pk.astype(np.float64) / fps


def cuts_to_shots(cut_times: np.ndarray, duration_s: float) -> list[dict]:
    """Contiguous shot spans tiling [0, duration]."""
    t = np.sort(np.asarray(cut_times, dtype=np.float64))
    t = t[(t > 0.0) & (t < duration_s)]
    edges = np.concatenate(([0.0], t, [float(duration_s)]))
    return [{"start": float(a), "end": float(b), "duration": float(b - a)}
            for a, b in zip(edges[:-1], edges[1:]) if b > a]


def shot_lengths(cut_times: np.ndarray, duration_s: float) -> np.ndarray:
    return np.array([s["duration"] for s in cuts_to_shots(cut_times, duration_s)])


# ----------------------------------------------------------------------- metrics
def beat_alignment_score(cut_times: np.ndarray, beat_times: np.ndarray,
                         sigma: float = 0.05) -> float:
    """Mean exp(-d^2/2sigma^2) over cuts, d = distance to the nearest beat."""
    c = np.asarray(cut_times, dtype=np.float64)
    b = np.asarray(beat_times, dtype=np.float64)
    if c.size == 0 or b.size == 0:
        return 0.0
    d = np.abs(c[:, None] - b[None, :]).min(axis=1)
    return float(np.exp(-(d ** 2) / (2 * sigma ** 2)).mean())


def cuts_per_minute(cut_times: np.ndarray, duration_s: float) -> float:
    if duration_s <= 0:
        return 0.0
    return float(60.0 * len(np.asarray(cut_times)) / duration_s)


def calibrate_threshold(
    cut_channels: list[np.ndarray],
    durations: list[float],
    target_cpm: float,
    fps: int,
    min_distance_s: float = 0.25,
    lo: float = 0.02,
    hi: float = 0.95,
    n: int = 40,
) -> tuple[float, float]:
    """Threshold whose mean cuts/min best matches `target_cpm`. Returns (thr, cpm)."""
    best = (lo, float("inf"), 0.0)
    for thr in np.linspace(lo, hi, n):
        cpms = [cuts_per_minute(pick_cuts(ch, fps, thr, min_distance_s), d)
                for ch, d in zip(cut_channels, durations)]
        cpm = float(np.mean(cpms)) if cpms else 0.0
        err = abs(cpm - target_cpm)
        if err < best[1]:
            best = (float(thr), err, cpm)
    return best[0], best[2]


# ------------------------------------------------------------------- evaluation
def evaluate_against_real(
    model,
    diffusion,
    cfg,
    root,
    normaliser,
    split: str = "val",
    max_videos: int = 2,
    device="cpu",
    threshold: float | None = None,
) -> dict:
    """Generate a few held-out songs and compare cut statistics to the real ones.

    Instrumentation for watching a training run, not an evaluation protocol --
    with 41 videos and a model that will memorise, these numbers say whether the
    pipeline is producing something shaped like a real cut track, nothing more.
    """
    from common.dataset import load_song          # deferred: avoids an import cycle
    from common.splits import videos_in_split
    from timeline.sample import generate_song

    vids = videos_in_split(root, split)[:max_videos]
    if not vids:
        raise ValueError(f"no videos in split {split!r}")

    gen_ch, durs, real_sl, gen_sl = [], [], [], []
    bas_g, bas_r, cpm_g, cpm_r = [], [], [], []

    was_training = model.training
    model.eval()
    for vid in vids:
        song = load_song(root, cfg, vid, normaliser)
        x = generate_song(model, diffusion, cfg, song, device=device,
                          normaliser=normaliser, progress=False)
        cut = x[:, 0]
        gen_ch.append(cut)
        durs.append(song.duration_s)

        thr = 0.25 if threshold is None else threshold
        gc = pick_cuts(cut, cfg.data.fps, thr)
        rc = song.cut_times

        gen_sl.append(shot_lengths(gc, song.duration_s))
        real_sl.append(shot_lengths(rc, song.duration_s))
        bas_g.append(beat_alignment_score(gc, song.beat_times))
        bas_r.append(beat_alignment_score(rc, song.beat_times))
        cpm_g.append(cuts_per_minute(gc, song.duration_s))
        cpm_r.append(cuts_per_minute(rc, song.duration_s))
    if was_training:
        model.train()

    gsl = np.concatenate(gen_sl) if gen_sl else np.array([0.0])
    rsl = np.concatenate(real_sl) if real_sl else np.array([0.0])
    w1 = float(wasserstein_distance(gsl, rsl)) if gsl.size and rsl.size else float("nan")

    thr_cal, cpm_cal = calibrate_threshold(gen_ch, durs, float(np.mean(cpm_r)), cfg.data.fps)
    return {
        "videos": vids,
        "w1_shot_len": w1,
        "bas_gen": float(np.mean(bas_g)),
        "bas_real": float(np.mean(bas_r)),
        "cpm_gen": float(np.mean(cpm_g)),
        "cpm_real": float(np.mean(cpm_r)),
        "median_shot_gen": float(np.median(gsl)),
        "median_shot_real": float(np.median(rsl)),
        "calibrated_threshold": thr_cal,
        "calibrated_cpm": cpm_cal,
    }
