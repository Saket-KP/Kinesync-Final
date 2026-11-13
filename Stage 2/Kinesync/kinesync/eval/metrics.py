"""Metrics with per-item weights, so an artist bootstrap is just a different weight vector.

Resampling artists with replacement = giving each shot the number of times its artist was drawn.
Every metric below takes such a weight vector `w` (all ones = the plain metric).
"""
from __future__ import annotations

from typing import Callable

import numpy as np


def macro_f1(y: np.ndarray, pred: np.ndarray, w: np.ndarray, k: int) -> float:
    """Macro-F1 over classes that occur in y or pred (sklearn's convention)."""
    cm = np.bincount(y * k + pred, weights=w, minlength=k * k).reshape(k, k)
    tp = np.diag(cm)
    fp = cm.sum(0) - tp
    fn = cm.sum(1) - tp
    present = (cm.sum(0) + cm.sum(1)) > 0
    f1 = np.where(present, 2 * tp / np.maximum(2 * tp + fp + fn, 1e-12), 0.0)
    return float(f1[present].mean()) if present.any() else 0.0


def accuracy(y, pred, w) -> float:
    return float((w * (y == pred)).sum() / max(w.sum(), 1e-12))


def binary_f1(y, pred, w) -> float:
    tp = (w * (y == 1) * (pred == 1)).sum()
    fp = (w * (y == 0) * (pred == 1)).sum()
    fn = (w * (y == 1) * (pred == 0)).sum()
    return float(2 * tp / max(2 * tp + fp + fn, 1e-12))


class AveragePrecision:
    """AP of one score vector; sorts once, then scores any weight vector in O(n)."""

    def __init__(self, y: np.ndarray, score: np.ndarray):
        order = np.argsort(-score, kind="stable")
        self.y = y[order].astype(np.float64)
        self.order = order
        s = score[order]
        # last index of each run of tied scores = one threshold (same as sklearn)
        self.cut = np.r_[np.nonzero(np.diff(s))[0], len(s) - 1]

    def __call__(self, w: np.ndarray) -> float:
        ws = w[self.order]
        pos = ws * self.y
        total = pos.sum()
        if total <= 0:
            return float("nan")
        tp = np.cumsum(pos)[self.cut]
        n = np.cumsum(ws)[self.cut]
        prec = tp / np.maximum(n, 1e-12)
        d_recall = np.diff(np.r_[0.0, tp]) / total
        return float((prec * d_recall).sum())


def mean_ap(y: np.ndarray, scores: np.ndarray) -> Callable[[np.ndarray], float]:
    """mAP over columns (phrases) of y/scores; phrases without positives are skipped."""
    aps = [AveragePrecision(y[:, j], scores[:, j]) for j in range(y.shape[1])]

    def f(w):
        v = [a(w) for a in aps]
        v = [x for x in v if not np.isnan(x)]
        return float(np.mean(v)) if v else float("nan")
    return f


def artist_weights(artists: np.ndarray, n: int, seed: int = 0):
    """Yield n weight vectors, one per bootstrap resample of artists."""
    uniq, inv = np.unique(artists, return_inverse=True)
    rng = np.random.default_rng(seed)
    for _ in range(n):
        counts = np.bincount(rng.integers(0, len(uniq), len(uniq)), minlength=len(uniq))
        yield counts[inv].astype(np.float64)


def bootstrap(fns: dict[str, Callable], artists_of: dict[str, np.ndarray], n: int = 1000,
              seed: int = 0) -> dict[str, list]:
    """fns[name](w) -> float, where w is a weight over the items of `artists_of[name]`.

    All metrics share the same artist draws (so differences are paired). Returns
    {name: [point, lo95, hi95]}.
    """
    all_art = np.unique(np.concatenate(list(artists_of.values())))
    rng = np.random.default_rng(seed)
    pos = {name: np.searchsorted(all_art, a) for name, a in artists_of.items()}
    samples = {name: [] for name in fns}
    for _ in range(n):
        counts = np.bincount(rng.integers(0, len(all_art), len(all_art)), minlength=len(all_art))
        for name, f in fns.items():
            samples[name].append(f(counts[pos[name]].astype(np.float64)))
    out = {}
    for name, f in fns.items():
        point = f(np.ones(len(artists_of[name])))
        s = np.array([x for x in samples[name] if not np.isnan(x)])
        lo, hi = (np.percentile(s, [2.5, 97.5]) if len(s) else (np.nan, np.nan))
        out[name] = [round(float(point), 4), round(float(lo), 4), round(float(hi), 4)]
    return out
