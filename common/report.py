"""Per-channel feature report, printed at the start of every run.

The data is loaded as-is, which means `downbeat_phase` (all zeros) and `valid`
(99.99% ones) are still in the files and still occupy conditioning channels. Rather
than silently fixing them, every run prints what each channel actually contains and
flags the ones carrying no information -- so a dead input is visible in the log
instead of being discovered three days into debugging.
"""
from __future__ import annotations

import numpy as np

from common.config import COND_SPEC, OUT_CHANNELS, Config
from common.features import build_cond, build_target, load_raw


def _stats(a: np.ndarray) -> tuple[float, float, float, float]:
    a = np.asarray(a, dtype=np.float64)
    return float(a.min()), float(a.max()), float(a.mean()), float(a.std())


def channel_report(root, cfg: Config, video_ids: list[str], max_videos: int = 12) -> dict:
    """Aggregate per-channel stats over the built target/cond tensors."""
    vids = list(video_ids)[:max_videos]
    out_rows: dict[str, list] = {c: [] for c in OUT_CHANNELS}
    cond_names = [n for n, w in COND_SPEC for _ in range(w)]
    cond_rows: dict[int, list] = {i: [] for i in range(len(cond_names))}

    for vid in vids:
        raw = load_raw(root, vid)
        x, _ = build_target(raw, cfg)
        c = build_cond(raw, cfg)
        for i, name in enumerate(OUT_CHANNELS):
            out_rows[name].append(_stats(x[:, i]))
        for i in range(c.shape[1]):
            cond_rows[i].append(_stats(c[:, i]))

    def agg(rows):
        r = np.array(rows)
        return r[:, 0].min(), r[:, 1].max(), r[:, 2].mean(), r[:, 3].mean()

    dead_out, dead_cond, big = [], [], []

    print(f"\n--- channel report ({len(vids)} videos, {cfg.data.fps} fps grid) ---")
    print(f"{'OUTPUT channel':<22}{'min':>10}{'max':>12}{'mean':>10}{'std':>10}   note")
    for name in OUT_CHANNELS:
        lo, hi, mu, sd = agg(out_rows[name])
        note = ""
        if sd == 0:
            note = "DEAD (constant)"
            dead_out.append(name)
        elif abs(hi) > 20 or abs(lo) > 20:
            note = "wide range -> needs normalisation"
            big.append(name)
        print(f"{name:<22}{lo:>10.3f}{hi:>12.1f}{mu:>10.3f}{sd:>10.3f}   {note}")

    print(f"\n{'COND channel':<22}{'min':>10}{'max':>12}{'mean':>10}{'std':>10}   note")
    seen: dict[str, list[int]] = {}
    for i, n in enumerate(cond_names):
        seen.setdefault(n, []).append(i)
    for name, idxs in seen.items():
        lo, hi, mu, sd = agg([s for i in idxs for s in cond_rows[i]])
        note = ""
        if sd == 0:
            note = "DEAD (constant) -- occupies conditioning capacity"
            dead_cond.append(name)
        elif abs(hi) > 20 or abs(lo) > 20:
            note = "wide range -> needs normalisation"
            big.append(name)
        label = f"{name}[{len(idxs)}]" if len(idxs) > 1 else name
        print(f"{label:<22}{lo:>10.3f}{hi:>12.1f}{mu:>10.3f}{sd:>10.3f}   {note}")

    if dead_out or dead_cond:
        print(f"\n  !! dead channels: out={dead_out or '-'} cond={dead_cond or '-'}")
        print("     loaded as-is on purpose; see DATA_AUDIT.md Q1/Q2.")
    if big:
        print(f"  !! wide-range channels: {sorted(set(big))}")
        print("     train.normalize=zscore handles these; 'none' will likely diverge.")
    print()
    return {"dead_out": dead_out, "dead_cond": dead_cond, "wide": sorted(set(big))}
