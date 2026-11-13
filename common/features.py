"""Feature reader and the 25 fps Stage-1 grid.

The stored data lives on three different rates (see DATA_AUDIT.md section B1):

    audio  10 fps   onset, flux, chroma, beat_phase, downbeat_phase, section
    video   2 fps   cut, flow, color_drift, brightness, saturation, valid
    CLAP    1 fps   clap                                    (Stage 2 uses this, not Stage 1)

Stage 1 trains on a single 25 fps grid:

  * `cut` is rasterised from `cut_times` at full precision -- NOT upsampled from the
    2 fps `cut` channel, which loses ~3.4% of cuts and quantises the rest to 500 ms.
    Max timing error is 20 ms, well under the ~50 ms scale of the off-beat effect.
  * the soft target is that impulse train convolved with a Gaussian (sigma ~ 80 ms),
    normalised so an isolated cut peaks at exactly 1.0. Peak *locations* stay exact.
    A 1-hot spike train at 25 fps is only ~3% positive and diffusion trains poorly
    on it; a smooth cut-likelihood curve is both learnable and exactly what the
    downstream peak-picker expects.
  * the other four video channels are linearly upsampled 2 -> 25 fps. They are
    flagged native_fps=2 so interpolation is never mistaken for measurement.

Nothing here corrects the data: no dedup, no metadata remapping, no normalisation.
Normalisation is a modelling choice and lives in timeline/ via data/norm_stats.json.

    python -m common.features --selftest
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks

from common.config import COND_SPEC, GENRES, OUT_CHANNELS, Config

VIDEO_RATE_KEYS = ("flow", "color_drift", "brightness", "saturation")
AUDIO_RATE_KEYS = ("onset", "flux", "chroma", "beat_phase", "downbeat_phase", "section")

# cond channels that get z-scored; the section one-hot block is left alone
COND_NORMALISE = tuple(n for n, _ in COND_SPEC if n != "section_onehot")


# --------------------------------------------------------------------------- io
def load_raw(root: str | Path, video_id: str) -> dict[str, np.ndarray]:
    p = Path(root) / "features" / f"{video_id}.npz"
    with np.load(p, allow_pickle=False) as z:
        return {k: z[k] for k in z.files}


def n_cuts(root: str | Path, video_id: str) -> int:
    """Number of detected cuts, without reading the rest of the file."""
    p = Path(root) / "features" / f"{video_id}.npz"
    with np.load(p, allow_pickle=False) as z:
        return int(z["cut_times"].shape[0])


def usable_videos(root: str | Path, video_ids: list[str], min_cuts: int) -> list[str]:
    """Videos with at least `min_cuts` cuts. Used by BOTH the dataset and the
    norm-stats builder so the statistics describe exactly what gets trained on."""
    if min_cuts <= 0:
        return list(video_ids)
    return [v for v in video_ids if n_cuts(root, v) >= min_cuts]


def n_frames(raw: dict[str, np.ndarray], cfg: Config) -> int:
    """Length of the target grid, derived from the audio span.

    audio_N/audio_fps and video_N/video_fps agree exactly in this corpus (ratio 5.00),
    so either works; audio is used because the conditioning must match the output length.
    """
    return int(round(raw["onset"].shape[0] * cfg.data.fps / cfg.data.audio_fps))


# -------------------------------------------------------------------- resampling
def resample_linear(a: np.ndarray, n_out: int) -> np.ndarray:
    """Linear interpolation along axis 0, mapping both endpoints onto the new grid."""
    a = np.asarray(a, dtype=np.float32)
    n_in = a.shape[0]
    if n_in == n_out:
        return a.copy()
    if n_in == 1:
        return np.repeat(a, n_out, axis=0)
    src = np.linspace(0.0, 1.0, n_in, dtype=np.float64)
    dst = np.linspace(0.0, 1.0, n_out, dtype=np.float64)
    if a.ndim == 1:
        return np.interp(dst, src, a).astype(np.float32)
    out = np.empty((n_out, a.shape[1]), dtype=np.float32)
    for j in range(a.shape[1]):
        out[:, j] = np.interp(dst, src, a[:, j])
    return out


def resample_nearest_int(a: np.ndarray, n_out: int) -> np.ndarray:
    """Nearest-neighbour for integer labels (section ids), which must not be blended."""
    a = np.asarray(a)
    n_in = a.shape[0]
    if n_in == n_out:
        return a.copy()
    idx = np.clip((np.arange(n_out) * (n_in / n_out)).astype(np.int64), 0, n_in - 1)
    return a[idx]


# ------------------------------------------------------------------ cut channel
def rasterise_cuts(cut_times: np.ndarray, n: int, fps: int) -> tuple[np.ndarray, int]:
    """Impulse train at `fps`. Returns (impulses, n_dropped_outside_grid)."""
    t = np.asarray(cut_times, dtype=np.float64)
    t = t[t >= 0.0]
    frames = np.rint(t * fps).astype(np.int64)
    inside = (frames >= 0) & (frames < n)
    dropped = int((~inside).sum())
    out = np.zeros(n, dtype=np.float32)
    # np.add.at so two cuts in one frame are visible as a value of 2
    np.add.at(out, frames[inside], 1.0)
    return out, dropped


def _impulse_peak_gain(sigma: float) -> float:
    """Peak height of a unit impulse after gaussian_filter1d, measured not assumed."""
    width = max(8, int(np.ceil(sigma * 8)) | 1)
    probe = np.zeros(width, dtype=np.float64)
    probe[width // 2] = 1.0
    return float(gaussian_filter1d(probe, sigma, mode="constant").max())


def soft_cuts(impulses: np.ndarray, sigma: float) -> np.ndarray:
    """Gaussian-smoothed cut likelihood; an isolated cut peaks at exactly 1.0."""
    if sigma <= 0:
        return impulses.astype(np.float32)
    sm = gaussian_filter1d(impulses.astype(np.float64), sigma, mode="constant")
    return (sm / _impulse_peak_gain(sigma)).astype(np.float32)


# ------------------------------------------------------------- target and cond
def build_target(raw: dict[str, np.ndarray], cfg: Config) -> tuple[np.ndarray, dict]:
    """(T, 5) float32 in OUT_CHANNELS order, plus provenance info."""
    T = n_frames(raw, cfg)
    d = cfg.data
    impulses, dropped = rasterise_cuts(raw["cut_times"], T, d.fps)
    cols = [soft_cuts(impulses, d.cut_sigma_frames)]
    for k in VIDEO_RATE_KEYS:
        cols.append(resample_linear(raw[k], T))
    x = np.stack(cols, axis=1).astype(np.float32)
    info = {
        "T": T,
        "n_cut_times": int(len(raw["cut_times"])),
        "n_impulses": int(impulses.sum()),
        "cuts_dropped_outside_grid": dropped,
        "native_fps": {"cut": "exact(cut_times)", **{k: d.video_fps for k in VIDEO_RATE_KEYS}},
    }
    return x, info


def build_cond(raw: dict[str, np.ndarray], cfg: Config) -> np.ndarray:
    """(T, 26) float32 rhythm conditioning in COND_SPEC order."""
    T = n_frames(raw, cfg)
    d = cfg.data
    parts: list[np.ndarray] = []
    for name, width in COND_SPEC:
        if name == "section_onehot":
            sec = resample_nearest_int(raw["section"], T).astype(np.int64)
            oh = np.zeros((T, width), dtype=np.float32)
            valid = (sec >= 0) & (sec < width)
            oh[np.arange(T)[valid], sec[valid]] = 1.0
            parts.append(oh)
        else:
            a = resample_linear(raw[name], T)
            a = a[:, None] if a.ndim == 1 else a
            if a.shape[1] != width:
                raise ValueError(f"{name}: expected width {width}, got {a.shape[1]}")
            parts.append(a)
    cond = np.concatenate(parts, axis=1).astype(np.float32)
    if cond.shape != (T, d.n_cond_ch):
        raise ValueError(f"cond shape {cond.shape} != {(T, d.n_cond_ch)}")
    return cond


def genre_id(genre: str) -> int:
    return GENRES.index(genre) if genre in GENRES else len(GENRES) - 1


# ------------------------------------------------------------------- norm stats
def compute_norm_stats(root: str | Path, cfg: Config, video_ids: list[str]) -> dict:
    """Per-channel mean/std over the given videos (pass the TRAIN split only).

    The cut channel is excluded: it is mapped by the fixed affine 2x-1, which needs no
    statistics and preserves the 'baseline 0, isolated peak 1' structure the
    peak-picker relies on.
    """
    out_sum = np.zeros(cfg.data.n_out_ch, dtype=np.float64)
    out_sq = np.zeros(cfg.data.n_out_ch, dtype=np.float64)
    cond_sum = np.zeros(cfg.data.n_cond_ch, dtype=np.float64)
    cond_sq = np.zeros(cfg.data.n_cond_ch, dtype=np.float64)
    n = 0
    for vid in video_ids:
        raw = load_raw(root, vid)
        x, _ = build_target(raw, cfg)
        c = build_cond(raw, cfg)
        out_sum += x.sum(0, dtype=np.float64)
        out_sq += (x.astype(np.float64) ** 2).sum(0)
        cond_sum += c.sum(0, dtype=np.float64)
        cond_sq += (c.astype(np.float64) ** 2).sum(0)
        n += x.shape[0]
    if n == 0:
        raise ValueError("no videos given to compute_norm_stats")

    def mstd(s, sq):
        m = s / n
        v = np.maximum(sq / n - m * m, 0.0)
        return m, np.sqrt(v)

    om, osd = mstd(out_sum, out_sq)
    cm, csd = mstd(cond_sum, cond_sq)

    cond_names = [n_ for n_, w in COND_SPEC for _ in range(w)]
    cond_mask = np.array([n_ in COND_NORMALISE for n_ in cond_names], dtype=bool)

    dead_out = [OUT_CHANNELS[i] for i in range(len(osd)) if osd[i] == 0]
    dead_cond = sorted({cond_names[i] for i in range(len(csd)) if csd[i] == 0})

    return {
        "n_videos": len(video_ids),
        "n_frames": int(n),
        "fps": cfg.data.fps,
        "video_ids": sorted(video_ids),
        "out_channels": list(OUT_CHANNELS),
        "out_mean": om.tolist(),
        "out_std": osd.tolist(),
        "out_skip_first": True,          # cut uses the fixed 2x-1 map
        "cond_mean": cm.tolist(),
        "cond_std": csd.tolist(),
        "cond_normalise_mask": cond_mask.tolist(),
        "dead_out_channels": dead_out,
        "dead_cond_channels": dead_cond,
    }


def save_norm_stats(root: str | Path, stats: dict) -> Path:
    p = Path(root) / "norm_stats.json"
    p.write_text(json.dumps(stats, indent=2))
    return p


def load_norm_stats(root: str | Path) -> dict:
    p = Path(root) / "norm_stats.json"
    if not p.exists():
        raise FileNotFoundError(f"{p} not found -- run `python -m common.dataset --build-stats`")
    return json.loads(p.read_text())


class Normaliser:
    """Applies the normalisation described by norm_stats.json. mode='none' is identity."""

    EPS = 1e-6

    def __init__(self, stats: dict | None, mode: str = "zscore"):
        self.mode = mode
        if mode == "none" or stats is None:
            self.enabled = False
            return
        self.enabled = True
        self.out_mean = np.asarray(stats["out_mean"], dtype=np.float32)
        self.out_std = np.asarray(stats["out_std"], dtype=np.float32)
        self.cond_mean = np.asarray(stats["cond_mean"], dtype=np.float32)
        self.cond_std = np.asarray(stats["cond_std"], dtype=np.float32)
        self.cond_mask = np.asarray(stats["cond_normalise_mask"], dtype=bool)
        self.skip_first = bool(stats.get("out_skip_first", True))

    def norm_out(self, x: np.ndarray) -> np.ndarray:
        if not self.enabled:
            return x
        y = x.copy()
        start = 1 if self.skip_first else 0
        if self.skip_first:
            y[..., 0] = y[..., 0] * 2.0 - 1.0          # cut: [0,1] -> [-1,1]
        sd = np.maximum(self.out_std[start:], self.EPS)
        y[..., start:] = (y[..., start:] - self.out_mean[start:]) / sd
        return y

    def denorm_out(self, y: np.ndarray) -> np.ndarray:
        if not self.enabled:
            return y
        x = y.copy()
        start = 1 if self.skip_first else 0
        sd = np.maximum(self.out_std[start:], self.EPS)
        x[..., start:] = x[..., start:] * sd + self.out_mean[start:]
        if self.skip_first:
            x[..., 0] = (x[..., 0] + 1.0) / 2.0
        return x

    def norm_cond(self, c: np.ndarray) -> np.ndarray:
        if not self.enabled:
            return c
        y = c.copy()
        m = self.cond_mask
        sd = np.maximum(self.cond_std[m], self.EPS)
        y[..., m] = (y[..., m] - self.cond_mean[m]) / sd
        return y


# -------------------------------------------------------------------- self-test
def selftest(root: str | Path, cfg: Config, limit: int | None = None) -> int:
    from common.manifest import load_index, summarise

    index = load_index(root)
    print(summarise(index))
    vids = sorted(index)[: limit or None]
    print(f"\nchecking {len(vids)} videos at {cfg.data.fps} fps, "
          f"cut sigma {cfg.data.cut_sigma_frames} frames "
          f"({1000*cfg.data.cut_sigma_frames/cfg.data.fps:.0f} ms)\n")

    print(f"{'vid':<8}{'T':>7}{'dur_s':>8}{'cuts':>6}{'imp':>6}{'peaks':>7}"
          f"{'merged':>8}{'cond':>12}  status")
    fails = 0
    tot_cuts = tot_peaks = 0
    for vid in vids:
        raw = load_raw(root, vid)
        x, info = build_target(raw, cfg)
        cond = build_cond(raw, cfg)

        # impulse round-trip: every cut time must be represented
        n_cuts = info["n_cut_times"] - info["cuts_dropped_outside_grid"]
        exact = info["n_impulses"] == n_cuts

        # how many survive smoothing as separable peaks
        pk, _ = find_peaks(x[:, 0], height=0.25)
        merged = n_cuts - len(pk)
        tot_cuts += n_cuts
        tot_peaks += len(pk)

        ok_len = x.shape[0] == cond.shape[0] == info["T"]
        ok_fin = bool(np.isfinite(x).all() and np.isfinite(cond).all())
        status = []
        if not exact:
            status.append(f"IMPULSE MISMATCH {info['n_impulses']}!={n_cuts}")
        if not ok_len:
            status.append("LENGTH MISMATCH")
        if not ok_fin:
            status.append("NON-FINITE")
        if info["cuts_dropped_outside_grid"]:
            status.append(f"{info['cuts_dropped_outside_grid']} cut(s) outside grid")
        if status:
            fails += 1

        print(f"{vid:<8}{info['T']:>7}{info['T']/cfg.data.fps:>8.1f}"
              f"{info['n_cut_times']:>6}{info['n_impulses']:>6}{len(pk):>7}{merged:>8}"
              f"{str(cond.shape):>12}  {'; '.join(status) or 'ok'}")

    print(f"\nimpulse round-trip exact for {len(vids)-fails}/{len(vids)} videos")
    print(f"cuts {tot_cuts} -> separable peaks {tot_peaks} "
          f"({100*tot_peaks/max(1,tot_cuts):.1f}% recovered after smoothing; "
          f"the rest are cuts closer than ~2 sigma)")
    return fails


def main() -> None:
    from common.config import add_config_args

    ap = argparse.ArgumentParser()
    add_config_args(ap)
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--limit", type=int, default=None)
    a = ap.parse_args()
    cfg = Config.load(a.config, a.overrides)

    if a.selftest:
        fails = selftest(cfg.data.root, cfg, a.limit)
        raise SystemExit(1 if fails else 0)
    ap.print_help()


if __name__ == "__main__":
    main()
