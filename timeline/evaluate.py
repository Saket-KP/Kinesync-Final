"""Stage-1 evaluation protocol.

The readout inside the training loop is instrumentation: two videos, a fixed 0.25
threshold. This is the thing you quote. Three rules it follows that the training
readout does not:

1. **The threshold is calibrated on val, never on test.** Threshold alone sets cut
   density, so a threshold picked on test would make the shot-length comparison
   partly a measurement of how the threshold was picked.
2. **Baselines share the model's cut count, per song.** BAS and shot-length W1 both
   move with cut density, so a baseline at a different rate is not comparable. Every
   baseline here emits exactly as many cuts as the model did for that song, which
   makes the comparison about *where* the cuts are, not how many.
3. **A mismatched-conditioning arm.** The claim of Stage 1 is that editing rhythm is
   predictable *from the music*. The test of that claim is not a good BAS -- it is
   that BAS drops when you feed the model a different song's rhythm while scoring
   against the original's beats. Without this arm, a model that ignored its
   conditioning entirely and emitted a plausible cut rate would look successful.

Metric note: BAS needs sigma well below the beat period (median 511 ms in this
corpus), hence sigma=0.05 s; see peakpick.py and DATA_AUDIT.md Q4. BAS is also
reported next to a uniform-random floor and a beat-grid ceiling at the same cut
count, because a bare BAS number is not interpretable on its own.

    python -m timeline.evaluate --config timeline/configs/stage1_v2.yaml \
        --ckpt runs/stage1_v2/best.pt --out runs/stage1_v2/eval
"""
from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch
from scipy.stats import wasserstein_distance

from common.config import Config, add_config_args
from common.dataset import load_song
from common.features import Normaliser, load_norm_stats, load_raw
from common.manifest import load_index
from common.splits import videos_in_split
from timeline.diffusion import GaussianDiffusion
from timeline.model import build_model, load_checkpoint_into, pick_weights
from timeline.peakpick import (
    beat_alignment_score,
    calibrate_threshold,
    cuts_per_minute,
    pick_cuts,
    shot_lengths,
)
from timeline.sample import generate_song

BAS_SIGMA = 0.05


# --------------------------------------------------------------- extra statistics
def beat_phase_stats(cut_times: np.ndarray, beat_times: np.ndarray) -> dict:
    """Rayleigh test on cut phase within the beat interval.

    More informative than BAS: BAS saturates when sigma approaches the beat period,
    whereas R is scale-free. R is the resultant length of the phase distribution
    (0 = uniform, 1 = all cuts at the same point in the beat) and z = n*R^2 is the
    Rayleigh statistic, ~Exp(1) under uniformity, so z > 3 is p < 0.05.
    """
    c = np.asarray(cut_times, dtype=np.float64)
    b = np.sort(np.asarray(beat_times, dtype=np.float64))
    if c.size == 0 or b.size < 2:
        return {"n": 0, "R": float("nan"), "z": float("nan")}
    # keep only cuts inside the tracked beat span, where a phase is defined
    c = c[(c >= b[0]) & (c < b[-1])]
    if c.size == 0:
        return {"n": 0, "R": float("nan"), "z": float("nan")}
    i = np.searchsorted(b, c, side="right") - 1
    i = np.clip(i, 0, len(b) - 2)
    period = b[i + 1] - b[i]
    ok = period > 1e-6
    phase = np.zeros_like(c)
    phase[ok] = (c[ok] - b[i][ok]) / period[ok]
    phase = phase[ok]
    if phase.size == 0:
        return {"n": 0, "R": float("nan"), "z": float("nan")}
    ang = 2 * np.pi * phase
    R = float(np.abs(np.mean(np.exp(1j * ang))))
    return {"n": int(phase.size), "R": R, "z": float(phase.size * R * R)}


def train_shot_lengths(root, video_ids: list[str], durations: dict[str, float]) -> np.ndarray:
    """Pooled real shot lengths over the train split -- the marginal baseline draws
    from this, so 'can conditioning beat the global distribution?' has an answer."""
    out = []
    for v in video_ids:
        d = durations.get(v)
        if d is None or d <= 0:
            continue
        ct = np.asarray(load_raw(root, v)["cut_times"], dtype=np.float64)
        if ct.size:
            out.append(shot_lengths(ct, d))
    return np.concatenate(out) if out else np.array([1.0])


# ------------------------------------------------------------------- baseline cuts
def cuts_uniform(n: int, duration_s: float, rng: np.random.Generator) -> np.ndarray:
    """n cut times drawn uniformly -- the BAS floor at this cut count."""
    if n <= 0:
        return np.array([])
    return np.sort(rng.uniform(0.0, duration_s, size=n))


def cuts_beat_grid(n: int, beat_times: np.ndarray, duration_s: float) -> np.ndarray:
    """n cuts placed exactly on beats, spread evenly -- the BAS ceiling."""
    b = np.sort(np.asarray(beat_times, dtype=np.float64))
    b = b[(b > 0) & (b < duration_s)]
    if n <= 0 or b.size == 0:
        return np.array([])
    if n >= b.size:
        return b
    idx = np.unique(np.round(np.linspace(0, b.size - 1, n)).astype(int))
    return b[idx]


def pick_top_n_cuts(cut_channel: np.ndarray, fps: int, n: int,
                    min_distance_s: float = 0.25) -> np.ndarray:
    """The n strongest peaks, ignoring any height threshold.

    Used only for the mismatched-conditioning arm. A fixed threshold gives the two
    arms different cut counts, and BAS moves with cut count, so comparing them at a
    fixed threshold would confound 'where the cuts are' with 'how many there are' --
    the same confound the baselines are density-matched to avoid. Forcing the same
    count makes the ablation a pure test of placement.
    """
    from scipy.signal import find_peaks

    if n <= 0:
        return np.array([])
    dist = max(1, int(round(min_distance_s * fps)))
    pk, props = find_peaks(np.asarray(cut_channel, dtype=np.float64),
                           height=-np.inf, distance=dist)
    if pk.size == 0:
        return np.array([])
    order = np.argsort(props["peak_heights"])[::-1][:n]
    return np.sort(pk[order].astype(np.float64) / fps)


def cuts_from_marginal(n: int, duration_s: float, pool: np.ndarray,
                       rng: np.random.Generator) -> np.ndarray:
    """n cuts whose gaps are drawn from the train shot-length marginal, rescaled to
    fill the song -- matches the global shot-length distribution but ignores the
    music, so it is the baseline a conditioned model has to beat."""
    if n <= 0:
        return np.array([])
    gaps = rng.choice(pool, size=n + 1, replace=True)
    gaps = gaps / gaps.sum() * duration_s
    t = np.cumsum(gaps)[:-1]
    return t[(t > 0) & (t < duration_s)]


# ------------------------------------------------------------------------ the run
@torch.no_grad()
def generate_split(model, diff, cfg, root, norm, vids, device, guidance, steps,
                   cond_from: dict[str, str] | None = None) -> dict[str, dict]:
    """video_id -> {cut channel, song}. cond_from swaps in another song's rhythm."""
    out = {}
    for k, vid in enumerate(vids):
        song = load_song(root, cfg, vid, norm)
        use = song
        if cond_from is not None:
            other = load_song(root, cfg, cond_from[vid], norm)
            c = other.cond
            T = song.cond.shape[0]
            if c.shape[0] >= T:
                c = c[:T]
            else:                       # edge-repeat a shorter donor to length
                c = np.concatenate([c, np.repeat(c[-1:], T - c.shape[0], axis=0)])
            use = replace(song, cond=c)
        x = generate_song(model, diff, cfg, use, device=device, normaliser=norm,
                          guidance=guidance, steps=steps, progress=False,
                          seed=1000 + k)
        assert np.isfinite(x).all(), f"{vid}: non-finite output"
        out[vid] = {"cut": x[:, 0], "song": song, "full": x}
        print(f"  [{k+1}/{len(vids)}] {vid}", flush=True)
    return out


def score(cut_times: np.ndarray, song, pooled_train: np.ndarray) -> dict:
    """Metrics for one arm on one song.

    Two different W1s, because they answer different questions: `w1_vs_train_pool`
    is this song's generated shot lengths against the corpus-wide train marginal
    (does it look like a music video at all?), `w1_vs_own_real` is against this
    song's own real shot lengths (did it get *this* song right?). The second is the
    one conditioning is supposed to improve.
    """
    sl = shot_lengths(cut_times, song.duration_s)
    own = shot_lengths(song.cut_times, song.duration_s)
    ph = beat_phase_stats(cut_times, song.beat_times)
    return {
        "n_cuts": int(len(cut_times)),
        "cpm": cuts_per_minute(cut_times, song.duration_s),
        "median_shot": float(np.median(sl)) if sl.size else 0.0,
        "bas": beat_alignment_score(cut_times, song.beat_times, BAS_SIGMA),
        "phase_R": ph["R"],
        "phase_z": ph["z"],
        "w1_vs_train_pool": (float(wasserstein_distance(sl, pooled_train))
                             if sl.size else float("nan")),
        "w1_vs_own_real": (float(wasserstein_distance(sl, own))
                           if sl.size and own.size else float("nan")),
        "_sl": sl,
    }


def aggregate(rows: list[dict], pooled_real: np.ndarray) -> dict:
    sl = np.concatenate([r["_sl"] for r in rows]) if rows else np.array([0.0])
    keep = ("cpm", "median_shot", "bas", "phase_R",
            "w1_vs_train_pool", "w1_vs_own_real")
    agg = {k: float(np.nanmean([r[k] for r in rows])) for k in keep}
    agg["n_cuts_total"] = int(sum(r["n_cuts"] for r in rows))
    agg["w1_pooled"] = float(wasserstein_distance(sl, pooled_real))
    agg["phase_z_pooled"] = float(np.nansum([r["phase_z"] for r in rows]))
    return agg


def main() -> None:
    ap = argparse.ArgumentParser()
    add_config_args(ap)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--split", default="test")
    ap.add_argument("--calib-split", default="val")
    ap.add_argument("--max-videos", type=int, default=None)
    ap.add_argument("--calib-videos", type=int, default=20)
    ap.add_argument("--guidance", type=float, default=None)
    ap.add_argument("--steps", type=int, default=None)
    ap.add_argument("--threshold", type=float, default=None,
                    help="skip calibration and use this threshold")
    ap.add_argument("--no-shuffled-cond", action="store_true",
                    help="skip the mismatched-conditioning arm (halves runtime)")
    ap.add_argument("--out", default=None)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--ema", action="store_true",
                    help="evaluate the EMA average instead of the raw weights (EDGE samples from the EMA)")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    a = ap.parse_args()

    cfg = Config.load(a.config, a.overrides)
    root = cfg.data.root
    device = torch.device(a.device)
    rng = np.random.default_rng(a.seed)

    ck = torch.load(a.ckpt, map_location="cpu", weights_only=False)
    model = build_model(cfg).to(device)
    load_checkpoint_into(model, pick_weights(ck, a.ema))
    model.eval()
    diff = GaussianDiffusion(cfg.diffusion.timesteps, cfg.diffusion.cosine_s,
                             cfg.diffusion.vel_loss_weight, device)
    stats = None if cfg.train.normalize == "none" else load_norm_stats(root)
    norm = Normaliser(stats, cfg.train.normalize)
    guidance = cfg.diffusion.guidance if a.guidance is None else a.guidance
    steps = cfg.diffusion.ddim_steps if a.steps is None else a.steps

    index = load_index(root)
    durations = {v: r.duration_s for v, r in index.items()}
    test_vids = videos_in_split(root, a.split)
    if a.max_videos:
        test_vids = test_vids[: a.max_videos]

    print(f"checkpoint {a.ckpt} @ step {ck.get('step','?')}")
    print(f"split {a.split}: {len(test_vids)} songs, guidance {guidance}, "
          f"{steps} DDIM steps, BAS sigma {BAS_SIGMA}s\n")

    # the shot-length distribution every W1 is measured against
    train_vids = videos_in_split(root, "train")
    pooled_train_sl = train_shot_lengths(root, train_vids, durations)
    print(f"train shot-length pool: {pooled_train_sl.size} shots, "
          f"median {np.median(pooled_train_sl):.2f} s\n")

    # ------------------------------------------------- 1. calibrate on val, not test
    if a.threshold is not None:
        thr = a.threshold
        print(f"threshold {thr:.3f} (given, calibration skipped)\n")
    else:
        cal_vids = videos_in_split(root, a.calib_split)[: a.calib_videos]
        print(f"calibrating threshold on {len(cal_vids)} {a.calib_split} songs")
        cal = generate_split(model, diff, cfg, root, norm, cal_vids, device,
                             guidance, steps)
        target = float(np.mean([cuts_per_minute(cal[v]["song"].cut_times,
                                                cal[v]["song"].duration_s)
                                for v in cal_vids]))
        thr, cpm = calibrate_threshold([cal[v]["cut"] for v in cal_vids],
                                       [cal[v]["song"].duration_s for v in cal_vids],
                                       target, cfg.data.fps)
        print(f"  {a.calib_split} real {target:.1f} cuts/min -> threshold {thr:.3f} "
              f"gives {cpm:.1f} cuts/min\n")
        del cal

    # ------------------------------------------------------------- 2. generate test
    print(f"generating {a.split}")
    gen = generate_split(model, diff, cfg, root, norm, test_vids, device,
                         guidance, steps)

    shuf = None
    if not a.no_shuffled_cond and len(test_vids) > 1:
        donor = {v: test_vids[(i + 1) % len(test_vids)]
                 for i, v in enumerate(test_vids)}
        print(f"\ngenerating {a.split} with MISMATCHED conditioning (ablation)")
        shuf = generate_split(model, diff, cfg, root, norm, test_vids, device,
                              guidance, steps, cond_from=donor)

    # --------------------------------------------------------------- 3. score arms
    arms: dict[str, list[dict]] = {k: [] for k in
                                   ("real", "model", "beat_grid", "marginal", "uniform")}
    if shuf is not None:
        arms["model_shuffled_cond"] = []
    per_song = []

    for vid in test_vids:
        song = gen[vid]["song"]
        mc = pick_cuts(gen[vid]["cut"], cfg.data.fps, thr)
        n = len(mc)
        row = {
            "real": score(song.cut_times, song, pooled_train_sl),
            "model": score(mc, song, pooled_train_sl),
            "beat_grid": score(cuts_beat_grid(n, song.beat_times, song.duration_s),
                               song, pooled_train_sl),
            "marginal": score(cuts_from_marginal(n, song.duration_s,
                                                 pooled_train_sl, rng),
                              song, pooled_train_sl),
            "uniform": score(cuts_uniform(n, song.duration_s, rng), song,
                             pooled_train_sl),
        }
        if shuf is not None:
            # density-matched to the matched arm, so this compares placement only
            sc = pick_top_n_cuts(shuf[vid]["cut"], cfg.data.fps, n)
            row["model_shuffled_cond"] = score(sc, song, pooled_train_sl)
        for k, v in row.items():
            arms[k].append(v)
        per_song.append({
            "video_id": vid, "artist": index[vid].artist, "genre": index[vid].genre,
            "duration_s": song.duration_s,
            **{f"{k}_{m}": v[m] for k, v in row.items()
               for m in ("n_cuts", "cpm", "median_shot", "bas", "phase_R")},
        })

    # Cache the generated cut curves. Peak-picking is pure post-processing, so with the
    # curves on disk any threshold / min-separation can be swept on the CPU afterwards
    # instead of re-generating 63 songs on the GPU for every readout we want to try.
    # Wrapped: a failure here must never cost a finished evaluation.
    try:
        cdir = Path(a.out) / "curves"
        cdir.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            cdir / "model.npz",
            **{v: gen[v]["cut"].astype(np.float32) for v in test_vids})
        if shuf is not None:
            np.savez_compressed(
                cdir / "shuffled.npz",
                **{v: shuf[v]["cut"].astype(np.float32) for v in test_vids})
        print(f"\ncached generated cut curves -> {cdir}")
    except Exception as e:
        print(f"note: could not cache cut curves ({type(e).__name__}: {e})")

    summary = {k: aggregate(v, pooled_train_sl) for k, v in arms.items()}

    # ------------------------------------------------------------------ 4. report
    order = ["real", "model"]
    if shuf is not None:
        order.append("model_shuffled_cond")
    order += ["beat_grid", "marginal", "uniform"]
    cols = [("cuts/min", "cpm"), ("med shot s", "median_shot"), ("BAS", "bas"),
            ("phase R", "phase_R"), ("W1 train", "w1_pooled"),
            ("W1 own", "w1_vs_own_real")]
    print(f"\n{'='*77}\n{a.split} split, {len(test_vids)} songs, "
          f"threshold {thr:.3f} (calibrated on {a.calib_split})\n{'='*77}")
    print(f"{'arm':<22}" + "".join(f"{lab:>11}" for lab, _ in cols))
    for k in order:
        s = summary[k]
        print(f"{k:<22}" + "".join(f"{s[key]:>11.3f}" for _, key in cols))

    # The pooled z is a SUM of per-song z, each ~Exp(1) under uniformity, so the
    # null expectation is the number of songs -- not 3, which is the threshold for a
    # single test. Printing the null makes the row readable instead of impressive.
    print(f"\nRayleigh z on cut phase, summed over songs "
          f"(null expectation ~{len(test_vids)}; higher = more beat-locked):")
    for k in order:
        z = summary[k]["phase_z_pooled"]
        print(f"  {k:<22}{z:>12.1f}   {z/max(1,len(test_vids)):>6.2f}x null")

    if shuf is not None:
        d_bas = summary["model"]["bas"] - summary["model_shuffled_cond"]["bas"]
        d_r = summary["model"]["phase_R"] - summary["model_shuffled_cond"]["phase_R"]
        print(f"\nconditioning ablation: matched - mismatched = "
              f"{d_bas:+.4f} BAS, {d_r:+.4f} phase R")
        print("  (a value near zero means the model is ignoring the music; the cut "
              "rate\n   alone would still look plausible, so this is the arm that "
              "tests the claim)")

    if a.out:
        od = Path(a.out)
        od.mkdir(parents=True, exist_ok=True)
        (od / "summary.json").write_text(json.dumps({
            "ckpt": a.ckpt, "step": ck.get("step"), "split": a.split,
            "threshold": thr, "calib_split": a.calib_split,
            "guidance": guidance, "ddim_steps": steps, "bas_sigma": BAS_SIGMA,
            "n_songs": len(test_vids), "summary": summary,
        }, indent=2))
        (od / "per_song.json").write_text(json.dumps(per_song, indent=2))
        print(f"\nwrote {od/'summary.json'} and {od/'per_song.json'}")


if __name__ == "__main__":
    main()
