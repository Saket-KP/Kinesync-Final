"""Full-song Stage-1 generation.

A trained window is 10 s; a song is 3-6 minutes. Following EDGE, long sequences are
produced by generating overlapping windows and *constraining* the overlap: at every
denoising step the first half of window k+1 is replaced by the correctly-noised
second half of window k, so consecutive windows agree where they overlap and the
result has no seam. (A plain cross-fade would blur exactly the sharp cut peaks the
model exists to place.)

    python -m timeline.sample --ckpt runs/stage1/last.pt --video v0032
    python -m timeline.sample --ckpt runs/stage1/last.pt --split test --out out/
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from common.config import OUT_CHANNELS, Config, add_config_args
from common.dataset import Song, load_song
from common.features import Normaliser, load_norm_stats
from common.manifest import load_index
from common.splits import videos_in_split
from timeline.diffusion import GaussianDiffusion
from timeline.model import build_model, load_checkpoint_into, pick_weights
from timeline.peakpick import (
    beat_alignment_score,
    cuts_per_minute,
    cuts_to_shots,
    pick_cuts,
    shot_lengths,
)


@torch.no_grad()
def generate_song_batch(
    model,
    diffusion: GaussianDiffusion,
    cfg: Config,
    song: Song,
    device="cpu",
    normaliser: Normaliser | None = None,
    guidance: float | None = None,
    steps: int | None = None,
    progress: bool = True,
    seed: int | None = None,
) -> np.ndarray:
    """EDGE's long_ddim_sample: the whole song at once, as one batch of windows.

    Every window of the song is denoised simultaneously, and after each step the
    overlap constraint `x[1:, :half] = x[:-1, half:]` is re-applied. Nothing is ever
    frozen, so a decision made at the start of the song can still be revised by what
    the model is doing at the end -- which the sequential path cannot do, because there
    each window's prefix is fixed the moment its predecessor finishes.

    Requires hop == window/2 (checked in Config.validate).
    """
    W = cfg.data.window_frames
    half = W // 2
    T = song.cond.shape[0]
    C = cfg.data.n_out_ch
    guidance = cfg.diffusion.guidance if guidance is None else guidance
    steps = cfg.diffusion.ddim_steps if steps is None else steps

    # n windows at stride `half` cover (n+1)*half frames
    n = max(2, int(np.ceil(T / half)) - 1)
    cond_all = torch.from_numpy(song.cond).to(device)                  # (T, 26)
    cw = torch.empty(n, W, cond_all.shape[1], device=device)
    for k in range(n):
        s = k * half
        e = min(s + W, T)
        cw[k, : e - s] = cond_all[s:e]
        if e - s < W:                              # edge-repeat past the end of the song
            cw[k, e - s:] = cond_all[T - 1]
    genre = torch.full((n,), song.genre, device=device, dtype=torch.long)

    gen = None
    if seed is not None:
        gen = torch.Generator(device=device).manual_seed(seed)

    x = diffusion.ddim_sample(
        model, (n, W, C), cw, genre, steps=steps, guidance=guidance,
        generator=gen, progress=progress,
        guidance_target=cfg.diffusion.guidance_target,
        eta=cfg.diffusion.ddim_eta,
        guidance_ramp=cfg.diffusion.guidance_ramp,
        clip_cut=cfg.diffusion.clip_cut,
        overlap_half=half,
    ).float().cpu().numpy()

    # The constraint holds exactly on the last step too, so x[k+1,:half] == x[k,half:]
    # and the song is a plain concatenation -- no blending, no smeared peaks.
    out = np.concatenate([x[0]] + [x[k, half:] for k in range(1, n)], axis=0)[:T]
    assert out.shape == (T, C), f"assembled {out.shape}, expected {(T, C)}"
    if normaliser is not None:
        out = normaliser.denorm_out(out)
    return out


@torch.no_grad()
def generate_song(
    model,
    diffusion: GaussianDiffusion,
    cfg: Config,
    song: Song,
    device="cpu",
    normaliser: Normaliser | None = None,
    guidance: float | None = None,
    steps: int | None = None,
    progress: bool = True,
    seed: int | None = None,
) -> np.ndarray:
    """(T, 5) de-normalised timeline for a whole song, at cfg.data.fps."""
    if cfg.diffusion.long_mode == "batch":
        return generate_song_batch(model, diffusion, cfg, song, device=device,
                                   normaliser=normaliser, guidance=guidance,
                                   steps=steps, progress=progress, seed=seed)
    W = cfg.data.window_frames
    hop = cfg.data.hop_frames
    T = song.cond.shape[0]
    C = cfg.data.n_out_ch
    guidance = cfg.diffusion.guidance if guidance is None else guidance
    steps = cfg.diffusion.ddim_steps if steps is None else steps

    cond_all = torch.from_numpy(song.cond).to(device).unsqueeze(0)   # (1,T,26)
    genre = torch.tensor([song.genre], device=device, dtype=torch.long)

    gen = None
    if seed is not None:
        gen = torch.Generator(device=device).manual_seed(seed)

    out = np.zeros((T, C), dtype=np.float32)
    starts = list(range(0, max(1, T - W + 1), hop))
    if starts[-1] + W < T:                      # make sure the tail is covered
        starts.append(T - W)

    it = starts
    if progress:
        import sys

        from tqdm import tqdm
        # only draw a bar on a real terminal; piped output would be flooded
        it = tqdm(starts, desc=f"sample {song.video_id}", leave=False,
                  disable=not sys.stderr.isatty())

    prev_end: int | None = None
    for s in it:
        e = min(s + W, T)
        cw = cond_all[:, s:e]
        n = e - s
        if n < W:                                # pad the final window by edge-repeat
            pad = cond_all[:, e - 1:e].expand(1, W - n, -1)
            cw = torch.cat([cw, pad], dim=1)

        # The overlap is computed from where the previous window actually ENDED, not
        # assumed to be W - hop. The tail window is appended at T - W to cover the end
        # of the song, so it is generally NOT hop-spaced from its predecessor; taking
        # the known prefix on faith would pin it to the wrong point in time (up to a
        # full hop out, ~5 s at the default settings) and corrupt the last window of
        # every song whose length is not an exact multiple of the hop.
        known = known_mask = None
        ov = 0 if prev_end is None else min(prev_end - s, W)
        if ov > 0:
            known = torch.zeros(1, W, C, device=device)
            known[:, :ov] = torch.from_numpy(out[s:s + ov].copy()).to(device).unsqueeze(0)
            known_mask = torch.zeros(W, dtype=torch.bool, device=device)
            known_mask[:ov] = True

        x = diffusion.ddim_sample(
            model, (1, W, C), cw, genre, steps=steps, guidance=guidance,
            known=known, known_mask=known_mask, generator=gen, progress=False,
            guidance_target=cfg.diffusion.guidance_target,
            eta=cfg.diffusion.ddim_eta,
            guidance_ramp=cfg.diffusion.guidance_ramp,
            clip_cut=cfg.diffusion.clip_cut,
        )
        xw = x[0].float().cpu().numpy()

        # keep the already-generated overlap, write only the genuinely new frames
        out[s + ov:e] = xw[ov:e - s]
        prev_end = e

    if normaliser is not None:
        out = normaliser.denorm_out(out)
    return out


def song_report(x: np.ndarray, song: Song, cfg: Config, threshold: float = 0.25) -> dict:
    cut = x[:, 0]
    gc = pick_cuts(cut, cfg.data.fps, threshold)
    rc = song.cut_times
    gsl, rsl = shot_lengths(gc, song.duration_s), shot_lengths(rc, song.duration_s)
    return {
        "video_id": song.video_id,
        "duration_s": song.duration_s,
        "frames": int(x.shape[0]),
        "n_cuts_gen": int(len(gc)),
        "n_cuts_real": int(len(rc)),
        "cpm_gen": cuts_per_minute(gc, song.duration_s),
        "cpm_real": cuts_per_minute(rc, song.duration_s),
        "median_shot_gen": float(np.median(gsl)) if gsl.size else 0.0,
        "median_shot_real": float(np.median(rsl)) if rsl.size else 0.0,
        "bas_gen": beat_alignment_score(gc, song.beat_times),
        "bas_real": beat_alignment_score(rc, song.beat_times),
        "threshold": threshold,
        "shots": cuts_to_shots(gc, song.duration_s),
    }


def _hist(vals: np.ndarray, bins: np.ndarray) -> list[int]:
    h, _ = np.histogram(vals, bins=bins)
    return h.tolist()


def print_report(rep: dict, song: Song) -> None:
    print(f"\n--- {rep['video_id']}  ({rep['duration_s']:.0f} s, {rep['frames']} frames) ---")
    print(f"{'':<18}{'generated':>12}{'real':>12}")
    for label, kg, kr in (
        ("cuts", "n_cuts_gen", "n_cuts_real"),
        ("cuts/min", "cpm_gen", "cpm_real"),
        ("median shot (s)", "median_shot_gen", "median_shot_real"),
        ("BAS", "bas_gen", "bas_real"),
    ):
        g, r = rep[kg], rep[kr]
        fmt = "{:>12.2f}" if isinstance(g, float) else "{:>12}"
        print(f"{label:<18}" + fmt.format(g) + fmt.format(r))

    bins = np.array([0, 0.5, 1, 1.5, 2, 3, 4, 6, 8, 12, 1e9])
    gsl = np.array([s["duration"] for s in rep["shots"]])
    rsl = shot_lengths(song.cut_times, song.duration_s)
    print(f"\nshot-length histogram  {'gen':>6} {'real':>6}")
    for i in range(len(bins) - 1):
        a, b = bins[i], bins[i + 1]
        lab = f"{a:g}-{b:g}s" if b < 1e8 else f">{a:g}s"
        print(f"  {lab:<20}{_hist(gsl, bins)[i]:>6} {_hist(rsl, bins)[i]:>6}")


def main() -> None:
    ap = argparse.ArgumentParser()
    add_config_args(ap)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--video", default=None, help="one video id")
    ap.add_argument("--split", default=None, help="sample every video in a split")
    ap.add_argument("--out", default=None, help="directory for .npy/.json output")
    ap.add_argument("--threshold", type=float, default=0.25)
    ap.add_argument("--guidance", type=float, default=None)
    ap.add_argument("--steps", type=int, default=None)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--ema", action="store_true",
                    help="evaluate the EMA average instead of the raw weights (EDGE samples from the EMA)")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    a = ap.parse_args()

    ck = torch.load(a.ckpt, map_location="cpu", weights_only=False)
    cfg = Config.load(a.config, a.overrides)
    device = torch.device(a.device)

    model = build_model(cfg).to(device)
    load_checkpoint_into(model, pick_weights(ck, a.ema))
    model.eval()
    diff = GaussianDiffusion(cfg.diffusion.timesteps, cfg.diffusion.cosine_s,
                             cfg.diffusion.vel_loss_weight, device)
    stats = None if cfg.train.normalize == "none" else load_norm_stats(cfg.data.root)
    norm = Normaliser(stats, cfg.train.normalize)

    if a.video:
        vids = [a.video]
    elif a.split:
        vids = videos_in_split(cfg.data.root, a.split)
    else:
        vids = videos_in_split(cfg.data.root, "test")[:1]

    index = load_index(cfg.data.root)
    print(f"checkpoint step {ck.get('step','?')}, {len(vids)} song(s), "
          f"guidance {a.guidance if a.guidance is not None else cfg.diffusion.guidance}, "
          f"{a.steps or cfg.diffusion.ddim_steps} DDIM steps")

    outdir = Path(a.out) if a.out else None
    if outdir:
        outdir.mkdir(parents=True, exist_ok=True)

    for vid in vids:
        song = load_song(cfg.data.root, cfg, vid, norm)
        x = generate_song(model, diff, cfg, song, device=device, normaliser=norm,
                          guidance=a.guidance, steps=a.steps, seed=a.seed)
        rep = song_report(x, song, cfg, a.threshold)
        rep["artist"] = index[vid].artist
        rep["genre"] = index[vid].genre
        print_report(rep, song)

        # sanity assertions the plan asks sampling to satisfy
        expect = int(round(song.duration_s * cfg.data.fps))
        assert abs(x.shape[0] - expect) <= cfg.data.fps, \
            f"{vid}: generated {x.shape[0]} frames, expected ~{expect}"
        assert np.isfinite(x).all(), f"{vid}: non-finite output"

        if outdir:
            np.save(outdir / f"{vid}_timeline.npy", x)
            (outdir / f"{vid}_shots.json").write_text(json.dumps(
                {"video_id": vid, "fps": cfg.data.fps,
                 "channels": list(OUT_CHANNELS), **rep}, indent=2))
            print(f"\nwrote {outdir/f'{vid}_timeline.npy'} and "
                  f"{outdir/f'{vid}_shots.json'}")


if __name__ == "__main__":
    main()
