"""Stage-1 training loop.

    python -m timeline.train --limit 5 --set train.max_steps=2000
    python -m timeline.train --resume runs/stage1/last.pt

Resumable: a checkpoint carries model, optimiser, scaler and step, so a run killed
overnight continues exactly where it stopped. Reads only cached .npz/.json.
"""
from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from common.config import Config, add_config_args
from common.dataset import TimelineWindows
from common.features import Normaliser, load_norm_stats
from common.manifest import load_index, summarise
from common.report import channel_report
from common.splits import videos_in_split
from timeline.diffusion import GaussianDiffusion
from timeline.model import build_model, load_checkpoint_into
from timeline.peakpick import evaluate_against_real


def lr_at(step: int, cfg: Config) -> float:
    w, total, base = cfg.train.warmup_steps, cfg.train.max_steps, cfg.train.lr
    if step < w:
        return base * (step + 1) / max(1, w)
    p = (step - w) / max(1, total - w)
    return base * 0.5 * (1.0 + math.cos(math.pi * min(1.0, p)))


class EMA:
    """Exponential moving average of the weights, as in EDGE.

    EDGE keeps a deepcopy of the model updated as
    `old * beta + (1 - beta) * new` every step from step 0, with no warmup, and
    samples from it rather than from the raw weights. With beta = 0.9999 the average
    spans ~10k steps, which is half of our run, so early in training the EMA lags
    badly -- that is expected and is why it is only read at the end.

    Buffers are copied rather than averaged: they are not learned, and averaging a
    counter or a running statistic would corrupt it.
    """

    def __init__(self, model, decay: float):
        self.decay = decay
        self.shadow = {k: v.detach().clone().float()
                       for k, v in model.state_dict().items()
                       if v.dtype.is_floating_point}
        self.other = {k for k, v in model.state_dict().items()
                      if not v.dtype.is_floating_point}

    @torch.no_grad()
    def update(self, model) -> None:
        for k, v in model.state_dict().items():
            if k in self.shadow:
                self.shadow[k].mul_(self.decay).add_(v.detach().float(),
                                                     alpha=1.0 - self.decay)

    def state_dict(self, model) -> dict:
        """EMA weights, with non-float entries taken from the live model."""
        live = model.state_dict()
        out = {k: v.clone() for k, v in self.shadow.items()}
        for k in self.other:
            out[k] = live[k].clone()
        return out


def save_ckpt(path: Path, model, opt, scaler, step: int, cfg: Config,
              best_vl: float = float("inf"), ema: "EMA | None" = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model": model.state_dict(),
            # Stored beside the raw weights, never instead of them, so a checkpoint can
            # be evaluated either way and the comparison is available later.
            "ema": ema.state_dict(model) if ema is not None else None,
            "opt": opt.state_dict(),
            "scaler": scaler.state_dict() if scaler is not None else None,
            "step": step,
            # Carried so a resumed run does not forget the best val loss it had already
            # reached and overwrite a better best.pt with a worse one.
            "best_vl": best_vl,
            "config": cfg.to_dict(),
        },
        path,
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    add_config_args(ap)
    ap.add_argument("--limit", type=int, default=None, help="use only N training videos")
    ap.add_argument("--resume", default=None, help="checkpoint to resume from")
    ap.add_argument("--best-vl", type=float, default=None,
                    help="seed the best-val record when resuming a checkpoint saved "
                         "before best_vl was stored (read it off the old log)")
    ap.add_argument("--no-report", action="store_true")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    a = ap.parse_args()

    cfg = Config.load(a.config, a.overrides)
    torch.manual_seed(cfg.train.seed)
    np.random.seed(cfg.train.seed)
    root = cfg.data.root
    device = torch.device(a.device)
    out_dir = Path(cfg.train.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(summarise(load_index(root)))
    if not a.no_report:
        channel_report(root, cfg, videos_in_split(root, "train"), max_videos=8)

    stats = None if cfg.train.normalize == "none" else load_norm_stats(root)
    norm = Normaliser(stats, cfg.train.normalize)
    if stats is not None and stats.get("dead_cond_channels"):
        print(f"note: dead conditioning channels present: {stats['dead_cond_channels']}")

    train_ds = TimelineWindows(root, cfg, "train", a.limit, norm)
    val_ds = TimelineWindows(root, cfg, "val", None, norm)
    train_dl = DataLoader(
        train_ds, batch_size=cfg.train.batch_size, shuffle=True, drop_last=True,
        num_workers=cfg.train.num_workers, pin_memory=(device.type == "cuda"),
    )
    val_dl = DataLoader(val_ds, batch_size=cfg.train.batch_size, shuffle=False,
                        num_workers=0)

    model = build_model(cfg).to(device)
    diff = GaussianDiffusion(cfg.diffusion.timesteps, cfg.diffusion.cosine_s,
                             cfg.diffusion.vel_loss_weight, device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.train.lr,
                            weight_decay=cfg.train.weight_decay)
    use_amp = cfg.train.amp and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    start = 0
    resumed_best = float("inf")
    if a.resume:
        ck = torch.load(a.resume, map_location=device, weights_only=False)
        load_checkpoint_into(model, ck["model"])
        # AdamW's state is a flat list matched POSITIONALLY to the parameters, so a
        # checkpoint saved before a parameter was added cannot restore it even though
        # the weights themselves load fine. Say so plainly instead of surfacing
        # "loaded state dict contains a parameter group that doesn't match".
        try:
            opt.load_state_dict(ck["opt"])
        except ValueError as e:
            n_now = sum(len(g["params"]) for g in opt.param_groups)
            n_ck = sum(len(g["params"]) for g in ck["opt"]["param_groups"])
            raise RuntimeError(
                f"the weights loaded but the optimiser state does not fit: this model "
                f"has {n_now} parameter tensors, the checkpoint has {n_ck}. AdamW "
                f"matches its state to parameters by position, so a checkpoint from "
                f"before a parameter was added (null_cond, ada_x) cannot be resumed "
                f"into a model that has it. Start this config from scratch -- the "
                f"checkpoint is still fine to evaluate and sample from."
            ) from e
        if ck.get("scaler") and scaler is not None:
            scaler.load_state_dict(ck["scaler"])
        start = int(ck["step"])
        resumed_best = float(ck.get("best_vl", float("inf")))
        if a.best_vl is not None:
            resumed_best = a.best_vl
        print(f"resumed from {a.resume} at step {start}"
              + (f", best val so far {resumed_best:.4f}" if resumed_best < float("inf")
                 else " (checkpoint predates best-val tracking, so best.pt will be "
                      "the best from here on, not overall)"))

    # After any resume, so the average starts from the weights actually loaded rather
    # than from a fresh init it would then take ~10k steps to forget.
    ema = EMA(model, cfg.train.ema_decay) if cfg.train.ema_decay > 0 else None
    if ema is not None:
        if a.resume and ck.get("ema"):
            ema.shadow = {k: v.detach().clone().float().to(device)
                          for k, v in ck["ema"].items() if k in ema.shadow}
            print(f"restored EMA from {a.resume}")
        elif a.resume:
            print("note: that checkpoint carries no EMA, so the average restarts from "
                  "its weights; its first few thousand steps of history are lost.")
        print(f"EMA on, decay {cfg.train.ema_decay} every {cfg.train.ema_every} step(s)")

    print(f"\nmodel {model.n_params()/1e6:.2f} M params on {device}"
          f"  (amp={use_amp}, batch={cfg.train.batch_size})")
    if device.type == "cuda":
        print(f"gpu   {torch.cuda.get_device_name(0)}, "
              f"{torch.cuda.get_device_properties(0).total_memory/1e9:.1f} GB")
    print(f"steps {start} -> {cfg.train.max_steps}\n")

    log_path = out_dir / "log.jsonl"
    cfg.save(out_dir / "config.yaml")

    def val_loss() -> float:
        """Validation loss with a FIXED timestep/noise draw.

        The denoising loss depends strongly on t (near t=T it is bounded below by the
        data variance; near t=0 it goes to zero), so drawing t afresh at every eval
        adds variance that has nothing to do with the weights -- two evals of the same
        model would disagree. Here t is a deterministic stratified sweep over [0, T)
        and the noise is seeded, so the curve across evals is comparable and changes
        in it are changes in the model.
        """
        model.eval()
        tot = n = 0.0
        g_cpu = torch.Generator().manual_seed(1234)   # same noise at every eval
        with torch.no_grad():
            for bi, b in enumerate(val_dl):
                x = b["x"].to(device); c = b["cond"].to(device); g = b["genre"].to(device)
                B = x.shape[0]
                d = torch.zeros(B, dtype=torch.bool, device=device)
                # stratified over the whole schedule, offset per batch so the union
                # of batches covers [0, T) evenly rather than repeating one grid
                frac = (torch.arange(B, dtype=torch.float64) + (bi % 8) / 8.0) / B
                t = (frac * diff.T).long().clamp_(0, diff.T - 1).to(device)
                noise = torch.randn(x.shape, generator=g_cpu).to(device)
                with torch.amp.autocast("cuda", enabled=use_amp):
                    out = diff.training_loss(model, x, c, g, d, t=t, noise=noise)
                tot += float(out["loss"]) * B
                n += B
        model.train()
        return tot / max(1.0, n)

    model.train()
    step = start
    t0 = time.time()
    run = {"loss": 0.0, "cut_mse": 0.0, "k": 0}
    best_vl = resumed_best
    while step < cfg.train.max_steps:
        for b in train_dl:
            if step >= cfg.train.max_steps:
                break
            for gp in opt.param_groups:
                gp["lr"] = lr_at(step, cfg)

            x = b["x"].to(device, non_blocking=True)
            c = b["cond"].to(device, non_blocking=True)
            g = b["genre"].to(device, non_blocking=True)
            drop = torch.rand(x.shape[0], device=device) < cfg.diffusion.cfg_dropout
            # Independent of the genre mask: a sample may lose the genre, the rhythm,
            # both or neither, so the network learns every combination it will be
            # asked for at sampling time.
            # None, not an all-False mask, when rhythm dropout is off: that way a model
            # built without a null_cond token is never handed a drop request at all.
            drop_c = (torch.rand(x.shape[0], device=device) < cfg.diffusion.cond_dropout
                      if cfg.diffusion.cond_dropout > 0 else None)

            with torch.amp.autocast("cuda", enabled=use_amp):
                out = diff.training_loss(model, x, c, g, drop, drop_cond=drop_c)
            opt.zero_grad(set_to_none=True)
            scaler.scale(out["loss"]).backward()
            scaler.unscale_(opt)
            gn = torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.train.grad_clip)
            scaler.step(opt)
            scaler.update()

            run["loss"] += float(out["loss"]); run["cut_mse"] += float(out["cut_mse"])
            run["k"] += 1
            step += 1

            if ema is not None and step % cfg.train.ema_every == 0:
                ema.update(model)

            if step % cfg.train.log_every == 0:
                k = max(1, run["k"])
                msg = (f"step {step:>6}/{cfg.train.max_steps}  "
                       f"loss {run['loss']/k:.4f}  cut_mse {run['cut_mse']/k:.4f}  "
                       f"lr {lr_at(step, cfg):.2e}  gnorm {float(gn):.2f}  "
                       # steps done in THIS process, not the absolute step count, or a
                       # resumed run reports a wildly inflated rate
                       f"{(step-start)/(time.time()-t0+1e-9):.1f} it/s")
                print(msg)
                with open(log_path, "a") as f:
                    f.write(json.dumps({"step": step, "loss": run["loss"]/k,
                                        "cut_mse": run["cut_mse"]/k}) + "\n")
                run = {"loss": 0.0, "cut_mse": 0.0, "k": 0}

            if step % cfg.train.eval_every == 0:
                vl = val_loss()
                print(f"  [eval] step {step}  val_loss {vl:.4f}")
                # Keep the minimum-val checkpoint separately: the last step is not
                # necessarily the best one, and evaluation should use the best.
                if vl < best_vl:
                    best_vl = vl
                    save_ckpt(out_dir / "best.pt", model, opt, scaler, step, cfg,
                              best_vl, ema=ema)
                    print(f"  [ckpt] best so far -> {out_dir/'best.pt'} (val {vl:.4f})")
                try:
                    m = evaluate_against_real(model, diff, cfg, root, norm,
                                              split="val", max_videos=2, device=device)
                    print(f"  [eval] shot-len W1 {m['w1_shot_len']:.3f} s  "
                          f"BAS gen {m['bas_gen']:.3f} / real {m['bas_real']:.3f}  "
                          f"cuts/min gen {m['cpm_gen']:.1f} / real {m['cpm_real']:.1f}")
                except Exception as e:  # instrumentation must never kill a run
                    print(f"  [eval] generation metrics skipped: {type(e).__name__}: {e}")
                with open(log_path, "a") as f:
                    f.write(json.dumps({"step": step, "val_loss": vl}) + "\n")

            if step % cfg.train.ckpt_every == 0:
                save_ckpt(out_dir / "last.pt", model, opt, scaler, step, cfg,
                          best_vl, ema=ema)
                print(f"  [ckpt] {out_dir/'last.pt'} @ step {step}")

    # best_vl was being dropped here, so the final last.pt reported best val inf and a
    # run resumed from it would overwrite a better best.pt with a worse one.
    save_ckpt(out_dir / "last.pt", model, opt, scaler, step, cfg, best_vl, ema=ema)
    print(f"\ndone at step {step}; checkpoint {out_dir/'last.pt'}"
          f"  (best val {best_vl:.4f} -> {out_dir/'best.pt'})")


if __name__ == "__main__":
    main()
