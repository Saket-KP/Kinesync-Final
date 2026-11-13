"""Train the Stage 2 model (handoff 4.3).

    python -m kinesync.stage2.train --config configs/stage2/run_A.yaml
    python -m kinesync.stage2.train --config configs/stage2/run_A.yaml --overfit 5 --steps 400

Checkpoints: checkpoints/stage2/{run_name}/best.pt (lowest val NLL) and last.pt.
Log: checkpoints/stage2/{run_name}/log.csv (+ TensorBoard if installed).
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import time
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from kinesync.common import (REPO_ROOT, data_root, genre_list, load_splits, load_vocab,
                             load_yaml, seed_everything, videos_in)
from kinesync.stage2.dataset import ShotDataset, collate, label_stats, load_record
from kinesync.stage2.model import Stage2CrossModel, Stage2Ensemble, Stage2Model, stage2_loss


def load_records(vids, vocab, kin_source="real", stage1_sample=0, root=None, tag_min_z=None):
    sp = load_splits(root)
    return [load_record(v, vocab, kin_source, stage1_sample, root, artist=sp.loc[v, "artist"],
                        tag_min_z=tag_min_z) for v in vids]


def to_device(b: dict, device) -> dict:
    return {k: v.to(device) for k, v in b.items()}


def build_model(cfg: dict, vocab, n_genres: int) -> Stage2Model:
    """model.type: "self" (default, the handoff's design) or "cross" (adds cross-attention to the song)."""
    m = cfg["model"]
    cls = Stage2CrossModel if m.get("type", "self") == "cross" else Stage2Model
    return cls(vocab, n_genres, d=m["d"], layers=m["layers"], heads=m["heads"], ff=m["ff"], dropout=m["dropout"])


def load_checkpoint(path, device="cpu", root=None):
    ck = torch.load(path, map_location=device, weights_only=False)
    vocab = load_vocab(root)
    assert vocab.version == ck["vocab_version"], "checkpoint was trained on another vocab"
    model = build_model(ck["config"], vocab, len(ck["genres"])).to(device)
    model.load_state_dict(ck["state_dict"])
    model.eval()
    return model, ck


def load_any(path, device="cpu", root=None):
    """A single checkpoint (.pt) or an ensemble listed in a YAML file (configs/stage2/final_ensemble.yaml).
    Returns (model, checkpoint dict); for an ensemble the dict is the first member's plus "members"."""
    path = Path(path)
    if path.suffix not in (".yaml", ".yml"):
        return load_checkpoint(path, device, root)
    members = [REPO_ROOT / p for p in load_yaml(path)["members"]]
    loaded = [load_checkpoint(p, device, root) for p in members]
    ck = dict(loaded[0][1])
    for _, other in loaded[1:]:
        assert other["genres"] == ck["genres"] and other["label_stats"] == ck["label_stats"],             "ensemble members must share genres and training data"
    ck["members"] = [str(p.relative_to(REPO_ROOT)) for p in members]
    ck["sampling_overrides"] = load_yaml(path).get("sampling", {})   # e.g. a tau_tag tuned for this ensemble
    return Stage2Ensemble([m for m, _ in loaded]).eval(), ck


def amp_context(device: torch.device, mode: str):
    if mode == "off" or device.type != "cuda":
        return nullcontext(), None
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    scaler = torch.amp.GradScaler("cuda") if dtype == torch.float16 else None
    return torch.autocast("cuda", dtype=dtype), scaler


@torch.no_grad()
def evaluate_nll(model, loader, vocab, device, sw, pw) -> dict:
    """Shot-weighted mean of each term over a split. 'nll_total' (unweighted) drives early stopping."""
    model.eval()
    sums, n = {}, 0
    for b in loader:
        b = to_device(b, device)
        out = model(b)
        k = int(b["mask"].sum())
        for tag, weighted in (("nll", False), ("loss", True)):
            for name, v in stage2_loss(out, b, vocab, sw, pw, weighted=weighted).items():
                sums[f"{tag}_{name}"] = sums.get(f"{tag}_{name}", 0.0) + float(v) * k
        n += k
    model.train()
    return {k: v / max(n, 1) for k, v in sums.items()}


def train(cfg: dict, overfit: int | None = None, steps: int | None = None, device: str | None = None,
          root=None, quiet: bool = False) -> dict:
    root = root or data_root()
    seed_everything(cfg["seed"])
    device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    vocab = load_vocab(root)
    genres = genre_list(root)
    log = print if not quiet else (lambda *a, **k: None)

    train_vids = videos_in("train", root)
    if overfit:
        train_vids = train_vids[:overfit]
        cfg = json.loads(json.dumps(cfg))
        cfg["model"]["dropout"] = 0.0
        cfg["optim"]["warmup_steps"] = min(cfg["optim"]["warmup_steps"], 20)
        cfg["run_name"] = f"{cfg['run_name']}_overfit{overfit}"
    t0 = time.time()
    zcut = cfg.get("tag_min_z")                      # None = the release's tags (z >= 1.0)
    train_recs = load_records(train_vids, vocab, root=root, tag_min_z=zcut)
    val_recs = [] if overfit else load_records(videos_in("val", root), vocab, root=root, tag_min_z=zcut)
    log(f"loaded {len(train_recs)} train / {len(val_recs)} val songs in {time.time() - t0:.1f}s")

    stats = label_stats(train_recs, vocab)
    log(f"setting repetition: the most frequent setting covers {stats['setting_max_share_p50']:.2f} of a "
        f"song's shots (median), {stats['setting_max_share_p95']:.2f} (95th pct) -> suggested rep_limit "
        f"= {stats['setting_max_share_p95']:.2f}")
    sw = torch.tensor(stats["scale_weight"], dtype=torch.float32, device=device)
    pw = torch.tensor(stats["tag_pos_weight"], dtype=torch.float32, device=device)

    oc = cfg["optim"]
    tr_ds = ShotDataset(train_recs, cfg["max_len"], train=True, kin_noise=cfg["kin_noise"],
                        jitter_frames=cfg["jitter_frames"], seed=cfg["seed"])
    g = torch.Generator().manual_seed(cfg["seed"])
    tr_dl = DataLoader(tr_ds, batch_size=oc["batch_size"], shuffle=True, collate_fn=collate,
                       generator=g, num_workers=cfg.get("num_workers", 0))
    va_dl = (DataLoader(ShotDataset(val_recs, cfg["max_len"]), batch_size=oc["batch_size"],
                        collate_fn=collate) if val_recs else None)

    model = build_model(cfg, vocab, len(genres)).to(device)
    log(f"Stage 2 parameters: {model.n_params():,}")
    opt = torch.optim.AdamW(model.parameters(), lr=oc["lr"], weight_decay=oc["weight_decay"])
    total_steps = steps or oc["max_epochs"] * len(tr_dl)
    warm = oc["warmup_steps"]

    def lr_lambda(s):
        if s < warm:
            return (s + 1) / warm
        return 0.5 * (1 + math.cos(math.pi * min(1.0, (s - warm) / max(1, total_steps - warm))))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)
    autocast, scaler = amp_context(device, cfg.get("amp", "auto"))

    out_dir = Path(root) / "checkpoints" / "stage2" / cfg["run_name"]
    out_dir.mkdir(parents=True, exist_ok=True)
    tb = None
    try:
        from torch.utils.tensorboard import SummaryWriter
        tb = SummaryWriter(str(out_dir / "tb"))
    except Exception:
        pass

    def save(name, epoch, val_nll):
        torch.save({"state_dict": model.state_dict(), "config": cfg, "genres": genres,
                    "vocab_version": vocab.version, "label_stats": stats, "epoch": epoch,
                    "val_nll": val_nll}, out_dir / name)

    rows, best, bad, step, epoch = [], float("inf"), 0, 0, 0
    model.train()
    while step < total_steps:
        epoch += 1
        sums, nb = {}, 0
        for b in tr_dl:
            b = to_device(b, device)
            with autocast:
                out = model(b)
            out = {k: v.float() for k, v in out.items()}
            terms = stage2_loss(out, b, vocab, sw, pw, weighted=True)
            opt.zero_grad(set_to_none=True)
            if scaler:
                scaler.scale(terms["total"]).backward()
                scaler.unscale_(opt)
            else:
                terms["total"].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), oc["grad_clip"])
            (scaler.step(opt), scaler.update()) if scaler else opt.step()
            sched.step()
            step += 1
            for k, v in terms.items():
                sums[k] = sums.get(k, 0.0) + v.item()
            nb += 1
            if step >= total_steps:
                break
        row = {"epoch": epoch, "step": step, "lr": sched.get_last_lr()[0]}
        row.update({f"train_{k}": v / nb for k, v in sums.items()})
        if va_dl is not None:
            row.update({f"val_{k}": v for k, v in evaluate_nll(model, va_dl, vocab, device, sw, pw).items()})
        rows.append(row)
        if tb:
            for k, v in row.items():
                if k not in ("epoch", "step"):
                    tb.add_scalar(k, v, step)
        msg = f"ep {epoch:3d} step {step:5d} train {row['train_total']:.3f}"
        if va_dl is not None:
            msg += (f" | val nll {row['val_nll_total']:.3f} (scale {row['val_nll_scale']:.3f}, "
                    f"moving {row['val_nll_moving']:.3f})")
        log(msg, flush=True)
        with open(out_dir / "log.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[-1].keys()))
            w.writeheader()
            w.writerows(rows)
        crit = row.get("val_nll_total", row["train_total"])
        if crit < best:
            best, bad = crit, 0
            save("best.pt", epoch, crit)
        else:
            bad += 1
            if va_dl is not None and bad >= oc["patience"]:
                log(f"early stop: no val NLL improvement for {bad} epochs")
                break
    save("last.pt", epoch, rows[-1].get("val_nll_total"))
    if tb:
        tb.close()
    log(f"best {'val NLL' if va_dl else 'train loss'} {best:.4f}; checkpoints in {out_dir}")
    return {"best": best, "final_train": rows[-1]["train_total"], "rows": rows, "out_dir": str(out_dir),
            "n_params": model.n_params()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--overfit", type=int, help="train on N train songs only, no validation")
    ap.add_argument("--steps", type=int, help="stop after this many optimiser steps")
    ap.add_argument("--max_epochs", type=int)
    ap.add_argument("--device")
    a = ap.parse_args()
    cfg = load_yaml(a.config)
    if a.max_epochs:
        cfg["optim"]["max_epochs"] = a.max_epochs
    res = train(cfg, overfit=a.overfit, steps=a.steps, device=a.device)
    if a.overfit:
        print(f"OVERFIT {a.overfit}: final train loss {res['final_train']:.4f}")


if __name__ == "__main__":
    main()
