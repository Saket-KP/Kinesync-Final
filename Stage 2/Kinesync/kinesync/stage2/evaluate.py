"""Evaluate Stage 2 checkpoints on test artists (handoff 4.5).

    python -m kinesync.stage2.evaluate --runs run_A run_C --split test

For each run: scale macro-F1 / accuracy, moving F1, tag mAP per group, with 95% CIs from
1,000 bootstrap resamples of ARTISTS. Each run is scored with
  real    real spans + real video kinetics
  stage1  real spans + Stage 1 kinetics resampled onto them (one score per Stage 1 sample,
          averaged). For Run A this is "Run B"; the drop is stage1 - real (paired bootstrap).
Predictions are per shot with ground-truth previous shots (teacher forcing).
A genre-prior row (train label frequencies per genre) is included as a floor.

Writes results/stage2_eval.json and results/stage2_eval.md.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from kinesync.common import REPO_ROOT, data_root, load_vocab, videos_in
from kinesync.eval.metrics import accuracy, binary_f1, bootstrap, macro_f1, mean_ap
from kinesync.stage2.dataset import ShotDataset, collate
from kinesync.stage2.model import group_targets
from kinesync.stage2.train import load_any, load_records, to_device

METRICS = ["scale_macro_f1", "scale_acc", "moving_f1"]


@torch.no_grad()
def predict(model, records, max_len=128) -> dict:
    """Per-shot probabilities, concatenated over all songs, in record order."""
    vocab = model.vocab
    dl = DataLoader(ShotDataset(records, max_len), batch_size=16, collate_fn=collate)
    acc = {"scale": [], "moving": [], **{g.name: [] for g in vocab.groups}, "song": [], "shot": []}
    for b in dl:
        b = to_device(b, "cpu")
        out = model(b)
        m = b["mask"]
        acc["scale"].append(torch.softmax(out["scale"][m], -1).numpy())
        acc["moving"].append(torch.sigmoid(out["moving"][m]).numpy())
        for g in vocab.groups:
            lg = out["tag_" + g.name][m]
            p = torch.softmax(lg, -1)[:, :g.n] if g.mode == "single" else torch.sigmoid(lg)
            acc[g.name].append(p.numpy())
        acc["song"].append(b["song_idx"][m].numpy())
        acc["shot"].append(b["shot_idx"][m].numpy())
    out = {k: np.concatenate(v) for k, v in acc.items()}
    order = np.lexsort((out["shot"], out["song"]))      # back to record order
    return {k: v[order] for k, v in out.items()}


def truth(records, vocab) -> dict:
    return {
        "scale": np.concatenate([r.scale for r in records]),
        "moving": np.concatenate([r.moving for r in records]).astype(int),
        "tags": np.concatenate([r.tags for r in records]),
        "tag_mask": np.concatenate([r.tag_mask for r in records]) > 0,
        "artist": np.concatenate([[r.artist] * len(r.scale) for r in records]),
    }


def metric_fns(pred: dict, y: dict, vocab, prefix="") -> tuple[dict, dict]:
    """Closures f(w) for every metric, plus the artist label of each item they score."""
    fns, arts = {}, {}
    sp = pred["scale"].argmax(1)
    mp = (pred["moving"] >= 0.5).astype(int)
    fns[prefix + "scale_macro_f1"] = lambda w: macro_f1(y["scale"], sp, w, 5)
    fns[prefix + "scale_acc"] = lambda w: accuracy(y["scale"], sp, w)
    fns[prefix + "moving_f1"] = lambda w: binary_f1(y["moving"], mp, w)
    for k in ("scale_macro_f1", "scale_acc", "moving_f1"):
        arts[prefix + k] = y["artist"]
    tm = y["tag_mask"]
    for g in vocab.groups:
        f = mean_ap(y["tags"][tm][:, g.ids], pred[g.name][tm])
        fns[prefix + f"map_{g.name}"] = f
        arts[prefix + f"map_{g.name}"] = y["artist"][tm]
    return fns, arts


def genre_prior(train_recs, test_recs, vocab) -> dict:
    """Predict train label frequencies of the song's genre for every shot."""
    pred ={"scale": [], "moving": [], **{g.name: [] for g in vocab.groups}}
    by_genre = {}
    for r in train_recs:
        by_genre.setdefault(r.song.genre_id, []).append(r)
    tables = {}
    for gid, recs in by_genre.items():
        sc = np.concatenate([r.scale for r in recs])
        mv = np.concatenate([r.moving for r in recs])
        tg = np.concatenate([r.tags[r.tag_mask > 0] for r in recs])
        tables[gid] = (np.bincount(sc, minlength=5) / len(sc), mv.mean(), tg.mean(0))
    for r in test_recs:
        s, m, t = tables.get(r.song.genre_id, next(iter(tables.values())))
        n = len(r.scale)
        pred["scale"].append(np.tile(s, (n, 1)))
        pred["moving"].append(np.full(n, m))
        for g in vocab.groups:
            pred[g.name].append(np.tile(t[g.ids], (n, 1)))
    return {k: np.concatenate(v) for k, v in pred.items()}


def available_stage1_samples(vids, root, max_k=10) -> list[int]:
    d = Path(root) / "stage1"
    return [k for k in range(max_k) if all((d / f"{v}_s{k}.npz").exists() for v in vids)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", default=["run_A", "run_C"])
    ap.add_argument("--split", default="test")
    ap.add_argument("--n_boot", type=int, default=1000)
    ap.add_argument("--out", default=str(REPO_ROOT / "results"))
    ap.add_argument("--tag_min_z", type=float, default=None,
                    help="score against tags with z >= this (default: the release's tags)")
    ap.add_argument("--ensemble", default=str(REPO_ROOT / "configs" / "stage2" / "final_ensemble.yaml"),
                    help="what the run name 'final' loads")
    a = ap.parse_args()
    root = data_root()
    vocab = load_vocab(root)
    vids = videos_in(a.split, root)
    ks = available_stage1_samples(vids, root)
    real_recs = load_records(vids, vocab, "real", root=root, tag_min_z=a.tag_min_z)
    s1_recs = {k: load_records(vids, vocab, "stage1", k, root=root, tag_min_z=a.tag_min_z) for k in ks}
    y = truth(real_recs, vocab)
    any_mock = any(json.load(open(Path(root) / "stage1" / f"{vids[0]}_s{k}_shots.json")).get("mock")
                   for k in ks) if ks else False

    fns, arts = {}, {}
    prior = genre_prior(load_records(videos_in("train", root), vocab, root=root, tag_min_z=a.tag_min_z),
                        real_recs, vocab)
    f, ar = metric_fns(prior, y, vocab, "genre_prior/")
    fns.update(f), arts.update(ar)
    names = [n.split("/", 1)[1] for n in f]          # metric names, e.g. scale_macro_f1

    for run in a.runs:
        ckpt = (Path(a.ensemble) if run == "final"
                else Path(root) / "checkpoints" / "stage2" / run / "best.pt")
        model, ck = load_any(ckpt, root=root)
        f, ar = metric_fns(predict(model, real_recs), y, vocab, f"{run}/real/")
        fns.update(f), arts.update(ar)
        if ks:
            per_k = []
            for k in ks:
                fk, ark = metric_fns(predict(model, s1_recs[k]), y, vocab, f"{run}/stage1_s{k}/")
                per_k.append(fk)
                arts.update(ark)
            for n in names:   # mean over Stage 1 samples, and the paired drop vs real
                ks_f = [pk[f"{run}/stage1_s{k}/{n}"] for pk, k in zip(per_k, ks)]
                fns[f"{run}/stage1/{n}"] = (lambda fl: lambda w: float(np.mean([g(w) for g in fl])))(ks_f)
                fns[f"{run}/drop/{n}"] = (lambda fl, fr: lambda w: float(np.mean([g(w) for g in fl])) - fr(w))(
                    ks_f, fns[f"{run}/real/{n}"])
                arts[f"{run}/stage1/{n}"] = arts[f"{run}/drop/{n}"] = arts[f"{run}/real/{n}"]
            for pk in per_k:
                for key in pk:
                    arts.pop(key, None)
    fns = {k: v for k, v in fns.items() if k in arts}
    res = bootstrap(fns, arts, n=a.n_boot, seed=0)

    out = {"split": a.split, "n_songs": len(vids), "n_artists": int(len(np.unique(y["artist"]))),
           "n_shots": int(len(y["scale"])), "n_tag_shots": int(y["tag_mask"].sum()),
           "stage1_samples": ks, "stage1_is_mock": bool(any_mock), "n_boot": a.n_boot,
           "protocol": ("per-shot predictions with ground-truth previous shots (teacher forcing); "
                        "stage1 rows use REAL spans with Stage 1 kinetics resampled onto them; "
                        "tag mAP only on shots >= 0.5 s; 95% CI from artist bootstrap"),
           "metrics": res}
    Path(a.out).mkdir(parents=True, exist_ok=True)
    with open(Path(a.out) / "stage2_eval.json", "w") as fjs:
        json.dump(out, fjs, indent=1)

    def cell(key):
        v = res.get(key)
        return f"{v[0]:.3f} [{v[1]:.3f}, {v[2]:.3f}]" if v else "-"
    lines = ["# Stage 2 evaluation", ""]
    if any_mock:
        lines += ["> **Stage 1 inputs are MOCK** (`kinesync/stage1/mock.py`, perturbed real kinetics). "
                  "The `stage1` rows and the Run B drop measure robustness to noisy kinetics, "
                  "not the real Stage 1. Re-run when Stage 1 exists.", ""]
    lines += ["> Labels are automatic (CLIP-based) and unchecked; `moving` uses a placeholder threshold "
              "(~51% positive).", "",
              f"Split: **{a.split}**, {out['n_songs']} songs, {out['n_artists']} artists, {out['n_shots']} shots "
              f"({out['n_tag_shots']} with tag labels). {out['protocol']}.", ""]
    header = "| model | kinetics | " + " | ".join(names) + " |"
    lines += [header, "|" + "---|" * (len(names) + 2)]
    lines.append("| genre prior | - | " + " | ".join(cell(f"genre_prior/{n}") for n in names) + " |")
    for run in a.runs:
        lines.append(f"| {run} | real | " + " | ".join(cell(f"{run}/real/{n}") for n in names) + " |")
        if ks:
            lines.append(f"| {run} | stage1 | " + " | ".join(cell(f"{run}/stage1/{n}") for n in names) + " |")
            lines.append(f"| {run} | drop (stage1 - real) | " + " | ".join(cell(f"{run}/drop/{n}") for n in names) + " |")
    if ks and "run_A" in a.runs:
        lines += ["", "**Run B** = the `run_A` / `stage1` row; its drop row is the cost of swapping real "
                  "kinetics for Stage 1 kinetics."]
    with open(Path(a.out) / "stage2_eval.md", "w", encoding="utf-8") as fmd:
        fmd.write("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
