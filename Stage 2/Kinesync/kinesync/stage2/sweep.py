"""Small hyperparameter search for Stage 2, scored on VAL only (never test).

    python -m kinesync.stage2.sweep                 # trains every variant, writes results/stage2_sweep.md

Each variant changes one thing from Run A. Selection uses the same criterion as early stopping
(unweighted val NLL, lower is better). Val metrics are reported alongside. Three seeds of the
baseline show how much of a difference is just training noise.
"""
from __future__ import annotations

import argparse
import copy
import json
import time
from pathlib import Path

import numpy as np

from kinesync.common import REPO_ROOT, load_vocab, load_yaml, videos_in
from kinesync.stage2.evaluate import metric_fns, predict, truth
from kinesync.stage2.train import load_checkpoint, load_records, train

VARIANTS = {   # name -> changes to configs/stage2/run_A.yaml
    "base seed 1234": {},
    "base seed 1": {"seed": 1},
    "base seed 2": {"seed": 2},
    "dropout 0.2": {"model.dropout": 0.2},
    "dropout 0.3": {"model.dropout": 0.3},
    "weight decay 0.05": {"optim.weight_decay": 0.05},
    "lr 1e-4": {"optim.lr": 1e-4},
    "lr 1e-3": {"optim.lr": 1e-3},
    "d 128, ff 512": {"model.d": 128, "model.ff": 512},
    "2 layers": {"model.layers": 2},
    "6 layers": {"model.layers": 6},
    "kin noise 0.3 + jitter (Run C)": {"kin_noise": 0.3, "jitter_frames": 2},
    # cross-attention to the song (model.py, Stage2CrossModel), same members as the final ensemble
    "cross seed 1234": {"model.type": "cross"},
    "cross seed 1": {"model.type": "cross", "seed": 1},
    "cross seed 2": {"model.type": "cross", "seed": 2},
    "cross d 128, ff 512": {"model.type": "cross", "model.d": 128, "model.ff": 512},
}


def apply(cfg: dict, changes: dict) -> dict:
    cfg = copy.deepcopy(cfg)
    for key, val in changes.items():
        d = cfg
        *path, last = key.split(".")
        for p in path:
            d = d[p]
        d[last] = val
    return cfg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default=None)
    ap.add_argument("--only", nargs="*", help="run only these variant names")
    ap.add_argument("--out_name", default="stage2_sweep", help="results/<out_name>.{json,md}")
    a = ap.parse_args()
    base = load_yaml(REPO_ROOT / "configs" / "stage2" / "run_A.yaml")
    vocab = load_vocab()
    val = load_records(videos_in("val"), vocab)
    y = truth(val, vocab)
    rows = []
    for i, (name, changes) in enumerate(VARIANTS.items()):
        if a.only and name not in a.only:
            continue
        cfg = apply(base, changes)
        cfg["run_name"] = f"sweep_{i:02d}"
        t0 = time.time()
        res = train(cfg, device=a.device, quiet=True)
        model, _ = load_checkpoint(Path(res["out_dir"]) / "best.pt")
        fns, _ = metric_fns(predict(model.cpu(), val), y, vocab)
        m = {k: round(f(np.ones(len(y["artist"]) if "map" not in k else int(y["tag_mask"].sum()))), 4)
             for k, f in fns.items()}
        best_epoch = min(res["rows"], key=lambda r: r["val_nll_total"])["epoch"]
        row = {"variant": name, "run": cfg["run_name"], "changes": changes, "val_nll": round(res["best"], 4),
               "best_epoch": best_epoch, "params": res["n_params"], "minutes": round((time.time() - t0) / 60, 1), **m}
        rows.append(row)
        print(json.dumps(row), flush=True)
    out = REPO_ROOT / "results"
    with open(out / f"{a.out_name}.json", "w") as f:
        json.dump(rows, f, indent=1)
    seeds = [r["val_nll"] for r in rows if r["variant"].startswith("base seed")]
    noise = (max(seeds) - min(seeds)) if len(seeds) > 1 else None
    keys = ["val_nll", "scale_macro_f1", "moving_f1", "map_setting", "map_subject", "map_action", "map_lighting", "map_style"]
    lines = ["# Stage 2 hyperparameter search (VAL only)", "",
             "Each variant changes one thing from Run A; early stopping and selection both use val NLL (lower is better).",
             f"Seed noise: the three baseline seeds span {noise:.3f} in val NLL, so smaller gaps are not meaningful." if noise is not None else "",
             "", "| variant | " + " | ".join(keys) + " | best epoch | params |", "|---|" + "---|" * (len(keys) + 2)]
    for r in sorted(rows, key=lambda r: r["val_nll"]):
        lines.append(f"| {r['variant']} | " + " | ".join(f"{r[k]:.3f}" for k in keys) + f" | {r['best_epoch']} | {r['params']:,} |")
    (out / f"{a.out_name}.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
