"""Train the strict-tag Stage 2 ensemble and tune its tag temperature on VAL.

    python -m kinesync.stage2.train_strict --tag_min_z 2.0

Same four members as configs/stage2/final_ensemble.yaml (three seeds of Run A + the d = 128 variant), but trained
on tags with stored z-score >= tag_min_z instead of the release's 1.0 (docs/data_issues.md, section 5). Writes
checkpoints/stage2/strict/*.pt and configs/stage2/strict_ensemble.yaml, including the val-tuned tau_tag.
"""
from __future__ import annotations

import argparse
import collections
import shutil
from pathlib import Path

import numpy as np
import yaml

from kinesync.common import REPO_ROOT, load_shots, load_vocab, load_yaml, videos_in
from kinesync.stage2.dataset import strict_tags
from kinesync.stage2.sample import storyboard
from kinesync.stage2.sweep import apply
from kinesync.stage2.train import load_any, train

MEMBERS = {"seed1234": {}, "seed1": {"seed": 1}, "seed2": {"seed": 2}, "d128": {"model.d": 128, "model.ff": 512}}
TAUS = (1.0, 1.1, 1.2, 1.3, 1.45, 1.6)


def tag_rates(shot_tag_lists, vocab) -> dict:
    """Tags per shot in each group."""
    c, n = collections.Counter(), 0
    for tags in shot_tag_lists:
        n += 1
        for t in tags:
            c[vocab.to_local(t)[0]] += 1
    return {g.name: c[g.name] / max(n, 1) for g in vocab.groups}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag_min_z", type=float, default=2.0)
    ap.add_argument("--device", default=None)
    a = ap.parse_args()
    base = load_yaml(REPO_ROOT / "configs" / "stage2" / "run_A.yaml")
    out = REPO_ROOT / "checkpoints" / "stage2" / "strict"
    out.mkdir(parents=True, exist_ok=True)
    for name, changes in MEMBERS.items():
        cfg = apply(base, {**changes, "tag_min_z": a.tag_min_z})
        cfg["run_name"] = f"strict_{name}"
        res = train(cfg, device=a.device, quiet=True)
        shutil.copy(Path(res["out_dir"]) / "best.pt", out / f"{name}.pt")
        print(f"{name}: best val NLL {res['best']:.3f}", flush=True)

    ens_path = REPO_ROOT / "configs" / "stage2" / "strict_ensemble.yaml"
    doc = {"tag_min_z": a.tag_min_z,
           "members": [f"checkpoints/stage2/strict/{n}.pt" for n in MEMBERS]}
    ens_path.write_text(yaml.safe_dump(doc, sort_keys=False), encoding="utf-8")

    # tau_tag: match sampled tag rates to strict label rates on VAL (as was done for the release's tags)
    vocab = load_vocab()
    model, ck = load_any(ens_path)
    vids = videos_in("val")
    real = tag_rates([strict_tags(s, a.tag_min_z) for v in vids for s in load_shots(v)], vocab)
    best = None
    for tau in TAUS:
        sampled = [sum(s["tags"].values(), []) for v in vids
                   for s in storyboard(model, ck, v, 0, 1234, {"tau_tag": tau}, timing="real")["shots"]]
        got = tag_rates([[vocab.id_of(p) for p in tags] for tags in sampled], vocab)
        err = float(np.mean([abs(np.log(max(got[g], 1e-4) / max(real[g], 1e-4))) for g in real]))
        print(f"tau_tag {tau}: mean |log(sampled / label rate)| {err:.3f}", flush=True)
        if best is None or err < best[1]:
            best = (tau, err)
    doc["sampling"] = {"tau_tag": best[0]}
    doc["note"] = (f"Trained on tags with z >= {a.tag_min_z}. tau_tag {best[0]} matches val tag rates "
                   f"(mean |log ratio| {best[1]:.3f}).")
    ens_path.write_text(yaml.safe_dump(doc, sort_keys=False), encoding="utf-8")
    print(f"label tag rates on val: {', '.join(f'{g} {r:.3f}' for g, r in real.items())}")
    print(f"wrote {ens_path} with tau_tag {best[0]}")


if __name__ == "__main__":
    main()
