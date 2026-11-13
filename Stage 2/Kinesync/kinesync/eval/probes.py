"""Probes for the research question: do audio and lyrics predict shot content beyond genre?
(handoff section 5)

    python -m kinesync.eval.probes

Per-shot features -> logistic regression and a 1-hidden-layer MLP. Train on train, pick C /
hidden size on val, report on test with 95% CIs from 1,000 artist bootstrap resamples.

Feature sets: genre | genre+rhythm | genre+CLAP | genre+lyric | all (the handoff's five), plus
genre+CLAP+lyric. "rhythm" includes the REAL video kinetics (flow, colour drift, brightness,
saturation), which are measured on the same frames the labels come from; so the clean test of
"audio and words beyond genre" is genre+CLAP+lyric vs genre (and all vs genre+rhythm).
Controls: shuffle (CLAP + lyric permuted among shots of the same song) and shift (CLAP taken
from a window moved +-5 s), each retrained from scratch.
The oracle row (CLIP image embedding of the keyframe) needs video frames, which this release
does not have, so it is reported as unavailable.

To keep ~550 fits fast, CLAP (512-d) and lyric (384-d) are reduced to `--pca` dims with PCA fit
on train shots only. Writes results/probes.json and results/probes.md.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from kinesync.common import N_SECTIONS, REPO_ROOT, data_root, genre_list, load_vocab, videos_in
from kinesync.eval.metrics import binary_f1, bootstrap, macro_f1, mean_ap
from kinesync.stage2.features import shot_inputs
from kinesync.stage2.model import group_targets
from kinesync.stage2.train import load_records

SETS = {
    "genre": ["genre"],
    "genre+rhythm": ["genre", "rhythm"],
    "genre+clap": ["genre", "clap"],
    "genre+lyric": ["genre", "lyric"],
    "genre+clap+lyric": ["genre", "clap", "lyric"],
    "all": ["genre", "rhythm", "clap", "lyric"],
}
CONTROLS = {   # name -> (feature set, which variant of clap / lyric to use)
    "shuffle: genre+clap": ("genre+clap", "shuf"),
    "shuffle: genre+lyric": ("genre+lyric", "shuf"),
    "shuffle: all": ("all", "shuf"),
    "shift +5s: genre+clap": ("genre+clap", "shift+5"),
    "shift -5s: genre+clap": ("genre+clap", "shift-5"),
    "shift +5s: all": ("all", "shift+5"),
    "shift -5s: all": ("all", "shift-5"),
}


# --------------------------------------------------------------------------- features

def split_features(split: str, vocab, n_genres: int, seed: int = 0) -> dict:
    recs = load_records(videos_in(split), vocab, "real")
    rng = np.random.default_rng(seed)
    blocks = {k: [] for k in ("genre", "rhythm", "clap", "lyric", "clap_shuf", "lyric_shuf",
                              "clap_shift+5", "clap_shift-5")}
    for r in recs:
        x = shot_inputs(r.song, r.starts, r.ends, r.kin)
        n = len(r.starts)
        g = np.zeros((n, n_genres + 1), np.float32)
        g[:, r.song.genre_id] = 1
        sec = np.eye(N_SECTIONS, dtype=np.float32)[x["section"]]
        lyr = np.concatenate([x["lyric"], x["has_lyric"][:, None]], 1)
        perm = rng.permutation(n)
        blocks["genre"].append(g)
        blocks["rhythm"].append(np.concatenate([x["log_dur"][:, None], x["kin"], sec], 1))
        blocks["clap"].append(x["clap"])
        blocks["lyric"].append(lyr)
        blocks["clap_shuf"].append(x["clap"][perm])
        blocks["lyric_shuf"].append(lyr[perm])
        for s in (5, -5):
            blocks[f"clap_shift{s:+d}"].append(shot_inputs(r.song, r.starts, r.ends, r.kin, clap_shift=s)["clap"])
    out = {k: np.concatenate(v).astype(np.float32) for k, v in blocks.items()}
    out["scale"] = np.concatenate([r.scale for r in recs])
    out["moving"] = np.concatenate([r.moving for r in recs]).astype(int)
    out["tags"] = np.concatenate([r.tags for r in recs])
    out["tag_mask"] = np.concatenate([r.tag_mask for r in recs]) > 0
    out["artist"] = np.concatenate([[r.artist] * len(r.scale) for r in recs])
    return out


def prepare(data: dict, pca_dim: int) -> dict:
    """PCA + standardisation, fitted on train only, applied to every split and variant."""
    out = {s: {} for s in data}
    for base in ("clap", "lyric"):
        pca = PCA(pca_dim, random_state=0).fit(data["train"][base])
        for s in data:
            for var in [base] + [k for k in data[s] if k.startswith(base + "_")]:
                out[s][var] = pca.transform(data[s][var])
    for s in data:
        out[s]["genre"] = data[s]["genre"]
        out[s]["rhythm"] = data[s]["rhythm"]
    for var in out["train"]:
        if var == "genre":
            continue
        sc = StandardScaler().fit(out["train"][var.split("_")[0] if "_" in var else var])
        for s in data:
            out[s][var] = sc.transform(out[s][var]).astype(np.float32)
    return out


def design(feats: dict, set_name: str, variant: str | None) -> np.ndarray:
    cols = []
    for b in SETS[set_name]:
        key = b
        if variant and b in ("clap", "lyric"):
            if variant == "shuf":
                key = f"{b}_shuf"
            elif b == "clap":
                key = f"clap_shift{variant[5:]}"
        cols.append(feats[key])
    return np.concatenate(cols, 1)


# --------------------------------------------------------------------------- targets

def targets(d: dict, vocab) -> list[dict]:
    """Each target: name, kind, y, row mask, and the metric family."""
    t = [{"name": "scale", "kind": "multiclass", "k": 5, "y": d["scale"], "rows": slice(None)},
         {"name": "moving", "kind": "binary", "y": d["moving"], "rows": slice(None)}]
    tm = d["tag_mask"]
    tags_t = torch.as_tensor(d["tags"])
    for g in vocab.groups:
        if g.mode == "single":
            t.append({"name": g.name, "kind": "single", "k": g.n + 1, "group": g,
                      "y": group_targets(tags_t, g).numpy()[tm], "rows": tm})
        else:
            t.append({"name": g.name, "kind": "multi", "group": g,
                      "y": d["tags"][tm][:, g.ids].astype(int), "rows": tm})
    return t


def score_fn(t: dict, probs: np.ndarray):
    """Closure w -> metric for one target. probs: (n,k) class probs, (n,) for binary,
    (n, n_phrases) for multi."""
    if t["kind"] == "multiclass":
        pred = probs.argmax(1)
        return lambda w: macro_f1(t["y"], pred, w, t["k"])
    if t["kind"] == "binary":
        pred = (probs >= 0.5).astype(int)
        return lambda w: binary_f1(t["y"], pred, w)
    g = t["group"]
    if t["kind"] == "single":
        y = np.eye(g.n + 1, dtype=int)[t["y"]][:, :g.n]
        return mean_ap(y, probs[:, :g.n])
    return mean_ap(t["y"], probs)


# --------------------------------------------------------------------------- models

def fit_lr(Xtr, ytr, kind, C):
    """One multinomial model, or one binary model per phrase for multi groups."""
    if kind == "multi":
        return [LogisticRegression(C=C, max_iter=300).fit(Xtr, ytr[:, j]) for j in range(ytr.shape[1])]
    # scale is scored by macro-F1: balanced class weights, as in the Stage 2 loss
    cw = "balanced" if kind == "multiclass" else None
    return LogisticRegression(C=C, max_iter=300, class_weight=cw).fit(Xtr, ytr)


def predict_lr(m, X, kind, k=1):
    if kind == "multi":
        return np.stack([mm.predict_proba(X)[:, 1] for mm in m], 1)
    p = m.predict_proba(X)
    if kind == "binary":
        return p[:, 1]
    full = np.zeros((len(X), k))                 # classes missing from train get probability 0
    full[:, m.classes_] = p
    return full


class MLP(nn.Module):
    def __init__(self, d_in, hidden, out):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(d_in, hidden), nn.ReLU(), nn.Dropout(0.2), nn.Linear(hidden, out))

    def forward(self, x):
        return self.net(x)


def fit_mlp(Xtr, ytr, Xva, yva, kind, k, hidden, seed=0, epochs=30, patience=4):
    """One small MLP per target; early stopping on validation loss."""
    torch.manual_seed(seed)
    out = {"multiclass": k, "single": k, "binary": 1, "multi": ytr.shape[1] if ytr.ndim > 1 else 1}[kind]
    net = MLP(Xtr.shape[1], hidden, out)
    opt = torch.optim.Adam(net.parameters(), lr=1e-3, weight_decay=1e-4)
    Xt, Xv = torch.as_tensor(Xtr), torch.as_tensor(Xva)
    if kind in ("multiclass", "single"):
        yt, yv = torch.as_tensor(ytr).long(), torch.as_tensor(yva).long()
        cw = None
        if kind == "multiclass":   # balanced class weights for macro-F1 (scale)
            cnt = torch.bincount(yt, minlength=k).float()
            cw = cnt.sum() / (k * cnt.clamp(min=1))
        lossf = nn.CrossEntropyLoss(weight=cw)
    else:
        yt = torch.as_tensor(ytr).float().reshape(len(ytr), -1)
        yv = torch.as_tensor(yva).float().reshape(len(yva), -1)
        lossf = nn.BCEWithLogitsLoss()
    g = torch.Generator().manual_seed(seed)
    best, best_state, bad = float("inf"), None, 0
    for _ in range(epochs):
        net.train()
        for idx in torch.randperm(len(Xt), generator=g).split(512):
            opt.zero_grad()
            lossf(net(Xt[idx]), yt[idx]).backward()
            opt.step()
        net.eval()
        with torch.no_grad():
            vl = float(lossf(net(Xv), yv))
        if vl < best - 1e-4:
            best, best_state, bad = vl, {k_: v.clone() for k_, v in net.state_dict().items()}, 0
        else:
            bad += 1
            if bad >= patience:
                break
    net.load_state_dict(best_state)
    net.eval()
    return net


def predict_mlp(net, X, kind):
    with torch.no_grad():
        z = net(torch.as_tensor(X))
    if kind in ("multiclass", "single"):
        return torch.softmax(z, -1).numpy()
    p = torch.sigmoid(z).numpy()
    return p[:, 0] if kind == "binary" else p


# --------------------------------------------------------------------------- main

def run_probe(feats, tg, set_name, variant, Cs, hiddens, log):
    """Fit LR and MLP for every target; select on val; return test probabilities."""
    X = {s: design(feats[s], set_name, variant) for s in feats}
    res = {}
    for ti, t_tr in enumerate(tg["train"]):
        t_va, t_te = tg["val"][ti], tg["test"][ti]
        rows = {s: tg[s][ti]["rows"] for s in tg}
        Xtr, Xva, Xte = X["train"][rows["train"]], X["val"][rows["val"]], X["test"][rows["test"]]
        best = None
        for C in Cs:
            m = fit_lr(Xtr, t_tr["y"], t_tr["kind"], C)
            v = score_fn(t_va, predict_lr(m, Xva, t_tr["kind"], t_tr.get("k", 1)))(np.ones(len(Xva)))
            if best is None or v > best[0]:
                best = (v, C, m)
        res[("lr", t_tr["name"])] = (predict_lr(best[2], Xte, t_tr["kind"], t_tr.get("k", 1)),
                                     {"C": best[1], "val": best[0]})
        best = None
        for h in hiddens:
            net = fit_mlp(Xtr, t_tr["y"], Xva, t_va["y"], t_tr["kind"], t_tr.get("k", 1), h)
            v = score_fn(t_va, predict_mlp(net, Xva, t_tr["kind"]))(np.ones(len(Xva)))
            if best is None or v > best[0]:
                best = (v, h, net)
        res[("mlp", t_tr["name"])] = (predict_mlp(best[2], Xte, t_tr["kind"]), {"hidden": best[1], "val": best[0]})
    log(f"  {set_name}{' [' + variant + ']' if variant else ''} done")
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pca", type=int, default=64)
    ap.add_argument("--n_boot", type=int, default=1000)
    ap.add_argument("--C", type=float, nargs="+", default=[0.01, 0.1, 1.0])
    ap.add_argument("--hidden", type=int, nargs="+", default=[64, 256])
    ap.add_argument("--out", default=str(REPO_ROOT / "results"))
    a = ap.parse_args()
    t0 = time.time()
    log = lambda *x: print(f"[{time.time() - t0:6.0f}s]", *x, flush=True)
    vocab = load_vocab()
    n_genres = len(genre_list())
    cache = data_root() / "cache" / "probe_features.npz"
    if cache.exists():
        z = np.load(cache, allow_pickle=True)
        raw = z["raw"].item()
    else:
        raw = {s: split_features(s, vocab, n_genres) for s in ("train", "val", "test")}
        cache.parent.mkdir(parents=True, exist_ok=True)
        np.savez(cache, raw=np.array(raw, dtype=object))
    log("features:", {s: len(raw[s]["scale"]) for s in raw}, "shots")
    feats = prepare(raw, a.pca)
    tg = {s: targets(raw[s], vocab) for s in raw}
    names = [t["name"] for t in tg["test"]]
    metric_of = {"scale": "macro-F1", "moving": "F1"}

    runs = {s: (s, None) for s in SETS}
    runs.update(CONTROLS)
    preds, chosen = {}, {}
    for rname, (set_name, variant) in runs.items():
        r = run_probe(feats, tg, set_name, variant, a.C, a.hidden, log)
        for (model, tname), (p, info) in r.items():
            preds[(rname, model, tname)] = p
            chosen[f"{rname}|{model}|{tname}"] = info

    # bootstrap: every (probe, model, target) plus paired differences
    test_t = {t["name"]: t for t in tg["test"]}
    art = raw["test"]["artist"]
    art_of_t = {n: art[test_t[n]["rows"]] for n in names}
    fns, arts = {}, {}
    for (rname, model, tname), p in preds.items():
        key = f"{rname}|{model}|{tname}"
        fns[key] = score_fn(test_t[tname], p)
        arts[key] = art_of_t[tname]
    diffs = [("all", "genre"), ("genre+clap+lyric", "genre"), ("all", "genre+rhythm"),
             ("genre+clap", "genre"), ("genre+lyric", "genre"), ("genre+rhythm", "genre")]
    diffs += [(f"shuffle: {s}", s) for s in ("genre+clap", "genre+lyric", "all")]
    diffs += [(f"shift {d}: {s}", s) for s in ("genre+clap", "all") for d in ("+5s", "-5s")]
    for model in ("lr", "mlp"):
        for x, y in diffs:
            for tname in names:
                key = f"diff {x} - {y}|{model}|{tname}"
                fx, fy = fns[f"{x}|{model}|{tname}"], fns[f"{y}|{model}|{tname}"]
                fns[key] = (lambda f1, f2: lambda w: f1(w) - f2(w))(fx, fy)
                arts[key] = art_of_t[tname]
    log(f"bootstrapping {len(fns)} metrics x {a.n_boot}")
    res = bootstrap(fns, arts, n=a.n_boot, seed=0)

    # ------------------------------------------------------------- report
    def cell(k):
        v = res[k]
        return f"{v[0]:.3f} [{v[1]:.3f}, {v[2]:.3f}]"

    def beats(x, y, model, tname):
        return res[f"diff {x} - {y}|{model}|{tname}"][1] > 0

    headline = []
    checks = [("all", "genre", "handoff test"),
              ("genre+clap+lyric", "genre", "audio + words only, no video kinetics"),
              ("all", "genre+rhythm", "audio + words on top of timing and kinetics")]
    for model in ("lr", "mlp"):
        for x, y, why in checks:
            no = [t for t in names if not beats(x, y, model, t)]
            yes = [t for t in names if beats(x, y, model, t)]
            head = f"{model.upper()}, {x} vs {y} ({why})"
            if no:
                headline.append(f"- **{head}: does NOT beat {y} beyond its 95% CI for: {', '.join(no)}.**"
                                + (f" Beats it for: {', '.join(yes)}." if yes else ""))
            else:
                headline.append(f"- {head}: beats {y} (CI of the difference above 0) for every target.")
    lines = ["# Probes: do audio and lyrics predict shot content beyond genre?", "", "## Headline", ""]
    lines += headline
    lines += ["", "> Labels are automatic (CLIP-based) and not yet checked against hand labels; "
              "`moving` uses a placeholder threshold (~51% positive), so its scores say little.",
              "> \"rhythm\" includes REAL video kinetics. `flow` is optical flow from the same frames the "
              "`moving` label is computed from, and brightness / saturation track lighting and style tags, so "
              "rows with rhythm partly measure video-to-label leakage, not audio. Read genre+clap+lyric vs genre "
              "for the audio question.",
              "> Oracle ceiling (CLIP image embedding of the keyframe): **not available**, no video frames "
              "in data release v2.", "",
              f"Test split: {len(art)} shots, {len(np.unique(art))} artists. Train on train, C and hidden size "
              f"chosen on val, scores on test. 95% CI from {a.n_boot} artist bootstrap resamples. CLAP and "
              f"lyric reduced to {a.pca} dims by PCA fit on train. Scale: macro-F1; moving: F1; tag groups: mAP "
              f"(shots >= 0.5 s only).", ""]
    for model in ("lr", "mlp"):
        lines += [f"## {'Logistic regression' if model == 'lr' else '1-hidden-layer MLP'}", "",
                  "| features | " + " | ".join(f"{n} ({metric_of.get(n, 'mAP')})" for n in names) + " |",
                  "|---|" + "---|" * len(names)]
        for rname in list(SETS) + ["oracle (CLIP keyframe)"] + list(CONTROLS):
            if rname.startswith("oracle"):
                lines.append(f"| {rname} | " + " | ".join("n/a" for _ in names) + " |")
                continue
            lines.append(f"| {rname} | " + " | ".join(cell(f"{rname}|{model}|{t}") for t in names) + " |")
        lines += ["", "Paired differences (95% CI; above 0 = first set is better):", "",
                  "| difference | " + " | ".join(names) + " |", "|---|" + "---|" * len(names)]
        for x, y in diffs:
            lines.append(f"| {x} - {y} | " + " | ".join(cell(f"diff {x} - {y}|{model}|{t}") for t in names) + " |")
        lines.append("")
    Path(a.out).mkdir(parents=True, exist_ok=True)
    with open(Path(a.out) / "probes.md", "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    with open(Path(a.out) / "probes.json", "w") as f:
        json.dump({"metrics": res, "selected_hparams": chosen, "pca_dim": a.pca, "n_boot": a.n_boot,
                   "oracle": "unavailable (no video frames in release v2)",
                   "n_test_shots": int(len(art)), "n_test_artists": int(len(np.unique(art)))}, f, indent=1)
    log("wrote", Path(a.out) / "probes.md")
    print("\n".join(lines[:12]))


if __name__ == "__main__":
    main()
