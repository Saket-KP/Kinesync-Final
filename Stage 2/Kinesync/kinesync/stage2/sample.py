"""Sample a storyboard for one song and one Stage 1 sample (handoff 4.4).

    python -m kinesync.stage2.sample --ckpt checkpoints/stage2/run_C/best.pt --split test --n_samples 3

Writes stage2/{video_id}_s{k}_storyboard.json. Same seed + same inputs -> identical output.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from kinesync.common import (SCALE_NAMES, SHORT_SHOT_S, KIN_KEYS, data_root, load_features,
                             load_shots, load_yaml, REPO_ROOT, videos_in)
from kinesync.stage1.adapter import load_stage1
from kinesync.stage2.features import KinSource, Song, shot_inputs, song_memory
from kinesync.stage2.train import load_any

DEFAULTS = {"tau_scale": 1.0, "tau_tag": 1.0, "rep_limit": "auto", "rep_penalty": 2.0,
            "rep_min_shots": 5, "calibrate": True}


def logit_corrections(ck, vocab, n_tags: int) -> dict:
    """Training up-weights rare scale classes (CE class weights w_c) and positive tags (BCE
    pos_weight w_j). The optimum of those losses predicts weighted odds:
        scale:  p_w(c) proportional to w_c * p(c)      tag:  odds_w = w_j * odds
    so subtracting log(w) from the logits recovers calibrated probabilities for sampling."""
    st = ck["label_stats"]
    return {"scale": torch.log(torch.tensor(st["scale_weight"], dtype=torch.float32)),
            "tag": torch.log(torch.tensor(st["tag_pos_weight"], dtype=torch.float32))}


def resolve_rep_limit(value, ck) -> float:
    """'auto' = 95th percentile, over train songs, of the share of shots taken by a song's
    most frequent setting (computed at training time and stored in the checkpoint)."""
    if value == "auto":
        est = ck["label_stats"].get("setting_max_share_p95")
        return float(est) if est else 0.6
    return float(value)


@torch.no_grad()
def sample_shots(model, x: dict, n_tags: int, vocab, seed: int, tau_scale=1.0, tau_tag=1.0,
                 rep_limit=0.6, rep_penalty=2.0, rep_min_shots=5, max_len=128, corr: dict | None = None,
                 mem: dict | None = None):
    """Autoregressive sampling over all shots of a song. Returns per-shot dicts with
    scale (int), moving (bool) and tag ids (list[int])."""
    gen = torch.Generator().manual_seed(int(seed))
    N = len(x["start"])
    dur = x["end"] - x["start"]
    t = {k: torch.as_tensor(x[k]) for k in ("clap", "lyric", "has_lyric", "log_dur", "kin", "pos",
                                             "mid_s", "section", "genre")}
    m = {k: torch.as_tensor(v).unsqueeze(0) for k, v in (mem or {}).items()}   # whole song, every step
    p_scale = torch.zeros(N, dtype=torch.long)
    p_moving = torch.zeros(N, dtype=torch.long)
    p_tags = torch.zeros(N, n_tags)
    has_prev = torch.zeros(N)
    setting = vocab.group_by_name.get("setting")
    setting_counts = np.zeros(setting.n if setting else 0)
    out = []
    for i in range(N):
        a = max(0, i - max_len + 1)          # sliding causal window
        b = {k: v[a:i + 1].unsqueeze(0) for k, v in t.items()}
        b.update(prev_scale=p_scale[a:i + 1][None], prev_moving=p_moving[a:i + 1][None],
                 prev_tags=p_tags[a:i + 1][None], has_prev=has_prev[a:i + 1][None], **m)
        o = {k: v[0, -1] for k, v in model(b).items()}

        s_logits = o["scale"] - corr["scale"] if corr else o["scale"]
        scale = int(torch.multinomial(torch.softmax(s_logits / tau_scale, -1), 1, generator=gen))
        moving = int(torch.bernoulli(torch.sigmoid(o["moving"]), generator=gen))
        tags = []
        for g in vocab.groups:
            logits = o["tag_" + g.name] / tau_tag
            if g.mode == "single":
                p = torch.softmax(logits, -1).clone()
                if g.name == "setting" and i >= rep_min_shots:
                    over = setting_counts / i > rep_limit       # repetition control
                    if over.any():
                        p[:g.n][torch.as_tensor(over)] /= rep_penalty
                        p = p / p.sum()
                j = int(torch.multinomial(p, 1, generator=gen))
                if j < g.n:
                    tags.append(g.ids[j])
                    if g.name == "setting":
                        setting_counts[j] += 1
            else:
                if corr:
                    logits = (o["tag_" + g.name] - corr["tag"][g.ids]) / tau_tag
                p = torch.sigmoid(logits)
                on = torch.bernoulli(p, generator=gen).bool()
                idx = [int(j) for j in torch.argsort(p, descending=True) if on[j]][:g.max_picks]
                tags += [g.ids[j] for j in sorted(idx)]
        out.append({"scale": scale, "moving": bool(moving), "tags": tags})
        if i + 1 < N:   # feed what we sampled as the next shot's "previous shot"
            p_scale[i + 1] = scale
            p_moving[i + 1] = moving
            if dur[i] >= SHORT_SHOT_S:                    # short shots carry no tags in training
                p_tags[i + 1, tags] = 1.0
            has_prev[i + 1] = 1.0
    return out


TIMINGS = ("stage1", "real")


def sample_id(k: int, timing: str = "stage1") -> str:
    """Name of one storyboard of a song: s0, s1, ... from Stage 1 sample k, or real0, real1, ... when the
    timing is the real video's (then only the Stage 2 draw differs between them)."""
    return f"real{k}" if timing == "real" else f"s{k}"


def storyboard(model, ck, vid: str, k: int, seed: int, opts: dict | None = None, root=None,
               genre: str | None = None, timing: str = "stage1") -> dict:
    """timing='stage1': shots and kinetics from Stage 1 sample k (the handoff's design).
    timing='real': shots and kinetics copied from the real video of a song in the dataset. Without a
    trained Stage 1 this is the most accurate timing there is; it is NOT generated, and the sheet says so.
    Every real-timing sample uses the same cuts, so each gets its own Stage 2 seed (seed + k)."""
    assert timing in TIMINGS, timing
    opts = {**DEFAULTS, **(opts or {}), **ck.get("sampling_overrides", {})}
    vocab = model.vocab
    song = Song.load(vid, root, genre=genre)
    if timing == "real":
        shots_real = load_shots(vid, root)
        starts = np.array([s["start"] for s in shots_real])
        ends = np.array([s["end"] for s in shots_real])
        kin, mock, seed = KinSource.from_features(load_features(vid, root)), False, seed + k
    else:
        s1 = load_stage1(vid, k, root)
        starts = np.array([s for s, _ in s1["shots"]])
        ends = np.array([e for _, e in s1["shots"]])
        kin, mock = KinSource.from_stage1(s1), bool(s1["mock"])
    x = shot_inputs(song, starts, ends, kin)
    rep_limit = resolve_rep_limit(opts["rep_limit"], ck)
    corr = logit_corrections(ck, vocab, vocab.n_tags) if opts["calibrate"] else None
    shots = sample_shots(model, x, vocab.n_tags, vocab, seed, opts["tau_scale"], opts["tau_tag"],
                         rep_limit, opts["rep_penalty"], opts["rep_min_shots"], ck["config"]["max_len"], corr,
                         song_memory(song))
    genres = ck["genres"]
    rows = []
    for i, s in enumerate(shots):
        rows.append({
            "start": round(float(starts[i]), 4), "end": round(float(ends[i]), 4),
            "section": f"segment {int(x['section'][i])}",
            "scale": SCALE_NAMES[s["scale"]], "moving": s["moving"],
            "tags": vocab.by_group(s["tags"]),
            "lyric": x["first_lyric"][i],
            "kin": {kk: round(float(x["kin_raw"][i, j]), 4) for j, kk in enumerate(KIN_KEYS)},
            "short": bool(ends[i] - starts[i] < SHORT_SHOT_S),
        })
    return {
        "video_id": vid, "stage1_sample": k, "sample_id": sample_id(k, timing), "timing": timing,
        "stage2_seed": int(seed), "vocab_version": vocab.version,
        "genre": genres[song.genre_id] if song.genre_id < len(genres) else "unknown",
        "stage1_mock": mock,
        "stage2_checkpoint": ck.get("path", ""),
        "sampling": {**opts, "rep_limit": rep_limit},
        "shots": rows,
    }


def write_storyboard(sb: dict, root=None) -> Path:
    out = Path(root or data_root()) / "stage2"
    out.mkdir(parents=True, exist_ok=True)
    p = out / f"{sb['video_id']}_{sb.get('sample_id', 's%d' % sb['stage1_sample'])}_storyboard.json"
    with open(p, "w", encoding="utf-8") as f:
        json.dump(sb, f, indent=1, ensure_ascii=False)
    return p


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=str(REPO_ROOT / "configs" / "stage2" / "final_ensemble.yaml"),
                    help="a .pt checkpoint or an ensemble .yaml")
    ap.add_argument("--split", default="test")
    ap.add_argument("--video_id", nargs="*")
    ap.add_argument("--n_samples", type=int, default=3)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--config", default=str(REPO_ROOT / "configs" / "sampling.yaml"))
    a = ap.parse_args()
    opts = load_yaml(a.config) if Path(a.config).exists() else {}
    model, ck = load_any(a.ckpt)
    ck["path"] = str(a.ckpt)
    print(f"rep_limit = {resolve_rep_limit(opts.get('rep_limit', 'auto'), ck):.3f} "
          f"(estimated from train: 95th pct share of a song's most frequent setting)")
    vids = a.video_id or videos_in(a.split)
    n = 0
    for vid in vids:
        for k in range(a.n_samples):
            write_storyboard(storyboard(model, ck, vid, k, a.seed, opts))
            n += 1
    print(f"wrote {n} storyboards to {data_root() / 'stage2'}")


if __name__ == "__main__":
    main()
