"""CLAP context: describe each shot's audio in words, for the panel prompt.

A shot's CLAP vector (the mean of the 1 Hz rows inside it, as in Stage 2) is compared with the CLAP
*text* embeddings of short audio descriptions. That only works with the text tower of the same
checkpoint that made the stored audio vectors: laion/clap-htsat-unfused, identified by testing five
LAION checkpoints (configs/render.yaml, docs/IMPLEMENTATION.md).

Raw cosine similarities are not comparable across phrases (some phrases are close to all music), so
each phrase's similarity is standardised over TRAIN shots (a z-score), the same idea as the tag labels.
Per group: `single` takes the best phrase; `multi` takes up to `max_picks` phrases with z above `min_z`.

Vocals are different. Picking the best z-score among "man / woman / rapper / no vocals" fails: for a
string-heavy song "no vocals" is unusually high for that song, so it won on 73 of 82 shots of a song
that is sung from 28 s on. So singing is detected by the contrast "singing vocals" minus "instrumental"
(raw similarities), with a cut-off learned on TRAIN shots against where kept lyric lines are. Only then
the panel told "singer". Choosing the kind of voice was tried and dropped: both Bruno Mars test songs
came out mostly "female singer", and one got "rapper" on 23 shots, so CLAP cannot tell voices apart
here and a wrong one would be drawn.

Each descriptor has an `audio` text (what CLAP is asked) and a `visual` phrase (what SDXL is told).

    python -m kinesync.render.audio_context        # builds the cache and prints VAL diagnostics
"""
from __future__ import annotations

import collections
import json
from pathlib import Path

import numpy as np

from kinesync.common import data_root, load_features, load_lyric_lines, load_shots, videos_in
from kinesync.stage2.features import _range_mean, clap_bounds

# group -> (mode, max_picks, [(audio description for CLAP, visual phrase for the prompt or None)])
GROUPS = {
    "mood": ("single", 1, [
        ("happy, upbeat and joyful music", "joyful"),
        ("sad and melancholic music", "melancholic"),
        ("romantic, tender love song", "romantic"),
        ("dark, tense and ominous music", "dark and tense"),
        ("aggressive, intense and angry music", "aggressive"),
        ("calm, peaceful and relaxing music", "calm"),
        ("dreamy, atmospheric and ethereal music", "dreamy"),
        ("epic, triumphant and powerful music", "triumphant"),
    ]),
    "energy": ("single", 1, [
        ("slow, quiet and soft music", "stillness"),
        ("mid-tempo groovy music", None),          # the default case: say nothing
        ("fast, loud, high-energy music", "energetic motion"),
    ]),
    # Named as PLAYERS, not instruments: "brass horns" drew heaps of floating trumpets, which reads as
    # clutter rather than a music-video shot. This adds people the tags did not ask for (see prompt.py).
    "instruments": ("multi", 2, [
        ("music with acoustic guitar", "acoustic guitarist"),
        ("music with distorted electric guitar", "electric guitarist"),
        ("piano music", "pianist"),
        ("music with synthesizers and electronic beats", "keyboard player"),
        ("music with heavy drums and bass", "drummer"),
        ("music with orchestral strings", "violinist"),
        ("music with brass horns", "trumpet player"),
    ]),
}
SINGING = ("a song with singing vocals", "instrumental music with no vocals")


def _slug(model_id: str) -> str:
    return model_id.replace("/", "__")


def shot_clap(clap: np.ndarray, starts, ends) -> np.ndarray:
    """Unit-length mean CLAP vector per shot (same rows as Stage 2's `clap` input)."""
    i0, i1 = clap_bounds(np.asarray(starts, float), np.asarray(ends, float), len(clap))
    v, _ = _range_mean(clap, i0, i1)
    return v / np.maximum(np.linalg.norm(v, axis=1, keepdims=True), 1e-9)


def lyric_overlap(vid: str, starts, ends, root=None) -> np.ndarray:
    """True where a shot overlaps a kept lyric line (the only singing signal in the data)."""
    lines = [l for l in load_lyric_lines(vid, root) if l.get("kept")]
    s, e = np.asarray(starts, float), np.asarray(ends, float)
    if not lines:
        return np.zeros(len(s), bool)
    ls = np.array([l["start"] for l in lines])
    le = np.array([l["end"] for l in lines])
    return ((ls[None] < e[:, None]) & (le[None] > s[:, None])).any(1)


def _text_embeddings(model_id: str, revision: str | None, texts: list[str]) -> np.ndarray:
    import torch
    from transformers import ClapModel, ClapProcessor
    # safetensors, not the .bin. transformers refuses to torch.load a .bin below
    # torch 2.6 (CVE-2025-32434) and raises rather than falling back, which breaks
    # CLAP context entirely on torch 2.5. The restriction does not apply to
    # safetensors, and the weights are identical.
    try:
        model = ClapModel.from_pretrained(model_id, revision=revision,
                                          use_safetensors=True).eval()
    except Exception as e:
        raise RuntimeError(
            f"could not load {model_id} as safetensors ({type(e).__name__}: {e}). "
            f"transformers will not load the .bin on torch {torch.__version__} "
            f"(it requires >= 2.6). Upgrade torch, or run with --clap_context off "
            f"to skip the audio-derived prompt content."
        ) from e
    proc = ClapProcessor.from_pretrained(model_id, revision=revision)
    with torch.no_grad():
        out = model.get_text_features(**proc(text=texts, return_tensors="pt", padding=True))
    out = out if torch.is_tensor(out) else out.pooler_output
    return torch.nn.functional.normalize(out, dim=-1).numpy()


def _balanced_cut(score: np.ndarray, truth: np.ndarray) -> float:
    """Cut-off on `score` that best separates truth True/False (max balanced accuracy)."""
    best, cut = -1.0, 0.0
    for c in np.percentile(score, np.arange(5, 96, 1)):
        pred = score > c
        bacc = (pred[truth].mean() + (~pred[~truth]).mean()) / 2
        if bacc > best:
            best, cut = bacc, float(c)
    return cut


class AudioDescriber:
    def __init__(self, cfg: dict, root=None):
        self.cfg = cfg
        self.root = Path(root or data_root())
        self.audio_texts = [a for _, _, items in GROUPS.values() for a, _ in items] + list(SINGING)
        self.cache = self.root / "cache" / f"clap_context_{_slug(cfg['model_id'])}.npz"
        if self.cache.exists():
            z = np.load(self.cache, allow_pickle=False)
            if list(z["texts"]) == self.audio_texts:
                self.T, self.mu, self.sd, self.sing_cut = z["T"], z["mu"], z["sd"], float(z["sing_cut"])
                return
        self._build()

    def _build(self):
        """Text embeddings, per-phrase mean/std of similarity over train shots, and the singing
        cut-off learned against kept lyrics on train shots. Done once (~20 s), then cached."""
        T = _text_embeddings(self.cfg["model_id"], self.cfg.get("revision"), self.audio_texts)
        sims, sung = [], []
        for vid in videos_in("train", self.root):
            shots = load_shots(vid, self.root)
            st, en = [s["start"] for s in shots], [s["end"] for s in shots]
            sims.append(shot_clap(load_features(vid, self.root)["clap"], st, en) @ T.T)
            sung.append(lyric_overlap(vid, st, en, self.root))
        sims, sung = np.concatenate(sims), np.concatenate(sung)
        self.T, self.mu, self.sd = T, sims.mean(0), sims.std(0) + 1e-9
        self.sing_cut = _balanced_cut(sims[:, -2] - sims[:, -1], sung)
        self.cache.parent.mkdir(parents=True, exist_ok=True)
        np.savez(self.cache, T=T, mu=self.mu, sd=self.sd, sing_cut=self.sing_cut,
                 texts=np.array(self.audio_texts))

    def scores(self, clap: np.ndarray, starts, ends):
        """(z-scores of every phrase, singing contrast) per shot."""
        sim = shot_clap(clap, starts, ends) @ self.T.T
        return (sim - self.mu) / self.sd, sim[:, -2] - sim[:, -1]

    def _pick_players(self, zs, k, used, n_done, rng):
        """Players for one shot. CLAP mostly describes the whole song, so always taking the top-z players gave
        the same one on 45% of the shots it filled ("pianist" 150 times in one song). Instead: every player
        the audio supports (z >= min_z) is a candidate, weighted by exp(z / temperature); a player already on
        more than `player_rep_limit` of the song's shots so far has its weight divided by
        `player_rep_penalty`; up to k are drawn without replacement. The generator is seeded from the song,
        so a song always gets the same players."""
        if not self.cfg.get("player_sampling", True):
            return [int(j) for j in np.argsort(-zs)[:k] if zs[j] >= self.cfg.get("min_z", 0.5)]
        k = self.cfg.get("players_per_shot", k)
        floor = self.cfg.get("player_min_z", self.cfg.get("min_z", 0.5))
        cand = [j for j in np.argsort(-zs) if zs[j] >= floor]
        if len(cand) <= 1:
            return [int(j) for j in cand[:k]]
        w = np.exp(zs[cand] / self.cfg.get("player_temperature", 1.0))
        if n_done >= 5:
            over = np.array([used[j] / n_done > self.cfg.get("player_rep_limit", 0.25) for j in cand])
            w = np.where(over, w / self.cfg.get("player_rep_penalty", 3.0), w)
        take = rng.choice(len(cand), size=min(k, len(cand)), replace=False, p=w / w.sum())
        return sorted((int(cand[t]) for t in take), key=lambda j: -zs[j])

    def describe(self, vid: str, starts, ends) -> list[dict]:
        """Per shot: {"mood": [...], "energy": [...], "instruments": [...], "vocals": [...]} of visual
        phrases, plus "heard": the audio descriptions that won (for the sheet)."""
        import hashlib
        z, sing = self.scores(load_features(vid, self.root)["clap"], starts, ends)
        min_z = self.cfg.get("min_z", 0.5)
        rng = np.random.default_rng(int.from_bytes(hashlib.sha256(vid.encode()).digest()[:4], "little"))
        used = collections.Counter()
        out = []
        for row, sung in zip(z, sing > self.sing_cut):
            d, heard, col = {}, [], 0
            for g, (mode, k, items) in GROUPS.items():
                zs = row[col:col + len(items)]
                col += len(items)
                if mode == "single":
                    picks = [int(np.argmax(zs))]
                elif g == "instruments":
                    picks = self._pick_players(zs, k, used, len(out), rng)
                    used.update(picks)
                else:
                    picks = [int(j) for j in np.argsort(-zs)[:k] if zs[j] >= min_z]
                d[g] = [items[j][1] for j in picks if items[j][1]]
                heard += [items[j][0] for j in picks]
            d["vocals"] = ["singer"] if sung else []
            heard.append(SINGING[0] if sung else SINGING[1])
            d["heard"] = heard
            out.append(d)
        return out


def diagnostics(cfg: dict, root=None) -> dict:
    """Does the singing detector agree with kept lyrics on VAL shots (never used for fitting)?"""
    from sklearn.metrics import roc_auc_score
    root = Path(root or data_root())
    desc = AudioDescriber(cfg, root)
    score, sung = [], []
    for vid in videos_in("val", root):
        shots = load_shots(vid, root)
        st, en = [s["start"] for s in shots], [s["end"] for s in shots]
        score.append(desc.scores(load_features(vid, root)["clap"], st, en)[1])
        sung.append(lyric_overlap(vid, st, en, root))
    score, sung = np.concatenate(score), np.concatenate(sung)
    pred = score > desc.sing_cut
    return {"val_shots": int(len(sung)), "share_with_kept_lyric": round(float(sung.mean()), 3),
            "singing_auroc": round(float(roc_auc_score(sung, score)), 3),
            "singing_detected_when_lyric": round(float(pred[sung].mean()), 3),
            "singing_detected_when_no_lyric": round(float(pred[~sung].mean()), 3),
            "cut_learned_on_train": round(desc.sing_cut, 4)}


if __name__ == "__main__":
    import yaml
    from kinesync.common import REPO_ROOT
    cfg = yaml.safe_load(open(REPO_ROOT / "configs" / "render.yaml", encoding="utf-8"))["clap"]
    print(json.dumps(diagnostics(cfg), indent=1))
