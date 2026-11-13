"""Shared helpers: data paths, splits, norm stats, tag vocabulary, per-video loaders.

Time grids in data release v2 (see docs/stage2_preflight.md, D1):
  audio features 10 fps, video features 2 fps, CLAP 1 Hz, everything else float seconds.
"""
from __future__ import annotations

import json
import os
import random
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent

AUDIO_FPS = 10   # onset, flux, chroma, beat_phase, section
VIDEO_FPS = 2    # cut, flow, color_drift, brightness, saturation, valid
CLAP_HZ = 1      # clap rows
KIN_KEYS = ("flow", "color_drift", "brightness", "saturation")
SCALE_NAMES = ("long shot", "full shot", "medium shot", "close-up", "extreme close-up")
N_SECTIONS = 8
SHORT_SHOT_S = 0.5  # shots under this have no tags in the labels


def data_root() -> Path:
    """Root of the data release. Override with KINESYNC_ROOT (tests point it at mock data)."""
    return Path(os.environ.get("KINESYNC_ROOT", REPO_ROOT))


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
    except ImportError:
        pass


def load_yaml(path) -> dict:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


# --------------------------------------------------------------------------- splits / stats

@lru_cache(maxsize=4)
def _splits(root: str) -> pd.DataFrame:
    df = pd.read_csv(Path(root) / "splits.csv", dtype={"video_id": str})
    return df.set_index("video_id", drop=False)


def load_splits(root: Path | None = None) -> pd.DataFrame:
    return _splits(str(root or data_root()))


def videos_in(split: str, root: Path | None = None) -> list[str]:
    df = load_splits(root)
    return df.index[df["split"] == split].tolist()


def genre_list(root: Path | None = None) -> list[str]:
    """Sorted genre names; the id of a genre is its position. id len(list) = unknown."""
    return sorted(load_splits(root)["genre"].unique().tolist())


@lru_cache(maxsize=4)
def _norm(root: str) -> dict:
    with open(Path(root) / "stats" / "norm_train.json", encoding="utf-8") as f:
        return json.load(f)


def load_norm(root: Path | None = None) -> dict:
    """Train-only mean/std per channel. Never recomputed here."""
    return _norm(str(root or data_root()))


def kin_mean_std(root: Path | None = None) -> tuple[np.ndarray, np.ndarray]:
    n = load_norm(root)
    mean = np.array([n[k]["mean"] for k in KIN_KEYS], dtype=np.float32)
    std = np.array([n[k]["std"] for k in KIN_KEYS], dtype=np.float32)
    return mean, std


# --------------------------------------------------------------------------- vocabulary

@dataclass
class TagGroup:
    name: str            # lower-case group name, e.g. "setting"
    mode: str            # "single" (softmax over phrases + none) or "multi" (sigmoid per phrase)
    max_picks: int
    ids: list[int]       # global tag ids, in vocab order
    phrases: list[str]

    @property
    def n(self) -> int:
        return len(self.ids)


class Vocab:
    """Tag vocabulary. Global tag id = position in the flat list in vocab/tags_v1.json.

    The release file has no mode / max_picks, so they come from configs/vocab_groups.yaml
    (preflight D3). Groups keep the order in which they first appear in the file.
    """

    def __init__(self, vocab_path: Path, groups_cfg: dict):
        with open(vocab_path, encoding="utf-8") as f:
            raw = json.load(f)
        self.version = raw.get("vocab_version", raw.get("version"))
        tags = sorted(raw["tags"], key=lambda t: t["id"])
        assert [t["id"] for t in tags] == list(range(len(tags))), "tag ids must be 0..n-1"
        self.phrases = [t["phrase"] for t in tags]
        self.tag_group = [t["group"].lower() for t in tags]
        order = list(dict.fromkeys(self.tag_group))
        self.groups: list[TagGroup] = []
        for g in order:
            cfg = groups_cfg[g]
            ids = [i for i, tg in enumerate(self.tag_group) if tg == g]
            self.groups.append(TagGroup(g, cfg["mode"], int(cfg["max_picks"]), ids,
                                        [self.phrases[i] for i in ids]))
        self._phrase_to_id = {p: i for i, p in enumerate(self.phrases)}
        self._local = {}
        for g in self.groups:
            for j, i in enumerate(g.ids):
                self._local[i] = (g.name, j)
        self.group_by_name = {g.name: g for g in self.groups}

    @property
    def n_tags(self) -> int:
        return len(self.phrases)

    # global id <-> (group, local index) <-> phrase
    def to_local(self, tag_id: int) -> tuple[str, int]:
        return self._local[tag_id]

    def to_global(self, group: str, local: int) -> int:
        return self.group_by_name[group].ids[local]

    def phrase(self, tag_id: int) -> str:
        return self.phrases[tag_id]

    def id_of(self, phrase: str) -> int:
        return self._phrase_to_id[phrase]

    def multi_hot(self, tag_ids) -> np.ndarray:
        v = np.zeros(self.n_tags, dtype=np.float32)
        v[list(tag_ids)] = 1.0
        return v

    def by_group(self, tag_ids) -> dict[str, list[str]]:
        out = {g.name: [] for g in self.groups}
        for i in tag_ids:
            g, _ = self.to_local(int(i))
            out[g].append(self.phrases[int(i)])
        return out


def load_vocab(root: Path | None = None) -> Vocab:
    root = root or data_root()
    cfg = load_yaml(REPO_ROOT / "configs" / "vocab_groups.yaml")
    return Vocab(Path(root) / "vocab" / f"{cfg['vocab_version']}.json", cfg["groups"])


# --------------------------------------------------------------------------- per-video loaders

def load_features(vid: str, root: Path | None = None) -> dict[str, np.ndarray]:
    with np.load(Path(root or data_root()) / "features" / f"{vid}.npz") as d:
        return {k: d[k] for k in d.files}


def load_shots(vid: str, root: Path | None = None) -> list[dict]:
    with open(Path(root or data_root()) / "shots" / f"{vid}_shots.json", encoding="utf-8") as f:
        return json.load(f)["shots"]


def load_lyric_lines(vid: str, root: Path | None = None) -> list[dict]:
    """All lines, in file order (index matches the lyric-embedding cache). Use l['kept']."""
    p = Path(root or data_root()) / "lyrics" / f"{vid}_lyrics.json"
    if not p.exists():
        return []
    with open(p, encoding="utf-8") as f:
        return json.load(f).get("lines", [])
