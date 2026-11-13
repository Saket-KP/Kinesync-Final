"""Stage 2 dataset (handoff 4.1). One item = one chunk of up to `max_len` consecutive shots of a song.

Spans always come from the real shot labels (we need their targets). The kinetics source is:
  real    mean of the real 2 fps video kinetics over each span
  stage1  mean of a Stage 1 sample's 25 fps kinetics over the REAL spans (Run B scoring;
          handoff 4.5 says to resample Stage 1 kinetics onto real spans)
Training augmentation (Run C): `kin_noise` adds N(0, sigma) in normalised units to `kin`, and
`jitter_frames` moves every internal shot boundary by up to +-N frames at 25 fps.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch.utils.data import Dataset

from kinesync.common import SHORT_SHOT_S, Vocab, load_features, load_shots
from kinesync.stage1.adapter import load_stage1
from kinesync.stage2.features import KinSource, Song, shot_inputs, song_memory

JITTER_FPS = 25
FLOAT_KEYS = ("clap", "lyric", "has_lyric", "log_dur", "kin", "pos", "mid_s")
INT_KEYS = ("section", "genre")


@dataclass
class SongRecord:
    song: Song
    kin: KinSource
    starts: np.ndarray
    ends: np.ndarray
    scale: np.ndarray       # (N,) int
    moving: np.ndarray      # (N,) float 0/1
    tags: np.ndarray        # (N, n_tags) multi-hot
    tag_mask: np.ndarray    # (N,) 1 if the shot is long enough to carry tag labels
    artist: str = ""


def strict_tags(shot: dict, min_z: float | None) -> list[int]:
    """Tags whose stored z-score (tag_scores) is at least min_z. The release keeps a tag at z >= 1.0, which
    about 16% of all shots clear for any phrase (docs/data_issues.md, section 5); a higher cut keeps only
    tags that stand out more. None keeps the release's tags unchanged."""
    if min_z is None:
        return list(shot["tags"])
    return [t for t, z in zip(shot["tags"], shot.get("tag_scores", [])) if z >= min_z]


def load_record(vid: str, vocab: Vocab, kin_source: str = "real", stage1_sample: int = 0,
                root=None, artist: str = "", tag_min_z: float | None = None) -> SongRecord:
    feat = load_features(vid, root)
    song = Song.load(vid, root, feat=feat)
    shots = load_shots(vid, root)
    starts = np.array([s["start"] for s in shots], np.float64)
    ends = np.array([s["end"] for s in shots], np.float64)
    if kin_source == "real":
        kin = KinSource.from_features(feat)
    elif kin_source == "stage1":
        kin = KinSource.from_stage1(load_stage1(vid, stage1_sample, root))
    else:
        raise ValueError(kin_source)
    return SongRecord(
        song=song, kin=kin, starts=starts, ends=ends,
        scale=np.array([s["scale"] for s in shots], np.int64),
        moving=np.array([s["moving"] for s in shots], np.float32),
        tags=np.stack([vocab.multi_hot(strict_tags(s, tag_min_z)) for s in shots]) if shots
        else np.zeros((0, vocab.n_tags), np.float32),
        tag_mask=((ends - starts) >= SHORT_SHOT_S).astype(np.float32),
        artist=artist,
    )


def prev_attributes(scale, moving, tags, tag_mask):
    """Shift attributes right by one shot: position i gets shot i-1. Short shots carry no tag
    labels, so their tags are fed as zeros (sampling does the same)."""
    n = len(scale)
    p_scale = np.zeros(n, np.int64)
    p_moving = np.zeros(n, np.int64)
    p_tags = np.zeros((n, tags.shape[1]), np.float32)
    has_prev = np.zeros(n, np.float32)
    if n > 1:
        p_scale[1:] = scale[:-1]
        p_moving[1:] = moving[:-1].astype(np.int64)
        p_tags[1:] = tags[:-1] * tag_mask[:-1, None]
        has_prev[1:] = 1.0
    return p_scale, p_moving, p_tags, has_prev


class ShotDataset(Dataset):
    def __init__(self, records: list[SongRecord], max_len: int = 128, train: bool = False,
                 kin_noise: float = 0.0, jitter_frames: int = 0, seed: int = 0):
        self.records = records
        self.max_len = max_len
        self.train = train
        self.kin_noise = kin_noise
        self.jitter_frames = jitter_frames
        self.rng = np.random.default_rng(seed)
        # consecutive chunks of max_len shots
        self.index = [(r, a, min(a + max_len, len(rec.starts)))
                      for r, rec in enumerate(records)
                      for a in range(0, len(rec.starts), max_len)]

    def __len__(self):
        return len(self.index)

    def _spans(self, rec: SongRecord):
        starts, ends = rec.starts, rec.ends
        if self.train and self.jitter_frames > 0 and len(starts) > 1:
            cuts = starts[1:] + self.rng.integers(-self.jitter_frames, self.jitter_frames + 1,
                                                  len(starts) - 1) / JITTER_FPS
            # keep every shot at least one frame long and the cuts in order
            edges = np.concatenate([[starts[0]], cuts, [ends[-1]]])
            for i in range(1, len(edges)):
                edges[i] = max(edges[i], edges[i - 1] + 1.0 / JITTER_FPS)
            edges[-1] = ends[-1]
            edges[:-1] = np.minimum(edges[:-1], edges[-1] - 1e-3)
            starts, ends = edges[:-1], edges[1:]
        return starts, ends

    def __getitem__(self, i):
        r, a, b = self.index[i]
        rec = self.records[r]
        starts, ends = self._spans(rec)
        x = shot_inputs(rec.song, starts[a:b], ends[a:b], rec.kin)
        if self.train and self.kin_noise > 0:
            x["kin"] = x["kin"] + self.rng.normal(0, self.kin_noise, x["kin"].shape).astype(np.float32)
        p_scale, p_moving, p_tags, has_prev = prev_attributes(
            rec.scale, rec.moving, rec.tags, rec.tag_mask)
        item = {k: x[k] for k in FLOAT_KEYS + INT_KEYS}
        item.update(
            prev_scale=p_scale[a:b], prev_moving=p_moving[a:b], prev_tags=p_tags[a:b],
            has_prev=has_prev[a:b],
            scale=rec.scale[a:b], moving=rec.moving[a:b], tags=rec.tags[a:b],
            tag_mask=rec.tag_mask[a:b],
            song_idx=np.full(b - a, r, np.int64), shot_idx=np.arange(a, b),
        )
        item.update(song_memory(rec.song))          # used only by the cross-attention model
        return item


def collate(items: list[dict]) -> dict[str, torch.Tensor]:
    """Pad to the longest chunk. mask[b, t] = True for real shots. Song-memory keys (mem_*) have their
    own lengths and are padded separately; their *_valid arrays mark real entries."""
    L = max(len(it["scale"]) for it in items)
    out = {}
    for k in items[0]:
        arrs = [np.asarray(it[k]) for it in items]
        n = max(len(a) for a in arrs) if k.startswith("mem_") else L
        shape = (len(items), n) + arrs[0].shape[1:]
        buf = np.zeros(shape, dtype=arrs[0].dtype)
        for j, a in enumerate(arrs):
            buf[j, :len(a)] = a
        out[k] = torch.from_numpy(buf)
    mask = np.zeros((len(items), L), bool)
    for j, it in enumerate(items):
        mask[j, :len(it["scale"])] = True
    out["mask"] = torch.from_numpy(mask)
    return out


def label_stats(records: list[SongRecord], vocab: Vocab) -> dict:
    """Inverse-frequency scale weights, multi-tag pos_weight, and the repetition-control
    statistic for settings. Computed on whatever records are passed (train only)."""
    scale = np.concatenate([r.scale for r in records])
    counts = np.bincount(scale, minlength=5).astype(np.float64)
    w = counts.sum() / (5 * np.maximum(counts, 1))
    tags = np.concatenate([r.tags[r.tag_mask > 0] for r in records])
    pos = tags.mean(0)
    pos_weight = np.clip((1 - pos) / np.maximum(pos, 1e-4), 1.0, 20.0)
    # share of shots in a song taken by its most frequent setting (for rep_limit)
    g = vocab.group_by_name["setting"]
    shares = []
    for r in records:
        if len(r.tags) >= 10:
            shares.append(r.tags[:, g.ids].sum(0).max() / len(r.tags))
    return {
        "scale_counts": counts.tolist(),
        "scale_weight": (w / w.mean()).tolist(),
        "tag_pos_rate": pos.tolist(),
        "tag_pos_weight": pos_weight.tolist(),
        "setting_max_share_p50": float(np.percentile(shares, 50)) if shares else None,
        "setting_max_share_p95": float(np.percentile(shares, 95)) if shares else None,
    }
