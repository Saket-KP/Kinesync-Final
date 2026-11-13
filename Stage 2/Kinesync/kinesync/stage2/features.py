"""Per-shot input features (handoff 4.1), shared by training, sampling, evaluation and probes.

Everything is vectorised over shots with prefix sums, so features can be rebuilt every
epoch (boundary jitter) at almost no cost.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from kinesync.common import (AUDIO_FPS, CLAP_HZ, KIN_KEYS, N_SECTIONS, VIDEO_FPS,
                             genre_list, kin_mean_std, load_features, load_lyric_lines,
                             load_splits)
from kinesync.stage2.lyric_cache import get_line_embeddings

LYRIC_DIM = 384
CLAP_DIM = 512


# --------------------------------------------------------------------------- span arithmetic

def _range_mean(arr: np.ndarray, i0: np.ndarray, i1: np.ndarray, w: np.ndarray | None = None):
    """Mean of arr[i0:i1] for each (i0, i1) pair; arr is (T, D). Optional per-frame weights w.

    Returns (means (N, D), counts (N,)). Rows with zero weight return zeros and count 0.
    """
    arr = arr.astype(np.float64)
    if w is None:
        w = np.ones(len(arr))
    w = w.astype(np.float64)
    cs = np.concatenate([np.zeros((1, arr.shape[1])), np.cumsum(arr * w[:, None], axis=0)])
    cw = np.concatenate([[0.0], np.cumsum(w)])
    tot = cs[i1] - cs[i0]
    cnt = cw[i1] - cw[i0]
    mean = np.where(cnt[:, None] > 0, tot / np.maximum(cnt, 1e-9)[:, None], 0.0)
    return mean, cnt


def frame_bounds(starts, ends, fps: float, T: int):
    """Frames whose interval [i/fps, (i+1)/fps) overlaps [start, end). Empty spans fall back
    to the single frame containing the shot midpoint."""
    i0 = np.clip(np.floor(starts * fps + 1e-6).astype(int), 0, T)
    i1 = np.clip(np.ceil(ends * fps - 1e-6).astype(int), 0, T)
    mid = np.clip(np.floor((starts + ends) / 2 * fps).astype(int), 0, T - 1)
    empty = i1 <= i0
    i0 = np.where(empty, mid, i0)
    i1 = np.where(empty, mid + 1, i1)
    return i0, i1


def clap_bounds(starts, ends, S: int, shift: float = 0.0):
    """CLAP rows whose second s satisfies start <= s < end (row s = second s, 1 Hz).
    Shots that contain no whole second (usually < 1 s) take the row nearest the midpoint.
    `shift` moves the window in seconds (probe control); rows are clamped to the song."""
    s, e = starts + shift, ends + shift
    i0 = np.clip(np.ceil(s * CLAP_HZ - 1e-6).astype(int), 0, S)
    i1 = np.clip(np.ceil(e * CLAP_HZ - 1e-6).astype(int), 0, S)
    near = np.clip(np.round((s + e) / 2 * CLAP_HZ).astype(int), 0, S - 1)
    empty = i1 <= i0
    i0 = np.where(empty, near, i0)
    i1 = np.where(empty, near + 1, i1)
    return i0, i1


# --------------------------------------------------------------------------- kinetics sources

@dataclass
class KinSource:
    """Kinetic channels (flow, color_drift, brightness, saturation) on some time grid, raw units."""
    arrays: np.ndarray          # (T, 4)
    fps: float
    valid: np.ndarray           # (T,) 1 = real frame

    @staticmethod
    def from_features(feat: dict) -> "KinSource":
        arr = np.stack([feat[k] for k in KIN_KEYS], axis=1).astype(np.float32)
        valid = feat.get("valid", np.ones(len(arr), np.uint8)).astype(np.float32)
        return KinSource(arr, VIDEO_FPS, valid)

    @staticmethod
    def from_stage1(s1: dict) -> "KinSource":
        arr = np.stack([s1[k] for k in KIN_KEYS], axis=1).astype(np.float32)
        return KinSource(arr, float(s1["fps"]), np.ones(len(arr), np.float32))

    def span_means(self, starts, ends) -> np.ndarray:
        i0, i1 = frame_bounds(starts, ends, self.fps, len(self.arrays))
        m, cnt = _range_mean(self.arrays, i0, i1, self.valid)
        if (cnt == 0).any():   # span made only of padded frames: use them anyway
            m2, _ = _range_mean(self.arrays, i0, i1)
            m[cnt == 0] = m2[cnt == 0]
        return m.astype(np.float32)


# --------------------------------------------------------------------------- song container

@dataclass
class Song:
    """Everything needed to build shot inputs for one song, independent of the spans."""
    vid: str
    duration: float
    genre_id: int
    clap: np.ndarray            # (S, 512) float16
    section: np.ndarray         # (T10,) int
    line_start: np.ndarray      # (L,) kept lines only
    line_end: np.ndarray
    line_emb: np.ndarray        # (L, 384)
    line_text: list = field(default_factory=list)

    @staticmethod
    def load(vid: str, root=None, genre: str | None = None, feat: dict | None = None) -> "Song":
        feat = feat if feat is not None else load_features(vid, root)
        genres = genre_list(root)
        if genre is None:
            sp = load_splits(root)
            genre = sp.loc[vid, "genre"] if vid in sp.index else None
        gid = genres.index(genre) if genre in genres else len(genres)
        lines = load_lyric_lines(vid, root)
        emb = get_line_embeddings(vid, lines, root)
        keep = [i for i, l in enumerate(lines) if l.get("kept")]
        return Song(
            vid=vid,
            duration=len(feat["onset"]) / AUDIO_FPS,
            genre_id=gid,
            clap=feat["clap"],              # float16 as stored; averaged in float64
            section=feat["section"].astype(np.int64),
            line_start=np.array([lines[i]["start"] for i in keep], dtype=np.float64),
            line_end=np.array([lines[i]["end"] for i in keep], dtype=np.float64),
            line_emb=emb[keep].astype(np.float32) if keep else np.zeros((0, LYRIC_DIM), np.float32),
            line_text=[lines[i]["text"].strip() for i in keep],
        )


def song_memory(song: Song) -> dict:
    """What the cross-attention model can look up for the whole song: one CLAP row per second (time
    s + 0.5) and one embedding per kept lyric line (time = line midpoint), all in seconds."""
    S = len(song.clap)
    L = len(song.line_start)
    return {
        "mem_clap": song.clap.astype(np.float32),
        "mem_clap_t": (np.arange(S) + 0.5).astype(np.float32),
        "mem_clap_valid": np.ones(S, np.float32),
        "mem_lyric": song.line_emb.astype(np.float32).reshape(L, LYRIC_DIM),
        "mem_lyric_t": ((song.line_start + song.line_end) / 2).astype(np.float32),
        "mem_lyric_valid": np.ones(L, np.float32),
    }


def shot_inputs(song: Song, starts, ends, kin: KinSource, clap_shift: float = 0.0) -> dict:
    """Build the per-shot inputs of handoff table 4.1 for spans [starts, ends) in seconds."""
    starts = np.asarray(starts, dtype=np.float64)
    ends = np.asarray(ends, dtype=np.float64)
    N = len(starts)
    dur = np.maximum(ends - starts, 1e-3)

    i0, i1 = clap_bounds(starts, ends, len(song.clap), clap_shift)
    clap, _ = _range_mean(song.clap, i0, i1)

    if len(song.line_start):
        ov = (song.line_start[None, :] < ends[:, None]) & (song.line_end[None, :] > starts[:, None])
        cnt = ov.sum(1)
        lyric = (ov.astype(np.float32) @ song.line_emb) / np.maximum(cnt, 1)[:, None]
        first = [song.line_text[int(np.argmax(r))] if r.any() else None for r in ov]
    else:
        cnt = np.zeros(N, int)
        lyric = np.zeros((N, LYRIC_DIM), np.float32)
        first = [None] * N

    kin_raw = kin.span_means(starts, ends)
    mean, std = kin_mean_std()
    mid = np.clip(np.floor((starts + ends) / 2 * AUDIO_FPS).astype(int), 0, len(song.section) - 1)

    return {
        "start": starts.astype(np.float32),
        "end": ends.astype(np.float32),
        "clap": clap.astype(np.float32),
        "lyric": lyric.astype(np.float32),
        "has_lyric": (cnt > 0).astype(np.float32),
        "log_dur": np.log(dur).astype(np.float32),
        "kin": ((kin_raw - mean) / std).astype(np.float32),
        "kin_raw": kin_raw,
        "section": np.clip(song.section[mid], 0, N_SECTIONS - 1).astype(np.int64),
        "pos": (starts / max(song.duration, 1e-3)).astype(np.float32),
        "mid_s": ((starts + ends) / 2).astype(np.float32),          # for cross-attention timing
        "genre": np.full(N, song.genre_id, np.int64),
        "first_lyric": first,
    }
