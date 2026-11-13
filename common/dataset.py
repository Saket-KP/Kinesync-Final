"""Windowed Stage-1 dataset.

Reads only cached .npz / .json -- never raw video, per project convention. The whole
corpus of built tensors is ~30 MB at 25 fps, so every video is held in RAM and
windows are cut on the fly.

    python -m common.dataset --build-stats     # writes data/norm_stats.json (train split)
    python -m common.dataset --report
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from common.config import Config, add_config_args
from common.features import (
    Normaliser,
    build_cond,
    build_target,
    compute_norm_stats,
    genre_id,
    load_norm_stats,
    load_raw,
    save_norm_stats,
    usable_videos,
)
from common.manifest import load_index, summarise
from common.splits import videos_in_split


@dataclass
class Song:
    video_id: str
    genre: int
    x: np.ndarray        # (T, 5) normalised target
    cond: np.ndarray     # (T, 26) normalised conditioning
    cut_times: np.ndarray
    beat_times: np.ndarray
    duration_s: float


class TimelineWindows(Dataset):
    """Fixed-length windows of the 5-channel timeline, with rhythm conditioning."""

    def __init__(
        self,
        root: str | Path,
        cfg: Config,
        split: str = "train",
        limit: int | None = None,
        normaliser: Normaliser | None = None,
        verbose: bool = True,
    ):
        self.cfg = cfg
        self.root = Path(root)
        self.split = split
        self.W = cfg.data.window_frames
        self.hop = cfg.data.hop_frames

        index = load_index(root)
        vids = [v for v in videos_in_split(root, split) if v in index]
        if limit is not None:
            vids = vids[:limit]
        if not vids:
            raise ValueError(f"no videos for split={split!r}")

        if normaliser is None:
            stats = None if cfg.train.normalize == "none" else load_norm_stats(root)
            normaliser = Normaliser(stats, cfg.train.normalize)
        self.norm = normaliser

        self.songs: list[Song] = []
        self.windows: list[tuple[int, int]] = []
        self.degenerate: list[str] = []     # videos whose cut target is all zero
        self.dropped: list[str] = []
        for vid in vids:
            raw = load_raw(root, vid)
            n_cuts = int(len(raw["cut_times"]))
            if n_cuts == 0:
                self.degenerate.append(vid)
            if n_cuts < cfg.data.min_cuts_per_video:
                self.dropped.append(vid)
                continue
            x, _ = build_target(raw, cfg)
            c = build_cond(raw, cfg)
            song = Song(
                video_id=vid,
                genre=genre_id(index[vid].genre),
                x=self.norm.norm_out(x),
                cond=self.norm.norm_cond(c),
                cut_times=np.asarray(raw["cut_times"], dtype=np.float64),
                beat_times=np.asarray(raw["beat_times"], dtype=np.float64),
                duration_s=index[vid].duration_s,
            )
            si = len(self.songs)
            self.songs.append(song)
            T = song.x.shape[0]
            if T < self.W:
                continue
            for s in range(0, T - self.W + 1, self.hop):
                self.windows.append((si, s))

        if verbose:
            mb = sum(s.x.nbytes + s.cond.nbytes for s in self.songs) / 1e6
            print(f"[{split}] {len(self.songs)} videos, {len(self.windows)} windows "
                  f"of {self.W} frames ({cfg.data.window_s:.0f} s, hop {cfg.data.hop_s:.0f} s), "
                  f"{mb:.0f} MB cached, normalize={self.norm.mode}")
            if self.dropped:
                print(f"[{split}] dropped {len(self.dropped)} video(s) with "
                      f"< {cfg.data.min_cuts_per_video} cuts: {self.dropped[:8]}")
            kept_degen = [v for v in self.degenerate if v not in self.dropped]
            if kept_degen:
                print(f"[{split}] !! {len(kept_degen)} video(s) have ZERO cuts, so their "
                      f"cut target is all zero: {kept_degen[:8]}")
                print(f"[{split}]    they teach the model 'never cut'; set "
                      f"data.min_cuts_per_video=1 to drop them")

    def __len__(self) -> int:
        return len(self.windows)

    def __getitem__(self, i: int) -> dict:
        si, s = self.windows[i]
        song = self.songs[si]
        e = s + self.W
        return {
            "x": torch.from_numpy(song.x[s:e].copy()),
            "cond": torch.from_numpy(song.cond[s:e].copy()),
            "genre": torch.tensor(song.genre, dtype=torch.long),
        }

    # ------------------------------------------------------------- eval helpers
    def song_by_id(self, video_id: str) -> Song:
        for s in self.songs:
            if s.video_id == video_id:
                return s
        raise KeyError(video_id)


def load_song(root, cfg: Config, video_id: str, normaliser: Normaliser | None = None) -> Song:
    """A single full-length song, for sampling and evaluation."""
    index = load_index(root)
    if video_id not in index:
        raise KeyError(f"{video_id} has no feature file under {root}")
    if normaliser is None:
        stats = None if cfg.train.normalize == "none" else load_norm_stats(root)
        normaliser = Normaliser(stats, cfg.train.normalize)
    raw = load_raw(root, video_id)
    x, _ = build_target(raw, cfg)
    c = build_cond(raw, cfg)
    return Song(
        video_id=video_id,
        genre=genre_id(index[video_id].genre),
        x=normaliser.norm_out(x),
        cond=normaliser.norm_cond(c),
        cut_times=np.asarray(raw["cut_times"], dtype=np.float64),
        beat_times=np.asarray(raw["beat_times"], dtype=np.float64),
        duration_s=index[video_id].duration_s,
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    add_config_args(ap)
    ap.add_argument("--build-stats", action="store_true",
                    help="compute norm stats from the TRAIN split and write data/norm_stats.json")
    ap.add_argument("--report", action="store_true", help="print the per-channel report")
    ap.add_argument("--split", default="train")
    ap.add_argument("--limit", type=int, default=None)
    a = ap.parse_args()
    cfg = Config.load(a.config, a.overrides)
    root = cfg.data.root

    if a.build_stats:
        all_train = videos_in_split(root, "train")
        train_vids = usable_videos(root, all_train, cfg.data.min_cuts_per_video)
        print(summarise(load_index(root)))
        skipped = sorted(set(all_train) - set(train_vids))
        if skipped:
            print(f"\nexcluding {len(skipped)} train video(s) with "
                  f"< {cfg.data.min_cuts_per_video} cuts: {skipped}")
        print(f"\ncomputing norm stats over {len(train_vids)} TRAIN videos "
              f"(val/test excluded to avoid leakage)")
        stats = compute_norm_stats(root, cfg, train_vids)
        p = save_norm_stats(root, stats)
        print(f"wrote {p}  ({stats['n_frames']} frames)")
        for n, m, s in zip(stats["out_channels"], stats["out_mean"], stats["out_std"]):
            print(f"   out {n:<14} mean {m:>10.4f}  std {s:>10.4f}")
        print(f"   dead out channels : {stats['dead_out_channels'] or '-'}")
        print(f"   dead cond channels: {stats['dead_cond_channels'] or '-'}")
        return

    if a.report:
        from common.report import channel_report

        vids = videos_in_split(root, a.split)
        channel_report(root, cfg, vids)
        return

    ds = TimelineWindows(root, cfg, a.split, a.limit)
    b = ds[0]
    print({k: tuple(v.shape) if v.ndim else v.item() for k, v in b.items()})


if __name__ == "__main__":
    main()
