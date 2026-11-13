"""Video index, supporting both data layouts.

v2 release (one merged archive, `splits.csv` carries metadata and the split):
    splits.csv: video_id,youtube_id,artist,genre,year,video_type,duration_s,split

v1 pilot (two per-worker archives): the archives number videos independently and
disagree about 5 of the 14 video_ids they share (same artist, different song), so a
video's metadata is read from *its own* archive's manifest, resolved via
archive_map.json. That is not a correction -- it is reading the right file for each
video. See DATA_AUDIT.md section B2.

The layout is detected from what is on disk; callers do not need to care.
"""
from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path

MANIFESTS = {0: "manifest_0.csv", 1: "manifest_1.csv"}


@dataclass(frozen=True)
class VideoRec:
    video_id: str
    youtube_id: str
    artist: str
    genre: str
    duration_s: float
    year: str = ""
    video_type: str = ""
    title: str = ""
    archive: int = -1          # -1 for the merged v2 release
    status: str = "ok"
    split: str = ""            # set by the v2 release; empty in the v1 pilot

    @property
    def features_path_name(self) -> str:
        return f"{self.video_id}.npz"


def _read_csv(path: Path) -> dict[str, dict]:
    with open(path, newline="", encoding="utf-8", errors="replace") as f:
        return {r["video_id"]: r for r in csv.DictReader(f)}


def detect_layout(root: str | Path) -> str:
    """'v2' for the merged release, 'v1' for the two-worker pilot."""
    root = Path(root)
    if (root / "archive_map.json").exists() and (root / "manifests").is_dir():
        return "v1"
    sp = root / "splits.csv"
    if sp.exists():
        with open(sp, newline="", encoding="utf-8", errors="replace") as f:
            cols = set(next(csv.reader(f), []))
        if {"artist", "genre", "duration_s"} <= cols:
            return "v2"
    raise FileNotFoundError(
        f"{root} is neither a v2 release (splits.csv with metadata) nor a v1 pilot "
        f"(archive_map.json + manifests/) -- run scripts/extract_zips.py for v1")


def load_index(root: str | Path) -> dict[str, VideoRec]:
    """Every video with a feature file on disk, with its metadata."""
    root = Path(root)
    layout = detect_layout(root)
    on_disk = sorted(p.stem for p in (root / "features").glob("*.npz"))

    if layout == "v2":
        rows = _read_csv(root / "splits.csv")
        index: dict[str, VideoRec] = {}
        missing = []
        for vid in on_disk:
            r = rows.get(vid)
            if r is None:
                missing.append(vid)
                continue
            index[vid] = VideoRec(
                video_id=vid,
                youtube_id=r["youtube_id"],
                artist=r["artist"],
                genre=r["genre"],
                duration_s=float(r["duration_s"]),
                year=r.get("year", ""),
                video_type=r.get("video_type", ""),
                title=r.get("title", ""),
                split=r.get("split", ""),
            )
        if missing:
            raise RuntimeError(f"no splits.csv row for: {missing[:10]}")
        return index

    # ---- v1 pilot: metadata comes from each video's own archive manifest
    archive_map = {k: int(v) for k, v in
                   json.loads((root / "archive_map.json").read_text()).items()}
    manifests = {aid: _read_csv(root / "manifests" / name)
                 for aid, name in MANIFESTS.items()
                 if (root / "manifests" / name).exists()}

    index = {}
    missing_meta: list[str] = []
    for vid in on_disk:
        aid = archive_map.get(vid)
        if aid is None or aid not in manifests or vid not in manifests[aid]:
            missing_meta.append(vid)
            continue
        r = manifests[aid][vid]
        index[vid] = VideoRec(
            video_id=vid,
            youtube_id=r["youtube_id"],
            artist=r["artist"],
            genre=r["genre"],
            duration_s=float(r["duration_s"]),
            year=r.get("year", ""),
            video_type=r.get("video_type", ""),
            title=r.get("title", ""),
            archive=aid,
            status=r.get("status", "ok"),
        )
    if missing_meta:
        raise RuntimeError(f"no manifest row found for: {missing_meta}")
    return index


def summarise(index: dict[str, VideoRec]) -> str:
    """One-line provenance summary. Prints 43 files / 41 unique videos for the full set,
    which is the reminder that the duplicate pair is being trained on (load-as-is)."""
    yt = {r.youtube_id for r in index.values()}
    artists = {r.artist for r in index.values()}
    dup = len(index) - len(yt)
    s = (f"{len(index)} feature files / {len(yt)} unique youtube_ids / "
         f"{len(artists)} artists")
    if dup:
        groups: dict[str, list[str]] = {}
        for v, r in index.items():
            groups.setdefault(r.youtube_id, []).append(v)
        pairs = [sorted(g) for g in groups.values() if len(g) > 1]
        s += f"  [{dup} duplicate file(s), trained on as-is: {pairs}]"
    return s


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="data")
    a = ap.parse_args()
    idx = load_index(a.root)
    print(summarise(idx))
    print()
    print(f"{'video_id':<9}{'arch':>5}{'genre':>12}{'dur':>7}  {'artist':<24}youtube_id")
    for vid, r in sorted(idx.items()):
        print(f"{vid:<9}{r.archive:>5}{r.genre:>12}{r.duration_s:>7.0f}  "
              f"{r.artist[:22]:<24}{r.youtube_id}")
