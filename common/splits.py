"""Artist-wise train/val/test split.

Splitting by artist, not by video, is what keeps the evaluation honest: the same
artist never appears on both sides. It also means the two duplicated videos
(v0162/v0163, v0714/v0715) land in the same split as each other, so loading the
data as-is costs double weight on those songs but leaks nothing.

Deterministic: artists are ordered by a hash of their name, then cut at the
requested proportions, so the split is stable across machines and reruns and does
not depend on how many videos happen to be extracted.

    python -m common.splits --root data
"""
from __future__ import annotations

import argparse
import csv
import hashlib
from collections import Counter
from pathlib import Path

from common.manifest import VideoRec, load_index, summarise

SPLITS = ("train", "val", "test")


def _artist_key(artist: str) -> str:
    return hashlib.md5(artist.strip().lower().encode("utf-8")).hexdigest()


def assign_splits(
    index: dict[str, VideoRec],
    frac_train: float = 0.70,
    frac_val: float = 0.15,
) -> dict[str, str]:
    """video_id -> split, grouped by artist."""
    artists = sorted({r.artist for r in index.values()}, key=_artist_key)
    n = len(artists)
    n_train = int(round(frac_train * n))
    n_val = int(round(frac_val * n))
    # make sure val and test are non-empty when there are enough artists
    if n >= 3:
        n_train = min(n_train, n - 2)
        n_val = max(1, min(n_val, n - n_train - 1))

    by_artist: dict[str, str] = {}
    for i, a in enumerate(artists):
        if i < n_train:
            by_artist[a] = "train"
        elif i < n_train + n_val:
            by_artist[a] = "val"
        else:
            by_artist[a] = "test"
    return {vid: by_artist[r.artist] for vid, r in index.items()}


def write_splits(root: str | Path, index: dict[str, VideoRec],
                 split_of: dict[str, str]) -> Path:
    out = Path(root) / "splits.csv"
    with open(out, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["video_id", "youtube_id", "artist", "genre", "split"])
        for vid in sorted(index):
            r = index[vid]
            w.writerow([vid, r.youtube_id, r.artist, r.genre, split_of[vid]])
    return out


def load_splits(root: str | Path) -> dict[str, str]:
    p = Path(root) / "splits.csv"
    if not p.exists():
        raise FileNotFoundError(f"{p} not found -- run `python -m common.splits` first")
    with open(p, newline="", encoding="utf-8") as f:
        return {r["video_id"]: r["split"] for r in csv.DictReader(f)}


def videos_in_split(root: str | Path, split: str) -> list[str]:
    if split == "all":
        return sorted(load_splits(root))
    return sorted(v for v, s in load_splits(root).items() if s == split)


def check(index: dict[str, VideoRec], split_of: dict[str, str]) -> None:
    """No artist may span two splits."""
    seen: dict[str, str] = {}
    for vid, s in split_of.items():
        a = index[vid].artist
        if a in seen and seen[a] != s:
            raise AssertionError(f"artist {a!r} spans splits {seen[a]} and {s}")
        seen[a] = s


def release_splits(index: dict[str, VideoRec]) -> dict[str, str] | None:
    """The split column shipped with the release, if every video has one."""
    got = {v: r.split for v, r in index.items() if r.split}
    return got if len(got) == len(index) and index else None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="data")
    ap.add_argument("--frac-train", type=float, default=0.70)
    ap.add_argument("--frac-val", type=float, default=0.15)
    ap.add_argument("--regenerate", action="store_true",
                    help="recompute the split even if the release ships one "
                         "(OVERWRITES splits.csv)")
    a = ap.parse_args()

    index = load_index(a.root)
    print(summarise(index))

    shipped = release_splits(index)
    if shipped is not None and not a.regenerate:
        # The v2 release already splits by artist (seed 42). Validate, do not overwrite:
        # regenerating would silently reshuffle which artists are held out.
        split_of = shipped
        check(index, split_of)
        print(f"\nusing the split shipped in {Path(a.root)/'splits.csv'} "
              f"(not regenerated; pass --regenerate to override)")
    else:
        split_of = assign_splits(index, a.frac_train, a.frac_val)
        check(index, split_of)
        out = write_splits(a.root, index, split_of)
        print(f"\nwrote {out}")

    vids = Counter(split_of.values())
    print(f"\n{'split':<8}{'videos':>8}{'artists':>9}{'hours':>8}  genres")
    for s in SPLITS:
        sel = [v for v, x in split_of.items() if x == s]
        if not sel:
            print(f"{s:<8}{0:>8}{0:>9}{0.0:>8.2f}  {{}}")
            continue
        arts = {index[v].artist for v in sel}
        hrs = sum(index[v].duration_s for v in sel) / 3600
        g = Counter(index[v].genre for v in sel)
        print(f"{s:<8}{vids[s]:>8}{len(arts):>9}{hrs:>8.2f}  {dict(g)}")
    print("\nno artist spans two splits: OK")


if __name__ == "__main__":
    main()
