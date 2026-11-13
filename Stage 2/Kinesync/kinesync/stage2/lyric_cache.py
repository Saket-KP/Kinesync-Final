"""Sentence embeddings of lyric lines, cached per video in cache/lyric_emb/{video_id}.npy.

One row per line in the lyrics file (same order). Lines with kept=false get a zero row and
are never used. Encoder: sentence-transformers/all-MiniLM-L6-v2 (384-d), pinned in
configs/versions.yaml.

    python -m kinesync.stage2.lyric_cache            # build the cache for every video
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from kinesync.common import data_root, load_lyric_lines, load_splits

MODEL_ID = "sentence-transformers/all-MiniLM-L6-v2"
DIM = 384
_model = None


def cache_path(vid: str, root=None) -> Path:
    return Path(root or data_root()) / "cache" / "lyric_emb" / f"{vid}.npy"


def _encoder():
    global _model
    if _model is None:
        from sentence_transformers import SentenceTransformer
        _model = SentenceTransformer(MODEL_ID, device="cpu")
    return _model


def encode_lines(lines: list[dict]) -> np.ndarray:
    out = np.zeros((len(lines), DIM), np.float32)
    idx = [i for i, l in enumerate(lines) if l.get("kept") and l.get("text", "").strip()]
    if idx:
        emb = _encoder().encode([lines[i]["text"].strip() for i in idx], batch_size=64,
                                normalize_embeddings=True, show_progress_bar=False)
        out[idx] = emb
    return out


def get_line_embeddings(vid: str, lines: list[dict] | None = None, root=None) -> np.ndarray:
    p = cache_path(vid, root)
    if lines is None:
        lines = load_lyric_lines(vid, root)
    if p.exists():
        emb = np.load(p)
        if len(emb) == len(lines):
            return emb
    emb = encode_lines(lines)
    p.parent.mkdir(parents=True, exist_ok=True)
    np.save(p, emb)
    return emb


def main():
    ap = argparse.ArgumentParser()
    ap.parse_args()
    vids = load_splits().index.tolist()
    for n, vid in enumerate(vids):
        get_line_embeddings(vid)
        if n % 50 == 0:
            print(f"{n}/{len(vids)}", flush=True)
    print("done:", len(vids), "videos ->", cache_path(vids[0]).parent)


if __name__ == "__main__":
    main()
