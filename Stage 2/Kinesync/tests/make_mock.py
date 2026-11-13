"""Minimal mock data release with the same layout and keys as kinesync_data_v2.

    python tests/make_mock.py OUT_DIR
Used by the tests (KINESYNC_ROOT points at it). Lyric embeddings are written straight into the
cache as random vectors, so no sentence encoder is downloaded.
"""
from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
GENRES = ["electronic", "hiphop", "pop", "rnb_latin", "rock"]


def make_mock(root: Path, n_songs: int = 8, seed: int = 0) -> list[str]:
    root = Path(root)
    rng = np.random.default_rng(seed)
    for d in ("features", "shots", "lyrics", "stats", "vocab", "cache/lyric_emb"):
        (root / d).mkdir(parents=True, exist_ok=True)
    shutil.copy(REPO / "vocab" / "tags_v1.json", root / "vocab" / "tags_v1.json")
    vocab = json.load(open(root / "vocab" / "tags_v1.json"))["tags"]
    groups = {}
    for t in vocab:
        groups.setdefault(t["group"].lower(), []).append(t["id"])
    max_picks = {"setting": 1, "lighting": 1, "subject": 2, "action": 2, "style": 2}

    split_of = ["train"] * (n_songs - 3) + ["val", "val", "test"]
    rows, vids = [], []
    for n in range(n_songs):
        vid = f"v{n:04d}"
        vids.append(vid)
        dur = float(rng.integers(30, 50))
        T10 = int(dur * 10)
        T2, S = T10 // 5, max(1, T10 // 10)
        cut_times = np.sort(rng.uniform(0.6, dur - 0.6, size=int(dur / 1.5))).astype(np.float32)
        np.savez(root / "features" / f"{vid}.npz",
                 onset=rng.random(T10).astype(np.float32) * 5, flux=rng.random(T10).astype(np.float32) * 900,
                 chroma=rng.random((T10, 12)).astype(np.float32), beat_phase=rng.random((T10, 2)).astype(np.float32),
                 downbeat_phase=np.zeros((T10, 2), np.float32),
                 section=np.repeat(np.arange(8), int(np.ceil(T10 / 8)))[:T10].astype(np.int16),
                 clap=(rng.normal(size=(S, 512)) * 0.05).astype(np.float16),
                 beat_times=np.arange(0.5, dur, 0.5).astype(np.float32), cut_times=cut_times,
                 cut=np.zeros(T2, np.uint8), flow=rng.random(T2).astype(np.float32) * 3,
                 color_drift=rng.random(T2).astype(np.float32) * 0.1,
                 brightness=rng.random(T2).astype(np.float32), saturation=rng.random(T2).astype(np.float32),
                 valid=np.ones(T2, np.uint8))
        edges = np.concatenate([[0.0], cut_times, [T10 / 10]])
        shots = []
        for a, b in zip(edges[:-1], edges[1:]):
            tags = []
            if b - a >= 0.5:
                for g, ids in groups.items():
                    k = int(rng.integers(0, max_picks[g] + 1))
                    tags += sorted(rng.choice(ids, size=k, replace=False).tolist())
            shots.append({"start": round(float(a), 4), "end": round(float(b), 4), "scale": int(rng.integers(0, 5)),
                          "scale_conf": 0.9, "moving": int(rng.integers(0, 2)), "camera_motion": 1.0,
                          "tags": tags, "tag_scores": [1.5] * len(tags)})
        json.dump({"vocab_version": "tags_v1", "shots": shots}, open(root / "shots" / f"{vid}_shots.json", "w"))
        lines = [{"start": float(s), "end": float(s + 3), "text": f"mock lyric line number {i} la la la la la",
                  "avg_logprob": -0.3, "no_speech_prob": 0.1, "compression_ratio": 1.2, "kept": bool(i % 3)}
                 for i, s in enumerate(np.arange(2.0, dur - 4, 5.0))]
        json.dump({"language": "en", "lines": lines}, open(root / "lyrics" / f"{vid}_lyrics.json", "w"))
        np.save(root / "cache" / "lyric_emb" / f"{vid}.npy", rng.normal(size=(len(lines), 384)).astype(np.float32))
        rows.append(f"{vid},yt{n},Artist {n},{GENRES[n % 5]},2020,mixed,{dur},{split_of[n]}")
    (root / "splits.csv").write_text("video_id,youtube_id,artist,genre,year,video_type,duration_s,split\n"
                                     + "\n".join(rows) + "\n")
    shutil.copy(REPO / "stats" / "norm_train.json", root / "stats" / "norm_train.json")
    # Fake CLAP text cache (random unit vectors): lets the CLAP-context code run without the model.
    sys.path.insert(0, str(REPO))
    from kinesync.render.audio_context import AudioDescriber, GROUPS, SINGING, _slug
    import yaml
    cfg = yaml.safe_load(open(REPO / "configs" / "render.yaml", encoding="utf-8"))["clap"]
    texts = [a for _, _, items in GROUPS.values() for a, _ in items] + list(SINGING)
    T = rng.normal(size=(len(texts), 512)).astype(np.float32)
    T /= np.linalg.norm(T, axis=1, keepdims=True)
    np.savez(root / "cache" / f"clap_context_{_slug(cfg['model_id'])}.npz", T=T, mu=np.zeros(len(texts)),
             sd=np.full(len(texts), 0.05), sing_cut=0.0, texts=np.array(texts))
    return vids


if __name__ == "__main__":
    print(make_mock(Path(sys.argv[1])))
