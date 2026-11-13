"""Audio extractor for NEW songs: an audio file -> features/{id}.npz + lyrics/{id}_lyrics.json.

    python -m kinesync.features.extract path/to/song.mp3 --genre pop

Follows data release v2 (the data team's note), which differs from the blueprint in places:
  - grids: audio features at 10 fps (hop 2205 at 22,050 Hz), CLAP at 1 Hz, times in seconds
  - CLAP: laion/clap-htsat-unfused (identified from the stored vectors, docs/IMPLEMENTATION.md 15b),
    48 kHz, one unit-length vector per second from a 5 s window centred on that second, float16
  - section: 8 contiguous segments in time order (every release song has exactly that), here from
    librosa's agglomerative segmentation of MFCC + chroma; a plain segment index, not verse/chorus
  - downbeat_phase: all zeros, as in the release
  - lyrics: Whisper "base" on the full mix (no Demucs). A line is kept when avg_logprob >= -1.0 and
    compression_ratio <= 2.4, the rule that reproduces 98.5% of the release's kept flags; songs whose
    language is not en/es have every line kept = false, as in the release
Not reproduced exactly (Stage 2 does not use them; the mock Stage 1 uses onset only after per-song
normalisation): the scale of onset and flux, and chroma (chroma_stft here; the release's method is
not documented).

There are no video features (no cut_times, flow, ...): a new song has no video. Its shot timing then
comes from Stage 1 (today the audio-mode mock).

A song's id is "u" + the first 10 hex digits of the file's SHA-256, so uploading the same file again
reuses the extraction. user_songs.csv records each one.
"""
from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import json
import os
import shutil
import subprocess
import time
from datetime import datetime
from pathlib import Path

import numpy as np

from kinesync.common import AUDIO_FPS, REPO_ROOT, data_root

# librosa's numba functions cache compiled code next to librosa itself. When librosa is installed in a
# read-only folder (here C:\Program Files), every cache write fails and Windows' tempfile retries
# thousands of times, so importing librosa takes many minutes. Point the cache somewhere writable first.
os.environ.setdefault("NUMBA_CACHE_DIR", str(REPO_ROOT / "cache" / "numba"))

SR = 22050
HOP = SR // AUDIO_FPS          # 2205 samples = exactly 10 frames per second
CLAP_SR = 48000
CLAP_WINDOW_S = 5.0
CLAP_MODEL = "laion/clap-htsat-unfused"
CLAP_REVISION = "8fa0f1c6d0433df6e97c127f64b2a1d6c0dcda8a"
WHISPER_MODEL = "base"
N_SECTIONS = 8
KEPT_LANGUAGES = ("en", "es")
REGISTRY = "user_songs.csv"
REGISTRY_COLS = ["video_id", "file_name", "genre", "duration_s", "sha256", "language", "kept_lines",
                 "lines", "created"]


def song_id(path: Path) -> tuple[str, str]:
    h = hashlib.sha256(Path(path).read_bytes()).hexdigest()
    return "u" + h[:10], h


def decode(path: Path, sr: int) -> np.ndarray:
    """Any format ffmpeg reads -> mono float32 at `sr`."""
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        raise RuntimeError("ffmpeg is not on PATH; it is needed to decode audio files")
    out = subprocess.run([ffmpeg, "-nostdin", "-v", "error", "-i", str(path), "-ac", "1", "-ar", str(sr),
                          "-f", "f32le", "-"], capture_output=True, check=True).stdout
    y = np.frombuffer(out, np.float32).copy()
    if y.size < sr:
        raise ValueError(f"{path.name}: less than 1 second of audio decoded")
    return y


def fit(x: np.ndarray, T: int) -> np.ndarray:
    """Crop or edge-pad the first axis to length T."""
    if len(x) >= T:
        return x[:T]
    pad = [(0, T - len(x))] + [(0, 0)] * (x.ndim - 1)
    return np.pad(x, pad, mode="edge")


def rhythm_features(y: np.ndarray, T10: int) -> dict:
    import librosa
    onset = fit(librosa.onset.onset_strength(y=y, sr=SR, hop_length=HOP), T10)
    mag = np.abs(librosa.stft(y, n_fft=4096, hop_length=HOP))
    flux = np.concatenate([[0.0], np.maximum(np.diff(mag, axis=1), 0).sum(0)])
    chroma = librosa.feature.chroma_stft(S=mag ** 2, sr=SR, hop_length=HOP)          # max 1 per frame
    _, beats = librosa.beat.beat_track(y=y, sr=SR, hop_length=512, units="time")
    beats = np.asarray(beats, np.float32)

    # beat phase: fraction of the way from the previous beat to the next, as sin/cos (0 at a beat)
    t = (np.arange(T10) + 0.5) / AUDIO_FPS
    if len(beats) >= 2:
        step = float(np.median(np.diff(beats)))
        grid = np.concatenate([[beats[0] - step * k for k in range(64, 0, -1)], beats,
                               [beats[-1] + step * k for k in range(1, 65)]])
        i = np.clip(np.searchsorted(grid, t, side="right") - 1, 0, len(grid) - 2)
        phase = (t - grid[i]) / np.maximum(grid[i + 1] - grid[i], 1e-3)
    else:
        phase = np.zeros(T10)
    beat_phase = np.stack([np.sin(2 * np.pi * phase), np.cos(2 * np.pi * phase)], 1)

    # sections: 8 contiguous segments in time order (as in every release song)
    mfcc = librosa.feature.mfcc(y=y, sr=SR, hop_length=HOP, n_mfcc=13)
    feat = librosa.util.normalize(np.vstack([fit(mfcc.T, T10).T, fit(chroma.T, T10).T]), axis=1)
    k = min(N_SECTIONS, T10)
    bounds = librosa.segment.agglomerative(feat, k)                    # sorted start frames, first is 0
    section = np.zeros(T10, np.int16)
    for idx, b in enumerate(bounds):
        section[b:] = idx
    return {"onset": onset.astype(np.float32), "flux": fit(flux, T10).astype(np.float32),
            "chroma": fit(chroma.T, T10).astype(np.float32), "beat_phase": beat_phase.astype(np.float32),
            "downbeat_phase": np.zeros((T10, 2), np.float32), "beat_times": beats, "section": section}


def clap_features(y48: np.ndarray, S: int, device: str, log=print) -> np.ndarray:
    """One embedding per second s, from the 5 s window centred on s + 0.5 (zero-padded at the ends)."""
    import torch
    from transformers import ClapModel, ClapProcessor
    from transformers.utils import logging as hf_logging
    hf_logging.set_verbosity_error()
    model = ClapModel.from_pretrained(CLAP_MODEL, revision=CLAP_REVISION).to(device).eval()
    proc = ClapProcessor.from_pretrained(CLAP_MODEL, revision=CLAP_REVISION)
    half = int(CLAP_WINDOW_S / 2 * CLAP_SR)
    padded = np.pad(y48, (half, half))
    out = []
    for a in range(0, S, 16):
        clips = []
        for s in range(a, min(a + 16, S)):
            c = int((s + 0.5) * CLAP_SR) + half                         # centre, in padded samples
            clips.append(padded[c - half:c + half])
        inp = proc(audio=clips, sampling_rate=CLAP_SR, return_tensors="pt").to(device)
        with torch.no_grad():
            e = model.get_audio_features(**inp)
        e = e if torch.is_tensor(e) else e.pooler_output
        out.append(torch.nn.functional.normalize(e, dim=-1).float().cpu().numpy())
        log(f"[extract] CLAP {min(a + 16, S)}/{S} seconds")
    del model
    gc.collect()
    if device == "cuda":
        torch.cuda.empty_cache()
    return np.concatenate(out).astype(np.float16)


def is_kept(avg_logprob: float, compression_ratio: float, language: str) -> bool:
    """The release's rule (98.5% agreement with its kept flags on 23,447 en/es lines): Whisper's own
    fallback thresholds, and only English or Spanish songs."""
    return avg_logprob >= -1.0 and compression_ratio <= 2.4 and language in KEPT_LANGUAGES


def lyric_lines(path: Path, device: str, log=print) -> dict:
    import torch
    import whisper
    y16 = decode(path, 16000)
    model = whisper.load_model(WHISPER_MODEL, device=device)
    res = model.transcribe(y16, verbose=None, fp16=(device == "cuda"))
    del model
    gc.collect()
    if device == "cuda":
        torch.cuda.empty_cache()
    lang = res.get("language", "")
    lines = []
    for seg in res["segments"]:
        text = seg["text"].strip()
        if not text:
            continue
        kept = is_kept(seg["avg_logprob"], seg["compression_ratio"], lang)
        lines.append({"start": round(float(seg["start"]), 2), "end": round(float(seg["end"]), 2), "text": text,
                      "avg_logprob": round(float(seg["avg_logprob"]), 4),
                      "no_speech_prob": round(float(seg["no_speech_prob"]), 4),
                      "compression_ratio": round(float(seg["compression_ratio"]), 3), "kept": bool(kept)})
    return {"language": lang, "lines": lines, "whisper_model": WHISPER_MODEL,
            "masked": lang not in KEPT_LANGUAGES}


def validate(feat: dict, T10: int) -> list[str]:
    """The release's contract checks that apply to audio-only files."""
    errs = []
    S = max(1, T10 // 10)
    for k in ("onset", "flux", "section"):
        if feat[k].shape != (T10,):
            errs.append(f"{k} shape {feat[k].shape} != ({T10},)")
    for k, d in (("chroma", 12), ("beat_phase", 2), ("downbeat_phase", 2)):
        if feat[k].shape != (T10, d):
            errs.append(f"{k} shape {feat[k].shape}")
    if feat["clap"].shape != (S, 512):
        errs.append(f"clap shape {feat['clap'].shape} != ({S}, 512)")
    for k, v in feat.items():
        if np.issubdtype(v.dtype, np.floating) and not np.isfinite(v).all():
            errs.append(f"{k} has NaN or inf")
    if len(feat["beat_times"]) > 1 and (np.diff(feat["beat_times"]) <= 0).any():
        errs.append("beat_times not increasing")
    if feat["section"].max() >= N_SECTIONS or (np.diff(feat["section"].astype(int)) < 0).any():
        errs.append("section not contiguous 0..7")
    return errs


def register(root: Path, row: dict) -> None:
    p = root / REGISTRY
    rows = []
    if p.exists():
        with open(p, encoding="utf-8", newline="") as f:
            rows = [r for r in csv.DictReader(f) if r["video_id"] != row["video_id"]]
    rows.append(row)
    with open(p, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=REGISTRY_COLS)
        w.writeheader()
        w.writerows(rows)


def extract(path, genre: str | None = None, root=None, force: bool = False, log=print,
            title: str | None = None) -> str:
    """Contract used by kinesync.pipeline --audio: writes the files and returns the song's id.
    `title` is the name shown on the sheet (the portal stores uploads under a hash, so it passes the
    original file name); default: the file's own name."""
    import torch
    path = Path(path)
    root = Path(root or data_root())
    vid, sha = song_id(path)
    npz, lyr = root / "features" / f"{vid}.npz", root / "lyrics" / f"{vid}_lyrics.json"
    if npz.exists() and lyr.exists() and not force:
        log(f"[extract] {path.name} already extracted as {vid}; reusing it")
        if title:                                      # keep the newest name for the sheet
            reg = root / REGISTRY
            if reg.exists():
                with open(reg, encoding="utf-8", newline="") as f:
                    rows = list(csv.DictReader(f))
                for r in rows:
                    if r["video_id"] == vid:
                        register(root, {**r, "file_name": title})
        return vid
    device = "cuda" if torch.cuda.is_available() else "cpu"
    t0 = time.time()
    log(f"[extract] decoding {path.name}")
    y = decode(path, SR)
    dur = len(y) / SR
    T10 = int(np.floor(dur * AUDIO_FPS))
    log(f"[extract] {dur:.1f} s of audio; rhythm features, beats and sections")
    feat = rhythm_features(y, T10)
    log(f"[extract] CLAP ({CLAP_MODEL}) on {device}")
    feat["clap"] = clap_features(decode(path, CLAP_SR), max(1, T10 // 10), device, log)
    errs = validate(feat, T10)
    if errs:
        raise ValueError("extracted features fail the contract: " + "; ".join(errs))
    log(f"[extract] lyrics (Whisper {WHISPER_MODEL}) on {device}")
    lyrics = lyric_lines(path, device, log)
    npz.parent.mkdir(parents=True, exist_ok=True)
    lyr.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(npz, **feat)
    lyr.write_text(json.dumps(lyrics, ensure_ascii=False, indent=1), encoding="utf-8")
    kept = sum(l["kept"] for l in lyrics["lines"])
    register(root, {"video_id": vid, "file_name": title or path.name, "genre": genre or "", "duration_s": round(dur, 2),
                    "sha256": sha, "language": lyrics["language"], "kept_lines": kept,
                    "lines": len(lyrics["lines"]), "created": datetime.now().isoformat(timespec="seconds")})
    log(f"[extract] done in {time.time() - t0:.0f} s: {len(feat['beat_times'])} beats, "
        f"{kept} of {len(lyrics['lines'])} lyric lines kept, language {lyrics['language'] or '?'}")
    return vid


def compare_with_release(vid_new: str, vid_release: str, root=None, max_shift_s: int = 5) -> dict:
    """Check the extractor against the data team's: extract a song that IS in the release (your own copy
    of it), then compare its CLAP rows with the release's, trying shifts of up to +-max_shift_s seconds
    (a music video's audio may start a little earlier or later than the track). If both used the same
    model and windows, the best shift gives a mean cosine near 1.0."""
    from kinesync.common import load_features
    a = load_features(vid_new, root)["clap"].astype(np.float32)
    b = load_features(vid_release, root)["clap"].astype(np.float32)
    best = None
    for s in range(-max_shift_s, max_shift_s + 1):
        x, y = (a[s:], b) if s >= 0 else (a, b[-s:])
        n = min(len(x), len(y))
        if n < 10:
            continue
        cos = float(np.mean(np.sum(x[:n] * y[:n], 1)))
        if best is None or cos > best["mean_cosine"]:
            best = {"shift_s": s, "mean_cosine": round(cos, 4), "seconds_compared": n}
    return best


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("audio")
    ap.add_argument("--genre")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--title", help="name shown on the sheet (default: the file name)")
    ap.add_argument("--compare_with", metavar="VIDEO_ID",
                    help="the release's id for this same song: report how closely the CLAP vectors match")
    a = ap.parse_args()
    vid = extract(a.audio, a.genre, force=a.force, log=lambda m: print(m, flush=True), title=a.title)
    print(vid)
    if a.compare_with:
        r = compare_with_release(vid, a.compare_with)
        print(f"CLAP match with {a.compare_with}: mean cosine {r['mean_cosine']} at a {r['shift_s']:+d} s shift "
              f"over {r['seconds_compared']} s (near 1.0 = same extraction; the same model with different "
              f"windows or audio typically scores well below)")


if __name__ == "__main__":
    main()
