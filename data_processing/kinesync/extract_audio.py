import librosa
import numpy as np
import torch
from sklearn.cluster import AgglomerativeClustering
from transformers import ClapModel, ClapProcessor
from kinesync.constants import AUDIO_FPS, CLAP_SR, HOP, SR, n_audio, n_clap

FINE_HOP = 512  # Finer ~23ms hop for accurate beat tracking

def _compute_beat_phase(frame_times: np.ndarray, beat_times: np.ndarray) -> np.ndarray:
    """Continuous cyclical beat position encoded as (sin, cos)."""
    idx = np.clip(np.searchsorted(beat_times, frame_times, side="right") - 1, 0, len(beat_times) - 2)
    t0, t1 = beat_times[idx], beat_times[idx + 1]
    frac = np.clip((frame_times - t0) / np.maximum(t1 - t0, 1e-4), 0.0, 1.0)
    angle = 2.0 * np.pi * frac
    return np.stack([np.sin(angle), np.cos(angle)], axis=-1).astype(np.float32)

def _compute_sections(chroma_raw: np.ndarray, T: int, n_segments: int = 8) -> np.ndarray:
    """Temporally-constrained Ward hierarchical clustering on 1 Hz chroma."""
    chroma_1hz = chroma_raw[:, ::AUDIO_FPS]
    n_steps = chroma_1hz.shape[1]
    # Tridiagonal adjacency matrix: only adjacent seconds can merge
    grid = np.eye(n_steps, k=1) + np.eye(n_steps, k=-1)
    labels_1hz = AgglomerativeClustering(
        n_clusters=min(n_segments, n_steps), connectivity=grid, linkage="ward"
    ).fit_predict(chroma_1hz.T)
    
    # Relabel clusters 0..7 in chronological order of first appearance
    _, first_idx = np.unique(labels_1hz, return_index=True)
    remap = {old: new for new, old in enumerate(labels_1hz[np.sort(first_idx)])}
    ordered = np.array([remap[x] for x in labels_1hz], dtype=np.int16)
    return ordered[np.minimum(np.arange(T) // AUDIO_FPS, n_steps - 1)]

def _compute_clap_1hz(y48: np.ndarray, S: int, model: ClapModel, proc: ClapProcessor) -> np.ndarray:
    """Extract L2-normalized 512-d CLAP embeddings using a 5s window centered on each second."""
    win = 5 * CLAP_SR
    padded = np.pad(y48, (win // 2, win // 2))
    windows = [padded[int((s + 0.5) * CLAP_SR) : int((s + 0.5) * CLAP_SR) + win] for s in range(S)]
    
    inputs = proc(audio=windows, sampling_rate=CLAP_SR, return_tensors="pt", padding=True)
    with torch.no_grad():
        feats = model.get_audio_features(**inputs)
        feats = feats / torch.clamp(feats.norm(dim=-1, keepdim=True), min=1e-8)
    return feats.cpu().numpy().astype(np.float16)[:S]

def filter_lyrics(raw_lines: list, y22: np.ndarray) -> list:
    """Filter Whisper hallucinations using confidence scores, RMS energy, and repeat caps."""
    filtered, prev_text, run = [], None, 0
    for line in raw_lines:
        s_idx, e_idx = int(line["start"] * SR), max(int(line["start"] * SR) + 1, int(line["end"] * SR))
        rms = float(np.sqrt(np.mean(y22[s_idx:e_idx] ** 2)))
        
        kept = (line["no_speech_prob"] <= 0.95 and line["avg_logprob"] >= -1.0
                and line.get("compression_ratio", 0.0) <= 2.4 and bool(line["text"]) and rms >= 1e-3)
        if kept:
            run = run + 1 if line["text"].lower() == prev_text else 1
            prev_text = line["text"].lower()
            if run > 2:  # Drop 3rd+ consecutive identical lyric line
                kept = False
        filtered.append({**line, "kept": kept})
    return filtered

def extract_audio_features(y: np.ndarray, dur: float):
    """Main 10 fps spectral & rhythmic feature extraction."""
    T = n_audio(dur)
    S_mag = np.abs(librosa.stft(y, n_fft=2048, hop_length=HOP))[:, :T]
    
    # 1. Spectral Onset & Half-Wave Rectified Spectral Flux
    onset = librosa.onset.onset_strength(S=librosa.amplitude_to_db(S_mag), sr=SR, hop_length=HOP)[:T]
    flux = np.sum(np.maximum(0.0, np.diff(S_mag, axis=1, prepend=S_mag[:, :1])), axis=0)[:T]
    
    # 2. Constant-Q Chroma (12 pitch classes)
    chroma = librosa.feature.chroma_cqt(y=y, sr=SR, hop_length=HOP)[:, :T].T
    
    # 3. High-resolution Beat Tracking & Phase
    onset_fine = librosa.onset.onset_strength(y=y, sr=SR, hop_length=FINE_HOP)
    _, beat_frames = librosa.beat.beat_track(onset_envelope=onset_fine, sr=SR, hop_length=FINE_HOP)
    beat_times = librosa.frames_to_time(beat_frames, sr=SR, hop_length=FINE_HOP)
    
    frame_times = (np.arange(T) + 0.5) / AUDIO_FPS
    beat_phase = _compute_beat_phase(frame_times, beat_times)
    section = _compute_sections(chroma.T, T)
    
    return onset, flux, chroma, beat_times, beat_phase, section