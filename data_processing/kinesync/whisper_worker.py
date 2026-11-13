import librosa
from faster_whisper import WhisperModel

MAX_SEG = 8.0   # Split segments longer than 8.0s
MIN_PART = 1.5  # Minimum allowed sub-segment length

def split_words(words):
    """Recursively split a long segment at its largest inter-word pause."""
    if len(words) < 2 or words[-1].end - words[0].start <= MAX_SEG:
        return [words]
    best_i, best_gap = None, -1.0
    for i in range(1, len(words)):
        if words[i].start - words[0].start < MIN_PART or words[-1].end - words[i].start < MIN_PART:
            continue
        gap = words[i].start - words[i - 1].end
        if gap > best_gap:
            best_gap, best_i = gap, i
    return [words] if best_i is None else split_words(words[:best_i]) + split_words(words[best_i:])

def transcribe_audio(wav_path: str):
    """Run int8 faster-whisper (base) with word timestamps and segment splitting."""
    audio, _ = librosa.load(wav_path, sr=16000, mono=True)
    model = WhisperModel("base", device="cpu", compute_type="int8")
    segs, info = model.transcribe(audio, beam_size=5, condition_on_previous_text=False,
                                  word_timestamps=True, vad_filter=False)
    lines = []
    for s in segs:
        for p in split_words(list(s.words or [])):
            lines.append({
                "start": round(float(p[0].start), 3),
                "end": round(float(p[-1].end), 3),
                "text": "".join(w.word for w in p).strip(),
                "avg_logprob": round(float(s.avg_logprob), 4),
                "no_speech_prob": round(float(s.no_speech_prob), 4),
                "compression_ratio": round(float(s.compression_ratio), 3),
            })
    return {"language": info.language or "en", "lines": lines}