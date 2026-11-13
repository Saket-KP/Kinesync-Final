"""Cut list (handoff 7.2): cutlist.csv, and an EXPERIMENTAL CMX 3600 EDL (cutlist.edl).

Both use the storyboard shot times (song time, 25 fps timecode).
"""
from __future__ import annotations

import csv
from pathlib import Path

from kinesync.output.timecode import TC_FPS, frames_to_tc, to_frames, to_tc

COLUMNS = ["shot", "start_tc", "end_tc", "start_s", "end_s", "duration_s", "section", "scale",
           "camera", "tags", "palette", "lyric", "audio_clap"]     # audio_clap: added after the handoff


def audio_words(a: dict | None) -> str:
    """CLAP context as one short string, e.g. 'romantic mood; violins, piano; singer'."""
    if not a:
        return ""
    parts = [", ".join(a.get(g, [])) for g in ("mood", "energy", "instruments", "vocals")]
    return "; ".join(p for p in parts if p)


def shot_rows(sb: dict, palettes: list[list[str]] | None = None, audio: list | None = None) -> list[dict]:
    rows = []
    for i, s in enumerate(sb["shots"]):
        tags = "; ".join(f"{g}: {', '.join(p)}" for g, p in s["tags"].items() if p)
        rows.append({
            "shot": i + 1,
            "start_tc": to_tc(s["start"]), "end_tc": to_tc(s["end"]),
            "start_s": f"{s['start']:.3f}", "end_s": f"{s['end']:.3f}",
            "duration_s": f"{s['end'] - s['start']:.3f}",
            "section": s["section"], "scale": s["scale"],
            "camera": "moving" if s["moving"] else "static",
            "tags": tags,
            "palette": " ".join(palettes[i]) if palettes else "",
            "lyric": s.get("lyric") or "",
            "audio_clap": audio_words(audio[i]) if audio else "",
        })
    return rows


def write_csv(rows: list[dict], path: Path) -> Path:
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=COLUMNS)
        w.writeheader()
        w.writerows(rows)
    return path


def write_edl(sb: dict, rows: list[dict], path: Path) -> Path:
    """CMX 3600: one cut event per shot on reel AX (no source media), source TC = record TC =
    song time. The shot description goes in comment lines. Lyrics are left out of the EDL."""
    title = f"KINESYNC {sb['video_id']} {sb.get('sample_id', 's%d' % sb['stage1_sample']).upper()}"
    lines = [f"TITLE: {title}", "FCM: NON-DROP FRAME", ""]
    for r, s in zip(rows, sb["shots"]):
        a, b = to_frames(s["start"]), to_frames(s["end"])
        if b <= a:
            b = a + 1
        tin, tout = frames_to_tc(a, TC_FPS), frames_to_tc(b, TC_FPS)
        lines.append(f"{r['shot']:03d}  AX       V     C        {tin} {tout} {tin} {tout}")
        lines.append(f"* FROM CLIP NAME: SHOT {r['shot']:03d}")
        desc = f"{r['scale']}, {r['camera']}" + (f", {r['tags']}" if r["tags"] else "") + \
               (f", {r['palette']}" if r["palette"] else "") + f", {r['section']}"
        lines.append(f"* COMMENT: {desc.upper()}")
        lines.append("")
    path.write_text("\n".join(lines), encoding="ascii", errors="replace")
    return path
