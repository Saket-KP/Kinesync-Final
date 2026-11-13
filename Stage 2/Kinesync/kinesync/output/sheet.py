"""Storyboard sheet (handoff 7.1): one self-contained HTML file (panels embedded as base64),
one row per shot, print-friendly. Optional PDF through weasyprint when it is installed.

    python -m kinesync.output.sheet --dir outputs/v0012_s0
"""
from __future__ import annotations

import argparse
import base64
import html
import json
from pathlib import Path

from kinesync.common import REPO_ROOT, load_lyric_lines, load_yaml
from kinesync.output.cutlist import shot_rows, write_csv, write_edl
from kinesync.output.timecode import to_tc


def song_title(vid: str, root=None) -> str | None:
    """Uploaded songs (ids starting with "u") are named after their file, from user_songs.csv."""
    import csv
    from kinesync.common import data_root
    p = Path(root or data_root()) / "user_songs.csv"
    if not vid.startswith("u") or not p.exists():
        return None
    with open(p, encoding="utf-8", newline="") as f:
        for r in csv.DictReader(f):
            if r["video_id"] == vid:
                return Path(r["file_name"]).stem
    return None


def lyric_summary(vid: str, root=None) -> str:
    """Why some rows have no lyric: only Whisper lines marked kept are used (handoff), and a song's
    intro, outro and instrumental breaks have none."""
    lines = load_lyric_lines(vid, root)
    kept = [l for l in lines if l.get("kept")]
    if not lines:
        return "no lyrics file"
    if not kept:
        return f"none of {len(lines)} Whisper lines kept (masked: wrong language detected), so no lyrics"
    return (f"{len(kept)} of {len(lines)} Whisper lines kept, from {to_tc(kept[0]['start'])} to "
            f"{to_tc(kept[-1]['end'])}; shots outside kept lines show no lyric")

CSS = """
:root { --ink:#1b1b1b; --muted:#666; --line:#d8d8d8; --bg:#ffffff; --head:#f4f4f2; --warn:#a33; }
* { box-sizing: border-box; }
body { margin: 0; padding: 24px; background: var(--bg); color: var(--ink);
       font: 13px/1.4 -apple-system, "Segoe UI", Roboto, Helvetica, Arial, sans-serif; }
header h1 { font-size: 20px; margin: 0 0 6px; }
header dl { display: grid; grid-template-columns: max-content 1fr; gap: 2px 14px; margin: 0 0 8px; }
header dt { color: var(--muted); } header dd { margin: 0; }
.warn { color: var(--warn); font-weight: 600; margin: 6px 0 14px; }
.scroll { overflow-x: auto; }                      /* narrow windows scroll the table, not the page */
table { border-collapse: collapse; width: 100%; min-width: 960px; }
th, td { border-bottom: 1px solid var(--line); padding: 6px 8px; vertical-align: top; text-align: left; }
th { background: var(--head); font-weight: 600; position: sticky; top: 0; }
td.n { font-variant-numeric: tabular-nums; white-space: nowrap; }
img { width: 180px; height: 180px; object-fit: contain; border: 1px solid var(--line); display: block; }
.tags b { color: var(--muted); font-weight: 500; }
.lyric { font-style: italic; color: var(--muted); max-width: 22ch; }
.none { font-style: normal; color: #aaa; font-size: 11px; }
@media print {
  .scroll { overflow: visible; } table { min-width: 0; }
  body { padding: 0; font-size: 10px; }
  th { position: static; }
  tr { break-inside: avoid; }
  img { width: 130px; height: 130px; }
}
"""


def _img_tag(path: Path) -> str:
    """Panel embedded as JPEG (quality 88): about 5x smaller than the PNG, no visible loss at sheet size.
    The full-quality PNGs stay in panels/."""
    if not path.exists():
        return "<em>no panel</em>"
    import io
    from PIL import Image
    buf = io.BytesIO()
    Image.open(path).convert("RGB").save(buf, "JPEG", quality=88)
    b64 = base64.b64encode(buf.getvalue()).decode()
    return f'<img alt="panel" src="data:image/jpeg;base64,{b64}">'


def build_sheet(sb: dict, out_dir: Path, versions: dict | None = None, pdf: bool = True, root=None) -> dict:
    """Writes storyboard.html (+ .pdf if weasyprint is installed), cutlist.csv and cutlist.edl."""
    out_dir = Path(out_dir)
    palettes = audio = None
    pj = out_dir / "prompts.jsonl"
    if pj.exists():
        recs = [json.loads(l) for l in pj.read_text(encoding="utf-8").splitlines() if l.strip()]
        palettes = [r["palette"] for r in recs]
        audio = [r.get("audio") for r in recs]
    rows = shot_rows(sb, palettes, audio)
    csv_p = write_csv(rows, out_dir / "cutlist.csv")
    edl_p = write_edl(sb, rows, out_dir / "cutlist.edl")

    render = {}
    if (out_dir / "render_log.json").exists():
        render = json.loads((out_dir / "render_log.json").read_text())
    v = versions or {}
    title = song_title(sb["video_id"], root)
    meta = [("Song", f"{title} ({sb['video_id']})" if title else sb["video_id"]), ("Genre", sb.get("genre", "?")),
            ("Shot timing", "copied from the real video (not generated)" if sb.get("timing") == "real"
             else f"Stage 1 sample {sb['stage1_sample']}" + (" (MOCK Stage 1)" if sb.get("stage1_mock") else "")),
            ("Stage 2 seed", sb["stage2_seed"]),
            ("Stage 2 checkpoint", sb.get("stage2_checkpoint", "")),
            ("Shots", len(sb["shots"])),
            ("Lyrics", lyric_summary(sb["video_id"], root)),
            ("Panels", f"{render.get('renderer', 'none')}, {render.get('steps', '-')} steps"),
            ("Prompts", f"{render.get('prompt_backend', '-')}, CLAP context: {render.get('clap_context', 'off')}"),
            # only models this sheet actually used: the prompt LLMs are listed in versions.yaml but optional
            ("Models", ", ".join(f"{k}: {val}" for k, val in v.get("models", {}).items()
                                 if not k.startswith("prompt_llm") or
                                 k == f"prompt_llm_{render.get('prompt_backend')}"))]
    warn = []
    if sb.get("stage1_mock"):
        warn.append("Shot timing and kinetics come from the MOCK Stage 1, not the trained model.")
    if sb.get("timing") == "real":
        warn.append("Shot timing and kinetics are copied from the real video, as no trained Stage 1 exists. "
                    "Shot content (scale, camera, tags) and panels are generated.")
    if render.get("renderer") == "placeholder":
        warn.append("Panels are placeholders (rendered without a GPU).")

    sid = sb.get("sample_id", f"s{sb['stage1_sample']}")

    def tags_html(t):
        return "<br>".join(f"<b>{html.escape(g)}:</b> {html.escape(', '.join(p))}" for g, p in t.items() if p) or "-"

    body = []
    for i, (r, s) in enumerate(zip(rows, sb["shots"])):
        body.append(
            "<tr>"
            f"<td class=n>{r['shot']}</td>"
            f"<td>{_img_tag(out_dir / 'panels' / f'panel_{i + 1:03d}.png')}</td>"
            f"<td class=n>{r['start_tc']}<br>{r['end_tc']}</td>"
            f"<td class=n>{float(r['duration_s']):.2f} s</td>"
            f"<td>{html.escape(r['section'])}</td>"
            f"<td>{html.escape(r['scale'])}</td>"
            f"<td>{r['camera']}</td>"
            f"<td class=tags>{tags_html(s['tags'])}</td>"
            f"<td>{html.escape(r['palette']) or '-'}</td>"
            f"<td class=tags>{html.escape(r['audio_clap']) or '-'}</td>"
            f"<td class=lyric>{html.escape(r['lyric']) or '<span class=none>no kept line</span>'}</td>"
            "</tr>")
    doc = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Storyboard {html.escape(sb['video_id'])} {html.escape(sid)}</title><style>{CSS}</style></head>
<body><header><h1>KineSync storyboard: {html.escape(title or sb['video_id'])}, sample {html.escape(sid)}</h1>
<dl>{''.join(f'<dt>{html.escape(str(k))}</dt><dd>{html.escape(str(val))}</dd>' for k, val in meta)}</dl>
{''.join(f'<p class=warn>{html.escape(w)}</p>' for w in warn)}
</header>
<div class=scroll><table><thead><tr><th>#</th><th>Panel</th><th>In / out (25 fps)</th><th>Duration</th><th>Section</th>
<th>Scale</th><th>Camera</th><th>Tags</th><th>Palette</th><th>Audio (CLAP)</th><th>Lyric</th></tr></thead>
<tbody>{''.join(body)}</tbody></table></div>
<p style="color:#666">Lyric text is for internal use only. Do not publish.</p>
</body></html>"""
    html_p = out_dir / "storyboard.html"
    html_p.write_text(doc, encoding="utf-8")
    out = {"html": str(html_p), "csv": str(csv_p), "edl": str(edl_p), "pdf": None}
    if pdf:
        try:
            from weasyprint import HTML
            HTML(str(html_p)).write_pdf(str(out_dir / "storyboard.pdf"))
            out["pdf"] = str(out_dir / "storyboard.pdf")
        except Exception:      # weasyprint missing or its system libraries missing: skip quietly
            pass
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True, help="output folder holding storyboard.json and panels/")
    a = ap.parse_args()
    d = Path(a.dir)
    sb = json.loads((d / "storyboard.json").read_text(encoding="utf-8"))
    print(build_sheet(sb, d, load_yaml(REPO_ROOT / "configs" / "versions.yaml")))


if __name__ == "__main__":
    main()
