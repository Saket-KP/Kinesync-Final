"""Contact sheet: every panel of one storyboard on one PNG, numbered, "C" where CLAP supplied the content."""
from __future__ import annotations

import json
from pathlib import Path

from kinesync.render.prompt import drawn_phrases


def contact_sheet(out_dir: Path, cols: int = 10, size: int = 200) -> Path:
    from PIL import Image, ImageDraw
    out_dir = Path(out_dir)
    panels = sorted((out_dir / "panels").glob("panel_*.png"))
    recs = [json.loads(l) for l in (out_dir / "prompts.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
    sb = json.loads((out_dir / "storyboard.json").read_text(encoding="utf-8"))
    rows = max(1, (len(panels) + cols - 1) // cols)
    sheet = Image.new("RGB", (cols * size, rows * size), "white")
    draw = ImageDraw.Draw(sheet)
    for i, p in enumerate(panels):
        x, y = (i % cols) * size, (i // cols) * size
        sheet.paste(Image.open(p).resize((size, size)), (x, y))
        shot = {**sb["shots"][i], "audio": recs[i].get("audio")} if i < len(recs) else sb["shots"][i]
        mode = recs[i].get("clap_context", "off") if i < len(recs) else "off"
        tagged = any(shot["tags"].get(g) for g in ("subject", "action", "setting"))
        from_clap = mode != "off" and bool(drawn_phrases(shot, mode)) and (mode == "only" or not tagged)
        label = f"{i + 1}{' C' if from_clap else ''}"
        draw.rectangle([x, y, x + 8 + 7 * len(label), y + 14], fill="white")
        draw.text((x + 3, y + 1), label, fill="red")
    path = out_dir / "contact_sheet.png"
    sheet.save(path)
    return path
