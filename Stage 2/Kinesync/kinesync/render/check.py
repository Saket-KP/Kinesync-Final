"""Redraw check (handoff 6.3). A QUALITY FILTER, never an evaluation result.

SigLIP (not the CLIP model used for labelling) scores each panel against its tag phrases. Panels
below the threshold are redrawn with a new seed, up to `max_tries` draws in total; the best
draw is kept and every try is logged.

Set the threshold from a val run (20th percentile of first-draw scores):
    python -m kinesync.render.check --calibrate outputs/v0004_s0 outputs/v0005_s0 ...
which writes cache/redraw_threshold.json; `threshold: auto` in configs/render.yaml reads it.
"""
from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path

import numpy as np

from kinesync.common import data_root


def shot_texts(shot: dict, mode: str = "off") -> list[str]:
    """What the panel should visibly show: the same content phrases the prompt asked for (subject,
    action and setting tags, or CLAP's vocals and instruments, per prompt.clap_context). Lighting,
    style and CLAP mood are left out: a grey ink panel cannot show them, so they would score near 0
    however good the panel is. A shot with nothing drawable has nothing to check."""
    from kinesync.render.prompt import drawn_phrases
    return [f"a sketch of {p}" for p in drawn_phrases(shot, mode)]


class PanelChecker:
    def __init__(self, model_id: str, device: str | None = None, revision: str | None = None):
        import torch
        from transformers import AutoModel, AutoProcessor
        from transformers.utils import logging as hf_logging
        hf_logging.set_verbosity_error()          # SigLIP's config triggers harmless token-id notices
        hf_logging.disable_progress_bar()
        self.torch = torch
        self.dev = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = AutoModel.from_pretrained(model_id, revision=revision).to(self.dev).eval()
        self.proc = AutoProcessor.from_pretrained(model_id, revision=revision)

    def score(self, image, texts: list[str]) -> float:
        """Mean SigLIP match probability between the panel and each phrase."""
        inp = self.proc(text=texts, images=image, padding="max_length", return_tensors="pt").to(self.dev)
        with self.torch.no_grad():
            logits = self.model(**inp).logits_per_image[0]
        return float(self.torch.sigmoid(logits).mean())

    def close(self):
        del self.model
        gc.collect()
        if self.torch.cuda.is_available():
            self.torch.cuda.empty_cache()


def threshold_path(backend: str) -> Path:
    """One threshold per image model: SigLIP's raw scores depend on the drawing style (Sana-Sprint's panels
    score lower than SDXL-Turbo's for the same content), so a threshold set on one model's panels would
    redraw far too often, or never, on another's. SDXL-Turbo keeps the original file name."""
    name = "redraw_threshold.json" if backend == "sdxl-turbo" else f"redraw_threshold_{backend}.json"
    return data_root() / "cache" / name


def resolve_threshold(cfg: dict, backend: str = "sdxl-turbo") -> float | None:
    th = cfg.get("threshold", "auto")
    if th == "auto":
        p = threshold_path(backend)
        return json.loads(p.read_text())["threshold"] if p.exists() else None   # not calibrated: log only
    return None if th is None else float(th)


def redraw(renderer, checker, sb, prompts, images, seeds, cfg, vid, k, mode="off"):
    from kinesync.render.panels import shot_seed
    th = resolve_threshold(cfg, renderer.name)
    log = {"threshold": th, "shots": []}
    for i, shot in enumerate(sb["shots"][:len(images)]):      # --max_shots may draw only the first few
        texts = shot_texts(shot, mode)
        if not texts:                         # abstract panel: nothing to compare against
            log["shots"].append({"shot": i + 1, "tries": [], "kept_seed": seeds[i]})
            continue
        tries = [{"seed": seeds[i], "score": checker.score(images[i], texts)}]
        best_img, kept = images[i], seeds[i]
        while th is not None and max(t["score"] for t in tries) < th and len(tries) < cfg.get("max_tries", 3):
            s = shot_seed(vid, k, i, attempt=len(tries))
            img = renderer([prompts[i]], [s])[0]
            sc = checker.score(img, texts)
            tries.append({"seed": s, "score": sc})
            if sc >= max(t["score"] for t in tries[:-1]):
                best_img, kept = img, s
        images[i] = best_img
        log["shots"].append({"shot": i + 1, "tries": tries, "kept_seed": kept})
    first = [s["tries"][0]["score"] for s in log["shots"] if s["tries"]]
    log["first_draw_scores_p20"] = float(np.percentile(first, 20)) if first else None
    log["redrawn"] = sum(len(s["tries"]) > 1 for s in log["shots"])
    return images, log


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--calibrate", nargs="+", required=True, help="output folders of a VAL render run")
    a = ap.parse_args()
    scores, backends = [], set()
    for d in a.calibrate:
        lg = json.load(open(Path(d) / "render_log.json"))
        if lg.get("check"):
            scores += [s["tries"][0]["score"] for s in lg["check"]["shots"] if s["tries"]]
            backends.add(lg["renderer"])
    if len(backends) != 1:
        raise SystemExit(f"the folders mix image models {sorted(backends)}; calibrate one model at a time")
    backend = backends.pop()
    if not scores:
        raise SystemExit("no check scores found; render the val run with check.enabled and threshold null first")
    th = float(np.percentile(scores, 20))
    p = threshold_path(backend)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps({"threshold": th, "backend": backend, "n_panels": len(scores), "source": a.calibrate},
                            indent=1))
    print(f"{backend}: redraw threshold = {th:.4f} (20th pct of {len(scores)} first-draw scores) -> {p}")


if __name__ == "__main__":
    main()
