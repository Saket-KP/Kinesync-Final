"""One command: song -> Stage 1 -> Stage 2 -> panels -> sheet + cut list (handoff section 8).

    python -m kinesync.pipeline --video_id v0012 --n_samples 3 --out outputs/
    python -m kinesync.pipeline --audio path/to/song.mp3 --genre pop --n_samples 3 --out outputs/

--video_id  a song already in features/. By default its shot timing and kinetics are copied from the
            real video (--timing real): with no trained Stage 1, that is the most accurate timing
            available, and the sheet says it is not generated. --timing stage1 uses Stage 1 instead.
--audio     needs the team's audio feature extractor (librosa features, sections, the same CLAP
            checkpoint, Whisper lyrics). It is not in this repo, so this mode stops and says so.
Stage 1 is MOCKED (kinesync/stage1/mock.py) until the real sampler exists; existing Stage 1
files for a song are reused, so dropping in real Stage 1 outputs needs no change here.
"""
from __future__ import annotations

import argparse
import importlib
import importlib.util
import json
import shutil
import sys
import time
from pathlib import Path

from kinesync.common import REPO_ROOT, data_root, load_yaml

EXTRACTOR_MODULE = "kinesync.features.extract"   # expected home of the team's extractor


def run_song(vid: str, n_samples: int, out_root: Path, ckpt: Path, genre: str | None = None,
             seed: int = 1234, stage1_mode: str = "perturb", placeholder: bool = False,
             regen_stage1: bool = False, root=None, timing: str = "stage1",
             samples: list[int] | None = None, render_overrides: dict | None = None) -> list[dict]:
    from kinesync.output.contact import contact_sheet
    from kinesync.output.sheet import build_sheet
    from kinesync.render.panels import render_storyboard
    from kinesync.stage1.mock import write_mock
    from kinesync.stage2.sample import storyboard, write_storyboard
    from kinesync.stage2.train import load_any

    root = Path(root or data_root())
    render_cfg = load_yaml(REPO_ROOT / "configs" / "render.yaml")
    for section, values in (render_overrides or {}).items():      # e.g. {"panels": {"steps": 4}}
        render_cfg[section].update(values)
    sample_cfg = load_yaml(REPO_ROOT / "configs" / "sampling.yaml")
    versions = load_yaml(REPO_ROOT / "configs" / "versions.yaml")
    results = []
    ks = list(samples) if samples is not None else list(range(n_samples))

    # Stage 1 (mock unless real outputs already exist); skipped when the timing is the real video's
    print(f"[stage1] {'shot timing from the real video' if timing == 'real' else 'shot timing from Stage 1'}",
          flush=True)
    for k in (ks if timing == "stage1" else []):
        if regen_stage1 or not (root / "stage1" / f"{vid}_s{k}.npz").exists():
            write_mock(vid, k, stage1_mode, root=root)

    # Stage 2: small model on CPU; freed before rendering
    print("[stage2] sampling shot content", flush=True)
    model, ck = load_any(ckpt, root=root)
    ck["path"] = str(Path(ckpt).relative_to(REPO_ROOT)) if Path(ckpt).is_relative_to(REPO_ROOT) else str(ckpt)
    boards = []
    for k in ks:
        sb = storyboard(model, ck, vid, k, seed, sample_cfg, root, genre, timing)
        boards.append((sb, write_storyboard(sb, root)))
    del model

    # Rendering + output, one sample at a time
    for sb, sb_path in boards:
        out = out_root / f"{vid}_{sb['sample_id']}"
        out.mkdir(parents=True, exist_ok=True)
        shutil.copy(sb_path, out / "storyboard.json")
        print(f"[render] {sb['sample_id']}: {len(sb['shots'])} panels", flush=True)
        log = render_storyboard(sb, out, render_cfg, placeholder, root)
        files = build_sheet(sb, out, versions, root=root)
        files["contact_sheet"] = str(contact_sheet(out))
        results.append({"sample_id": sb["sample_id"], "out_dir": str(out), "storyboard": str(sb_path), **files,
                        "renderer": log["renderer"], "peak_gpu_gib": log["peak_gpu_gib"],
                        "n_shots": len(sb["shots"])})
    return results


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--video_id")
    src.add_argument("--audio")
    ap.add_argument("--genre")
    ap.add_argument("--title", help="with --audio: the song's name for the sheet (default: the file name)")
    ap.add_argument("--n_samples", type=int, default=3)
    ap.add_argument("--samples", type=int, nargs="+", help="render only these sample numbers, e.g. --samples 1")
    ap.add_argument("--out", default="outputs/")
    ap.add_argument("--ckpt", default=str(REPO_ROOT / "configs" / "stage2" / "final_ensemble.yaml"),
                    help="Stage 2 model: a .pt checkpoint or an ensemble .yaml (default: the final ensemble)")
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--timing", choices=["real", "stage1"], default=None,
                    help="real: cuts and kinetics from the real video (default for --video_id); stage1: from Stage 1")
    ap.add_argument("--stage1_mode", choices=["perturb", "audio"], default="perturb")
    ap.add_argument("--regen_stage1", action="store_true")
    ap.add_argument("--placeholder", action="store_true", help="skip SDXL; draw text placeholders")
    ap.add_argument("--steps", type=int, choices=[1, 2, 3, 4], help="SDXL-Turbo steps (default from render.yaml)")
    ap.add_argument("--clap_context", choices=["off", "add", "only"], help="override prompt.clap_context")
    ap.add_argument("--image_model", choices=["sdxl-turbo", "sana-sprint"], help="override panels.backend")
    a = ap.parse_args()
    t0 = time.time()

    if a.audio:
        try:
            found = importlib.util.find_spec(EXTRACTOR_MODULE) is not None
        except ModuleNotFoundError:          # parent package kinesync.features does not exist
            found = False
        if not found:
            sys.exit("STOP: --audio needs the audio feature extractor (onset, flux, chroma, beats, sections, "
                     "CLAP with the same checkpoint as the training data, Whisper lyrics). It is not in this "
                     f"repo (expected module {EXTRACTOR_MODULE}). Use --video_id for songs in features/.")
        ext = importlib.import_module(EXTRACTOR_MODULE)
        # contract: writes features/{vid}.npz (+ lyrics) and returns vid
        vid = ext.extract(a.audio, genre=a.genre, log=lambda m: print(m, flush=True), title=a.title)
    else:
        vid = a.video_id

    timing = a.timing or ("real" if a.video_id else "stage1")   # a new song has no real video
    overrides = {}
    if a.steps:
        overrides["panels"] = {"steps": a.steps}
    if a.clap_context:
        overrides["prompt"] = {"clap_context": a.clap_context}
    if a.image_model:
        overrides.setdefault("panels", {})["backend"] = a.image_model
    res = run_song(vid, a.n_samples, Path(a.out), Path(a.ckpt), a.genre, a.seed, a.stage1_mode,
                   a.placeholder, a.regen_stage1, timing=timing, samples=a.samples, render_overrides=overrides)
    for r in res:
        print("[result] " + json.dumps(r), flush=True)
    print(f"total time {time.time() - t0:.1f} s")


if __name__ == "__main__":
    main()
