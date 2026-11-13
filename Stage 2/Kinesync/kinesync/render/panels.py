"""Panel renderer (handoff 6.2): tags -> prompt -> SDXL-Turbo -> one 512x512 panel per shot.

    python -m kinesync.render.panels --storyboard stage2/v0012_s0_storyboard.json
    python -m kinesync.render.panels --storyboard ... --max_shots 12 --steps 4

GPU memory, in order, one thing at a time:
    prompt LLM (if any) -> freed
    text encoders       -> every prompt encoded once -> freed
    UNet + VAE          -> panels drawn (offloaded in pieces if the GPU is small)
So the same code runs on the 16 GB box and on a 4 GB laptop GPU (about 4 s per panel, 2 steps).

Panels are saved as they are drawn and listed in panels/manifest.json, so an interrupted run picks
up where it stopped. A panel is reused only if its prompt, seed, steps, size, model and renderer
all still match, so stale panels (or old placeholders) are never kept by accident.
--placeholder draws text placeholders instead, which needs no GPU; it must be asked for.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import textwrap
import time
from pathlib import Path

from kinesync.common import REPO_ROOT, load_yaml

MANIFEST = "manifest.json"


def shot_seed(vid: str, sample: int, shot: int, attempt: int = 0) -> int:
    """Stable across runs and machines (Python's hash() is salted per process)."""
    h = hashlib.sha256(f"{vid}|{sample}|{shot}|{attempt}".encode()).digest()
    return int.from_bytes(h[:4], "little") & 0x7FFFFFFF


def full_prompt(prompt: str, cfg: dict) -> str:
    """Style goes around the content. Words early in a prompt weigh more, so a suffix alone leaves
    the panels in colour. See configs/render.yaml for why these exact words."""
    return cfg.get("style_prefix", "") + prompt + cfg.get("style_suffix", "")


def pick_offload(cfg: dict, vram_gib: float) -> str:
    """How to fit the UNet (4.78 GiB in fp16) on the GPU.
    none = everything on the GPU; model = one whole model at a time;
    sequential = one layer at a time (slowest, but peaks under 1 GiB)."""
    choice = cfg.get("offload", "auto")
    if choice != "auto":
        return choice
    if vram_gib >= 10:
        return "none"
    return "model" if vram_gib >= 7 else "sequential"


class SDXLTurbo:
    name = "sdxl-turbo"

    def __init__(self, cfg: dict):
        import torch
        from diffusers.utils import logging as diffusers_logging
        diffusers_logging.set_verbosity_error()      # hides a repeated VAE upcast notice per panel
        self.torch = torch
        self.cfg = cfg
        self.vram_gib = torch.cuda.get_device_properties(0).total_memory / 2**30
        self.mode = pick_offload(cfg, self.vram_gib)
        self.batch_size = 1 if self.mode == "sequential" else cfg.get("batch_size", 4)
        self.embeds = {}                             # full prompt -> (prompt_embeds, pooled), on CPU
        self.pipe = None
        torch.cuda.reset_peak_memory_stats()

    def _load(self, **parts):
        from diffusers import AutoPipelineForText2Image
        return AutoPipelineForText2Image.from_pretrained(
            self.cfg["model_id"], revision=self.cfg.get("revision"),
            torch_dtype=self.torch.float16, variant="fp16", **parts)

    def _free(self):
        gc.collect()
        self.torch.cuda.empty_cache()

    def prepare(self, prompts: list[str]) -> None:
        """Encode every new prompt once, with the text encoders alone, then free them (1.5 GiB)."""
        missing = sorted({full_prompt(p, self.cfg) for p in prompts} - self.embeds.keys())
        if not missing:
            return
        te = self._load(unet=None, vae=None).to("cuda")
        for i in range(0, len(missing), 16):
            batch = missing[i:i + 16]
            with self.torch.no_grad():
                pe, _, pooled, _ = te.encode_prompt(batch, device="cuda", num_images_per_prompt=1,
                                                    do_classifier_free_guidance=False)
            for j, text in enumerate(batch):
                self.embeds[text] = (pe[j].cpu(), pooled[j].cpu())
        del te
        self._free()

    def _denoiser(self):
        if self.pipe is None:
            pipe = self._load(text_encoder=None, text_encoder_2=None, tokenizer=None, tokenizer_2=None)
            if self.mode == "sequential":
                pipe.enable_sequential_cpu_offload()
            elif self.mode == "model":
                pipe.enable_model_cpu_offload()
            else:
                pipe.to("cuda")
            pipe.set_progress_bar_config(disable=True)
            self.pipe = pipe
        return self.pipe

    def __call__(self, prompts: list[str], seeds: list[int]):
        """One image per (prompt, seed). Seeds use CPU generators: the starting noise is then the
        same on any GPU and any batch size."""
        self.prepare(prompts)
        pipe = self._denoiser()
        stack = self.torch.stack
        imgs = []
        for i in range(0, len(prompts), self.batch_size):
            emb = [self.embeds[full_prompt(p, self.cfg)] for p in prompts[i:i + self.batch_size]]
            gens = [self.torch.Generator("cpu").manual_seed(s) for s in seeds[i:i + self.batch_size]]
            out = pipe(prompt_embeds=stack([e[0] for e in emb]).to("cuda"),
                       pooled_prompt_embeds=stack([e[1] for e in emb]).to("cuda"),
                       num_inference_steps=self.cfg["steps"], guidance_scale=0.0,
                       width=self.cfg["size"], height=self.cfg["size"], generator=gens)
            imgs += out.images
        if self.cfg.get("grayscale", True):          # warm tags keep some colour otherwise
            imgs = [im.convert("L").convert("RGB") for im in imgs]
        return imgs

    def peak_gib(self) -> float:
        return self.torch.cuda.max_memory_allocated() / 2**30

    def close(self):
        self.pipe = None
        self.embeds = {}
        self._free()


class SanaSprint(SDXLTurbo):
    """NVIDIA Sana-Sprint (1-4 steps, 1024 px native, Gemma-2-2B text encoder). Same two-phase memory plan
    as SDXL-Turbo. On a 4 GB GPU the text encoder (about 5 GB) runs layer by layer, and the 3.2 GB image
    model runs whole ("model" offload, peak 3.15 GiB measured). It must run at 1024 px: at 512 px its
    figures smear (bake-off in results/README.md)."""
    name = "sana-sprint"

    def __init__(self, cfg: dict):
        super().__init__(cfg)
        self.mode = cfg.get("offload", "auto")
        if self.mode == "auto":
            self.mode = "none" if self.vram_gib >= 10 else "model" if self.vram_gib >= 3.8 else "sequential"
        self.batch_size = 1 if self.mode != "none" else cfg.get("batch_size", 4)

    def _load(self, **parts):
        from diffusers import SanaSprintPipeline
        return SanaSprintPipeline.from_pretrained(self.cfg["model_id"], revision=self.cfg.get("revision"),
                                                  torch_dtype=self.torch.bfloat16, **parts)

    def prepare(self, prompts: list[str]) -> None:
        missing = sorted({full_prompt(p, self.cfg) for p in prompts} - self.embeds.keys())
        if not missing:
            return
        te = self._load(transformer=None, vae=None)
        if self.vram_gib >= 8:
            te.to("cuda")
        else:
            te.enable_sequential_cpu_offload()             # Gemma-2-2B does not fit a 4 GB GPU whole
        for text in missing:
            with self.torch.no_grad():
                pe, mask = te.encode_prompt(text, num_images_per_prompt=1, device="cuda")
            self.embeds[text] = (pe[0].cpu(), mask[0].cpu())
        del te
        self._free()

    def _denoiser(self):
        if self.pipe is None:
            pipe = self._load(text_encoder=None, tokenizer=None)
            if self.mode == "sequential":
                pipe.enable_sequential_cpu_offload()
            elif self.mode == "model":
                pipe.enable_model_cpu_offload()
            else:
                pipe.to("cuda")
            pipe.set_progress_bar_config(disable=True)
            self.pipe = pipe
        return self.pipe

    def __call__(self, prompts: list[str], seeds: list[int]):
        self.prepare(prompts)
        pipe = self._denoiser()
        imgs = []
        for p, s in zip(prompts, seeds):                   # prompts differ in token length: one at a time
            pe, mask = self.embeds[full_prompt(p, self.cfg)]
            out = pipe(prompt_embeds=pe[None].to("cuda"), prompt_attention_mask=mask[None].to("cuda"),
                       num_inference_steps=self.cfg["steps"], width=self.cfg["size"], height=self.cfg["size"],
                       use_resolution_binning=False, generator=self.torch.Generator("cpu").manual_seed(s))
            imgs.append(out.images[0])
        size = self.cfg.get("save_size", 512)              # stored like SDXL-Turbo's, so sheets stay small
        imgs = [im.resize((size, size)) for im in imgs]
        if self.cfg.get("grayscale", True):
            imgs = [im.convert("L").convert("RGB") for im in imgs]
        return imgs


BACKENDS = {"sdxl-turbo": SDXLTurbo, "sana-sprint": SanaSprint}


def backend_cfg(pc: dict) -> dict:
    """panels config for the chosen model: shared settings, then that model's own block (model_id, revision,
    size). The default backend's settings sit at the top level for older configs."""
    name = pc.get("backend", "sdxl-turbo")
    return {**pc, **pc.get("models", {}).get(name, {}), "backend": name}


class Placeholder:
    """Stand-in that needs no GPU: a white panel with the prompt on it, clearly marked."""
    name = "placeholder"
    mode = "none"
    batch_size = 8

    def __init__(self, cfg: dict):
        self.size = cfg.get("size", 512)

    def prepare(self, prompts):
        pass

    def __call__(self, prompts, seeds):
        from PIL import Image, ImageDraw
        imgs = []
        for p, s in zip(prompts, seeds):
            im = Image.new("RGB", (self.size, self.size), "white")
            d = ImageDraw.Draw(im)
            d.rectangle([4, 4, self.size - 5, self.size - 5], outline="black", width=3)
            d.text((16, 14), "PLACEHOLDER (no GPU render)", fill="red")
            y = 44
            for line in textwrap.wrap(p, 52):
                d.text((16, y), line, fill="black")
                y += 16
            d.text((16, self.size - 28), f"seed {s}", fill="gray")
            imgs.append(im)
        return imgs

    def peak_gib(self):
        return None

    def close(self):
        pass


def gpu_available() -> bool:
    try:
        import torch
        return torch.cuda.is_available()
    except ImportError:
        return False


def _panel_key(renderer, cfg: dict, prompt: str, seed: int) -> dict:
    """Everything that decides what a panel looks like. Reuse needs an exact match."""
    return {"renderer": renderer.name, "model": cfg.get("model_id"), "revision": cfg.get("revision"),
            "steps": cfg.get("steps"), "size": cfg.get("size"), "grayscale": cfg.get("grayscale", True),
            "prompt": full_prompt(prompt, cfg), "seed": seed}


def _load_manifest(path: Path) -> dict:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError):
        return {}


def _write_manifest(path: Path, manifest: dict) -> None:
    tmp = path.with_suffix(".tmp")               # never leave a half-written manifest after a crash
    tmp.write_text(json.dumps(manifest, indent=0), encoding="utf-8")
    tmp.replace(path)


def render_storyboard(sb: dict, out_dir: Path, cfg: dict, placeholder: bool = False, root=None,
                      max_shots: int | None = None, force: bool = False) -> dict:
    """Prompts -> panels -> (redraw check) for one storyboard. Writes prompts.jsonl, panels/ and
    render_log.json in out_dir, and returns the log. max_shots draws only the first N panels."""
    from PIL import Image
    from kinesync.render.prompt import PromptWriter

    if not placeholder and not gpu_available():
        raise RuntimeError("No CUDA GPU visible to this Python, so SDXL-Turbo cannot run. Use the project "
                           "environment (.venv/Scripts/python.exe), or pass --placeholder to draw text "
                           "placeholders instead.")
    out_dir = Path(out_dir)
    panel_dir = out_dir / "panels"
    panel_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    vid, k = sb["video_id"], sb["stage1_sample"]
    if sb.get("timing") == "real":     # real-timing samples get their own panel seeds (s0 keeps its old ones)
        k = sb["sample_id"]
    pc = backend_cfg(cfg["panels"])

    mode = cfg["prompt"].get("clap_context", "off")
    shots = sb["shots"]
    if mode != "off":                                  # describe each shot's audio in words (CPU, cached)
        from kinesync.render.audio_context import AudioDescriber
        audio = AudioDescriber(cfg["clap"], root).describe(vid, [s["start"] for s in shots],
                                                           [s["end"] for s in shots])
        shots = [{**s, "audio": a} for s, a in zip(shots, audio)]
    writer = PromptWriter(cfg["prompt"], root)
    prompts, recs = [], []
    for i, shot in enumerate(shots):
        p, rec = writer.write(shot)
        prompts.append(p)
        recs.append({"shot": i + 1, **rec, "clap_context": mode, "audio": shot.get("audio"),
                     "full_prompt": full_prompt(p, pc)})
    writer.close()                                     # free the LLM before the image model
    with open(out_dir / "prompts.jsonl", "w", encoding="utf-8") as f:
        for r in recs:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    n = len(prompts) if max_shots is None else min(max_shots, len(prompts))
    seeds = [shot_seed(vid, k, i) for i in range(n)]
    renderer = Placeholder(pc) if placeholder else BACKENDS[pc["backend"]](pc)
    path = lambda i: panel_dir / f"panel_{i + 1:03d}.png"

    # Reuse what an earlier run already drew; draw the rest, saving as we go.
    man_path = panel_dir / MANIFEST
    manifest = {} if force else _load_manifest(man_path)
    images, kept_seeds, todo = {}, list(seeds), []
    for i in range(n):
        entry = manifest.get(str(i + 1))
        if entry and entry.get("key") == _panel_key(renderer, pc, prompts[i], seeds[i]) and path(i).exists():
            images[i] = Image.open(path(i)).convert("RGB")
            kept_seeds[i] = entry.get("kept_seed", seeds[i])
        else:
            todo.append(i)
    renderer.prepare([prompts[i] for i in todo])
    t_draw = time.time()
    for a in range(0, len(todo), renderer.batch_size):
        chunk = todo[a:a + renderer.batch_size]
        for i, im in zip(chunk, renderer([prompts[i] for i in chunk], [seeds[i] for i in chunk])):
            im.save(path(i))
            images[i] = im
            manifest[str(i + 1)] = {"key": _panel_key(renderer, pc, prompts[i], seeds[i]), "kept_seed": seeds[i]}
        _write_manifest(man_path, manifest)
        done = a + len(chunk)
        if not placeholder and (done % 10 < len(chunk) or done == len(todo)):
            print(f"  {vid}_s{k}: {done}/{len(todo)} panels, {(time.time() - t_draw) / done:.1f} s/panel", flush=True)
    seconds_per_panel = (time.time() - t_draw) / len(todo) if todo else None

    check_log = None
    if not placeholder and n and cfg["check"].get("enabled", True):
        from kinesync.render.check import PanelChecker, redraw
        # Under 8 GB, SigLIP goes on the CPU: a redraw needs the image model on the GPU at the same time, and
        # Sana-Sprint (3.2 GB, "model" offload) plus SigLIP ran a 4 GB GPU out of memory. SigLIP is fast on a CPU.
        device = "cuda" if getattr(renderer, "vram_gib", 0) >= 8 else "cpu"
        checker = PanelChecker(cfg["check"]["model_id"], device, cfg["check"].get("revision"))
        ordered = [images[i] for i in range(n)]
        ordered, check_log = redraw(renderer, checker, {**sb, "shots": shots}, prompts[:n], ordered,
                                    kept_seeds, cfg["check"], vid, k, mode)
        checker.close()
        for i, entry in enumerate(check_log["shots"]):
            if entry["kept_seed"] != kept_seeds[i]:      # a redraw won: replace the saved panel
                ordered[i].save(path(i))
                manifest[str(i + 1)]["kept_seed"] = entry["kept_seed"]
        _write_manifest(man_path, manifest)
    peak = renderer.peak_gib() if todo else None     # nothing drawn: no meaningful GPU peak
    offload, name = renderer.mode, renderer.name
    renderer.close()

    log = {"video_id": vid, "stage1_sample": k, "n_shots": len(prompts), "n_panels": n,
           "panels_drawn": len(todo), "panels_reused": n - len(todo),
           "renderer": name, "offload": offload, "steps": pc["steps"], "peak_gpu_gib": peak,
           "seconds": round(time.time() - t0, 1),
           "seconds_per_panel": round(seconds_per_panel, 2) if seconds_per_panel else None,
           "prompt_backend": cfg["prompt"]["backend"], "clap_context": mode,
           "llm_fallbacks": sum(r["used"] == "template" for r in recs) if cfg["prompt"]["backend"] != "template" else 0,
           "check": check_log}
    with open(out_dir / "render_log.json", "w") as f:
        json.dump(log, f, indent=1)
    if peak is not None:
        print(f"peak GPU memory {peak:.2f} GiB ({'OK' if peak < 16 else 'OVER'} the 16 GB budget), offload: {offload}")
    return log


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--storyboard", required=True)
    ap.add_argument("--out")
    ap.add_argument("--config", default=str(REPO_ROOT / "configs" / "render.yaml"))
    ap.add_argument("--placeholder", action="store_true", help="draw text placeholders; no GPU needed")
    ap.add_argument("--max_shots", type=int, help="draw only the first N panels")
    ap.add_argument("--steps", type=int, help="override the configured steps (1 to 4)")
    ap.add_argument("--image_model", choices=sorted(BACKENDS), help="override panels.backend")
    ap.add_argument("--force", action="store_true", help="redraw panels even if already drawn")
    a = ap.parse_args()
    cfg = load_yaml(a.config)
    if a.steps:
        cfg["panels"]["steps"] = a.steps
    if a.image_model:
        cfg["panels"]["backend"] = a.image_model
    with open(a.storyboard, encoding="utf-8") as f:
        sb = json.load(f)
    sid = sb.get("sample_id", f"s{sb['stage1_sample']}")
    out = Path(a.out) if a.out else REPO_ROOT / "outputs" / f"{sb['video_id']}_{sid}"
    log = render_storyboard(sb, out, cfg, a.placeholder, max_shots=a.max_shots, force=a.force)
    print(json.dumps({k: v for k, v in log.items() if k != "check"}))


if __name__ == "__main__":
    main()
