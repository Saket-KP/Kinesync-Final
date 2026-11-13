"""Prompt writer (handoff 6.1): sampled shot attributes -> one line-art prompt sentence.

Backends (configs/render.yaml, `prompt.backend`):
  template  deterministic string built only from the attributes. Always works; the fallback.
  local     small instruction-tuned model via transformers (model ID in configs/versions.yaml).
  api       chat API; key read from an environment variable, never stored.
LLM output is validated; on failure the template prompt is used and the reason is logged.
"""
from __future__ import annotations

import gc
import json
import os
import re
import urllib.request
from pathlib import Path

import numpy as np

from kinesync.common import (KIN_KEYS, data_root, kin_mean_std, load_features, load_shots,
                             load_splits, videos_in)

SYSTEM_PROMPT = (
    "You write image prompts for a sketch artist. Write exactly ONE sentence describing a single "
    "black-ink line-art frame. Describe only the elements you are given (shot scale, camera, tags, "
    "audio context, palette). Do not add any objects, people, text, logos or brand names that are not given. "
    "Never name a real person or artist. Do not quote the lyric; you may only let it set the mood. "
    "Never use the word storyboard. Reply with the sentence only."
)
# "camera" in an image prompt draws an actual camera, so the movement is phrased as a kind of shot.
CAMERA = {False: "static shot", True: "moving shot"}
# Shots with no subject, action or setting (about 28% of sampled shots) still need a panel. This phrase
# renders as plain ink strokes; it adds no object or person, unlike "scene" or "empty space".
NO_CONTENT = "abstract light and shadow"
BRIGHT_WORDS = ("low-key", "even", "high-key")
SAT_WORDS = ("muted", "natural", "vivid")
TEMPLATE_WORDS = {"static", "moving", "shot", "in", "palette", "mood"} | set(NO_CONTENT.split())
# How CLAP audio context enters the prompt (configs/render.yaml, prompt.clap_context):
#   off   tags only (the handoff's design)
#   add   tags, plus CLAP mood/energy on every shot; a shot with no subject, action or setting tag
#         takes CLAP's vocals/instruments as its content instead of abstract strokes
#   only  CLAP context alone: no tags at all (scale, camera and palette words stay)
CLAP_MODES = ("off", "add", "only")


# --------------------------------------------------------------------------- palette

def palette_thresholds(root=None) -> dict:
    """Tertiles of per-shot mean brightness and saturation over TRAIN shots (real video).
    Cached in cache/palette_tertiles.json."""
    root = Path(root or data_root())
    p = root / "cache" / "palette_tertiles.json"
    if p.exists():
        return json.loads(p.read_text())
    from kinesync.stage2.features import KinSource
    vals = []
    for vid in videos_in("train", root):
        shots = load_shots(vid, root)
        kin = KinSource.from_features(load_features(vid, root))
        vals.append(kin.span_means(np.array([s["start"] for s in shots]), np.array([s["end"] for s in shots])))
    v = np.concatenate(vals)
    j_b, j_s = KIN_KEYS.index("brightness"), KIN_KEYS.index("saturation")
    th = {"brightness": np.percentile(v[:, j_b], [100 / 3, 200 / 3]).tolist(),
          "saturation": np.percentile(v[:, j_s], [100 / 3, 200 / 3]).tolist(),
          "source": "train-shot tertiles of real video kinetics"}
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(th, indent=1))
    return th


def palette_words(kin: dict, th: dict) -> list[str]:
    """Stage 1 kinetics are the only colour signal (no hue feature)."""
    b = BRIGHT_WORDS[int(np.searchsorted(th["brightness"], kin["brightness"]))]
    s = SAT_WORDS[int(np.searchsorted(th["saturation"], kin["saturation"]))]
    return [b, s]


# --------------------------------------------------------------------------- template backend

def drawn_phrases(shot: dict, mode: str = "off") -> list[str]:
    """What the panel should show: subject, action and setting tags, or CLAP's vocals and
    instruments (see CLAP_MODES). Shared by the prompt and the redraw check."""
    t = shot["tags"]
    a = shot.get("audio") or {}
    tagged = t.get("subject", []) + t.get("action", []) + t.get("setting", [])
    heard = a.get("vocals", []) + a.get("instruments", [])
    if mode == "only":
        return heard
    if mode == "add":
        return tagged or heard
    return tagged


def atmosphere(shot: dict, mode: str = "off") -> list[str]:
    """CLAP mood and energy as atmosphere words, e.g. ["melancholic mood", "stillness"]."""
    a = shot.get("audio") or {}
    if mode == "off" or not a:
        return []
    return [f"{m} mood" for m in a.get("mood", [])] + a.get("energy", [])


def template_prompt(shot: dict, palette: list[str], mode: str = "off") -> str:
    t = shot["tags"]
    settings = set(t.get("setting", [])) if mode != "only" else set()
    content = [f"in {p}" if p in settings else p for p in drawn_phrases(shot, mode)] or [NO_CONTENT]
    head = f"{shot['scale'].capitalize()}, {CAMERA[bool(shot['moving'])]}: {content[0]}"
    extras = [] if mode == "only" else t.get("lighting", []) + t.get("style", [])
    parts = [head] + content[1:] + extras + atmosphere(shot, mode)
    parts.append(f"{palette[0]} {palette[1]} palette")
    return ", ".join(parts) + "."


def allowed_words(shot: dict, palette: list[str]) -> set[str]:
    words = set(TEMPLATE_WORDS) | set(palette)
    words |= set(_tokens(shot["scale"]))
    groups = list(shot["tags"].values()) + [v for k, v in (shot.get("audio") or {}).items() if k != "heard"]
    for phrases in groups:
        for p in phrases:
            words |= set(_tokens(p))
    return words


def _tokens(s: str) -> list[str]:
    return re.findall(r"[a-z0-9]+(?:-[a-z0-9]+)*", s.lower())


# --------------------------------------------------------------------------- validator

def _person_names(root=None) -> list[str]:
    try:
        return sorted({a.lower() for a in load_splits(root)["artist"].astype(str)}, key=len, reverse=True)
    except Exception:
        return []


def validate(text: str, shot: dict, palette: list[str], names: list[str]) -> list[str]:
    """Reasons to reject an LLM prompt (empty list = accept)."""
    reasons = []
    if not text.strip():
        return ["empty"]
    words = text.split()
    if len(words) > 60:
        reasons.append(f"{len(words)} words > 60")
    if "storyboard" in text.lower():
        reasons.append("says 'storyboard', which draws a grid of panels")
    lyric = shot.get("lyric") or ""
    lt, ot = _tokens(lyric), " ".join(_tokens(text))
    for i in range(len(lt) - 6):                         # any 7 consecutive lyric words
        if " " + " ".join(lt[i:i + 7]) + " " in f" {ot} ":
            reasons.append("quotes more than 6 words of the lyric")
            break
    low = f" {text.lower()} "
    for n in names:
        if len(n) > 3 and re.search(r"(?<![a-z])" + re.escape(n) + r"(?![a-z])", low):
            reasons.append(f"names an artist ({n})")
            break
    # heuristic for other real names: two capitalised words in a row, not at the start
    allowed = allowed_words(shot, palette)
    for m in re.finditer(r"(?<!^)(?<![.!?]\s)\b([A-Z][a-z]+)\s+([A-Z][a-z]+)\b", text.strip()):
        if not {m.group(1).lower(), m.group(2).lower()} <= allowed:
            reasons.append(f"possible person name '{m.group(0)}'")
            break
    return reasons


# --------------------------------------------------------------------------- LLM backends

def _user_message(shot: dict, palette: list[str], mode: str = "off") -> str:
    tags = [] if mode == "only" else [p for ps in shot["tags"].values() for p in ps]
    msg = {"shot_scale": shot["scale"], "camera": CAMERA[bool(shot["moving"])],
           "tags": tags, "palette": palette, "lyric_line": shot.get("lyric")}
    if mode != "off":
        msg["audio_context"] = {"content": drawn_phrases(shot, mode) if not tags else [],
                                "atmosphere": atmosphere(shot, mode)}
    return json.dumps(msg)


class LocalLLM:
    def __init__(self, model_id: str, max_new_tokens: int = 80, load_in_4bit: bool = False):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        self.torch = torch
        self.tok = AutoTokenizer.from_pretrained(model_id)
        kw = {"torch_dtype": torch.float16, "device_map": "cuda"} if torch.cuda.is_available() else {}
        if load_in_4bit:
            from transformers import BitsAndBytesConfig
            kw["quantization_config"] = BitsAndBytesConfig(load_in_4bit=True)
        self.model = AutoModelForCausalLM.from_pretrained(model_id, **kw)
        self.max_new_tokens = max_new_tokens

    def __call__(self, system: str, user: str) -> str:
        msgs = [{"role": "system", "content": system}, {"role": "user", "content": user}]
        ids = self.tok.apply_chat_template(msgs, add_generation_prompt=True, return_tensors="pt").to(self.model.device)
        with self.torch.no_grad():
            out = self.model.generate(ids, max_new_tokens=self.max_new_tokens, do_sample=False)
        return self.tok.decode(out[0, ids.shape[1]:], skip_special_tokens=True).strip()

    def close(self):
        del self.model
        gc.collect()
        if self.torch.cuda.is_available():
            self.torch.cuda.empty_cache()


class ApiLLM:
    """Anthropic Messages API or any OpenAI-compatible chat endpoint. Key from an env var."""

    def __init__(self, fmt: str, model: str, key_env: str, url: str | None = None, max_tokens: int = 120):
        self.key = os.environ.get(key_env)
        if not self.key:
            raise RuntimeError(f"environment variable {key_env} is not set")
        self.fmt, self.model, self.max_tokens = fmt, model, max_tokens
        self.url = url or ("https://api.anthropic.com/v1/messages" if fmt == "anthropic"
                           else "https://api.openai.com/v1/chat/completions")

    def __call__(self, system: str, user: str) -> str:
        if self.fmt == "anthropic":
            body = {"model": self.model, "max_tokens": self.max_tokens, "system": system,
                    "messages": [{"role": "user", "content": user}]}
            headers = {"x-api-key": self.key, "anthropic-version": "2023-06-01"}
        else:
            body = {"model": self.model, "max_tokens": self.max_tokens,
                    "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}]}
            headers = {"Authorization": f"Bearer {self.key}"}
        req = urllib.request.Request(self.url, json.dumps(body).encode(), method="POST",
                                     headers={"content-type": "application/json", **headers})
        with urllib.request.urlopen(req, timeout=60) as r:
            data = json.load(r)
        if self.fmt == "anthropic":
            return "".join(b.get("text", "") for b in data["content"]).strip()
        return data["choices"][0]["message"]["content"].strip()

    def close(self):
        pass


# --------------------------------------------------------------------------- writer

class PromptWriter:
    def __init__(self, cfg: dict, root=None):
        self.cfg = cfg
        self.backend = cfg.get("backend", "template")
        self.mode = cfg.get("clap_context", "off")
        assert self.mode in CLAP_MODES, self.mode
        self.th = palette_thresholds(root)
        self.names = _person_names(root)
        self.llm = None
        if self.backend == "local":
            self.llm = LocalLLM(cfg["local_model"], load_in_4bit=cfg.get("load_in_4bit", False))
        elif self.backend == "api":
            self.llm = ApiLLM(cfg.get("api_format", "anthropic"), cfg["api_model"], cfg["api_key_env"],
                              cfg.get("api_url"))

    def write(self, shot: dict) -> tuple[str, dict]:
        """Returns (prompt, log record)."""
        pal = palette_words(shot["kin"], self.th)
        tmpl = template_prompt(shot, pal, self.mode)
        rec = {"backend": self.backend, "palette": pal, "template": tmpl}
        if self.llm is None or not drawn_phrases(shot, self.mode):
            # nothing to describe: an LLM could only invent content, which the rules forbid
            return tmpl, {**rec, "used": "template", "prompt": tmpl}
        try:
            raw = self.llm(SYSTEM_PROMPT, _user_message(shot, pal, self.mode))
        except Exception as e:                        # network / model failure -> template
            return tmpl, {**rec, "used": "template", "error": repr(e), "prompt": tmpl}
        text = " ".join(raw.replace('"', "").split())
        reasons = validate(text, shot, pal, self.names)
        if reasons:
            return tmpl, {**rec, "used": "template", "llm_raw": raw, "rejected": reasons, "prompt": tmpl}
        return text, {**rec, "used": self.backend, "llm_raw": raw, "prompt": text}

    def close(self):
        """Free the LLM before the image model is loaded (one model on the GPU at a time)."""
        if self.llm is not None:
            self.llm.close()
            self.llm = None
