"""Stage-1 configuration.

Plain dataclasses plus a YAML loader, so a run is fully described by one file and
any field can be overridden from the command line as --section.key=value.
"""
from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

# ---------------------------------------------------------------- channel layout
# Stage-1 output channels, in order. `cut` is channel 0 everywhere.
OUT_CHANNELS = ("cut", "flow", "color_drift", "brightness", "saturation")

# Rhythm conditioning, in order. Widths must sum to DataConfig.n_cond_ch.
COND_SPEC = (
    ("onset", 1),
    ("flux", 1),
    ("chroma", 12),
    ("beat_phase", 2),
    ("downbeat_phase", 2),
    ("section_onehot", 8),
)

GENRES = ("electronic", "hiphop", "pop", "rnb_latin", "rock")


@dataclass
class DataConfig:
    root: str = "data"
    fps: int = 25            # Stage-1 output grid
    audio_fps: int = 10      # native rate of onset/flux/chroma/beat_phase/section
    video_fps: int = 2       # native rate of flow/color_drift/brightness/saturation
    window_s: float = 10.0
    hop_s: float = 5.0
    cut_sigma_frames: float = 2.0   # Gaussian width of the soft cut target (~80 ms)
    n_sections: int = 8
    n_cond_ch: int = 26
    n_out_ch: int = 5
    # Videos with fewer than this many detected cuts are dropped from training and
    # from the norm-stats set. 0 keeps everything (verbatim load). Default 1 drops
    # single-shot videos, whose cut target is entirely zero and which would teach the
    # model "never cut" -- the v2 release has 4 of them, all in train.
    min_cuts_per_video: int = 1

    @property
    def window_frames(self) -> int:
        return int(round(self.window_s * self.fps))

    @property
    def hop_frames(self) -> int:
        return int(round(self.hop_s * self.fps))


@dataclass
class ModelConfig:
    d_model: int = 256
    n_layers: int = 6
    n_heads: int = 8
    d_ff: int = 1024
    dropout: float = 0.1
    # EDGE feeds the music in TWICE: as cross-attention memory, and as a mean-pooled
    # vector added to the timestep embedding (`t += cond_hidden`), which then drives
    # every FiLM modulation in every block. We only ever had the cross-attention path,
    # and measurement showed the model barely using it. This adds the second path.
    # Off by default so the first three runs stay reproducible.
    cond_to_time: bool = False


@dataclass
class DiffusionConfig:
    timesteps: int = 1000
    cosine_s: float = 0.008
    ddim_steps: int = 50
    cfg_dropout: float = 0.1     # prob. of dropping the genre token during training
    guidance: float = 2.0        # classifier-free guidance weight at sampling
    vel_loss_weight: float = 0.05  # first-difference loss, EDGE's auxiliary-loss analogue
    # Rhythm (music) conditioning dropout + what guidance is applied to. The first
    # trained run used cond_dropout=0 and guidance_target="genre": guidance was applied
    # to the one input that did not need strengthening and withheld from the one the
    # research claim depends on, and the model learned to ignore the music for cut
    # placement. Defaults keep that old behaviour so earlier runs stay reproducible;
    # set cond_dropout=0.1 and guidance_target="cond" to actually guide on rhythm.
    cond_dropout: float = 0.0
    guidance_target: str = "genre"   # "genre" | "cond" | "both"

    # ---------------------------------------------------- from EDGE's released code
    # The paper describes ancestral sampling over the full schedule; the code ships
    # something different, and these are the differences that are ours to choose.
    # Defaults below reproduce our first three runs; stage1_edge.yaml sets EDGE's.
    #
    # eta=0 is deterministic DDIM. EDGE samples with eta=1, i.e. a stochastic step:
    # part of the variance is re-injected as fresh noise at every step, which trades
    # reproducibility for diversity.
    ddim_eta: float = 0.0
    # EDGE ramps the guidance weight over the sampling trajectory --
    # np.clip(np.linspace(0, 2w, steps), None, w) -- so the first steps are nearly
    # unconditional and full guidance only arrives at the halfway point. Appendix G
    # attributes a diversity gain to this.
    guidance_ramp: bool = False
    # EDGE clamps its x0 prediction to [-1, 1] at every sampling step, which is sound
    # for them because every channel of their pose representation is normalised into
    # that range. Here only the cut channel is ([0,1] -> [-1,1] in norm_out); the other
    # four are z-scored and genuinely unbounded, so clamping all five would clip real
    # signal. This clamps channel 0 alone.
    clip_cut: bool = False
    # "sequential": generate windows in order, freezing each overlap once written.
    # "batch": EDGE's long_ddim_sample -- hold the whole song as a batch of windows and
    # re-assert the overlap after every denoising step, so the entire song co-adapts
    # throughout the trajectory instead of being pinned by its own first window.
    long_mode: str = "sequential"


@dataclass
class TrainConfig:
    batch_size: int = 64
    lr: float = 2e-4
    weight_decay: float = 0.01
    warmup_steps: int = 500
    max_steps: int = 20000
    grad_clip: float = 1.0
    amp: bool = True
    num_workers: int = 0
    ckpt_every: int = 1000
    eval_every: int = 1000
    log_every: int = 50
    out_dir: str = "runs/stage1"
    normalize: str = "zscore"   # "zscore" | "none"  (see data/norm_stats.json)
    seed: int = 0
    # Exponential moving average of the weights. EDGE keeps one at decay 0.9999,
    # updates it every step from step 0 with no warmup, and SAMPLES FROM IT rather
    # than from the raw weights -- their test path loads "ema_state_dict". We had none
    # at all, which is an omission rather than a decision: EMA is standard for
    # diffusion and usually worth a visible margin. 0.0 disables it.
    ema_decay: float = 0.0
    ema_every: int = 1


@dataclass
class Config:
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    diffusion: DiffusionConfig = field(default_factory=DiffusionConfig)
    train: TrainConfig = field(default_factory=TrainConfig)

    # ------------------------------------------------------------------ plumbing
    @classmethod
    def load(cls, path: str | Path | None = None, overrides: list[str] | None = None) -> "Config":
        cfg = cls()
        if path is not None:
            raw = yaml.safe_load(Path(path).read_text()) or {}
            for section, values in raw.items():
                if not hasattr(cfg, section):
                    raise KeyError(f"unknown config section: {section}")
                sub = getattr(cfg, section)
                for k, v in (values or {}).items():
                    if not hasattr(sub, k):
                        raise KeyError(f"unknown key: {section}.{k}")
                    setattr(sub, k, v)
        for ov in overrides or []:
            cfg.apply_override(ov)
        cfg.validate()
        return cfg

    def apply_override(self, spec: str) -> None:
        """--data.fps=25 style override."""
        if "=" not in spec:
            raise ValueError(f"bad override {spec!r}, expected section.key=value")
        lhs, value = spec.split("=", 1)
        section, _, key = lhs.partition(".")
        if not key:
            raise ValueError(f"bad override {spec!r}, expected section.key=value")
        sub = getattr(self, section, None)
        if sub is None or not hasattr(sub, key):
            raise KeyError(f"unknown config field: {lhs}")
        cur = getattr(sub, key)
        if isinstance(cur, bool):
            cast: Any = value.lower() in ("1", "true", "yes")
        elif isinstance(cur, int):
            cast = int(value)
        elif isinstance(cur, float):
            cast = float(value)
        else:
            cast = value
        setattr(sub, key, cast)

    def validate(self) -> None:
        d = self.data
        got = sum(w for _, w in COND_SPEC)
        if got != d.n_cond_ch:
            raise ValueError(f"COND_SPEC sums to {got} but n_cond_ch={d.n_cond_ch}")
        if len(OUT_CHANNELS) != d.n_out_ch:
            raise ValueError(f"OUT_CHANNELS has {len(OUT_CHANNELS)} but n_out_ch={d.n_out_ch}")
        if d.fps % d.audio_fps and d.audio_fps % d.fps:
            pass  # non-integer ratios are fine, we interpolate
        if self.model.d_model % self.model.n_heads:
            raise ValueError("d_model must be divisible by n_heads")
        if self.train.normalize not in ("zscore", "none"):
            raise ValueError("train.normalize must be 'zscore' or 'none'")
        if self.diffusion.guidance_target not in ("genre", "cond", "both"):
            raise ValueError("diffusion.guidance_target must be 'genre', 'cond' or 'both'")
        if self.diffusion.guidance_target in ("cond", "both") and self.diffusion.cond_dropout <= 0:
            raise ValueError(
                "guidance_target includes 'cond' but cond_dropout is 0: guiding on a "
                "condition the model was never trained without extrapolates away from "
                "a prediction it has never made. Set diffusion.cond_dropout > 0."
            )
        if d.hop_frames < 1 or d.hop_frames > d.window_frames:
            raise ValueError("hop_s must be in (0, window_s]")
        if self.diffusion.long_mode not in ("sequential", "batch"):
            raise ValueError("diffusion.long_mode must be 'sequential' or 'batch'")
        if self.diffusion.long_mode == "batch" and d.hop_frames * 2 != d.window_frames:
            raise ValueError(
                f"long_mode='batch' re-asserts the overlap as x[1:, :half] = "
                f"x[:-1, half:], which assumes the hop is exactly half the window "
                f"(EDGE: 2.5 s hop on a 5 s window). Here hop={d.hop_s}s on a "
                f"{d.window_s}s window. Set hop_s={d.window_s / 2} or use "
                f"long_mode='sequential'."
            )
        if not 0.0 <= self.diffusion.ddim_eta <= 1.0:
            raise ValueError("diffusion.ddim_eta must be in [0, 1]")
        if not 0.0 <= self.train.ema_decay < 1.0:
            raise ValueError("train.ema_decay must be in [0, 1)")
        if self.train.ema_every < 1:
            raise ValueError("train.ema_every must be >= 1")

    def to_dict(self) -> dict:
        return dataclasses.asdict(self)

    def save(self, path: str | Path) -> None:
        Path(path).write_text(yaml.safe_dump(self.to_dict(), sort_keys=False))


def add_config_args(ap) -> None:
    ap.add_argument("--config", default=None, help="YAML config path")
    ap.add_argument("--set", dest="overrides", action="append", default=[],
                    metavar="section.key=value", help="override a config field (repeatable)")
