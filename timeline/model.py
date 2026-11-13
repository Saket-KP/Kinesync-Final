"""EDGE-style transformer denoiser for the Stage-1 timeline.

Architecture, following EDGE (Tseng et al., CVPR 2023):

    x_t (B,T,5) -> in_proj -> + learned positional embedding
      -> N x [ self-attn | cross-attn to music | FFN ]  with adaLN on all three
         sub-blocks, conditioned on the diffusion timestep
      -> out_proj -> x0_hat (B,T,5)

Cross-attention memory is [genre_token] + proj(rhythm), i.e. 1 + T tokens.

Cross-attention carries the music, and it is modulated by the timestep like everything
else (`ada_x`). It was not in the first three runs: the branch was a plain ungated
`x = x + a`, so the model's reliance on the music was forced to be constant across the
noise schedule, when the information it needs from the music -- where in the beat to
place a cut -- is only usable once the signal is clean enough to place anything.

What classifier-free guidance drops depends on the config: the genre token always, and
the rhythm stream too when `diffusion.cond_dropout > 0`. Dropping rhythm was off for
the first run on the reasoning that rhythm is the input signal rather than a style
label; measurement later showed the model was barely using it, so it became optional
rather than forbidden.

adaLN-Zero init (modulation projections start at zero) makes every block an
identity map at step 0, which is what keeps a ~9M-param diffusion transformer
stable without warmup tricks.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn

from common.config import GENRES, Config


def timestep_embedding(t: torch.Tensor, dim: int, max_period: float = 10000.0) -> torch.Tensor:
    """Sinusoidal embedding of diffusion timesteps. t is (B,) long."""
    half = dim // 2
    freqs = torch.exp(
        -math.log(max_period) * torch.arange(half, dtype=torch.float32, device=t.device) / half
    )
    args = t.float()[:, None] * freqs[None]
    emb = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
        emb = torch.cat([emb, torch.zeros_like(emb[:, :1])], dim=-1)
    return emb


def modulate(x: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    return x * (1.0 + scale.unsqueeze(1)) + shift.unsqueeze(1)


class Block(nn.Module):
    """Self-attention + cross-attention + FFN, with adaLN on self-attn and FFN."""

    def __init__(self, d: int, n_heads: int, d_ff: int, dropout: float):
        super().__init__()
        self.norm1 = nn.LayerNorm(d, elementwise_affine=False, eps=1e-6)
        self.self_attn = nn.MultiheadAttention(d, n_heads, dropout=dropout, batch_first=True)

        self.norm2 = nn.LayerNorm(d, eps=1e-6)
        self.cross_attn = nn.MultiheadAttention(d, n_heads, dropout=dropout, batch_first=True)

        self.norm3 = nn.LayerNorm(d, elementwise_affine=False, eps=1e-6)
        self.ff = nn.Sequential(
            nn.Linear(d, d_ff), nn.GELU(), nn.Dropout(dropout), nn.Linear(d_ff, d)
        )
        # shift/scale/gate for self-attn and for FFN
        self.ada = nn.Sequential(nn.SiLU(), nn.Linear(d, 6 * d))
        nn.init.zeros_(self.ada[1].weight)
        nn.init.zeros_(self.ada[1].bias)
        # Cross-attention modulation, added after the first three trained runs.
        #
        # Deliberately a SEPARATE module rather than widening `ada` to 9*d: widening it
        # would change the shape of an existing tensor, so no earlier checkpoint could
        # load at all. As its own module, a checkpoint from before this change simply
        # lacks `ada_x` and leaves it at the zero init -- and because the gate enters as
        # (1 + g) below, zero means "fully on", which is exactly the ungated
        # `x = x + a` those runs were trained with. Old checkpoints are bit-identical.
        self.ada_x = nn.Sequential(nn.SiLU(), nn.Linear(d, 3 * d))
        nn.init.zeros_(self.ada_x[1].weight)
        nn.init.zeros_(self.ada_x[1].bias)

    def forward(self, x: torch.Tensor, t_emb: torch.Tensor, mem: torch.Tensor) -> torch.Tensor:
        sh1, sc1, g1, sh2, sc2, g2 = self.ada(t_emb).chunk(6, dim=-1)
        shx, scx, gx = self.ada_x(t_emb).chunk(3, dim=-1)

        h = modulate(self.norm1(x), sh1, sc1)
        a, _ = self.self_attn(h, h, h, need_weights=False)
        x = x + g1.unsqueeze(1) * a

        h = modulate(self.norm2(x), shx, scx)
        a, _ = self.cross_attn(h, mem, mem, need_weights=False)
        # (1 + gx), not gx as adaLN-Zero would have it: cross-attention is the ONLY
        # path by which the music reaches the output, so a zero-init gate would start
        # the model deaf and leave it to climb out. At 1 the model begins exactly where
        # the ungated version did, and the timestep can open or close the path from
        # there -- which is the point of the change, since placement information only
        # becomes usable at low noise.
        x = x + (1.0 + gx).unsqueeze(1) * a

        h = modulate(self.norm3(x), sh2, sc2)
        x = x + g2.unsqueeze(1) * self.ff(h)
        return x


class TimelineDenoiser(nn.Module):
    def __init__(self, cfg: Config, max_len: int = 1024):
        super().__init__()
        m, d_cfg = cfg.model, cfg.data
        d = m.d_model
        self.d_model = d
        self.n_out = d_cfg.n_out_ch

        self.in_proj = nn.Linear(d_cfg.n_out_ch, d)
        self.pos_emb = nn.Parameter(torch.zeros(1, max_len, d))
        nn.init.trunc_normal_(self.pos_emb, std=0.02)

        self.t_mlp = nn.Sequential(nn.Linear(d, d), nn.SiLU(), nn.Linear(d, d))
        self.cond_proj = nn.Linear(d_cfg.n_cond_ch, d)
        self.genre_emb = nn.Embedding(len(GENRES), d)
        self.null_genre = nn.Parameter(torch.zeros(1, 1, d))
        nn.init.trunc_normal_(self.null_genre, std=0.02)
        # Learned stand-in for "no music", broadcast over time when the rhythm stream
        # is dropped. Dropping it in training is what teaches the model that the music
        # is informative, and it is what makes guidance on the music possible at all.
        #
        # Created ONLY when the config uses it. If it existed unconditionally, a model
        # built today would carry one more parameter than a checkpoint trained before
        # it, and the optimiser state -- which is a flat list positionally matched to
        # the parameters -- would refuse to load, so older runs could no longer be
        # resumed at all. Gating it on the config keeps those runs exactly reproducible.
        dcfg = cfg.diffusion
        self.uses_null_cond = dcfg.cond_dropout > 0 or dcfg.guidance_target in ("cond", "both")
        if self.uses_null_cond:
            self.null_cond = nn.Parameter(torch.zeros(1, 1, d))
            nn.init.trunc_normal_(self.null_cond, std=0.02)
        else:
            self.null_cond = None

        # EDGE's second route for the music: mean-pool the projected rhythm and add it
        # to the timestep vector, which then drives the modulation in every block. So
        # the music reaches the output both through attention (where it must compete for
        # attention mass) and globally (where it cannot be ignored). Zero-initialised,
        # so a model built with this on starts out computing exactly what one built
        # without it computes, and an older checkpoint keeps its behaviour.
        self.cond_to_time = m.cond_to_time
        if self.cond_to_time:
            self.cond_time_proj = nn.Linear(d, d)
            nn.init.zeros_(self.cond_time_proj.weight)
            nn.init.zeros_(self.cond_time_proj.bias)

        self.blocks = nn.ModuleList(
            [Block(d, m.n_heads, m.d_ff, m.dropout) for _ in range(m.n_layers)]
        )
        self.norm_out = nn.LayerNorm(d, elementwise_affine=False, eps=1e-6)
        self.ada_out = nn.Sequential(nn.SiLU(), nn.Linear(d, 2 * d))
        nn.init.zeros_(self.ada_out[1].weight)
        nn.init.zeros_(self.ada_out[1].bias)
        self.out_proj = nn.Linear(d, d_cfg.n_out_ch)
        nn.init.zeros_(self.out_proj.weight)
        nn.init.zeros_(self.out_proj.bias)

    def forward(
        self,
        x_t: torch.Tensor,        # (B, T, C_out) noisy timeline
        t: torch.Tensor,          # (B,) long
        cond: torch.Tensor,       # (B, T, C_cond) rhythm
        genre: torch.Tensor,      # (B,) long
        drop_genre: torch.Tensor, # (B,) bool -- True replaces genre with the null token
        drop_cond: torch.Tensor | None = None,  # (B,) bool -- True drops the rhythm
    ) -> torch.Tensor:
        B, T, _ = x_t.shape
        if T > self.pos_emb.shape[1]:
            raise ValueError(f"sequence length {T} exceeds max_len {self.pos_emb.shape[1]}")

        h = self.in_proj(x_t) + self.pos_emb[:, :T]
        t_emb = self.t_mlp(timestep_embedding(t, self.d_model))

        g = self.genre_emb(genre).unsqueeze(1)                     # (B,1,d)
        g = torch.where(drop_genre.view(B, 1, 1), self.null_genre.expand(B, 1, -1), g)
        c = self.cond_proj(cond)                                   # (B,T,d)
        if drop_cond is not None:
            if self.null_cond is None:
                raise RuntimeError(
                    "drop_cond was passed but this model has no null_cond: it was "
                    "built from a config with cond_dropout=0 and guidance_target="
                    "'genre'. Set diffusion.cond_dropout > 0 to train with rhythm "
                    "dropout."
                )
            c = torch.where(drop_cond.view(B, 1, 1), self.null_cond.expand(B, T, -1), c)
        if self.cond_to_time:
            # after the dropout, so a dropped rhythm stream is dropped on BOTH paths --
            # otherwise guidance on 'cond' would be amplifying a difference the
            # unconditional branch had partly kept
            t_emb = t_emb + self.cond_time_proj(c.mean(dim=1))
        mem = torch.cat([g, c], dim=1)                             # (B, 1+T, d)

        for blk in self.blocks:
            h = blk(h, t_emb, mem)

        sh, sc = self.ada_out(t_emb).chunk(2, dim=-1)
        h = modulate(self.norm_out(h), sh, sc)
        return self.out_proj(h)

    def n_params(self) -> int:
        return sum(p.numel() for p in self.parameters())


def build_model(cfg: Config, max_len: int | None = None) -> TimelineDenoiser:
    if max_len is None:
        max_len = max(1024, cfg.data.window_frames * 2)
    return TimelineDenoiser(cfg, max_len=max_len)


# Parameters added after earlier runs were trained. A checkpoint from before each one
# still loads AND still computes exactly what it did when it was trained:
#
#   null_cond -- those runs had cond_dropout=0, so the rhythm stream was never dropped
#     and the token was never read. Only guidance_target='genre' is meaningful for
#     them; guiding on 'cond' would extrapolate away from a prediction they were never
#     trained to make.
#   ada_x     -- cross-attention modulation. Its gate enters as (1 + g), so the zero
#     init it is left at reproduces the ungated `x = x + a` identically.
#
# Matched as substrings, because the per-block ones appear as
# `blocks.<i>.ada_x.1.weight`. Anything else missing is a real mismatch.
#   cond_time_proj -- the global music path. Zero-initialised, so leaving it at init
#     adds exactly nothing to the timestep vector.
_POST_HOC_PARAMS = ("null_cond", "ada_x.", "cond_time_proj.")


def pick_weights(ck: dict, ema: bool = False) -> dict:
    """The weights to evaluate: the raw ones, or the EMA average EDGE samples from."""
    if not ema:
        return ck["model"]
    if not ck.get("ema"):
        raise RuntimeError(
            "--ema was requested but this checkpoint carries no EMA weights: it was "
            "trained with train.ema_decay = 0. Evaluate it without --ema."
        )
    return ck["ema"]


def load_checkpoint_into(model: TimelineDenoiser, state: dict, quiet: bool = False) -> None:
    """Load weights, tolerating only the parameters added after a run was trained."""
    missing, unexpected = model.load_state_dict(state, strict=False)
    unknown = [k for k in missing if not any(p in k for p in _POST_HOC_PARAMS)]
    if unknown or unexpected:
        raise RuntimeError(
            f"checkpoint does not match this model: missing {sorted(unknown)}, "
            f"unexpected {sorted(unexpected)}"
        )
    if missing and not quiet:
        kinds = sorted({p.rstrip(".") for p in _POST_HOC_PARAMS
                        if any(p in k for k in missing)})
        print(f"note: this checkpoint predates {kinds}, left at init. Those defaults "
              f"reproduce the behaviour it was trained with, so it evaluates as it did "
              f"originally -- but its optimiser state cannot be resumed into this "
              f"model, and guidance_target='cond' is only meaningful if it was trained "
              f"with cond_dropout > 0.")


if __name__ == "__main__":
    cfg = Config()
    net = build_model(cfg)
    B, T = 2, cfg.data.window_frames
    x = torch.randn(B, T, cfg.data.n_out_ch)
    c = torch.randn(B, T, cfg.data.n_cond_ch)
    t = torch.randint(0, 1000, (B,))
    g = torch.randint(0, len(GENRES), (B,))
    d = torch.zeros(B, dtype=torch.bool)
    y = net(x, t, c, g, d)
    print(f"params      : {net.n_params()/1e6:.2f} M")
    print(f"input  x_t  : {tuple(x.shape)}")
    print(f"cond        : {tuple(c.shape)}")
    print(f"output x0^  : {tuple(y.shape)}")
    assert y.shape == x.shape
    print("forward pass ok")
