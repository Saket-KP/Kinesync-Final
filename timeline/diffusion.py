"""Diffusion core for the Stage-1 timeline model.

Follows EDGE (Tseng et al., CVPR 2023) in the choices that matter here:

  * cosine beta schedule (Nichol & Dhariwal)
  * the network predicts x0, not epsilon -- more stable over short horizons and it
    lets us apply losses directly on the signal we care about
  * an auxiliary first-difference loss, the analogue of EDGE's velocity/contact
    losses, which discourages jittery frame-to-frame output
  * classifier-free guidance, here on the genre token only
  * long-form generation by constraining overlapping windows (see sample.py); the
    `known`/`known_mask` arguments to ddim_sample implement that constraint
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F


def cosine_alphas_cumprod(T: int, s: float = 0.008) -> torch.Tensor:
    """alpha_bar(t) for t = 0..T-1, from Nichol & Dhariwal's cosine schedule."""
    t = torch.arange(T + 1, dtype=torch.float64) / T
    f = torch.cos((t + s) / (1.0 + s) * np.pi * 0.5) ** 2
    acp = (f / f[0]).clamp(1e-8, 1.0)
    return acp[1:].to(torch.float32)


class GaussianDiffusion:
    """x0-prediction DDPM training + DDIM sampling."""

    def __init__(self, timesteps: int = 1000, cosine_s: float = 0.008,
                 vel_loss_weight: float = 0.05, device: str | torch.device = "cpu"):
        self.T = timesteps
        self.vel_w = vel_loss_weight
        acp = cosine_alphas_cumprod(timesteps, cosine_s).to(device)
        self.acp = acp
        self.sqrt_acp = acp.sqrt()
        self.sqrt_one_minus_acp = (1.0 - acp).sqrt()
        self.device = device

    def to(self, device) -> "GaussianDiffusion":
        self.acp = self.acp.to(device)
        self.sqrt_acp = self.sqrt_acp.to(device)
        self.sqrt_one_minus_acp = self.sqrt_one_minus_acp.to(device)
        self.device = device
        return self

    # ------------------------------------------------------------------ forward
    def q_sample(self, x0: torch.Tensor, t: torch.Tensor,
                 noise: torch.Tensor | None = None) -> torch.Tensor:
        """x_t ~ q(x_t | x_0). t is (B,) long; x0 is (B, T, C)."""
        if noise is None:
            noise = torch.randn_like(x0)
        a = self.sqrt_acp[t].view(-1, 1, 1)
        b = self.sqrt_one_minus_acp[t].view(-1, 1, 1)
        return a * x0 + b * noise

    def training_loss(self, model, x0: torch.Tensor, cond: torch.Tensor,
                      genre: torch.Tensor, drop_genre: torch.Tensor,
                      t: torch.Tensor | None = None,
                      noise: torch.Tensor | None = None,
                      drop_cond: torch.Tensor | None = None) -> dict:
        """One denoising-loss step. `t` and `noise` default to fresh random draws.

        Passing them explicitly is what makes a *validation* loss comparable between
        evals: with fresh draws each time, the loss varies with the timestep sample
        as well as with the model, so two evals of the same weights differ. See
        train.py's val_loss, which supplies a stratified t and seeded noise.
        """
        B = x0.shape[0]
        if t is None:
            t = torch.randint(0, self.T, (B,), device=x0.device)
        if noise is None:
            noise = torch.randn_like(x0)
        x_t = self.q_sample(x0, t, noise)
        x0_hat = model(x_t, t, cond, genre, drop_genre, drop_cond)

        mse = F.mse_loss(x0_hat, x0)
        if self.vel_w > 0:
            dv = F.mse_loss(x0_hat[:, 1:] - x0_hat[:, :-1], x0[:, 1:] - x0[:, :-1])
        else:
            dv = torch.zeros((), device=x0.device)
        loss = mse + self.vel_w * dv
        # cut channel reported separately -- it is the one the research claim rests on
        with torch.no_grad():
            cut_mse = F.mse_loss(x0_hat[..., 0], x0[..., 0])
        return {"loss": loss, "mse": mse.detach(), "vel": dv.detach(),
                "cut_mse": cut_mse.detach()}

    # ----------------------------------------------------------------- sampling
    @torch.no_grad()
    def ddim_sample(
        self,
        model,
        shape: tuple[int, int, int],
        cond: torch.Tensor,
        genre: torch.Tensor,
        steps: int = 50,
        guidance: float = 2.0,
        known: torch.Tensor | None = None,
        known_mask: torch.Tensor | None = None,
        generator: torch.Generator | None = None,
        progress: bool = False,
        guidance_target: str = "genre",
        eta: float = 0.0,
        guidance_ramp: bool = False,
        clip_cut: bool = False,
        overlap_half: int | None = None,
    ) -> torch.Tensor:
        """Deterministic DDIM (eta=0) with classifier-free guidance on genre.

        `known` (B,T,C) + `known_mask` (T,) bool constrains part of the window to an
        already-generated signal: at every step the known region is replaced by the
        correctly-noised target, which is how overlapping windows are stitched into a
        full song without a seam.

        `overlap_half` switches on EDGE's long-form constraint instead: the batch is
        treated as consecutive 50%-overlapping windows of one song, and after every
        step the first half of each window is overwritten with the second half of its
        predecessor. Unlike `known`, nothing is frozen -- the whole song keeps
        co-adapting until the trajectory ends.

        `eta` > 0 makes the step stochastic (EDGE uses 1.0); 0 is deterministic DDIM.
        `guidance_ramp` follows EDGE in raising the weight from 0 to its full value
        over the first half of the trajectory. `clip_cut` clamps channel 0 of the x0
        prediction to [-1, 1] at every step, as EDGE clamps its whole representation.
        """
        device = cond.device
        B, T, C = shape
        x = torch.randn(shape, device=device, generator=generator)
        if known is not None and known_mask is not None:
            m = known_mask.view(1, T, 1)
            x = torch.where(m, self.q_sample(known, torch.full((B,), self.T - 1,
                                                               device=device,
                                                               dtype=torch.long)), x)

        ts = torch.linspace(self.T - 1, 0, steps, device=device).round().long()
        it = enumerate(ts)
        if progress:
            from tqdm import tqdm
            it = tqdm(list(it), total=len(ts), leave=False, desc="ddim")

        off = torch.zeros(B, dtype=torch.bool, device=device)
        on = torch.ones(B, dtype=torch.bool, device=device)
        # A model trained without rhythm dropout has no null_cond token, and handing it
        # any drop_cond at all -- even an all-False mask -- is a misconfiguration it
        # must reject. So pass None rather than "all False" for such a model, keeping
        # the old guidance_target='genre' path byte-identical to before.
        keep_cond = off if getattr(model, "null_cond", None) is not None else None
        # What the unconditional branch drops decides what guidance amplifies: the
        # extrapolation x0_u + w*(x0_c - x0_u) scales exactly the component of the
        # prediction that the dropped condition is responsible for.
        drop_g, drop_c = {
            "genre": (on, keep_cond),
            "cond": (off, on),
            "both": (on, on),
        }[guidance_target]

        # EDGE: weights = np.clip(np.linspace(0, 2*w, steps), None, w), indexed from the
        # noisiest step down, so guidance is off at the start and full from the midpoint.
        if guidance_ramp:
            w_sched = torch.linspace(0.0, 2.0 * guidance, steps,
                                     device=device).clamp(max=guidance)
        else:
            w_sched = torch.full((steps,), float(guidance), device=device)

        for i, t in it:
            tb = t.expand(B)
            w = float(w_sched[i])
            if w == 0.0:                       # purely unconditional; one pass suffices
                x0_hat = model(x, tb, cond, genre, drop_g, drop_c)
            elif w != 1.0:
                x0_c = model(x, tb, cond, genre, off, keep_cond)
                x0_u = model(x, tb, cond, genre, drop_g, drop_c)
                x0_hat = x0_u + w * (x0_c - x0_u)
            else:
                x0_hat = model(x, tb, cond, genre, off, keep_cond)

            if clip_cut:
                # Only channel 0. EDGE clamps its whole pose vector because every one of
                # its channels is normalised into [-1, 1]; here the other four are
                # z-scored and unbounded, so clamping them would cut away real signal.
                x0_hat = torch.cat([x0_hat[..., :1].clamp(-1.0, 1.0),
                                    x0_hat[..., 1:]], dim=-1)

            acp_t = self.acp[t]
            eps = (x - acp_t.sqrt() * x0_hat) / (1.0 - acp_t).sqrt().clamp_min(1e-8)

            if i + 1 < len(ts):
                t_prev = ts[i + 1]
                acp_prev = self.acp[t_prev]
                if eta > 0.0:
                    # stochastic DDIM: give back part of the variance as fresh noise
                    sigma = eta * (((1.0 - acp_t / acp_prev)
                                    * (1.0 - acp_prev) / (1.0 - acp_t))
                                   .clamp_min(0.0)).sqrt()
                    c = (1.0 - acp_prev - sigma ** 2).clamp_min(0.0).sqrt()
                    z = torch.randn(x.shape, device=device, generator=generator)
                    x = acp_prev.sqrt() * x0_hat + c * eps + sigma * z
                else:
                    x = acp_prev.sqrt() * x0_hat + (1.0 - acp_prev).sqrt() * eps
                if known is not None and known_mask is not None:
                    m = known_mask.view(1, T, 1)
                    x_known = self.q_sample(known, t_prev.expand(B))
                    x = torch.where(m, x_known, x)
            else:
                x = x0_hat
                if known is not None and known_mask is not None:
                    x = torch.where(known_mask.view(1, T, 1), known, x)

            # EDGE's long-form constraint, re-asserted every step so no window is ever
            # frozen. We also apply it on the FINAL step, where EDGE does not: that
            # makes window k+1's first half exactly equal to window k's second half, so
            # the song assembles by concatenation. EDGE instead blends the seams
            # afterwards (linear fade / slerp), which for us would smear precisely the
            # sharp cut peaks the model exists to place.
            if overlap_half is not None and B > 1:
                x[1:, :overlap_half] = x[:-1, overlap_half:]
        return x
