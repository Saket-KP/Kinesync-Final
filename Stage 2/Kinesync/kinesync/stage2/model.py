"""Stage 2 shot-content model (handoff 4.2): a small causal transformer over a song's shots.

Per shot, every input is projected to d and summed, plus an embedding of the PREVIOUS shot's
attributes (teacher-forced in training, sampled at inference). Heads predict scale, moving
and one head per tag group.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from kinesync.common import N_SECTIONS, Vocab


class Stage2Model(nn.Module):
    def __init__(self, vocab: Vocab, n_genres: int, d: int = 256, layers: int = 4, heads: int = 4,
                 ff: int = 1024, dropout: float = 0.1, clap_dim: int = 512, lyric_dim: int = 384):
        super().__init__()
        self.vocab = vocab
        self.n_tags = vocab.n_tags
        # ---- per-shot inputs -> d, summed
        self.in_clap = nn.Linear(clap_dim, d)
        self.in_lyric = nn.Linear(lyric_dim + 1, d)      # lyric embedding + has_lyric flag
        self.in_dur = nn.Linear(1, d)
        self.in_kin = nn.Linear(4, d)
        self.in_section = nn.Embedding(N_SECTIONS, d)
        self.in_genre = nn.Embedding(n_genres + 1, d)     # last index = unknown genre
        self.in_pos = nn.Linear(1, d)
        # ---- previous-shot conditioning
        self.prev_scale = nn.Embedding(5, d)
        self.prev_moving = nn.Embedding(2, d)
        self.prev_tags = nn.Linear(self.n_tags, d)
        self.start = nn.Parameter(torch.zeros(d))         # used where there is no previous shot
        nn.init.normal_(self.start, std=0.02)
        self.in_drop = nn.Dropout(dropout)
        # ---- causal transformer (pre-norm)
        layer = nn.TransformerEncoderLayer(d, heads, ff, dropout, batch_first=True, norm_first=True,
                                           activation="gelu")
        self.encoder = nn.TransformerEncoder(layer, layers, enable_nested_tensor=False)
        self.out_norm = nn.LayerNorm(d)
        # ---- heads
        self.head_scale = nn.Linear(d, 5)
        self.head_moving = nn.Linear(d, 1)
        self.head_tags = nn.ModuleDict({
            g.name: nn.Linear(d, g.n + 1 if g.mode == "single" else g.n) for g in vocab.groups
        })

    def n_params(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def embed(self, b: dict) -> torch.Tensor:
        lyric = torch.cat([b["lyric"], b["has_lyric"].unsqueeze(-1)], -1)
        x = (self.in_clap(b["clap"]) + self.in_lyric(lyric)
             + self.in_dur(b["log_dur"].unsqueeze(-1)) + self.in_kin(b["kin"])
             + self.in_section(b["section"]) + self.in_genre(b["genre"])
             + self.in_pos(b["pos"].unsqueeze(-1)))
        prev = (self.prev_scale(b["prev_scale"]) + self.prev_moving(b["prev_moving"])
                + self.prev_tags(b["prev_tags"]))
        hp = b["has_prev"].unsqueeze(-1)
        x = x + hp * prev + (1 - hp) * self.start
        return self.in_drop(x)

    def forward(self, b: dict) -> dict[str, torch.Tensor]:
        x = self.embed(b)
        L = x.shape[1]
        causal = torch.triu(torch.ones(L, L, dtype=torch.bool, device=x.device), diagonal=1)
        pad = ~b["mask"] if "mask" in b else None
        h = self.encoder(x, mask=causal, src_key_padding_mask=pad)
        h = self.out_norm(h)
        out = {"scale": self.head_scale(h), "moving": self.head_moving(h).squeeze(-1)}
        for name, head in self.head_tags.items():
            out["tag_" + name] = head(h)
        return out


def time_features(t: torch.Tensor, n: int = 32) -> torch.Tensor:
    """Sinusoidal features of a time in seconds, periods from 0.5 s to about 1,000 s."""
    periods = torch.logspace(-0.3, 3.0, n // 2, device=t.device)
    ang = 2 * torch.pi * t.unsqueeze(-1) / periods
    return torch.cat([torch.sin(ang), torch.cos(ang)], -1)


class Stage2CrossModel(Stage2Model):
    """Stage 2 with cross-attention. Shot tokens are built exactly as in Stage2Model (so the shot-averaged
    CLAP and lyrics are still there); each layer adds cross-attention from the shots to a memory of the
    whole song: one token per second of CLAP and one per kept lyric line. Shots and memory tokens both
    carry sinusoidal time features (seconds), so a shot can learn to look at the seconds around it, or
    at the chorus, or anywhere else. Self-attention over shots stays causal; the memory is the song's
    audio and lyrics, which are known in full before any shot is sampled, so it is not masked."""

    def __init__(self, vocab: Vocab, n_genres: int, d: int = 256, layers: int = 4, heads: int = 4,
                 ff: int = 1024, dropout: float = 0.1, clap_dim: int = 512, lyric_dim: int = 384):
        super().__init__(vocab, n_genres, d, layers, heads, ff, dropout, clap_dim, lyric_dim)
        del self.encoder                                     # replaced by a decoder stack below
        self.time = nn.Linear(32, d)                         # shared by shots and memory tokens
        self.mem_clap = nn.Linear(clap_dim, d)
        self.mem_lyric = nn.Linear(lyric_dim, d)
        self.mem_type = nn.Embedding(2, d)                   # 0 = audio second, 1 = lyric line
        self.mem_norm = nn.LayerNorm(d)
        layer = nn.TransformerDecoderLayer(d, heads, ff, dropout, batch_first=True, norm_first=True,
                                           activation="gelu")
        self.decoder = nn.TransformerDecoder(layer, layers)

    def memory(self, b: dict):
        dev = b["clap"].device
        a = self.mem_clap(b["mem_clap"]) + self.mem_type(torch.zeros(1, dtype=torch.long, device=dev)) \
            + self.time(time_features(b["mem_clap_t"]))
        l = self.mem_lyric(b["mem_lyric"]) + self.mem_type(torch.ones(1, dtype=torch.long, device=dev)) \
            + self.time(time_features(b["mem_lyric_t"]))
        mem = self.mem_norm(torch.cat([a, l], 1))
        valid = torch.cat([b["mem_clap_valid"], b["mem_lyric_valid"]], 1) > 0
        return self.in_drop(mem), ~valid

    def forward(self, b: dict) -> dict[str, torch.Tensor]:
        x = self.embed(b) + self.time(time_features(b["mid_s"]))
        L = x.shape[1]
        causal = torch.triu(torch.ones(L, L, dtype=torch.bool, device=x.device), diagonal=1)
        pad = ~b["mask"] if "mask" in b else None
        mem, mem_pad = self.memory(b)
        h = self.decoder(x, mem, tgt_mask=causal, tgt_key_padding_mask=pad, memory_key_padding_mask=mem_pad)
        h = self.out_norm(h)
        out = {"scale": self.head_scale(h), "moving": self.head_moving(h).squeeze(-1)}
        for name, head in self.head_tags.items():
            out["tag_" + name] = head(h)
        return out


class Stage2Ensemble(nn.Module):
    """Average of several trained Stage 2 models. Logits are averaged, so softmax and sigmoid heads stay
    valid and the sampler, evaluator and calibration code need no change. Members must share the vocab,
    genre list and training data (their label_stats are then identical)."""

    def __init__(self, members: list[Stage2Model]):
        super().__init__()
        self.members = nn.ModuleList(members)
        self.vocab = members[0].vocab

    def n_params(self) -> int:
        return sum(m.n_params() for m in self.members)

    def forward(self, b: dict) -> dict[str, torch.Tensor]:
        outs = [m(b) for m in self.members]
        return {k: torch.stack([o[k] for o in outs]).mean(0) for k in outs[0]}


# --------------------------------------------------------------------------- targets and loss

def group_targets(tags: torch.Tensor, g) -> torch.Tensor:
    """Single-mode group: class index of the (at most one) tag present, or g.n for 'none'."""
    sub = tags[..., g.ids]
    has = sub.sum(-1) > 0
    return torch.where(has, sub.argmax(-1), torch.full_like(has, g.n, dtype=torch.long))


def stage2_loss(out: dict, b: dict, vocab: Vocab, scale_weight: torch.Tensor | None = None,
                pos_weight: torch.Tensor | None = None, weighted: bool = True) -> dict:
    """Returns each loss term separately plus 'total'. weighted=False gives plain NLL terms
    (used for early stopping and reporting val NLL)."""
    m = b["mask"]
    tm = m & (b["tag_mask"] > 0)
    terms = {}
    terms["scale"] = F.cross_entropy(out["scale"][m], b["scale"][m],
                                     weight=scale_weight if weighted else None)
    terms["moving"] = F.binary_cross_entropy_with_logits(out["moving"][m], b["moving"][m])
    for g in vocab.groups:
        logits = out["tag_" + g.name][tm]
        if logits.numel() == 0:
            terms["tag_" + g.name] = logits.sum() * 0
            continue
        if g.mode == "single":
            terms["tag_" + g.name] = F.cross_entropy(logits, group_targets(b["tags"], g)[tm])
        else:
            pw = pos_weight[g.ids] if (weighted and pos_weight is not None) else None
            terms["tag_" + g.name] = F.binary_cross_entropy_with_logits(
                logits, b["tags"][..., g.ids][tm], pos_weight=pw)
    terms["total"] = sum(terms.values())
    return terms
