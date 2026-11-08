"""A tiny stand-in for alphagenome_pytorch.AlphaGenome, for CPU tests.

It keeps only what the intervention code relies on: an encoder that pools the
sequence to 128 bp bins, an organism embedding, a tower whose residual adds
live in `_forward_block` (with `mha` as a module whose pre-hook sees the
stream entering the block), heads that read the final stream, and a
`predict(dna, organism, resolutions, heads, channels_last, embeddings_only)`
entry point.
"""

from __future__ import annotations

import torch
from torch import nn

BIN = 128


class _MHA(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.proj = nn.Linear(dim, dim)

    def forward(self, x, bias, compute_dtype=None):
        # Mixes positions so an edit at one bin reaches its neighbours.
        mixed = x + 0.5 * torch.roll(x, 1, dims=1) + 0.5 * torch.roll(x, -1, dims=1)
        return torch.tanh(self.proj(mixed)) + 0 * bias


class _Tower(nn.Module):
    def __init__(self, dim, n_blocks):
        super().__init__()
        self.blocks = nn.ModuleList(
            nn.ModuleDict({"mha": _MHA(dim), "mlp": nn.Linear(dim, dim)})
            for _ in range(n_blocks))

    def _forward_block(self, block, x, pair_x, compute_dtype):
        x = x + block["mha"](x, 0.0, compute_dtype=compute_dtype)
        x = x + torch.tanh(block["mlp"](x))
        return x, pair_x

    def forward(self, x):
        pair_x = None
        for block in self.blocks:
            x, pair_x = self._forward_block(block, x, pair_x, None)
        return x


class FakeAlphaGenome(nn.Module):
    def __init__(self, dim=16, n_blocks=4, tracks=None, seed=0, autocast=False):
        super().__init__()
        # The real predict() runs forward under bf16 autocast with the
        # mixed-precision policy; autocast=True reproduces that.
        self.autocast = autocast
        torch.manual_seed(seed)
        self.encoder = nn.Linear(4, dim)
        self.organism_embed = nn.Embedding(2, dim)
        self.tower = _Tower(dim, n_blocks)
        self.tracks = tracks or {"chip_tf": 3, "atac": 2, "dnase": 2, "cage": 2, "procap": 2,
                                 "rna_seq": 2, "chip_histone": 2}
        self.heads = nn.ModuleDict({h: nn.Linear(dim, n) for h, n in self.tracks.items()})
        self.calls = 0

    def predict(self, dna, organism, **kwargs):
        with torch.autocast(dna.device.type, dtype=torch.bfloat16, enabled=self.autocast):
            return self._forward(dna, organism, **kwargs)

    def encode(self, dna, organism_index=0, resolutions=(128,)):
        # Extraction's entry point; it has no autocast wrapper here.
        return self._forward(dna, organism_index, resolutions=resolutions, embeddings_only=True)

    def _forward(self, dna, organism, resolutions=(128,), heads=None,
                 channels_last=True, embeddings_only=False):
        self.calls += 1
        dna = dna.float()
        b, length, _ = dna.shape
        x = self.encoder(dna.reshape(b, length // BIN, BIN, 4).mean(2))
        if isinstance(organism, int):
            organism = torch.full((b,), organism, dtype=torch.long)
        x = x + self.organism_embed(organism.to(x.device)).unsqueeze(1)
        x = self.tower(x)
        if embeddings_only:
            return {"embeddings_128bp": x}
        names = heads if heads is not None else tuple(self.tracks)
        return {h: {128: torch.nn.functional.softplus(self.heads[h](x))} for h in names}
