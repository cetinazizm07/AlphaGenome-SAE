"""One split of one tap, held in memory as a single tensor.

`ShardStore` reads rows from memory-mapped float16 shards and converts them to
float32 on the CPU, one microbatch at a time. On an L4 that kept the GPU
waiting most of each step. A tap's training split is ~25 GB in float16
(1000 windows x 8192 positions x 1536 channels), which fits on an 80 GB A100
next to several SAEs, so the pool loads it once and every batch becomes a GPU
gather.

When the split does not fit, the pool falls back to pinned host memory and
copies each batch over asynchronously.
"""

from __future__ import annotations

import warnings
from typing import Iterator

import numpy as np
import torch

from ag_sae.extract import ShardStore

#: GPU memory to leave free for the SAE, its optimiser and activations.
GPU_HEADROOM_BYTES = 12 * 2**30


class ActivationPool:
    """All rows of a ShardStore as one float16 tensor, on GPU or pinned host."""

    def __init__(self, store: ShardStore, device: str = "cuda",
                 placement: str = "auto", log=print) -> None:
        if placement not in ("auto", "gpu", "host"):
            raise ValueError("placement must be auto, gpu or host")
        self.store = store
        self.device = torch.device(device)
        self.dim, self.rows = store.dim, len(store)
        self.tap, self.split, self.identity = store.tap, store.split, store.identity
        nbytes = self.rows * self.dim * 2
        on_gpu = self.device.type == "cuda" and (
            placement == "gpu" or (placement == "auto" and _fits_on_gpu(nbytes, self.device)))
        # Filled one shard at a time, straight into its final home, so the
        # split is never held twice.
        if on_gpu:
            self.data = torch.empty((self.rows, self.dim), dtype=torch.float16,
                                    device=self.device)
        else:
            self.data = torch.empty((self.rows, self.dim), dtype=torch.float16,
                                    pin_memory=self.device.type == "cuda")
        offset = 0
        for array in store.arrays:
            with warnings.catch_warnings():
                # The shards are read-only memory maps; we only read from them.
                warnings.simplefilter("ignore", UserWarning)
                chunk = torch.from_numpy(np.ascontiguousarray(array))
            self.data[offset:offset + len(array)].copy_(chunk)
            offset += len(array)
        self.placement = "gpu" if on_gpu else "host"
        log(f"pool {store.tap}/{store.split}: {self.rows:,} rows, "
            f"{nbytes / 2**30:.1f} GiB on {self.placement}")

    def __len__(self) -> int:
        return self.rows

    def take(self, rows: torch.Tensor) -> torch.Tensor:
        """float32 rows on the compute device. `rows` is a 1-D index tensor."""
        if self.placement == "gpu":
            return self.data.index_select(0, rows.to(self.device)).float()
        picked = self.data.index_select(0, rows.cpu())
        if self.device.type == "cuda":
            picked = picked.pin_memory().to(self.device, non_blocking=True)
        elif self.device.type != "cpu":
            picked = picked.to(self.device)
        return picked.float()

    def order(self, seed: int, epoch: int, mode: str = "global") -> torch.Tensor:
        """Row order for one pass over the split.

        "global" draws every batch from the whole split, so a step mixes
        chromosomes. "shard" reproduces ShardStore.order (the first campaign).
        """
        if mode == "shard":
            order = self.store.order(seed, epoch)
        elif mode == "global":
            rng = np.random.default_rng(np.random.SeedSequence([seed, epoch]))
            order = rng.permutation(self.rows)
        else:
            raise ValueError("mode must be global or shard")
        return torch.from_numpy(order.astype(np.int64))

    def batches(self, size: int) -> Iterator[torch.Tensor]:
        """Sequential float32 batches on the compute device, for evaluation."""
        for start in range(0, self.rows, size):
            rows = torch.arange(start, min(start + size, self.rows))
            yield self.take(rows)

    def sample(self, n: int, seed: int) -> torch.Tensor:
        """A reproducible random subset of rows, for data-dependent init."""
        rng = np.random.default_rng(seed)
        rows = rng.choice(self.rows, size=min(n, self.rows), replace=False)
        return self.take(torch.from_numpy(np.sort(rows).astype(np.int64)))


def _fits_on_gpu(nbytes: int, device: torch.device) -> bool:
    free, _ = torch.cuda.mem_get_info(device)
    return nbytes + GPU_HEADROOM_BYTES <= free
