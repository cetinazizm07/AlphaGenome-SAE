"""Batched edits of AlphaGenome's residual stream, entirely on the GPU.

The first campaign ran every condition as its own forward pass with batch 1,
plus a separate pass just to capture the residual stream, and moved the full
(8192, 1536) stream to the CPU and back on every call. A 2x2 motif window cost
six forwards; a triad window six.

Here a window's conditions are stacked along the batch axis and run together.
The edit happens inside the forward, at the entrance of the chosen tower
block, so no capture pass is needed: the stream entering block i depends only
on the input sequence, so each batch element sees exactly what a capture pass
would have returned. A condition can read what an earlier condition computed
(for example "restore the feature to its wild-type value") through a shared
memo dictionary; conditions run in order, and the memo survives across
batches when a window has more conditions than fit in memory at once.

Why patch `_forward_block` rather than register a hook: the residual adds
happen in `TransformerTower._forward_block`, a plain method, so the stream
entering a block is nobody's input or output. A pre-hook on `mha` can change
what attention sees but not the residual the block adds to.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Iterable, Sequence

import numpy as np
import torch

#: rows (n_bins, D) float32, memo -> replacement rows (n_bins, D)
RowEdit = Callable[[torch.Tensor, dict], torch.Tensor]
#: (model outputs for the whole batch, batch index) -> anything picklable
Readout = Callable[[dict, int], Any]


@dataclass(frozen=True)
class Condition:
    """One forward pass of the experiment.

    `sequence` indexes the input sequences of the window. When `edit` is None
    the stream is left alone (the unedited model); otherwise the rows at
    `bins` are replaced by ``edit(rows, memo)``.
    """

    name: str
    sequence: int = 0
    bins: Sequence[int] | None = None
    edit: RowEdit | None = None

    def __post_init__(self):
        if (self.edit is None) != (self.bins is None):
            raise ValueError(f"{self.name}: give both bins and edit, or neither")
        if self.bins is not None and len(self.bins) == 0:
            raise ValueError(f"{self.name}: no bins to edit")


class ResidualEdit:
    """Replace the residual stream entering one tower block, per batch element.

    `edit_fn(x)` receives the stream (B, S, D) in the model's compute dtype and
    returns a tensor of the same shape and dtype. It runs once per forward.
    """

    def __init__(self, model: Any, block_index: int,
                 edit_fn: Callable[[torch.Tensor], torch.Tensor]) -> None:
        self.tower = model.tower
        self.block = self.tower.blocks[int(block_index)]
        self.edit_fn = edit_fn
        self.previous = None

    def __enter__(self) -> "ResidualEdit":
        self.previous = previous = self.tower._forward_block

        def on_block(block, x, pair_x, compute_dtype):
            if block is self.block:
                edited = self.edit_fn(x)
                if edited.shape != x.shape or edited.dtype != x.dtype:
                    raise ValueError("edit changed the stream's shape or dtype")
                if not torch.isfinite(edited).all():
                    raise ValueError("edit produced non-finite values")
                x = edited
            return previous(block, x, pair_x, compute_dtype)

        self.tower._forward_block = on_block
        return self

    def __exit__(self, *_exc) -> None:
        if self.previous is not None:
            self.tower._forward_block = self.previous
            self.previous = None


class _Stop(Exception):
    def __init__(self, stream: torch.Tensor) -> None:
        super().__init__("stopped at the requested block")
        self.stream = stream


@torch.inference_mode()
def capture_residual(model: Any, block_index: int, sequences: torch.Tensor,
                     organism: int = 0) -> torch.Tensor:
    """Residual stream entering `block_index`, without running the rest.

    `sequences` is (B, L, 4) on the model's device. Everything after the block
    is skipped, so for early blocks this costs little more than the encoder.
    """
    block = model.tower.blocks[int(block_index)]

    def stop(_module, args, _kwargs):
        raise _Stop(args[0])

    handle = block["mha"].register_forward_pre_hook(stop, with_kwargs=True)
    try:
        model.predict(sequences, organism, resolutions=(128,), embeddings_only=True)
    except _Stop as caught:
        return caught.stream
    finally:
        handle.remove()
    raise RuntimeError(f"block {block_index} was never reached")


def _chunks(items: Sequence, size: int) -> Iterable[Sequence]:
    for start in range(0, len(items), size):
        yield items[start:start + size]


@torch.inference_mode()
def run_conditions(
    model: Any,
    block_index: int,
    sequences: torch.Tensor,
    conditions: Sequence[Condition],
    readout: Readout,
    *,
    heads: Sequence[str],
    organism: int = 0,
    max_batch: int = 4,
    memo: dict | None = None,
) -> dict[str, Any]:
    """Run every condition and return {condition name: readout}.

    `sequences` is (n, L, 4) on the model's device, in any float or uint8
    dtype. Conditions are run in order, `max_batch` at a time. For every
    edited condition memo["edit_norms"][name] holds the L2 size of the change
    the edit asked for and of the change that reached the stream.
    """
    if max_batch < 1:
        raise ValueError("max_batch must be positive")
    names = [c.name for c in conditions]
    if len(set(names)) != len(names):
        raise ValueError("condition names must be unique")
    memo = {} if memo is None else memo
    results: dict[str, Any] = {}
    device = sequences.device
    def forward(chunk: Sequence[Condition]) -> dict:
        index = torch.tensor([c.sequence for c in chunk], device=device)
        batch = sequences.index_select(0, index).float()
        positions = [None if c.bins is None else
                     torch.as_tensor(np.asarray(c.bins), dtype=torch.long, device=device)
                     for c in chunk]

        def edit_stream(x: torch.Tensor) -> torch.Tensor:
            if all(c.edit is None for c in chunk):
                return x
            # predict() runs the model under bf16 autocast; the SAE must not
            # inherit it, or TopK supports and reconstructions shift.
            with torch.autocast(x.device.type, enabled=False):
                return _edited(x)

        def _edited(x: torch.Tensor) -> torch.Tensor:
            out = x.clone()
            for b, (condition, where) in enumerate(zip(chunk, positions)):
                if condition.edit is None:
                    continue
                rows = x[b, where].float()
                replacement = condition.edit(rows, memo)
                if replacement.shape != rows.shape:
                    raise ValueError(f"{condition.name}: edit changed the row shape")
                stored = replacement.to(x.dtype)
                # In a bf16 stream a change below half an ulp is rounded away;
                # recording both norms shows when that happens.
                memo.setdefault("edit_norms", {})[condition.name] = {
                    "intended": float((replacement - rows).norm()),
                    "realized": float((stored.float() - rows).norm()),
                    "stream_dtype": str(x.dtype).removeprefix("torch."),
                }
                out[b, where] = stored
            return out

        with ResidualEdit(model, block_index, edit_stream):
            return model.predict(batch, organism, resolutions=(128,),
                                 heads=tuple(heads), channels_last=True)

    def run(chunk: Sequence[Condition]) -> None:
        # A chunk that does not fit (fragmentation can make the same batch fit
        # one window and not the next) is split in half and retried; edits are
        # deterministic, so conditions re-run in a smaller chunk give the same
        # memo entries.
        try:
            outputs = forward(chunk)
        except torch.cuda.OutOfMemoryError:
            if len(chunk) == 1:
                raise
            outputs = None
        if outputs is None:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            half = (len(chunk) + 1) // 2
            run(chunk[:half])
            run(chunk[half:])
            return
        for b, condition in enumerate(chunk):
            results[condition.name] = readout(outputs, b)
        del outputs

    for chunk in _chunks(list(conditions), max_batch):
        run(chunk)
    return results


# --------------------------------------------------------------------------
# SAE edits
# --------------------------------------------------------------------------

CodeChange = Callable[[torch.Tensor, dict], torch.Tensor]


def sae_edit(sae: Any, record: str, change: CodeChange | None = None, *,
             preserve_error: bool = False) -> RowEdit:
    """Edit rows through the SAE, optionally after changing codes.

    The unchanged codes are stored as memo[record], so later conditions can
    refer to them. `change(codes, memo)` gets a copy and returns the codes to
    decode.

    By default the rows are replaced by the decoded codes, which also drops
    what the SAE cannot reconstruct; the right baseline is then the plain
    reconstruction. With `preserve_error` only the change is applied:
    ``rows + decode(changed) - decode(codes)``, so the reconstruction error
    stays in the stream, an unchanged code leaves the rows untouched, and the
    right baseline is the unedited model.
    """
    def edit(rows: torch.Tensor, memo: dict) -> torch.Tensor:
        codes, params = sae.encode_codes(rows)
        memo[record] = codes
        if change is None:
            return rows if preserve_error else sae.decode_codes(codes, params)
        changed = sae.decode_codes(change(codes.clone(), memo), params)
        if preserve_error:
            return rows + (changed - sae.decode_codes(codes, params))
        return changed
    return edit


def zero_feature(feature: int) -> CodeChange:
    return zero_features([feature])


def zero_features(features: Sequence[int]) -> CodeChange:
    """Silence several features at once (joint ablation)."""
    index = list(dict.fromkeys(int(f) for f in features))
    if not index:
        raise ValueError("no features to zero")

    def change(codes, _memo):
        codes[:, index] = 0
        return codes
    return change


def set_feature_from(feature: int, source: str) -> CodeChange:
    """Set `feature` to the value it had in the condition recorded as `source`."""
    def change(codes, memo):
        if source not in memo:
            raise KeyError(f"condition {source!r} must run before it is referenced")
        codes[:, feature] = memo[source][:, feature]
        return codes
    return change


def add_to_feature(feature: int, amount: float) -> CodeChange:
    """Raise a feature by a fixed amount, whatever its current value."""
    def change(codes, _memo):
        codes[:, feature] = codes[:, feature] + amount
        return codes
    return change


def clamp_min_feature(feature: int, level: float) -> CodeChange:
    def change(codes, _memo):
        codes[:, feature] = torch.clamp_min(codes[:, feature], level)
        return codes
    return change


# --------------------------------------------------------------------------
# Readouts
# --------------------------------------------------------------------------

def head_tensor(outputs: dict, head: str) -> torch.Tensor:
    """(B, bins, tracks) prediction of one head at 128 bp."""
    value = outputs[head]
    if isinstance(value, dict):
        value = value[128]
    if not torch.is_tensor(value):
        raise ValueError(f"unexpected output for head {head!r}: {type(value)}")
    return value


def track_mean_profile(heads: Sequence[str]) -> Readout:
    """Per head, the mean over tracks at every bin: {head: (bins,) float32}."""
    def readout(outputs, b):
        return {head: head_tensor(outputs, head)[b].float().mean(-1).cpu().numpy()
                for head in heads}
    return readout
