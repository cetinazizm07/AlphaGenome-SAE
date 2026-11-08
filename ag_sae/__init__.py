"""Sparse autoencoders on AlphaGenome activations."""

import os as _os

# Batched 1 Mb forwards fragment the CUDA cache (an A100 run hit OOM with 33 GB
# reserved but free); expandable segments avoid that. Set before torch starts
# CUDA, and only if the user has not chosen something else.
_os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

__all__ = ["FrozenSAE", "MatchResult", "match_concepts", "read_ccre_bed"]


def __getattr__(name: str):
    # Imported lazily: `python -m ag_sae.concepts` would otherwise load the
    # module twice and warn, and importing the package should not pull in scipy.
    if name in __all__:
        from ag_sae import concepts

        return getattr(concepts, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
