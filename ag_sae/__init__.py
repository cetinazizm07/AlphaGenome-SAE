"""Sparse autoencoders on AlphaGenome activations."""

__all__ = ["FrozenSAE", "MatchResult", "match_concepts", "read_ccre_bed"]


def __getattr__(name: str):
    # Imported lazily: `python -m ag_sae.concepts` would otherwise load the
    # module twice and warn, and importing the package should not pull in scipy.
    if name in __all__:
        from ag_sae import concepts

        return getattr(concepts, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
