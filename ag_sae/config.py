"""Analysis identity, dimensions and frozen-panel validation."""
ANALYSIS_VERSION = "v3fix3"
BIN_BP = 128
WIN = 131072
BINS = WIN // BIN_BP
DIM = 3072
ORG_HUMAN = 0
def validate_panel(reference, candidate):
    """Reject changes to frozen roles, consensus threshold, or data provenance."""
    roles = ("confirmatory", "exploratory", "negative_control", "artifact_control")
    for key in roles:
        if sorted(reference.get(key, [])) != sorted(candidate.get(key, [])):
            raise ValueError(f"Frozen panel mismatch: {key}")
    for key in ("encode_min_cells", "rule", "gencode", "provenance"):
        if reference.get(key) != candidate.get(key):
            raise ValueError(f"Frozen panel mismatch: {key}")
