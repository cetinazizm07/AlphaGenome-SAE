"""Command dispatch. Heavy dependencies load only for the selected stage."""
import importlib
import sys

STAGES = {
    "folds": ("prepare.folds", "cmd_folds"),
    "matrix": ("prepare.annotations", "cmd_matrix"),
    "fetch": ("prepare.sources", "cmd_fetch"),
    "panel": ("prepare.annotations", "cmd_panel"),
    "paper": ("paper", "cmd_paper"),
    "extract": ("backbone", "cmd_extract"),
    "sae": ("train", "cmd_sae"),
    "match": ("match", "cmd_match"),
    "intervene": ("intervene", "cmd_intervene"),
}


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv or argv[0] in ("-h", "--help"):
        print("Usage: python -m ag_sae <stage> [options]\nStages: " + ", ".join(STAGES))
        return 0 if argv else 2
    if argv[0] not in STAGES:
        print(f"Unknown stage: {argv[0]}", file=sys.stderr)
        return 2
    module, function = STAGES[argv[0]]
    return getattr(importlib.import_module(f"ag_sae.{module}"), function)(argv[1:])
