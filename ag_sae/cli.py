"""AlphaGenome-SAE command line interface dispatcher.

Heavy dependencies load only when the requested stage is invoked.
"""

from __future__ import annotations

import importlib
import sys

STAGES: dict[str, tuple[str, str]] = {
    "extract": ("extract", "main"),
    "train": ("train", "main"),
    "intervene": ("intervene", "cmd_intervene"),
    "campaign": ("matrix", "cmd_matrix"),
    "matrix": ("matrix", "cmd_matrix"),
}


def main(argv: list[str] | None = None) -> int:
    """Dispatch command to target stage module."""
    args = list(sys.argv[1:] if argv is None else argv)
    if not args or args[0] in ("-h", "--help"):
        print("Usage: python -m ag_sae <stage> [options]\nAvailable stages: " + ", ".join(STAGES.keys()))
        return 0 if args else 2
    stage = args[0]
    if stage not in STAGES:
        print(f"Unknown stage: {stage!r}. Choose from: {', '.join(STAGES.keys())}", file=sys.stderr)
        return 2

    module_name, func_name = STAGES[stage]
    module = importlib.import_module(f"ag_sae.{module_name}")
    func = getattr(module, func_name)
    return int(func(args[1:]) or 0)


if __name__ == "__main__":
    sys.exit(main())
