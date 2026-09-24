# Experimental AlphaGenome SAE ablation tools

This folder contains source and tests for exploratory SAE interventions on the AlphaGenome tower stream:

- feature ablation/injection at explicitly selected 128-bp bins, with the original activation's SAE reconstruction residual preserved;
- PLS-bin control selection, cohort freezing, and output validation;
- one-bin injection at a preselected protein-coding gene TSS and gene-level readouts.

These scripts are research tooling, not a claim that causal validation is complete. Some plans are exploratory and fold-specific. Model weights, SAE checkpoints, activation caches, annotations, manifests, and output archives are external inputs and are intentionally not included here. Keep each run's frozen plan and source hashes with its results.

From the repository root, run the unit tests with the project environment:

```sh
PYTHONPATH=.:scripts/ablation .venv/bin/python -m pytest scripts/ablation/test_*.py
```
