The intervention entry point cannot run because it calls a nonexistent SAE loader. Conditional paths also cause dry-run side effects and can produce stale or misleading analysis summaries.

Full review comments:

- [P1] Load the SAE checkpoint with an implemented API — /Users/azizcetin/PycharmProjects/AlphaGenome-SAE/ag_sae/intervene.py:919-919
  Every intervention run that reaches model loading fails here because `BorzoiSAE` has no `from_checkpoint` method. Training writes `best.pt` through `save_inference_checkpoint`; reconstruct the SAE from that checkpoint format before starting discovery.

- [P2] Keep dry runs from syncing completed configurations — /Users/azizcetin/PycharmProjects/AlphaGenome-SAE/ag_sae/matrix.py:50-54
  When both local summaries already exist, `--dry-run` enters this branch and executes `rclone copy` before reaching the dry-run check. A supposedly non-executing preview can upload data or fail because `rclone` is unavailable; check `dry_run` before any sync.

- [P2] Invalidate summaries when intervention inputs change — /Users/azizcetin/PycharmProjects/AlphaGenome-SAE/ag_sae/intervene.py:930-934
  If the same output directory is rerun with a changed SAE checkpoint, model weights, annotations, or window plan, this existence-only check reuses the previous triad summary; the analogous gene-summary check does the same. That can report stale results or combine analyses from different inputs. Record input identity and skip only when it matches.

- [P2] Do not substitute pinned IDs for undiscovered concepts — /Users/azizcetin/PycharmProjects/AlphaGenome-SAE/ag_sae/intervene.py:240-242
  When calibration contains no positive-versus-negative bins for a concept, this branch assigns a hard-coded feature ID to the current independently trained SAE. The subsequent ablation reports that ID as a discovered biological feature despite having no evidence it represents the concept. Mark the concept unavailable or require a validated feature instead.

- [P2] Exclude windows lacking concept bins from paired tests — /Users/azizcetin/PycharmProjects/AlphaGenome-SAE/ag_sae/intervene.py:370-377
  If a test window has no bins for a concept, `eval_mask` is empty and `delta_on_concept` is recorded as `0.0`. The final paired test then treats that unmeasured concept effect as an observed zero, biasing its mean and p-values. Omit that window for the concept or record it as missing.