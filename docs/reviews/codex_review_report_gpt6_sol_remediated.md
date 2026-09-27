The intervention entry point cannot load the SAE checkpoint, so the main campaign fails. Several conditional paths can also produce stale cohorts, invalid feature assignments, or misleading completion and significance reports.

Full review comments:

- [P1] Load the SAE checkpoint through an implemented API — /Users/azizcetin/PycharmProjects/AlphaGenome-SAE/ag_sae/intervene.py:907-907
  Every intervention run that reaches model loading fails here: `BorzoiSAE` has no `from_checkpoint` method, while training writes `best.pt` through `save_inference_checkpoint`. Reconstruct the SAE from that checkpoint's state and recipe before starting feature discovery.

- [P2] Retry remote sync for locally complete configurations — /Users/azizcetin/PycharmProjects/AlphaGenome-SAE/ag_sae/matrix.py:51-54
  If `rclone` fails after both summaries are written, a campaign retry takes this skip path and never attempts the upload again, yet reports the configuration as complete. Check or retry remote synchronization before skipping a locally complete configuration.

- [P2] Do not assign feature zero when calibration has no evidence — /Users/azizcetin/PycharmProjects/AlphaGenome-SAE/ag_sae/intervene.py:228-234
  When none of the calibration windows has both positive and negative bins for a concept, `diffs[c]` is empty and that concept is silently assigned feature 0. The subsequent ablation then measures an unrelated latent, potentially the same one for multiple concepts; fail or omit that concept instead of reporting it as discovered.

- [P2] Hash all cohort-selection inputs in the cache key — /Users/azizcetin/PycharmProjects/AlphaGenome-SAE/ag_sae/intervene.py:567-577
  If manifest coordinates or `cCRE_PLS` labels change after row 25 while the manifest length stays the same, this key is unchanged and `eligible_genes` returns the previous cohort and TSS coordinates from disk. Hash the full selection-relevant columns, not just their first 25 rows.

- [P2] Keep calibration off evaluated test windows — /Users/azizcetin/PycharmProjects/AlphaGenome-SAE/ag_sae/intervene.py:895-902
  With fewer than ten train/validation windows and at most forty test windows, this fallback selects `windows[:10]` for discovery and then evaluates ablations on those same windows. That leaks evaluation labels into feature selection; use the available non-test windows or stop when disjoint calibration is unavailable.

- [P2] Avoid a zero t-test p-value for one observation — /Users/azizcetin/PycharmProjects/AlphaGenome-SAE/ag_sae/stats.py:90-93
  When only one gene is measured or one ablation window is supplied, any nonzero delta enters this identical-values branch and is reported with an infinite t-statistic and `p=0`. A one-sample t-test cannot establish significance from one observation; handle `len(arr) < 2` separately.