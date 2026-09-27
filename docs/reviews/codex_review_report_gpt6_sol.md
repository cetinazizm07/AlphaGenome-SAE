The campaign currently reports successful runs without executing interventions, and direct intervention fails while loading the SAE. Several additional issues can abort execution or silently invalidate the reported analysis.

Full review comments:

- [P1] Invoke the intervention CLI in campaign subprocesses — /Users/azizcetin/PycharmProjects/AlphaGenome-SAE/ag_sae/matrix.py:56-62
  For every campaign step, `python -m ag_sae.intervene` only loads definitions: `intervene.py` has no `__main__` entry point. The subprocess exits successfully without producing either summary, yet the campaign counts the configuration as complete. Invoke the registered `ag_sae intervene` CLI or add a module entry point.

- [P1] Load SAE checkpoints through an implemented method — /Users/azizcetin/PycharmProjects/AlphaGenome-SAE/ag_sae/intervene.py:865-866
  When `intervene` is invoked directly, every configuration fails here: `BorzoiSAE` has no `from_checkpoint` method. Training writes `best.pt` via `save_inference_checkpoint`, so this stage needs a loader for that checkpoint format before discovery or intervention can run.

- [P1] Disable autograd during feature discovery — /Users/azizcetin/PycharmProjects/AlphaGenome-SAE/ag_sae/intervene.py:217-220
  With a normally constructed `BorzoiSAE`, its parameters require gradients, while the SAE encoding here runs outside `torch.inference_mode()`. As soon as a calibration window has both positive and negative bins, `codes[mask].mean(...).cpu().numpy()` raises because the tensor requires grad. Wrap SAE inference in a no-grad context or detach before conversion.

- [P1] Include manifest and annotation contents in the gene cache key — /Users/azizcetin/PycharmProjects/AlphaGenome-SAE/ag_sae/intervene.py:560-569
  When two folds have the same manifest length and annotation split counts, this key is identical despite different window coordinates and `cCRE_PLS` labels. The committed fold manifests all have the same split sizes, and the default cache directory is shared, so a later fold can silently reuse another fold's eligible genes and TSS coordinates. Hash the selection-relevant data or partition the cache by fold.

- [P1] Discover features on windows separate from the test set — /Users/azizcetin/PycharmProjects/AlphaGenome-SAE/ag_sae/intervene.py:873-881
  For the default run, `windows` comes from the held-out test split and is passed both to feature discovery and to triad ablation. The first ten evaluated windows therefore select the feature IDs used to test those same windows, biasing the reported effect and p-values, including the shuffled control. Use a separate calibration or dev split for discovery.

- [P2] Propagate Drive sync failures — /Users/azizcetin/PycharmProjects/AlphaGenome-SAE/ag_sae/matrix.py:81-86
  If `rclone copy` fails, its exit status is discarded and the campaign still reports success. On the next run, the two local summaries cause this configuration to be skipped, so the failed upload is never retried. Check the copy result and only mark the step complete after a successful sync.

- [P2] Do not report identical nonzero effects as null — /Users/azizcetin/PycharmProjects/AlphaGenome-SAE/ag_sae/stats.py:81-89
  When every paired delta is the same nonzero value, this branch reports both p-values as 1.0. Those observations all point in one direction; a Wilcoxon signed-rank test is valid and would not give a null result. Only all-zero differences warrant this shortcut, and `safe_wilcoxon` has the same identical-nonzero check.

- [P2] Handle a cohort with no measurable genes — /Users/azizcetin/PycharmProjects/AlphaGenome-SAE/ag_sae/intervene.py:806-809
  If no transcript passes eligibility, or every injection is marked unmeasurable, `diffs` is empty and `compute_paired_statistics` raises `ValueError`. That prevents the cohort summary from being written and aborts the campaign instead of recording a zero-measured cohort. Handle this case before requesting paired statistics.

- [P2] Align the shuffled-control evaluation mask with edited bins — /Users/azizcetin/PycharmProjects/AlphaGenome-SAE/ag_sae/intervene.py:286-290
  When shuffled dELS bins fall outside `cCRE_any`, the shuffled feature is ablated only at `cCRE_any` bins but its `delta_on_concept` is averaged over the shuffled bins. Most of that control region receives no local edit, so its effect is not comparable to the other concept effects. Edit the shuffled bins too, or evaluate the control on the bins actually edited.