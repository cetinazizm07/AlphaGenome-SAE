The intervention entry point calls a nonexistent checkpoint loader. Additional calibration, annotation-alignment, and resumption defects can invalidate or discard results, while scoped bridge reviews fail with custom instructions.

Full review comments:

- [P1] Load SAE checkpoints through an implemented API — /Users/azizcetin/PycharmProjects/AlphaGenome-SAE/ag_sae/intervene.py:940-940
  `BorzoiSAE` in `ag_sae/sae.py` has no `from_checkpoint` method. Consequently, every intervention run reaching this line raises `AttributeError`, and matrix campaigns stop at their first configuration. Implement loading for the existing `{state, recipe}` checkpoint format before calling this API.

- [P1] Exclude the gene cohort from fallback calibration — /Users/azizcetin/PycharmProjects/AlphaGenome-SAE/ag_sae/intervene.py:926-931
  When the manifest lacks train/validation windows, this fallback can select test windows outside the first 40 ablation windows. However, `run_gene_injection_cohort` receives the entire manifest and evaluates eligible genes in those same windows. Feature discovery therefore uses gene-cohort test data despite the claimed separation. Restrict calibration to a non-test split or fail when none is available.

- [P2] Map annotation coordinates onto the activation grid — /Users/azizcetin/PycharmProjects/AlphaGenome-SAE/ag_sae/intervene.py:304-304
  For annotation tables derived from extraction, which samples positions and drops ambiguous bins, dataframe row indices are not activation indices. This selects incorrect positions whenever a preceding bin is missing; subsequent boolean masking also fails because predictions contain the full grid. Discovery has the same alignment assumption. Reindex annotations onto the complete window grid using `bin_start`, or explicitly reject incomplete grids before inference.

- [P2] Avoid checkpoint-independent fallback feature IDs — /Users/azizcetin/PycharmProjects/AlphaGenome-SAE/ag_sae/intervene.py:239-242
  When calibration provides no contrast for a concept, these fixed IDs are reused across every layer and seed even though latent numbering is checkpoint-specific. This can silently label an unrelated feature as the requested concept. It can also crash: the default tower SAE has 6,144 features, making enhancer ID 7,205 invalid. Fail or mark the concept unavailable rather than substituting an unverified feature.

- [P2] Validate planned windows against the requested fold — /Users/azizcetin/PycharmProjects/AlphaGenome-SAE/ag_sae/intervene.py:1008-1009
  If this shared `local_plan.json` exists, it overrides manifest-derived windows for every `--fold`. A plan created for another fold can therefore select windows outside the current test split. The ablation then receives missing or incomplete test annotations, or evaluates territory that is not held out for the selected model. Scope the default plan by fold and validate its windows against the current test manifest.

- [P2] Write summary completion markers atomically — /Users/azizcetin/PycharmProjects/AlphaGenome-SAE/ag_sae/intervene.py:873-873
  An interruption or disk-full error during this write can leave an empty or truncated summary file. Both `run_single_configuration` and `run_matrix_campaign` treat file existence alone as completion, so restarting can permanently skip the unfinished stage and upload corrupt results. The triad summary has the same problem. Write summaries through temporary files with atomic replacement and validate existing markers before skipping.

- [P2] Handle scoped reviews with custom instructions explicitly — /Users/azizcetin/PycharmProjects/AlphaGenome-SAE/scripts/codex_bridge.py:140-141
  Combining `--instructions` with either `--base` or `--commit` appends a positional prompt to the scoped review command. Codex CLI 0.157.1 rejects both combinations with an argument-conflict error and exits before reviewing anything; only the `--uncommitted` conflict is handled here. Handle these combinations explicitly rather than forwarding an invalid command.