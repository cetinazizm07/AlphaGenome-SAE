Sparse-grid remapping introduces incorrect negative labels during feature discovery, which can change downstream intervention results. Focused tests passed (16 passed, 2 skipped), but do not cover this case.

Review comment:

- [P2] Exclude unannotated positions from calibration negatives — /Users/azizcetin/PycharmProjects/AlphaGenome-SAE/ag_sae/intervene.py:261-264
  For sparse annotations produced by sampling positions or dropping ambiguous bins, the new mapping marks every missing position as False. Consequently, `codes[~mask]` treats unobserved positions as known negatives, potentially changing the selected SAE feature. It even creates apparent contrast when every supplied annotation is positive. Track which grid positions have annotations and restrict the negative mean to observed rows explicitly labelled False, or require dense annotations.