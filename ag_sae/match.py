"""Feature matching with exact sparse ranks and genomic permutation nulls."""
import argparse
import json
import os
import numpy as np
import pandas as pd
from scipy import sparse
from scipy.stats import rankdata
from .config import ANALYSIS_VERSION
from .data import log, read_annotation, ActivationStore
from .statistics import (average_precision, domain_precision_recall_f1,
                         event_enrichment_with_block_bootstrap,
                         genomic_block_ids)
def rank_once_dense(F, feat_chunk=1024):
    """F:(N,nf) real -> R:(N,nf) float32 average ranks per column (exact:
    N<1.6e7 so integer/half ranks are exact in float32)."""
    N, nf = F.shape
    R = np.empty((N, nf), np.float32)
    for s in range(0, nf, feat_chunk):
        e = min(s + feat_chunk, nf)
        R[:, s:e] = rankdata(F[:, s:e], axis=0).astype(np.float32)
    return R

def auroc_from_dense_ranks(R, Y, chunk=512):
    """R:(N,nf) float32 ranks, Y:(N,nc) bool -> (nf,nc) AUROC."""
    N, nf = R.shape
    Yf = Y.astype(np.float64); n1 = Yf.sum(0); n0 = N - n1
    valid = (n1 > 0) & (n0 > 0)
    out = np.full((nf, Y.shape[1]), np.nan)
    for s in range(0, nf, chunk):
        e = min(s + chunk, nf)
        r1 = R[:, s:e].T.astype(np.float64) @ Yf
        auc = (r1 - n1 * (n1 + 1) / 2) / (n1 * n0)
        auc[:, ~valid] = np.nan
        out[s:e] = auc
    return out

def build_sparse_rank_struct(Fcsc):
    """Fcsc: scipy CSC (N,nf), nonneg (TopK+relu). Returns (Rsp, Bpat, base, N):
       Rsp  = CSC of average ranks at the true-nonzero entries,
       Bpat = CSC binary pattern of true nonzeros,
       base = (nf,) average rank of each feature's zero block."""
    Fcsc = Fcsc.tocsc(copy=True)
    Fcsc.sum_duplicates()
    Fcsc.eliminate_zeros()
    if not np.isfinite(Fcsc.data).all() or (Fcsc.data < 0).any():
        raise ValueError("Sparse AUROC requires finite, nonnegative activations")
    N, nf = Fcsc.shape
    Rdat = np.zeros_like(Fcsc.data, dtype=np.float64)
    base = np.zeros(nf)
    for f in range(nf):
        s, e = Fcsc.indptr[f], Fcsc.indptr[f + 1]
        vals = Fcsc.data[s:e]
        pos = vals > 0                       # relu may store exact zeros
        m = int(pos.sum())
        base[f] = (N - m + 1) / 2.0
        if m:
            rr = np.full(vals.shape, base[f], dtype=np.float64)
            rr[pos] = rankdata(vals[pos]) + (N - m)
            Rdat[s:e] = rr
        else:
            Rdat[s:e] = base[f]
    Rsp = sparse.csc_matrix((Rdat, Fcsc.indices, Fcsc.indptr), shape=(N, nf))
    Bpat = Fcsc.copy(); Bpat.data = (Fcsc.data > 0).astype(np.float64)
    return Rsp, Bpat, base, N

def auroc_from_sparse_ranks(Rsp, Bpat, base, N, Y):
    """Exact AUROC (nf,nc) reusing precomputed sparse ranks."""
    Yf = Y.astype(np.float64)
    n1 = Yf.sum(0); n0 = N - n1
    valid = (n1 > 0) & (n0 > 0)
    r1_nz = Rsp.T @ Yf                        # rank-sum over nonzero rows
    k1 = Bpat.T @ Yf                          # # concept-pos rows that are nonzero
    r1 = r1_nz + base[:, None] * (n1[None, :] - k1)   # zero-block contribution
    auc = (r1 - n1 * (n1 + 1) / 2) / (n1 * n0)
    auc[:, ~valid] = np.nan
    return auc

ORIGINAL_8 = ["cCRE_PLS", "cCRE_pELS", "cCRE_dELS", "cCRE_CTCF", "TF_binding",
              "TSS_promoter", "splice_donor", "splice_acceptor"]

META_COLS = {"chrom", "bin_start", "bin_end", "split", "n_mask", "window_idx"}

def sparse_auroc_ceiling(prevalence, firing_rate, sign=1):
    """Directional upper bound for nonnegative scores with a zero mass.

    Positive associations separate concept-positive bins; inverse associations
    separate concept-negative bins. Population-average sparsity is not a bound
    on the firing rate of any individual feature.
    """
    p, q, sign = np.broadcast_arrays(prevalence, firing_rate, sign)
    if np.any((p <= 0) | (p >= 1) | (q < 0) | (q > 1)):
        raise ValueError("Require 0 < prevalence < 1 and 0 <= firing_rate <= 1")
    mass = np.where(sign >= 0, p, 1 - p)
    return np.minimum(1.0, 0.5 + q / (2 * mass))

def genomic_runs(coords):
    """Return positional indices for contiguous 128-bp runs on each chromosome.

    Gaps from split selection or N masking end a run. This avoids pretending
    that bins separated by unobserved genomic territory are adjacent.
    """
    coords = coords.reset_index(drop=True)
    runs = []
    for _, group in coords.groupby("chrom", sort=False):
        group = group.sort_values("bin_start")
        starts, ends = group.bin_start.to_numpy(), group.bin_end.to_numpy()
        if np.any(ends - starts != 128) or np.any(starts[1:] < ends[:-1]):
            raise ValueError("Permutation coordinates must be nonoverlapping 128-bp bins")
        cuts = np.flatnonzero(starts[1:] != ends[:-1]) + 1
        runs.extend(np.split(group.index.to_numpy(), cuts))
    return runs

def permuted_indices(n_rows, rng, mode, runs):
    """Shift all concept columns together, retaining within-run label structure.

    Circular shifts include the identity (required for the full shift group).
    The wrap boundary and stationarity within runs remain null assumptions.
    Global shuffling is available only for historical sensitivity comparisons.
    """
    if mode == "global":
        return rng.permutation(n_rows)
    if mode != "circular":
        raise ValueError(f"Unknown null mode: {mode}")
    perm = np.arange(n_rows)
    for idx in runs:
        perm[idx] = np.roll(idx, int(rng.integers(len(idx))))
    return perm

def resolve_concepts(ann, panel_path=None):
    """Return (all_concepts, primary, exploratory) from the matrix schema.

    v1 davranisi (panel_path=None): primary = ORIGINAL_8 ile kesisim.
    v2 davranisi (panel_path verilir): primary = JSON'daki 'confirmatory'
    listesi; negatif kontrol ve kesifsel konseptler puanlanir ama dogrulayici
    teste girmez. Panel build_concepts_v2.py tarafindan, sonuclara BAKILMADAN,
    yalnizca yayginlik kuralindan uretilir."""
    concepts = [c for c in ann.columns if c not in META_COLS]
    if panel_path:
        import json as _json
        with open(panel_path) as _f:
            _p = _json.load(_f)
        benchmark = list(_p.get("benchmark_only", []))
        if benchmark:
            missing = [c for c in benchmark if c not in concepts]
            if missing:
                raise ValueError(f"Benchmark panel concepts missing from matrix: {missing}")
            # A benchmark matrix may be built beside a richer project matrix.
            # Restrict scoring to the declared external test family so the
            # frozen primary hypotheses and its FDR family cannot leak in.
            concepts = benchmark
        primary = [c for c in _p.get("confirmatory", []) if c in concepts]
        exploratory = [c for c in concepts if c not in primary]
        # v3: negatif kontrolun ADI panelden okunur. Onceden "cCRE_dELS" sabit
        # yazilmisti ve panel degisince kontrol SESSIZCE atlaniyordu -- yani
        # kod calisiyor ama saglama yapilmiyordu. Artik ad panelden gelir ve
        # panelde negatif kontrol yoksa asagida yuksek sesle hata verilir.
        roles = {"negative_control": [c for c in _p.get("negative_control", []) if c in concepts],
                 "artifact_control": [c for c in _p.get("artifact_control", []) if c in concepts],
                 "confirmatory": primary,
                 "benchmark_only": [c for c in benchmark if c in concepts],
                 "sparse_event": [c for c in _p.get("sparse_event", []) if c in concepts],
                 "event_enrichment": _p.get("event_enrichment")}
        print(f"[panel] {os.path.basename(panel_path)}: dogrulayici {len(primary)}, "
              f"kesifsel {len(exploratory)}, negatif kontrol {roles['negative_control']}")
        return concepts, primary, exploratory, roles
    primary = [c for c in ORIGINAL_8 if c in concepts]
    exploratory = [c for c in concepts if c not in ORIGINAL_8]
    # panel yoksa v1 davranisi: negatif kontrol adi eski sabitten gelir
    roles = {"negative_control": [c for c in ["cCRE_dELS"] if c in concepts],
             "artifact_control": [], "confirmatory": primary}
    return concepts, primary, exploratory, roles

def cmd_match(argv):
    import pandas as pd
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["raw", "sae"], required=True)
    ap.add_argument("--sae", help="SAE checkpoint (mode=sae)")
    ap.add_argument("--act-dir", default="activations")
    ap.add_argument("--ann", default="annotation_matrix.parquet")
    ap.add_argument("--split", default="test")
    ap.add_argument("--n-control", type=int, default=1000,
                    help="Permutation draws; 319 resolves rank-one BH at q=.05 for 16 concepts")
    ap.add_argument("--out", default="match_out")
    ap.add_argument("--null-mode", choices=["circular", "global"], default="circular")
    ap.add_argument("--null-seed", type=int, default=0)
    ap.add_argument("--panel", default=None,
                    help="concept_panel_v2.json; verilmezse v1 ORIGINAL_8 paneli kullanilir")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--batch-size", type=int, default=1024, help="SAE encoding batch size")
    ap.add_argument("--max-sae-nnz", type=int, default=250_000_000,
                    help="upper bound for materialized sparse entries (use 0 to disable)")
    ap.add_argument("--allow-large-sae", action="store_true",
                    help="override the sparse-entry memory guard; may exhaust RAM")
    args = ap.parse_args(argv)
    os.makedirs(args.out, exist_ok=True)

    ann = read_annotation(args.ann)
    CONCEPTS, PRIMARY, EXPLORATORY, ROLES = resolve_concepts(ann, args.panel)
    log(f"concepts: {len(CONCEPTS)} total | primary {len(PRIMARY)} | exploratory {len(EXPLORATORY)}")
    sel = (ann.split == args.split).values & ann.n_mask.values
    Y = ann.loc[sel, CONCEPTS].values.astype(bool)
    if args.n_control < 1:
        raise ValueError("--n-control must be positive")
    constant = [c for c, y in zip(CONCEPTS, Y.T) if not y.any() or y.all()]
    if constant:
        raise ValueError(f"AUROC is undefined for constant concepts in {args.split}: {constant}")
    runs = genomic_runs(ann.loc[sel, ["chrom", "bin_start", "bin_end"]])
    if args.null_mode == "circular" and not any(len(r) > 1 for r in runs):
        raise ValueError("No contiguous genomic runs available for circular permutations")

    # load activations for split, apply same mask
    data = ActivationStore(args.act_dir, args.split, ann)
    if args.batch_size < 1:
        raise ValueError("batch-size must be positive")
    if len(data) != len(Y):
        raise ValueError("Activation/label row mismatch")
    log(f"{args.split}: {len(data)} kept bins, {data.dim} raw dims")

    # Build the feature representation. RAW = dense real-valued neurons;
    # SAE = TopK-sparse -> assembled directly as CSC (never densified: the
    # dense (N, 49152) form would be ~54 GB; sparse is <100 MB).
    if args.mode == "raw":
        X = np.empty((len(data), data.dim), dtype=np.float32)
        offset = 0
        for batch in data.batches(args.batch_size):
            X[offset:offset + len(batch)] = batch
            offset += len(batch)
        nfeat = data.dim
        feat_names = [f"neuron_{i}" for i in range(nfeat)]
        tag = "raw"
        sparse_mode = False
    else:
        import torch
        from .sae import TopKSAE
        if not args.sae:
            raise ValueError("--sae checkpoint is required in SAE mode")
        model = TopKSAE.from_checkpoint(args.sae, args.device)
        if model.d_in != data.dim:
            raise ValueError("SAE/activation dimension mismatch")
        nfeat, k = model.n_feat, model.k
        estimated_nnz = len(data) * k
        if args.max_sae_nnz < 0:
            raise ValueError("--max-sae-nnz must be nonnegative")
        if (args.max_sae_nnz and estimated_nnz > args.max_sae_nnz
                and not args.allow_large_sae):
            raise ValueError(
                f"SAE match would materialize up to {estimated_nnz:,} sparse entries "
                f"({len(data):,} rows x k={k}); limit is {args.max_sae_nnz:,}. "
                "Use a smaller k, blockwise matching, or explicitly pass "
                "--allow-large-sae after checking available RAM.")
        log(f"SAE sparse-entry upper bound: {estimated_nnz:,}")
        # collect COO triplets of TopK activations across row-batches
        rows_l, cols_l, vals_l = [], [], []
        offset = 0
        with torch.no_grad():
            for batch in data.batches(args.batch_size):
                xb = torch.from_numpy(batch).to(args.device)
                topv, topi = model.encode_topk(xb)
                nb = xb.shape[0]
                rr = (torch.arange(nb, device=xb.device).unsqueeze(1)
                      .expand(nb, k) + offset)
                rows_l.append(rr.reshape(-1).cpu().numpy())
                cols_l.append(topi.reshape(-1).cpu().numpy())
                vals_l.append(topv.reshape(-1).cpu().numpy().astype(np.float32))
                offset += nb
        rows = np.concatenate(rows_l); cols = np.concatenate(cols_l)
        vals = np.concatenate(vals_l)
        Fcsc = sparse.csc_matrix((vals, (rows, cols)),
                                 shape=(len(data), nfeat), dtype=np.float32)
        Fcsc.sum_duplicates()
        feat_names = [f"sae_{i}" for i in range(nfeat)]
        tag = f"sae_{os.path.basename(args.sae)}"
        sparse_mode = True

    log(f"scoring {nfeat} features x {len(CONCEPTS)} concepts by AUROC "
        f"({'sparse' if sparse_mode else 'dense'} rank-once)...")

    if sparse_mode:
        Rsp, Bpat, base, N = build_sparse_rank_struct(Fcsc)
        score = lambda Ym: auroc_from_sparse_ranks(Rsp, Bpat, base, N, Ym)
    else:
        R = rank_once_dense(X)
        score = lambda Ym: auroc_from_dense_ranks(R, Ym)

    A = score(Y)                              # (feat, concept) — HAM (katlanmamis) AUROC
    # fold AUROC<0.5 (anti-correlated features are still informative): use max(a,1-a)
    A_dir = np.maximum(A, 1 - A)

    if sparse_mode:
        _Fnz = Fcsc.copy()
        _Fnz.eliminate_zeros()            # topk+relu acik sifir saklayabilir; onlari at
        firing_rate = (np.diff(_Fnz.indptr) / float(Fcsc.shape[0])).astype(np.float64)
        del _Fnz
    else:
        # Ham noron temel cizgisi: aktivasyonlar neredeyse hic tam sifir degildir,
        # dolayisiyla q ~ 1.0 ve tavan ~ 1.0 cikar. Bu KASITLI: normalizasyon
        # seyrek koda hakkini verir, yogun temel cizgiye avantaj vermez.
        firing_rate = (np.count_nonzero(X, axis=0) / float(X.shape[0])).astype(np.float64)

    # best feature per concept
    best_feat = np.nanargmax(A_dir, axis=0)
    best_auroc = np.nanmax(A_dir, axis=0)

    # Default null: independently rotate labels within contiguous genomic runs.
    # Each rotation preserves prevalence and cross-concept co-occurrence, and
    # retains local runs up to the circular boundary. Max over ALL features is
    # recalculated for every draw to include feature-selection multiplicity.
    rng = np.random.default_rng(args.null_seed)
    null_best = np.zeros((args.n_control, len(CONCEPTS)))
    for t in range(args.n_control):
        perm = permuted_indices(Y.shape[0], rng, args.null_mode, runs)
        Ap = score(Y[perm])                   # (feat, concept) under permuted labels
        Ap = np.maximum(Ap, 1 - Ap)
        null_best[t] = np.nanmax(Ap, axis=0)
        if (t + 1) % 25 == 0:
            log(f"  control perm {t+1}/{args.n_control}")
    ctrl_95 = np.nanpercentile(null_best, 95, axis=0)

    recovered = (best_auroc >= 0.70) & (best_auroc > ctrl_95)

    # Normalize using the selected feature's measured rate and association sign.
    # k/n_features describes average sparsity only; it cannot establish that
    # the best feature cannot exceed a fixed AUROC threshold.
    _nc      = len(CONCEPTS)
    p_conc   = np.asarray(Y).mean(axis=0).astype(np.float64)          # yayginlik
    q_best   = firing_rate[best_feat]                                  # eslesen ozelligin q'su
    match_sign = np.where(A[best_feat, np.arange(_nc)] >= 0.5, 1, -1).astype(np.int8)

    # AUPRC complements AUROC for rare annotations.  It is evaluated on the
    # AUROC-selected feature, so it is an explicitly descriptive companion;
    # permutation p-values below continue to use the selection-aware AUROC null.
    best_auprc = np.empty(_nc, dtype=np.float64)
    domain_precision = np.full(_nc, np.nan)
    domain_recall = np.full(_nc, np.nan)
    domain_f1 = np.full(_nc, np.nan)
    event_log_enrichment = np.full(_nc, np.nan)
    event_enrichment_ci_low = np.full(_nc, np.nan)
    event_enrichment_ci_high = np.full(_nc, np.nan)
    event_activation_mean = np.full(_nc, np.nan)
    event_background_mean = np.full(_nc, np.nan)
    event_n = np.full(_nc, np.nan)
    event_background_n = np.full(_nc, np.nan)
    event_bootstrap_valid = np.full(_nc, np.nan)
    eval_coords = ann.loc[sel, ["chrom", "bin_start"]]
    event_names = set(ROLES.get("sparse_event", []))
    event_config = ROLES.get("event_enrichment")
    if event_names:
        required_event_keys = {
            "radius_bins", "sigma_bins", "block_bp", "bootstrap_replicates",
            "bootstrap_seed", "confidence",
        }
        if not isinstance(event_config, dict) or not required_event_keys <= set(event_config):
            raise ValueError("Sparse-event concepts require a complete event_enrichment panel config")
        event_blocks = genomic_block_ids(
            eval_coords.chrom.to_numpy(), eval_coords.bin_start.to_numpy(),
            int(event_config["block_bp"]))
    for c in range(_nc):
        if sparse_mode:
            selected_scores = np.asarray(Fcsc.getcol(best_feat[c]).toarray()).ravel()
        else:
            selected_scores = X[:, best_feat[c]]
        oriented = selected_scores * match_sign[c]
        best_auprc[c] = average_precision(Y[:, c], oriented)
        # The paper's domain F1 uses activation > 0.  This threshold has a
        # defined meaning only for positive TopK SAE activations.  Inverse
        # matches and dense raw neurons are left unreported instead of silently
        # inventing a test-set threshold.
        if sparse_mode and match_sign[c] > 0:
            domain = domain_precision_recall_f1(
                selected_scores > 0, Y[:, c], eval_coords.chrom.to_numpy(),
                eval_coords.bin_start.to_numpy())
            domain_precision[c] = domain["precision"]
            domain_recall[c] = domain["recall"]
            domain_f1[c] = domain["f1"]
        if sparse_mode and match_sign[c] > 0 and CONCEPTS[c] in event_names:
            event = event_enrichment_with_block_bootstrap(
                selected_scores, Y[:, c], eval_coords.chrom.to_numpy(),
                eval_coords.bin_start.to_numpy(), event_blocks,
                radius_bins=int(event_config["radius_bins"]),
                sigma_bins=float(event_config["sigma_bins"]),
                n_bootstrap=int(event_config["bootstrap_replicates"]),
                seed=int(event_config["bootstrap_seed"]),
                confidence=float(event_config["confidence"]))
            event_log_enrichment[c] = event["log_enrichment"]
            event_enrichment_ci_low[c] = event["ci_low"]
            event_enrichment_ci_high[c] = event["ci_high"]
            event_activation_mean[c] = event["event_mean"]
            event_background_mean[c] = event["background_mean"]
            event_n[c] = event["n_events"]
            event_background_n[c] = event["n_background_bins"]
            event_bootstrap_valid[c] = event["bootstrap_valid"]
    # Signed raw neurons do not satisfy the nonnegative sparse-score assumption.
    ceiling = (sparse_auroc_ceiling(p_conc, q_best, match_sign) if sparse_mode
               else np.ones(_nc))
    captured = np.clip((best_auroc - 0.5) / np.maximum(ceiling - 0.5, 1e-12), 0.0, 1.0)

    # v2 birincil geri kazanim olcutu
    recovered_v2 = (captured >= 0.50) & (best_auroc > ctrl_95)

    # BH controls FDR under independence or suitable positive dependence.
    # Overlapping annotations alone do not prove that dependence condition.
    FDR_Q = 0.05
    n_perm = int(null_best.shape[0])
    # ampirik p: +1 duzeltmesi (Phipson & Smyth 2010) -- p asla 0 olamaz
    p_emp = (1.0 + (null_best >= best_auroc[None, :]).sum(axis=0)) / (1.0 + n_perm)

    # The frozen primary panel and the external paper benchmark are distinct
    # multiplicity families.  A benchmark-only run receives its own BH step;
    # it is never pooled with (or allowed to enlarge) the primary family.
    benchmark_names = ROLES.get("benchmark_only", [])
    fdr_names = PRIMARY if PRIMARY else benchmark_names
    fdr_indices = [i for i, c in enumerate(CONCEPTS) if c in fdr_names]
    m_tests = len(fdr_indices)
    p_min = 1.0 / (1.0 + n_perm)
    # q/m is the rank-one BH threshold, not a necessary threshold for every
    # rejection: a group of discoveries can pass a higher-rank threshold.
    if m_tests and p_min > FDR_Q / m_tests:
        n_need = int(np.ceil(m_tests / FDR_Q)) - 1
        log(f"WARNING: {n_perm} permutations cannot resolve the rank-one BH "
            f"threshold; use >= {n_need} for that resolution. Joint rejections "
            "at higher ranks can still be possible.")

    bh_pass = np.zeros(len(CONCEPTS), dtype=bool)
    if m_tests:
        order = sorted(fdr_indices, key=lambda i: p_emp[i])       # p'ye gore artan
        kmax = 0
        for rank, i in enumerate(order, start=1):
            if p_emp[i] <= FDR_Q * rank / m_tests:
                kmax = rank
        for i in order[:kmax]:                                    # BH: ilk kmax reddedilir
            bh_pass[i] = True

    # v3 birincil olcut: v2 yakalama esigi VE panel-genisliginde FDR kontrolu
    recovered_v3 = recovered_v2 & bh_pass

    # panel tag + cell-type parse for the exploratory TF tracks (CELLLINE_TF)
    def parse(c):
        if c in PRIMARY:
            return ("primary", "", "")
        if c in ROLES.get("benchmark_only", []):
            return ("benchmark_only", "", "")
        cell, _, tf = c.partition("_")
        return ("exploratory", cell, tf)
    panel, cell_line, tf = zip(*[parse(c) for c in CONCEPTS])
    res = pd.DataFrame(dict(analysis_version=ANALYSIS_VERSION, concept=CONCEPTS, panel=panel, cell_line=cell_line, tf=tf,
                            best_feature=[feat_names[i] for i in best_feat],
                            best_auroc=best_auroc, control_95=ctrl_95, recovered=recovered,
                            match_sign=match_sign,          # +1 duz, -1 ters korelasyonlu
                            firing_rate=q_best,             # eslesen ozelligin atesleme orani
                            prevalence=p_conc,              # konsept yayginligi
                            best_feature_auprc=best_auprc,
                            auprc_lift=best_auprc / p_conc,
                            rank_biserial=2 * best_auroc - 1,
                            domain_precision=domain_precision,
                            domain_recall=domain_recall,
                            domain_f1=domain_f1,
                            event_log_enrichment=event_log_enrichment,
                            event_enrichment_ci_low=event_enrichment_ci_low,
                            event_enrichment_ci_high=event_enrichment_ci_high,
                            event_activation_mean=event_activation_mean,
                            event_background_mean=event_background_mean,
                            event_n=event_n,
                            event_background_n=event_background_n,
                            event_bootstrap_valid=event_bootstrap_valid,
                            auroc_ceiling=ceiling,          # bu q,p ile ulasilabilir en yuksek AUROC
                            captured=captured,              # yakalanan sinyalin kesri [0,1]
                            recovered_v2=recovered_v2,
                            p_empirical=p_emp, bh_pass=bh_pass,
                            recovered_v3=recovered_v3))
    res.to_csv(os.path.join(args.out, f"match_{tag}_{args.split}.csv"), index=False)
    # full AUROC matrix (all features reported)
    np.save(os.path.join(args.out, f"auroc_matrix_{tag}_{args.split}.npy"), A_dir.astype(np.float32))
    np.save(os.path.join(args.out, f"auroc_raw_{tag}_{args.split}.npy"), A.astype(np.float32))
    np.save(os.path.join(args.out, f"null_best_{tag}_{args.split}.npy"), null_best.astype(np.float32))
    np.save(os.path.join(args.out, f"firing_rate_{tag}_{args.split}.npy"), firing_rate.astype(np.float32))
    log("\n" + res.to_string(index=False))

    pidx = [i for i, c in enumerate(CONCEPTS) if c in PRIMARY]
    prim_recovered = int(recovered[pidx].sum())
    prim_mean = float(np.nanmean(best_auroc[pidx])) if pidx else None
    prim_recovered_v2 = int(recovered_v2[pidx].sum())
    prim_captured = float(np.nanmean(captured[pidx])) if pidx else None
    if pidx:
        log(f"\nPRIMARY recovered (v1, AUROC>=0.70): {prim_recovered}/{len(PRIMARY)} "
            f"| mean best-AUROC {prim_mean:.4f}")
        log(f"PRIMARY recovered (v2, yakalama>=0.50): {prim_recovered_v2}/{len(PRIMARY)} "
            f"| mean yakalama {prim_captured:.4f}")
    elif ROLES.get("benchmark_only"):
        log(f"\nEXTERNAL BENCHMARK: {len(ROLES['benchmark_only'])} concepts; "
            "no primary claims or intervention hypotheses changed")
    # NEGATIF KONTROL saglamasi (on kayit v2 §3.2 + v3 eki).
    # Kontrolun ADI panelden gelir; sabit yazilmaz. Panelde negatif kontrol
    # tanimli degilse bu SESSIZ GECILMEZ -- yuksek sesle bildirilir, cunku
    # saglamasi yapilmamis bir kosu ile saglamasi gecmis bir kosu ayni
    # gorunmemelidir.
    neg_names = ROLES.get("negative_control", [])
    neg_failed = None
    if not neg_names and not ROLES.get("benchmark_only"):
        log("!!! UYARI: panelde negatif kontrol yok. Eslestirme prosedurunun "
            "yanlis-pozitif orani BU KOSUDA OLCULMEDI.")
    else:
        neg_failed = False
        for _n in neg_names:
            _i = CONCEPTS.index(_n)
            log(f"negatif kontrol {_n}: best_auroc={best_auroc[_i]:.4f} "
                f"tavan={ceiling[_i]:.4f} yakalama={captured[_i]:.4f} "
                f"p={p_emp[_i]:.4f} recovered_v2={bool(recovered_v2[_i])}")
            if bool(recovered_v2[_i]):
                neg_failed = True
                log(f"!!! NEGATIF KONTROL BASARISIZ: {_n} recovered_v2=True. "
                    "Eslestirme prosedurunun kendisi yanlis pozitif uretiyor; "
                    "on kayit v2 §3.2 geregi bu kosunun geri kazanim "
                    "sonuclari YORUMLANAMAZ.")
    if EXPLORATORY and not ROLES.get("benchmark_only"):
        eidx = [i for i, c in enumerate(CONCEPTS) if c in EXPLORATORY]
        log(f"EXPLORATORY (cell-type TF) recovered: {int(recovered[eidx].sum())}/{len(EXPLORATORY)} "
            f"| mean best-AUROC {float(np.nanmean(best_auroc[eidx])):.4f}")
    with open(os.path.join(args.out, f"summary_{tag}_{args.split}.json"), "w") as f:
        json.dump(dict(analysis_version=ANALYSIS_VERSION,
                       null_mode=args.null_mode, null_seed=args.null_seed,
                       null_run_lengths=[len(r) for r in runs],
                       mode=args.mode, split=args.split, tag=tag,
                       primary_concepts=PRIMARY, exploratory_concepts=EXPLORATORY,
                       benchmark_concepts=ROLES.get("benchmark_only", []),
                       fdr_family=("primary" if PRIMARY else
                                   "external_benchmark" if benchmark_names else "none"),
                       fdr_family_concepts=fdr_names,
                       sparse_event_concepts=sorted(event_names),
                       event_enrichment_config=event_config,
                       n_recovered_primary=prim_recovered, n_primary=len(PRIMARY),
                       mean_best_auroc_primary=prim_mean,
                       n_recovered_exploratory=int(recovered[[i for i, c in enumerate(CONCEPTS)
                                                              if c in EXPLORATORY]].sum()) if EXPLORATORY else 0,
                       mean_best_auroc_all=float(np.nanmean(best_auroc)),
                       n_recovered_primary_v2=prim_recovered_v2,
                       mean_captured_primary=prim_captured,
                       negative_control_concepts=neg_names,
                       negative_control_failed=neg_failed,
                       n_recovered_primary_v3=int(recovered_v3[pidx].sum()),
                       n_recovered_benchmark_v3=int(recovered_v3[
                           [i for i, c in enumerate(CONCEPTS) if c in benchmark_names]
                       ].sum()) if benchmark_names else 0,
                       fdr_q=FDR_Q, n_permutations=n_perm,
                       bh_rank_one_reachable=bool(p_min <= FDR_Q / max(m_tests, 1)),
                       fdr_reachable=bool(p_min <= FDR_Q),
                       per_concept=res.to_dict("records")), f, indent=2)
