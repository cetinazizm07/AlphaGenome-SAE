"""Genomic overlaps, consensus annotations and panel roles."""
import argparse
import gzip
import json
import os
import time
import numpy as np
import pandas as pd
from ..config import validate_panel
from ..data import log, read_annotation, load_chrom_fasta
def read_bed(path):
    """BED (gz olabilir) -> {kromozom: (start,end) dizisi}, 0-tabanli yarim acik.
    BED zaten 0-tabanli yarim aciktir; donusum gerekmez."""
    per = {}
    opn = gzip.open if path.endswith(".gz") else open
    with opn(path, "rt") as f:
        for line in f:
            if not line or line[0] in "#t":      # '#' yorum, 'track'/'browser'
                continue
            p = line.split("\t", 3)
            if len(p) < 3:
                continue
            try:
                per.setdefault(p[0], []).append((int(p[1]), int(p[2])))
            except ValueError:
                continue
    return {c: np.array(sorted(v), dtype=np.int64) for c, v in per.items()}

def mark_bins(ann, peaks):
    """Vectorized per-chromosome overlap: bin flagged if any peak overlaps."""
    flag = np.zeros(len(ann), dtype=np.uint8)
    for c, sub_idx in ann.groupby("chrom").groups.items():
        iv = peaks.get(c)
        if iv is None or len(iv) == 0:
            continue
        idx = ann.index.get_indexer(sub_idx)
        bs = ann["bin_start"].values[idx]
        be = ann["bin_end"].values[idx]
        ps, pe = iv[:, 0], iv[:, 1]
        # for each bin, does any peak overlap? peak.start < bin.end AND peak.end > bin.start
        # use searchsorted on peak starts sorted; check the candidate window
        order = np.argsort(ps); ps_s = ps[order]; pe_s = pe[order]
        # a peak overlaps bin[bs,be) if ps < be and pe > bs
        # find peaks with ps < be  -> right = searchsorted(ps_s, be, 'left')
        right = np.searchsorted(ps_s, be, side="left")
        hit = np.zeros(len(idx), dtype=bool)
        # cumulative max of pe over sorted-by-start peaks lets us test "any pe > bs"
        cummax_pe = np.maximum.accumulate(pe_s) if len(pe_s) else pe_s
        for i in range(len(idx)):
            r = right[i]
            if r == 0:
                continue
            if cummax_pe[r - 1] > bs[i]:
                hit[i] = True
        flag[idx] = hit.astype(np.uint8)
    return flag

def cmd_matrix(argv):
    ap = argparse.ArgumentParser()
    ap.add_argument("--ann", required=True)
    ap.add_argument("--peaks", nargs="+", required=True,
                    help="NAME=path.bed.gz entries, one per new concept")
    ap.add_argument("--out", default="annotation_matrix_expanded.parquet")
    args = ap.parse_args(argv)

    ann = read_annotation(args.ann)
    log(f"loaded matrix {ann.shape}, cols={list(ann.columns)}")
    base_concepts = [c for c in ann.columns
                     if c not in ("chrom", "bin_start", "bin_end", "split", "n_mask")]
    log(f"{len(base_concepts)} existing concepts")

    for spec in args.peaks:
        assert "=" in spec, f"bad --peaks entry {spec!r}; use NAME=path"
        name, path = spec.split("=", 1)
        assert name not in ann.columns, f"column {name} already exists"
        assert os.path.exists(path), f"missing {path}"
        t0 = time.time()
        peaks = read_bed(path)
        npk = sum(len(v) for v in peaks.values())
        ann[name] = mark_bins(ann, peaks)
        # n_mask==True marks KEPT (valid) bins; ==False marks N-masked bins
        prev = ann.loc[ann.n_mask, name].mean() if "n_mask" in ann else ann[name].mean()
        log(f"+ {name}: {npk} peaks, prevalence(kept bins)={prev:.4%}  ({time.time()-t0:.1f}s)")

    ann.to_parquet(args.out, index=False)
    new = [c for c in ann.columns
           if c not in ("chrom", "bin_start", "bin_end", "split", "n_mask")]
    log(f"wrote {args.out}: {ann.shape}, {len(new)} concepts total")

PREV_MIN = 0.001

PREV_MAX = 0.030

NEGATIVE_CONTROL = ["shuffled_PLS"]

FORCE_EXPLORATORY = ["cCRE_pELS", "cCRE_dELS"]

SHUFFLE_SEED = 20260910

V4_EXTRA_CLASSES = {"CA-TF": "cCRE_CA_TF", "CA-H3K4me3": "cCRE_CA_H3K4me3",
                    "TF": "cCRE_TF", "CA": "cCRE_CA"}

V4_KNOWN_CLASSES = {"PLS", "pELS", "dELS", "CA-CTCF", *V4_EXTRA_CLASSES}

CELL_LINES = ("K562", "GM12878", "HepG2", "A549", "H1", "MCF-7")

def is_cell_type_specific(c):
    return any(c.startswith(cl + "_") for cl in CELL_LINES)

CPG_MIN_LEN = 200

CPG_MIN_GC = 0.50

CPG_MIN_OE = 0.60

CPG_STEP = 100

def parse_gtf_features(path, wanted_types):
    """GTF -> {tip: {kromozom: (start,end) int dizisi}}, 0-tabanli yarim acik."""
    out = {t: {} for t in wanted_types}
    opn = gzip.open if path.endswith(".gz") else open
    with opn(path, "rt") as f:
        for line in f:
            if line[0] == "#":
                continue
            p = line.split("\t", 8)
            if len(p) < 8:
                continue
            ftype = p[2]
            if ftype not in out:
                continue
            # GTF 1-tabanli kapali -> 0-tabanli yarim acik
            out[ftype].setdefault(p[0], []).append((int(p[3]) - 1, int(p[4])))
    for t in out:
        out[t] = {c: np.array(sorted(v), dtype=np.int64) for c, v in out[t].items()}
    return out

def encode_consensus_columns(ann, encode_dir, min_cells):
    """Her histon isareti icin: her hucre hattinda bin'leri isaretle, topla,
    >=min_cells esigini uygula. Sonuc hucre-tipinden BAGIMSIZ bir konsepttir.

    Ayrica blacklist tek dosya oldugu icin dogrudan kolon olur."""
    man_p = os.path.join(encode_dir, "encode_sources.json")
    if not os.path.exists(man_p):
        raise SystemExit(f"ENCODE kunyesi yok: {man_p} (once fetch_encode.py calistir)")
    man = json.load(open(man_p))
    cols, prov = {}, {}
    by_mark = {}
    for e in man["files"]:
        by_mark.setdefault(e["mark"], []).append(e)

    for mark, entries in sorted(by_mark.items()):
        # The combined cCRE registry is an input to class-specific annotations,
        # not an additional biological concept in the frozen panel.
        if mark == "cCRE_v4":
            continue
        # Tekil anotasyonlar (blacklist, konsensus DHS, TSS): hucre hatti yok,
        # konsensus alinmaz, dosya dogrudan kolon olur.
        if len(entries) == 1 and entries[0].get("cell") == "-":
            e = entries[0]
            p = os.path.join(encode_dir, f"{e['label']}__{e['accession']}.bed.gz")
            if not os.path.exists(p):
                log(f"  {e['label']}: dosya yok, atlaniyor"); continue
            cols[e["label"]] = mark_bins(ann, read_bed(p))
            prov[e["label"]] = [e["accession"]]
            log(f"  {e['label']}: tekil anotasyon, yayginlik {cols[e['label']].mean():.4%}")
            continue

        counts = np.zeros(len(ann), dtype=np.uint8)
        used = []
        for e in entries:
            p = os.path.join(encode_dir, f"{e['label']}__{e['accession']}.bed.gz")
            if not os.path.exists(p):
                log(f"  {e['label']}: dosya yok, atlaniyor"); continue
            counts += mark_bins(ann, read_bed(p))
            used.append(e["accession"])
        if not used:
            continue
        if len(used) < min_cells:
            raise ValueError(f"{mark}: {len(used)} files cannot meet frozen threshold {min_cells}")
        thr = min_cells
        name = f"{mark}_consensus"
        cols[name] = (counts >= thr).astype(np.uint8)
        prov[name] = used
        log(f"  {name}: {len(used)} hucre hatti, esik >={thr}, "
            f"yayginlik {cols[name].mean():.4%}")
    return cols, prov

def parse_first_exons(path):
    """Her transkriptin ILK ekzonunu dondurur (exon_number 1). Yonu dikkate alir:
    GTF'te exon_number zaten transkript yonunde numaralandirilmistir."""
    per = {}
    opn = gzip.open if path.endswith(".gz") else open
    with opn(path, "rt") as f:
        for line in f:
            if line[0] == "#":
                continue
            p = line.split("\t", 8)
            if len(p) < 9 or p[2] != "exon":
                continue
            attr = p[8]
            i = attr.find("exon_number ")
            if i < 0:
                continue
            num = attr[i + 12:].split(";", 1)[0].strip().strip('"')
            if num != "1":
                continue
            per.setdefault(p[0], []).append((int(p[3]) - 1, int(p[4])))
    return {c: np.array(sorted(v), dtype=np.int64) for c, v in per.items()}

def cpg_islands(seq, lo, hi):
    """[lo,hi) araliginda GGF olcutlerini saglayan pencereleri birlestirip dondurur.
    Kayan 200 bp pencere, 100 bp adim; gecen komsu pencereler birlestirilir."""
    s = np.frombuffer(seq[lo:hi].encode("ascii"), dtype=np.uint8)
    isC = (s == ord("C")); isG = (s == ord("G"))
    # CpG dinukleotidi: pozisyon i'de C ve i+1'de G
    isCpG = np.zeros(len(s), bool)
    if len(s) > 1:
        isCpG[:-1] = isC[:-1] & isG[1:]
    cC, cG, cCpG = (np.concatenate(([0], np.cumsum(x))) for x in (isC, isG, isCpG))
    hits = []
    for st in range(0, max(0, len(s) - CPG_MIN_LEN + 1), CPG_STEP):
        en = st + CPG_MIN_LEN
        nC, nG = cC[en] - cC[st], cG[en] - cG[st]
        nCpG = cCpG[en] - cCpG[st]
        gc = (nC + nG) / CPG_MIN_LEN
        if gc < CPG_MIN_GC or nC == 0 or nG == 0:
            continue
        oe = (nCpG * CPG_MIN_LEN) / (nC * nG)      # gozlenen/beklenen CpG
        if oe >= CPG_MIN_OE:
            hits.append((lo + st, lo + en))
    # ortusen/komsu pencereleri birlestir
    merged = []
    for a, b in hits:
        if merged and a <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], b)
        else:
            merged.append([a, b])
    return np.array(merged, dtype=np.int64) if merged else np.zeros((0, 2), np.int64)

def read_ccre_v4(path):
    """v4 agnostik cCRE BED'ini SINIFA GORE ayirir.

    Dosya biciminde son kolon sinif etiketidir (PLS / pELS / dELS / CA-CTCF /
    CA-H3K4me3 / CA-TF / TF / CA). Donus: {sinif: {kromozom: (start,end) dizisi}}
    """
    per = {}
    opn = gzip.open if path.endswith(".gz") else open
    with opn(path, "rt") as f:
        for line in f:
            if not line or line[0] in "#t":
                continue
            p = line.rstrip("\n").split("\t")
            if len(p) < 4:
                continue
            cls = p[-1]
            try:
                per.setdefault(cls, {}).setdefault(p[0], []).append((int(p[1]), int(p[2])))
            except ValueError:
                continue
    out = {}
    for cls, d in per.items():
        out[cls] = {c: np.array(sorted(v), dtype=np.int64) for c, v in d.items()}
    return out

def shuffled_intervals(per_chrom, ann, seed):
    """Araliklari KROMOZOM ICINDE rastgele tasir; uzunluklar ve kromozom basina
    aralik sayisi korunur, dolayisiyla yayginlik (ve AUROC tavani) degismez.

    Tasima araligi, o kromozomda ANALIZ EDILEN bin'lerin kapsadigi bolgedir --
    tum kromozom degil. Boylece kontrol, gercek konseptle ayni alanda yasar.
    """
    rng = np.random.default_rng(seed)
    span = (ann.groupby("chrom")
               .agg(lo=("bin_start", "min"), hi=("bin_end", "max")))
    out = {}
    for c, iv in per_chrom.items():
        if c not in span.index or len(iv) == 0:
            continue
        lo, hi = int(span.loc[c, "lo"]), int(span.loc[c, "hi"])
        lens = iv[:, 1] - iv[:, 0]
        lens = lens[lens < (hi - lo)]              # bolgeye sigmayanlari at
        if len(lens) == 0:
            continue
        starts = rng.integers(lo, hi - lens, size=len(lens), endpoint=False)
        out[c] = np.array(sorted(zip(starts, starts + lens)), dtype=np.int64)
    return out

def cmd_panel(argv):
    ap = argparse.ArgumentParser()
    ap.add_argument("--ann", required=True)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--encode-dir", default=None,
                    help="fetch_encode.py ciktisi; ENCODE konsensus konseptleri buradan")
    ap.add_argument("--min-cells", type=int, default=4,
                    help="bir bin'in konsensus sayilmasi icin gereken hucre hatti sayisi")
    ap.add_argument("--prev-split", choices=["train", "all"], default="train",
                    help="Yayginligin hangi bin'lerden hesaplanacagi. "
                         "'train': o fold'un egitim bolmesi (v2 davranisi). "
                         "'all': butun kromozomlar -- panel FOLD'DAN BAGIMSIZ "
                         "olur, uc fold'da ayni roller kullanilir. Uc fold "
                         "birlikte raporlanacaksa 'all' kullanin (bkz. asagidaki not).")
    ap.add_argument("--ccre-v4", default=None,
                    help="ENCODE v4 agnostik cCRE BED (ENCFF420VPZ). Verilirse "
                         "kullanilmayan 4 sinif kolon olarak eklenir ve "
                         "shuffled_PLS negatif kontrolu uretilir.")
    ap.add_argument("--gencode", default=None,
                    help="ISTEGE BAGLI. GENCODE gen-modeli konseptleri (kodon/ekzon/polyA). "
                         "Varsayilan panel ENCODE'dur; bu verilmezse eklenmez.")
    ap.add_argument("--polya", default=None)
    ap.add_argument("--fasta-dir", default=None, help="verilmezse CpG_island atlanir")
    ap.add_argument("--out-ann", required=True)
    ap.add_argument("--out-json", required=True)
    ap.add_argument("--out-csv", default="concept_prevalence_v2.csv")
    ap.add_argument("--reference-panel", help="Require exact agreement with this frozen panel")
    args = ap.parse_args(argv)

    ann = read_annotation(args.ann)
    man = pd.read_parquet(args.manifest)
    log(f"annotation_matrix: {ann.shape} | manifest: {man.shape}")
    assert {"chrom", "bin_start", "bin_end"} <= set(ann.columns), "bin koordinatlari yok"

    provenance = {}

    if args.encode_dir:
        log(f"ENCODE konsensus kolonlari (esik >={args.min_cells}/6 hucre hatti)...")
        enc_cols, enc_prov = encode_consensus_columns(ann, args.encode_dir, args.min_cells)
        for name, v in enc_cols.items():
            ann[name] = v
        provenance.update(enc_prov)

    if args.ccre_v4:
        log("v4 cCRE kaydi sinifa gore ayristiriliyor...")
        v4 = read_ccre_v4(args.ccre_v4)
        bilinmeyen = set(v4) - V4_KNOWN_CLASSES
        if bilinmeyen:
            log(f"  UYARI: beklenmeyen sinif etiketleri: {sorted(bilinmeyen)}")
        # (a) kullanilmayan 4 sinif
        for cls, kolon in V4_EXTRA_CLASSES.items():
            if cls not in v4:
                log(f"  UYARI: {cls} dosyada yok, {kolon} atlaniyor")
                continue
            ann[kolon] = mark_bins(ann, v4[cls])
            provenance[kolon] = {"kaynak": os.path.basename(args.ccre_v4),
                                 "v4_sinif": cls,
                                 "eleman": int(sum(len(a) for a in v4[cls].values()))}
            log(f"  {kolon}: yayginlik {ann[kolon].mean():.4%}")
        # (b) negatif kontrol: PLS'in kromozom ici karistirilmisi
        if "PLS" in v4:
            sh = shuffled_intervals(v4["PLS"], ann, SHUFFLE_SEED)
            ann["shuffled_PLS"] = mark_bins(ann, sh)
            provenance["shuffled_PLS"] = {
                "kaynak": "cCRE_PLS araliklarinin kromozom ici rastgele tasinmasi",
                "tohum": SHUFFLE_SEED,
                "not": "yayginlik ve AUROC tavani PLS ile eslesir; biyoloji yok. "
                       "Bir ozellik bunu null ustunde yakalarsa eslestirme bozuktur."}
            log(f"  shuffled_PLS: yayginlik {ann['shuffled_PLS'].mean():.4%} "
                f"(cCRE_PLS: {ann['cCRE_PLS'].mean():.4%})"
                if "cCRE_PLS" in ann.columns else
                f"  shuffled_PLS: yayginlik {ann['shuffled_PLS'].mean():.4%}")
        else:
            log("  UYARI: PLS sinifi yok, shuffled_PLS uretilemedi")

    new_cols = {}
    if not args.gencode:
        log("GENCODE verilmedi — gen-modeli konseptleri eklenmiyor")
    else:
        log("GENCODE ayristiriliyor...")
        feats = parse_gtf_features(args.gencode, {"start_codon", "stop_codon"})
        new_cols = {"start_codon": feats["start_codon"], "stop_codon": feats["stop_codon"]}
        log("ilk ekzonlar...")
        new_cols["first_exon"] = parse_first_exons(args.gencode)
    if args.gencode and args.polya:
        pa = parse_gtf_features(args.polya, {"polyA_site"})
        if any(len(v) for v in pa["polyA_site"].values()):
            new_cols["polyA_site"] = pa["polyA_site"]
        else:   # bazi surumlerde tip adi farkli olabilir -> tum satirlari al
            log("uyari: polyA_site tipi bulunamadi, polyA dosyasindaki tum araliklar aliniyor")
            allp = {}
            with gzip.open(args.polya, "rt") as f:
                for line in f:
                    if line[0] == "#":
                        continue
                    p = line.split("\t", 8)
                    if len(p) >= 5:
                        allp.setdefault(p[0], []).append((int(p[3]) - 1, int(p[4])))
            new_cols["polyA_site"] = {c: np.array(sorted(v), np.int64) for c, v in allp.items()}

    if args.fasta_dir:
        log("CpG adalari diziden hesaplaniyor (GGF 1987)...")
        cpg = {}
        for c in sorted(ann.chrom.unique()):
            try:
                seq = load_chrom_fasta(args.fasta_dir, c)
            except FileNotFoundError:
                log(f"  {c}: FASTA yok, atlaniyor")
                continue
            sub = ann[ann.chrom == c]
            lo, hi = int(sub.bin_start.min()), int(sub.bin_end.max())
            cpg[c] = cpg_islands(seq, lo, hi)
            log(f"  {c}: {len(cpg[c])} ada")
            del seq
        new_cols["CpG_island"] = cpg

    for name, per in new_cols.items():
        ann[name] = mark_bins(ann, per)
        log(f"kolon eklendi: {name}  (yayginlik {ann[name].mean():.4%})")

    if args.prev_split == "all":
        is_train = np.ones(len(ann), dtype=bool)
        log(f"yayginlik BUTUN bin'lerden (fold'dan bagimsiz panel): {len(ann)}")
    elif "split" in ann.columns:
        is_train = (ann["split"].values == "train")
        log(f"train bin sayisi: {int(is_train.sum())} / {len(ann)}")
    else:
        n_win = len(man)
        assert len(ann) == n_win * 1024, f"beklenmeyen boyut: {len(ann)} vs {n_win}*1024"
        is_train = man["split"].values[np.repeat(np.arange(n_win), 1024)] == "train"
        log(f"train bin sayisi: {int(is_train.sum())} / {len(ann)}")

    META = {"chrom", "bin_start", "bin_end", "split", "n_mask", "window_idx"}
    concept_cols = [c for c in ann.columns
                    if c not in META and ann[c].dtype.kind in "uib"]
    is_train &= ann.n_mask.to_numpy()
    if not is_train.any():
        raise ValueError("No valid bins available for panel prevalence")
    prev = {c: float(ann.loc[is_train, c].mean()) for c in concept_cols}

    rows, panel = [], {"confirmatory": [], "exploratory": [],
                       "negative_control": [], "artifact_control": []}
    for c in sorted(prev, key=lambda x: prev[x]):
        p = prev[c]
        if c == "ENCODE_blacklist":
            # Artefakt kontrolu: biyolojik bir konsept degil. Bir SAE ozelligi
            # buraya eslesirse model olcum artefakti kodluyor demektir.
            # Dogrulayici teste GIRMEZ, ayri raporlanir.
            rol = "artifact_control"
        elif c in NEGATIVE_CONTROL:
            rol = "negative_control"
        elif c in FORCE_EXPLORATORY:
            rol = "exploratory"
        elif is_cell_type_specific(c):
            rol = "exploratory"          # A5/A6 atamasi korunur (yukaridaki nota bak)
        elif PREV_MIN <= p <= PREV_MAX:
            rol = "confirmatory"
        else:
            rol = "exploratory" if p > PREV_MAX else "excluded_too_rare"
        q_mean = 32 / (3072 * 16)
        rows.append(dict(concept=c, prevalence=p,
                         ceiling_mean_feature=min(1.0, 0.5 + q_mean / (2 * max(p, 1e-12))),
                         q_needed_for_auroc_070=2 * p * 0.20,
                         role=rol))
        if rol in panel:
            panel[rol].append(c)

    P = pd.DataFrame(rows)
    panel["rule"] = dict(prev_min=PREV_MIN, prev_max=PREV_MAX,
                         source="preregistration_v2.md §3.1",
                         frozen_before_run=True)
    panel["prevalence_population"] = "valid_bins"
    panel["gencode"] = os.path.basename(args.gencode) if args.gencode else None
    panel["encode_min_cells"] = args.min_cells
    panel["provenance"] = provenance          # konsept -> ENCODE accession listesi
    if args.reference_panel:
        with open(args.reference_panel) as f:
            validate_panel(json.load(f), panel)
    P.to_csv(args.out_csv, index=False)
    ann.to_parquet(args.out_ann, index=False)
    with open(args.out_json, "w") as f:
        json.dump(panel, f, indent=2)

    log("\n" + P.to_string(index=False))
    log(f"\ndogrulayici : {panel['confirmatory']}")
    log(f"kesifsel    : {panel['exploratory']}")
    log(f"negatif ktrl: {panel['negative_control']}")
    log(f"yazildi: {args.out_ann}, {args.out_json}, {args.out_csv}")
