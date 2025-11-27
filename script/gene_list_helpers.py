import glob
import os
from typing import Any, Dict, List
import pandas as pd


_META_GENE_COLUMNS = {
    "tile",
    "patient",
    "x",
    "y",
    "px",
    "py",
    "row",
    "col",
    "barcode",
    "spot",
    "spot_id",
    "spot_index",
    "section",
    "sample",
}


def _is_meta_gene_column(name: Any) -> bool:
    try:
        s = str(name).strip().lower()
    except Exception:
        s = str(name)
    if s in _META_GENE_COLUMNS:
        return True
    for prefix in ("coord_", "spatial_", "pixel_", "image_", "img_", "spot_"):
        if s.startswith(prefix):
            return True
    for suffix in ("_x", "_y", "_row", "_col", "_coord", "_coords"):
        if s.endswith(suffix):
            return True
    return False


_GENE_SETS: Dict[str, List[str]] = {
    "hvg": [
        "ABCF1","ACAT2","ACD","ADAT1","AGL","ALAS1","ALDOC","ANO10","APOE","APPBP2","ARHGEF12","ARID4B","ARID5B","ASCL2","ATAD2","ATF3","ATG3","ATP11B","ATP2C1","AURKB","AXIN1","BAG3","BAMBI","BHLHE40","BIRC5","BLM","BMP4","BPHL","BRAF","BRCA1","BZW2","C2CD2L","C2CD5","CAMSAP2","CASK","CASP10","CCNB1","CCND1","CCNE2","CCP110","CD58","CDC25A","CDC25B","CDH3","CDK2","CENPU","CEP57","CFLAR","CIAPIN1","CKAP2","CKAP2L","CKAP5","COASY","COG4","COL4A1","CORO1A","CPNE3","CREB1","CREBBP","CRTAP","CRYZ","CSNK1E","CTCF","CTNND1","CTTN","CXCL2","DLGAP5","DMAC2L","DNM1L","DNMT3A","DNTTIP2","DSCC1","DSG2","DUSP4","DUSP6","DYNLT3","E2F8","EBP","ECT2","EDN1","EED","EFCAB14","EGFR","EGR1","EIF5","ELF1","ELF5","EPB41L2","EPRS","ETV1","EXO1","EXOSC4","EXT1","EZH2","FAIM","FASTKD5","FBXL12","FCHO1","FOSL1","GABPB1","GFPT1","GINS2","GNA15","GTF2A2","HAT1","HDAC2","HELLS","HJURP","HMGCR","HMMR","HMOX1","HNF4A","HOXA10","HOXA5","HOXA9","HS2ST1","HSPA1A","HSPD1","HTRA1","HYOU1","IARS2","ID1","IER3","IFRD2","IGF2BP2","IGF2R","IGFBP3","IL1B","JAK1","JPT2","JUN","KAT6A","KDM3A","KEAP1","KIAA0100","KIF14","KIF20A","KIF23","KLHDC2","KTN1","LAMA3","LBR","LYN","MAP3K4","MAST2","MAT2A","MCM2","MCM3","MCM4","MEF2C","MKI67","MMP1","MTF2","MTFR1","MTHFD2","MUC1","MYBL2","MYC","MYCBP","MYLK","NFKBIA","NGRN","NNT","NOTCH1","NRAS","NSDHL","NT5DC2","NUDCD3","NUP133","NUP93","NUSAP1","ORC1","OXSR1","PACSIN3","PAFAH1B3","PAK1","PARP2","PCCB","PCNA","PGM1","PHKA1","PHKB","PIK3CA","PKIG","PKMYT1","PLA2G4A","PLK1","PMM2","PNP","POLB","POLD3","POLR1C","PPP2R5E","PRCP","PRIM1","PRKCD","PRPF4","PRSS23","PSMF1","PSMG1","PTGS2","PTPRK","PWP1","PYCR1","RAF1","RANGAP1","RNMT","RPP38","RRAGA","RRM2","RRS1","RUVBL1","SACM1L","SCRN1","SCYL3","SENP6","SERPINE1","SESN1","SGK2","SLC35A1","SLC35A3","SLC35F2","SLC37A4","SLC5A6","SMARCA4","SMARCC1","SMC4","SNCA","SOX4","SOX9","SPDEF","SPEN","SPI1","SPP1","SPTAN1","SRC","STAT1","STXBP1","SUPV3L1","TACC3","TARBP1","TBP","TES","TEX10","TFAP2C","TFCP2L1","TGFBR2","TIPARP","TM9SF2","TMCO1","TOMM34","TOR1A","TP53","TPX2","TRAPPC3","TRIB1","TRIB3","TSKU","TSPAN6","TTK","TWF2","TXNL4B","TXNRD1","TYMS","UFM1","USP1","VAPB","VAV3","VPS72","WASF3","WDR76","WFS1","YKT6","ZMYM2","ZNF274","ZW10"
    ],
    "icms2down": [
        "ENOSF1","HIF1A","AKT1","RAP1GAP","FOXA1","LAMA3","NUCB2","TMED10","CANT1","MUC1","SLC35A1","KCNK1","CLSTN1","SCP2","PTPRF","POLD3","LRP10","SPDEF","UGDH","CRELD2","PCM1","NPDC1","TYMS","GSTZ1","KLF9"
    ],
    "icms2up": [
        "TRIB3","SCARB1","NSDHL","TIMM17B","ASCL2","CDK4","TOMM34","MYC","SOX4","YTHDF1","GNPDA1","PRSS23","SLC5A6","CDC25B","EDN1","PGRMC1","RAE1","TRAP1","SCAND1","UTP14A","AURKA","CSNK2A2","RPIA","CCDC85B","NCOA3","LAGE3","BLCAP","HSD17B10","TXNDC9","TSPAN6","RRS1","TPD52L2","UBE2C","VAPB","TPX2","DNAJC15"
    ],
    "icms3down": [
        "CEBPA","PKIG","TFCP2L1","HNF4A","VAV3","MAPK13","ID1","CDX2"
    ],
    "icms3up": [
        "RGS2","BNIP3L","S100A13","DUSP4","TSTA3","ATP1B1"
    ],
}

def get_full_gene_list(cfg: Dict[str, Any]) -> List[str]:
    data_dir = cfg.get("data_dir")
    meta_dir = str(cfg.get("meta_data_dir", "meta_data")).strip("/")
    fname = cfg.get("gene_data_filename")

    files = []
    scp = cfg.get("single_csv_path")
    if scp:
        # Resolve relative to data_dir if needed
        path = None
        if not os.path.isabs(scp) and cfg.get("data_dir"):
            cand = os.path.join(cfg["data_dir"], scp)
            if os.path.isfile(cand):
                path = cand
        if path is None and os.path.isfile(scp):
            path = scp
        if path is None:
            raise FileNotFoundError(f"single_csv_path not found: '{scp}'. Tried: '{scp}' and data_dir-joined '{os.path.join(cfg.get('data_dir',''), scp)}'")
        files = [path]
    elif cfg.get("train_csv_path") and os.path.isfile(cfg["train_csv_path"]):
        files = [cfg["train_csv_path"]]
    else:
        sample_ids = cfg.get("sample_ids")
        if sample_ids:
            for sid in sample_ids:
                path = os.path.join(data_dir, sid, meta_dir, fname)
                if os.path.isfile(path):
                    files.append(path)
        else:
            files = glob.glob(os.path.join(data_dir, "*", meta_dir, fname))
            files.sort()

    if not files:
        raise FileNotFoundError(f"No gene data files found under {data_dir}/*/{meta_dir}/{fname}")

    order = None
    inter = None
    for i, path in enumerate(files):
        df = pd.read_csv(path, nrows=1)
        # Only consider numeric gene columns; never include unnamed/index-like columns
        cols = []
        for c in df.columns:
            s = str(c)
            if s.strip() == "" or s.strip().lower().startswith("unnamed"):
                continue
            if s.endswith("_lds_w") or _is_meta_gene_column(s):
                continue
            if pd.api.types.is_numeric_dtype(df[c]):
                cols.append(c)
        s = set(cols)
        if i == 0:
            order = cols[:]
            inter = s
        else:
            inter &= s
        if not inter:
            break

    genes = [c for c in order if c in inter] if order else []
    if not genes:
        raise ValueError("Could not infer any gene columns. Check your CSV headers and dtypes.")
    return genes

def prepare_gene_list(cfg: Dict[str, Any]) -> List[List[str]]:
    """Prepare gene chunks for downstream usage.

    Behavior:
    - If cfg contains an explicit flat list of genes (cfg["genes"] as list[str]), use it as the base list.
    - Otherwise, infer the full gene list from the dataset via get_full_gene_list(cfg).
    - If split_genes_by is set, split the base list into contiguous chunks of size k.
      Store both cfg["genes"] (chunks) and cfg["gene_chunks"].
    - If split_genes_by is not set, keep the base list and wrap it in a single chunk.
    """
    base_genes = None
    g = cfg.get("genes")
    if isinstance(g, list) and (len(g) == 0 or not isinstance(g[0], list)):
        # Provided a flat list of genes: trust and use as base
        base_genes = [str(x) for x in g if str(x).strip() != ""]
    else:
        # Infer from dataset
        base_genes = get_full_gene_list(cfg)

    if cfg.get("split_genes_by"):
        k = int(cfg["split_genes_by"])
        if k <= 0:
            raise ValueError("split_genes_by must be a positive integer.")
        chunks = [base_genes[i:i+k] for i in range(0, len(base_genes), k)]
        cfg["genes"] = chunks
        cfg["gene_chunks"] = chunks
    else:
        cfg["genes"] = base_genes
        cfg["gene_chunks"] = [base_genes]

    cfg["n_gene_chunks"] = len(cfg["gene_chunks"])
    return cfg["genes"]


def get_active_chunk_idx(cfg: Dict[str, Any], chunk: List[str] = None) -> int:
    chunks = cfg.get("gene_chunks")
    if not chunks:
        return 0
    if chunk is None:
        if cfg.get("genes") and isinstance(cfg["genes"][0], list):
            idx = int(cfg.get("gene_list_index", 1)) - 1
            return max(0, min(idx, len(chunks) - 1))
        # if cfg["genes"] is already a single chunk
        chunk = cfg.get("genes")
    try:
        return next(i for i, ch in enumerate(chunks) if ch == chunk)
    except StopIteration:
        raise RuntimeError(f"chunk not found {chunk} in {chunks}")

def had_split_genes(cfg_dict: Dict[str, Any]) -> bool:
    if "split_genes_by" in cfg_dict and cfg_dict["split_genes_by"] is not None:
        try:
            return int(cfg_dict["split_genes_by"]) > 0
        except Exception:
            return False
    p = cfg_dict.get("parameters", {}).get("split_genes_by")
    if isinstance(p, dict) and "value" in p and p["value"] is not None:
        try:
            return int(p["value"]) > 0
        except Exception:
            return False
    return False
