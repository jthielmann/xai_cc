from typing import List, Dict, Any

from script.gene_list_helpers import get_full_gene_list
from script.configs.dataset_config import DATASETS


_HVG: List[str] = [
    "ABCF1","ACAT2","ACD","ADAT1","AGL","ALAS1","ALDOC","ANO10","APOE","APPBP2","ARHGEF12","ARID4B","ARID5B","ASCL2","ATAD2","ATF3","ATG3","ATP11B","ATP2C1","AURKB","AXIN1","BAG3","BAMBI","BHLHE40","BIRC5","BLM","BMP4","BPHL","BRAF","BRCA1","BZW2","C2CD2L","C2CD5","CAMSAP2","CASK","CASP10","CCNB1","CCND1","CCNE2","CCP110","CD58","CDC25A","CDC25B","CDH3","CDK2","CENPU","CEP57","CFLAR","CIAPIN1","CKAP2","CKAP2L","CKAP5","COASY","COG4","COL4A1","CORO1A","CPNE3","CREB1","CREBBP","CRTAP","CRYZ","CSNK1E","CTCF","CTNND1","CTTN","CXCL2","DLGAP5","DMAC2L","DNM1L","DNMT3A","DNTTIP2","DSCC1","DSG2","DUSP4","DUSP6","DYNLT3","E2F8","EBP","ECT2","EDN1","EED","EFCAB14","EGFR","EGR1","EIF5","ELF1","ELF5","EPB41L2","EPRS","ETV1","EXO1","EXOSC4","EXT1","EZH2","FAIM","FASTKD5","FBXL12","FCHO1","FOSL1","GABPB1","GFPT1","GINS2","GNA15","GTF2A2","HAT1","HDAC2","HELLS","HJURP","HMGCR","HMMR","HMOX1","HNF4A","HOXA10","HOXA5","HOXA9","HS2ST1","HSPA1A","HSPD1","HTRA1","HYOU1","IARS2","ID1","IER3","IFRD2","IGF2BP2","IGF2R","IGFBP3","IL1B","JAK1","JPT2","JUN","KAT6A","KDM3A","KEAP1","KIAA0100","KIF14","KIF20A","KIF23","KLHDC2","KTN1","LAMA3","LBR","LYN","MAP3K4","MAST2","MAT2A","MCM2","MCM3","MCM4","MEF2C","MKI67","MMP1","MTF2","MTFR1","MTHFD2","MUC1","MYBL2","MYC","MYCBP","MYLK","NFKBIA","NGRN","NNT","NOTCH1","NRAS","NSDHL","NT5DC2","NUDCD3","NUP133","NUP93","NUSAP1","ORC1","OXSR1","PACSIN3","PAFAH1B3","PAK1","PARP2","PCCB","PCNA","PGM1","PHKA1","PHKB","PIK3CA","PKIG","PKMYT1","PLA2G4A","PLK1","PMM2","PNP","POLB","POLD3","POLR1C","PPP2R5E","PRCP","PRIM1","PRKCD","PRPF4","PRSS23","PSMF1","PSMG1","PTGS2","PTPRK","PWP1","PYCR1","RAF1","RANGAP1","RNMT","RPP38","RRAGA","RRM2","RRS1","RUVBL1","SACM1L","SCRN1","SCYL3","SENP6","SERPINE1","SESN1","SGK2","SLC35A1","SLC35A3","SLC35F2","SLC37A4","SLC5A6","SMARCA4","SMARCC1","SMC4","SNCA","SOX4","SOX9","SPDEF","SPEN","SPI1","SPP1","SPTAN1","SRC","STAT1","STXBP1","SUPV3L1","TACC3","TARBP1","TBP","TES","TEX10","TFAP2C","TFCP2L1","TGFBR2","TIPARP","TM9SF2","TMCO1","TOMM34","TOR1A","TP53","TPX2","TRAPPC3","TRIB1","TRIB3","TSKU","TSPAN6","TTK","TWF2","TXNL4B","TXNRD1","TYMS","UFM1","USP1","VAPB","VAV3","VPS72","WASF3","WDR76","WFS1","YKT6","ZMYM2","ZNF274","ZW10"
]

_ICMS: Dict[str, List[str]] = {
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

_HARDCODED: Dict[str, List[str]] = {
    "hvg": _HVG,
    **_ICMS,
}


def _dataset_minimal_cfg(name: str) -> Dict[str, Any]:
    if name not in DATASETS:
        raise ValueError(f"unknown dataset {name!r}; available: {sorted(DATASETS.keys())}")
    ds = DATASETS[name]
    for k in ("data_dir", "meta_data_dir", "gene_data_filename"):
        if k not in ds or not ds[k]:
            raise ValueError(f"dataset {name!r} missing required key {k!r}; ds={ds}")
    return {
        "data_dir": ds["data_dir"],
        "meta_data_dir": ds["meta_data_dir"],
        "gene_data_filename": ds["gene_data_filename"],
    }


def get_geneset(geneset: str, dataset: str) -> List[str]:
    if geneset == "cmmn":
        cfg = _dataset_minimal_cfg(dataset)
        return get_full_gene_list(cfg)
    if geneset in _HARDCODED:
        return list(_HARDCODED[geneset])
    raise ValueError(f"unknown geneset {geneset!r}; expected one of: 'cmmn', {sorted(_HARDCODED.keys())}")


def available_genesets() -> List[str]:
    return ["cmmn", *sorted(_HARDCODED.keys())]

