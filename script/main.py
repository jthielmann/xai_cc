import os


# Make numba avoid OpenMP/TBB to prevent clashes with PyTorch/MKL on HPC
os.environ.setdefault("NUMBA_THREADING_LAYER", "workqueue")
# Be conservative with thread pools by default (can be overridden by user env)
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("MPLBACKEND", "Agg")
# Required for deterministic CuBLAS ops on CUDA >= 10.2 when Lightning sets deterministic=True
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
import numpy as np
import torch
from umap import UMAP
import csv, os, random, numpy, torch, yaml, pandas as pd, wandb

import sys
sys.path.insert(0, '..')

from script.gene_list_helpers import prepare_gene_list, get_full_gene_list, had_split_genes
from script.train.lit_train_sae import SAETrainerPipeline
from typing import Dict, Any, List
from script.configs.dataset_config import get_dataset_cfg
from script.train.lit_train import TrainerPipeline
from main_utils import (
    ensure_free_disk_space,
    parse_args,
    parse_yaml_config,
    read_config_parameter,
    get_sweep_parameter_names,
    make_run_name_from_config,
    setup_dump_env,
    prepare_cfg,
    compute_genes_id,
)


import os, glob
import pandas as pd
from typing import Dict, Any, List

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

def _apply_gene_set_inplace(cfg: Dict[str, Any]) -> None:
    gs = cfg.get("gene_set")
    if gs is None:
        return
    if cfg.get("genes") is not None:
        raise ValueError("cannot set both 'genes' and 'gene_set'")
    key = str(gs).strip().lower()
    if key == "common":
        return
    if key not in _GENE_SETS:
        raise ValueError(f"unknown gene_set {gs!r}")
    cfg["genes"] = list(_GENE_SETS[key])

def _train(cfg: Dict[str, Any]) -> None:
    # Flatten only at handoff to training
    flat_cfg = _flatten_params(cfg)
    _apply_gene_set_inplace(flat_cfg)

    # Initialize W&B with flattened config
    log_to_wandb = bool(read_config_parameter(cfg, "log_to_wandb"))
    if log_to_wandb:
        run = wandb.init(
            project=read_config_parameter(cfg, "project"),
            name=read_config_parameter(cfg, "run_name"),
            group=read_config_parameter(cfg, "group"),
            job_type=read_config_parameter(cfg, "job_type"),
            tags=read_config_parameter(cfg, "tags"),
            config=flat_cfg,
        )
    else:
        run = None

    # Use flattened config for downstream pipeline
    cfg = dict(run.config) if run else flat_cfg

    # Ensure dump_dir and prepare cfg
    cfg.setdefault("dump_dir", setup_dump_env())

    cfg = prepare_cfg(cfg)

    if bool(cfg.get("train_sae", False)):
        # No gene list inference needed — training uses encoder features only
        print("SAETrainerPipeline debug")
        SAETrainerPipeline(cfg, run=run).run()
    else:
        # Use provided genes if available; otherwise infer from dataset
        if cfg.get("genes") is None:
            cfg["genes"] = get_full_gene_list(cfg)
        # If split_genes_by is set, split the provided list into chunks and pick an index
        split_k = cfg.get("split_genes_by")
        genes = cfg.get("genes")
        if split_k:
            k = int(split_k)
            if k <= 0:
                raise ValueError("split_genes_by must be a positive integer.")
            # Do not accept nested lists silently; require a flat list of genes
            if isinstance(genes, list) and genes and isinstance(genes[0], list):
                raise ValueError(
                    "Config 'genes' must be a flat list of gene names when using split_genes_by; "
                    "got a list of lists. Provide a flat list and use 'gene_list_index' to select a chunk."
                )
            tgt = [str(g) for g in genes]
            chunks = [tgt[i:i+k] for i in range(0, len(tgt), k)]
            # Select chunk via 1-based gene_list_index if provided, else first chunk
            idx = int(cfg.get("gene_list_index", 1)) - 1
            idx = max(0, min(idx, len(chunks) - 1)) if chunks else 0
            cfg["genes"] = chunks[idx] if chunks else []
            cfg["genes_id"] = f"c{idx+1:03d}"
        else:
            # No chunking requested; derive a stable id from the provided list
            cfg["genes_id"] = compute_genes_id(cfg.get("genes"))
        TrainerPipeline(cfg, run=run).run()
    if run: run.finish()

 # Locate the config by name or path
def _resolve_config_path(name: str) -> str:
    if os.path.isfile(name):
        return name
    # search common config roots
    roots = ["../sweeps/configs", "../sweeps/to_train"]
    for root in roots:
        cand = os.path.join(root, name)
        if os.path.isfile(cand):
            return cand
    cwd = os.getcwd()
    raise FileNotFoundError(f"Could not resolve config_name '{name}' from cwd={cwd!r}; searched: {roots}")

def _flatten_base_fixed(cfg: Dict[str, Any]) -> Dict[str, Any]:
    """
    Extracts fixed values from a base config that may use {"value": ...}
    wrappers and a 'parameters' section. Only fixed values are retained.
    """
    out: Dict[str, Any] = {}

    # Top-level fields: allow either raw values or {"value": ...}
    for k, v in cfg.items():
        if k in ("parameters", "metric", "method", "name", "sweep_parameter_names"):
            continue
        if isinstance(v, dict) and "value" in v:
            out[k] = v["value"]
        else:
            out[k] = v

    # Fixed values inside the 'parameters' section (ignore sweep search specs)
    params = cfg.get("parameters")
    if isinstance(params, dict):
        for pk, pv in params.items():
            if isinstance(pv, dict) and "value" in pv and not ("values" in pv or "distribution" in pv):
                out[pk] = pv["value"]
    return out

def _sweep_run():
    # Ensure dump env in agent subprocess before init
    setup_dump_env()
    run = wandb.init()
    run_config = dict(run.config)
    config_name = run_config.get("config_name")
    if not config_name:
        raise RuntimeError("Sweep run missing 'config_name' in parameters")

    base_config_path = _resolve_config_path(run_config["config_name"])
    base_config = parse_yaml_config(base_config_path)
    parameter_names = get_sweep_parameter_names(base_config)
    auto_name = make_run_name_from_config(run_config, parameter_names)
    run.name = auto_name

    base_fixed = _flatten_base_fixed(base_config)
    merged = dict(base_fixed)
    merged.update({k: v for k, v in run_config.items() if k != "config_name"})
    merged["name"] = auto_name
    merged["sweep_parameter_names"] = list(parameter_names)

    project = merged.get("project") or read_config_parameter(base_config, "project")
    base_model_dir = "../models/"
    project_dir = os.path.join(base_model_dir, project)
    os.makedirs(project_dir, exist_ok=True)
    ensure_free_disk_space(project_dir)

    merged["model_dir"] = project_dir
    merged["sweep_dir"] = project_dir
    # Tag runs/dirs with an identifier derived from the selected genes
    if merged.get("genes") is not None:
        try:
            merged["genes_id"] = compute_genes_id(merged.get("genes"))
        except Exception:
            merged["genes_id"] = "genes"
    merged = prepare_cfg(merged)
    run.config.update(merged, allow_val_change=True)

    # Ensure dump_dir present and env set before training
    if bool(merged.get("train_sae", False)):
        SAETrainerPipeline(merged, run=run).run()
    else:
        TrainerPipeline(merged, run=run).run()
    run.finish()

def _extract_hyperparams(pdict: Dict[str, Any]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for k, v in pdict.items():
        if isinstance(v, dict) and ("values" in v or "distribution" in v):
            out[k] = v
    return out

def _flatten_params(raw: Dict[str, Any]) -> Dict[str, Any]:
    cfg: Dict[str, Any] = {}
    for k, v in raw.items():
        if type(v) == str:
            cfg[k] = v
        elif k in ("parameters", "metric"):
            continue
        else:
            cfg[k] = v["value"]
    params = raw.get("parameters")
    for pk, pv in params.items():
        cfg[pk] = pv["value"]
    return cfg

def _build_sweep_config(raw_cfg: Dict[str, Any], config_basename: str = None) -> Dict[str, Any]:
    """Build a W&B sweep spec from a raw config with a 'parameters' section."""
    params_dict = (raw_cfg.get("parameters", {})) if isinstance(raw_cfg, dict) else {}
    hyper_params = _extract_hyperparams(params_dict)

    # Add config name so each run can load the base config for fixed parameters
    hyper_params["config_name"] = {"value": config_basename}

    # Minimal fields for dataset resolution and gene discovery
    cfg_for_genes: Dict[str, Any] = {}
    for key in ("dataset", "debug", "gene_data_filename", "meta_data_dir", "sample_ids", "split_genes_by", "genes", "gene_set"):
        val = read_config_parameter(raw_cfg, key)
        cfg_for_genes[key] = val
    if read_config_parameter(raw_cfg, "gene_set") is not None and read_config_parameter(raw_cfg, "genes") is not None:
        raise ValueError("cannot set both 'genes' and 'gene_set'")
    cfg_for_genes = prepare_cfg(cfg_for_genes)
    _apply_gene_set_inplace(cfg_for_genes)
    gene_chunks = prepare_gene_list(dict(cfg_for_genes))

    sweep_config = {
        "name": raw_cfg["name"],
        "method": read_config_parameter(raw_cfg, "method"),
        "metric": read_config_parameter(raw_cfg, "metric"),
        "parameters": hyper_params,
    }

    if "genes" not in hyper_params:
        if gene_chunks:
            sweep_config["parameters"]["genes"] = {"values": gene_chunks}
        else:
            fixed_genes = read_config_parameter(raw_cfg, "genes")
            sweep_config["parameters"]["genes"] = {"value": fixed_genes}


    sweep_param_names = get_sweep_parameter_names(raw_cfg)
    sweep_config["sweep_parameter_names"] = sweep_param_names


    return sweep_config

def _has_sweep_params(config: Dict[str, Any]) -> bool:
    params = config.get("parameters", {}) if isinstance(config, dict) else {}
    if not isinstance(params, dict):
        return False
    return any(isinstance(v, dict) and ("values" in v or "distribution" in v) for v in params.values())

def main():
    args = parse_args()

    raw_cfg = parse_yaml_config(args.config)

    xai_pipeline = (read_config_parameter(raw_cfg, "xai_pipeline"))
    if xai_pipeline:
        raise RuntimeError("[info] Detected xai_pipeline config (manual/auto). Instead run:\n"
                           f"  python script/eval_main.py --config {args.config}")

    setup_dump_env()

    debug = read_config_parameter(raw_cfg, "debug")
    project = read_config_parameter(raw_cfg, "project") if not debug else "_debug_" + random.randbytes(4).hex()

    if debug:
        print("python version:", sys.version)
        print("numpy version:", numpy.version.version)
        print("torch version:", torch.__version__)
        print("cuda available:", torch.cuda.is_available())

    is_sweep = _has_sweep_params(raw_cfg)
    if is_sweep:
        name = raw_cfg["name"]
        sweep_id_dir = os.path.join("..", "wandb_sweep_ids", project, name)
        sweep_id_file = os.path.join(sweep_id_dir, "sweep_id.txt")
        os.environ["WANDB_RUN_GROUP"] = str(read_config_parameter(raw_cfg, "group"))
        os.environ["WANDB_JOB_TYPE"] = str(read_config_parameter(raw_cfg, "job_type"))
        tags = read_config_parameter(raw_cfg, "tags")
        os.environ["WANDB_TAGS"] = ",".join(map(str, tags))
        if os.path.exists(sweep_id_file):
            with open(sweep_id_file, "r") as f: sweep_id = f.read().strip()
        else:
            os.makedirs(sweep_id_dir, exist_ok=True)
            sweep_config = _build_sweep_config(raw_cfg, os.path.basename(args.config))
            sweep_id = wandb.sweep(sweep_config, project=project)
            with open(sweep_id_file, "w") as f:
                f.write(sweep_id)
        wandb.agent(sweep_id, function=_sweep_run, project=project)
    else:
        _train(raw_cfg)

if __name__ == "__main__":
    main()
