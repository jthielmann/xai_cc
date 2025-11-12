import os
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import yaml
from script.main_utils import compute_genes_id


def _list_results(eval_root: str) -> List[str]:
    if not eval_root or not os.path.isdir(eval_root):
        raise FileNotFoundError(f"eval_root not found or not dir: {eval_root}")
    hits: List[str] = []
    for dirpath, dirnames, filenames in os.walk(eval_root):
        # debug is usually just 100 samples and often failed runs, so it does not make sense to include it here
        if "/debug/" in (dirpath + "/"):
            continue
        if os.path.basename(dirpath) != "predictions":
            continue
        if "results.csv" in filenames:
            hits.append(os.path.join(dirpath, "results.csv"))
    if not hits:
        raise RuntimeError("no predictions/results.csv found under eval_root")
    return sorted(hits)


def _read_eval_config(eval_root: str) -> Dict:
    cfg_path = os.path.join(eval_root, "config")
    if not os.path.exists(cfg_path):
        raise FileNotFoundError(f"missing eval config at {cfg_path}")
    with open(cfg_path, "r") as f:
        cfg = yaml.safe_load(f)
    if not isinstance(cfg, dict):
        raise RuntimeError("eval config invalid; expected mapping")
    return cfg


def _read_model_config(models_root: str, rel_run_dir: str) -> Dict:
    if not models_root or not os.path.isdir(models_root):
        raise FileNotFoundError(f"models_root not found or not dir: {models_root}")
    cfg_path = os.path.join(models_root, rel_run_dir, "config")
    if not os.path.exists(cfg_path):
        raise FileNotFoundError(f"missing model config: {cfg_path}")
    with open(cfg_path, "r") as f:
        cfg = yaml.safe_load(f)
    if not isinstance(cfg, dict):
        raise RuntimeError("model config invalid; expected mapping")
    return cfg


def _infer_training_metric(cfg: Dict) -> str:
    mc = cfg.get("model_config")
    if not isinstance(mc, dict):
        raise RuntimeError("eval config missing nested model_config")
    loss = str(mc.get("loss_fn_switch", "")).strip().lower()
    if not loss:
        raise RuntimeError("model_config.loss_fn_switch missing in eval config")
    if loss not in {"mse", "pearson", "wmse", "weighted mse"}:
        raise RuntimeError(f"unsupported loss_fn_switch: {loss}")
    # bit hardcoded, mapping wmse to mse because we would just need normal mse as metric without reweighting which only impacts the weights anyways
    if loss in {"wmse", "weighted mse"}:
        return "mse"
    return loss


def _pearson(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    if y_true.size < 2:
        return float("nan")
    v = np.corrcoef(y_true, y_pred)[0, 1]
    return float(v)


def _mse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    if y_true.size == 0:
        return float("nan")
    return float(np.mean((y_true - y_pred) ** 2))


def _collect_genes(df: pd.DataFrame) -> List[str]:
    genes: List[str] = []
    for col in df.columns:
        if col.startswith("label_"):
            g = col[len("label_"):]
            pcol = f"pred_{g}"
            if pcol in df.columns:
                gs = str(g).strip()
                if not gs:
                    continue
                if gs.lower().startswith("unnamed"):
                    continue
                genes.append(gs)
    if not genes:
        raise RuntimeError("no label_/pred_ gene columns detected in results.csv")
    return genes


def _compute_metrics_for_csv(csv_path: str) -> Tuple[Dict[str, float], Dict[str, int]]:
    df = pd.read_csv(csv_path)
    genes = _collect_genes(df)
    per_gene_pearson: Dict[str, float] = {}
    per_gene_mse: Dict[str, float] = {}
    counts: Dict[str, int] = {}
    for g in genes:
        a = pd.to_numeric(df[f"label_{g}"], errors="coerce").to_numpy()
        b = pd.to_numeric(df[f"pred_{g}"], errors="coerce").to_numpy()
        m = np.isfinite(a) & np.isfinite(b)
        a = a[m]
        b = b[m]
        counts[g] = int(a.shape[0])
        per_gene_pearson[g] = _pearson(a, b)
        per_gene_mse[g] = _mse(a, b)
    merged = {f"pearson_{g}": per_gene_pearson[g] for g in genes}
    merged.update({f"mse_{g}": per_gene_mse[g] for g in genes})
    return merged, counts


def _weighted_mean(values: Dict[str, float], counts: Dict[str, int]) -> float:
    num = 0.0
    den = 0
    for g, v in values.items():
        if not np.isfinite(v):
            continue
        c = counts.get(g, 0)
        num += float(v) * int(c)
        den += int(c)
    if den == 0:
        return float("nan")
    return float(num / den)


def gather_forward_metrics(eval_root: str, output_csv: str = None) -> str:
    if not eval_root:
        raise ValueError("eval_root is required")
    if not output_csv:
        output_csv = os.path.join(eval_root, "forward_metrics.csv")
    cfg = _read_eval_config(eval_root)
    enc_global = cfg.get("encoder_type") or (cfg.get("model_config") or {}).get("encoder_type")
    if not isinstance(enc_global, str) or not enc_global.strip():
        raise RuntimeError("encoder_type missing in eval config")
    models_root = os.path.normpath(os.path.join("..", "models"))

    rows: List[Dict[str, object]] = []
    existing_df = None
    existing_run_dirs: set[str] = set()
    existing_run_names: set[str] = set()
    existing_proj_run_keys: set[str] = set()
    if os.path.exists(output_csv):
        existing_df = pd.read_csv(output_csv)
        if "run_dir" in existing_df.columns:
            existing_run_dirs = set(map(str, existing_df["run_dir"].astype(str).tolist()))
        if "run_name" in existing_df.columns:
            existing_run_names = set(map(str, existing_df["run_name"].astype(str).tolist()))
        if "project" in existing_df.columns and "run_name" in existing_df.columns:
            existing_proj_run_keys = set(
                f"{str(p)}/{str(r)}" for p, r in zip(existing_df["project"].astype(str), existing_df["run_name"].astype(str))
            )
    for csv_path in _list_results(eval_root):
        run_dir = os.path.dirname(csv_path)
        model_dir = os.path.dirname(run_dir)
        run_name = os.path.basename(model_dir)
        rel_run_dir = os.path.relpath(model_dir, eval_root)

        metrics, counts = _compute_metrics_for_csv(csv_path)
        pearson_vals = {k[len("pearson_"):]: v for k, v in metrics.items() if k.startswith("pearson_")}
        mse_vals = {k[len("mse_"):]: v for k, v in metrics.items() if k.startswith("mse_")}
        mc = _read_model_config(models_root, rel_run_dir)
        enc = mc.get("encoder_type") or enc_global
        if not isinstance(enc, str) or not enc.strip():
            raise RuntimeError("encoder_type missing in model config")
        train_metric = _infer_training_metric({"model_config": mc})
        loss_name = str(mc.get("loss_fn_switch", "")).strip().lower() or "unknown"
        train_vals = mse_vals if train_metric == "mse" else pearson_vals
        project = mc.get("project")
        if not isinstance(project, str) or not project.strip():
            parts = os.path.normpath(rel_run_dir).split(os.sep)
            project = parts[0] if parts else "project"
        gene_set = str(mc.get("gene_set", "custom"))
        genes_id = mc.get("genes_id")
        if not genes_id:
            genes = mc.get("genes")
            if genes is None:
                raise RuntimeError("model config missing 'genes' to compute genes_id")
            genes_id = compute_genes_id(genes)
        run_key = f"{project}/{run_name}"
        if rel_run_dir in existing_run_dirs or run_name in existing_run_names or run_key in existing_proj_run_keys:
            continue

        row: Dict[str, object] = {
            "encoder_type": enc,
            "project": project,
            "run_dir": rel_run_dir,
            "run_name": run_name,
            "metric_name": train_metric,
            "loss_name": loss_name,
            "gene_set": gene_set,
            "genes_id": genes_id,
            "pearson_mean": _weighted_mean(pearson_vals, counts),
            "mse_mean": _weighted_mean(mse_vals, counts),
            f"{train_metric}_mean": _weighted_mean(train_vals, counts),
        }
        row.update(metrics)
        rows.append(row)

    if existing_df is not None:
        if rows:
            df_new = pd.DataFrame(rows)
            df = pd.concat([existing_df, df_new], ignore_index=True)
            os.makedirs(os.path.dirname(output_csv) or ".", exist_ok=True)
            df.to_csv(output_csv, index=False)
        return output_csv
    df = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(output_csv) or ".", exist_ok=True)
    df.to_csv(output_csv, index=False)
    return output_csv


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser(description="Aggregate forward predictions metrics into CSV")
    p.add_argument("--eval-root", required=True, help="Path to evaluation/<encoder_type>")
    p.add_argument("--out", required=False, help="Output CSV path (defaults to <eval_root>/forward_metrics.csv)")
    args = p.parse_args()
    path = gather_forward_metrics(args.eval_root, args.out)
    print(path)
