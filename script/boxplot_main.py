import os
import sys

sys.path.insert(0, "..")


from script.main_utils import parse_args, parse_yaml_config, setup_dump_env
from script.boxplot_helpers import (
    _require_keys,
    _load_forward_metrics_recursive,
    _validate_gene_sets,
    _maybe_init_wandb_and_update_cfg,
    _plot_all_sets,
    _apply_filters,
)


EVAL_ROOT = "../evaluation"
OUT_DIR = os.path.join(EVAL_ROOT, "boxplots")


def main() -> None:
    args = parse_args()
    raw_cfg = parse_yaml_config(args.config)

    if not isinstance(raw_cfg, dict):
        raise RuntimeError("config must be a mapping")

    setup_dump_env()

    _require_keys(
        raw_cfg,
        ["gene_sets", "log_to_wandb", "plot_box", "plot_violin", "scan_root"],
    )

    scan_root = raw_cfg.get("scan_root") or None
    plot_box = raw_cfg.get("plot_box")
    plot_violin = raw_cfg.get("plot_violin")

    if not isinstance(plot_box, bool) or not isinstance(plot_violin, bool):
        raise ValueError("plot_box and plot_violin must be bools")
    if not (plot_box or plot_violin):
        raise ValueError("at least one of plot_box/plot_violin must be true")

    if not isinstance(scan_root, str) or not scan_root.strip():
        raise ValueError(
            f"scan_root must be a non-empty string; got {scan_root!r}"
        )
    scan_path = (
        scan_root if os.path.isabs(scan_root) else os.path.join(EVAL_ROOT, scan_root)
    )
    df = _load_forward_metrics_recursive(scan_path)

    include_projects = raw_cfg.get("include_projects")
    include_encoders = raw_cfg.get("include_encoders")
    include_run_name_regex = raw_cfg.get("include_run_name_regex")
    exclude_run_name_regex = raw_cfg.get("exclude_run_name_regex")
    df = _apply_filters(
        df,
        include_projects=include_projects,
        include_encoders=include_encoders,
        include_run_name_regex=include_run_name_regex,
        exclude_run_name_regex=exclude_run_name_regex,
    )

    gene_sets = raw_cfg.get("gene_sets")
    if isinstance(gene_sets, str):
        token = gene_sets.strip().lower()
        if token in {"all", "__all__"}:
            genes = []
            for c in df.columns:
                if not c.startswith("pearson_"):
                    continue
                g = c[len("pearson_") :].strip()
                if not g:
                    continue
                if g.lower().startswith("unnamed"):
                    continue
                genes.append(g)
            if not genes:
                raise RuntimeError("no pearson_* columns found to infer genes")
            gene_sets = {"all": genes}
        else:
            raise ValueError("gene_sets string must be 'all' or '__ALL__'")
    else:
        # sanitize any provided gene sets: drop empty/unnamed genes
        if isinstance(gene_sets, dict):
            cleaned = {}
            for name, gl in gene_sets.items():
                cleaned_list = []
                for g in gl:
                    gs = str(g).strip()
                    if not gs:
                        continue
                    if gs.lower().startswith("unnamed"):
                        continue
                    cleaned_list.append(gs)
                cleaned[name] = cleaned_list
            gene_sets = cleaned
    _validate_gene_sets(gene_sets)

    run, raw_cfg = _maybe_init_wandb_and_update_cfg(raw_cfg)

    group_by = str(raw_cfg.get("group_by", "encoder_type")).strip().lower()
    if group_by not in {"encoder_type", "encoder_type+loss", "project", "project+encoder_type"}:
        raise ValueError(f"unsupported group_by: {group_by}")
    group_col = "__group__"
    if group_by == "encoder_type":
        df[group_col] = df["encoder_type"].astype(str)
    elif group_by == "encoder_type+loss":
        if "loss_name" not in df.columns:
            raise RuntimeError("loss_name column missing in metrics; re-run aggregation")
        df[group_col] = df[["encoder_type", "loss_name"]].astype(str).agg(lambda r: f"{r[0]} ({r[1]})", axis=1)
    elif group_by == "project":
        df[group_col] = df["project"].astype(str)
    elif group_by == "project+encoder_type":
        df[group_col] = df[["project", "encoder_type"]].astype(str).agg(lambda r: f"{r[0]}::{r[1]}", axis=1)

    saved_paths = _plot_all_sets(
        df=df,
        gene_sets=gene_sets,
        plot_box=plot_box,
        plot_violin=plot_violin,
        skip_non_finite=bool(raw_cfg.get("skip_non_finite", False)),
        run=run,
        out_dir=OUT_DIR,
        group_key=group_col,
    )

    if run is not None:
        run.finish()

    # Print saved files for quick inspection
    for p in saved_paths:
        print(p)


if __name__ == "__main__":
    main()
