import os
from typing import Optional

import wandb


def _resolve_out_path(out_path_arg: Optional[str], config) -> str:
    cfg_path = None
    if isinstance(config, dict):
        cfg_path = config.get("out_path")
    if (
        out_path_arg
        and cfg_path
        and os.path.abspath(str(cfg_path)) != os.path.abspath(str(out_path_arg))
    ):
        raise ValueError(f"Conflicting out_path arg and config: {out_path_arg} vs {cfg_path}")
    if out_path_arg:
        return str(out_path_arg)
    if cfg_path:
        return str(cfg_path)
    raise ValueError("out_path required")


def plot_pcx(model, config, run=None, out_path: Optional[str] = None):
    from script.evaluation.cluster_functions import cluster
    resolved_out_path = _resolve_out_path(out_path, config)
    results = cluster(
        cfg=config,
        model=model,
        data_dir=config["data_dir"],
        samples=config.get("test_samples"),
        genes=config["genes"],
        out_path=resolved_out_path,
        debug=bool(config.get("debug")),
    )
    if run is not None:
        images = [wandb.Image(p) for p in results]
        run.log({"pcx/cluster_images": images})
