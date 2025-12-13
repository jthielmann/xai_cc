import os
from typing import Optional

import wandb


def plot_pcx(model, config, out_path: str, run=None):
    from script.evaluation.cluster_functions import cluster
    patients = config.get("patients")
    if not patients:
        patients = config.get("test_samples")
    print (f"-----------------------------------------\n{patients}\n-------------------------------------------------")
    results = cluster(
        cfg=config,
        model=model,
        data_dir=config["data_dir"],
        samples=patients,
        genes=config["genes"],
        out_path=out_path,
        debug=bool(config.get("debug")),
    )
    if run is not None:
        images = [wandb.Image(p) for p in results]
        run.log({"pcx/cluster_images": images})
