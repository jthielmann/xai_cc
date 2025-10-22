import wandb


def plot_pcx(model, config, run=None):
    from script.evaluation.cluster_functions import cluster
    results = cluster(
        cfg=config,
        model=model,
        data_dir=config["data_dir"],
        samples=config.get("test_samples"),
        genes=config["genes"],
        out_path=config["out_path"],
        debug=bool(config.get("debug")),
    )
    if run is not None:
        images = [wandb.Image(p) for p in results]
        run.log({"pcx/cluster_images": images})
