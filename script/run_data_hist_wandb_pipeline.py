import sys
sys.path.insert(0, '..')
from evaluation.generate_data_hists import plot_data_hists
from main_utils import parse_yaml_config, parse_args, read_config_parameter
from script.configs.dataset_config import get_dataset_cfg

args = parse_args()
cfg = parse_yaml_config(args.config)

flat_params = {
    k: (v["value"] if isinstance(v, dict) and "value" in v else v)
    for k, v in cfg.get("parameters", {}).items()
}

ds_cfg = get_dataset_cfg(flat_params)
# Build a flattened config for W&B and plotting (avoid nested parameters/metric)
flat_cfg = {k: v for k, v in cfg.items() if k not in ("parameters", "metric", "method")}
flat_cfg.update(flat_params)
flat_cfg.update(ds_cfg)

out_paths = plot_data_hists(
    config=flat_cfg,
    save_dir="plots/hists/crc_base",
    overlay_per_gene=True
)
print(out_paths)
