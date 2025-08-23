import torch

from script.configs.config_factory import get_dataset_cfg
from torchvision.models import resnet50
print(torch.__version__)
from script.evaluation.cluster_functions import cluster_explanations_genes_loop
import wandb
from script.model.lit_model import load_model
from script.main_utils import parse_yaml_config, read_config_parameter
import random
import os
from script.main_utils import ensure_free_disk_space
import torch.nn as nn
from script.data_processing.image_transforms import get_transforms
from script.data_processing.data_loader import get_dataset


model = resnet50()
print(model)


def find_output_layer_name(model: nn.Module, example_input):
    """
    Returns the name of the last *used* module during forward.
    Prefers modules that have parameters (e.g., the final Linear).
    """
    model_was_training = model.training
    model.eval()

    last_any = None
    last_with_params = None
    handles = []

    def make_hook(name, module):
        def _hook(module, inp, out):
            nonlocal last_any, last_with_params
            last_any = name
            try:
                has_params = any(p.requires_grad for p in module.parameters(recurse=False))
            except Exception:
                has_params = False
            if has_params:
                last_with_params = name
        return _hook

    for name, m in model.named_modules():
        handles.append(m.register_forward_hook(make_hook(name, m)))

    with torch.no_grad():
        _ = model(example_input)

    for h in handles:
        h.remove()
    if model_was_training:
        model.train()

    # Prefer the last module with parameters; fallback to last module of any kind
    return last_with_params or last_any

def cluster(cfg, model, data_dir, samples, genes, out_path, debug=False, target_layer_name=None):
    dataset = get_dataset(data_dir, genes=genes, samples=samples, transforms=get_transforms(cfg),
                                max_len=1, only_inputs=False)
    if target_layer_name is None:
        # create or fetch a single example batch (X, y) -> take X
        # ensure it's on the same device as the model
        device = next(model.parameters()).device
        example_x = dataset[0][0].unsqueeze(0).to(device)
        target_layer_name = find_output_layer_name(model, example_x)

    print("clustering start; target layer:", target_layer_name)
    results_paths = cluster_explanations_genes_loop(
        cfg, model, data_dir, out_path,
        target_layer_name=target_layer_name,
        genes=genes, debug=debug, samples=samples
    )
    for i in range(len(results_paths)):
        wandb.log({f"clustering results for {genes[i]}": wandb.Image(results_paths[i])})
    print("clustering done")


base_path = "sweeps/debug_local/"
model_string = "best_model.pt"
config_string = "config"
cfg = parse_yaml_config(base_path + config_string)

flat_params = {
    k: (v["value"] if isinstance(v, dict) and "value" in v else v["values"])
    for k, v in cfg.get("parameters", {}).items()
}

project = cfg["project"] if not read_config_parameter(cfg, "debug") else "debug_" + random.randbytes(4).hex()
out_path = "../crp_out/" + project
if not os.path.exists(out_path):
    os.makedirs(out_path, exist_ok=True)
ensure_free_disk_space(out_path)
flat_params["out_path"] = out_path

debug = read_config_parameter(flat_params, "debug")
ds_cfg = get_dataset_cfg(flat_params["dataset"], debug=debug)
flat_params.update(ds_cfg)
data_dir = read_config_parameter(flat_params, "data_dir")
out_path = read_config_parameter(flat_params, "out_path")
train_samples = read_config_parameter(flat_params, "train_samples")
genes = read_config_parameter(flat_params, "genes")
cluster(flat_params, model, data_dir, train_samples, genes, out_path, debug)

exit(0)













#model = load_model(path=base_path+model_string, config=cfg)
model = resnet50()
print(model)