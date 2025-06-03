import wandb, argparse, os, torch, json
from script.model.lit_model import load_model
from script.evaluation.cluster_functions import cluster

parser = argparse.ArgumentParser()
parser.add_argument("--wandb-id", required=True)
args = parser.parse_args()

# must resume as this is supposed to continue a wandb run after the model training to log pcx
run = wandb.init(project="dino",
                 id=args.wandb_id,
                 resume="must")

api = wandb.Api()
artifact = api.artifact(f"{run.entity}/{run.project}/model-best:latest", type="model")
model_dir = artifact.download()                # e.g. ./artifacts/model-best:v0
ckpt_path = os.path.join(model_dir, "model.ckpt")

with open(os.path.join(model_dir, "config.json")) as f:
    cfg = json.load(f)

device = "cuda" if torch.cuda.is_available() else "cpu"
model  = load_model(ckpt_path, cfg).to(device)

result_paths = cluster(model=model,
                  data_dir  = cfg["data_dir"],
                  samples   = cfg["val_samples"],
                  genes     = cfg["genes"],
                  out_dir   = "../crp_out/",
                  debug     = False)

from pathlib import Path
import wandb

MAX_PER_STEP = 50

for step, i in enumerate(range(0, len(result_paths), MAX_PER_STEP)):
    batch_paths  = result_paths[i : i + MAX_PER_STEP]
    batch_genes  = model.genes[i : i + MAX_PER_STEP]

    run.log(
        {"cluster_images": [
             wandb.Image(p, caption=gene)
             for p, gene in zip(batch_paths, batch_genes)
        ]},
        step=step
    )

run.finish()