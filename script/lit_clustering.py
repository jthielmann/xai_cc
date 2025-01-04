from lit_config import config
from cluster_functions import cluster_explanations
from model import load_model
import torch

device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
model = load_model(config["model_dir"], config["model_dir"] + "/best_model.pth", squelch=True).to(device)

# clustering
debug = True
for gene in config["genes"]:
    cluster_explanations(model, config["data_dir"], config["output_dir"], genes=config["genes"], debug=debug)