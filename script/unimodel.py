import os
import torch
import timm
from huggingface_hub import login
from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform

# ---------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------
MODEL_ID   = "hf-hub:MahmoodLab/UNI2-h"   # timm knows how to fetch this
MODEL_FILE = "../../models/UNI2-h_state.pt"            # local cache of the weights
CACHE_DIR  = "../../models/h"             # put HF files here (optional)

# timm kwargs for the UNI2-h backbone
timm_kwargs = {
    "img_size": 224,
    "patch_size": 14,
    "depth": 24,
    "num_heads": 24,
    "init_values": 1e-5,
    "embed_dim": 1536,
    "mlp_ratio": 2.66667 * 2,
    "num_classes": 0,
    "no_embed_class": True,
    "mlp_layer": timm.layers.SwiGLUPacked,
    "act_layer": torch.nn.SiLU,
    "reg_tokens": 8,
    "dynamic_img_size": True,
}


def build_model(pretrained: bool, **kwargs):
    return timm.create_model(
        MODEL_ID,
        pretrained=pretrained,
        cache_dir=CACHE_DIR,
        **kwargs,
    )

if os.path.exists(MODEL_FILE):
    print("loading model from local file")
    model = build_model(pretrained=False, **timm_kwargs)  # no net access
    state_dict = torch.load(MODEL_FILE, map_location="cpu")
    model.load_state_dict(state_dict)
else:
    print("downloading model")
    token = os.environ["HUGGINGFACE_HUB_TOKEN"]   # must be set in the shell
    login(token=token)

    model = build_model(
        pretrained=True,
        hf_hub_token=token,   # passed down to hf_hub_download
        **timm_kwargs,
    )
    torch.save(model.state_dict(), MODEL_FILE)    # cache for next run
    print(f"Weights saved to {MODEL_FILE}")

print(os.path.exists(MODEL_FILE))
print(MODEL_FILE)