import os
import torch
import timm
from huggingface_hub import login
from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform
from data_processing.data_loader import get_dataset
from data_processing.lit_STDataModule import get_data_module
from main_utils import parse_args, parse_yaml_config
from configs.dataset_config import get_dataset_cfg
# ---------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------
if __name__ == "__main__":
    args = parse_args()
    config = parse_yaml_config(args.config)

    params = config.get("parameters", {})
    # Detect if any parameter defines multiple 'values'

    # Single-run: flatten each parameter 'value' into cfg dict
    cfg = {k: v for k, v in config.items() if k != "parameters"}
    for key, param in params.items():
        if isinstance(param, dict) and "value" in param:
            cfg[key] = param["value"]


    ds_cfg = get_dataset_cfg(cfg)
    cfg.update(ds_cfg)

    MODEL_ID   = "hf-hub:MahmoodLab/UNI2-h"
    MODEL_FILE = "../encoders/UNI2-h_state.pt"
    CACHE_DIR  = "../../models/h"

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


    def build_model(encoder: bool, **kwargs):
        return timm.create_model(
            MODEL_ID,
            encoder=encoder,
            cache_dir=CACHE_DIR,
            **kwargs,
        )

    if os.path.exists(MODEL_FILE):
        print("loading model from local file")
        model = build_model(encoder=False, **timm_kwargs)  # no net access
        state_dict = torch.load(MODEL_FILE, map_location="cpu", weights_only=True)
        model.load_state_dict(state_dict)
    else:
        print("downloading model")
        token = os.environ["HUGGINGFACE_HUB_TOKEN"]   # must be set in the shell
        login(token=token)

        model = build_model(
            encoder=True,
            hf_hub_token=token,   # passed down to hf_hub_download
            **timm_kwargs,
        )
        torch.save(model.state_dict(), MODEL_FILE)    # cache for next run
        print(f"Weights saved to {MODEL_FILE}")

    print(os.path.exists(MODEL_FILE))
    print(MODEL_FILE)

    IMG_SIZE   = 224          # as in timm_kwargs
    PATCH_SIZE = 14           # as in timm_kwargs
    REG_TOKENS = 8            # as in timm_kwargs
    EMBED_DIM  = 1536         # as in timm_kwargs
    BATCH_SIZE = 1            # use any batch size you like

    num_patches = (IMG_SIZE // PATCH_SIZE) ** 2

    dummy_img = torch.randn(1, 3, 224, 224)
    tokens = model(dummy_img)
    print(tokens.shape)

    data_module = get_data_module(cfg)
    data_module.setup("fit")
    val_loader = data_module.val_dataloader()

    for batch_idx, batch in enumerate(val_loader):
        # If your dataloader returns (images, labels) –––– adjust if it returns a dict
        images, labels = batch

        with torch.no_grad():
            tokens = model(images)  # (B, reg_tokens + num_patches, embed_dim)

        print(f"{batch_idx:02d} | imgs: {images.shape} → tokens: {tokens.shape}")

        if batch_idx == 9:  # stop after 10 batches
            break

