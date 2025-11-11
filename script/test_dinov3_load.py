import gc
import logging
import sys
from pathlib import Path

import torch

# add parent so `import script.*` works when running from script/
sys.path.insert(0, "..")

from script.model.dinov3_local import (
    available_dinov3_models,
    load_dinov3_local,
)
from script.model.model_factory import resolve_unique_model_file


log = logging.getLogger(__name__)


def test_load_all_dinov3_cpu(encoders_dir: str = "../encoders") -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required for this test")
    if not isinstance(encoders_dir, str) or not encoders_dir:
        raise ValueError(f"invalid encoders_dir: {encoders_dir!r}")
    if not log.handlers:
        logging.basicConfig(level=logging.DEBUG, format="%(levelname)s:%(message)s")
    root = Path(encoders_dir).expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"encoders_dir not found: {root}")

    names = available_dinov3_models()
    log.debug("testing %d models from %s", len(names), root)
    for name in names:
        ckpt = resolve_unique_model_file(name, encoders_dir=str(root))
        log.debug("loading %s from %s", name, ckpt)
        model = load_dinov3_local(name, str(ckpt))
        n_params = sum(p.numel() for p in model.parameters())
        log.debug("loaded %s params=%d", name, n_params)

        model = model.to("cuda")
        model.eval()
        x = torch.zeros(1, 3, 224, 224, device="cuda")
        with torch.no_grad():
            if hasattr(model, "forward_features"):
                out = model.forward_features(x)
            else:
                out = model(x)
        if isinstance(out, (list, tuple)) and len(out):
            out = out[0]
        if hasattr(out, "last_hidden_state"):
            out = out.last_hidden_state
        if not isinstance(out, torch.Tensor):
            raise RuntimeError(f"unexpected forward output type for {name}: {type(out)}")
        log.debug("forward %s ok; out shape=%s", name, tuple(out.shape))

        del model, out, x
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        log.debug("unloaded %s", name)


if __name__ == "__main__":
    test_load_all_dinov3_cpu("../encoders")
