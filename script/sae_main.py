import os
import sys

sys.path.insert(0, '..')

os.environ.setdefault("NUMBA_THREADING_LAYER", "workqueue")

from typing import Any, Dict

import wandb

from script.configs.dataset_config import get_dataset_cfg
from script.main_utils import (
    ensure_free_disk_space,
    parse_args,
    parse_yaml_config,
    setup_dump_env,
)
from script.evaluation.eval_helpers import (
    load_state_dict_from_path,
    normalize_state_dicts,
)
from script.model.lit_model import load_lit_regressor
from script.train.lit_train_sae import SAETrainerPipeline


def _sanitize_token(s: str) -> str:
    return (
        str(s).replace("\\", "/").rstrip("/").replace("/", "__").replace(" ", "_")
    )[:128]


def _rel_model_path(p: str) -> str:
    s = str(p).replace("\\", "/")
    if "/models/" in s:
        s = s.split("/models/", 1)[1]
    elif s.startswith("../models/"):
        s = s[len("../models/") :]
    return s.strip("/")


def _require_model_dir(cfg: Dict[str, Any]) -> str:
    p = cfg.get("model_state_path")
    if not p or not isinstance(p, str):
        raise RuntimeError("sae_main requires 'model_state_path' directory")
    base = p if os.path.isabs(p) else os.path.join("../models", p)
    if not os.path.isdir(base):
        raise RuntimeError(f"model_state_path not a directory: {base}")
    if not os.path.isfile(os.path.join(base, "config")):
        raise RuntimeError(f"missing model config: {os.path.join(base, 'config')}")
    if not os.path.isfile(os.path.join(base, "best_model.pth")):
        raise RuntimeError(f"missing best_model.pth in: {base}")
    return os.path.abspath(base)


def _prepare_eval_paths(cfg: Dict[str, Any], model_dir: str) -> Dict[str, Any]:
    merged = dict(cfg)
    merged.update(get_dataset_cfg(merged))

    model_cfg_path = os.path.join(model_dir, "config")
    model_cfg = parse_yaml_config(model_cfg_path)
    if not isinstance(model_cfg, dict):
        raise RuntimeError(f"invalid model config: {model_cfg_path}")
    enc = model_cfg.get("encoder_type")
    if not isinstance(enc, str) or not enc.strip():
        raise RuntimeError("encoder_type missing in model config")
    enc_token = _sanitize_token(enc)
    base = os.path.join("../evaluation", enc_token)
    eval_path = os.path.join(base, "debug") if bool(merged.get("debug", False)) else base
    os.makedirs(eval_path, exist_ok=True)
    ensure_free_disk_space(eval_path)
    merged["eval_path"] = eval_path
    merged["out_path"] = eval_path
    merged["model_config"] = model_cfg
    merged["encoder_type"] = enc

    rel_model = _rel_model_path(model_dir)
    sae_dir = os.path.join(eval_path, rel_model, "sae")
    os.makedirs(sae_dir, exist_ok=True)
    merged["model_dir"] = sae_dir
    merged["out_path"] = sae_dir
    # use default run name to keep everything simple
    if not merged.get("run_name"):
        merged["run_name"] = f"sae__{_sanitize_token(rel_model)}"[:128]
    if merged.get("log_to_wandb") and not merged.get("name"):
        merged["name"] = merged["run_name"]
    return merged


def main() -> None:
    args = parse_args()
    raw_cfg = parse_yaml_config(args.config)

    model_dir = _require_model_dir(raw_cfg)
    cfg = _prepare_eval_paths(raw_cfg, model_dir)

    setup_dump_env()

    run = None
    if bool(cfg.get("log_to_wandb")):
        for key in ("run_name", "group", "job_type", "tags"):
            if key not in cfg:
                raise ValueError(f"Missing required parameter '{key}' in config")
        wb_cfg = {
            k: v
            for k, v in cfg.items()
            if k
            not in ("project", "metric", "method", "run_name", "group", "job_type", "tags")
        }
        run = wandb.init(
            project=cfg.get("project", "xai"),
            name=cfg["run_name"],
            group=cfg["group"],
            job_type=cfg["job_type"],
            tags=cfg["tags"],
            config=wb_cfg,
        )
        cfg = dict(cfg)
        cfg.update(dict(run.config))
        cfg["run_name"] = raw_cfg.get("run_name", cfg["run_name"]) 

    best_path = os.path.join(model_dir, "best_model.pth")
    bundled = load_state_dict_from_path(best_path)
    state_dicts = normalize_state_dicts(bundled)
    model = load_lit_regressor(cfg["model_config"], state_dicts)

    cfg_sae = dict(cfg)
    cfg_sae["train_sae"] = True
    pipeline = SAETrainerPipeline(cfg_sae, run=run, encoder=model.encoder)
    pipeline.setup()
    pipeline.run()

    if run:
        run.finish()


if __name__ == "__main__":
    main()
