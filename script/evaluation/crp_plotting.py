import torch
import os
import wandb
import torch.nn as nn
import numpy as np
from PIL import Image
import matplotlib.cm as cm
import zennit.image as zimage
from zennit.composites import EpsilonPlusFlat
from zennit.torchvision import VGGCanonizer, ResNetCanonizer
from crp.attribution import CondAttribution


def _get_layer_names(model, types):
    names = []
    for n, m in model.named_modules():
        for t in types:
            if isinstance(m, t):
                names.append(n)
                break
    return names


def _get_composite_and_layer(encoder):
    if type(encoder).__name__ == "VGG":
        composite = EpsilonPlusFlat(canonizers=[VGGCanonizer()])
        layer_type = nn.Linear
        layer_name = _get_layer_names(encoder, [nn.Linear])[-3]
    else:
        composite = EpsilonPlusFlat(canonizers=[ResNetCanonizer()])
        layer_type = type(getattr(encoder, "layer1", [nn.Identity()])[0])
        layer_name = _get_layer_names(encoder, [layer_type])[-1]
    return composite, layer_name

def plot_crp_zennit(model, dataset, run=None, layer_name: str = None, max_items: int = None, out_dir: str = None):
    """CRP using zennit-crp CondAttribution on a small dataset subset."""
    model.eval()
    device = next(model.parameters()).device
    enc = getattr(model, "encoder", model)
    composite, default_layer_name = _get_composite_and_layer(enc)
    target_layer = layer_name or default_layer_name

    attribution = CondAttribution(model)
    count = 0
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    for i in range(len(dataset)):
        if max_items is not None and count >= max_items:
            break
        sample = dataset[i]
        x = sample[0] if isinstance(sample, (tuple, list)) else sample
        x = x.to(device)
        if x.dim() == 3:
            x = x.unsqueeze(0)
        # Make input require grad and be non-leaf so zennit can attach grad_fn hooks
        x.requires_grad_(True)
        x = x + x.new_zeros(())

        attr = attribution(x, [{"y": [0]}], composite, record_layer=[target_layer])
        rel = attr.relevances[target_layer].sum(1).detach().cpu()
        # Sanitize and normalize to avoid NaN/Inf during visualization
        rel = torch.nan_to_num(rel, nan=0.0, posinf=1.0, neginf=-1.0)
        denom = torch.nan_to_num(rel.abs(), nan=0.0).amax(dim=(1, 2), keepdim=True).clamp_min(1e-12)
        rel = rel / denom
        img = zimage.imgify(rel, symmetric=True, cmap='coldnhot', vmin=-1, vmax=1)
        if run is not None:
            run.log({f"crp/attribution[{i}]": wandb.Image(img)})
        if out_dir:
            fn = f"crp_{i:04d}.png"
            img.save(os.path.join(out_dir, fn))
        count += 1


def _find_module_by_name(model: nn.Module, name: str) -> nn.Module:
    for n, m in model.named_modules():
        if n == name:
            return m
    raise KeyError(f"Layer '{name}' not found in model.")


def _default_layer_name(model: nn.Module, types=(nn.Conv2d, nn.Linear)) -> str:
    names = _get_layer_names(model, list(types))
    if len(names) == 0:
        raise ValueError("No layer found matching provided types.")
    return names[-1]


def _tensor_to_pil(arr: torch.Tensor, cmap_name: str = "bwr", symmetric: bool = True) -> Image.Image:
    a = arr.detach().cpu().numpy()
    # Replace NaN/Inf to keep downstream conversion stable
    a = np.nan_to_num(a, nan=0.0, posinf=1.0, neginf=-1.0)
    if symmetric:
        # Compute range robustly in presence of NaNs
        v = float(max(abs(np.nanmin(a)), abs(np.nanmax(a))))
        vmin, vmax = -v, v
    else:
        vmin, vmax = float(np.nanmin(a)), float(np.nanmax(a))
    if not np.isfinite(vmin) or not np.isfinite(vmax):
        vmin, vmax = -1.0, 1.0
    if vmax == vmin:
        vmax = vmin + 1e-12
    norm = np.clip((a - vmin) / (vmax - vmin), 0.0, 1.0)
    rgba = cm.get_cmap(cmap_name)(norm)
    # Guard against NaNs from the colormap by sanitizing before cast
    rgb = np.nan_to_num(rgba[..., :3], nan=0.0)
    rgb = (rgb * 255.0).astype(np.uint8)
    return Image.fromarray(rgb)


def make_capture_forward(store: dict):
    """Factory: forward hook to capture layer activations into 'store'."""
    def capture_forward(module, inp, out):
        store["act"] = out if not isinstance(out, tuple) else out[0]
    return capture_forward


def make_mask_forward_hook(selected_channels: torch.Tensor):
    """Factory: forward hook that registers a gradient mask for selected channels."""
    def hook(module, inp, out):
        o = out if not isinstance(out, tuple) else out[0]

        def grad_mask(grad):
            shape = grad.shape
            base = torch.zeros((shape[0], shape[1]) + (1,) * (grad.dim() - 2), device=grad.device, dtype=grad.dtype)
            for b in range(shape[0]):
                base[b, selected_channels[b]] = 1
            return grad * base

        o.register_hook(grad_mask)
    return hook


def _forward_capture_targets(model: nn.Module, x: torch.Tensor, layer: nn.Module, target):
    """Forward pass to capture activations and select targets."""
    act_store = {}
    h_cap = layer.register_forward_hook(make_capture_forward(act_store))
    logits = model(x)
    h_cap.remove()
    if logits.dim() != 2:
        raise ValueError("Model output must be 2D logits [B, num_classes].")
    batch_size = logits.shape[0]
    if target is None:
        targets_idx = torch.argmax(logits, dim=1)
    else:
        if isinstance(target, int):
            targets_idx = torch.full((batch_size,), target, device=logits.device, dtype=torch.long)
        elif isinstance(target, (list, tuple)):
            targets_idx = torch.as_tensor(target, device=logits.device, dtype=torch.long)
        else:
            targets_idx = target.to(logits.device)
    activation = act_store["act"]
    return logits, targets_idx, activation, batch_size


def _select_concept_channels(activation: torch.Tensor, concept_ids, top_k: int):
    """Select concept channels per sample."""
    batch_size = activation.shape[0]
    if isinstance(concept_ids, (list, tuple)) and len(concept_ids) > 0 and isinstance(concept_ids[0], (list, tuple)):
        selected_channels = torch.zeros((batch_size, len(concept_ids[0])), dtype=torch.long, device=activation.device)
        for b in range(batch_size):
            ids = concept_ids[b]
            selected_channels[b, : len(ids)] = torch.as_tensor(ids, device=activation.device)
    elif concept_ids is None:
        ch_view = activation.view(activation.shape[0], activation.shape[1], -1)
        scores = torch.sum(torch.abs(ch_view), dim=-1)
        k = min(top_k, scores.shape[1])
        selected_channels = torch.topk(scores, k=k, dim=1).indices
    else:
        base = torch.as_tensor(concept_ids, device=activation.device, dtype=torch.long)
        selected_channels = base.unsqueeze(0).repeat(batch_size, 1)
    return selected_channels


def _mask_and_backward(model: nn.Module, x: torch.Tensor, layer: nn.Module, selected_channels: torch.Tensor, targets_idx: torch.Tensor):
    """Register masking hook and backprop selected target scores."""
    h_mask = layer.register_forward_hook(make_mask_forward_hook(selected_channels))
    model.zero_grad(set_to_none=True)
    if x.grad is not None:
        x.grad = None
    logits2 = model(x)
    scores = logits2[torch.arange(logits2.shape[0], device=logits2.device), targets_idx]
    scores.sum().backward()
    h_mask.remove()


def _finalize_and_render(x: torch.Tensor, abs_norm: bool, cmap: str, run):
    """Reduce to heatmaps, normalize, render, and optionally log."""
    heat = x.grad.detach().sum(dim=1)
    if abs_norm:
        B = heat.shape[0]
        hs = torch.nan_to_num(torch.abs(heat), nan=0.0)
        denom = torch.amax(hs.view(B, -1), dim=1).clamp_min(1e-12).view(B, 1, 1)
        heat = torch.nan_to_num(heat, nan=0.0) / denom
    images = [_tensor_to_pil(heat[i], cmap, symmetric=True) for i in range(heat.shape[0])]
    if run is not None:
        run.log({"crp2/attribution": [wandb.Image(img) for img in images]})
    return heat, images


def _prepare_single_input(model: nn.Module, sample):
    """Convert dataset sample to a 4D input tensor [1,C,H,W] and optional label."""
    if isinstance(sample, tuple):
        x_raw, label = sample[0], sample[1]
    else:
        x_raw, label = sample, None
    device = next(model.parameters()).device
    x = x_raw.to(device)
    x = x.unsqueeze(0) if x.dim() == 3 else x
    if x.dim() != 4:
        raise ValueError("Input sample must be 3D (C,H,W) or 4D (B,C,H,W).")
    x = x.detach().requires_grad_(True)
    return x, label


def _resolve_target_arg(target, label, index):
    """Choose target class for the current sample based on provided target or dataset label."""
    if isinstance(target, (list, tuple)):
        return target[index]
    if target is None and label is not None:
        return int(label) if not torch.is_tensor(label) else int(label.item())
    return target


def _resolve_concept_ids_for_sample(concept_ids, index):
    """Return concept ids for this sample (supports list-of-lists or global list)."""
    if isinstance(concept_ids, (list, tuple)) and len(concept_ids) > 0 and isinstance(concept_ids[0], (list, tuple)):
        return concept_ids[index]
    return concept_ids


def plot_crp(
    model: nn.Module,
    dataset,
    run=None,
    layer_name: str = None,
    concept_ids=None,
    target=None,
    top_k: int = 5,
    abs_norm: bool = True,
    cmap: str = "bwr",
    out_dir: str = None,
):
    """
    CRP-like conditional attribution reimplemented with pure PyTorch hooks.
    """

    # Resolve target layer for CRP masking
    model.eval()
    resolved_layer_name = layer_name if layer_name is not None else _default_layer_name(model, (nn.Conv2d, nn.Linear))
    layer = _find_module_by_name(model, resolved_layer_name)

    heatmaps_list, selected_channels_list, targets_list = [], [], []
    all_images = []

    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    for i in range(len(dataset)):
        # Load one datapoint and prepare input tensor, unsqueeze if dim == 3
        x, label = _prepare_single_input(model, dataset[i])

        # Forward to capture activations and select targets
        target_arg = _resolve_target_arg(target, label, i)
        logits, targets_idx, activation, batch_size = _forward_capture_targets(model, x, layer, target_arg)

        # Select concept channels for this sample
        concept_ids_for_sample = _resolve_concept_ids_for_sample(concept_ids, i)
        selected_channels = _select_concept_channels(activation, concept_ids_for_sample, top_k)

        # Mask backward, compute heatmap, normalize, render, and log
        _mask_and_backward(model, x, layer, selected_channels, targets_idx)
        heat, images = _finalize_and_render(x, abs_norm, cmap, run)

        heatmaps_list.append(heat[0])
        selected_channels_list.append(selected_channels[0].detach().cpu())
        targets_list.append(targets_idx.detach().cpu())
        all_images.extend(images)
        if out_dir:
            for j, im in enumerate(images):
                im.save(os.path.join(out_dir, f"crp_custom_{i:04d}_{j:02d}.png"))


    same_shape = all(h.shape == heatmaps_list[0].shape for h in heatmaps_list)
    heatmaps = torch.stack(heatmaps_list, dim=0) if same_shape else heatmaps_list
    return {"heatmaps": heatmaps, "layer": resolved_layer_name, "selected_channels": selected_channels_list, "targets": targets_list}
