import torch
import torch.nn as nn
from typing import Dict, Optional, Tuple, Union

class MultiGeneWeightedMSE(nn.Module):
    """Weighted MSE for multi-gene regression with robust shape/broadcast handling."""

    _VALID_REDUCTIONS = {"mean", "sum", "none"}
    _VALID_NORMALIZATIONS = {"global", "per_sample", "per_gene"}

    def __init__(
        self,
        eps: float = 1e-8,
        reduction: str = "mean",
        normalize: str = "global",
        clip_weights: Optional[float] = None,
        check_finite: bool = True,
    ):
        super().__init__()
        if reduction not in self._VALID_REDUCTIONS:
            raise ValueError(f"Unsupported reduction '{reduction}'. Expected one of {self._VALID_REDUCTIONS}.")
        if normalize not in self._VALID_NORMALIZATIONS:
            raise ValueError(
                f"Unsupported normalize '{normalize}'. Expected one of {self._VALID_NORMALIZATIONS}."
            )
        self.eps = float(eps)
        self.reduction = reduction
        self.normalize = normalize
        self.clip_weights = clip_weights
        self.check_finite = check_finite

    def _ensure_shapes(self, pred: torch.Tensor, target: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if pred.shape != target.shape:
            raise ValueError(f"Shape mismatch pred{tuple(pred.shape)} vs target{tuple(target.shape)}")
        return pred, target

    @staticmethod
    def _broadcast_weights(weights: torch.Tensor, ref_shape: torch.Size) -> torch.Tensor:
        if weights.dim() == 0:
            return weights.expand(ref_shape)

        if weights.shape == ref_shape:
            return weights

        if weights.dim() == 1:
            if weights.shape[0] == ref_shape[0]:
                return weights.view(ref_shape[0], 1).expand(ref_shape)
            if weights.shape[0] == ref_shape[-1]:
                return weights.view(1, ref_shape[-1]).expand(ref_shape)

        if weights.dim() == 2:
            if weights.shape[0] == ref_shape[0] and weights.shape[1] == 1:
                return weights.expand(ref_shape)
            if weights.shape[0] == 1 and weights.shape[1] == ref_shape[-1]:
                return weights.expand(ref_shape)

        try:
            return weights.expand(ref_shape)
        except RuntimeError as exc:  # pragma: no cover - defensive branch
            raise ValueError(
                f"sample_weights shape {tuple(weights.shape)} cannot broadcast to {tuple(ref_shape)}"
            ) from exc

    def _normalize_weights(self, weights: torch.Tensor) -> torch.Tensor:
        if self.clip_weights is not None:
            weights = weights.clamp(max=self.clip_weights)
        if self.check_finite:
            if not torch.isfinite(weights).all():
                raise ValueError("sample_weights contains non-finite values (inf or NaN).")
            if (weights < 0).any():
                raise ValueError("sample_weights must be non-negative.")
        return weights

    def _apply_reduction(self, value: torch.Tensor) -> torch.Tensor:
        if self.reduction == "none":
            return value
        if self.reduction == "mean":
            return value.mean()
        if self.reduction == "sum":
            return value.sum()
        raise RuntimeError(f"Unhandled reduction '{self.reduction}'.")  # pragma: no cover

    def _finalize_loss(
        self,
        per_elem: torch.Tensor,
        w32: torch.Tensor,
        *,
        return_stats: bool,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, Dict[str, torch.Tensor]]]:
        stats: Dict[str, torch.Tensor] = {
            "weight_mean": w32.mean().detach(),
            "weight_max": w32.max().detach(),
            "weight_min": w32.min().detach(),
            "weight_nonzero_frac": (w32 > 0).float().mean().detach(),
        }

        if self.normalize == "global":
            numerator = per_elem.sum()
            denominator = torch.clamp(w32.sum(), min=self.eps)
            value = numerator / denominator
            stats["numerator"] = numerator.detach()
            stats["denominator"] = denominator.detach()
        elif self.normalize == "per_sample":
            numerator = per_elem.sum(dim=1)
            denominator = torch.clamp(w32.sum(dim=1), min=self.eps)
            value = numerator / denominator
            stats["numerator_vector"] = numerator.detach()
            stats["denominator_vector"] = denominator.detach()
        else:  # per_gene
            numerator = per_elem.sum(dim=0)
            denominator = torch.clamp(w32.sum(dim=0), min=self.eps)
            value = numerator / denominator
            stats["numerator_vector"] = numerator.detach()
            stats["denominator_vector"] = denominator.detach()

        loss = self._apply_reduction(value)
        if return_stats:
            stats["loss_before_cast"] = value.detach()
            return loss, stats
        return loss

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        sample_weights: torch.Tensor,
        *,
        return_stats: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, Dict[str, torch.Tensor]]]:
        if sample_weights is None:
            raise ValueError("sample_weights must be provided for MultiGeneWeightedMSE.")

        pred, target = self._ensure_shapes(pred, target)
        ref_shape = pred.shape
        w = torch.as_tensor(sample_weights, device=pred.device, dtype=pred.dtype)
        w = self._broadcast_weights(w, ref_shape)
        w = self._normalize_weights(w)

        pred32 = pred.to(torch.float32)
        target32 = target.to(torch.float32)
        w32 = w.to(torch.float32)
        per_elem = (pred32 - target32).pow(2) * w32
        result = self._finalize_loss(per_elem, w32, return_stats=return_stats)
        if isinstance(result, tuple):
            loss, stats = result
            return loss.to(pred.dtype), stats
        return result.to(pred.dtype)


class WeightedHuberLoss(MultiGeneWeightedMSE):
    """Weighted Huber (smooth L1) loss sharing normalization semantics with WMSE."""

    def __init__(
        self,
        delta: float = 1.0,
        eps: float = 1e-8,
        reduction: str = "mean",
        normalize: str = "global",
        clip_weights: Optional[float] = None,
        check_finite: bool = True,
    ):
        super().__init__(
            eps=eps,
            reduction=reduction,
            normalize=normalize,
            clip_weights=clip_weights,
            check_finite=check_finite,
        )
        self.delta = float(delta)

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        sample_weights: torch.Tensor,
        *,
        return_stats: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, Dict[str, torch.Tensor]]]:
        if sample_weights is None:
            raise ValueError("sample_weights must be provided for WeightedHuberLoss.")

        pred, target = self._ensure_shapes(pred, target)
        ref_shape = pred.shape
        w = torch.as_tensor(sample_weights, device=pred.device, dtype=pred.dtype)
        w = self._broadcast_weights(w, ref_shape)
        w = self._normalize_weights(w)

        pred32 = pred.to(torch.float32)
        target32 = target.to(torch.float32)
        w32 = w.to(torch.float32)
        diff = pred32 - target32
        abs_diff = diff.abs()

        delta_tensor = torch.tensor(self.delta, device=diff.device, dtype=diff.dtype)
        quadratic = torch.minimum(abs_diff, delta_tensor)
        linear = abs_diff - quadratic
        per_elem = (0.5 * quadratic.pow(2) + delta_tensor * linear) * w32
        result = self._finalize_loss(per_elem, w32, return_stats=return_stats)
        if isinstance(result, tuple):
            loss, stats = result
            return loss.to(pred.dtype), stats
        return result.to(pred.dtype)


class PearsonCorrLoss(nn.Module):
    def __init__(self, dim: int = 0, eps: float = 1e-8, reduction: str = "mean"):
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.reduction = reduction

    def forward(self, preds: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        x = preds
        y = target
        x = x - x.mean(dim=self.dim, keepdim=True)
        y = y - y.mean(dim=self.dim, keepdim=True)

        xy = (x * y).sum(dim=self.dim)
        xx = (x * x).sum(dim=self.dim)
        yy = (y * y).sum(dim=self.dim)

        r = xy / (torch.sqrt(xx + self.eps) * torch.sqrt(yy + self.eps))  # shape: (G,) or scalar
        loss = 1 - r
        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        return loss
