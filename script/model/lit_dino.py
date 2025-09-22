import os
import copy

import torch
import torch.nn as nn
import torchvision
import lightning as L
from torch.optim.lr_scheduler import OneCycleLR

import wandb
from lightly.loss import DINOLoss
from lightly.models.modules import DINOProjectionHead
from lightly.models.utils import deactivate_requires_grad, update_momentum
from lightly.utils.scheduler import cosine_schedule
try:
    from transformers import AutoImageProcessor, Dinov3Model
except Exception:
    AutoImageProcessor = None  # type: ignore
    Dinov3Model = None  # type: ignore


class _BackboneOut:
    def __init__(self, last_hidden_state):
        self.last_hidden_state = last_hidden_state


class DummyBackbone(nn.Module):
    def __init__(self, hidden_size: int = 768):
        super().__init__()
        self.config = type('cfg', (), {'hidden_size': hidden_size})
        self.stem = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.proj = nn.Linear(32, hidden_size)

    def forward(self, pixel_values: torch.Tensor):
        # pixel_values: (B,3,H,W)
        x = self.stem(pixel_values)          # (B,32,1,1)
        x = x.flatten(1)                     # (B,32)
        cls = self.proj(x)                   # (B,D)
        # Construct tiny token sequence: [CLS] + 1 dummy patch token
        B, D = cls.shape
        patch = torch.zeros(B, 1, D, device=cls.device, dtype=cls.dtype)
        tokens = torch.cat([cls.unsqueeze(1), patch], dim=1)
        return _BackboneOut(tokens)

    # keep API parity
    def gradient_checkpointing_enable(self):
        return None

class DINO(L.LightningModule):
    def __init__(self, dino_config, lr=0.001):
        super().__init__()
        self.config = dino_config
        self.lr = lr

        # --- NEW: DINOv3 backbone setup ---
        # config knobs you can pass in:
        # self.config.get("model_id", "facebook/dinov3-vitb16-pretrain-lvd1689m")
        # self.config.get("use_hf_normalize", True)

        model_id = self.config.get(
            "model_id", "facebook/dinov3-vitb16-pretrain-lvd1689m"
        )
        # Default to False to avoid double-normalization when transforms already normalize
        self.use_hf_normalize = bool(self.config.get("use_hf_normalize", False))
        # Backbone training knobs
        self.grad_checkpointing = bool(self.config.get("grad_checkpointing", True))
        # Unfreeze last N transformer blocks (0 = none; -1 = all)
        self.unfreeze_last_n = int(self.config.get("unfreeze_last_n", -1))
        # Backbone LR multiplier (default 1.0 to avoid behavior change)
        self.backbone_lr_mult = float(self.config.get("backbone_lr_mult", 1.0))
        # Optional explicit backbone LR overrides multiplier
        self.backbone_lr = self.config.get("backbone_lr", None)

        # HF processor (resize/normalize) – keep toggle so you can use your own transforms instead
        if self.config.get("use_hf_normalize", False):
            if AutoImageProcessor is None:
                raise ImportError("transformers.AutoImageProcessor is required when use_hf_normalize=True")
            self.processor = AutoImageProcessor.from_pretrained(model_id)
        else:
            self.processor = None

        # Student / teacher backbones (optionally dummy for smoke tests)
        if bool(self.config.get("use_dummy_backbone", False)):
            hidden_size = int(self.config.get("dummy_hidden_size", 256))
            self.student_backbone = DummyBackbone(hidden_size)
        else:
            if Dinov3Model is None:
                raise ImportError("transformers.Dinov3Model is required unless use_dummy_backbone=True")
            self.student_backbone = Dinov3Model.from_pretrained(model_id)
        # Enable gradient checkpointing to save VRAM
        if self.grad_checkpointing and hasattr(self.student_backbone, "gradient_checkpointing_enable"):
            try:
                self.student_backbone.gradient_checkpointing_enable()
            except Exception:
                pass
        self.teacher_backbone = copy.deepcopy(self.student_backbone)
        deactivate_requires_grad(self.teacher_backbone)
        self.teacher_backbone.eval()

        # Optionally restrict trainable layers of the student backbone
        self._configure_backbone_trainability(self.unfreeze_last_n)

        # Hidden size drives head input dim (e.g., 768 for ViT-B/16)
        hidden = int(getattr(self.student_backbone.config, 'hidden_size', 768))

        # Head dimensions from config to ensure consistency with loss
        head_hidden_dim = int(self.config.get('head_hidden_dim', 512))
        head_bottleneck_dim = int(self.config.get('head_bottleneck_dim', 64))
        head_output_dim = int(self.config.get('head_output_dim', 2048))
        freeze_last_layer = int(self.config.get('freeze_last_layer_epochs', 1))

        # Projection heads
        self.student_head  = DINOProjectionHead(hidden, head_hidden_dim, head_bottleneck_dim, head_output_dim, freeze_last_layer=freeze_last_layer)
        self.teacher_head  = DINOProjectionHead(hidden, head_hidden_dim, head_bottleneck_dim, head_output_dim)
        deactivate_requires_grad(self.teacher_head)
        self.teacher_head.eval()

        # DINOLoss parameters (temps/centering) driven by config
        per_step_ttemp = bool(self.config.get('teacher_temp_per_step', False))
        warmup_teacher_temp_epochs = int(self.config.get('warmup_teacher_temp_epochs', 30))
        if per_step_ttemp:
            warmup_teacher_temp_epochs = 0  # we'll schedule temperature per step
        self.criterion = DINOLoss(
            output_dim=head_output_dim,
            warmup_teacher_temp=float(self.config.get('teacher_temp_warmup', 0.04)),
            teacher_temp=float(self.config.get('teacher_temp_final', 0.07)),
            warmup_teacher_temp_epochs=warmup_teacher_temp_epochs,
            student_temp=float(self.config.get('student_temp', 0.1)),
            center_momentum=float(self.config.get('center_momentum', 0.9)),
        )
        self._teacher_temp_per_step = per_step_ttemp
        self.best_loss = torch.tensor(float('inf'))
        self.current_loss = torch.tensor(0.)
        self.num_training_batches = 0

    def _backbone_tokens(self, pixel_values: torch.Tensor, model: nn.Module):
        """Return (cls, patches) from backbone with output_hidden_states disabled.

        - cls: (B, D)
        - patches: (B, N, D)
        """
        if hasattr(getattr(model, 'config', None), 'output_hidden_states'):
            try:
                model.config.output_hidden_states = False  # type: ignore[attr-defined]
            except Exception:
                pass
        try:
            out = model(pixel_values=pixel_values, output_hidden_states=False)
        except TypeError:
            out = model(pixel_values=pixel_values)
        tokens = out.last_hidden_state
        cls = tokens[:, 0]
        patches = tokens[:, 1:]
        return cls, patches

    def _find_transformer_layers(self, model: nn.Module):
        # Try common HF ViT/DINOv3 layer containers
        candidates = [
            "vit.encoder.layer",
            "dinov3.encoder.layer",
            "encoder.layer",
            "blocks",
            "layers",
        ]
        for path in candidates:
            obj = model
            ok = True
            for attr in path.split('.'):
                if hasattr(obj, attr):
                    obj = getattr(obj, attr)
                else:
                    ok = False
                    break
            if ok and isinstance(obj, (nn.ModuleList, list, tuple)) and len(obj) > 0:
                return obj
        # Fallback: scan named_modules for a ModuleList that looks like transformer blocks
        for name, mod in model.named_modules():
            if isinstance(mod, nn.ModuleList) and len(mod) > 0 and any(k in name.lower() for k in ["layer", "block", "encoder"]):
                return mod
        return None

    def _configure_backbone_trainability(self, unfreeze_last_n: int):
        # Default: train all; allow selectively unfreezing last N blocks
        if unfreeze_last_n is None:
            return
        if unfreeze_last_n == -1:
            # train all backbone params
            for p in self.student_backbone.parameters():
                p.requires_grad = True
            return
        # freeze all first
        for p in self.student_backbone.parameters():
            p.requires_grad = False
        # unfreeze last N blocks if we can find them
        blocks = self._find_transformer_layers(self.student_backbone)
        if isinstance(blocks, (nn.ModuleList, list, tuple)) and len(blocks) > 0 and unfreeze_last_n > 0:
            n = min(unfreeze_last_n, len(blocks))
            for blk in blocks[-n:]:
                for p in blk.parameters():
                    p.requires_grad = True
        # Also allow final layernorm/head norms to train for stability
        for name, module in self.student_backbone.named_modules():
            if any(k in name.lower() for k in ["layernorm", "ln", "final_norm"]):
                for p in module.parameters():
                    p.requires_grad = True

    def on_fit_start(self):
        os.makedirs(self.config.get('out_path', '.'), exist_ok=True)
        # Ensure teacher stays in eval mode
        self.teacher_backbone.eval()
        self.teacher_head.eval()

        # ---- total_steps for per-step cosine schedules ----
        # Prefer Lightning's estimate; fallback to epochs * num_training_batches
        if hasattr(self.trainer, "estimated_stepping_batches") and self.trainer.estimated_stepping_batches:
            self.total_steps = int(self.trainer.estimated_stepping_batches)
        else:
            self.total_steps = int(self.config['epochs'] * max(1, self.num_training_batches))
        # Setup per-step teacher temperature schedule if enabled
        if getattr(self, '_teacher_temp_per_step', False):
            t_start = float(self.config.get('teacher_temp_warmup', 0.04))
            t_end = float(self.config.get('teacher_temp_final', 0.07))
            warmup_steps = int(self.config.get('warmup_teacher_temp_steps', 0))
            if warmup_steps <= 0:
                warmup_epochs = int(self.config.get('warmup_teacher_temp_epochs', 0))
                warmup_steps = int(warmup_epochs * max(1, self.num_training_batches))
            self._ttemp_schedule = {'start': t_start, 'end': t_end, 'warmup_steps': max(1, warmup_steps)}
        else:
            self._ttemp_schedule = None

    def update_lr(self, lr):
        self.lr = lr

    def forward(self, pixel_values):
        cls, _ = self._backbone_tokens(pixel_values, self.student_backbone)
        z = self.student_head(cls)
        return z

    def forward_teacher(self, pixel_values):
        with torch.no_grad():
            cls, _ = self._backbone_tokens(pixel_values, self.teacher_backbone)
        return self.teacher_head(cls)

    def common_step(self, batch, batch_idx):
        # Safety: teacher always eval (no BN/dropout updates)
        self.teacher_backbone.eval()
        self.teacher_head.eval()
        # Step-based EMA momentum
        step_now = min(self.global_step, max(0, self.total_steps - 1))
        m_start = float(self.config.get('ema_momentum_start', 0.996))
        m_end   = float(self.config.get('ema_momentum_end', 0.9995))
        momentum = cosine_schedule(
            step=step_now,
            max_steps=self.total_steps,
            start_value=m_start,
            end_value=m_end,
        )
        update_momentum(self.student_backbone, self.teacher_backbone, m=momentum)
        update_momentum(self.student_head, self.teacher_head, m=momentum)
        # Teacher temperature per step (optional)
        if getattr(self, '_ttemp_schedule', None):
            s = self._ttemp_schedule
            ws = float(s['warmup_steps'])
            if step_now < ws:
                ttemp = float(s['start'] + (s['end'] - s['start']) * (step_now / ws))
            else:
                ttemp = float(s['end'])
            try:
                self.criterion.teacher_temp = float(ttemp)
            except Exception:
                pass
        # Diagnostics: log EMA momentum, LR, teacher temp, and weight decay
        self.log('ema_m', momentum, on_step=True, prog_bar=False)
        try:
            if self.trainer and self.trainer.optimizers:
                lrs = [pg.get('lr', 0.0) for pg in self.trainer.optimizers[0].param_groups]
                if lrs:
                    self.log('lr', float(max(lrs)), on_step=True, prog_bar=False)
                wd = None
                for pg in self.trainer.optimizers[0].param_groups:
                    if pg.get('weight_decay', 0.0) > 0.0:
                        wd = pg['weight_decay']
                        break
                if wd is not None:
                    self.log('weight_decay', float(wd), on_step=True, prog_bar=False)
            ttemp = float(getattr(self.criterion, 'teacher_temp', 0.0))
            self.log('teacher_temp', ttemp, on_step=True, prog_bar=False)
        except Exception:
            pass

        # Views handling: support (views, ...) or plain views
        views_in = batch[0] if (isinstance(batch, (list, tuple)) and batch and isinstance(batch[0], (list, tuple))) else batch
        views_in = views_in if isinstance(views_in, (list, tuple)) else [views_in]

        # Helper: apply HF normalization only when needed to avoid double-normalization
        def _maybe_hf_normalize(x: torch.Tensor) -> torch.Tensor:
            # x: (B,C,H,W) float tensor
            if not self.use_hf_normalize:
                return x
            # Detect range to infer whether already normalized
            with torch.no_grad():
                x_min = x.amin().item()
                x_max = x.amax().item()
            mean = torch.tensor(self.processor.image_mean, device=x.device, dtype=x.dtype).view(1, -1, 1, 1)
            std  = torch.tensor(self.processor.image_std,  device=x.device, dtype=x.dtype).view(1, -1, 1, 1)
            # If in [0,1] → normalize; if in [0,255] → scale then normalize; otherwise assume already normalized
            if -1e-3 <= x_min <= 1.0 + 1e-3 and -1e-3 <= x_max <= 1.0 + 1e-3:
                return (x - mean) / std
            if 1.0 + 1e-3 < x_max <= 255.0 + 1e-3:
                x01 = x / 255.0
                return (x01 - mean) / std
            return x

        # Prepare list of views as (B,C,H,W) tensors on device
        views = []
        for v in views_in:
            if torch.is_tensor(v):
                pv = v if v.dim() == 4 else v.unsqueeze(0)
                pv = pv.to(self.device, non_blocking=True)
                pv = _maybe_hf_normalize(pv)
            else:
                # PIL image(s) path: rely on HF processor to convert + normalize
                if self.use_hf_normalize:
                    pv = self.processor(images=v, return_tensors="pt")["pixel_values"].to(self.device)
                else:
                    # Convert to tensor without normalization
                    pv = torchvision.transforms.functional.pil_to_tensor(v).float().div(255.0).unsqueeze(0).to(self.device)
            views.append(pv)

        # teacher: 2× global crops (first two); student: all crops
        teacher_out, student_out = [], []
        # Keep patch tokens for potential Gram anchoring (not used here yet)
        teacher_patches, student_patches = [], []
        for i, v in enumerate(views):
            # Student on all views
            s_cls, s_p = self._backbone_tokens(v, self.student_backbone)
            student_out.append(self.student_head(s_cls))
            student_patches.append(s_p)
            # Feature norm diagnostics (first student view only to reduce overhead)
            if i == 0:
                try:
                    self.log('cls_norm', s_cls.norm(dim=-1).mean(), on_step=True, prog_bar=False)
                    if s_p.numel() > 0:
                        self.log('patch_norm', s_p.norm(dim=-1).mean(), on_step=True, prog_bar=False)
                except Exception:
                    pass
            # Teacher only on first 2 (global) views
            if i < 2:
                with torch.no_grad():
                    t_cls, t_p = self._backbone_tokens(v, self.teacher_backbone)
                    teacher_out.append(self.teacher_head(t_cls))
                    teacher_patches.append(t_p)
        # Optional distributed gather before loss (ensures cross-GPU centering)
        if bool(self.config.get('gather_distributed', False)) and getattr(getattr(self, 'trainer', None), 'world_size', 1) > 1:
            try:
                teacher_out = [self.all_gather(t).reshape(-1, t.size(-1)) for t in teacher_out]
                student_out = [self.all_gather(s).reshape(-1, s.size(-1)) for s in student_out]
            except Exception:
                pass

        # DINO objective on CLS tokens
        loss = self.criterion(teacher_out, student_out, epoch=self.current_epoch)

        # Diagnostics: teacher entropy, center norm, student variance
        try:
            if teacher_out:
                temp = float(getattr(self.criterion, 'teacher_temp', 0.07))
                p = torch.softmax(teacher_out[0] / max(temp, 1e-6), dim=-1)
                entropy = (-p * (p.clamp_min(1e-9).log())).sum(dim=-1).mean()
                self.log('teacher_entropy', entropy, on_step=True, prog_bar=False)
            center = getattr(self.criterion, 'center', None)
            if center is not None:
                self.log('center_norm', center.norm().detach(), on_step=True, prog_bar=False)
            if student_out:
                self.log('student_var', student_out[0].var(dim=0).mean(), on_step=True, prog_bar=False)
        except Exception:
            pass

        # Optional Gram anchoring on patch tokens to stabilize features
        ga_weight = float(self.config.get('gram_anchor_weight', 0.0))
        if ga_weight > 0.0 and len(teacher_patches) == 2 and len(student_patches) >= 2:
            ga_loss = self._gram_anchor_loss(student_patches[:2], teacher_patches)
            self.log('gram_loss', ga_loss, prog_bar=False, on_step=True, on_epoch=False)
            loss = loss + ga_weight * ga_loss
        return loss

    def _gram_anchor_loss(self, student_patch_list, teacher_patch_list):
        """Compute Gram anchoring loss between student and teacher patch tokens.

        - Normalizes tokens along feature dim (cosine sim of patch tokens).
        - Optionally subsamples tokens to limit O(T^2) cost.
        - Uses MSE between Gram matrices, averaged over views and batch.
        """
        token_cap = int(self.config.get('gram_anchor_token_subsample', 0))
        total = 0.0
        count = 0
        for s_p, t_p in zip(student_patch_list, teacher_patch_list):
            # s_p, t_p: (B, T, D)
            if s_p.numel() == 0 or t_p.numel() == 0:
                continue
            B, T, D = s_p.shape
            # Normalize across D for cosine similarity
            s_n = torch.nn.functional.normalize(s_p, dim=-1)
            with torch.no_grad():
                t_n = torch.nn.functional.normalize(t_p, dim=-1)

            # Subsample tokens if requested
            if token_cap and T > token_cap:
                idx = torch.randperm(T, device=s_n.device)[:token_cap]
                s_n = s_n[:, idx]
                t_n = t_n[:, idx]

            # Gram matrices (B, T, T)
            Gs = torch.einsum('btd,bsd->bts', s_n, s_n)
            Gt = torch.einsum('btd,bsd->bts', t_n, t_n)

            # MSE over all entries; averaging scales by token count
            view_loss = torch.mean((Gs - Gt) ** 2)
            total = total + view_loss
            count += 1

        return total / max(count, 1)

    def training_step(self, batch, batch_idx):
        loss = self.common_step(batch, batch_idx)
        self.log('train_loss', loss, prog_bar=True, on_step=True, on_epoch=False)
        return loss

    def validation_step(self, batch, batch_idx):
        loss = self.common_step(batch, batch_idx)
        self.log('val_loss', loss, on_epoch=True)
        self.current_loss += loss
        return loss

    def on_after_backward(self):
        # freeze last layer gradients periodically
        self.student_head.cancel_last_layer_gradients(current_epoch=self.current_epoch)

    def configure_optimizers(self):
        if not hasattr(self, 'num_training_batches') or self.num_training_batches <= 0:
            raise ValueError(
                '`num_training_batches` must be set before configuring optimizers'
            )
        wd = float(self.config.get('weight_decay', 0.0))
        scale_lr_with_batch = bool(self.config.get('scale_lr_with_batch', True))
        base_batch_size = int(self.config.get('base_batch_size', max(1, int(self.config.get('batch_size', 1)))))
        # Attempt to estimate effective batch size (approximate)
        eff_bs = int(self.config.get('effective_batch_size', 0))
        if eff_bs <= 0:
            try:
                devices = getattr(self.trainer, 'num_devices', 1) or 1
                accum = getattr(self.trainer, 'accumulate_grad_batches', 1) or 1
            except Exception:
                devices, accum = 1, 1
            eff_bs = int(max(1, int(self.config.get('batch_size', 1))) * devices * accum)
        lr_scale = (eff_bs / base_batch_size) if scale_lr_with_batch and base_batch_size > 0 else 1.0

        # Build optimizer parameter groups: bias/norm no-decay, lower LR for backbone
        def split_decay_groups(module: nn.Module):
            # Identify norm-layer parameters to exclude from decay
            norm_types = (
                nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d,
                nn.GroupNorm, nn.LayerNorm, nn.InstanceNorm1d,
                nn.InstanceNorm2d, nn.InstanceNorm3d,
            )
            norm_param_ids = set()
            for m in module.modules():
                if isinstance(m, norm_types):
                    for _, p in m.named_parameters(recurse=False):
                        if p.requires_grad:
                            norm_param_ids.add(id(p))

            decay, no_decay = [], []
            embed_keys = (
                'pos_embed', 'position_embeddings', 'absolute_pos_embed',
                'relative_position_bias_table', 'cls_token'
            )
            for name, p in module.named_parameters(recurse=True):
                if not p.requires_grad:
                    continue
                if id(p) in norm_param_ids or name.endswith('bias') or any(k in name.lower() for k in embed_keys):
                    no_decay.append(p)
                else:
                    decay.append(p)
            return decay, no_decay

        # Backbone groups (trainable only)
        layer_lr_decay = float(self.config.get('layer_lr_decay', 0.0) or 0.0)
        bb_decay, bb_no_decay = [], []
        if layer_lr_decay and layer_lr_decay > 0.0 and layer_lr_decay != 1.0:
            # Layer-wise LR decay for transformer blocks
            blocks = self._find_transformer_layers(self.student_backbone)
            considered = set()
            if isinstance(blocks, (nn.ModuleList, list, tuple)) and len(blocks) > 0:
                L = len(blocks)
                for idx, blk in enumerate(blocks):
                    if not any(p.requires_grad for p in blk.parameters(recurse=True)):
                        continue
                    lr_factor = layer_lr_decay ** (L - 1 - idx)
                    dec, nde = split_decay_groups(blk)
                    for p in dec + nde:
                        considered.add(id(p))
                    if dec:
                        bb_decay.append((dec, lr_factor))
                    if nde:
                        bb_no_decay.append((nde, lr_factor))
                # leftover backbone params (e.g., embeddings) → smallest LR (extra decay step)
                rest_decay, rest_no_decay = [], []
                for name, p in self.student_backbone.named_parameters(recurse=True):
                    if not p.requires_grad or id(p) in considered:
                        continue
                    # classify to decay/no-decay
                    if name.endswith('bias') or any(k in name.lower() for k in (
                        'pos_embed','position_embeddings','absolute_pos_embed','relative_position_bias_table'
                    )):
                        rest_no_decay.append(p)
                    else:
                        rest_decay.append(p)
                if rest_decay:
                    bb_decay.append((rest_decay, layer_lr_decay ** L))
                if rest_no_decay:
                    bb_no_decay.append((rest_no_decay, layer_lr_decay ** L))
            else:
                # Fallback to flat grouping if blocks not found
                d, nd = split_decay_groups(self.student_backbone)
                if d:
                    bb_decay.append((d, 1.0))
                if nd:
                    bb_no_decay.append((nd, 1.0))
        else:
            d, nd = split_decay_groups(self.student_backbone)
            if d:
                bb_decay.append((d, 1.0))
            if nd:
                bb_no_decay.append((nd, 1.0))
        # Head groups
        head_decay, head_no_decay = split_decay_groups(self.student_head)

        # Compute backbone LR (scaled)
        head_lr = float(self.lr) * float(lr_scale)
        backbone_lr = float(self.backbone_lr) if self.backbone_lr is not None else head_lr * float(self.backbone_lr_mult)

        param_groups = []
        for params, factor in bb_decay:
            param_groups.append({"params": params, "lr": backbone_lr * factor, "weight_decay": wd})
        for params, factor in bb_no_decay:
            param_groups.append({"params": params, "lr": backbone_lr * factor, "weight_decay": 0.0})
        if head_decay:
            param_groups.append({"params": head_decay, "lr": head_lr, "weight_decay": wd})
        if head_no_decay:
            param_groups.append({"params": head_no_decay, "lr": head_lr, "weight_decay": 0.0})

        # Fallback in unlikely case groups are empty
        if not param_groups:
            param_groups = [{"params": [p for p in self.parameters() if p.requires_grad], "lr": float(self.lr), "weight_decay": wd}]

        # Optionally use fused AdamW on CUDA
        use_fused = bool(self.config.get('use_fused_adamw', False)) and torch.cuda.is_available()
        if use_fused:
            try:
                optimizer = torch.optim.AdamW(param_groups, fused=True)
            except TypeError:
                optimizer = torch.optim.AdamW(param_groups)
        else:
            optimizer = torch.optim.AdamW(param_groups)

        # Cosine LR with warmup (per-step)
        warmup_epochs = int(self.config.get('warmup_epochs', 10))
        warmup_steps = warmup_epochs * self.num_training_batches
        total_steps = int(self.config.get('epochs', 1)) * self.num_training_batches
        import math
        def lr_lambda(step):
            if total_steps <= 0:
                return 1.0
            if step < warmup_steps and warmup_steps > 0:
                return float(step) / float(max(1, warmup_steps))
            # Cosine from 1 → 0
            progress = float(step - warmup_steps) / float(max(1, total_steps - warmup_steps))
            return 0.5 * (1.0 + math.cos(math.pi * min(1.0, max(0.0, progress))))

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)
        out = {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'interval': 'step',
                'frequency': 1,
            }
        }
        # Optional cosine schedule for weight decay across steps
        if bool(self.config.get('weight_decay_cosine', False)):
            self._wd_schedule = {
                'start': float(self.config.get('weight_decay', 0.0)),
                'end': float(self.config.get('weight_decay_end', self.config.get('weight_decay', 0.0))),
                'total': int(self.config.get('epochs', 1)) * self.num_training_batches,
            }
        else:
            self._wd_schedule = None
        return out

    def on_before_optimizer_step(self, optimizer, optimizer_idx=None):
        # Update param group weight decay with cosine schedule if enabled
        s = getattr(self, '_wd_schedule', None)
        if s:
            total = max(1, int(s['total']))
            step = min(self.global_step, total)
            import math
            progress = step / total
            wd_now = float(s['end'] + (s['start'] - s['end']) * 0.5 * (1.0 + math.cos(math.pi * progress)))
            for pg in optimizer.param_groups:
                if pg.get('weight_decay', 0.0) > 0.0:
                    pg['weight_decay'] = wd_now

    # rely on Trainer's built-in gradient clipping via precision plugin

    def on_train_epoch_end(self):
        if not self.config.get('debug'):
            latest_path = os.path.join(self.config.get('out_path', '.'), 'latest.pth')
            torch.save(self.state_dict(), latest_path)

    def on_validation_start(self):
        self.current_loss = torch.tensor(0.).to(self.device)

    def on_validation_end(self):
        # save latest checkpoint
        if not self.config.get('debug'):
            latest = os.path.join(self.config.get('out_path', '.'), 'latest.pth')
            torch.save(self.state_dict(), latest)

        # save best model if improved
        if self.current_loss < self.best_loss:
            self.best_loss = self.current_loss
            if wandb.run:
                wandb.run.summary['best_val_loss'] = float(self.best_loss)
                wandb.run.summary['best_val_epoch'] = int(self.current_epoch)
                wandb.log({'epoch': int(self.current_epoch)})
            best = os.path.join(self.config.get('out_path', '.'), 'best_model.pth')
            torch.save(self.state_dict(), best)

    # ---- Feature extraction helpers for integration ----
    @torch.no_grad()
    def encode_features(self, pixel_values: torch.Tensor, pool: str = 'cls') -> torch.Tensor:
        """Extract backbone features with pooling.

        pool: 'cls' or 'mean' over patch tokens.
        Returns (B, D).
        """
        self.eval()
        cls, patches = self._backbone_tokens(pixel_values.to(self.device), self.student_backbone)
        if pool == 'cls':
            return cls
        elif pool == 'mean':
            return patches.mean(dim=1) if patches.numel() > 0 else cls
        else:
            raise ValueError(f"Unknown pool: {pool}")

    @torch.no_grad()
    def cache_features(self, dataloader, out_path: str, pool: str = 'cls', dtype: str = 'float16') -> str:
        """Cache features from a dataloader to a .pt file.

        - Runs in eval mode, disables dropout, keeps same normalization as training.
        - dtype: 'float16', 'bfloat16', or 'float32'.
        """
        self.eval()
        if dtype == 'float16':
            cast = torch.float16
        elif dtype == 'bfloat16':
            cast = torch.bfloat16
        else:
            cast = torch.float32

        feats = []
        for batch in dataloader:
            # Accept (inputs, ...) or just inputs
            x = batch[0] if isinstance(batch, (list, tuple)) else batch
            if isinstance(x, (list, tuple)):
                # If multi-crop list, take first global view only
                x = x[0]
            x = x.to(self.device, non_blocking=True)
            f = self.encode_features(x, pool=pool).to(dtype=cast).cpu()
            feats.append(f)
        feats = torch.cat(feats, dim=0)
        os.makedirs(os.path.dirname(out_path) or '.', exist_ok=True)
        torch.save({'features': feats, 'pool': pool, 'dtype': str(cast)}, out_path)
        return out_path

    def export_backbone(self, out_path: str) -> str:
        """Export backbone-only state dict for downstream use."""
        state = {'backbone': self.student_backbone.state_dict()}
        os.makedirs(os.path.dirname(out_path) or '.', exist_ok=True)
        torch.save(state, out_path)
        return out_path

    def set_num_training_batches(self, num_batches):
        self.num_training_batches = num_batches
