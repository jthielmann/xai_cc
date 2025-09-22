import sys
from pathlib import Path

# Ensure repo root is on sys.path so `import script...` works when running this file directly
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from lightly.loss import DINOLoss  # type: ignore
except Exception:
    # Minimal stand-in for DINOLoss to keep the sanity test offline.
    # It aligns teacher's 2 global outputs with student's first two outputs using simple MSE.
    class DINOLoss(nn.Module):
        def __init__(self, output_dim: int = 2048, **kwargs):
            super().__init__()
            self.output_dim = output_dim

        def forward(self, teacher_out, student_out, epoch=None):
            # teacher_out: list of 2 tensors [(B,D), (B,D)]
            # student_out: list of N tensors
            if len(teacher_out) != 2:
                raise ValueError("teacher_out must have exactly 2 views")
            if len(student_out) < 2:
                raise ValueError("student_out must have at least 2 views")
            t0, t1 = teacher_out[0].detach(), teacher_out[1].detach()
            s0, s1 = student_out[0], student_out[1]
            return F.mse_loss(s0, t0) + F.mse_loss(s1, t1)

from lightly.models.utils import update_momentum


class DummyProcessor:
    # Minimal HF-like processor for normalization values
    image_mean = [0.7406, 0.5331, 0.7059]
    image_std = [0.1651, 0.2174, 0.1574]


class MinimalDINO:
    """A minimal object compatible with lit_dino.DINO.common_step.

    Avoids any network/model downloads. It exposes only the attributes/methods
    that common_step uses: momentum update on student/teacher modules, forward
    for student/teacher, a DINOLoss criterion, and the HF normalization toggle.
    """

    def __init__(self, use_hf_normalize: bool = False, device: str = "cpu"):
        self.device = torch.device(device)
        self.config = {"epochs": 1, "out_path": ".", "debug": True}
        self.global_step = 0
        self.total_steps = 10
        self.current_epoch = 0

        # Normalization control used by common_step
        self.use_hf_normalize = bool(use_hf_normalize)
        self.processor = DummyProcessor()

        # Minimal student/teacher modules with parameters (so EMA has something to update)
        # We keep the head input-dim at 3 (channel avg) to keep it simple
        self.student_backbone = nn.Identity()
        self.teacher_backbone = nn.Identity()
        self.student_head = nn.Linear(3, 2048, bias=False)
        self.teacher_head = nn.Linear(3, 2048, bias=False)

        self.criterion = DINOLoss(output_dim=2048, warmup_teacher_temp_epochs=0)

        # Counters to verify view-routing
        self.calls_student = 0
        self.calls_teacher = 0

        # Bind the method from the real class at runtime to reuse the pipeline logic
        # Import the real method while avoiding heavy optional deps
        import sys, types, types as _types, types as _t
        if 'transformers' not in sys.modules:
            import types as _types
            dummy = _types.ModuleType('transformers')
            class _Auto:
                @classmethod
                def from_pretrained(cls, *a, **k):
                    raise RuntimeError('from_pretrained called in offline sanity test')
            # Provide common names that importers might look for
            dummy.AutoImageProcessor = _Auto
            dummy.Dinov3Model = _Auto
            dummy.AutoModel = _Auto
            # Fallback for any other name imports
            def __getattr__(name):
                return _Auto
            dummy.__getattr__ = __getattr__
            sys.modules['transformers'] = dummy
        from script.model.lit_dino import DINO as RealDINO
        self.common_step = RealDINO.common_step.__get__(self, MinimalDINO)

    def forward(self, pixel_values: torch.Tensor):
        self.calls_student += 1
        # cheap CLS proxy: channel-wise avg pool, then linear head to 2048
        x = pixel_values.mean(dim=(2, 3))  # (B, 3)
        return self.student_head(x)

    def forward_teacher(self, pixel_values: torch.Tensor):
        self.calls_teacher += 1
        with torch.no_grad():
            x = pixel_values.mean(dim=(2, 3))  # (B, 3)
        return self.teacher_head(x)


def _make_multicrop_batch(batch_size=4, gsize=224, lsize=96, n_local=4, value_range=(0.0, 1.0)):
    low, high = value_range
    g1 = torch.rand(batch_size, 3, gsize, gsize) * (high - low) + low
    g2 = torch.rand(batch_size, 3, gsize, gsize) * (high - low) + low
    locals_ = [torch.rand(batch_size, 3, lsize, lsize) * (high - low) + low for _ in range(n_local)]
    return [g1, g2] + locals_


def run_once(use_hf_normalize: bool):
    model = MinimalDINO(use_hf_normalize=use_hf_normalize, device="cpu")
    views = _make_multicrop_batch()
    loss = model.common_step(views, batch_idx=0)
    assert torch.is_tensor(loss) and loss.dim() == 0, "Loss should be a scalar tensor"
    assert model.calls_teacher == 2, "Teacher must process exactly 2 global crops"
    assert model.calls_student == len(views), "Student must process all views"
    return float(loss.item()), model.calls_student, model.calls_teacher


if __name__ == "__main__":
    l0, s0, t0 = run_once(use_hf_normalize=False)
    print(f"OK: use_hf_normalize=False · loss={l0:.4f} · student_calls={s0} · teacher_calls={t0}")

    l1, s1, t1 = run_once(use_hf_normalize=True)
    print(f"OK: use_hf_normalize=True  · loss={l1:.4f} · student_calls={s1} · teacher_calls={t1}")
