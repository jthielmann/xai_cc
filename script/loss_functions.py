import torch
import torch.nn as nn
import torch.nn.functional as F

class SparsityLoss(nn.Module):
    def __init__(self, layername, model):
        super(SparsityLoss, self).__init__()
        self.activations = None
        for name, module in model.named_modules():
            if name == layername:
                module.register_forward_hook(self.fw_hook)

    def fw_hook(self, module, input, output):
        self.activations = output.clone()

    def forward(self, out, label):
        a = self.activations.clamp(min=0).amax((2, 3)) / (self.activations.shape[2] * self.activations.shape[3])
        sparsity_loss = (a[:, a.sum(0) > 0].mean())
        return sparsity_loss


class CompositeLoss(nn.Module):
    def __init__(self, losses, weights = None):
        super().__init__()
        self.losses = losses
        if weights:
            self.weights = weights
        else:
            self.weights = [torch.tensor(1/len(losses)) for _ in losses]
        if len(self.weights) != len(self.losses):
            print(len(self.weights), len(self.losses), len(losses), len(weights) if weights else "weights is None")
            raise ValueError("CompositeLoss: Number of weights must match number of losses")

    def forward(self, out, label):
        losses = []
        for idx, loss_fn in enumerate(self.losses):
            loss = loss_fn(out, label)
            loss = torch.mul(loss, self.weights[idx])
            losses.append(loss)
        losses = torch.stack(losses)
        result = torch.sum(losses)
        return result


def dino_loss(student_output, teacher_output, temperature):
    teacher_out = F.softmax(teacher_output / temperature, dim=-1).detach()
    student_out = F.log_softmax(student_output / temperature, dim=-1)
    return -torch.mean(torch.sum(teacher_out * student_out, dim=-1))
