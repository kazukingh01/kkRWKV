import torch
from torch.nn import functional as F


def calc_cross_entropy(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    _, n_task = targets.shape
    n_label = logits.shape[-1] // n_task
    total_loss = 0
    for task_idx in range(n_task):
        task_target = targets[:, task_idx]  # [batch_size]
        task_logit  = logits[:, task_idx * n_label:(task_idx + 1) * n_label]  # [batch_size, num_classes]
        task_loss   = F.cross_entropy(task_logit, task_target)
        total_loss += task_loss
    loss = total_loss / n_task
    return loss

def calc_accuracy(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    _, n_task = targets.shape
    n_label = logits.shape[-1] // n_task
    total_correct = 0
    total_samples = 0
    for task_idx in range(n_task):
        task_target    = targets[:, task_idx]
        task_logit     = logits[:, task_idx * n_label:(task_idx + 1) * n_label]
        task_pred      = torch.argmax(task_logit, dim=-1)
        task_correct   = (task_pred == task_target).sum().item()
        total_correct += task_correct
        total_samples += len(task_target)
    return total_correct / total_samples
