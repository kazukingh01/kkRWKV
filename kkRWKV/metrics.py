import torch
from torch.nn import functional as F


def calc_cross_entropy(logits: torch.Tensor, targets: torch.Tensor) -> list[torch.Tensor]:
    _, n_task = targets.shape
    n_label = logits.shape[-1] // n_task
    list_ce = []
    for task_idx in range(n_task):
        task_target = targets[:, task_idx]  # [batch_size]
        task_logit  = logits[:, task_idx * n_label:(task_idx + 1) * n_label]  # [batch_size, num_classes]
        task_loss   = F.cross_entropy(task_logit, task_target)
        list_ce.append(task_loss)
    return list_ce

def calc_accuracy(logits: torch.Tensor, targets: torch.Tensor) -> list[torch.Tensor]:
    _, n_task = targets.shape
    n_label  = logits.shape[-1] // n_task
    list_acc = []
    for task_idx in range(n_task):
        task_target    = targets[:, task_idx]
        task_logit     = logits[:, task_idx * n_label:(task_idx + 1) * n_label]
        task_pred      = torch.argmax(task_logit, dim=-1)
        task_correct   = (task_pred == task_target).sum().item()
        list_acc.append(task_correct / task_target.shape[0])
    return list_acc
