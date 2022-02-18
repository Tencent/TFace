import torch
import torch.nn.functional as F


def compute_metrics(model_outputs, labels):
    """
    Compute the accuracy metrics.
    """
    real_probs = F.softmax(model_outputs, dim=1)[:, 0]
    bin_preds = (real_probs <= 0.5).int()
    bin_labels = (labels != 0).int()

    real_cnt = (bin_labels == 0).sum()
    fake_cnt = (bin_labels == 1).sum()

    acc = (bin_preds == bin_labels).float().mean()

    real_acc = (bin_preds == bin_labels)[torch.where(bin_labels == 0)].sum() / (real_cnt + 1e-12)
    fake_acc = (bin_preds == bin_labels)[torch.where(bin_labels == 1)].sum() / (fake_cnt + 1e-12)

    return acc.item(), real_acc.item(), fake_acc.item(), real_cnt.item(), fake_cnt.item()
