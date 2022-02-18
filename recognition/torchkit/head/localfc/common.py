import torch
import torch.nn as nn
from torchkit.util.utils import l2_norm


def calc_logits(embeddings, kernel):
    """ calculate original logits
    """
    embeddings = l2_norm(embeddings, axis=1)
    kernel_norm = l2_norm(kernel, axis=0)
    cos_theta = torch.mm(embeddings, kernel_norm)
    cos_theta = cos_theta.clamp(-1, 1)  # for numerical stability
    with torch.no_grad():
        origin_cos = cos_theta.clone()
    return cos_theta, origin_cos
