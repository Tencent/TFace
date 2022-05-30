import torch
import torch.nn as nn
import torch.nn.functional as F


def euclidean_dist(x, y):
    """
    Args:
        x: pytorch Variable, with shape [m, d]
        y: pytorch Variable, with shape [n, d]
    Returns:
        dist: pytorch Variable, with shape [m, n]
    """

    m, n = x.size(0), y.size(0)
    xx = torch.pow(x, 2).sum(1, keepdim=True).expand(m, n)
    yy = torch.pow(y, 2).sum(1, keepdim=True).expand(n, m).t()
    dist = xx + yy
    dist = torch.addmm(dist, x, y.t(), beta=1, alpha=-2)
    dist = dist.clamp(min=1e-12).sqrt()  # for numerical stability
    return dist


class HardDarkRank(nn.Module):
    """ DarkRank: Accelerating Deep Metric Learning via Cross Sample Similarities Transfer. AAAI 2018
    """
    def __init__(self, alpha=3, beta=3, permute_len=3):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.permute_len = permute_len

    def forward(self, g_s, g_t):
        g_s = F.normalize(g_s)
        g_t = F.normalize(g_t)

        score_teacher = -1 * self.alpha * \
            euclidean_dist(g_t, g_t).pow(self.beta)
        score_student = -1 * self.alpha * \
            euclidean_dist(g_s, g_s).pow(self.beta)

        permute_idx = score_teacher.sort(dim=1, descending=True)[
            1][:, 1:(self.permute_len+1)]
        ordered_student = torch.gather(score_student, 1, permute_idx)

        log_prob = (ordered_student - torch.stack([torch.logsumexp(
            ordered_student[:, i:], dim=1) for i in range(permute_idx.size(1))], dim=1)).sum(dim=1)
        loss = (-1 * log_prob).mean()
        return loss
