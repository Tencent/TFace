import torch
import torch.nn as nn
import torch.nn.functional as F


class Similarity_perserving(nn.Module):
    """ Similarity-Preserving Knowledge Distillation, ICCV2019
    """

    def __init__(self):
        super().__init__()

    def forward(self, g_s, g_t):
        return self.similarity_loss(g_s, g_t)

    # different from the origin paper, here use the embedding feature
    def similarity_loss(self, f_s, f_t):
        batch_size = f_s.size(0)
        # just to make sure the tensor is 2D
        f_s = f_s.view(batch_size, -1)
        f_t = f_t.view(batch_size, -1)
        # normalize the feature
        f_s = F.normalize(f_s)
        f_t = F.normalize(f_t)
        s_similarity = torch.mm(f_s, torch.t(f_s))
        t_similarity = torch.mm(f_t, torch.t(f_t))

        diff_similarity = s_similarity - t_similarity
        diff_norm = torch.norm(diff_similarity)
        loss = diff_norm * diff_norm / (batch_size * batch_size)

        return loss
