import torch
import torch.nn as nn
import torch.nn.functional as F


class CosineDistance(nn.Module):
    """ Feature-Feature cosine distance
    """

    def __init__(self):
        super(CosineDistance, self).__init__()

    def forward(self, f_s, f_t):
        batch_size = f_s.size(0)
        # just to make sure the tensor is 2D
        f_s = f_s.view(batch_size, -1)
        f_t = f_t.view(batch_size, -1)
        # normalize the feature
        f_s = F.normalize(f_s)
        f_t = F.normalize(f_t)

        distance = F.cosine_similarity(f_s, f_t, dim=1, eps=1e-8)
        loss = 1 - torch.mean(distance)
        return loss
