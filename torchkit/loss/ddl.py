import torch
import torch.nn.functional as F


class DDL(torch.nn.Module):
    """ Implented of  "https://arxiv.org/abs/2002.03662"
    """
    def __init__(self, pos_kl_weight=0.1, neg_kl_weight=0.02,
                 order_loss_weight=0.5, positive_threshold=0.0):
        """ Args:
            pos_kl_weight: weight for positive kl loss
            neg_kl_weight: weight for negative kl loss
            order_loss_weight: weight for order loss
            positive_threshold:  threshold fo positive pair
        """
        super(DDL, self).__init__()
        self.pos_kl_weight = pos_kl_weight
        self.neg_kl_weight = neg_kl_weight
        self.order_loss_weight = order_loss_weight
        self.positive_threshold = positive_threshold
        self.register_buffer('t', torch.arange(0, 1.0, 0.001).view(-1, 1).t())

    def forward(self, neg_features, pos_pair_features_first, pos_pair_features_second):
        assert len(pos_pair_features_first) == len(pos_pair_features_second)
        assert len(neg_features) == len(pos_pair_features_first)

        neg_distributions, neg_distances = self._neg_distribution(neg_features)
        pos_distirbutions, pos_distances = self._pos_distribution(pos_pair_features_first, pos_pair_features_second)

        pos_kl_losses = []
        neg_kl_losses = []
        order_losses = []
        for i in range(1, len(pos_distirbutions)):  # first branch as the anchor
            pos_kl = self._kl(pos_distirbutions[0], pos_distirbutions[i])
            pos_kl_losses.append(pos_kl)
        for i in range(1, len(neg_distributions)):  # first branch as the anchor
            neg_kl = self._kl(neg_distributions[0], neg_distributions[i])
            neg_kl_losses.append(neg_kl)
        for neg in neg_distances:
            for pos in pos_distances:
                order_loss = torch.mean(neg) - torch.mean(pos)
                order_losses.append(order_loss)
        ddl_loss = sum(pos_kl_losses) * self.pos_kl_weight + sum(neg_kl_losses) * \
            self.neg_kl_weight + sum(order_losses) * self.order_loss_weight
        return ddl_loss, neg_distances, pos_distances

    def _kl(self, anchor_distribution, distribution):
        loss = F.kl_div(torch.log(distribution + 1e-9), anchor_distribution + 1e-9, reduction="batchmean")
        return loss

    def _histogram(self, dists):
        dists = dists.view(-1, 1)
        simi_p = torch.mm(dists, torch.ones_like(self.t)) - torch.mm(torch.ones_like(dists), self.t)
        simi_p = torch.sum(torch.exp(-0.5 * torch.pow((simi_p / 0.1), 2)), 0, keepdim=True)
        p_sum = torch.sum(simi_p, 1)
        simi_p_normed = simi_p / p_sum
        return simi_p_normed

    def _pos_distribution(self, first_features, second_features, positive_threshold=0.):
        pos_distirbutions = []
        pos_distances = []
        for first_feature, second_feature in zip(first_features, second_features):
            first_feature = F.normalize(first_feature)
            second_feature = F.normalize(second_feature)
            pos_distance = torch.mul(first_feature, second_feature).sum(dim=1)
            pos_distance = torch.masked_select(pos_distance, pos_distance > positive_threshold)
            pos_p = self._histogram(pos_distance)
            pos_distirbutions.append(pos_p)
            pos_distances.append(pos_distance)
        return pos_distirbutions, pos_distances

    def _neg_distribution(self, neg_features):
        neg_distributions = []
        neg_distances = []
        for neg_feature in neg_features:
            neg_feature = F.normalize(neg_feature)
            neg_distance = torch.mm(neg_feature, neg_feature.transpose(0, 1))
            neg_distance = torch.triu(neg_distance, diagonal=1)
            neg_distance, _ = torch.max(neg_distance, dim=1)
            neg_p = self._histogram(neg_distance)
            neg_distributions.append(neg_p)
            neg_distances.append(neg_distance)
        return neg_distributions, neg_distances
