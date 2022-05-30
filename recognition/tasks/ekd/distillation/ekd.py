import torch
import torch.nn as nn
import torch.nn.functional as F


class EKD(nn.Module):
    """ Evaluation-oriented knowledge distillation for deep face recognition, CVPR2022
    """

    def __init__(self):
        super().__init__()
        self.topk = 2000
        self.t = 0.01
        self.anchor = [10, 100, 1000, 10000, 100000, 1000000]
        self.momentum = 0.01
        self.register_buffer('s_anchor', torch.zeros(len(self.anchor)))
        self.register_buffer('t_anchor', torch.zeros(len(self.anchor)))

    def forward(self, g_s, g_t, labels):
        # normalize feature
        class_size = labels.size(0)
        g_s = g_s.view(class_size, -1)
        g_s = F.normalize(g_s)
        classes_eq = (labels.repeat(class_size, 1) == labels.view(-1, 1).repeat(1, class_size))
        # print("classes_eq = ", classes_eq)
        similarity_student = torch.mm(g_s, g_s.transpose(0, 1))
        s_inds = torch.triu(torch.ones(classes_eq.size(), device=g_s.device), 1).bool()

        pos_inds = classes_eq[s_inds]
        # print("pos_inds = ", pos_inds)
        neg_inds = ~classes_eq[s_inds]
        # print("neg_inds = ", neg_inds)
        s = similarity_student[s_inds]
        pos_similarity_student = torch.masked_select(s, pos_inds)
        neg_similarity_student = torch.masked_select(s, neg_inds)
        sorted_s_neg, sorted_s_index = torch.sort(neg_similarity_student, descending=True)

        with torch.no_grad():
            g_t = g_t.view(class_size, -1)
            g_t = F.normalize(g_t)
            similarity_teacher = torch.mm(g_t, g_t.transpose(0, 1))
            t = similarity_teacher[s_inds]
            pos_similarity_teacher = torch.masked_select(t, pos_inds)
            neg_similarity_teacher = torch.masked_select(t, neg_inds)
            sorted_t_neg, _ = torch.sort(neg_similarity_teacher, descending=True)
            length = sorted_s_neg.size(0)
            select_indices = [length // anchor for anchor in self.anchor]
            s_neg_thresholds = sorted_s_neg[select_indices]
            t_neg_thresholds = sorted_t_neg[select_indices]
            self.s_anchor = self.momentum * s_neg_thresholds + (1 - self.momentum) * self.s_anchor
            self.t_anchor = self.momentum * t_neg_thresholds + (1 - self.momentum) * self.t_anchor
        s_pos_kd_loss = self.relative_loss(pos_similarity_student, pos_similarity_teacher)

        s_neg_selected = neg_similarity_student[sorted_s_index[0:self.topk]]
        t_neg_selected = neg_similarity_teacher[sorted_s_index[0:self.topk]]

        s_neg_kd_loss = self.relative_loss(s_neg_selected, t_neg_selected)

        loss = s_pos_kd_loss * 0.02 + s_neg_kd_loss * 0.01

        return loss

    def sigmoid(self, inputs, temp=1.0):
        """ temperature controlled sigmoid
            takes as input a torch tensor (tensor) and passes it through a sigmoid, controlled by temperature: temp
        """
        exponent = -inputs / temp
        # clamp the input tensor for stability
        exponent = torch.clamp(exponent, min=-50, max=50)
        y = 1.0 / (1.0 + torch.exp(exponent))
        return y

    def relative_loss(self, s_similarity, t_similarity):
        s_distance = s_similarity.unsqueeze(1) - self.s_anchor.unsqueeze(0)
        t_distance = t_similarity.unsqueeze(1) - self.t_anchor.unsqueeze(0)

        s_rank = self.sigmoid(s_distance, self.t)
        t_rank = self.sigmoid(t_distance, self.t)

        s_rank_count = s_rank.sum(axis=1, keepdims=True)
        t_rank_count = t_rank.sum(axis=1, keepdims=True)

        s_kd_loss = F.mse_loss(s_rank_count, t_rank_count)
        return s_kd_loss
