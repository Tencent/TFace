import copy
import torch
import torch.nn as nn


class DCL(nn.Module):
    def __init__(self, base_encoder, dim=128, K=65536, m=0.999, T=0.07, threshold=5):
        """Dual Contrastive learning for Face Forgery Detection

        Args:
            base_encoder (nn.Module)
            dim (int, optional): Feature dimension. Defaults to 128.
            K (int, optional): Queue size. Defaults to 65536.
            m (float, optional): Parameters for moving average. Defaults to 0.999.
            T (float, optional): Temperature. Defaults to 0.07.
            threshold (int, optional): Parameters for hard sample selection. Defaults to 5.
        """
        super(DCL, self).__init__()

        self.K = K
        self.m = m
        self.T = T
        self.threshold = threshold
        self.margin = 0.9
        self.criterion = nn.CrossEntropyLoss().cuda()

        self.encoder_q = base_encoder
        self.encoder_k = copy.deepcopy(base_encoder)

        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.requires_grad = False  # ema update
        # prototype for real feature
        self.register_buffer("real_queue", torch.randn(dim))
        self.register_buffer("hard_fake_queue", torch.randn(dim, K + 128))  # hard queue for fake image

        self.register_buffer("hard_real_queue", torch.randn(dim, K + 128))  # hard queue for real image
        # prototype for fake feature
        self.register_buffer("ancor_queue", torch.randn(dim))
        self.criterion = nn.CrossEntropyLoss().cuda()

        self.real_queue = nn.functional.normalize(self.real_queue, dim=0)
        self.hard_fake_queue = nn.functional.normalize(self.hard_fake_queue, dim=0)
        self.hard_real_queue = nn.functional.normalize(self.hard_real_queue, dim=0)
        self.ancor_queue = nn.functional.normalize(self.ancor_queue, dim=0)

        self.register_buffer("real_queue_ptr_hard", torch.zeros(1, dtype=torch.long))  # pointer for queue
        self.register_buffer("fake_queue_ptr_hard", torch.zeros(1, dtype=torch.long))
        self.alpha = 0.9

    @torch.no_grad()
    def _momentum_update_key_encoder(self):
        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.data = param_k.data * self.m + param_q.data * (1. - self.m)

    @torch.no_grad()
    def _dequeue_and_enqueue_hard(self, keys, type_queue="real"):

        keys = concat_all_gather(keys)

        batch_size = keys.shape[0]
        if type_queue == "real":
            ptr = int(self.real_queue_ptr_hard)
            self.hard_real_queue[:, ptr:ptr + batch_size] = keys.T
            ptr = (ptr + batch_size) % self.K
            self.real_queue_ptr_hard[0] = ptr
        else:
            ptr = int(self.fake_queue_ptr_hard)
            self.hard_fake_queue[:, ptr:ptr + batch_size] = keys.T
            ptr = (ptr + batch_size) % self.K
            self.fake_queue_ptr_hard[0] = ptr

    @torch.no_grad()
    def _dequeue_and_enqueue_real(self, keys):
        # gather keys before updating queue
        keys = concat_all_gather(keys)

        batch_size = keys.shape[0]

        ptr = int(self.queue_ptr)
        assert self.K % batch_size == 0

        self.queue[:, ptr:ptr + batch_size] = keys.T
        ptr = (ptr + batch_size) % self.K

        self.queue_ptr[0] = ptr

    @torch.no_grad()
    def _batch_shuffle_ddp(self, x):

        batch_size_this = x.shape[0]
        x_gather = concat_all_gather(x)
        batch_size_all = x_gather.shape[0]

        num_gpus = batch_size_all // batch_size_this

        idx_shuffle = torch.randperm(batch_size_all).cuda()

        torch.distributed.broadcast(idx_shuffle, src=0)

        idx_unshuffle = torch.argsort(idx_shuffle)
        gpu_idx = torch.distributed.get_rank()
        idx_this = idx_shuffle.view(num_gpus, -1)[gpu_idx]

        return x_gather[idx_this], idx_unshuffle

    @torch.no_grad()
    def _batch_unshuffle_ddp(self, x, idx_unshuffle):

        # gather from all gpus
        batch_size_this = x.shape[0]
        x_gather = concat_all_gather(x)
        batch_size_all = x_gather.shape[0]

        num_gpus = batch_size_all // batch_size_this

        # restored index for this gpu
        gpu_idx = torch.distributed.get_rank()
        idx_this = idx_unshuffle.view(num_gpus, -1)[gpu_idx]

        return x_gather[idx_this]

    def _angular_norm(self, feature, w):
        feature = nn.functional.normalize(feature, dim=1)
        w = nn.functional.normalize(w, dim=1)
        feature = nn.functional.normalize((feature - w), dim=1)
        return feature

    def forward(self, im_q, mask=None, im_k=None, labels=None):
        """
        Input:
            im_q: a batch of query images
            im_k: a batch of key images
        Output:
            logits, targets
        """
        output_q, feature_map = self.encoder_q(im_q)  # queries: NxC

        if im_k == None and labels == None:
            # for inference
            return output_q, feature_map
        q = self.encoder_q.feature_squeeze(feature_map).squeeze().view(feature_map.shape[0], -1)
        q = nn.functional.normalize(q, dim=1)
        feature_map = feature_map.view(feature_map.shape[0], feature_map.shape[1], -1)

        feature_map = nn.functional.normalize(feature_map, dim=1)

        with torch.no_grad():
            self._momentum_update_key_encoder()

            im_k, idx_unshuffle = self._batch_shuffle_ddp(im_k)

            _, feature_k = self.encoder_k(im_k)  # keys: NxC
            k = self.encoder_k.feature_squeeze(feature_k).squeeze().view(feature_k.shape[0], -1)

            k = nn.functional.normalize(k, dim=1)
            k = self._batch_unshuffle_ddp(k, idx_unshuffle)
        labels = labels.float()
        real_k = k[labels == 0.0]
        fake_k = k[labels == 1.0]

        real_q = q[labels == 0.0]
        fake_q = q[labels == 1.0]

        real_feat = feature_map[labels == 0.0]
        fake_feat = feature_map[labels == 1.0]
        loss_intra = 0
        if mask != None:
            # if have mask, we compute intra-instance loss
            mask = mask.view(mask.shape[0], -1)
            real_mask = mask[labels == 0.0]
            fake_mask = mask[labels == 1.0]

            fake_part_threshold = (fake_mask > 0.01).unsqueeze(1)
            real_part_threshold = (fake_mask == 0.0).unsqueeze(1)

            fake_part = fake_feat * fake_part_threshold
            real_part = fake_feat * real_part_threshold
            sim_positive1 = torch.bmm(real_feat.permute(0, 2, 1), real_feat).view(real_feat.shape[0],
                                                                                  -1)  # for real image
            sim_positive2 = torch.bmm(real_part.permute(0, 2, 1), real_part).view(real_part.shape[0],
                                                                                  -1)  # for real part of fake image
            sim_negative = torch.bmm(fake_part.permute(0, 2, 1), real_part).view(fake_part.shape[0],
                                                                                 -1)  # for fake part of fake image

            sim_positive1 /= self.T
            l_real_sim1 = sim_positive1
            sim_positive2 /= self.T
            l_real_sim2 = sim_positive2
            sim_negative /= self.T
            l_negative_sim = sim_negative

            l_real_sim1 = torch.sum(torch.exp(l_real_sim1), dim=1)
            l_real_sim2 = torch.sum(torch.exp(l_real_sim2), dim=1)
            l_negative_sim = torch.sum(torch.exp(l_negative_sim), dim=1)
            loss_real_intra = -torch.log(torch.sum(l_real_sim1))
            loss_fake_intra = -torch.log((torch.sum(l_real_sim2)) / 
                                         (torch.sum(l_negative_sim) + torch.sum(l_real_sim2)))
            loss_intra = loss_fake_intra + 0.01 * loss_real_intra
        # calculate prototype for hard sample selection
        self.real_queue = self.alpha*self.real_queue + \
            (1-self.alpha)*torch.mean(real_k, dim=0)
        self.ancor_queue = self.alpha*self.ancor_queue + \
            (1-self.alpha)*torch.mean(fake_k, dim=0)
        l_real_pos1 = torch.einsum('nc,nc->n', [real_q, real_k]).unsqueeze(-1)
        l_real_pos = l_real_pos1

        l_fake_pos1 = torch.einsum('nc,nc->n', [fake_q, fake_k]).unsqueeze(-1)
        l_fake_pos = l_fake_pos1

        # Cross-contrast to select hard sample
        l_fake_real = torch.einsum('nc,ck->nk', [fake_k, self.real_queue.unsqueeze(-1).clone().detach()])
        l_real_fake = torch.einsum('nc,ck->nk', [real_k, self.ancor_queue.unsqueeze(-1).clone().detach()])

        l_fake_real = torch.mean(l_fake_real, dim=1)
        l_real_fake = torch.mean(l_real_fake, dim=1)
        _, fake_index = torch.sort(l_fake_real, dim=0, descending=True)
        _, real_index = torch.sort(l_real_fake, dim=0, descending=True)

        fake_k_hard = fake_k[fake_index[:self.threshold]].squeeze()
        fake_k_easy = fake_k[fake_index[-self.threshold:]].squeeze()

        real_k_hard = real_k[real_index[:self.threshold]].squeeze()
        real_k_easy = real_k[real_index[-self.threshold:]].squeeze()
        # hard sample enqueue
        if len(real_k_hard.shape) == 2:
            self._dequeue_and_enqueue_hard(real_k_hard, 'real')
        if len(fake_k_hard.shape) == 2:
            self._dequeue_and_enqueue_hard(fake_k_hard, 'fake')
        # inter-instance loss
        l_real_neg1 = torch.einsum('nc,ck->nk', [real_q, self.hard_fake_queue.clone().detach()])
        l_real_neg = l_real_neg1
        l_fake_neg1 = torch.einsum('nc,ck->nk', [fake_q, self.hard_real_queue.clone().detach()])
        l_fake_neg = l_fake_neg1

        l_neg = torch.cat([l_real_neg, l_fake_neg], dim=0)
        l_neg /= self.T
        l_pos = torch.cat([l_real_pos, l_fake_pos], dim=0)
        l_pos /= self.T
        logits1 = torch.cat([l_pos, l_neg], dim=1)
        labels = torch.zeros(logits1.shape[0], dtype=torch.long).cuda()

        loss_inter = self.criterion(logits1, labels)
        loss = loss_inter + 0.1 * loss_intra

        return output_q, loss


# utils
@torch.no_grad()
def concat_all_gather(tensor):

    tensors_gather = [torch.ones_like(tensor) for _ in range(torch.distributed.get_world_size())]
    torch.distributed.all_gather(tensors_gather, tensor, async_op=False)

    output = torch.cat(tensors_gather, dim=0)
    return output
