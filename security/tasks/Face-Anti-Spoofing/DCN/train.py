"""Train Task of l Destruction and Combination Network (DCN)
Reference: Structure Destruction and Content Combination for Face Anti-Spoofing, IJCB2021
"""
import os
import sys
import wandb
import importlib
import random
import numpy as np

import torch
import torch.nn.functional as F

sys.path.append(os.path.join(os.path.abspath(os.path.dirname(__file__)), '..', '..', '..'))

from common.task import BaseTask
from common.task.fas import test_module
from common.utils import *
from common import losses


class DCNTask(BaseTask):

    def __init__(self, args, task_dir):
        super(DCNTask, self).__init__(args, task_dir)

    def _build_inputs(self, **kwargs):
        """ build dataloaders
        """
        if self.cfg.local_rank == 0:
            self.logger.info('=> Building dataloader')
        datasets = importlib.import_module('datasets', package=self.task_dir)
        self.kwargs = getattr(self.cfg.dataset, self.cfg.dataset.name)

        self.train_pos = datasets.create_dataloader(self.cfg, split='train', category='pos')
        self.train_neg = datasets.create_dataloader(self.cfg, split='train', category='neg')
        self.val_pos = datasets.create_dataloader(self.cfg, split='val', category='pos')
        self.val_neg = datasets.create_dataloader(self.cfg, split='val', category='neg')
        self.test_pos = datasets.create_dataloader(self.cfg, split='test', category='pos')
        self.test_neg_1 = datasets.create_dataloader(self.cfg, split='test', category='neg_1')
        if 'test_neg_2_list_path' in self.kwargs.keys():
            self.test_neg_2 = datasets.create_dataloader(self.cfg, split='test', category='neg_2')

        length = min(len(self.train_pos.dataset), len(self.train_neg.dataset))
        self.epoch_size = length // (self.cfg.train.batch_size * self.cfg.world_size)

    def _build_losses(self, **kwargs):
        self.criterion = losses.__dict__[self.cfg.loss.name](**self.cfg.loss.params).to(self.device)
        self.criterion_2 = losses.__dict__[self.cfg.loss_2.name](**self.cfg.loss_2.params).to(self.device)

    def fit(self):
        """ control training and validating process
        """
        if self.cfg.local_rank == 0:
            self.logger.info('=> Starting training')
        for epoch in range(self.start_epoch, self.cfg.train.epochs + 1):
            if self.cfg.distributed and self.cfg.debug == False:
                self.train_pos.sampler.set_epoch(epoch)
                self.train_neg.sampler.set_epoch(epoch)

            self.train(epoch)
            self.validate(epoch)
            self.scheduler.step(epoch)

        if self.cfg.local_rank == 0:
            self.cfg.distributed = False
            self.test()
            wandb.finish()
            self.logger.info(self.cfg)

    def get_tile(self, img):
        if len(img.shape) == 3:
            w = int((img.shape[-1]) / 3)
            net_block = {}
            for i in range(3):
                for j in range(3):
                    net_block[i * 3 + j] = [i * w, (i + 1) * w, j * w, (j + 1) * w]
        else:
            w = int((img.shape[-1]) / 3)
            net_block = {}
            for i in range(3):
                for j in range(3):
                    net_block[i * 3 + j] = [i * w, (i + 1) * w, j * w, (j + 1) * w]
        return net_block

    def cutmix_withreflection(self, images_a, images_b, reflect_a, reflect_b):
        # to generate the similarity
        num = images_a.shape[0]
        sim_a_cutmix = torch.ones((num, 1, 3, 3))
        sim_b_cutmix = (torch.ones((num, 1, 3, 3)) * (-1))

        # get the locations of each patch
        net_block = self.get_tile(images_a[0])

        # initial the pathes for exchanging
        images_a_cutmix = images_a.clone().detach()
        images_b_cutmix = images_b.clone().detach()
        reflect_a_cutmix = reflect_a.clone().detach()
        reflect_b_cutmix = reflect_b.clone().detach()

        for i in range(num):
            # generate the number of patches for exchanging
            index = np.random.randint(0, 2)
            # generate the sequence of patches for exchanging
            block_sequence = np.random.permutation(range(9))
            # exchange the patch of different subdomains
            for j in range(index):
                ind = block_sequence[j]
                y_start = net_block[ind][0]
                y_end = net_block[ind][1]
                x_start = net_block[ind][2]
                x_end = net_block[ind][3]
                target = (i + 1) % num
                images_a_cutmix[i, :, y_start:y_end, x_start:x_end] = images_a[target, :, y_start:y_end, x_start:x_end]
                images_b_cutmix[i, :, y_start:y_end, x_start:x_end] = images_b[target, :, y_start:y_end, x_start:x_end]
                reflect_a_cutmix[i, y_start:y_end, x_start:x_end] = reflect_a[target, y_start:y_end, x_start:x_end]
                reflect_b_cutmix[i, y_start:y_end, x_start:x_end] = reflect_b[target, y_start:y_end, x_start:x_end]

        for i in range(num):
            # generate the number of patches for exchanging
            index = np.random.randint(0, 2)
            # generate the sequence of patches for exchanging
            block_sequence = np.random.permutation(range(9))
            # exchange the patch of different labels
            for j in range(index):
                ind = block_sequence[j]
                y_start = net_block[ind][0]
                y_end = net_block[ind][1]
                x_start = net_block[ind][2]
                x_end = net_block[ind][3]
                images_a_cutmix[i, :, y_start:y_end, x_start:x_end] = images_b[i, :, y_start:y_end, x_start:x_end]
                images_b_cutmix[i, :, y_start:y_end, x_start:x_end] = images_a[i, :, y_start:y_end, x_start:x_end]
                reflect_a_cutmix[i, y_start:y_end, x_start:x_end] = reflect_b[i, y_start:y_end, x_start:x_end]
                reflect_b_cutmix[i, y_start:y_end, x_start:x_end] = reflect_a[i, y_start:y_end, x_start:x_end]
                y = ind // 3
                x = ind % 3
                sim_a_cutmix[i, 0, y, x] = (-1)
                sim_b_cutmix[i, 0, y, x] = 1

        # calculate the Similarity Matrix
        w = F.unfold(sim_a_cutmix, kernel_size=1, stride=1, padding=0).permute(0, 2, 1)
        w_normed = w / (w * w).sum(dim=2, keepdim=True).sqrt()
        B, K = w.shape[:2]
        sim_a_cutmix = torch.einsum('bij,bjk->bik', w_normed, w_normed.permute(0, 2, 1))

        w = F.unfold(sim_b_cutmix, kernel_size=1, stride=1, padding=0).permute(0, 2, 1)
        w_normed = w / (w * w).sum(dim=2, keepdim=True).sqrt()
        B, K = w.shape[:2]
        sim_b_cutmix = torch.einsum('bij,bjk->bik', w_normed, w_normed.permute(0, 2, 1))

        B, H, W = reflect_a_cutmix.shape
        reflect_a_cutmix = F.interpolate(reflect_a_cutmix.view(B, 1, H, W), (32, 32), mode='bilinear')
        reflect_b_cutmix = F.interpolate(reflect_b_cutmix.view(B, 1, H, W), (32, 32), mode='bilinear')

        return images_a_cutmix, images_b_cutmix, reflect_a_cutmix, reflect_b_cutmix, sim_a_cutmix, sim_b_cutmix

    def mix_pos_neg(self, datas_pos, datas_neg):
        images_pos = datas_pos[0]
        targets_pos = datas_pos[1].long()
        reflect_pos = datas_pos[2]

        images_neg = datas_neg[0]
        targets_neg = datas_neg[1].long()
        reflect_neg = datas_neg[2]

        # cutmix images of different labels and subdomains
        images_pos, images_neg, \
        reflect_pos, reflect_neg, \
        sim_pos, sim_neg = self.cutmix_withreflection(images_pos,images_neg,reflect_pos,reflect_neg)

        images = torch.cat((images_pos, images_neg), 0)
        targets = torch.cat((targets_pos, targets_neg), 0)
        reflect_GT = torch.cat((reflect_pos, reflect_neg), 0)
        sim_GT = torch.cat((sim_pos, sim_neg), 0)

        # random the list
        batchid = list(range(len(images)))
        random.shuffle(batchid)
        images = images[batchid, :].detach()
        targets = targets[batchid].detach()
        reflect_GT = reflect_GT[batchid, :].detach()
        sim_GT = sim_GT[batchid, :].detach()

        return images, targets, reflect_GT, sim_GT

    def _model_forward(self, images):
        reflection, sim = self.model(images)
        if self.model.training:
            return reflection, sim
        else:
            probs = reflection.mean((1, 2, 3))
            return probs

    def train(self, epoch):
        # define the metric recoders
        train_losses = AverageMeter('train_Reflect', ':.5f')
        train_losses2 = AverageMeter('train_Sim', ':.5f')

        # aggregate all the recoders
        progress = ProgressMeter(self.epoch_size, [train_losses, train_losses2], prefix=f"Epoch:{epoch} ")

        self.model.train()

        for i, (datas_pos, datas_neg) in enumerate(zip(self.train_pos, self.train_neg)):
            # mix the pos and neg images and generate the similarity matrix
            images, labels, reflect_GT, sim_GT = self.mix_pos_neg(datas_pos, datas_neg)

            images = images.to(self.device)
            labels = labels.to(self.device)
            reflect_GT = reflect_GT.to(self.device)
            sim_GT = sim_GT.to(self.device)
            reflect, sim = self._model_forward(images)
            loss_1 = self.criterion(reflect, reflect_GT)
            loss_2 = self.criterion_2(sim, sim_GT)

            # all the loss
            loss = loss_1 + loss_2 * self.cfg.loss_2.weight

            # update the metrics
            if self.cfg.distributed:
                train_losses.update(reduce_tensor(loss_1.data).item(), images.size(0))
                train_losses2.update(reduce_tensor(loss_2.data).item(), images.size(0))
            else:
                train_losses.update(loss_1.item(), images.size(0))
                train_losses2.update(loss_2.item(), images.size(0))

            # update the loss
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            if self.cfg.local_rank == 0:
                if i % self.cfg.train.print_interval == 0:
                    self.logger.info(progress.display(i))

        # record the results via wandb
        if self.cfg.local_rank == 0:
            wandb.log({'train_Reflection': train_losses.avg, 'train_Sim': train_losses2.avg}, step=epoch)

    def validate(self, epoch):
        y_preds, y_trues = test_module(self.model, [self.val_pos, self.val_neg],
                                       self._model_forward, distributed=True)

        # calculate the metrics
        metrics = self._evaluate(y_preds, y_trues, threshold='auto')

        if self.cfg.local_rank == 0:
            self._save_checkpoints(metrics, epoch, monitor_metric='AUC')
            self._log_data(metrics, epoch, prefix='val')

    def test(self):
        ckpt_path = f'{self.cfg.exam_dir}/ckpts/model_best.pth.tar'
        checkpoint = torch.load(ckpt_path, weights_only=True)

        try:
            state_dict = {'module.' + k: w for k, w in checkpoint['state_dict'].items()}
            self.model.load_state_dict(state_dict)
        except Exception:
            self.model.load_state_dict(checkpoint['state_dict'])
        self.logger.info(f'resume model from {ckpt_path}')

        y_preds, y_trues = test_module(self.model, [self.val_pos, self.val_neg], self._model_forward)
        metric_val = cal_metrics(y_trues, y_preds, threshold='auto')

        y_preds, y_trues = test_module(self.model, [self.test_pos, self.test_neg_1], self._model_forward)
        metric = cal_metrics(y_trues, y_preds, threshold=metric_val.Thre)

        if 'test_neg_list_path_2' in self.kwargs.keys():
            y_preds, y_trues = test_module(self.model, [self.test_pos, self.test_neg_2], self._model_forward)
            metric_2 = cal_metrics(y_trues, y_preds, threshold=metric_val.Thre)
            metric = metric if metric.ACER >= metric_2.ACER else metric_2

        self._log_data(metric, prefix='test')


def main():
    args = get_parameters()
    task_dir = os.path.dirname(os.path.abspath(__file__))
    DCN_task = DCNTask(args, task_dir)
    DCN_task.prepare()
    DCN_task.fit()


if __name__ == '__main__':
    main()
