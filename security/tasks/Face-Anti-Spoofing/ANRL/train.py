"""Train Task of Adaptive Normalized Representation Learning (ANRL)
Reference: Adaptive Normalized Representation Learning for Generalizable Face Anti-Spoofing, ACM MM2021
"""
import os
import sys
import wandb
import importlib
import random
from collections import OrderedDict

import torch
import torch.nn.functional as F

sys.path.append(os.path.join(os.path.abspath(os.path.dirname(__file__)), '..', '..', '..'))

from common.task import BaseTask
from common.task.fas import test_module
from common.utils import *
from common import losses


class ANRLTask(BaseTask):

    def __init__(self, args, task_dir):
        super(ANRLTask, self).__init__(args, task_dir)

    def _build_inputs(self, **kwargs):
        """ build dataloaders
        """
        if self.cfg.local_rank == 0:
            self.logger.info('=> Building dataloader')
        datasets = importlib.import_module('datasets', package=self.task_dir)

        self.dg_train_pos = datasets.create_dataloader(self.cfg, split='train', category='pos')
        self.dg_train_neg = datasets.create_dataloader(self.cfg, split='train', category='neg')
        self.dg_val_pos = datasets.create_dataloader(self.cfg, split='val', category='pos')
        self.dg_val_neg = datasets.create_dataloader(self.cfg, split='val', category='neg')
        self.dg_test_pos = datasets.create_dataloader(self.cfg, split='test', category='pos')
        self.dg_test_neg = datasets.create_dataloader(self.cfg, split='test', category='neg')

        length = min(len(self.dg_train_pos.dataset), len(self.dg_train_neg.dataset))
        self.epoch_size = length // (self.cfg.train.batch_size * self.cfg.world_size)

    def _distribute_model(self):
        """setup distributed training
        """
        super(ANRLTask, self)._distribute_model(broadcast_buffers=False)

    def _build_losses(self, **kwargs):
        self.criterion = losses.__dict__[self.cfg.loss.name](**self.cfg.loss.params).to(self.device)
        self.criterion_2 = losses.__dict__[self.cfg.loss_2.name](**self.cfg.loss_2.params).to(self.device)
        self.criterion_3 = losses.__dict__[self.cfg.loss_3.name](**self.cfg.loss_3.params).to(self.device)

    def fit(self):
        """ control training and validating process
        """
        if self.cfg.local_rank == 0:
            self.logger.info('=> Starting training')
        for epoch in range(self.start_epoch, self.cfg.train.epochs + 1):
            if self.cfg.distributed and self.cfg.debug == False:
                self.dg_train_pos.sampler.set_epoch(epoch)
                self.dg_train_neg.sampler.set_epoch(epoch)

            self.train(epoch)
            self.validate(epoch)
            self.scheduler.step(epoch)

        if self.cfg.local_rank == 0:
            self.cfg.distributed = False
            self.test()
            wandb.finish()
            self.logger.info(self.cfg)

    def mix_pos_neg(self, datas_pos, datas_neg, meta_train_list, meta_test_list):
        B, N, C, H, W = datas_pos[0].shape
        # initial the meta_train_img, meta_test_img
        meta_train_pos_img = torch.ones((B, 0, C, H, W))
        meta_train_neg_img = torch.ones((B, 0, C, H, W))
        meta_test_pos_img = torch.ones((B, 0, C, H, W))
        meta_test_neg_img = torch.ones((B, 0, C, H, W))

        # initial the meta_train_label, meta_test_label
        meta_train_pos_label = torch.ones((B, 0))
        meta_train_neg_label = torch.ones((B, 0))
        meta_test_pos_label = torch.ones((B, 0))
        meta_test_neg_label = torch.ones((B, 0))

        dep_B, dep_N, dep_H, dep_W = datas_pos[2].shape

        # initial the meta_train_depth, meta_test_depth
        meta_train_pos_depth = torch.ones((dep_B, 0, dep_H, dep_W))
        meta_train_neg_depth = torch.ones((dep_B, 0, dep_H, dep_W))
        meta_test_pos_depth = torch.ones((dep_B, 0, dep_H, dep_W))
        meta_test_neg_depth = torch.ones((dep_B, 0, dep_H, dep_W))

        # assign the images into meta_train and meta_test according to the index
        for index in meta_train_list:
            meta_train_pos_img = torch.cat((meta_train_pos_img, datas_pos[0][:, index:index + 1]), 1)
            meta_train_neg_img = torch.cat((meta_train_neg_img, datas_neg[0][:, index:index + 1]), 1)

            meta_train_pos_label = torch.cat((meta_train_pos_label, datas_pos[1][:, index:index + 1]), 1)
            meta_train_neg_label = torch.cat((meta_train_neg_label, datas_neg[1][:, index:index + 1]), 1)

            meta_train_pos_depth = torch.cat((meta_train_pos_depth, datas_pos[2][:, index:index + 1]), 1)
            meta_train_neg_depth = torch.cat((meta_train_neg_depth, datas_neg[2][:, index:index + 1]), 1)

        for index in meta_test_list:
            meta_test_pos_img = torch.cat((meta_test_pos_img, datas_pos[0][:, index:index + 1]), 1)
            meta_test_neg_img = torch.cat((meta_test_neg_img, datas_neg[0][:, index:index + 1]), 1)

            meta_test_pos_label = torch.cat((meta_test_pos_label, datas_pos[1][:, index:index + 1]), 1)
            meta_test_neg_label = torch.cat((meta_test_neg_label, datas_neg[1][:, index:index + 1]), 1)

            meta_test_pos_depth = torch.cat((meta_test_pos_depth, datas_pos[2][:, index:index + 1]), 1)
            meta_test_neg_depth = torch.cat((meta_test_neg_depth, datas_neg[2][:, index:index + 1]), 1)

        # reshape the img from [B,N,C,H,W] to [B*N,C,H,W]
        # reshape the label from [B,N] to [B*N]
        _, _, C, H, W = meta_train_pos_img.shape
        meta_train_pos_img = meta_train_pos_img.reshape(-1, C, H, W)
        meta_train_pos_label = meta_train_pos_label.reshape(-1).long()

        meta_train_neg_img = meta_train_neg_img.reshape(-1, C, H, W)
        meta_train_neg_label = meta_train_neg_label.reshape(-1).long()

        meta_test_pos_img = meta_test_pos_img.reshape(-1, C, H, W)
        meta_test_pos_label = meta_test_pos_label.reshape(-1).long()

        meta_test_neg_img = meta_test_neg_img.reshape(-1, C, H, W)
        meta_test_neg_label = meta_test_neg_label.reshape(-1).long()

        meta_train_images = torch.cat((meta_train_pos_img, meta_train_neg_img), 0)
        meta_train_targets = torch.cat((meta_train_pos_label, meta_train_neg_label), 0)

        meta_test_images = torch.cat((meta_test_pos_img, meta_test_neg_img), 0)
        meta_test_targets = torch.cat((meta_test_pos_label, meta_test_neg_label), 0)

        # reshape the depth from [B,N,H,W] to [B*N,H,W]
        meta_train_pos_depth = meta_train_pos_depth.reshape(-1, meta_train_pos_depth.shape[2],
                                                            meta_train_pos_depth.shape[3])
        meta_train_neg_depth = meta_train_neg_depth.reshape(-1, meta_train_neg_depth.shape[2],
                                                            meta_train_neg_depth.shape[3])
        meta_train_depth = torch.cat((meta_train_pos_depth, meta_train_neg_depth), 0)

        meta_test_pos_depth = meta_test_pos_depth.reshape(-1, meta_test_pos_depth.shape[2],
                                                          meta_test_pos_depth.shape[3])
        meta_test_neg_depth = meta_test_neg_depth.reshape(-1, meta_test_neg_depth.shape[2],
                                                          meta_test_neg_depth.shape[3])
        meta_test_depth = torch.cat((meta_test_pos_depth, meta_test_neg_depth), 0)

        return meta_train_images, meta_train_targets, meta_train_depth, \
               meta_test_images, meta_test_targets, meta_test_depth

    def _model_forward(self, image, param=None):
        output, depth, feat = self.model(image, param)
        prediction = output.argmax(dim=1)
        if self.model.training:
            return prediction, output, depth, feat
        else:
            probs = torch.softmax(output, dim=1)[:, 1]
            return probs

    def compute_losses(self, output, depth, feat, domain_meta_targets, domain_meta_depth, center_real, center_fake,
                       num_real, index):
        loss_1 = self.criterion(output, domain_meta_targets)
        loss_2 = self.criterion_2(depth.squeeze(), domain_meta_depth)

        feat_real = feat[:num_real]
        feat_pool_real = F.adaptive_avg_pool2d(feat_real, 1).reshape((feat_real.shape[0], -1))
        center_real[index] = center_real[index] * 0.9 + feat_pool_real.mean(dim=0).detach() * 0.1

        feat_fake = feat[num_real:]
        feat_pool_fake = F.adaptive_avg_pool2d(feat_fake, 1).reshape((feat_fake.shape[0], -1))
        center_fake[index] = center_fake[index] * 0.9 + feat_pool_fake.mean(dim=0).detach() * 0.1

        # calculate the domain loss and discrimination loss
        AA_loss = self.criterion_3(feat_pool_real, center_real[index])
        AB_loss = 0
        for k in range(3):
            if not k == index:
                AB_loss += self.criterion_3(feat_pool_real, center_real[k])

        loss_domain = AB_loss / 2 - AA_loss

        RR_loss = self.criterion_3(feat_pool_real, center_real[index])
        RF_loss = self.criterion_3(feat_pool_real, center_fake[index])
        FR_loss = self.criterion_3(feat_pool_fake, center_real[index])

        loss_discri = RF_loss + FR_loss - RR_loss

        return loss_1, loss_2, loss_domain, loss_discri

    def train(self, epoch):
        # define the metric recoders
        train_acces = AverageMeter('Meta_train_Acc', ':.5f')
        train_losses = AverageMeter('Meta_train_Class', ':.5f')
        train_losses2 = AverageMeter('Meta_train_Depth', ':.5f')
        train_losses3 = AverageMeter('Meta_train_Domain', ':.5f')
        train_losses4 = AverageMeter('Meta_train_Discri', ':.5f')

        test_acces = AverageMeter('Meta_test_Acc', ':.5f')
        test_losses = AverageMeter('Meta_test_Class', ':.5f')
        test_losses2 = AverageMeter('Meta_test_Depth', ':.5f')
        test_losses3 = AverageMeter('Meta_test_Domain', ':.5f')
        test_losses4 = AverageMeter('Meta_test_Discri', ':.5f')

        # aggregate all the recoders
        progress = ProgressMeter(self.epoch_size, [
            train_acces, train_losses, train_losses2, train_losses3, train_losses4, 
            test_acces, test_losses, test_losses2, test_losses3, test_losses4
        ], prefix=f"Epoch:{epoch} ")

        self.model.train()

        # to initialize the centor
        center_real = torch.ones(3, 384).to(self.device)
        center_fake = torch.ones(3, 384).to(self.device)

        for i, (datas_pos, datas_neg) in enumerate(zip(self.dg_train_pos, self.dg_train_neg)):
            # generate the meta_train and meta_test index
            domain_list = list(range(datas_pos[0].shape[1]))
            random.shuffle(domain_list)

            meta_train_list = domain_list[:self.cfg.train.metasize]
            meta_test_list = domain_list[self.cfg.train.metasize:]

            # ============ meta-train ============= #
            # split meta-train data and meta-test data
            meta_train_images, \
            meta_train_targets, \
            meta_train_depth, \
            meta_test_images, \
            meta_test_targets, \
            meta_test_depth = self.mix_pos_neg(datas_pos, datas_neg, meta_train_list, meta_test_list)

            loss_1 = 0
            loss_2 = 0
            loss_domain = 0
            loss_discri = 0

            batch = int(meta_train_images.shape[0] / 2)
            length = int(batch / len(meta_train_list))

            # compute center
            for j in range(len(meta_train_list)):
                domain_select = list(range(j*length,(j*length+length)))+ \
                                list(range(j*length+batch,(j*length+length+batch)))
                domain_meta_train_images = meta_train_images[domain_select].to(self.device)
                domain_meta_train_targets = meta_train_targets[domain_select].to(self.device)
                domain_meta_train_depth = meta_train_depth[domain_select].to(self.device)

                prediction, output, depth, feat = self._model_forward(domain_meta_train_images)

                # generate the results
                index = meta_train_list[j]
                num_real = int(feat.shape[0] / 2)
                sub_loss_1, sub_loss_2, sub_loss_domain, sub_loss_discri = self.compute_losses(
                    output, depth, feat, domain_meta_train_targets, domain_meta_train_depth, center_real, center_fake,
                    num_real, index)
                loss_1 += sub_loss_1
                loss_2 += sub_loss_2
                loss_domain += sub_loss_domain
                loss_discri += sub_loss_discri

            # all the loss
            train_loss = loss_1 + \
                        loss_2 * self.cfg.loss_2.weight + \
                        loss_domain * self.cfg.loss_3.domain_weight + \
                        loss_discri * self.cfg.loss_3.discri_weight

            # the accuracy of current batch
            acc = (prediction == domain_meta_train_targets).float().mean()

            # update the metrics
            if self.cfg.distributed:
                train_acces.update(reduce_tensor(acc.data).item(), meta_train_targets.size(0))
                train_losses.update(reduce_tensor(loss_1.data).item(), meta_train_targets.size(0))
                train_losses2.update(reduce_tensor(loss_2.data).item(), meta_train_targets.size(0))
                train_losses3.update(reduce_tensor(loss_domain.data).item(), meta_train_targets.size(0))
                train_losses4.update(reduce_tensor(loss_discri.data).item(), meta_train_targets.size(0))
            else:
                train_acces.update(acc.item(), meta_train_targets.size(0))
                train_losses.update(loss_1.item(), meta_train_targets.size(0))
                train_losses2.update(loss_2.item(), meta_train_targets.size(0))
                train_losses3.update(loss_domain.item(), meta_train_targets.size(0))
                train_losses4.update(loss_discri.item(), meta_train_targets.size(0))

            # ============ update the meta parameter ============= #
            attention_parameters_extor = []
            # store the original parameters which need updated via meta-learning
            for k, v in self.model.module.FeatExtractor.named_parameters():
                if 'AttentionNet' in k:
                    if v.grad is not None:
                        v.grad.zero_()
                    attention_parameters_extor.append(v)

            attention_parameters = attention_parameters_extor

            # calculate the gradients
            grads_AttentionNet = torch.autograd.grad(train_loss,
                                                     attention_parameters,
                                                     create_graph=True,
                                                     allow_unused=True)

            fast_weights_AttentionNet_extor = {}
            for k, v in self.model.module.FeatExtractor.state_dict().items():
                if 'AttentionNet' in k:
                    fast_weights_AttentionNet_extor[k] = v

            # update the parameters on the stored paramters
            adapted_params = OrderedDict()
            for key, grad in zip(fast_weights_AttentionNet_extor, grads_AttentionNet):
                adapted_params[key] = fast_weights_AttentionNet_extor[key] - self.cfg.train.meta_step_size * grad
                fast_weights_AttentionNet_extor[key] = adapted_params[key]

            loss_1 = 0
            loss_2 = 0
            loss_domain = 0
            loss_discri = 0
            # ============ meta-test ============= #
            batch = int(meta_test_images.shape[0] / 2)
            length = int(batch / len(meta_test_list))

            for j in range(len(meta_test_list)):
                domain_select = list(range(j*length,(j*length+length)))+ \
                                list(range(j*length+batch,(j*length+length+batch)))

                domain_meta_test_images = meta_test_images[domain_select].to(self.device)
                domain_meta_test_targets = meta_test_targets[domain_select].to(self.device)
                domain_meta_test_depth = meta_test_depth[domain_select].to(self.device)

                prediction, output, depth, feat = self._model_forward(domain_meta_test_images,
                                                                      fast_weights_AttentionNet_extor)

                # generate the results
                index = meta_test_list[j]
                num_real = int(feat.shape[0] / 2)
                sub_loss_1, sub_loss_2, sub_loss_domain, sub_loss_discri = self.compute_losses(
                    output, depth, feat, domain_meta_test_targets, domain_meta_test_depth, center_real, center_fake,
                    num_real, index)
                loss_1 += sub_loss_1
                loss_2 += sub_loss_2
                loss_domain += sub_loss_domain
                loss_discri += sub_loss_discri

                acc = (prediction == domain_meta_test_targets).float().mean()

                if self.cfg.distributed:
                    test_acces.update(reduce_tensor(acc.data).item(), domain_meta_test_targets.size(0))
                    test_losses.update(reduce_tensor(loss_1.data).item(), domain_meta_test_targets.size(0))
                    test_losses2.update(reduce_tensor(loss_2.data).item(), domain_meta_test_targets.size(0))
                    test_losses3.update(reduce_tensor(loss_domain.data).item(), domain_meta_test_targets.size(0))
                    test_losses4.update(reduce_tensor(loss_discri.data).item(), domain_meta_test_targets.size(0))
                else:
                    test_acces.update(acc.item(), domain_meta_test_targets.size(0))
                    test_losses.update(loss_1.item(), domain_meta_test_targets.size(0))
                    test_losses2.update(loss_2.item(), domain_meta_test_targets.size(0))
                    test_losses3.update(loss_domain.item(), domain_meta_test_targets.size(0))
                    test_losses4.update(loss_discri.item(), domain_meta_test_targets.size(0))

            # all the loss
            test_loss = loss_1 + \
                        loss_2 * self.cfg.loss_2.weight + \
                        loss_domain * self.cfg.loss_3.domain_weight + \
                        loss_discri * self.cfg.loss_3.discri_weight

            # all the loss
            loss = train_loss + test_loss

            # update the loss
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            if self.cfg.local_rank == 0:
                if i % self.cfg.train.print_interval == 0:
                    self.logger.info(progress.display(i))

        # record the results via wandb
        if self.cfg.local_rank == 0:
            wandb.log(
                {
                    'Meta_train_Acc': train_acces.avg,
                    'Meta_train_Class': train_losses.avg,
                    'Meta_train_Depth': train_losses2.avg,
                    'Meta_train_Domain': train_losses3.avg,
                    'Meta_train_Discri': train_losses4.avg,
                    'Meta_test_Acc': test_acces.avg,
                    'Meta_test_Class': test_losses.avg,
                    'Meta_test_Depth': test_losses2.avg,
                    'Meta_test_Domain': test_losses3.avg,
                    'Meta_test_Discri': test_losses4.avg,
                },
                step=epoch)

    def validate(self, epoch):
        y_preds, y_trues = test_module(self.model, [self.dg_val_pos, self.dg_val_neg],
                                       self._model_forward, distributed=True)

        # calculate the metrics
        metrics = self._evaluate(y_preds, y_trues, threshold='auto')

        if self.cfg.local_rank == 0:
            self._save_checkpoints(metrics, epoch, monitor_metric='AUC')
            self._log_data(metrics, epoch, prefix='val')

    def test(self):
        ckpt_path = f'{self.cfg.exam_dir}/ckpts/model_best.pth.tar'
        checkpoint = torch.load(ckpt_path)

        try:
            state_dict = {'module.' + k: w for k, w in checkpoint['state_dict'].items()}
            self.model.load_state_dict(state_dict)
        except Exception:
            self.model.load_state_dict(checkpoint['state_dict'])
        self.logger.info(f'resume model from {ckpt_path}')

        y_preds, y_trues = test_module(self.model, [self.dg_test_pos, self.dg_test_neg], self._model_forward)
        metric = cal_metrics(y_trues, y_preds, threshold='auto')

        self._log_data(metric, prefix='test')

def main():
    args = get_parameters()
    task_dir = os.path.dirname(os.path.abspath(__file__))
    ANRL_task = ANRLTask(args, task_dir)
    ANRL_task.prepare()
    ANRL_task.fit()


if __name__ == '__main__':
    main()
