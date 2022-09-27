import os
import sys
import logging
import torch
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader
import torchvision.transforms as transforms

sys.path.append(os.path.join(os.path.abspath(os.path.dirname(__file__)), '..', '..'))

from torchkit.util import AverageMeter, Timer
from torchkit.util import accuracy_dist
from torchkit.util import AllGather

from torchkit.data import SingleDataset
from torchkit.backbone import get_model
from torchkit.loss import get_loss
from torchkit.task import BaseTask
from torchkit.util import CkptLoader

from dataset import create_label2index, BalancedBatchSampler
from distillation.ekd import EKD


class TrainTask(BaseTask):
    """ Knowledge distillation TrainTask
    """
    def __init__(self, cfg_file):
        super(TrainTask, self).__init__(cfg_file)

    def loop_step(self, epoch):
        """
        load_data
            |
        extract feature
            |
        optimizer step
            |
        print log and write summary
        """
        backbone, heads = self.backbone, list(self.heads.values())
        backbone.train()  # set to training mode
        for head in heads:
            head.train()
        t_backbone = self.t_backbone
        t_backbone.eval()

        am_loss = AverageMeter()
        am_kd_loss = AverageMeter()
        am_top1 = AverageMeter()
        am_top5 = AverageMeter()

        t = Timer()
        general_batch_size = self.cfg["GENERAL_BATCH_SIZE"]

        for step, samples in enumerate(self.train_loader):
            # call hook function before_train_iter
            self.call_hook("before_train_iter", step, epoch)
            backbone_opt, head_opts = self.opt['backbone'], list(self.opt['heads'].values())

            inputs = samples[0].cuda(non_blocking=True)
            labels = samples[1].cuda(non_blocking=True)

            splits = [general_batch_size, inputs.size(0) - general_batch_size]
            with torch.no_grad():
                t_features = t_backbone(inputs)
                features_gather = AllGather(t_features, self.world_size)
                features_gather = [torch.split(x, splits) for x in features_gather]

                t_features_general = torch.cat([x[0] for x in features_gather], dim=0)
                t_features_balance = torch.cat([x[1] for x in features_gather], dim=0)

            s_features = backbone(inputs)
            features_gather = AllGather(s_features, self.world_size)
            features_gather = [torch.split(x, splits) for x in features_gather]
            s_features_general = torch.cat([x[0] for x in features_gather], dim=0)
            s_features_balance = torch.cat([x[1] for x in features_gather], dim=0)

            with torch.no_grad():
                labels_gather = AllGather(labels, self.world_size)
                labels_gather = [torch.split(x, splits) for x in labels_gather]
                labels_general = torch.cat([x[0] for x in labels_gather], dim=0)
                labels_balance = torch.cat([x[1] for x in labels_gather], dim=0)

            s_features_all = torch.cat([s_features_general, s_features_balance], dim=0)
            t_features_all = torch.cat([t_features_general, t_features_balance], dim=0)
            labels_all = torch.cat([labels_general, labels_balance], dim=0)

            outputs, labels, original_outputs = heads[0](s_features_general, labels_general)
            softmax_loss = self.loss(outputs, labels)
            kd_loss = self.kd_loss(s_features_all, t_features_all, labels_all)
            # compute loss
            loss = softmax_loss + kd_loss

            prec1, prec5 = accuracy_dist(
                self.cfg,
                original_outputs.data,
                labels_general,
                self.class_shards[0],
                topk=(1, 5))

            am_loss.update(softmax_loss.data.item())
            am_kd_loss.update(kd_loss.data.item())
            am_top1.update(prec1.data.item())
            am_top5.update(prec5.data.item())

            # update summary and log_buffer
            scalars = {
                'train/softmax_loss': am_loss,
                'train/kd_loss': am_kd_loss,
                'train/top1': am_top1,
                'train/top5': am_top5,
            }
            self.update_summary({'scalars': scalars})
            log = {
                'softmax_loss': am_loss,
                'kd_loss': am_kd_loss,
                'prec@1': am_top1,
                'prec@5': am_top5,
            }
            self.update_log_buffer(log)

            # compute gradient and do SGD
            backbone_opt.zero_grad()
            for head_opt in head_opts:
                head_opt.zero_grad()

            loss.backward()
            backbone_opt.step()
            for head_opt in head_opts:
                head_opt.step()

            cost = t.get_duration()
            self.update_log_buffer({'time_cost': cost})

            # call hook function after_train_iter
            self.call_hook("after_train_iter", step, epoch)

    def make_inputs(self):
        """ buding dataset and dataloader
        """
        rgb_mean = self.cfg['RGB_MEAN']
        rgb_std = self.cfg['RGB_STD']
        transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=rgb_mean, std=rgb_std)
        ])

        assert len(self.branches) == 1
        ds_name = list(self.branches.keys())[0]
        ds = SingleDataset(self.cfg['DATA_ROOT'], self.cfg['INDEX_ROOT'], ds_name, transform)
        ds.make_dataset()

        self.class_nums = {ds.name: ds.class_num}

        general_batch_size = self.cfg["GENERAL_BATCH_SIZE"]
        balanced_batch_size = self.cfg["BALANCED_BATCH_SIZE"]

        sampler = BalancedBatchSampler(
            create_label2index(ds),
            general_batch_size,
            balanced_batch_size,
            self.world_size,
            ds.sample_num
        )
        self.train_loader = DataLoader(
            ds,
            batch_sampler=sampler,
            num_workers=self.cfg["NUM_WORKERS"],
            pin_memory=True)

    def make_model(self):
        """ building traning backbones and heads, including teacher and student
        """

        # First, build teacher backbone
        t_backbone_name = self.cfg["T_BACKBONE_NAME"]
        t_backbone_model = get_model(t_backbone_name)
        self.t_backbone = t_backbone_model(self.input_size)
        self.t_backbone.cuda()
        logging.info("Teacher {} Backbone Generated".format(t_backbone_name))

        # Second, build studentr backbone and head
        return super().make_model()

    def load_pretrain_model(self):
        """ load pretrain model ckpt if training mode is finetuning
        """
        t_backbone_resume = self.cfg.get('T_BACKBONE_RESUME', '')
        if t_backbone_resume != '':
            CkptLoader.load_backbone(self.t_backbone, t_backbone_resume, self.local_rank)

        return super().load_pretrain_model()

    def prepare(self):
        """ common prepare task for training
        """
        self.make_inputs()
        self.make_model()
        self.loss = get_loss('DistCrossEntropy').cuda()
        self.kd_loss = EKD().cuda()
        self.opt = self.get_optimizer()
        self.register_hooks()

    def train(self):
        """
        make inputs
            |
        make model
            |
        make loss function
            |
        make optimizer
            |
        register hooks
            |
        Distributed Data Parallel mode
            |
        loop_step
        """
        self.prepare()
        self.call_hook("before_run")
        self.backbone = DistributedDataParallel(self.backbone, device_ids=[self.local_rank])
        self.t_backbone = DistributedDataParallel(self.t_backbone, device_ids=[self.local_rank])
        for epoch in range(self.start_epoch, self.epoch_num):
            self.call_hook("before_train_epoch", epoch)
            self.loop_step(epoch)
            self.call_hook("after_train_epoch", epoch)
        self.call_hook("after_run")


def main():
    task_dir = os.path.dirname(os.path.abspath(__file__))
    task = TrainTask(os.path.join(task_dir, 'train.yaml'))
    task.init_env()
    task.train()


if __name__ == '__main__':
    main()
