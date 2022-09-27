import os
import sys
import logging
import torch
import torch.cuda.amp as amp
from torch.nn.parallel import DistributedDataParallel

sys.path.append(os.path.join(os.path.abspath(os.path.dirname(__file__)), '..', '..'))

import torch.optim as optim
import torch.nn.init as init
from torchkit.util import AverageMeter, Timer
from torchkit.util import accuracy_dist
from torchkit.util import AllGather
from torchkit.hooks.learning_rate_hook import adjust_lr
from torchkit.loss import get_loss
from torchkit.task import BaseTask
from torchkit.head import get_head
from torchkit.util import get_class_split
from utils import NoisyActivation, images_to_batch, get_model

logging.basicConfig(stream=sys.stdout, level=logging.INFO, format='%(asctime)s: %(message)s')


class TrainTask(BaseTask):
    """ TrainTask in distfc mode, which means classifier shards into multi workers
    """
    def __init__(self, cfg_file):
        super(TrainTask, self).__init__(cfg_file)

    def make_model(self):
        """ build training backbone and heads
        """
        backbone_name = self.cfg['BACKBONE_NAME']
        backbone_model = get_model(backbone_name)
        self.backbone = backbone_model(self.input_size)
        self.backbone.cuda()
        logging.info("{} Backbone Generated".format(backbone_name))

        embedding_size = self.cfg['EMBEDDING_SIZE']
        self.class_shards = []
        metric = get_head(self.cfg['HEAD_NAME'], dist_fc=self.dist_fc)

        for name, branch in self.branches.items():
            class_num = self.class_nums[name]
            class_shard = get_class_split(class_num, self.world_size)
            self.class_shards.append(class_shard)
            logging.info('Split FC: {}'.format(class_shard))

            init_value = torch.FloatTensor(embedding_size, class_num)
            init.normal_(init_value, std=0.01)
            head = metric(in_features=embedding_size,
                          gpu_index=self.rank,
                          weight_init=init_value,
                          class_split=class_shard,
                          scale=branch.scale,
                          margin=branch.margin)
            del init_value
            head = head.cuda()
            self.heads[name] = head

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
        backbone, heads, noise_model = self.backbone, list(self.heads.values()), self.noise_model
        noise_model.train()
        backbone.train()  # set to training mode
        for head in heads:
            head.train()

        batch_sizes = self.batch_sizes
        am_losses = [AverageMeter() for _ in batch_sizes]
        am_top1s = [AverageMeter() for _ in batch_sizes]
        am_top5s = [AverageMeter() for _ in batch_sizes]
        t = Timer()
        for step, samples in enumerate(self.train_loader):
            # call hook function before_train_iter
            self.call_hook("before_train_iter", step, epoch)
            backbone_opt, head_opts, noise_opt = self.opt['backbone'], list(self.opt['heads'].values()), self.noise_opt

            inputs = samples[0].cuda(non_blocking=True)
            labels = samples[1].cuda(non_blocking=True)

            inputs = images_to_batch(inputs)
            inputs = inputs.detach()
            inputs = noise_model(inputs)

            if self.amp:
                with amp.autocast():
                    features = backbone(inputs)
                features = features.float()
            else:
                features = backbone(inputs)

            # gather features
            features_gather = AllGather(features, self.world_size)
            features_gather = [torch.split(x, batch_sizes) for x in features_gather]
            all_features = []
            for i in range(len(batch_sizes)):
                all_features.append(torch.cat([x[i] for x in features_gather], dim=0).cuda())

            # gather labels
            with torch.no_grad():
                labels_gather = AllGather(labels, self.world_size)
            labels_gather = [torch.split(x, batch_sizes) for x in labels_gather]
            all_labels = []
            for i in range(len(batch_sizes)):
                all_labels.append(torch.cat([x[i] for x in labels_gather], dim=0).cuda())

            losses = []
            for i in range(len(batch_sizes)):
                # PartialFC need update optimizer state in training process
                if self.pfc:
                    outputs, labels, original_outputs = heads[i](all_features[i], all_labels[i], head_opts[i])
                else:
                    outputs, labels, original_outputs = heads[i](all_features[i], all_labels[i])

                loss = self.loss(outputs, labels) * self.branch_weights[i]
                losses.append(loss)
                prec1, prec5 = accuracy_dist(self.cfg,
                                             original_outputs.data,
                                             all_labels[i],
                                             self.class_shards[i],
                                             topk=(1, 5))
                am_losses[i].update(loss.data.item(), all_features[i].size(0))
                am_top1s[i].update(prec1.data.item(), all_features[i].size(0))
                am_top5s[i].update(prec5.data.item(), all_features[i].size(0))

            # update summary and log_buffer
            scalars = {
                'train/loss': am_losses,
                'train/top1': am_top1s,
                'train/top5': am_top5s,
            }
            self.update_summary({'scalars': scalars})
            log = {
                'loss': am_losses,
                'prec@1': am_top1s,
                'prec@5': am_top5s,
            }
            self.update_log_buffer(log)

            # compute loss
            total_loss = sum(losses)
            # compute gradient and do SGD
            backbone_opt.zero_grad()
            noise_opt.zero_grad()
            for head_opt in head_opts:
                head_opt.zero_grad()

            # Automatic Mixed Precision setting
            if self.amp:
                self.scaler.scale(total_loss).backward()
                self.scaler.step(backbone_opt)
                self.scaler.step(noise_opt)
                for head_opt in head_opts:
                    self.scaler.step(head_opt)
                self.scaler.update()
            else:
                total_loss.backward()
                backbone_opt.step()
                noise_opt.step()
                for head_opt in head_opts:
                    head_opt.step()

            # PartialFC need update weight and weight_norm manually
            if self.pfc:
                for head in heads:
                    head.update()

            cost = t.get_duration()
            self.update_log_buffer({'time_cost': cost})

            # call hook function after_train_iter
            self.call_hook("after_train_iter", step, epoch)

    def get_noise_opt(self, noise_model):
        optimizer = optim.Adam(list(noise_model.module.parameters()), lr=self.cfg['LRS_NOISE'][0])
        return optimizer

    def prepare(self):
        """ common prepare task for training
        """
        self.make_inputs()
        self.make_model()
        self.loss = get_loss('DistCrossEntropy').cuda()
        self.opt = self.get_optimizer()
        self.register_hooks()
        self.pfc = self.cfg['HEAD_NAME'] == 'PartialFC'
        self.noise_model = NoisyActivation().cuda()

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
        make auto mix precision grad scalar
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
        self.noise_model = DistributedDataParallel(self.noise_model, device_ids=[self.local_rank])
        self.noise_opt = self.get_noise_opt(self.noise_model)
        for epoch in range(self.start_epoch, self.epoch_num):
            self.call_hook("before_train_epoch", epoch)
            adjust_lr(epoch, self.cfg['LRS_NOISE'], self.cfg['STAGES'], self.noise_opt)
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
