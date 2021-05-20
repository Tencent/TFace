import os
import sys
import logging
import torch
import torch.cuda.amp as amp
from torch.distributed import ReduceOp
sys.path.append(os.path.join(os.path.abspath(os.path.dirname(__file__)), '..', '..'))

from torchkit.util.utils import AverageMeter, Timer
from torchkit.util.utils import adjust_learning_rate, warm_up_lr
from torchkit.util.utils import accuracy
from torchkit.loss import get_loss
from torchkit.head import get_head
from torchkit.backbone import get_model
from torchkit.task.base_task import BaseTask

logging.basicConfig(level=logging.INFO, format='%(asctime)s: %(message)s')


class TrainTask(BaseTask):
    """ Traintask in localfc mode, which means each worker has a full classifier
    """
    def __init__(self, cfg_file):
        super(TrainTask, self).__init__(cfg_file)

    def _make_model(self, class_nums):
        backbone_name = self.cfg['BACKBONE_NAME']
        backbone_model = get_model(backbone_name)
        backbone = backbone_model([112, 112])
        logging.info("{} Backbone Generated".format(backbone_name))

        embedding_size = self.cfg['EMBEDDING_SIZE']
        heads = []
        metric = get_head(self.cfg['HEAD_NAME'], dist_fc=False)

        for class_num in class_nums:
            head = metric(in_features=embedding_size,
                          out_features=class_num)
            heads.append(head)
        backbone.cuda()
        for head in heads:
            head.cuda()
        return backbone, heads

    def _loop_step(self, train_loaders, backbone, heads, criterion, opt,
                   scaler, epoch):
        """ load_data --> extract feature --> calculate loss and apply grad --> summary
        """
        backbone.train()  # set to training mode
        for head in heads:
            head.train()

        batch_sizes = self.batch_sizes

        am_top1s = [AverageMeter() for _ in batch_sizes]
        am_top5s = [AverageMeter() for _ in batch_sizes]
        am_losses = [AverageMeter() for _ in batch_sizes]
        t = Timer()
        for batch, samples in enumerate(zip(*train_loaders)):
            global_batch = epoch * self.step_per_epoch + batch
            if global_batch <= self.warmup_step:
                warm_up_lr(global_batch, self.warmup_step, self.cfg['LR'], opt)
            if batch >= self.step_per_epoch:
                break

            labels = torch.cat([x[1] for x in samples], dim=0)
            labels = labels.cuda(non_blocking=True)
            inputs = torch.cat([x[0] for x in samples], dim=0)
            inputs = inputs.cuda(non_blocking=True)

            if self.cfg['AMP']:
                with amp.autocast():
                    features = backbone(inputs)
                features = features.float()
            else:
                features = backbone(inputs)

            # split features
            features_split = torch.split(features, batch_sizes)

            # split labels
            labels_split = torch.split(labels, batch_sizes)

            step_losses = []
            step_original_outputs = []
            for i in range(len(batch_sizes)):
                outputs, original_outputs = heads[i](features_split[i], labels_split[i])
                step_original_outputs.append(original_outputs)
                loss = criterion(outputs, labels_split[i]) * self.branch_weights[i]
                step_losses.append(loss)

            total_loss = sum(step_losses)
            # compute gradient and do SGD step
            opt.zero_grad()
            # Automatic Mixed Precision setting
            if self.cfg['AMP']:
                scaler.scale(total_loss).backward()
                scaler.step(opt)
                scaler.update()
            else:
                total_loss.backward()
                opt.step()

            for i in range(len(batch_sizes)):
                # measure accuracy and record loss
                prec1, prec5 = accuracy(step_original_outputs[i].data,
                                        labels_split[i],
                                        topk=(1, 5))
                torch.distributed.all_reduce(prec1, ReduceOp.SUM)
                torch.distributed.all_reduce(prec5, ReduceOp.SUM)
                prec1 /= self.cfg["WORLD_SIZE"]
                prec5 /= self.cfg["WORLD_SIZE"]
                avg_losses = []
                for step_loss in step_losses:
                    torch.distributed.all_reduce(step_loss, ReduceOp.SUM)
                    step_loss /= self.cfg["WORLD_SIZE"]
                    avg_losses.append(step_loss)
                am_losses[i].update(avg_losses[i].data.item(),
                                    features_split[i].size(0))
                am_top1s[i].update(prec1.data.item(), features_split[i].size(0))
                am_top5s[i].update(prec5.data.item(), features_split[i].size(0))
                # wirte loss and acc to tensorboard
                summarys = {
                    'train/loss_%d' % i: am_losses[i].val,
                    'train/top1_%d' % i: am_top1s[i].val,
                    'train/top5_%d' % i: am_top5s[i].val
                }
                self._writer_summarys(summarys, batch, epoch)

            duration = t.get_duration()
            self._log_tensor(batch, epoch, duration, am_losses, am_top1s, am_top5s)

    def train(self):
        """ make_inputs --> make_model --> load_pretrain --> build DDP -->
            build optimizer --> loop step
        """
        train_loaders, class_nums = self._make_inputs()
        backbone, heads = self._make_model(class_nums)
        self._load_pretrain_model(backbone, self.cfg['BACKBONE_RESUME'], heads, self.cfg['HEAD_RESUME'], False)
        backbone = torch.nn.parallel.DistributedDataParallel(backbone, device_ids=[self.local_rank])
        new_heads = []
        for head in heads:
            head = torch.nn.parallel.DistributedDataParallel(head, device_ids=[self.local_rank])
            new_heads.append(head)
        loss = get_loss('Softmax').cuda()
        opt = self._get_optimizer(backbone, new_heads)
        scaler = amp.GradScaler()
        self._create_writer()
        for epoch in range(self.start_epoch, self.epoch_num):
            adjust_learning_rate(opt, epoch, self.cfg)
            self._loop_step(train_loaders, backbone, new_heads, loss, opt, scaler, epoch)
            self._save_ckpt(epoch, backbone, new_heads, opt, scaler, False)


def main():
    task_dir = os.path.dirname(os.path.abspath(__file__))
    task = TrainTask(os.path.join(task_dir, 'train_config.yaml'))
    task.init_env()
    task.train()


if __name__ == '__main__':
    main()
