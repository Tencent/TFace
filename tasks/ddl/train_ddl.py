import os
import sys
import logging
import torch
import torch.cuda.amp as amp
import torch.distributed as dist
import torchvision.transforms as transforms
sys.path.append(os.path.join(os.path.abspath(os.path.dirname(__file__)), '..', '..'))

from torchkit.util.utils import AverageMeter, Timer
from torchkit.util.utils import adjust_learning_rate, warm_up_lr
from torchkit.util.utils import accuracy_dist
from torchkit.util.distributed_functions import AllGather
from torchkit.loss import get_loss
from torchkit.data.balance_dataset import IDBalanceTFRDataset
from torchkit.data.balance_dataset import DistributedIDBalanceSampler
from torchkit.data.datasets import IterationDataloader
from torchkit.data.index_tfr_dataset import PairIndexTFRDataset
from torchkit.task.base_task import BaseTask



logging.basicConfig(level=logging.INFO, format='%(asctime)s: %(message)s')


class TrainDDLTask(BaseTask):
    """ TrainDDLTask usually consists of softmax-based loss and ddl loss
    """
    def __init__(self, cfg_file, pos_weight=0.1,
                 neg_weight=0.02, order_weight=0.5):
        super(TrainDDLTask, self).__init__(cfg_file)
        self.person_number = 1
        self.pos_weight = pos_weight
        self.neg_weight = neg_weight
        self.order_weight = order_weight
        self.is_pair_flags = []
        for branch in self.cfg['DATASETS']:
            if "IS_PAIR" not in branch:
                self.is_pair_flags.append(False)
            else:
                self.is_pair_flags.append(branch['IS_PAIR'])
        self.normal_branch_number = self.is_pair_flags.count(False)
        self.pair_branch_number = self.is_pair_flags.count(True)
        logging.info("normal_branch_number: {}".format(self.normal_branch_number))
        logging.info("pair_branch_number: {}".format(self.pair_branch_number))

    def _make_inputs(self):
        """ DDL task can process pair data branch
        """
        dataset_names = list(self.branches.keys())
        for name, batch_size in zip(dataset_names, self.batch_sizes):
            logging.info("branch_name: {}; batch_size: {}".format(name, batch_size))
        dataset_indexs = [os.path.join(self.cfg['INDEX_ROOT'],
                                       '%s.txt' % branch_name) for branch_name in dataset_names]
        rgb_mean = self.cfg['RGB_MEAN']
        rgb_std = self.cfg['RGB_STD']
        transform = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=rgb_mean, std=rgb_std)
        ])

        train_loaders = []
        class_nums = []
        for branch_index, (index_file, batch_size) in enumerate(zip(dataset_indexs, self.batch_sizes)):
            if self.is_pair_flags[branch_index]:
                dataset = PairIndexTFRDataset(self.cfg['DATA_ROOT'], index_file, transform)
                sampler = torch.utils.data.distributed.DistributedSampler(
                    dataset,
                    num_replicas=self.world_size,
                    rank=self.rank,
                    shuffle=True)
            else:
                dataset = IDBalanceTFRDataset(self.cfg['DATA_ROOT'], index_file, transform)
                sampler = DistributedIDBalanceSampler(
                    dataset,
                    self.person_number)
                class_nums.append(dataset.class_num)
            train_loader = torch.utils.data.DataLoader(
                dataset,
                batch_size=batch_size,
                shuffle=(sampler is None),
                num_workers=self.cfg['NUM_WORKERS'],
                pin_memory=True,
                sampler=sampler,
                drop_last=True)
            train_loaders.append(train_loader)

            self.step_per_epoch = max(
                self.step_per_epoch,
                int(dataset.sample_num / (batch_size * self.world_size)))

        train_loaders = [IterationDataloader(train_loader, self.step_per_epoch * self.epoch_num, 0)
                         for train_loader in train_loaders]

        return train_loaders, class_nums

    def _gather_pair_features(self, pair_features, pair_batch_sizes):
        """ Allgather pair features and split according pair batch sizes.
        """
        _features_gather = [torch.zeros_like(pair_features) for _ in range(self.world_size)]
        features_gather = AllGather(pair_features, *_features_gather)
        features_gather = [torch.split(x, pair_batch_sizes) for x in features_gather]
        all_pair_features = []
        for i in range(len(pair_batch_sizes)):
            all_pair_features.append(torch.cat([x[i] for x in features_gather], dim=0).cuda())
        return all_pair_features

    def _loop_step(self, train_loaders, backbone, heads, criterion, ddl, opt,
                   scaler, epoch, class_splits):
        """ load_data --> extract feature --> calculate loss and apply grad --> summary
        """
        backbone.train()  # set to training mode
        for head in heads:
            head.train()

        normal_batch_sizes = self.batch_sizes[:self.normal_branch_number]
        pair_batch_sizes = self.batch_sizes[self.normal_branch_number:]

        am_losses = [AverageMeter() for _ in normal_batch_sizes]
        am_top1s = [AverageMeter() for _ in normal_batch_sizes]
        am_top5s = [AverageMeter() for _ in normal_batch_sizes]
        t = Timer()
        for batch, samples in enumerate(zip(*train_loaders)):
            global_batch = epoch * self.step_per_epoch + batch
            if global_batch <= self.warmup_step:
                warm_up_lr(global_batch, self.warmup_step, self.cfg['LR'], opt)
            if batch >= self.step_per_epoch:
                break
            normal_branch_samples = samples[:self.normal_branch_number]  # single-data samples
            pair_branch_samples = samples[self.normal_branch_number:]  # pair-data samples

            normal_inputs = torch.cat([x[0] for x in normal_branch_samples], dim=0)
            normal_labels = torch.cat([x[1] for x in normal_branch_samples], dim=0)
            normal_labels = normal_labels.cuda(non_blocking=True)

            pair_inputs_first = torch.cat([x[0] for x in pair_branch_samples], dim=0)
            pair_inputs_second = torch.cat([x[1] for x in pair_branch_samples], dim=0)

            # total inputs
            total_inputs = torch.cat((normal_inputs, pair_inputs_first, pair_inputs_second), dim=0)
            total_inputs = total_inputs.cuda(non_blocking=True)

            # Automatic Mixed Precision setting
            if self.cfg['AMP']:
                with amp.autocast():
                    total_features = backbone(total_inputs)
                total_features = total_features.float()
            else:
                total_features = backbone(total_inputs)

            (features, pair_features_first, pair_features_second) = torch.split(
                total_features, [len(normal_inputs), len(pair_inputs_first), len(pair_inputs_second)])
            # gather normal features
            _features_gather = [torch.zeros_like(features) for _ in range(self.world_size)]
            features_gather = AllGather(features, *_features_gather)
            features_gather = [torch.split(x, normal_batch_sizes) for x in features_gather]
            all_normal_features = []
            for i in range(len(normal_batch_sizes)):
                all_normal_features.append(torch.cat([x[i] for x in features_gather], dim=0).cuda())

            # gather normal labels
            labels_gather = [torch.zeros_like(normal_labels) for _ in range(self.world_size)]
            dist.all_gather(labels_gather, normal_labels)
            labels_gather = [torch.split(x, normal_batch_sizes) for x in labels_gather]
            all_normal_labels = []
            for i in range(len(normal_batch_sizes)):
                all_normal_labels.append(torch.cat([x[i] for x in labels_gather], dim=0).cuda())

            # calculate softmax-based loss
            step_losses = []
            step_original_outputs = []
            for i in range(len(normal_batch_sizes)):
                outputs, part_labels, original_outputs = heads[i](all_normal_features[i], all_normal_labels[i])
                step_original_outputs.append(original_outputs)
                loss = criterion(outputs, part_labels)
                step_losses.append(loss)

            # gather pair features
            all_pair_features_first = self._gather_pair_features(pair_features_first, pair_batch_sizes)
            all_pair_features_second = self._gather_pair_features(pair_features_second, pair_batch_sizes)
            # calculate ddl loss
            ddl_loss, neg_distances, pos_distances = ddl(
                all_normal_features, all_pair_features_first, all_pair_features_second)
            total_loss = sum(step_losses) + ddl_loss
            # compute gradient and do SGD step
            opt.zero_grad()

            if self.cfg['AMP']:
                scaler.scale(total_loss).backward()
                scaler.step(opt)
                scaler.update()
            else:
                total_loss.backward()
                opt.step()

            for i in range(len(normal_batch_sizes)):
                # measure accuracy and record loss
                prec1, prec5 = accuracy_dist(self.cfg,
                                             step_original_outputs[i].data,
                                             all_normal_labels[i],
                                             class_splits[i],
                                             topk=(1, 5))

                am_losses[i].update(step_losses[i].data.item(),
                                    all_normal_features[i].size(0))
                am_top1s[i].update(prec1.data.item(), all_normal_features[i].size(0))
                am_top5s[i].update(prec5.data.item(), all_normal_features[i].size(0))
                # wirte loss and acc to tensorboard
                summarys = {
                    'train/ddl_loss': ddl_loss.item(),
                    'train/loss_%d' % i: am_losses[i].val,
                    'train/top1_%d' % i: am_top1s[i].val,
                    'train/top5_%d' % i: am_top5s[i].val
                }
                self._writer_summarys(summarys, batch, epoch)
            histograms = {}
            for index, neg in enumerate(neg_distances):
                    histograms['neg_%d' % index] = neg.detach().cpu().numpy()
            for index, pos in enumerate(pos_distances):
                    histograms['pos_%d' % index] = pos.detach().cpu().numpy()
            self._writer_histograms(histograms, batch, epoch)

            duration = t.get_duration()
            self._log_tensor(batch, epoch, duration, am_losses, am_top1s, am_top5s)

    def _prepare(self):
        """ common prepare task for training
        """
        train_loaders, class_nums = self._make_inputs()
        backbone, heads, class_splits = self._make_model(class_nums)
        self._load_pretrain_model(backbone, self.cfg['BACKBONE_RESUME'], heads, self.cfg['HEAD_RESUME'])
        backbone = torch.nn.parallel.DistributedDataParallel(backbone, device_ids=[self.local_rank])
        loss = get_loss('DistCrossEntropy').cuda()
        opt = self._get_optimizer(backbone, heads)
        scaler = amp.GradScaler()
        self._load_meta(opt, scaler, self.cfg['META_RESUME'])
        return train_loaders, backbone, heads, class_splits, loss, opt, scaler


    def train(self):
        """ make_inputs --> make_model --> load_pretrain --> build DDP -->
            build optimizer --> loop step
        """
        train_loaders, backbone, heads, class_splits, loss, opt, scaler = self._prepare()
        ddl = get_loss('DDL').cuda()
        self._create_writer()
        for epoch in range(self.start_epoch, self.epoch_num):
            adjust_learning_rate(opt, epoch, self.cfg)
            self._loop_step(train_loaders, backbone, heads, loss, ddl, opt, scaler,
                            epoch, class_splits)
            self._save_ckpt(epoch, backbone, heads, opt, scaler)


def main():
    task_dir = os.path.dirname(os.path.abspath(__file__))
    task = TrainDDLTask(os.path.join(task_dir, 'train_config.yaml'))
    task.init_env()
    task.train()


if __name__ == '__main__':
    main()
