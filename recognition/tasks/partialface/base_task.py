import os
import sys
import copy
import logging
from collections import OrderedDict
import torch
import torch.nn as nn
import torch.nn.init as init
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.transforms as transforms

from torchkit.task.base_task import BaseTask
from torchkit.backbone import get_model
from torchkit.head import get_head
from torchkit.util import get_class_split, separate_resnet_bn_paras
from torchkit.data import MultiDataset, MultiDistributedSampler

logging.basicConfig(stream=sys.stdout, level=logging.INFO, format='%(asctime)s: %(message)s')


class AugmentedMultiDataset(MultiDataset):
    def __init__(self, data_root, index_root, names, transform, num_aug, **kwargs) -> None:
        super().__init__(data_root, index_root, names, transform, **kwargs)
        self.num_aug = num_aug

    def _build_inputs(self, world_size=None, rank=None):
        """ Read index file and saved in ``self.inputs``
            If ``self.is_shard`` is True, ``total_sample_nums`` > ``sample_nums``
        """

        for i, name in enumerate(self.names):
            index_file = os.path.join(self.index_root, name + ".txt")
            self.index_parser.reset()
            self.inputs[name] = []
            with open(index_file, 'r') as f:
                for line_i, line in enumerate(f):
                    sample = self.index_parser(line)
                    if self.is_shard is False:
                        # ============== PartialFace / MinusFace ==============
                        for i in range(self.num_aug):
                            self.inputs[name].append(sample)
                        # =====================================================
                    else:
                        if line_i % world_size == rank:
                            self.inputs[name].append(sample)
                        else:
                            pass
            self.class_nums[name] = self.index_parser.class_num + 1
            self.sample_nums[name] = len(self.inputs[name])
            if self.is_shard:
                self.total_sample_nums[name] = self.index_parser.sample_num
                logging.info("Dataset %s, class_num %d, total_sample_num %d, sample_num %d" % (
                    name, self.class_nums[name], self.total_sample_nums[name], self.sample_nums[name]))
            else:
                logging.info("Dataset %s, class_num %d, sample_num %d" % (
                    name, self.class_nums[name], self.sample_nums[name]))


class LocalBaseTask(BaseTask):
    def __init__(self, cfg_file):
        super().__init__(cfg_file=cfg_file)
        # if self.cfg['METHOD'] == 'PartialFace':
        #     self.num_aug = self.cfg['NUM_AUG']
        #     self.num_chs = self.cfg['NUM_CHS']

    def make_inputs(self):
        """ make datasets
        """
        rgb_mean = self.cfg['RGB_MEAN']
        rgb_std = self.cfg['RGB_STD']
        transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=rgb_mean, std=rgb_std)
        ])

        ds_names = list(self.branches.keys())
        # ============== PartialFace / MinusFace ==============
        ds = AugmentedMultiDataset(self.cfg['DATA_ROOT'], self.cfg['INDEX_ROOT'], ds_names,
                                   transform, self.cfg['NUM_AUG'])
        # =====================================================

        ds.make_dataset(shard=False)
        self.class_nums = ds.class_nums

        sampler = MultiDistributedSampler(ds, self.batch_sizes)
        self.train_loader = DataLoader(ds, sum(self.batch_sizes), shuffle=False,
                                       num_workers=self.cfg["NUM_WORKERS"], pin_memory=True,
                                       sampler=sampler, drop_last=False)

        self.step_per_epoch = len(self.train_loader)
        logging.info("Step_per_epoch = %d" % self.step_per_epoch)

    def make_model(self):
        """ build training backbone and heads
        """

        # ============== PartialFace / MinusFace ==============
        if self.cfg['METHOD'] == 'PartialFace':

            backbone_name = self.cfg['BACKBONE_NAME']
            backbone_model = get_model(backbone_name)
            self.backbone = backbone_model(self.input_size)

            self.backbone.input_layer = nn.Sequential(nn.Conv2d(self.cfg['NUM_CHS'] * 3, 64, (3, 3), 1, 1, bias=False),
                                                      nn.BatchNorm2d(64), nn.PReLU(64))

            logging.info("{} Backbone Generated".format(backbone_name))

        else:  # self.cfg['METHOD'] == 'MinusFace'
            from tasks.minusface.minusface import MinusBackbone

            generator, recognizer = None, None

            recognizer = get_model(self.cfg['TASK_BACKBONE'])([112, 112])
            print('Recognizer is {}'.format(self.cfg['TASK_BACKBONE']))

            if self.cfg['TASK'] == 'stage2':
                pretrain_backbone = MinusBackbone(mode='stage1',
                                                  recognizer=get_model(self.cfg['TASK_BACKBONE'])([112, 112]))
                pretrain_backbone.load_state_dict(torch.load(self.cfg['PRETRAIN_CKPT']))
                pretrain_backbone.generator.mode = self.cfg['TASK']
                generator = copy.deepcopy(pretrain_backbone.generator)
                print('Load pretrain ckpt: ', self.cfg['PRETRAIN_CKPT'])

            self.backbone = MinusBackbone(mode=self.cfg['TASK'], n_duplicate=1, generator=generator,
                                          recognizer=recognizer)
            logging.info(f"Minus {self.cfg['TASK']} Backbone Generated")
        # =====================================================

        self.backbone.cuda()

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

    def get_optimizer(self):
        """ build optimizers
        """

        learning_rates = self.cfg['LRS']
        init_lr = learning_rates[0]
        weight_decay = self.cfg['WEIGHT_DECAY']
        momentum = self.cfg['MOMENTUM']

        # ===================== MinusFace =====================
        if self.cfg['METHOD'] == 'Minusface':
            if self.cfg['TASK'] == 'stage2':
                backbone_opt = optim.SGD([{'params': self.backbone.recognizer.parameters(), 'lr': init_lr}],
                                         weight_decay=weight_decay, lr=init_lr, momentum=momentum)
            else:
                backbone_opt = optim.SGD([{'params': self.backbone.generator.parameters(), 'lr': init_lr / 10.},
                                          {'params': self.backbone.recognizer.parameters(), 'lr': init_lr}],
                                         weight_decay=weight_decay, lr=init_lr, momentum=momentum)
        else:
            backbone_paras_only_bn, backbone_paras_wo_bn = separate_resnet_bn_paras(self.backbone)
            backbone_opt = optim.SGD([
                {'params': backbone_paras_wo_bn, 'weight_decay': weight_decay},
                {'params': backbone_paras_only_bn}], lr=init_lr, momentum=momentum)
        # =====================================================

        head_opts = OrderedDict()
        for name, head in self.heads.items():
            opt = optim.SGD([{'params': head.parameters()}], lr=init_lr, momentum=momentum,
                            weight_decay=weight_decay)
            head_opts[name] = opt

        optimizer = {
            'backbone': backbone_opt,
            'heads': head_opts,
        }
        return optimizer

    def loop_step(self, epoch):
        """ Implemented by sub class, which run in every training step
        """
        raise NotImplementedError()

    def train(self):
        raise NotImplementedError()
