import os
import logging
from collections import namedtuple
from collections import OrderedDict
import torch
import torch.optim as optim
import torch.distributed as dist
import torchvision.transforms as transforms
import torch.nn.init as init
from torch.utils.tensorboard import SummaryWriter

from torchkit.backbone import get_model
from torchkit.util.utils import load_config
from torchkit.util.utils import separate_resnet_bn_paras
from torchkit.util.utils import get_class_split
from torchkit.util.utils import load_pretrain_backbone
from torchkit.util.utils import load_pretrain_head
from torchkit.head import get_head
from torchkit.data.index_tfr_dataset import IndexTFRDataset
from torchkit.data.datasets import IterationDataloader

logging.basicConfig(level=logging.INFO, format='%(asctime)s: %(message)s')


class BranchMetaInfo(namedtuple('Branch', ['batch_size', 'weight'])):
    '''A named tuple describing a brach.'''


class BaseTask(object):
    def __init__(self, cfg_file):
        """ Create a ``BaseTask`` object
            A ``BaseTask`` object will create a base class about training scheduler

            Args:
            cfg_file: config file, which defines datasets, backbone and head names, loss name 
            and some training envs
        """
        self.cfg = load_config(cfg_file)
        self.rank = 0
        self.local_rank = 0
        self.world_size = 0
        self.step_per_epoch = 0
        self.warmup_step = self.cfg['WARMUP_STEP']
        self.start_epoch = self.cfg['START_EPOCH']
        self.epoch_num = self.cfg['NUM_EPOCH']
        self.log_root = self.cfg['LOG_ROOT']
        self.model_root = self.cfg['MODEL_ROOT']
        self.input_size = self.cfg['INPUT_SIZE']
        self.writer = None
        self.branches = OrderedDict()
        for branch in self.cfg['DATASETS']:
            self.branches[branch['name']] = BranchMetaInfo(branch['batch_size'], branch['weight'])
        self.batch_sizes = [branch.batch_size for branch in self.branches.values()]
        self.branch_weights = [branch.weight for branch in self.branches.values()]

    def init_env(self):
        """ Init distribution env
        """
        seed = self.cfg['SEED']
        torch.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = True
        dist.init_process_group(backend=self.cfg['DIST_BACKEND'],
                                init_method=self.cfg["DIST_URL"])
        self.world_size = dist.get_world_size()
        self.rank = dist.get_rank()
        self.local_rank = int(os.environ.get('LOCAL_RANK', 0))
        torch.cuda.set_device(self.local_rank)
        logging.info("world_size: %s, rank: %d, local_rank: %d" %
                     (self.world_size, self.rank, self.local_rank))
        self.cfg['WORLD_SIZE'] = self.world_size
        self.cfg['RANK'] = self.rank

    def _make_inputs(self):
        """ build training input datasets
        """
        dataset_names = list(self.branches.keys())
        for name, batch_size in zip(dataset_names, self.batch_sizes):
            logging.info("branch_name: {}; batch_size: {}".format(name, batch_size))
        dataset_indexs = [os.path.join(self.cfg['INDEX_ROOT'], '%s.txt' % branch_name)
                          for branch_name in dataset_names]
        rgb_mean = self.cfg['RGB_MEAN']
        rgb_std = self.cfg['RGB_STD']
        transform = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=rgb_mean, std=rgb_std)
        ])

        train_loaders = []
        class_nums = []
        for index_file, batch_size in zip(dataset_indexs, self.batch_sizes):
            dataset = IndexTFRDataset(self.cfg['DATA_ROOT'], index_file, transform)
            sampler = torch.utils.data.distributed.DistributedSampler(
                dataset,
                num_replicas=self.world_size,
                rank=self.rank,
                shuffle=True)
            train_loader = torch.utils.data.DataLoader(
                dataset,
                batch_size=batch_size,
                shuffle=(sampler is None),
                num_workers=self.cfg['NUM_WORKERS'],
                pin_memory=True,
                sampler=sampler,
                drop_last=True)
            train_loaders.append(train_loader)
            class_nums.append(dataset.class_num)
            self.step_per_epoch = max(
                self.step_per_epoch,
                int(dataset.sample_num / (batch_size * self.world_size)))

        train_loaders = [
            IterationDataloader(train_loader, self.step_per_epoch * self.epoch_num, 0)
            for train_loader in train_loaders]

        return train_loaders, class_nums

    def _make_model(self, class_nums):
        """ build training backbone and heads
        """
        backbone_name = self.cfg['BACKBONE_NAME']
        backbone_model = get_model(backbone_name)
        backbone = backbone_model(self.input_size)
        logging.info("{} Backbone Generated".format(backbone_name))

        embedding_size = self.cfg['EMBEDDING_SIZE']
        heads = []
        class_splits = []
        metric = get_head(self.cfg['HEAD_NAME'], dist_fc=self.cfg['DIST_FC'])

        for class_num in class_nums:
            class_split = get_class_split(class_num, self.world_size)
            class_splits.append(class_split)
            logging.info('Split FC: {}'.format(class_split))
            init_value = torch.FloatTensor(embedding_size, class_num)
            init.normal_(init_value, std=0.01)
            head = metric(in_features=embedding_size,
                          gpu_index=self.rank,
                          weight_init=init_value,
                          class_split=class_split)
            del init_value
            heads.append(head)
        backbone.cuda()
        for head in heads:
            head.cuda()
        return backbone, heads, class_splits

    def _get_optimizer(self, backbone, heads):
        """ build optimizers
        """
        backbone_paras_only_bn, backbone_paras_wo_bn = separate_resnet_bn_paras(backbone)
        lr = self.cfg['LR']
        weight_decay = self.cfg['WEIGHT_DECAY']
        momentum = self.cfg['MOMENTUM']
        head_params = []
        for head in heads:
            head_params += list(head.parameters())
        optimizer = optim.SGD([
            {'params': backbone_paras_wo_bn + head_params, 'weight_decay': weight_decay},
            {'params': backbone_paras_only_bn}],
            lr=lr,
            momentum=momentum)
        return optimizer

    def _create_writer(self):
        """ build tensorboard summary writer
        """
        self.writer = SummaryWriter(self.log_root) if self.rank == 0 else None

    def _save_backbone(self, epoch, backbone):
        """ save backbone ckpt
        """
        if self.rank == 0:
            backbone_path = os.path.join(
                self.model_root,
                "Backbone_Epoch_{}_checkpoint.pth".format(epoch + 1))
            torch.save(backbone.module.state_dict(), backbone_path)

    def _save_meta(self, epoch, opt, scaler):
        """ save optimizer and scaler
        """
        if self.rank == 0:
            save_dict = {
                'EPOCH': epoch + 1,
                'OPTIMIZER': opt.state_dict(),
                "AMP_SCALER": scaler.state_dict()}
            opt_path = os.path.join(
                self.model_root,
                "Optimizer_Epoch_{}_checkpoint.pth".format(epoch + 1))
            torch.save(save_dict, opt_path)

    def _save_heads(self, epoch, heads, dist_fc=True):
        """ save heads, if dist_fc is True, the head should be splited
        """
        head_dict = {}
        head_names = list(self.branches.keys())
        assert len(heads) == len(head_names)
        for i, head in enumerate(heads):
            head_dict[head_names[i]] = head.state_dict()
        if dist_fc:
            head_path = os.path.join(
                self.model_root,
                "HEAD_Epoch_{}_Split_{}_checkpoint.pth".format(epoch + 1, self.rank))
            torch.save(head_dict, head_path)
        elif self.rank == 0:
            head_path = os.path.join(
                self.model_root,
                "HEAD_Epoch_{}_checkpoint.pth".format(epoch + 1))
            torch.save(head_dict, head_path)

    def _save_ckpt(self, epoch, backbone, heads, opt, scaler, dist_fc=True):
        """ save ckpt
        """
        logging.info("Save checkpoint at epoch %d ..." % (epoch + 1))
        self._save_backbone(epoch, backbone)
        self._save_meta(epoch, opt, scaler)
        self._save_heads(epoch, heads, dist_fc)
        logging.info("Save checkpoint done")

    def _load_pretrain_model(self, backbone, backbone_resume, heads, head_resume, dist_fc=True):
        """ load pretrain model ckpt if training mode is finetuning
        """
        load_pretrain_backbone(backbone, backbone_resume)
        head_names = list(self.branches.keys())
        load_pretrain_head(heads, head_names, head_resume, dist_fc=dist_fc, rank=self.rank)

    def _load_meta(self, opt, scaler, meta_resume):
        """ load pretrain meta ckpt if training mode is finetuning
        """
        if os.path.exists(meta_resume) and os.path.isfile(meta_resume):
            logging.info("Loading meta Checkpoint '{}'".format(meta_resume))
            meta_dict = torch.load(meta_resume)
            self.start_epoch = meta_dict['EPOCH']
            opt.load_state_dict(meta_dict['OPTIMIZER'])
            scaler.load_state_dict(meta_dict['AMP_SCALER'])
        else:
            logging.info(("No Meta Found at '{}'"
                          "Please Have a Check or Continue to Train from Scratch").format(meta_resume))

    def _log_tensor(self, batch, epoch, duration, losses, top1, top5, log_step=100):
        """ logging training info
        """
        if self.rank == 0 and (batch == 0 or ((batch + 1) % log_step == 0)):
            logging.info("Epoch {} / {}, batch {} / {}, {:.4f} sec/batch".format(
                epoch + 1, self.epoch_num, batch + 1, self.step_per_epoch,
                duration))
            log_tensors = {}
            log_tensors['loss'] = [x.val for x in losses]
            log_tensors['prec@1'] = [x.val for x in top1]
            log_tensors['prec@5'] = [x.val for x in top5]

            log_str = " " * 25
            for k, v in log_tensors.items():
                s = ', '.join(['%.6f' % x for x in v])
                log_str += '{} = [{}] '.format(k, s)
            print(log_str)

    def _writer_summarys(self, summarys, batch, epoch, log_step=100):
        if self.rank == 0 and (batch == 0 or ((batch + 1) % log_step == 0)):
            global_step = batch + self.step_per_epoch * epoch
            for k, v in summarys.items():
                self.writer.add_scalar(k, v, global_step=global_step)

    def _writer_histograms(self, histograms, batch, epoch, log_step=100):
        if self.rank == 0 and (batch == 0 or ((batch + 1) % log_step == 0)):
            global_step = batch + self.step_per_epoch * epoch
            for k, v in histograms.items():
                self.writer.add_histogram(k, v, global_step=global_step)

    def _loop_step(self, train_loaders, backbone, heads, criterion, opt,
                   scaler, epoch, class_splits):
        """ Implemented by sub class, which run in every training step
        """
        raise NotImplementedError()

    def train(self):
        raise NotImplementedError()
