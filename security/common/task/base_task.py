import os
import sys
import importlib
import wandb
import loguru

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as NativeDDP

from timm.utils import CheckpointSaver
from timm.models import resume_checkpoint

sys.path.append(os.path.join(os.path.abspath(os.path.dirname(__file__)), '..', '..'))

from common import losses, optimizers, schedulers
from common.utils import *


class BaseTask(object):

    def __init__(self, cfg, task_dir, logger=None):
        self.cfg = cfg
        self.task_dir = task_dir
        self.device = torch.device("cuda", self.cfg.local_rank)

        if logger is None:
            self.logger = loguru.logger

    def prepare(self):
        self._init_env()
        self._build_inputs()
        self._build_model()
        self._distribute_model()
        self._build_optimizer()
        self._build_schedulers()
        self._build_losses()
        self._resume_model()
        self._build_saver()

    def _init_env(self, **kwargs):
        """ Init distribution env
        """
        if self.cfg.local_rank == 0:
            self.logger.info('=> Init environment')
            self.cfg = init_wandb_workspace(self.cfg)
            self.logger.add(f'{self.cfg.exam_dir}/train.log', level="INFO")

        if self.cfg.distributed:
            self.cfg.local_rank = int(os.environ.get('LOCAL_RANK', 0))
            dist.init_process_group(backend='nccl', init_method="env://")
            torch.cuda.set_device(self.cfg.local_rank)
            self.world_size = dist.get_world_size()

        setup(self.cfg)

    def _build_inputs(self, **kwargs):
        """ build dataloaders
        """
        if self.cfg.local_rank == 0:
            self.logger.info('=> Building dataloader')
        datasets = importlib.import_module('datasets', package=self.task_dir)
        self.train_dataloader = datasets.create_dataloader(self.cfg, split='train')
        self.val_dataloader = datasets.create_dataloader(self.cfg, split='val')
        self.test_dataloader = datasets.create_dataloader(self.cfg, split='test')
        self.epoch_size = len(self.train_dataloader.dataset) // (self.cfg.train.batch_size * self.cfg.world_size)

    def _build_model(self, **kwargs):
        """ build training backbone and heads
        """
        if self.cfg.local_rank == 0:
            self.logger.info('=> Building model')
        models = importlib.import_module('models', package=self.task_dir)
        self.model = models.__dict__[self.cfg.model.name](**self.cfg.model.params)
        self.model = self.model.to(self.device)

    def _distribute_model(self, **kwargs):
        """setup distributed training
        """
        if self.cfg.distributed:
            if self.cfg.local_rank == 0:
                self.logger.info("Using native Torch DistributedDataParallel.")
            self.model = NativeDDP(self.model, device_ids=[self.cfg.local_rank], find_unused_parameters=True, **kwargs)

    def _build_optimizer(self, **kwargs):
        """ build optimizers
        """
        self.optimizer = optimizers.__dict__[self.cfg.optimizer.name](self.model.parameters(),
                                                                      **self.cfg.optimizer.params)

    def _build_schedulers(self, **kwargs):
        self.scheduler = schedulers.__dict__[self.cfg.scheduler.name](self.optimizer, **self.cfg.scheduler.params)

    def _build_losses(self, **kwargs):
        """ build losses
        """
        self.criterion = losses.__dict__[self.cfg.loss.name](**self.cfg.loss.params).to(self.device)

    def _resume_model(self, **kwargs):
        """ load pretrain model ckpt if training mode is finetuning
        """
        self.start_epoch = 1
        if self.cfg.model.resume is not None:
            self.start_epoch = resume_checkpoint(self.model, self.cfg.model.resume, self.optimizer)
            if self.cfg.local_rank == 0:
                self.logger.info(f'resume model from {self.cfg.model.resume}')

    def _build_saver(self, **kwargs):
        """ build checkpoint saver
        """
        self.saver = None
        if self.cfg.local_rank == 0:
            self.logger.info('=> Building checkpoint saver')
            checkpoint_dir = os.path.join(self.cfg.exam_dir, 'ckpts')
            self.saver = CheckpointSaver(self.model,
                                         self.optimizer,
                                         args=self.cfg,
                                         checkpoint_dir=checkpoint_dir,
                                         max_history=5)

    def _model_forward(self, data, model):
        """ Implemented by sub class, model forward
        """
        output = model(data)
        if type(output) is tuple or type(output) is list:
            output = output[0]
        probs = 1 - torch.softmax(output, dim=1)[:, 0]
        return probs, output

    def _evaluate(self, y_preds, y_trues, threshold=0.5, to_numpy=True):
        y_preds = gather_tensor(y_preds, dist_=self.cfg.distributed, to_numpy=to_numpy)
        y_trues = gather_tensor(y_trues, dist_=self.cfg.distributed, to_numpy=to_numpy)
        metrics = cal_metrics(y_trues, y_preds, threshold=threshold)
        return metrics

    def _save_checkpoints(self, metrics, epoch, monitor_metric='ACC', **kwargs):
        """save checkpoints in validation

        Args:
            metrics (Dict): metrics
            epoch (int): current epoch
            monitor_metric (str, optional): validation monitor metric. Defaults to 'ACC'.
        """
        metric = metrics[monitor_metric]
        best_metric, best_epoch = self.saver.save_checkpoint(epoch, metric=metric)
        last_lr = [group['lr'] for group in self.scheduler.optimizer.param_groups][0]
        self.logger.info(f'Best_{monitor_metric}: {100 * best_metric:.4f} (Epoch-{best_epoch})')
        self.logger.info(f'learning rate: {last_lr}')

    def _log_data(self, data, epoch=0, prefix='val', **kwargs):
        """ display data and running status
        """
        results = {f'{prefix}_{k}': v for k, v in data.items()}

        for k, v in results.items():
            if 'loss' not in k:
                v *= 100
            self.logger.info(f'{k}: {v:.4f}')

        wandb.log(results, step=epoch)

    def fit(self):
        if self.cfg.local_rank == 0:
            self.logger.info('=> Starting training')

        for epoch in range(self.start_epoch, self.cfg.train.epochs + 1):
            if self.cfg.distributed and self.cfg.debug == False:
                self.train_dataloader.sampler.set_epoch(epoch)
            self.train(epoch)
            self.validate(epoch)

        if self.cfg.local_rank == 0:
            self.test()
            self.logger.info(self.cfg)
            wandb.finish()

    def train(self, epoch, **kwargs):
        raise NotImplementedError()

    def validate(self, epoch, **kwargs):
        raise NotImplementedError()

    def test(self, **kwargs):
        raise NotImplementedError()
