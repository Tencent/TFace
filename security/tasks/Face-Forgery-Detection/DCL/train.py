"""Train Task of Dual Constrative Learning (DCL)
Reference: Dual Constrative Learning for Face Forgery Detection, AAAI2022
"""
import os
import sys
import time
import wandb
import copy
import importlib
import torch
from tqdm import tqdm

sys.path.append(os.path.join(os.path.abspath(os.path.dirname(__file__)), '..', '..', '..'))

from common.task import BaseTask
from common.utils import *


class TrainTask(BaseTask):
    """ TrainTask in dcl mode, which means classifier shards into multi workers
    """

    def __init__(self, cfg_file, task_dir):
        super(TrainTask, self).__init__(cfg_file, task_dir)

    def _build_model(self, **kwargs):
        """ build training backbone and heads
        """
        if self.cfg.local_rank == 0:
            self.logger.info('=> Building model')
        models = importlib.import_module('models', package=self.task_dir)
        base_model = models.__dict__[self.cfg.base_model.name](**self.cfg.base_model.params)
        self.model = models.__dict__[self.cfg.model.name](base_model, **self.cfg.model.params)
        self.model = self.model.to(self.device)

    def _build_inputs(self, **kwargs):
        """ build dataloaders
        """
        if self.cfg.local_rank == 0:
            self.logger.info('=> Building dataloader')
        datasets = importlib.import_module('datasets', package=self.task_dir)
        self.train_dataloader = datasets.create_dataloader(self.cfg, split='train')
        cfg_celebdf = copy.deepcopy(self.cfg)
        cfg_celebdf.dataset['name'] = 'CelebDF'
        self.val_dataloader = datasets.create_dataloader(cfg_celebdf, split='val')
        self.test_dataloader = datasets.create_dataloader(cfg_celebdf, split='test')
        self.epoch_size = len(self.train_dataloader.dataset) // (self.cfg.train.batch_size * self.cfg.world_size)

    def train(self, epoch):
        # set monitoring indicators during training
        batch_time = AverageMeter('Time', ':6.5f')
        data_time = AverageMeter('Data', ':6.5f')
        acces = AverageMeter('Acc', ':.5f')
        losses = AverageMeter('Loss', ':.5f')
        progress = ProgressMeter(self.epoch_size, [acces, losses], prefix=f"Epoch:{epoch} ")

        # not use moco loss in warmup epochs
        alpha = 1 if epoch < self.cfg.train.warmup else 0.5

        self.model.train()
        end = time.time()
        for i, datas in enumerate(self.train_dataloader):
            # measure data loading time
            data_time.update(time.time() - end)

            # get input data from dataloader
            image1 = datas[0][0].to(self.device)
            image2 = datas[0][1].to(self.device)
            targets = datas[1].to(self.device)

            # forward
            if self.cfg.dataset['FaceForensics'].has_mask == True:
                mask = datas[3].float().to(self.device)
                output, loss_moco = self.model(im_q=image1, mask=mask, im_k=image2, labels=targets)
            else:
                output, loss_moco = self.model(im_q=image1, im_k=image2, labels=targets)

            probs = 1 - torch.softmax(output, dim=1)[:, 0]
            loss_ce = self.criterion(output, targets)
            loss = alpha * loss_ce + (1 - alpha) * loss_moco

            # backward
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            # compute accuracy metrics
            prediction = (probs > 0.5).int()
            targets[targets > 1] = 1
            acc = (prediction == targets).float().mean()

            # update statistical meters
            if self.cfg.distributed:
                acces.update(reduce_tensor(acc.data).item(), targets.size(0))
                losses.update(reduce_tensor(loss.data).item(), targets.size(0))
            else:
                acces.update(acc.item(), targets.size(0))
                losses.update(loss.item(), targets.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            # log training metrics at a certain frequency
            if self.cfg.local_rank == 0:
                if i % self.cfg.train.print_interval == 0:
                    self.logger.info(progress.display(i))

        if self.cfg.local_rank == 0:
            results = {
                'train_acc': acces.avg,
                'train_loss': losses.avg,
            }
            wandb.log(results, step=epoch)

    def validate(self, epoch):
        losses = AverageMeter('Loss', ':.5f')

        y_preds = []
        y_trues = []
        self.model.eval()
        for i, datas in enumerate(tqdm(self.val_dataloader)):
            with torch.no_grad():
                images = datas[0].to(self.device)
                targets = datas[1].long().to(self.device)

                probs, output = self._model_forward(images, self.model)

                loss = self.criterion(output, targets)
                if self.cfg.distributed:
                    losses.update(reduce_tensor(loss.data).item(), targets.size(0))
                else:
                    losses.update(loss.item(), targets.size(0))

                targets[targets > 1] = 1
                y_preds.extend(probs)
                y_trues.extend(targets)

        self.scheduler.step(epoch)
        metrics = self._evaluate(y_preds, y_trues, threshold=0.5)
        
        if self.cfg.local_rank == 0:
            self._save_checkpoints(metrics, epoch, monitor_metric='ACC')
            metrics.loss = losses.avg
            self._log_data(metrics, epoch, prefix='val')

    def test(self):
        # load checkpoint from model with the best validation metric
        ckpt_path = f'{self.cfg.exam_dir}/ckpts/model_best.pth.tar'
        checkpoint = torch.load(ckpt_path)

        try:
            state_dict = {'module.' + k: w for k, w in checkpoint['state_dict'].items()}
            self.model.load_state_dict(state_dict)
        except Exception:
            self.model.load_state_dict(checkpoint['state_dict'])

        if self.cfg.local_rank == 0:
            self.logger.info(f'resume model from {ckpt_path}')

        self.model.eval()

        y_preds = []
        y_trues = []
        for i, datas in enumerate(tqdm(self.test_dataloader)):
            with torch.no_grad():
                # get input data from dataloader
                images = datas[0].to(self.device)
                targets = datas[1].to(self.device)

                # model forward
                probs, _ = self._model_forward(images, self.model)

                y_preds.extend(probs)
                y_trues.extend(targets)

        metrics = self._evaluate(y_preds, y_trues, threshold=0.5)

        if self.cfg.local_rank == 0:
            self._log_data(metrics, prefix='test')


def main():
    args = get_parameters()
    task_dir = os.path.dirname(os.path.abspath(__file__))
    task = TrainTask(args, task_dir)
    task.prepare()
    task.fit()


if __name__ == '__main__':
    main()
