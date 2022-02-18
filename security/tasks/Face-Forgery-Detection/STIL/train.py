import os
import sys
from omegaconf import OmegaConf
import time
import numpy as np
from tqdm import tqdm
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.distributed as dist
import torch.utils.data as data
import torch.optim as optim
import torch.nn.functional as F

from models import *
from datasets import *

sys.path.append(os.path.join(os.path.abspath(os.path.dirname(__file__)), '..', '..', '..'))

from common import losses, optimizers
from common.utils import *
from utils import *

args = get_params()
setup(args)
init_exam_dir(args)


###########################
# main logic for training #
###########################
def main():
    # use distributed training with nccl backend 
    args.local_rank = int(os.environ.get('LOCAL_RANK', 0))
    dist.init_process_group(backend='nccl', init_method="env://")
    torch.cuda.set_device(args.local_rank)
    args.world_size = dist.get_world_size()
    
    # set logger
    logger = get_logger(str(args.local_rank), console=args.local_rank==0, 
        log_path=os.path.join(args.exam_dir, f'train_{args.local_rank}.log'))

    # get dataloaders for train and test
    train_dataloader = get_dataloader(args, 'train')
    test_dataloader = get_dataloader(args, 'test')

    # set model and wrap it with DistributedDataParallel
    model = eval(args.model.name)(**args.model.params)
    model.cuda(args.local_rank)
    model = nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank])

    # set optimizer
    optimizer = optimizers.__dict__[args.optimizer.name](model.parameters(), **args.optimizer.params)
    criterion = losses.__dict__[args.loss.name](
        **(args.loss.params if getattr(args.loss, "params", None) else {})
    ).cuda(args.local_rank)
    
    global_step = 1
    start_epoch = 1
    # resume model for a given checkpoint file
    if args.model.resume:
        logger.info(f'resume from {args.model.resume}')
        checkpoint = torch.load(args.model.resume, map_location='cpu')
        if 'state_dict' in checkpoint:
            sd = checkpoint['state_dict']
            if (not getattr(args.model, 'only_resume_model', False)):
                # loading optimizer status, global step and epoch 
                if 'optimizer' in checkpoint:
                    optimizer.load_state_dict(checkpoint['optimizer'])
                if 'global_step' in checkpoint:
                    global_step = checkpoint['global_step']
                if 'epoch' in checkpoint:
                    start_epoch = checkpoint['epoch'] + 1
        else:
            sd = checkpoint
        
        # specifing the layer names (key word match) not to load from the checkpoint
        not_resume_layer_names = args.model.not_resume_layer_names
        if not_resume_layer_names:
            for name in not_resume_layer_names:
                sd.pop(name)
                logger.info(f'Not loading layer {name}')
        model.load_state_dict(sd)
    
    # Training loops
    for epoch in range(start_epoch, args.train.max_epoches):
        # set train dataloader sampler random seed with the epoch param.
        train_dataloader.sampler.set_epoch(epoch)
        
        train(train_dataloader, model, criterion, optimizer, epoch, global_step, args, logger)
        global_step += len(train_dataloader)
        test(test_dataloader, model, criterion, optimizer, epoch, global_step, args, logger)


def train(dataloader, model, criterion, optimizer, epoch, global_step, args, logger):
    epoch_size = len(dataloader)

    # modify the STIL num segment (train and test may have different segments)
    model.module.set_segment(args.train.dataset.params.num_segments)

    # set statistical meters and progress meter
    acces = AverageMeter('Acc', ':.4f')
    real_acces = AverageMeter('RealAcc', ':.4f')
    fake_acces = AverageMeter('FakeACC', ':.4f')
    losses = AverageMeter('Loss', ':.4f')
    data_time = AverageMeter('Data', ':.4f')
    batch_time = AverageMeter('Time', ':.4f')
    progress = ProgressMeter(epoch_size, [acces, real_acces, fake_acces, losses, data_time, batch_time])

    model.train()
    end = time.time()
    for idx, datas in enumerate(dataloader):
        data_time.update(time.time() - end)

        # get input data from dataloader
        images, labels, video_paths, segment_indices = datas
        images = images.cuda(args.local_rank)
        labels = labels.cuda(args.local_rank)

        # tune learning rate
        cur_lr = lr_tuner(args.optimizer.params.lr, optimizer, epoch_size, args.scheduler, 
            global_step, args.train.use_warmup, args.train.warmup_epochs)

        # forward
        outputs = model(images)
        loss = criterion(outputs, labels)

        # backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # compute accuracy metrics
        acc, real_acc, fake_acc, real_cnt, fake_cnt = compute_metrics(outputs, labels)

        # update statistical meters 
        acces.update(acc, images.size(0))
        real_acces.update(real_acc, real_cnt)
        fake_acces.update(fake_acc, fake_cnt)
        losses.update(loss.item(), images.size(0))

        # log training metrics at a certain frequency
        if (idx + 1) % args.train.print_info_step_freq == 0:
            logger.info(f'TRAIN Epoch-{epoch}, Step-{global_step}: {progress.display(idx+1)} lr: {cur_lr:.7f}')
        
        global_step += 1

        batch_time.update(time.time() - end)
        end = time.time()


def test(dataloader, model, criterion, optimizer, epoch, global_step, args, logger):
    # modify the STIL num segment (train and test may have different segments)
    model.module.set_segment(args.test.dataset.params.num_segments)

    model.eval()
    y_outputs, y_labels = [], []
    loss_t = 0.
    with torch.no_grad():
        for idx, datas in enumerate(tqdm(dataloader)):
            # get input data from dataloader
            images, labels, video_paths, segment_indices = datas
            images = images.cuda(args.local_rank)
            labels = labels.cuda(args.local_rank)

            # model forward
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss_t += loss * labels.size(0)

            y_outputs.extend(outputs)
            y_labels.extend(labels)
    
    # gather outputs from all distributed nodes
    gather_y_outputs = gather_tensor(y_outputs, args.world_size, to_numpy=False)
    gather_y_labels  = gather_tensor(y_labels, args.world_size, to_numpy=False)
    # compute accuracy metrics
    acc, real_acc, fake_acc, _, _ = compute_metrics(gather_y_outputs, gather_y_labels)
    weight_acc = 0.
    if real_acc and fake_acc:
        weight_acc = 2 / (1 / real_acc + 1 / fake_acc)

    # compute loss
    loss_t = reduce_tensor(loss_t, mean=False)
    loss = (loss_t / len(dataloader.dataset)).item()

    # log test metrics and save the model into the checkpoint file
    lr = optimizer.param_groups[0]['lr']
    logger.info(
        '[TEST] EPOCH-{} Step-{} ACC: {:.4f} RealACC: {:.4f} FakeACC: {:.4f} Loss: {:.5f} lr: {:.7f}'.format(
            epoch, global_step, acc, real_acc, fake_acc, loss, lr
        )
    )
    if args.local_rank == 0:
        test_metrics = {
            'test_acc': acc,
            'test_weight_acc': weight_acc,
            'test_real_acc': real_acc,
            'test_fake_acc': fake_acc,
            'test_loss': loss,
            'lr': lr,
            "epoch": epoch
        }

        checkpoint = OrderedDict()
        checkpoint['state_dict'] = model.state_dict()
        checkpoint['optimizer'] = optimizer.state_dict()
        checkpoint['epoch'] = epoch
        checkpoint['global_step'] = global_step
        checkpoint['metrics'] = test_metrics
        checkpoint['args'] = args

        checkpoint_save_name = \
            "Epoch-{}-Step-{}-ACC-{:.4f}-RealACC-{:.4f}-FakeACC-{:.4f}-Loss-{:.5f}-LR-{:.6g}.tar".format(
                epoch, global_step, acc, real_acc, fake_acc, loss, lr
            )
        checkpoint_save_dir = os.path.join(
            os.path.join(args.exam_dir, 'ckpt'), 
            checkpoint_save_name
        )
        torch.save(checkpoint, checkpoint_save_dir)


if __name__ == '__main__':
    main()
