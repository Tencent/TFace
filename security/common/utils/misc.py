import os
import random
import wandb
import shutil
import time
import datetime
import warnings
import torch
import numpy as np


def set_seed(SEED):
    """This function set the random seed for the training process
    
    Args:
        SEED (int): the random seed
    """
    if SEED:
        random.seed(SEED)
        np.random.seed(SEED)
        torch.manual_seed(SEED)
        torch.cuda.manual_seed(SEED)
        torch.cuda.manual_seed_all(SEED)
        torch.backends.cudnn.deterministic = True


def setup(cfg):
    if getattr(cfg, 'torch_home', None):
        os.environ['TORCH_HOME'] = cfg.torch_home
    warnings.filterwarnings("ignore")
    seed = cfg.seed
    set_seed(seed)


def init_exam_dir(cfg):
    if cfg.local_rank == 0:
        if not os.path.exists(cfg.exam_dir):
            os.makedirs(cfg.exam_dir)
        ckpt_dir = os.path.join(cfg.exam_dir, 'ckpt')
        if not os.path.exists(ckpt_dir):
            os.makedirs(ckpt_dir)


def init_wandb_workspace(cfg):
    """This function initializes the wandb workspace
    """
    if cfg.wandb.name is None:
        cfg.wandb.name = cfg.config.split('/')[-1].replace('.yaml', '')
    wandb.init(**cfg.wandb)
    allow_val_change = False if cfg.wandb.resume is None else True
    wandb.config.update(cfg, allow_val_change)
    wandb.save(cfg.config)
    if cfg.debug or wandb.run.dir == '/tmp':
        cfg.exam_dir = 'wandb/debug'
        if os.path.exists(cfg.exam_dir):
            shutil.rmtree(cfg.exam_dir)
        os.makedirs(cfg.exam_dir, exist_ok=True)
    else:
        cfg.exam_dir = os.path.dirname(wandb.run.dir)
    os.makedirs(os.path.join(cfg.exam_dir, 'ckpts'), exist_ok=True)
    return cfg


def save_test_results(img_paths, y_preds, y_trues, filename='results.log'):
    assert len(y_trues) == len(y_preds) == len(img_paths)

    with open(filename, 'w') as f:
        for i in range(len(img_paths)):
            print(img_paths[i], end=' ', file=f)
            print(y_preds[i], file=f)
            print(y_trues[i], end=' ', file=f)
