import os
import sys
import logging
import torch
import torch.cuda.amp as amp
from torch.distributed import ReduceOp
sys.path.append(os.path.join(os.path.abspath(os.path.dirname(__file__)), '..', '..'))

from tasks.localfc.train_localfc import TrainTask


def main():
    """ main function with Traintask in localfc mode, which means each worker has a full classifier
    """
    task_dir = os.path.dirname(os.path.abspath(__file__))
    task = TrainTask(os.path.join(task_dir, 'train_config.yaml'))
    task.init_env()
    task.train()


if __name__ == '__main__':
    main()
