import os
import sys
from omegaconf import OmegaConf
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.distributed as dist
import torch.utils.data as data
import torch.nn.functional as F

from models import *
from datasets import *

sys.path.append(os.path.join(os.path.abspath(os.path.dirname(__file__)), '..', '..', '..'))

from common import losses, optimizers
from common.utils import *
from utils import *


args = get_params()
setup(args)


###########################
# main logic for test #
###########################
def main():
    # use distributed test with nccl backend 
    args.local_rank = int(os.environ.get('LOCAL_RANK', 0))
    dist.init_process_group(backend='nccl', init_method="env://")
    torch.cuda.set_device(args.local_rank)
    args.world_size = dist.get_world_size()

    # set model and wrap it with DistributedDataParallel
    model = eval(args.model.name)(**args.model.params)
    model.cuda(args.local_rank)
    model = nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank])

    # load the checkpoint file from model.resume in the input args
    # model.resume must be set here.
    ckpt_load_path = args.model.resume
    print(f'ckpt_load_path: {ckpt_load_path}')
    if not ckpt_load_path:
        raise ValueError("You must load a checkpoint by specifying the `model.resume` argument.")
    checkpoint = torch.load(ckpt_load_path, map_location='cpu')
    if 'state_dict' in checkpoint:
        sd = checkpoint['state_dict']
    else:
        sd = checkpoint
    model.load_state_dict(sd)

    args.test.dataset.params.split = 'test'
    test_dataloader = get_dataloader(args, 'test')

    # main test function
    test(test_dataloader, model, args)


def test(dataloader, model, args):
    # modify the STIL num segment (train and test may have different segments)
    model.module.set_segment(args.test.dataset.params.num_segments)
    model.eval()
    y_outputs, y_labels = [], []
    with torch.no_grad():
        for _, datas in enumerate(tqdm(dataloader)):
            images, labels, video_paths, segment_indices = datas
            images = images.cuda(args.local_rank)
            labels = labels.cuda(args.local_rank)

            outputs = model(images)

            y_outputs.extend(outputs)
            y_labels.extend(labels)
    
    gather_y_outputs = gather_tensor(y_outputs, args.world_size, to_numpy=False)
    gather_y_labels  = gather_tensor(y_labels, args.world_size, to_numpy=False)
    acc, real_acc, fake_acc, _, _ = compute_metrics(gather_y_outputs, gather_y_labels)
    
    result = f'ACC: {acc:.4f} RealACC: {real_acc:.4f} FakeACC: {fake_acc:.4f}'
    if args.local_rank == 0:
        print(result)

if __name__ == "__main__":
    main()
