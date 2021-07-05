#!/bin/bash

if [ ! -d "logs" ]; then
    mkdir logs
fi

if [ ! -d "ckpt" ]; then
    mkdir ckpt
fi

export CUDA_VISIBLE_DEVICES='0,1,2,3,4,5,6,7'
nohup python -u -m torch.distributed.launch --nproc_per_node=8 --nnodes=1 ./tasks/distfc/train_distfc.py > logs/$(date +%F-%H-%M-%S).log 2>&1 &
