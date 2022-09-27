#!/bin/bash

if [ ! -d "logs" ]; then
    mkdir logs
fi

if [ ! -d "ckpt" ]; then
    mkdir ckpt
fi

export CUDA_VISIBLE_DEVICES='0'
nohup python3 -u -m torch.distributed.launch --nproc_per_node=1 --nnodes=1 train.py > logs/$(date +%F-%H-%M-%S).log 2>&1 &
