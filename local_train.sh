#!/bin/bash

if [ ! -d "logs" ]; then
    mkdir logs
fi
export CUDA_VISIBLE_DEVICES='0,1,2,3,4,5,6,7'
nohup python -u train.py > logs/$(date +%F-%H-%M-%S).log 2>&1 &
