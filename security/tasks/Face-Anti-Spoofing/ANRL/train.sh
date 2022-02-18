#!/usr/bin/bash

if [ $# -lt 3 ]; then
    echo "USAGE: sh ./train.sh GPU_NUM CONFIG_PATH PORT"
    exit -1
fi

python3 -u -m torch.distributed.launch --nproc_per_node=$1 --master_port $3 ./train.py -c $2
