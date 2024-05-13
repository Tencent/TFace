export CUDA_VISIBLE_DEVICES='0'
python3 -u -m torch.distributed.launch --nproc_per_node=1 --nnodes=1 train.py
