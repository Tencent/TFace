set -e

# Manipulation method. One of ['Deepfakes', 'Face2Face', 'FaceSwap', 'NeuralTextures'].
METHOD=Deepfakes
# video compression rate. One of ['c23', 'c40'].
COMPRESSION=c23

python3 -m torch.distributed.launch --nproc_per_node=8 --master_port 12345 train.py -c configs/ffpp.yaml --method ${METHOD} --compression ${COMPRESSION}