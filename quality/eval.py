import os
import os.path as osp
import torch
import torch.nn as nn
import torch.optim as opti
from tqdm import tqdm
import torchvision.transforms as T
from generate_pseudo_labels.extract_embedding.model import model
import numpy as np
from scipy import stats
import pdb
from PIL import Image


def read_img(imgPath, device):     # read image & data pre-process
    data = torch.randn(1, 3, 112, 112).to(device)
    transform = T.Compose([
        T.Resize((112, 112)),
        T.ToTensor(),
        T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])
    img = Image.open(imgPath).convert("RGB")
    data[0, :, :, :] = transform(img).to(device)
    return data


def network(eval_model, device):
    net = model.R50([112, 112], use_type="Qua").to(device)
    net_dict = net.state_dict()     
    data_dict = {
        key.replace('module.', ''): value for key, value in torch.load(eval_model, map_location=device).items()}
    net_dict.update(data_dict)
    net.load_state_dict(net_dict)
    net.eval()
    return net

if __name__ == "__main__":
    imgpath = './demo_imgs/1.jpg'                         # [1,2,3.jpg]
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    eval_model = './model/SDD_FIQA_checkpoints_r50.pth'   # checkpoint
    net = network(eval_model, device)
    input_data = read_img(imgpath, device)
    pred_score = net(input_data).data.cpu().numpy().squeeze()
    print(f"Quality score = {pred_score}")
