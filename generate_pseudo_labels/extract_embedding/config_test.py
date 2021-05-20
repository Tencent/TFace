import torch
import torchvision.transforms as T


class Config:
    # dataset
    data_root = ''
    img_list = '../DATA.labelpath'
    eval_model = './model/MobileFaceNet_MS1M.pth'
    outfile = '../feats_npy/Embedding_Features.npy'
    # data preprocess
    transform = T.Compose([
        T.Resize((112, 112)),
        T.ToTensor(),
        T.Normalize(mean=[0.5,0.5,0.5], std=[0.5,0.5,0.5]),
    ])
    # network settings
    backbone = 'MFN'               # [MFN, R_50]
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    multi_GPUs = [0,1,2,3,4,5,6]
    embedding_size = 512
    batch_size = 2000
    pin_memory = True
    num_workers = 4
config = Config()
