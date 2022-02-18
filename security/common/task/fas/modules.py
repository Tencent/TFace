import os
import sys
import numpy as np
from tqdm import tqdm

import torch

sys.path.append(os.path.join(os.path.abspath(os.path.dirname(__file__)), '..', '..'))
from common.utils import *


def test_module(model, test_data_loaders, forward_function, device='cuda', distributed=False):
    """Test module for Face Anti-spoofing

    Args:
        model (nn.module): fas model
        test_data_loaders (torch.dataloader): list of test data loaders
        forward_function (function): model forward function
        device (str, optional): Defaults to 'cuda'.
        distributed (bool, optional): whether to use distributed training. Defaults to False.

    Returns:
        y_preds (list): predictions
        y_trues (list): ground truth labels
    """
    prob_dict = {}
    label_dict = {}

    y_preds = []
    y_trues = []

    model.eval()
    for loaders in test_data_loaders:
        for iter, datas in enumerate(tqdm(loaders)):
            with torch.no_grad():
                images = datas[0].to(device)
                targets = datas[1].to(device)
                map_GT = datas[2].to(device)
                img_path = datas[3]
                probs = forward_function(images)

                if not distributed:
                    probs = probs.cpu().data.numpy()
                    label = targets.cpu().data.numpy()

                    for i in range(len(probs)):
                        # the image of the same video share the same video_path
                        video_path = img_path[i].rsplit('/', 1)[0]
                        if (video_path in prob_dict.keys()):
                            prob_dict[video_path].append(probs[i])
                            label_dict[video_path].append(label[i])
                        else:
                            prob_dict[video_path] = []
                            label_dict[video_path] = []
                            prob_dict[video_path].append(probs[i])
                            label_dict[video_path].append(label[i])
                else:
                    y_preds.extend(probs)
                    y_trues.extend(targets)

    if not distributed:
        y_preds = []
        y_trues = []
        for key in prob_dict.keys():
            # calculate the scores in video-level via averaging the scores of the images from the same videos
            avg_single_video_prob = sum(prob_dict[key]) / len(prob_dict[key])
            avg_single_video_label = sum(label_dict[key]) / len(label_dict[key])
            y_preds = np.append(y_preds, avg_single_video_prob)
            y_trues = np.append(y_trues, avg_single_video_label)

    return y_preds, y_trues
