import os
import sys
import argparse
import numpy as np
from omegaconf import OmegaConf
from tqdm import tqdm
from glob import glob
from sklearn import metrics
from scipy.optimize import brentq
from scipy.interpolate import interp1d

import torch

import models
import datasets

sys.path.append(os.path.join(os.path.abspath(os.path.dirname(__file__)), '..', '..', '..'))

from common.utils import save_test_results


def load_model(args):
    """Load models with args
    Returns:
        [nn.Module]: [model]
    """
    if args.ckpt_path is not None:
        print(f'resume model from {args.ckpt_path}')
        checkpoint = torch.load(args.ckpt_path, weights_only=True)
        if getattr(args, 'transform', None) is None:
            args.transform = checkpoint['args'].transform
        if getattr(args, 'model', None) is None:
            args.model = checkpoint['args'].model
        if getattr(args, 'base_model', None) is None:
            args.base_model = checkpoint['args'].base_model

        if args.model["name"] == "DCL":
            base_model = models.__dict__[args.base_model.name](**args.base_model.params)
            model = models.__dict__[args.model.name](base_model, **args.model.params)
        else:
            model = models.__dict__[args.model.name](**args.model.params)

        state_dict = checkpoint if args.ckpt_path.endswith('pth') else checkpoint['state_dict']
        model.load_state_dict(state_dict)
    else:
        assert getattr(args, 'model', None) is not None
        model = models.__dict__[args.model.name](**args.model.params)
    print(args.model)
    return model, args


def model_forward(args, inputs, model):
    """Model forward

    Args:
        inputs ([Tensor]): [Input image]
        model ([nn.Module])

    Returns:
        [Tensor]
    """
    output = model(inputs)
    if type(output) is tuple or type(output) is list:
        output = output[0]
    if output.shape[1] == 2:
        prob = 1 - torch.softmax(output, dim=1)[:, 0].squeeze().cpu().numpy()
    else:
        prob = torch.sigmoid(output).squeeze().cpu().numpy()
    return prob


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', type=str, default='configs/test.yaml')
    parser.add_argument('--distributed', type=int, default=0)
    parser.add_argument('--exam_id', type=str, default='')
    parser.add_argument('--ckpt_path', type=str, default='')
    parser.add_argument('--dataset', type=str, default='')
    parser.add_argument('--compress', type=str, default='')
    parser.add_argument('--constract', type=bool, default=False)

    parser.add_argument('--debug', action='store_true', default=False)
    args = parser.parse_args()

    local_config = OmegaConf.load(args.config)
    for k, v in local_config.items():
        if args.dataset != '':
            if k == 'dataset':
                v["name"] = args.dataset
                if args.compress != '':
                    v[args.dataset]["compressions"] = args.compress

        setattr(args, k, v)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if args.exam_id:
        ckpt_path = glob(f'wandb/*{args.exam_id}/ckpts/model_best.pth.tar')

        if len(ckpt_path) >= 1:
            args.ckpt_path = ckpt_path[0]

    if args.ckpt_path:
        args.output_dir = os.path.dirname(args.ckpt_path)

    os.environ['TORCH_HOME'] = args.torch_home

    model, args = load_model(args)
    model = model.to(device)
    model.eval()

    test_dataloader = datasets.create_dataloader(args, split='test')

    y_trues = []
    y_preds = []
    acces = []
    img_paths = []
    for i, datas in enumerate(tqdm(test_dataloader)):
        with torch.no_grad():
            images = datas[0].to(device)
            targets = datas[1].float().numpy()
            y_trues.extend(targets)
            prob = model_forward(args, images, model)
            prediction = (prob >= args.test.threshold).astype(float)
            y_preds.extend(prob)
            acces.extend(targets == prediction)
            if args.test.record_results:
                img_paths.extend(datas[2])

    acc = np.mean(acces)
    fpr, tpr, thresholds = metrics.roc_curve(y_trues, y_preds, pos_label=1)
    AUC = metrics.auc(fpr, tpr)
    eer = brentq(lambda x: 1. - x - interp1d(fpr, tpr)(x), 0., 1.)
    thresh = interp1d(fpr, thresholds)(eer)
    print(f'#Total# ACC:{acc:.5f}  AUC:{AUC:.5f}  EER:{100*eer:.2f}(Thresh:{thresh:.3f})')

    preds = np.array(y_preds) >= args.test.threshold
    pred_fake_nums = np.sum(preds)
    pred_real_nums = len(preds) - pred_fake_nums
    print(f"pred dataset:{args.dataset.name},pred id: {args.exam_id},compress:{args.compress}")
    print(f'pred real nums:{pred_real_nums} pred fake nums:{pred_fake_nums}')

    if args.test.record_results:
        if args.exam_id is not None:
            task_id = args.exam_id
            filename = glob(f'wandb/*{task_id}/preds_{args.dataset.name}.log')
        else:
            task_id = args.model.params.model_path.split('/')[1]
            filename = f'logs/test/{task_id}_preds_{args.dataset.name}.log'
        save_test_results(y_trues, y_preds, img_paths, filename=filename)


if __name__ == '__main__':
    main()
