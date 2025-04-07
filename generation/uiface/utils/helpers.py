import os.path

import numpy as np
import torch
import torch.nn.functional as F
from pytorch_lightning.utilities import rank_zero_only
from utils.colored import colored


@rank_zero_only
def print_status(
    epoch, global_step, loss, metrics=None, time=None, lr=None, log_file_path=None
):
    colored_status = _get_status_string(
        epoch, global_step, loss, metrics, time, lr, use_colors=True
    )
    status = _get_status_string(
        epoch, global_step, loss, metrics, time, lr, use_colors=False
    )
    print(status)
    if log_file_path is not None:
        with open(log_file_path, "a") as f:
            print(status, file=f)


def _get_status_string(
    epoch, global_step, loss, metrics=None, time=None, lr=None, use_colors=True
):
    def maybe_colored(text, color):
        if use_colors:
            return colored(text, color)
        else:
            return text

    e = maybe_colored("Epoch: ", "white") + maybe_colored(str(epoch), "magenta")
    gs = maybe_colored("Global Step: ", "white") + maybe_colored(
        str(global_step), "magenta"
    )
    l = maybe_colored("Loss: ", "white") + maybe_colored(f"{loss: .8f}", "red")

    m = " - ".join(
        [
            maybe_colored(k + ": ", "white") + maybe_colored(str(v), "green")
            for k, v in metrics.items()
        ]
    )

    if time is not None:
        t = "Time:" + maybe_colored(f"{time: .4f}", "magenta")
    else:
        t = ""

    if lr is not None:
        lr = "lr:" + maybe_colored(f"{lr: .8f}", "yellow")
    else:
        lr = ""

    return " | ".join([e, gs, l, m, t, lr])


def count_model_parameters(model: torch.nn.Module):
    trainable_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    rest_count = sum(p.numel() for p in model.parameters() if not p.requires_grad)
    return trainable_count, rest_count, trainable_count + rest_count


def zero_module(module):
    """
    Zero out the parameters of a module and return it.
    """
    for p in module.parameters():
        p.detach().zero_()
    return module


def denormalise(x, mean, std):
    return (x * torch.tensor(std, device=x.device)[None, :, None, None]) + torch.tensor(
        mean, device=x.device
    )[None, :, None, None]


def normalize_to_neg_one_to_one(img):
    return img * 2.0 - 1.0


def denormalize_to_zero_to_one(img):
    img = img.clamp(-1, 1)
    return (img + 1.0) / 2.0


def standardize_to_unit_gaussian(img, mean, std):
    return (img - mean) / std


def destandardize_from_unit_gaussian(img, mean, std):
    img = (img * std) + mean
    return img.clamp(0, 1)


def ensure_path(path):
    if "." in path:
        os.makedirs(os.path.dirname(path), exist_ok=True)
    else:
        os.makedirs(path, exist_ok=True)
    return path


def ensure_path_join(*p):
    path = os.path.join(*p)
    return ensure_path(path)
