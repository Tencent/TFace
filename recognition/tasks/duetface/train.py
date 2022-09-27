import os
import sys
import torch
import torch.fft

import numpy as np
import torch.cuda.amp as amp
from torch.nn.parallel import DistributedDataParallel

from torchjpeg import dct
from torch.nn import functional as F
from scipy.spatial import ConvexHull, Delaunay

sys.path.append(os.path.join(os.path.abspath(os.path.dirname(__file__)), '..'))
from torchkit.util import AverageMeter, Timer
from torchkit.util import accuracy_dist
from torchkit.util import AllGather
from torchkit.loss import get_loss
from torchkit.task import BaseTask
from tasks.duetface.duetface_model import ClientBackbone
from tasks.duetface.pfld_model import PFLDInference


class TrainTask(BaseTask):
    """ TrainTask in distfc mode, which means classifier shards into multi workers
    """

    def __init__(self, cfg_file):
        super(TrainTask, self).__init__(cfg_file)

    def loop_step(self, epoch):
        backbone, heads = self.backbone, list(self.heads.values())
        backbone.train()  # set to training mode

        if self.cfg['MODE'] == 'INT' and self.cfg['LOAD_CKPT']:
            # freeze client-side model and facial landmark detector
            for module in backbone.modules():
                if isinstance(module, ClientBackbone):
                    module.eval()
                if isinstance(module, PFLDInference):
                    module.eval()

            # load checkpoint for facial landmark detector
            self.inference_model = PFLDInference()
            inference_model_dict = torch.load(self.cfg['LANDMARK_CKPT_PATH'])
            self.inference_model.load_state_dict(inference_model_dict['pfld_backbone'])
            self.inference_model.cuda()
            self.inference_model.eval()

        for head in heads:
            head.train()

        batch_sizes = self.batch_sizes
        am_losses = [AverageMeter() for _ in batch_sizes]
        am_top1s = [AverageMeter() for _ in batch_sizes]
        am_top5s = [AverageMeter() for _ in batch_sizes]
        t = Timer()
        for step, samples in enumerate(self.train_loader):
            self.call_hook("before_train_iter", step, epoch)
            backbone_opt, head_opts = self.opt['backbone'], list(self.opt['heads'].values())

            inputs = samples[0].cuda(non_blocking=True)
            labels = samples[1].cuda(non_blocking=True)

            training_mode = self.cfg['MODE']
            sub_channels = self.cfg['SUB_CHS']

            # provided training mode, adjust inputs
            if training_mode == 'RGB':
                pass  # do nothing
            if training_mode == 'INT':
                # transform RGB images into the frequency domain via BDCT, then perform channel splitting
                main_inputs, sub_inputs = _images_to_dct(inputs, sub_channels=sub_channels)

            # train
            if training_mode == 'RGB':
                if self.amp:
                    with amp.autocast():
                        features = backbone(inputs)
                    features = features.float()
                else:
                    features = backbone(inputs)
            if training_mode == 'INT':
                # warm up stages, use this config to pre-train client-side model alone
                if epoch < self.cfg['SUB_WARMUP_STEP']:
                    if self.amp:
                        with amp.autocast():
                            features = backbone(main_inputs, sub_inputs, warm_up=True)
                        features = features.float()
                    else:
                        features = backbone(main_inputs, sub_inputs, warm_up=True)
                else:
                    if self.amp:
                        with amp.autocast():
                            landmark_inferences = self.calculate_landmarks(inputs)
                            features = backbone(main_inputs, sub_inputs, landmark_inference=landmark_inferences,
                                                warm_up=False)
                        features = features.float()
                    else:
                        landmark_inferences = self.calculate_landmarks(inputs)
                        features = backbone(main_inputs, sub_inputs, landmark_inference=landmark_inferences,
                                            warm_up=False)

            # gather features
            features_gather = AllGather(features, self.world_size)
            features_gather = [torch.split(x, batch_sizes) for x in features_gather]
            all_features = []
            for i in range(len(batch_sizes)):
                all_features.append(torch.cat([x[i] for x in features_gather], dim=0).cuda())

            # gather labels
            with torch.no_grad():
                labels_gather = AllGather(labels, self.world_size)
            labels_gather = [torch.split(x, batch_sizes) for x in labels_gather]
            all_labels = []
            for i in range(len(batch_sizes)):
                all_labels.append(torch.cat([x[i] for x in labels_gather], dim=0).cuda())

            losses = []
            for i in range(len(batch_sizes)):
                # PartialFC need update optimizer state in training process
                if self.pfc:
                    outputs, labels, original_outputs = heads[i](all_features[i], all_labels[i], head_opts[i])
                else:
                    outputs, labels, original_outputs = heads[i](all_features[i], all_labels[i])

                loss = self.loss(outputs, labels) * self.branch_weights[i]
                losses.append(loss)
                prec1, prec5 = accuracy_dist(self.cfg,
                                             original_outputs.data,
                                             all_labels[i],
                                             self.class_shards[i],
                                             topk=(1, 5))
                am_losses[i].update(loss.data.item(), all_features[i].size(0))
                am_top1s[i].update(prec1.data.item(), all_features[i].size(0))
                am_top5s[i].update(prec5.data.item(), all_features[i].size(0))

            # update summary and log_buffer
            scalars = {
                'train/loss': am_losses,
                'train/top1': am_top1s,
                'train/top5': am_top5s,
            }
            self.update_summary({'scalars': scalars})
            log = {
                'loss': am_losses,
                'prec@1': am_top1s,
                'prec@5': am_top5s,
            }
            self.update_log_buffer(log)

            # compute loss
            total_loss = sum(losses)
            # compute gradient and do SGD
            backbone_opt.zero_grad()
            for head_opt in head_opts:
                head_opt.zero_grad()

            # Automatic Mixed Precision setting
            if self.amp:
                self.scaler.scale(total_loss).backward()
                self.scaler.step(backbone_opt)
                for head_opt in head_opts:
                    self.scaler.step(head_opt)
                self.scaler.update()
            else:
                total_loss.backward()
                backbone_opt.step()
                for head_opt in head_opts:
                    head_opt.step()

            # PartialFC need update weight and weight_norm manually
            if self.pfc:
                for head in heads:
                    head.update()

            cost = t.get_duration()

            self.update_log_buffer({'time_cost': cost})

            # call hook function after_train_iter
            self.call_hook("after_train_iter", step, epoch)

    def prepare(self):
        """ common prepare task for training
        """
        for key in self.cfg:
            print(key, self.cfg[key])
        self.make_inputs()
        if self.cfg['MODE'] == 'RGB':
            self.make_model()
        if self.cfg['MODE'] == 'INT':
            self.make_interactive_models()
        self.loss = get_loss('DistCrossEntropy').cuda()
        self.opt = self.get_optimizer()
        self.scaler = amp.GradScaler()
        self.register_hooks()
        self.pfc = self.cfg['HEAD_NAME'] == 'PartialFC'

    def train(self):
        self.prepare()
        self.call_hook("before_run")
        self.backbone = DistributedDataParallel(self.backbone,
                                                device_ids=[self.local_rank], find_unused_parameters=True)
        for epoch in range(self.start_epoch, self.epoch_num):
            self.call_hook("before_train_epoch", epoch)
            self.loop_step(epoch)
            self.call_hook("after_train_epoch", epoch)
        self.call_hook("after_run")

    def calculate_landmarks(self, inputs):
        size = 112
        inputs = inputs * 0.5 + 0.5  # PFLD requires inputs to be within [0, 1]
        _, landmarks = self.inference_model(inputs)
        landmarks = landmarks.detach().cpu().numpy()
        landmarks = landmarks.reshape(landmarks.shape[0], -1, 2)
        landmarks = landmarks * [size, size]
        landmark_masks = torch.zeros((landmarks.shape[0], size, size))

        def in_hull(p, hull):
            #  test if points in `p` are in `hull`
            if not isinstance(hull, Delaunay):
                hull = Delaunay(hull)
            return hull.find_simplex(p) >= 0

        x, y = np.mgrid[0:size:1, 0:size:1]
        grid = np.vstack((y.flatten(), x.flatten())).T  # swap axes

        for i in range(len(landmarks)):
            hull = ConvexHull(landmarks[i])
            points = landmarks[i, hull.vertices, :]
            mask = torch.from_numpy(in_hull(grid, points).astype(int).reshape(size, size)).unsqueeze(0)
            landmark_masks[i] = mask
        landmark_masks = landmark_masks.unsqueeze(dim=1).cuda()
        landmark_masks.requires_grad = False

        return landmark_masks


def main():
    task_dir = os.path.dirname(os.path.abspath(__file__))
    task = TrainTask(os.path.join(task_dir, 'train_dct.yaml'))
    task.init_env()
    task.train()


def _images_to_dct(x, sub_channels=None, size=8, stride=8, pad=0, dilation=1):
    x = x * 0.5 + 0.5  # x to [0, 1]

    x = F.interpolate(x, scale_factor=8, mode='bilinear', align_corners=True)
    x *= 255
    if x.shape[1] == 3:
        x = dct.to_ycbcr(x)
    x -= 128  # x to [-128, 127]
    bs, ch, h, w = x.shape
    block_num = h // stride
    x = x.view(bs * ch, 1, h, w)
    x = F.unfold(x, kernel_size=(size, size), dilation=dilation, padding=pad,
                 stride=(stride, stride))
    x = x.transpose(1, 2)
    x = x.view(bs, ch, -1, size, size)
    dct_block = dct.block_dct(x)
    dct_block = dct_block.view(bs, ch, block_num, block_num, size * size).permute(0, 1, 4, 2, 3)

    channels = list(set([i for i in range(64)]) - set(sub_channels))
    main_inputs = dct_block[:, :, channels, :, :]
    sub_inputs = dct_block[:, :, sub_channels, :, :]
    main_inputs = main_inputs.reshape(bs, -1, block_num, block_num)
    sub_inputs = sub_inputs.reshape(bs, -1, block_num, block_num)
    return main_inputs, sub_inputs


def _dct_to_images(x, size=8, stride=8, pad=0, dilation=1):
    bs, _, _, _ = x.shape
    sampling_rate = 8

    x = x.view(bs, 3, 64, 14 * sampling_rate, 14 * sampling_rate)
    x = x.permute(0, 1, 3, 4, 2)
    x = x.view(bs, 3, 14 * 14 * sampling_rate * sampling_rate, 8, 8)
    x = dct.block_idct(x)
    x = x.view(bs * 3, 14 * 14 * sampling_rate * sampling_rate, 64)
    x = x.transpose(1, 2)
    x = F.fold(x, output_size=(112 * sampling_rate, 112 * sampling_rate),
               kernel_size=(size, size), dilation=dilation, padding=pad, stride=(stride, stride))
    x = x.view(bs, 3, 112 * sampling_rate, 112 * sampling_rate)
    x += 128
    x = dct.to_rgb(x)
    x /= 255
    x = F.interpolate(x, scale_factor=1 / sampling_rate, mode='bilinear', align_corners=True)
    x = x.clamp(min=0.0, max=1.0)
    return x


if __name__ == '__main__':
    main()
