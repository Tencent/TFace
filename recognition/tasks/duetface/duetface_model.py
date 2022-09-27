import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.append(os.path.join(os.path.abspath(os.path.dirname(__file__)), '..'))
from tasks.duetface.local_backbones import ClientBackbone, ServerBackbone

activation = {}


def get_activation(name):
    def hook(model, input, output):
        # detach, otherwise the activation will not be removed during backward
        activation[name] = output.detach()

    return hook


class InteractiveBlock(nn.Module):
    def __init__(self):
        super(InteractiveBlock, self).__init__()
        self.activation = nn.Sigmoid()
        self.interface = None
        self.weight = nn.Parameter(torch.tensor(0.))

    def forward(self, x):

        def reshape_and_normalize_masks(inputs, to_bool=False, squeeze=True):
            if squeeze:
                mask = inputs.mean(1)
                mask = mask.unsqueeze(dim=1)
            else:
                mask = inputs
            n, _, h, w = mask.size()
            batch_min, batch_max = [], []
            for i in range(n):
                min, max = mask[i].min().item(), mask[i].max().item()
                img_min, img_max = torch.full((h, w), min).cuda(), torch.full((h, w), max).cuda()
                img_min, img_max = img_min.unsqueeze(dim=0), img_max.unsqueeze(dim=0)
                img_min, img_max = img_min.unsqueeze(dim=0), img_max.unsqueeze(dim=0)  # yes, do it twice (HW -> NCHW)
                batch_min.append(img_min)
                batch_max.append(img_max)

            batch_min = torch.cat(batch_min, dim=0)
            batch_max = torch.cat(batch_max, dim=0)
            mask = (mask - batch_min) / (batch_max - batch_min)
            if to_bool:
                mask = (mask > 0.5).float()  # turn into boolean mask
            return mask

        if len(x) == 3:
            main_inputs, embedding_inputs, inference_x = x[0], x[1], x[2]
        else:
            # deprecated
            main_inputs, embedding_inputs, inference_x = x[0], x[1], None

        shape = main_inputs.shape

        # dynamically produced to meet the client-side shape
        self.interface = nn.Sequential(
            nn.Upsample(size=(shape[2], shape[3]), mode='bilinear'),
            nn.Conv2d(embedding_inputs.shape[1], shape[1], (1, 1))
        ).cuda()
        embedding_inputs = self.interface(embedding_inputs)

        # mask can be smaller than 0, so align to [0, 1] first; reuse reshape_and_normalize_masks for simplicity
        mask = reshape_and_normalize_masks(embedding_inputs)

        # crop mask with the convex hull of facial landmarks to acquire ROI
        if inference_x is not None:
            scale_factor = mask.shape[2] / inference_x.shape[2]
            inference_mask = F.interpolate(inference_x, scale_factor=scale_factor, mode='bilinear', align_corners=True)
            mask = mask * inference_mask

        # again, align cropped mask to [0, 1]
        mask = reshape_and_normalize_masks(mask, squeeze=False)

        main_inputs = self.activation(main_inputs)

        # overlay mask to server-side feature maps
        main_outputs = main_inputs * mask * self.weight + main_inputs

        return main_outputs


class DuetFaceModel(nn.Module):
    def __init__(self, num_sub_channels, len_features, len_sub_features, main_model_name='IR_18',
                 sub_model_name='MobileFaceNet'):
        super(DuetFaceModel, self).__init__()
        if main_model_name == 'IR_18':
            model_size = 18
        else:
            model_size = 50
        # reshape in and output, feature length, and override the server-side unit module
        self.server_model = ServerBackbone([112, 112], model_size, 192 - num_sub_channels, len_features,
                                           len_sub_features, kernel_size=3, unit_module=DuetFaceBasicBlock)
        self.client_model = ClientBackbone(channels_in=num_sub_channels, channels_out=len_sub_features)

    def forward(self, x_server, x_client, landmark_inference=None, warm_up=False):
        if warm_up:
            sub_features = self.client_model(x_client)
            return sub_features
        else:
            # freeze the sub-model after pretraining
            for param in self.client_model.parameters():
                param.requires_grad = False

            # hook the feature maps of intermediate layers to retrieve attention
            body_blocks = list(self.client_model.client_backbone.children())[1:-1]  # retain only bottleneck blocks
            handles = {}
            for i in range(len(body_blocks)):
                name = 'body_{}'.format(str(i))
                handles[name] = body_blocks[i].register_forward_hook(get_activation(name))

            # perform forward to obtain activation for feature masks
            _ = self.client_model(x_client)
            intermediate_output = []
            for i in [1, 3, 5, 7]:
                output = activation['body_{}'.format(str(i))]
                intermediate_output.append(output)
            if landmark_inference is not None:
                main_features = self.server_model([x_server, intermediate_output, landmark_inference])
            else:
                main_features = self.server_model([x_server, intermediate_output])

            # clear activation and remove hook
            activation.clear()
            for key, _ in handles.items():
                handles[key].remove()

            return main_features


class DuetFaceBasicBlock(nn.Module):
    """ BasicBlock for IRNet
    """

    def __init__(self, in_channel, depth, stride, feature_channel, kernel_size, stage=0, embedding=False):
        super(DuetFaceBasicBlock, self).__init__()
        if in_channel == depth:
            self.shortcut_layer = nn.MaxPool2d(1, stride)
        else:
            self.shortcut_layer = nn.Sequential(
                nn.Conv2d(in_channel, depth, (1, 1), stride, bias=False),
                nn.BatchNorm2d(depth))
        self.res_layer = nn.Sequential(
            nn.BatchNorm2d(in_channel),
            nn.Conv2d(in_channel, depth, (3, 3), (1, 1), 1, bias=False),
            nn.BatchNorm2d(depth),
            nn.PReLU(depth),
            nn.Conv2d(depth, depth, (3, 3), stride, 1, bias=False),
            nn.BatchNorm2d(depth))
        if embedding:
            self.embedding_layer = InteractiveBlock()
        else:
            self.embedding_layer = None
        self.stage = stage

    def forward(self, x):
        if len(x) == 2:
            main_x, embedding_x = x[0], x[1]

            shortcut = self.shortcut_layer(main_x)
            res = self.res_layer(main_x)
            main_x = shortcut + res

            if self.embedding_layer is not None:
                main_x = self.embedding_layer([main_x, embedding_x[self.stage]])
            return [main_x, embedding_x]
        else:  # len(x) == 3
            main_x, embedding_x, inference_x = x[0], x[1], x[2]

            shortcut = self.shortcut_layer(main_x)
            res = self.res_layer(main_x)
            main_x = shortcut + res
            if self.embedding_layer is not None:
                main_x = self.embedding_layer([main_x, embedding_x[self.stage], inference_x])
            return [main_x, embedding_x, inference_x]


# test case
if __name__ == '__main__':
    duetface_model = DuetFaceModel(num_sub_channels=30, len_features=512, len_sub_features=512)
    server_input = torch.rand(16, 162, 112, 112)
    client_input = torch.rand(16, 30, 112, 112)
    landmark = torch.rand(16, 1, 112, 112)
    out = duetface_model(server_input, client_input, landmark)
    target = torch.rand(16, 512)
    print(out.shape == target.shape)
