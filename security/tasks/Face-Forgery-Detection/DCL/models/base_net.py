import numpy as np
import timm

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ['BinaryClassifier']

MODEL_DICTS = {}
MODEL_DICTS.update(timm.models.__dict__)


class BinaryClassifier(nn.Module):
    def __init__(self, encoder, num_classes=2, drop_rate=0.2, has_feature=False, pretrained=False, **kwargs) -> None:
        """Base binary classifier
        Args:
            encoder ([nn.Module]): Backbone of the DCL
            num_classes (int, optional): Defaults to 2.
            drop_rate (float, optional):  Defaults to 0.2.
            has_feature (bool, optional): Wthether to return feature maps. Defaults to False.
            pretrained (bool, optional): Whether to use a pretrained model. Defaults to False.
        """
        super().__init__()
        self.encoder = MODEL_DICTS[encoder](pretrained=pretrained, **kwargs)
        for k, v in kwargs.items():
            setattr(self, k, v)

        if hasattr(self.encoder, 'get_classifier'):
            self.num_features = self.encoder.get_classifier().in_features
        else:
            self.num_features = self.encoder.last_channel

        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(drop_rate)
        self.has_feature = has_feature
        self.feature_squeeze = nn.Conv2d(self.num_features, 1, 1)
        self.fc = nn.Linear(self.num_features, num_classes)

    def forward(self, x):
        featuremap = self.encoder.forward_features(x)
        x = self.global_pool(featuremap).flatten(1)
        x = self.dropout(x)
        x = self.fc(x)
        if self.has_feature:
            return x, featuremap
        return x


if __name__ == '__main__':
    name = "tf_efficientnet_b4_ns"
    device = 'cpu'
    model = BinaryClassifier(name)
    model = model.to(device)
    model.eval()
    with torch.no_grad():
        inputs = torch.rand(4, 3, 224, 224)
        inputs = inputs.to(device)
        out = model(inputs)
        print(out.shape)
