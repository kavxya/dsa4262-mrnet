import torch
import torch.nn as nn
from torchvision import models

import ssl
ssl._create_default_https_context = ssl._create_unverified_context


class MRNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.pretrained_model = models.alexnet(pretrained=True)
        self.pooling_layer = nn.AdaptiveAvgPool2d(1)
        self.classifer = nn.Linear(256, 2)

    def forward(self, x):
        x = torch.squeeze(x, dim=0) 
        # print(f"X: {x.shape}")
        features = self.pretrained_model.features(x)
        # print(f"Features: {features}")
        # print(f"Features: {features.shape}")
        pooled_features = self.pooling_layer(features)
        # print(f"Pooled Features 1: {pooled_features.shape}")
        pooled_features = pooled_features.view(pooled_features.size(0), -1)
        # print(f"Pooled Features 2: {pooled_features.shape}")
        flattened_features = torch.max(pooled_features, 0, keepdim=True)[0]
        # print(f"Flattened Features: {flattened_features.shape}")
        output = self.classifer(flattened_features)
        # print(f"Output: {output.shape}")
        return output