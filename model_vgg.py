import torch
import torch.nn as nn
from torchvision import models

import ssl
ssl._create_default_https_context = ssl._create_unverified_context


class MRNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.pretrained_model = models.vgg16_bn(weights=models.VGG16_BN_Weights.IMAGENET1K_V1)
        self.pretrained_model.fc = nn.Identity()


        self.pooling_layer = nn.AdaptiveAvgPool2d(1)
        self.classifer = nn.Linear(1000, 2)

    def forward(self, x):
        x = torch.squeeze(x, dim=0) 
        # print(f"Features: {x.shape}")
        # features = self.pretrained_model.features(x)
        features = self.pretrained_model(x)
        # print(f"Features: {features.shape}")
        pooled_features = features
        flattened_features = torch.max(pooled_features, 0, keepdim=True)[0]
        # print(f"Flattened Features: {flattened_features.shape}")
        output = self.classifer(flattened_features)
        # print(f"Output: {output.shape}")

        
        return output