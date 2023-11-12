import torch
import torch.nn as nn
from torchvision import models

import ssl
ssl._create_default_https_context = ssl._create_unverified_context

class AlexEnsembleModel(nn.Module):   
    def __init__(self, count):
        super().__init__()
        self.model = models.alexnet(pretrained=True)
        self.pooling_layer = nn.AdaptiveAvgPool2d(1)
        self.count = count
        self.classifier = nn.Linear(count, 2)
        
    def forward(self, x):
        x = torch.squeeze(x, dim=0) 
        x_list = []
        for i in range(self.count):
  
            x1 = self.model(x)
            x1 = self.pooling_layer(x1)
            # # print(f"Pooled Features 1: {pooled_features.shape}")
            # pooled_features = pooled_features.view(pooled_features.size(0), -1)
            # # print(f"Pooled Features 2: {pooled_features.shape}")
            # flattened_features = torch.max(pooled_features, 0, keepdim=True)[0]
            x_list.append(x1)
            

        x = torch.cat(x_list, dim=1)
        out = self.classifier(x)

        return out

