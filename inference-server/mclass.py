# bits/stdc++.h style: we just define your model here
import torch.nn as nn
from torchvision import models

class MultiTaskModel(nn.Module):
    def __init__(self, backbone_name="resnet18", num_shape_classes=5):
        super().__init__()
        backbone = getattr(models, backbone_name)(pretrained=True)
        self.features = nn.Sequential(*list(backbone.children())[:-1])
        feat_dim = backbone.fc.in_features
        self.gender_head = nn.Linear(feat_dim, 2)
        self.shape_head  = nn.Linear(feat_dim, num_shape_classes)

    #forward
    def forward(self, x):
        x = self.features(x).view(x.size(0), -1)
        return self.gender_head(x), self.shape_head(x)
