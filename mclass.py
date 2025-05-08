import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models, transforms, datasets
from torch.utils.data import DataLoader
from PIL import Image
from pathlib import Path
import matplotlib.pyplot as plt

class MultiTaskModel(nn.Module):
    def __init__(self, backbone_name="resnet18", num_shape_classes=5):
        super().__init__()
        backbone = getattr(models, backbone_name)(pretrained=True)
        self.features = nn.Sequential(*list(backbone.children())[:-1])
        feat_dim = backbone.fc.in_features
        self.gender_head = nn.Linear(feat_dim, 2)
        self.shape_head  = nn.Linear(feat_dim, num_shape_classes)

    def forward(self, x):
        x = self.features(x).view(x.size(0), -1)
        return self.gender_head(x), self.shape_head(x)
