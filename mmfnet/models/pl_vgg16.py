import torch.nn as nn
from torchvision.models import vgg16_bn

class PLVGG16(nn.Module):
    def __init__(self, num_classes=4, pretrained=False):
        self_super = super().__init__()
        self.backbone = vgg16_bn(weights=None if not pretrained else "IMAGENET1K_V1")
        self.backbone.classifier = nn.Sequential(
            nn.Linear(512*7*7, 4096), nn.ReLU(True), nn.Dropout(0.5),
            nn.Linear(4096, 4096),    nn.ReLU(True), nn.Dropout(0.5),
            nn.Linear(4096, num_classes),
        )
    def forward(self, x): return self.backbone(x)
