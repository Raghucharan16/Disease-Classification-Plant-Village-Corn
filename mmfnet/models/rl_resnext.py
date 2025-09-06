import torch
import torch.nn as nn
from torchvision.models.resnet import ResNet, Bottleneck

def _resnext_16x4d(num_classes=4, layers=(3,4,6,3)):
    model = ResNet(
        block=Bottleneck,
        layers=layers,
        groups=16,            # cardinality
        width_per_group=4
    )
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model

class RLResNeXt(nn.Module):
    def __init__(self, num_classes=4, pretrained=False):
        super().__init__()
        # pretrained=False keeps it internet-free by default
        self.net = _resnext_16x4d(num_classes=num_classes)
    def forward(self, x): return self.net(x)
