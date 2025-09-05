import torch, torch.nn as nn
from torchvision.models.resnet import ResNet, Bottleneck

def resnext(groups=16, width_per_group=4, layers=(3,4,6,3), num_classes=4):
    # ResNeXt-34-equivalent with Bottleneck is uncommon; layers here mirror ResNet-50 topology by default.
    # You can tweak `layers` if you want fewer blocks.
    model = ResNet(block=Bottleneck, layers=layers,
                   groups=groups, width_per_group=width_per_group)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model

class RLResNeXt(nn.Module):
    def __init__(self, num_classes=4):
        super().__init__()
        self.net = resnext(groups=16, width_per_group=4, layers=(3,4,6,3), num_classes=num_classes)
    def forward(self, x): return self.net(x)
