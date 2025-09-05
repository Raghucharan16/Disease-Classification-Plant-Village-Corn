import torch, torch.nn as nn, torch.nn.functional as F

class PL1D_AlexNet(nn.Module):
    def __init__(self, in_feats=4, num_classes=4):
        super().__init__()
        C = 64
        self.net = nn.Sequential(
            nn.Conv1d(in_feats, C, kernel_size=3, padding=1), nn.ReLU(), nn.MaxPool1d(2), nn.Dropout(0.25),
            nn.Conv1d(C, C*2, kernel_size=3, padding=1),     nn.ReLU(), nn.MaxPool1d(2), nn.Dropout(0.25),
            nn.Conv1d(C*2, C*4, kernel_size=3, padding=1),   nn.ReLU(), nn.AdaptiveMaxPool1d(1)
        )
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(C*4, 128), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(128, num_classes)
        )

    def forward(self, sensors):  # sensors: (B, F)
        x = sensors.unsqueeze(-1).transpose(1,2)  # (B, F, 1) -> (B, 1, F) -> for Conv1d we want (B, C_in, L)
        # Above we used in_feats=4 as channels; if you prefer L=4, switch dims accordingly
        # Alternative: treat features as channels with L=1
        x = sensors.unsqueeze(-1).transpose(1,2)  # (B, F, 1) => channels=F, L=1
        x = self.net(x)
        return self.fc(x)
