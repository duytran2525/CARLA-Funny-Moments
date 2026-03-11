import torch.nn as nn

class NvidiaCNN(nn.Module):
    def __init__(self):
        super(NvidiaCNN, self).__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(3, 24, kernel_size=5, stride=2),
            nn.ELU(),
            nn.Conv2d(24, 36, kernel_size=5, stride=2),
            nn.ELU(),
            nn.Conv2d(36, 48, kernel_size=5, stride=2),
            nn.ELU(),
            nn.Conv2d(48, 64, kernel_size=3, stride=1),
            nn.ELU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ELU(),
            
            nn.Flatten()
        )
        
        self.dense_layers = nn.Sequential(
            nn.Linear(1152, 100),
            nn.ELU(),
            nn.Linear(100, 50),
            nn.ELU(),
            nn.Linear(50, 10),
            nn.ELU(),

            nn.Linear(10, 1)
        )

    def forward(self, x):
        x = self.conv_layers(x)
        x = self.dense_layers(x)
        return x.squeeze()


class NvidiaCNNV2(nn.Module):
    """NvidiaCNN with BatchNorm after each Conv2d and Dropout in dense layers."""

    def __init__(self):
        super(NvidiaCNNV2, self).__init__()

        self.conv_layers = nn.Sequential(
            nn.Conv2d(3, 24, kernel_size=5, stride=2),
            nn.BatchNorm2d(24),
            nn.ELU(),
            nn.Conv2d(24, 36, kernel_size=5, stride=2),
            nn.BatchNorm2d(36),
            nn.ELU(),
            nn.Conv2d(36, 48, kernel_size=5, stride=2),
            nn.BatchNorm2d(48),
            nn.ELU(),
            nn.Conv2d(48, 64, kernel_size=3, stride=1),
            nn.BatchNorm2d(64),
            nn.ELU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.BatchNorm2d(64),
            nn.ELU(),
            nn.Flatten(),
        )

        self.dense_layers = nn.Sequential(
            nn.Linear(1152, 100),
            nn.ELU(),
            nn.Dropout(0.5),
            nn.Linear(100, 50),
            nn.ELU(),
            nn.Dropout(0.5),
            nn.Linear(50, 10),
            nn.ELU(),
            nn.Dropout(0.5),
            nn.Linear(10, 1),
        )

    def forward(self, x):
        x = self.conv_layers(x)
        x = self.dense_layers(x)
        return x.squeeze()
