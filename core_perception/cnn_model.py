import torch.nn as nn

class NvidiaCNN(nn.Module):
    def __init__(self):
        super(NvidiaCNN, self).__init__()
        
        # Kiß║┐n tr├║c Nvidia chuß║⌐n y├¬u cß║ºu ─æß║ºu v├áo ß║únh k├¡ch th╞░ß╗¢c (3, 66, 200)
        # Sß╗¡ dß╗Ñng h├ám k├¡ch hoß║ít ELU theo khuyß║┐n nghß╗ï cß╗ºa t├íc giß║ú ─æß╗â tr├ính hiß╗çn t╞░ß╗úng Dead ReLU
        self.conv_layers = nn.Sequential(
            # Khß╗æi 1: 3 Lß╗¢p Conv vß╗¢i Kernel 5x5, Stride 2x2
            nn.Conv2d(3, 24, kernel_size=5, stride=2),
            nn.ELU(),
            nn.Conv2d(24, 36, kernel_size=5, stride=2),
            nn.ELU(),
            nn.Conv2d(36, 48, kernel_size=5, stride=2),
            nn.ELU(),
            
            # Khß╗æi 2: 2 Lß╗¢p Conv vß╗¢i Kernel 3x3, Stride 1x1
            nn.Conv2d(48, 64, kernel_size=3, stride=1),
            nn.ELU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ELU(),
            
            nn.Flatten()
        )
        
        # C├íc lß╗¢p Dense (Fully Connected)
        self.dense_layers = nn.Sequential(
            # Output tß╗½ Conv layers sau khi flatten k├¡ch th╞░ß╗¢c l├á 64 * 1 * 18 = 1152
            nn.Linear(1152, 100),
            nn.ELU(),
            nn.Linear(100, 50),
            nn.ELU(),
            nn.Linear(50, 10),
            nn.ELU(),
            # Lß╗¢p Output cuß╗æi c├╣ng dß╗▒ ─æo├ín 1 gi├í trß╗ï g├│c l├íi (Kh├┤ng d├╣ng activation)
            nn.Linear(10, 1)
        )

    def forward(self, x):
        x = self.conv_layers(x)
        x = self.dense_layers(x)
        return x.squeeze() # Trß║ú vß╗ü mß║úng 1 chiß╗üu thay v├¼ [batch_size, 1]


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
