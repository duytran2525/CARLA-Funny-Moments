import torch.nn as nn

class NvidiaCNN(nn.Module):
    def __init__(self):
        super(NvidiaCNN, self).__init__()
        
        # Kiến trúc Nvidia chuẩn yêu cầu đầu vào ảnh kích thước (3, 66, 200)
        # Sử dụng hàm kích hoạt ELU theo khuyến nghị của tác giả để tránh hiện tượng Dead ReLU
        self.conv_layers = nn.Sequential(
            # Khối 1: 3 Lớp Conv với Kernel 5x5, Stride 2x2
            nn.Conv2d(3, 24, kernel_size=5, stride=2),
            nn.ELU(),
            nn.Conv2d(24, 36, kernel_size=5, stride=2),
            nn.ELU(),
            nn.Conv2d(36, 48, kernel_size=5, stride=2),
            nn.ELU(),
            
            # Khối 2: 2 Lớp Conv với Kernel 3x3, Stride 1x1
            nn.Conv2d(48, 64, kernel_size=3, stride=1),
            nn.ELU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ELU(),
            
            nn.Flatten()
        )
        
        # Các lớp Dense (Fully Connected)
        self.dense_layers = nn.Sequential(
            # Output từ Conv layers sau khi flatten kích thước là 64 * 1 * 18 = 1152
            nn.Linear(1152, 100),
            nn.ELU(),
            nn.Linear(100, 50),
            nn.ELU(),
            nn.Linear(50, 10),
            nn.ELU(),
            # Lớp Output cuối cùng dự đoán 1 giá trị góc lái (Không dùng activation)
            nn.Linear(10, 1)
        )

    def forward(self, x):
        x = self.conv_layers(x)
        x = self.dense_layers(x)
        return x.squeeze() # Trả về mảng 1 chiều thay vì [batch_size, 1]