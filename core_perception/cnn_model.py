import torch
import torch.nn as nn


class NvidiaCNN(nn.Module):
    """
    Kiến trúc CNN gốc của Nvidia dành cho bài toán Behavioral Cloning (Phase 1).

    Mô hình nhận một ảnh RGB/YUV đầu vào và dự đoán **góc lái** (steering angle)
    tương ứng, không sử dụng bất kỳ thông tin bổ sung nào (vận tốc, lệnh GPS).

    Kiến trúc tổng quan
    -------------------
    Input: (B, 3, 66, 200) – batch ảnh YUV đã chuẩn hoá về [-1, 1]

    Convolutional backbone (trích xuất đặc trưng hình ảnh):
        Conv(3→24,  5×5, stride=2) → ELU
        Conv(24→36, 5×5, stride=2) → ELU
        Conv(36→48, 5×5, stride=2) → ELU
        Conv(48→64, 3×3, stride=1) → ELU
        Conv(64→64, 3×3, stride=1) → ELU
        Flatten → vector 1 152 chiều

    Fully-connected regressor (ra góc lái):
        Linear(1152→100) → ELU
        Linear(100→50)   → ELU
        Linear(50→10)    → ELU
        Linear(10→1)

    Output: (B,) – góc lái trong khoảng [-1, 1]

    Lưu ý
    -----
    * Không có BatchNorm hay Dropout: mô hình này phù hợp để kiểm chứng nhanh
      nhưng dễ bị overfit trên tập dữ liệu nhỏ.  Dùng ``NvidiaCNNV2`` cho
      huấn luyện thực tế.
    * Kích thước ảnh đầu vào *phải* là 66×200.  Nếu CARLA thu ảnh ở kích thước
      khác, hãy resize trong ``dataset.py`` trước khi đưa vào mô hình.

    Tham khảo
    ---------
    Bojarski et al., "End to End Learning for Self-Driving Cars", Nvidia 2016.
    https://arxiv.org/abs/1604.07316
    """

    def __init__(self):
        """
        Khởi tạo NvidiaCNN và đăng ký toàn bộ trọng số với PyTorch.

        Hai thuộc tính chính được tạo ra:
        * ``self.conv_layers`` – Sequential chứa 5 lớp Conv2d + ELU + Flatten.
        * ``self.dense_layers`` – Sequential chứa 4 lớp Linear + ELU.
        """
        super(NvidiaCNN, self).__init__()

        # -----------------------------------------------------------------
        # Convolutional backbone
        # Mỗi lớp conv rút gọn kích thước không gian (spatial) của feature map:
        #   Input : (B, 3,  66, 200)
        #   After conv1 (5×5 s2): (B, 24, 31, 98)
        #   After conv2 (5×5 s2): (B, 36, 14, 47)
        #   After conv3 (5×5 s2): (B, 48,  5, 22)
        #   After conv4 (3×3 s1): (B, 64,  3, 20)
        #   After conv5 (3×3 s1): (B, 64,  1, 18)
        #   After Flatten        : (B, 1152)
        # -----------------------------------------------------------------
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

        # -----------------------------------------------------------------
        # Fully-connected regressor
        # Thu hẹp dần số chiều từ 1152 về 1 (góc lái).
        # ELU được chọn thay ReLU để tránh hiện tượng "Dying ReLU" – các
        # neuron bị khoá vĩnh viễn khi gradient âm liên tục.
        # -----------------------------------------------------------------
        self.dense_layers = nn.Sequential(
            nn.Linear(1152, 100),
            nn.ELU(),
            nn.Linear(100, 50),
            nn.ELU(),
            nn.Linear(50, 10),
            nn.ELU(),

            nn.Linear(10, 1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Thực hiện một lần lan truyền xuôi (forward pass).

        Parameters
        ----------
        x : torch.Tensor
            Batch ảnh đầu vào, shape ``(B, 3, 66, 200)``, kiểu ``float32``,
            đã chuẩn hoá về ``[-1, 1]`` (YUV hoặc RGB đều được, miễn là nhất
            quán với quá trình tiền xử lý trong ``dataset.py``).

        Returns
        -------
        torch.Tensor
            Góc lái dự đoán, shape ``(B,)`` khi B > 1 hoặc scalar khi B = 1
            (do ``.squeeze()`` loại bỏ chiều cuối kích thước 1).
            Giá trị nằm trong khoảng ``[-1, 1]`` nếu dữ liệu huấn luyện được
            clamp đúng cách.
        """
        x = self.conv_layers(x)
        x = self.dense_layers(x)
        return x.squeeze()


class NvidiaCNNV2(nn.Module):
    """
    Phiên bản cải tiến của NvidiaCNN với BatchNorm và Dropout (Phase 1 – phiên bản sản xuất).

    So với ``NvidiaCNN`` gốc, phiên bản này bổ sung:

    * **BatchNorm2d** sau mỗi lớp Conv2d: chuẩn hoá activation trong mini-batch,
      giúp gradient ổn định hơn, tốc độ hội tụ nhanh hơn và giảm độ nhạy cảm
      với learning rate.
    * **Dropout(0.5)** sau mỗi lớp Linear (trừ lớp đầu ra): kỹ thuật
      regularisation ngẫu nhiên làm cho mô hình bền vững hơn với tập dữ liệu
      nhỏ, hạn chế overfitting.

    Kiến trúc tổng quan
    -------------------
    Input: (B, 3, 66, 200) – batch ảnh YUV đã chuẩn hoá về [-1, 1]

    Convolutional backbone:
        Conv(3→24,  5×5, s2) → BN → ELU
        Conv(24→36, 5×5, s2) → BN → ELU
        Conv(36→48, 5×5, s2) → BN → ELU
        Conv(48→64, 3×3, s1) → BN → ELU
        Conv(64→64, 3×3, s1) → BN → ELU
        Flatten → 1 152 chiều

    Fully-connected regressor:
        Linear(1152→100) → ELU → Dropout(0.5)
        Linear(100→50)   → ELU → Dropout(0.5)
        Linear(50→10)    → ELU → Dropout(0.5)
        Linear(10→1)

    Output: (B,) – góc lái trong khoảng [-1, 1]

    Đây là mô hình **được dùng mặc định** trong ``scripts/train_cnn.py``.
    """

    def __init__(self):
        """
        Khởi tạo NvidiaCNNV2 và đăng ký toàn bộ trọng số với PyTorch.

        Hai thuộc tính chính được tạo ra:
        * ``self.conv_layers`` – Sequential chứa 5 lớp Conv2d + BatchNorm2d + ELU + Flatten.
        * ``self.dense_layers`` – Sequential chứa 4 lớp Linear + ELU + Dropout.
        """
        super(NvidiaCNNV2, self).__init__()

        # -----------------------------------------------------------------
        # Convolutional backbone + BatchNorm
        # BatchNorm2d(C) chuẩn hoá mỗi kênh (channel) độc lập trên toàn
        # mini-batch, giúp mô hình ít phụ thuộc vào phân phối dữ liệu ban đầu.
        # Thứ tự: Conv → BN → Activation (ELU) là thứ tự chuẩn được khuyến nghị.
        # -----------------------------------------------------------------
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

        # -----------------------------------------------------------------
        # Fully-connected regressor + Dropout
        # Dropout(p=0.5) tắt ngẫu nhiên 50% neuron trong lúc train.
        # Ở chế độ eval() (model.eval()), Dropout tự tắt và mô hình sử dụng
        # toàn bộ neuron – không cần thay đổi code khi inference.
        # -----------------------------------------------------------------
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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Thực hiện một lần lan truyền xuôi (forward pass).

        Parameters
        ----------
        x : torch.Tensor
            Batch ảnh đầu vào, shape ``(B, 3, 66, 200)``, kiểu ``float32``,
            đã chuẩn hoá về ``[-1, 1]``.

        Returns
        -------
        torch.Tensor
            Góc lái dự đoán, shape ``(B,)`` khi B > 1 hoặc scalar khi B = 1.

        Lưu ý
        -----
        Hành vi của BatchNorm và Dropout **phụ thuộc vào chế độ** của mô hình:

        * ``model.train()`` – BatchNorm dùng thống kê mini-batch, Dropout bật.
        * ``model.eval()``  – BatchNorm dùng running stats, Dropout tắt (dự đoán
          deterministc).  Luôn gọi ``model.eval()`` trước khi inference.
        """
        x = self.conv_layers(x)
        x = self.dense_layers(x)
        return x.squeeze()


class CIL_NvidiaCNN(nn.Module):
    """
    Phase 2 – Conditional Imitation Learning CNN.

    Mô hình này mở rộng NvidiaCNNV2 để hỗ trợ **điều hướng có điều kiện**:
    xe không chỉ nhìn ảnh mà còn "nghe" lệnh GPS (rẽ trái / phải / đi thẳng)
    và cảm nhận vận tốc hiện tại trước khi quyết định góc lái.

    Kiến trúc tổng quan
    -------------------

    ::

                     ┌─────────────────────────────┐
        image ──────►│  Conv backbone (NvidiaCNNV2)│──► vis_feat (1152-d)
                     └─────────────────────────────┘         │
                     ┌─────────────────────────────┐         │  cat                               
        speed ──────►│  Speed branch (FC 1→64→32)  │──► spd_feat (32-d) ──►│
                     └─────────────────────────────┘                       │
                                                              features (1184-d)
                                                                  │
                                              ┌───────────────────┴───────────────────┐
                                    Head-0    │  Head-1  │  Head-2  │  Head-3         │
                                  (Follow    )(Turn Left)(Turn Right)(Go Straight)    │
                                              └───────────────────────────────────────┘
                                                                  │
        command ─────────────────────────────────────────► select matching head
                                                                  │
                                                              steering (B,)

    Inputs
    ------
    image   : (B, 3, 66, 200) – ảnh YUV đã chuẩn hoá về [-1, 1]
    speed   : (B,)            – vận tốc đã chuẩn hoá về [0, 1]
                                 (chia cho ``CILCarlaDataset.MAX_SPEED_KMH = 120``)
    command : (B,)            – chỉ số lệnh GPS kiểu ``long``:
                                 0 = Follow Lane  (giữ làn, không có ngã tư)
                                 1 = Turn Left    (rẽ trái tại ngã tư)
                                 2 = Turn Right   (rẽ phải tại ngã tư)
                                 3 = Go Straight  (đi thẳng qua ngã tư)

    Output
    ------
    steering : (B,) – góc lái dự đoán trong [-1, 1]

    Tại sao dùng nhiều "expert head"?
    ----------------------------------
    Theo bài báo CIL gốc (Codevilla et al., 2018), việc dùng một mạng chung
    cho tất cả các lệnh khiến mô hình bị "nhầm lẫn" vì một ảnh ngã tư trông
    giống nhau dù ta muốn rẽ trái hay phải.  Mỗi head chuyên biệt cho một lệnh
    giải quyết vấn đề này: head không được chọn không đóng góp vào loss trong
    lúc train, nên mỗi head học hành vi riêng của nó.

    Tham khảo
    ---------
    Codevilla et al., "End-to-end Driving via Conditional Imitation Learning",
    ICRA 2018. https://arxiv.org/abs/1710.02410
    """

    NUM_COMMANDS = 4
    _CONV_FLAT_DIM = 1152  # 64 filters × 1 H × 18 W after 5 conv layers on a 66×200 input
    _SPEED_EMB_DIM = 32

    def __init__(self):
        """
        Khởi tạo CIL_NvidiaCNN và đăng ký toàn bộ trọng số với PyTorch.

        Ba nhóm thuộc tính được tạo ra:

        * ``self.conv_layers``   - Shared visual backbone (giống NvidiaCNNV2).
        * ``self.speed_branch``  - FC nhỏ nhúng vận tốc (scalar → 32-d vector).
        * ``self.command_heads`` - ``nn.ModuleList`` gồm ``NUM_COMMANDS = 4``
          expert head; mỗi head là một FC regressor độc lập.
        """
        super().__init__()

        # -----------------------------------------------------------------
        # Shared visual backbone (same as NvidiaCNNV2)
        # Backbone được chia sẻ giữa tất cả các command heads để tận dụng
        # việc học đặc trưng hình ảnh chung (làn đường, biển báo, v.v.).
        # -----------------------------------------------------------------
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

        # -----------------------------------------------------------------
        # Speed embedding branch
        # Vận tốc (1 scalar) được nhúng vào không gian 32 chiều để mạng có
        # thể học mối quan hệ phi tuyến giữa tốc độ và góc lái.
        # Ví dụ: ở tốc độ cao, góc lái tối ưu thường nhỏ hơn (xe quẹo ít hơn).
        # -----------------------------------------------------------------
        self.speed_branch = nn.Sequential(
            nn.Linear(1, 64),
            nn.ELU(),
            nn.Linear(64, self._SPEED_EMB_DIM),
            nn.ELU(),
        )

        # -----------------------------------------------------------------
        # Conditional command heads
        # Mỗi head nhận vector ghép (vis_feat ∥ spd_feat) 1184 chiều và ra
        # một giá trị góc lái.  ModuleList đảm bảo PyTorch theo dõi tham số
        # của tất cả heads trong quá trình tối ưu.
        # -----------------------------------------------------------------
        head_input_dim = self._CONV_FLAT_DIM + self._SPEED_EMB_DIM
        self.command_heads = nn.ModuleList(
            [self._make_head(head_input_dim) for _ in range(self.NUM_COMMANDS)]
        )

    @staticmethod
    def _make_head(input_dim: int) -> nn.Sequential:
        """
        Tạo một expert head (FC regressor) cho một lệnh điều hướng cụ thể.

        Mỗi head là một mạng fully-connected nhỏ với hai lớp ẩn và Dropout
        để regularize.  Không dùng activation ở lớp đầu ra để đầu ra là giá
        trị thực (regression).

        Parameters
        ----------
        input_dim : int
            Số chiều của vector đặc trưng đầu vào.  Bằng
            ``_CONV_FLAT_DIM + _SPEED_EMB_DIM = 1184``.

        Returns
        -------
        nn.Sequential
            Mạng FC: Linear(input_dim→256) → ELU → Dropout(0.5)
                     → Linear(256→64) → ELU → Dropout(0.3)
                     → Linear(64→1)
        """
        return nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ELU(),
            nn.Dropout(0.5),
            nn.Linear(256, 64),
            nn.ELU(),
            nn.Dropout(0.3),
            nn.Linear(64, 1),
        )

    def forward(self, image: torch.Tensor, speed: torch.Tensor,
                command: torch.Tensor) -> torch.Tensor:
        """
        Thực hiện một lần lan truyền xuôi cho CIL model.

        Quy trình bên trong
        -------------------
        1. Trích xuất đặc trưng hình ảnh qua ``conv_layers`` → ``vis_feat``.
        2. Nhúng vận tốc qua ``speed_branch`` → ``spd_feat``.
        3. Ghép nối ``[vis_feat ∥ spd_feat]`` → ``features``.
        4. Chạy ``features`` qua *tất cả* 4 heads song song → ``head_outputs``.
        5. Dùng ``torch.gather`` để chọn đúng head tương ứng với ``command``
           của từng mẫu trong batch.

        Parameters
        ----------
        image : torch.Tensor
            Batch ảnh, shape ``(B, 3, 66, 200)``, kiểu ``float32``,
            đã chuẩn hoá về ``[-1, 1]``.
        speed : torch.Tensor
            Vận tốc đã chuẩn hoá, shape ``(B,)``, kiểu ``float32``,
            trong ``[0, 1]``.
        command : torch.Tensor
            Chỉ số lệnh GPS, shape ``(B,)``, kiểu ``long`` (int64).
            Giá trị hợp lệ: 0, 1, 2, 3.

        Returns
        -------
        torch.Tensor
            Góc lái dự đoán, shape ``(B,)``, kiểu ``float32``.

        Ví dụ
        -----
        >>> model = CIL_NvidiaCNN().eval()
        >>> img  = torch.zeros(4, 3, 66, 200)
        >>> spd  = torch.tensor([0.3, 0.5, 0.2, 0.8])
        >>> cmd  = torch.tensor([0, 1, 2, 3])
        >>> out  = model(img, spd, cmd)
        >>> out.shape
        torch.Size([4])
        """
        # Visual features
        vis_feat = self.conv_layers(image)                        # (B, 1152)

        # Speed features
        spd_feat = self.speed_branch(speed.unsqueeze(1).float())  # (B, 32)

        # Concatenate
        features = torch.cat([vis_feat, spd_feat], dim=1)         # (B, 1184)

        # Apply all heads, then select the correct one per sample
        # head_outputs: (B, NUM_COMMANDS, 1)
        head_outputs = torch.stack(
            [head(features) for head in self.command_heads], dim=1
        )

        # Gather the output of the active command head for each sample
        cmd_idx = command.long().unsqueeze(1).unsqueeze(2)        # (B, 1, 1)
        steering = head_outputs.gather(1, cmd_idx).squeeze(1)     # (B, 1)

        return steering.squeeze(-1)                               # (B,)
