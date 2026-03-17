import os
import pandas as pd
import torch
from torch.utils.data import Dataset
import cv2
import random
import numpy as np


class CarlaDataset(Dataset):
    """
    Dataset PyTorch cho bài toán Behavioral Cloning (Phase 1) với dữ liệu thu từ CARLA.

    Pipeline tổng quan
    ------------------
    Khởi tạo (``__init__``)
        │
        ├─► đọc CSV qua ``_prepare_data``
        │       ├─ duyệt 3 thư mục camera (center / left / right)
        │       ├─ áp steering correction cho camera trái/phải
        │       └─ nếu ``is_training=True`` → ``_balance_steering_distribution``
        │
    Lấy mẫu (``__getitem__``)
        │
        ├─► đọc ảnh jpg bằng OpenCV (BGR → RGB)
        ├─► crop phần trời (~45 % trên)
        ├─► nếu ``is_training=True`` → áp dụng ngẫu nhiên các phép augmentation:
        │       translate, brightness, shadow, flip, blur, noise, rotation, contrast
        ├─► resize về 200x66
        ├─► chuyển không gian màu RGB → YUV  (định dạng Nvidia CNN kỳ vọng)
        ├─► áp ``transform`` tuỳ chọn (thường là ``ToTensor``)
        └─► clamp steering về [-1, 1] và trả về ``(image_tensor, steering_tensor)``

    Cấu trúc thư mục dữ liệu kỳ vọng
    ----------------------------------
    ::

        root_dir/
        ├── images_center/   (ảnh từ camera chính giữa)
        ├── images_left/     (ảnh từ camera trái, steering + correction)
        └── images_right/    (ảnh từ camera phải, steering - correction)

    File CSV phải có header: ``img_id, steering, throttle, brake, speed``.
    ``img_id`` là tên file ảnh không có phần mở rộng, được pad thành 8 chữ số.

    Parameters
    ----------
    csv_file : str
        Đường dẫn đến file CSV chứa nhật ký lái (driving log).
    root_dir : str
        Thư mục gốc chứa ba thư mục con camera.
    transform : callable, optional
        Transform tùy chọn (ví dụ ``torchvision.transforms.ToTensor()``) áp lên
        ảnh sau khi đã xử lý.  Nếu ``None``, ảnh trả về là ``np.ndarray``.
    steering_correction : float, optional
        Độ bù góc lái cho camera trái (+correction) và phải (-correction).
        Mặc định 0.2 - giá trị này buộc xe "nhìn về giữa làn" khi học từ
        ảnh của camera lệch.
    is_training : bool, optional
        Nếu ``True`` (mặc định): dùng cả 3 camera và bật augmentation.
        Nếu ``False``: chỉ dùng camera chính giữa, tắt augmentation.
    """

    def __init__(self, csv_file, root_dir, transform=None,
                 steering_correction=0.2, is_training=True):
        """
        Khởi tạo CarlaDataset và xây dựng danh sách mẫu.

        Gọi ``_prepare_data()`` để duyệt file CSV, khớp từng dòng với
        file ảnh trên đĩa và (nếu là tập huấn luyện) cân bằng phân phối
        steering.

        Parameters
        ----------
        csv_file : str
            Đường dẫn file CSV driving log.
        root_dir : str
            Thư mục gốc chứa ``images_center/``, ``images_left/``,
            ``images_right/``.
        transform : callable, optional
            Transform áp lên ảnh sau pipeline chuẩn bị.
        steering_correction : float, optional
            Bù góc lái cho camera trái/phải.  Mặc định ``0.2``.
        is_training : bool, optional
            Kích hoạt augmentation và dùng 3 camera.  Mặc định ``True``.
        """
        self.data_df = pd.read_csv(
            csv_file, header=0,
            names=['img_id', 'steering', 'throttle', 'brake', 'speed', 'command']
        )
        self.root_dir = root_dir
        self.transform = transform
        self.steering_correction = steering_correction
        self.is_training = is_training
        self.samples = []
        self._prepare_data()

    def _prepare_data(self):
        """
        Xây dựng ``self.samples`` bằng cách khớp CSV với file ảnh trên đĩa.

        Với mỗi dòng trong CSV:
        * Chuẩn hoá ``img_id`` thành tên file 8 chữ số + ``.jpg``.
        * Thử khớp với **3 thư mục camera** (center / left / right):
          - Camera trái:  ``steering + steering_correction``
          - Camera phải: ``steering - steering_correction``
          - Khi ``is_training=False``: chỉ lấy ảnh center.
        * Nếu file tồn tại → thêm ``(img_path, steering_corrected)`` vào
          ``self.samples``.

        Sau khi duyệt xong, nếu là tập huấn luyện thì gọi
        ``_balance_steering_distribution()`` để cắt bớt mẫu đi thẳng.
        """
        print("Đang rà soát file ảnh từ 3 camera...")
        for _, row in self.data_df.iterrows():
            steering = float(row['steering'])
            img_id_str = str(row['img_id']).strip()
            if '.' in img_id_str:
                img_id_str = img_id_str.split('.')[0]  # Bỏ phần thập phân nếu có
            img_id = img_id_str.zfill(8) + '.jpg'

            configs = [
                ('images_center', 0),
                ('images_left',  self.steering_correction),
                ('images_right', -self.steering_correction),
            ]

            for sub_dir, correction in configs:
                if not self.is_training and sub_dir != 'images_center':
                    continue
                img_path = os.path.join(self.root_dir, sub_dir, img_id)
                if os.path.exists(img_path):
                    self.samples.append((img_path, steering + correction))

        print(f"✅ Hoàn tất! Tổng số mẫu hợp lệ: {len(self.samples)}")
        if self.is_training:
            self._balance_steering_distribution()

    def _balance_steering_distribution(self, bins=25, max_per_bin=None):
        """
        Giảm số lượng mẫu đi thẳng (steering ≈ 0) để cân bằng dữ liệu.

        Vấn đề cần giải quyết
        ---------------------
        Trong lái xe thực tế, xe đi thẳng chiếm phần lớn thời gian.  Nếu
        huấn luyện trực tiếp trên dữ liệu thô, mô hình sẽ thiên về dự đoán
        steering ≈ 0 vì đây là giá trị phổ biến nhất → xe dễ bị lao thẳng
        ở các khúc cua.

        Thuật toán
        ----------
        1. Chia trục steering thành ``bins`` khoảng đều nhau.
        2. Tính ``max_per_bin = mean(histogram)`` nếu không truyền vào.
        3. Duyệt ngẫu nhiên (sau ``shuffle``) danh sách mẫu:
           - Xác định bin của mẫu đó.
           - Nếu bin chưa đầy → giữ lại; ngược lại bỏ qua.
        4. Kết quả: mỗi bin có tối đa ``max_per_bin`` mẫu.

        Parameters
        ----------
        bins : int, optional
            Số khoảng để chia phân phối steering.  Mặc định ``25``.
        max_per_bin : int, optional
            Ngưỡng tối đa mẫu mỗi bin.  Nếu ``None`` thì dùng
            ``int(mean(histogram))``.
        """
        steerings = [s for _, s in self.samples]
        hist, bin_edges = np.histogram(steerings, bins=bins)

        if max_per_bin is None:
            max_per_bin = int(np.mean(hist))

        balanced_samples = []
        bin_counts = {i: 0 for i in range(bins)}

        random.shuffle(self.samples)

        for img_path, steering in self.samples:
            bin_idx = min(np.digitize(steering, bin_edges[1:-1]), bins - 1)
            if bin_counts[bin_idx] < max_per_bin:
                balanced_samples.append((img_path, steering))
                bin_counts[bin_idx] += 1

        print(f"Cân bằng dữ liệu: {len(self.samples)} → {len(balanced_samples)} mẫu")
        self.samples = balanced_samples

        print(f"Hoàn tất! Tổng số mẫu hợp lệ: {len(self.samples)}")

    def __len__(self):
        """Trả về tổng số mẫu trong dataset sau khi đã cân bằng."""
        return len(self.samples)

    # --- CÁC HÀM AUGMENTATION ---

    def _random_brightness(self, image):
        """
        Thay đổi độ sáng ngẫu nhiên bằng cách chuyển sang hệ màu HSV.

        Kênh V (Value - độ sáng) được nhân với một hệ số ngẫu nhiên trong
        khoảng [0.5, 1.5]:

        * Hệ số < 1 → ảnh tối hơn (mô phỏng ban đêm / bóng tối).
        * Hệ số > 1 → ảnh sáng hơn (mô phỏng nắng gắt / lóa đèn).

        Parameters
        ----------
        image : np.ndarray
            Ảnh đầu vào ở không gian màu RGB, shape ``(H, W, 3)``,
            dtype ``uint8``.

        Returns
        -------
        np.ndarray
            Ảnh sau khi điều chỉnh độ sáng, cùng shape và dtype với đầu vào.
        """
        hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
        # Tỉ lệ sáng từ 0.5 (tối đi một nửa) đến 1.5 (sáng rực)
        ratio = 1.0 + (random.random() - 0.5)
        hsv[:, :, 2] = np.clip(hsv[:, :, 2] * ratio, 0, 255)
        return cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)

    def _random_shadow(self, image):
        """
        Tạo bóng râm ngẫu nhiên trên mặt đường.

        Một đa giác 4 đỉnh được vẽ ngẫu nhiên trên mask, sau đó kênh V
        trong không gian màu HSV tại vùng đó được giảm xuống còn 50%.

        Mục đích: mô phỏng bóng của cây cối, tòa nhà hoặc xe khác đổ lên
        mặt đường, giúp mô hình không bị nhầm bóng tối là vật cản.

        Parameters
        ----------
        image : np.ndarray
            Ảnh RGB đầu vào, shape ``(H, W, 3)``, dtype ``uint8``.

        Returns
        -------
        np.ndarray
            Ảnh RGB sau khi thêm bóng râm, cùng shape với đầu vào.
        """
        h, w = image.shape[:2]
        # Tạo 4 điểm ngẫu nhiên để vẽ một đa giác (polygon) làm bóng râm
        x1, y1 = random.randint(0, w), 0
        x2, y2 = random.randint(0, w), h
        x3, y3 = random.randint(0, w), h
        x4, y4 = random.randint(0, w), 0

        pts = np.array([[x1, y1], [x2, y2], [x3, y3], [x4, y4]], np.int32)

        hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
        mask = np.zeros_like(hsv[:, :, 2])
        cv2.fillPoly(mask, [pts], 255)

        # Giảm độ sáng (kênh V) tại vùng có mask xuống 50%
        hsv[:, :, 2][mask == 255] = hsv[:, :, 2][mask == 255] * 0.5
        return cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)

    def _random_translate(self, image, steering, range_x=20, range_y=10):
        """
        Dịch chuyển ảnh ngẫu nhiên và bù trừ góc lái tương ứng.

        Xe trong ảnh dịch chuyển theo chiều ngang (±20 px) và dọc (±10 px).
        Với mỗi pixel dịch ngang, góc lái được điều chỉnh +0.004 rad để mô
        hình học cách phục hồi khi xe bị lệch khỏi tâm làn.

        * ``trans_x > 0`` (xe lệch phải) → ``steering`` tăng (đánh lái phải
          nhiều hơn để kéo xe về giữa).
        * ``trans_x < 0`` (xe lệch trái)  → ``steering`` giảm (đánh lái
          trái nhiều hơn).

        Parameters
        ----------
        image : np.ndarray
            Ảnh RGB đầu vào, shape ``(H, W, 3)``.
        steering : float
            Góc lái hiện tại (trước khi bù).
        range_x : int, optional
            Biên độ dịch ngang tối đa (pixel).  Mặc định ``20``.
        range_y : int, optional
            Biên độ dịch dọc tối đa (pixel).  Mặc định ``10``.

        Returns
        -------
        image : np.ndarray
            Ảnh sau khi dịch chuyển, cùng kích thước với đầu vào.
        steering : float
            Góc lái đã được bù trừ tương ứng với độ lệch ngang.
        """
        h, w = image.shape[:2]
        trans_x = range_x * (random.random() - 0.5)
        trans_y = range_y * (random.random() - 0.5)

        # Nếu xe bị lệch sang phải (trans_x > 0), phải đánh lái thêm về bên phải để đưa xe về giữa
        steering += trans_x * 0.004

        trans_m = np.float32([[1, 0, trans_x], [0, 1, trans_y]])
        image = cv2.warpAffine(image, trans_m, (w, h))
        return image, steering

    def _random_blur(self, image):
        """
        Thêm Gaussian blur ngẫu nhiên để mô phỏng mờ camera.

        Kernel 3x3 hoặc 5x5 được chọn ngẫu nhiên với xác suất bằng nhau.
        Blur mô phỏng các điều kiện thực tế: ống kính bẩn, rung xe, tốc độ
        cao khiến ảnh bị nhoè (motion blur nhẹ).

        Parameters
        ----------
        image : np.ndarray
            Ảnh RGB, shape ``(H, W, 3)``, dtype ``uint8``.

        Returns
        -------
        np.ndarray
            Ảnh sau khi làm mờ, cùng shape với đầu vào.
        """
        kernel_size = random.choice([3, 5])
        image = cv2.GaussianBlur(image, (kernel_size, kernel_size), 0)
        return image

    def _random_noise(self, image):
        """
        Thêm nhiễu Gaussian ngẫu nhiên để mô phỏng nhiễu sensor.

        Độ lệch chuẩn ``sigma`` được lấy ngẫu nhiên trong [5, 15].  Nhiễu
        được cộng thêm vào từng pixel rồi clamp về [0, 255].

        Mục đích: giúp mô hình bền vững trước các điều kiện ánh sáng kém,
        camera chất lượng thấp, hoặc nhiễu điện tử từ sensor.

        Parameters
        ----------
        image : np.ndarray
            Ảnh RGB, shape ``(H, W, 3)``, dtype ``uint8``.

        Returns
        -------
        np.ndarray
            Ảnh sau khi thêm nhiễu, dtype ``uint8``.
        """
        sigma = random.uniform(5, 15)
        noise = np.random.normal(0, sigma, image.shape).astype(np.float32)
        image = np.clip(image.astype(np.float32) + noise, 0, 255).astype(np.uint8)
        return image

    def _random_rotation(self, image, steering, max_angle=5):
        """
        Xoay ảnh nhẹ ngẫu nhiên và bù steering tương ứng.

        Góc xoay được chọn ngẫu nhiên trong [-max_angle, +max_angle] độ.
        Ảnh được xoay quanh tâm bằng ``cv2.warpAffine``.

        Quy ước bù steering:
        * Xoay ngược kim đồng hồ (``angle > 0``): ảnh nghiêng trái → phần
          đường phía trước lệch phải tương đối → cần lái ít hơn sang phải,
          tức ``steering`` giảm.
        * Xoay thuận kim đồng hồ (``angle < 0``): ngược lại.

        Parameters
        ----------
        image : np.ndarray
            Ảnh RGB, shape ``(H, W, 3)``.
        steering : float
            Góc lái hiện tại.
        max_angle : float, optional
            Góc xoay tối đa (độ).  Mặc định ``5``.

        Returns
        -------
        image : np.ndarray
            Ảnh sau khi xoay.
        steering : float
            Góc lái đã bù.
        """
        angle = random.uniform(-max_angle, max_angle)
        h, w = image.shape[:2]
        M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
        image = cv2.warpAffine(image, M, (w, h))
        # Xoay ảnh theo chiều kim đồng hồ (angle âm) ≈ xe nghiêng phải → cần lái trái
        steering -= angle * 0.01
        return image, steering

    def _random_contrast(self, image, low=0.7, high=1.3):
        """
        Thay đổi contrast ngẫu nhiên.

        Công thức: ``output = clip((pixel - mean) * factor + mean, 0, 255)``

        * ``factor > 1`` → tăng contrast (màu đậm hơn, sáng sáng hơn).
        * ``factor < 1`` → giảm contrast (ảnh bị "mờ phẳng").

        Mục đích: mô phỏng điều kiện thời tiết khác nhau (sương mù → contrast
        thấp; nắng gắt → contrast cao).

        Parameters
        ----------
        image : np.ndarray
            Ảnh RGB, shape ``(H, W, 3)``, dtype ``uint8``.
        low : float, optional
            Hệ số contrast thấp nhất.  Mặc định ``0.7``.
        high : float, optional
            Hệ số contrast cao nhất.  Mặc định ``1.3``.

        Returns
        -------
        np.ndarray
            Ảnh sau khi điều chỉnh contrast, dtype ``uint8``.
        """
        factor = random.uniform(low, high)
        mean = np.mean(image, axis=(0, 1), keepdims=True)
        image = np.clip((image - mean) * factor + mean, 0, 255).astype(np.uint8)
        return image
    def _random_cutout(self, image, min_size_ratio=0.1, max_size_ratio=0.3):
        """
        Kỹ thuật Cutout Augmentation: Che khuất một vùng chữ nhật ngẫu nhiên.
        
        Mục đích: Mô phỏng một vật cản (như xe tải ở khoảng cách 10-30m) 
        che mất một phần mặt đường, ép mạng CNN phải học cách nhìn các 
        vạch kẻ đường còn sót lại xung quanh để nội suy góc lái.
        
        Parameters
        ----------
        image : np.ndarray
            Ảnh RGB đầu vào.
        min_size_ratio : float
            Tỷ lệ cạnh nhỏ nhất của hình chữ nhật so với ảnh (mặc định 10%).
        max_size_ratio : float
            Tỷ lệ cạnh lớn nhất của hình chữ nhật so với ảnh (mặc định 30%).
        """
        h, w, c = image.shape

        # Random kích thước của "vật cản" (hình chữ nhật đen)
        cut_h = int(h * random.uniform(min_size_ratio, max_size_ratio))
        cut_w = int(w * random.uniform(min_size_ratio, max_size_ratio))

        # Random tọa độ tâm của hình chữ nhật
        # Cho phép hình chữ nhật nằm lấp ló ở các rìa ảnh
        cy = random.randint(0, h)
        cx = random.randint(0, w)

        # Tính tọa độ 4 góc, dùng np.clip để không bị văng lỗi out-of-bounds
        y1 = np.clip(cy - cut_h // 2, 0, h)
        y2 = np.clip(cy + cut_h // 2, 0, h)
        x1 = np.clip(cx - cut_w // 2, 0, w)
        x2 = np.clip(cx + cut_w // 2, 0, w)

        # Tô đen vùng bị cắt (Giá trị 0 tương đương màu đen / bóng tối)
        # Bạn cũng có thể dùng màu xám trung bình np.mean(image) nếu muốn
        image[y1:y2, x1:x2, :] = 0

        return image
    def __getitem__(self, idx):
        """
        Trả về một mẫu dữ liệu đã qua toàn bộ pipeline xử lý.

        Pipeline chi tiết
        -----------------
        1. **Đọc ảnh**: ``cv2.imread`` (BGR) → chuyển sang RGB.
        2. **Crop**: cắt bỏ ~45 % trên cùng của ảnh (trời + chân trời) - phần
           không chứa thông tin về làn đường.
        3. **Augmentation** (chỉ khi ``is_training=True``, mỗi bước xác suất
           độc lập):

           +---+----------------------------+----------+
           | # | Augmentation               | P(bật)   |
           +===+============================+==========+
           | 1 | Random translate           | 0.5      |
           +---+----------------------------+----------+
           | 2 | Random brightness          | 0.5      |
           +---+----------------------------+----------+
           | 3 | Random shadow              | 0.5      |
           +---+----------------------------+----------+
           | 4 | Horizontal flip            | 0.5      |
           +---+----------------------------+----------+
           | 5 | Gaussian blur              | 0.5      |
           +---+----------------------------+----------+
           | 6 | Gaussian noise             | 0.3      |
           +---+----------------------------+----------+
           | 7 | Random rotation            | 0.5      |
           +---+----------------------------+----------+
           | 8 | Random contrast            | 0.5      |
           +---+----------------------------+----------+

        4. **Resize**: 200x66 px (đúng kích thước đầu vào của Nvidia CNN).
        5. **Colour space**: RGB → YUV (Nvidia CNN được thiết kế cho YUV).
        6. **Transform**: áp ``self.transform`` nếu có (thường ``ToTensor``
           để ra tensor ``float32`` trong [0, 1]).
        7. **Steering clamp**: clip về [-1, 1] rồi bọc trong ``torch.Tensor``.

        Parameters
        ----------
        idx : int
            Chỉ số mẫu trong ``self.samples``.

        Returns
        -------
        image : torch.Tensor hoặc np.ndarray
            Ảnh đã xử lý.  Nếu ``transform`` là ``ToTensor`` thì shape là
            ``(3, 66, 200)``, dtype ``float32``.
        steering : torch.Tensor
            Góc lái scalar, dtype ``float32``, trong [-1, 1].
        """
        img_path, steering = self.samples[idx]

        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
        height, width, _ = image.shape
        image = image[int(height * 0.45):, :, :]

        # 2. DATA AUGMENTATION NGẪU NHIÊN
        if self.is_training:
            if random.random() > 0.5:
                image, steering = self._random_translate(image, steering)
            if random.random() > 0.5:
                image = self._random_brightness(image)
            if random.random() > 0.5:
                image = self._random_shadow(image)
            if random.random() > 0.5:
                image = cv2.flip(image, 1)
                steering = -steering
            # --- Augmentation MỚI ---
            if random.random() > 0.5:                                  # [MỚI #2]
                image = self._random_blur(image)
            if random.random() > 0.7:                                  # [MỚI #3] xác suất thấp hơn
                image = self._random_noise(image)
            if random.random() > 0.5:                                  # [MỚI #5]
                image, steering = self._random_rotation(image, steering)
            if random.random() > 0.5:                                  # [MỚI #8]
                image = self._random_contrast(image)
            if random.random() > 0.5:                         # Xác suất 50% xuất hiện vật cản che mắt
                image = self._random_cutout(image)

        image = cv2.resize(image, (200, 66))
        image = cv2.cvtColor(image, cv2.COLOR_RGB2YUV)
        if self.transform:
            image = self.transform(image)

        steering = torch.tensor(np.clip(steering, -1.0, 1.0), dtype=torch.float32)

        return image, steering


class CILCarlaDataset(CarlaDataset):
    """
    Dataset Phase 2 - Conditional Imitation Learning (CIL).

    Kế thừa ``CarlaDataset`` và mở rộng để trả về thêm **vận tốc đã chuẩn
    hoá** và **lệnh điều hướng cấp cao** bên cạnh ảnh và nhãn góc lái.

    Sự khác biệt so với ``CarlaDataset``
    -------------------------------------
    * ``__getitem__`` trả về ``(image, steering, speed, command)`` thay vì
      ``(image, steering)``.
    * ``_prepare_data`` lưu ``(img_path, steering, speed_norm, command)``
      trong mỗi phần tử của ``self.samples``.
    * ``_balance_steering_distribution`` được ghi đè để làm việc với
      tuple 4 phần tử thay vì 2.
    * Vận tốc được chuẩn hoá về [0, 1] bằng cách chia cho
      ``MAX_SPEED_KMH = 120``.

    Định dạng CSV
    -------------
    CSV phải có 6 cột: ``img_id, steering, throttle, brake, speed, command``.
    Nếu thiếu ``command`` hoặc ``speed``, một cảnh báo sẽ được phát và giá
    trị mặc định (0) được điền vào - cho phép tải trọng số Phase 1 mà không
    cần sửa file CSV.

    Lệnh điều hướng
    ---------------
    =====================  =====
    Lệnh                   Index
    =====================  =====
    Follow Lane            0
    Turn Left              1
    Turn Right             2
    Go Straight            3
    =====================  =====
    """

    # Maximum speed (km/h) used for normalisation. CARLA's default speed limit on
    # urban roads is 30-90 km/h; 120 km/h provides headroom for highway scenarios
    # while keeping normalised values comfortably in [0, 1].
    MAX_SPEED_KMH = 120.0

    def __init__(self, csv_file, root_dir, transform=None,
                 steering_correction=0.2, is_training=True):
        """
        Khởi tạo CILCarlaDataset.

        Trước khi gọi ``super().__init__``, đọc CSV bằng pandas để kiểm tra
        sự hiện diện của các cột ``speed`` và ``command``:

        * Nếu thiếu ``command`` → cảnh báo + điền giá trị mặc định ``0``
          (Follow Lane).
        * Nếu thiếu ``speed``   → cảnh báo + điền giá trị mặc định ``0.0``.

        DataFrame đã xử lý được lưu vào ``self._raw_df`` để phương thức
        ``_prepare_data`` (được gọi bên trong ``super().__init__``) có thể
        truy cập các cột bổ sung này.

        Parameters
        ----------
        csv_file : str
            Đường dẫn file CSV driving log (Phase 2).
        root_dir : str
            Thư mục gốc chứa ba thư mục con camera.
        transform : callable, optional
            Transform tuỳ chọn áp lên ảnh đã xử lý.
        steering_correction : float, optional
            Bù góc lái cho camera trái/phải.  Mặc định ``0.2``.
        is_training : bool, optional
            Bật/tắt augmentation và sử dụng 3 camera.  Mặc định ``True``.
        """
        # Read the extended CSV *before* calling super().__init__ so that the
        # speed/command columns are available when _prepare_data runs.
        raw_df = pd.read_csv(csv_file)
        if 'command' not in raw_df.columns:
            import warnings
            warnings.warn(
                "CSV file does not contain a 'command' column - defaulting all "
                "commands to 0 (Follow Lane). Add a 'command' column to enable "
                "full CIL training for Phase 2.",
                UserWarning, stacklevel=2,
            )
            raw_df['command'] = 0
        if 'speed' not in raw_df.columns:
            import warnings
            warnings.warn(
                "CSV file does not contain a 'speed' column - defaulting all "
                "speeds to 0.0. Add a 'speed' column to enable speed-conditioned "
                "CIL training for Phase 2.",
                UserWarning, stacklevel=2,
            )
            raw_df['speed'] = 0.0

        # Persist as an attribute that _prepare_data (inside super().__init__)
        # can also reference.
        self._raw_df = raw_df

        super().__init__(csv_file=csv_file, root_dir=root_dir,
                         transform=transform,
                         steering_correction=steering_correction,
                         is_training=is_training)

    def _prepare_data(self):
        """
        Xây dựng ``self.samples`` dưới dạng tuple 4 phần tử
        ``(img_path, steering, speed_norm, command)``.

        Tương tự ``CarlaDataset._prepare_data`` nhưng đọc thêm ``speed`` và
        ``command`` từ ``self._raw_df``.  Vận tốc được chuẩn hoá ngay tại
        đây bằng cách chia cho ``MAX_SPEED_KMH``.

        Thứ tự ưu tiên nguồn dữ liệu:

        1. ``self._raw_df`` - DataFrame đã có cột ``speed`` và ``command``
           (được tạo trong ``__init__`` của class này).
        2. ``self.data_df`` - fallback nếu ``_raw_df`` chưa được gán (không
           nên xảy ra trong luồng bình thường).
        """
        print("Đang rà soát file ảnh từ 3 camera (CIL mode)...")

        # Use the raw df that already has 'speed' and 'command'
        source_df = self._raw_df if hasattr(self, '_raw_df') else self.data_df

        self.samples = []  # reset in case called a second time
        for _, row in source_df.iterrows():
            steering = float(row['steering'])
            speed_norm = float(row.get('speed', 0.0)) / self.MAX_SPEED_KMH
            command = int(row.get('command', 0))

            img_id_str = str(row['img_id']).strip()
            if '.' in img_id_str:
                img_id_str = img_id_str.split('.')[0]
            img_id = img_id_str.zfill(8) + '.jpg'

            configs = [
                ('images_center', 0),
                ('images_left',  self.steering_correction),
                ('images_right', -self.steering_correction),
            ]

            for sub_dir, correction in configs:
                if not self.is_training and sub_dir != 'images_center':
                    continue
                img_path = os.path.join(self.root_dir, sub_dir, img_id)
                if os.path.exists(img_path):
                    self.samples.append(
                        (img_path, steering + correction, speed_norm, command)
                    )

        print(f"✅ Hoàn tất! Tổng số mẫu hợp lệ: {len(self.samples)}")
        if self.is_training:
            self._balance_steering_distribution()

    def _balance_steering_distribution(self, bins=25, max_per_bin=None):
        """
        Cân bằng phân phối steering cho các mẫu 4-tuple của CIL dataset.

        Ghi đè phiên bản cha vì mỗi phần tử trong ``self.samples`` là
        tuple ``(img_path, steering, speed_norm, command)`` thay vì
        tuple 2 phần tử.

        Thuật toán giống hệt ``CarlaDataset._balance_steering_distribution``:
        histogram → tính ``max_per_bin`` → shuffle → giữ lại tối đa
        ``max_per_bin`` mẫu mỗi bin.

        Parameters
        ----------
        bins : int, optional
            Số bin phân phối steering.  Mặc định ``25``.
        max_per_bin : int, optional
            Ngưỡng tối đa mẫu mỗi bin.  Nếu ``None`` dùng
            ``int(mean(histogram))``.
        """
        steerings = [s for _, s, _, _ in self.samples]
        hist, bin_edges = np.histogram(steerings, bins=bins)

        if max_per_bin is None:
            max_per_bin = int(np.mean(hist))

        balanced_samples = []
        bin_counts = {i: 0 for i in range(bins)}

        random.shuffle(self.samples)

        for sample in self.samples:
            steering = sample[1]
            bin_idx = min(np.digitize(steering, bin_edges[1:-1]), bins - 1)
            if bin_counts[bin_idx] < max_per_bin:
                balanced_samples.append(sample)
                bin_counts[bin_idx] += 1

        print(f"Cân bằng dữ liệu: {len(self.samples)} → {len(balanced_samples)} mẫu")
        self.samples = balanced_samples
        print(f"Hoàn tất! Tổng số mẫu hợp lệ: {len(self.samples)}")

    def __getitem__(self, idx):
        """
        Trả về một mẫu dữ liệu CIL đã qua toàn bộ pipeline xử lý.

        Pipeline xử lý ảnh giống hệt ``CarlaDataset.__getitem__`` (crop,
        augmentation, resize 200x66, chuyển YUV, transform).  Ngoài ra còn
        đóng gói **vận tốc** và **lệnh điều hướng** thành tensor.

        Parameters
        ----------
        idx : int
            Chỉ số mẫu trong ``self.samples``.

        Returns
        -------
        image : torch.Tensor hoặc np.ndarray
            Ảnh đã xử lý, shape ``(3, 66, 200)`` nếu dùng ``ToTensor``.
        steering : torch.Tensor
            Góc lái scalar, dtype ``float32``, trong [-1, 1].
        speed : torch.Tensor
            Vận tốc đã chuẩn hoá, scalar, dtype ``float32``, trong [0, 1].
        command : torch.Tensor
            Lệnh điều hướng, scalar, dtype ``long`` (int64).
            Giá trị hợp lệ: 0 (Follow Lane), 1 (Turn Left),
            2 (Turn Right), 3 (Go Straight).
        """
        img_path, steering, speed_norm, command = self.samples[idx]

        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        height, width, _ = image.shape
        image = image[int(height * 0.45):, :, :]

        if self.is_training:
            if random.random() > 0.5:
                image, steering = self._random_translate(image, steering)
            if random.random() > 0.5:
                image = self._random_brightness(image)
            if random.random() > 0.5:
                image = self._random_shadow(image)
            if random.random() > 0.5:
                image = cv2.flip(image, 1)
                steering = -steering
            if random.random() > 0.5:
                image = self._random_blur(image)
            if random.random() > 0.7:
                image = self._random_noise(image)
            if random.random() > 0.5:
                image, steering = self._random_rotation(image, steering)
            if random.random() > 0.5:
                image = self._random_contrast(image)
            if random.random() > 0.5: 
                image = self._random_cutout(image)

        image = cv2.resize(image, (200, 66))
        image = cv2.cvtColor(image, cv2.COLOR_RGB2YUV)
        if self.transform:
            image = self.transform(image)

        steering = torch.tensor(np.clip(steering, -1.0, 1.0), dtype=torch.float32)
        speed = torch.tensor(speed_norm, dtype=torch.float32)
        command = torch.tensor(command, dtype=torch.long)

        return image, steering, speed, command