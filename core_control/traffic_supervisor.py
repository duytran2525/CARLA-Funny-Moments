from enum import Enum
from typing import List, Tuple, Dict, Any

# 🌟 [CẬP NHẬT] Enum định nghĩa trạng thái giao thông
class TrafficState(Enum):
    CAN_GO = 0
    MUST_STOP = 1

# 🌟 [CẬP NHẬT] Class giám sát luật lệ giao thông
class TrafficSupervisor:
    def __init__(self, config) -> None:
        # Nạp cấu hình ROI từ đối tượng config
        self.roi_max_dist = getattr(config, 'red_light_max_distance', 15.0)
        self.roi_width_left = getattr(config, 'roi_width_left', 2.0)
        self.roi_width_right = getattr(config, 'roi_width_right', 4.0)
        self.green_immunity_frames = getattr(config, 'green_immunity_frames', 50)
        
        # Biến đếm miễn nhiễm (tránh phanh lặp lại khi vừa qua đèn xanh)
        self._current_immunity_counter = 0

    def evaluate(self, tracks: List[Any]) -> Tuple[TrafficState, Dict[str, Any]]:
        """
        Đánh giá đèn giao thông dựa trên danh sách Track đã có tọa độ world_x, world_y.
        """
        # 1. Giảm bộ đếm immunity (nếu đang kích hoạt)
        if self._current_immunity_counter > 0:
            self._current_immunity_counter -= 1

        red_light_in_roi = False
        green_light_in_roi = False
        closest_red_dist = float('inf')

        # 2 & 3. Quét các track để tìm đèn giao thông trong ROI (Vùng Quan Tâm)
        for track in tracks:
            # Dựa vào mô hình YOLO cũ: 4 = traffic_light_red, 5 = traffic_light_green
            if getattr(track, 'class_id', -1) not in [4, 5]:
                continue
            
            # Kiểm tra xem đèn có nằm trong hộp ROI không:
            # X: 0 đến roi_max_dist (phía trước xe)
            # Y: -roi_width_left đến roi_width_right (bề ngang)
            in_roi = (
                0 < getattr(track, 'world_x', 0.0) <= self.roi_max_dist and 
                -self.roi_width_left <= getattr(track, 'world_y', 0.0) <= self.roi_width_right
            )
            
            if in_roi:
                if track.class_id == 4:  # Đèn Đỏ
                    red_light_in_roi = True
                    if track.world_x < closest_red_dist:
                        closest_red_dist = track.world_x
                elif track.class_id == 5:  # Đèn Xanh
                    green_light_in_roi = True

        # 4. Áp dụng Logic Miễn nhiễm (Immunity)
        # Nếu thấy đèn xanh trong ROI, nạp lại bộ đếm immunity để xe đi dứt khoát qua ngã tư
        if green_light_in_roi:
            self._current_immunity_counter = self.green_immunity_frames

        debug_info = {
            "red_in_roi": red_light_in_roi,
            "closest_red_dist": closest_red_dist if red_light_in_roi else -1,
            "immunity_counter": self._current_immunity_counter
        }

        # 5. Đưa ra quyết định cuối cùng
        # Nếu có đèn đỏ trong ROI VÀ không có tấm khiên miễn nhiễm từ đèn xanh trước đó
        if red_light_in_roi and self._current_immunity_counter <= 0:
            return TrafficState.MUST_STOP, debug_info
        
        return TrafficState.CAN_GO, debug_info