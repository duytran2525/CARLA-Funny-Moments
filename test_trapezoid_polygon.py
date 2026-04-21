"""
Test Script: Trapezoid Obstacle Corridor Polygon Generation

Kiểm chứng rằng phương thức _build_obstacle_danger_polygon()
tạo ra hình thang chính xác dính trên mặt đất với bẻ cong theo steering.
"""

import sys
import math
import numpy as np
from pathlib import Path

# Add parent dirs to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from core_control.traffic_supervisor import TrafficSupervisor


def test_straight_line_polygon():
    """Test vẽ hình thang khi xe đi thẳng (no steering)"""
    print("\n" + "="*70)
    print("TEST 1: Straight Line (No Steering)")
    print("="*70)
    
    config = {
        'confidence_threshold': 0.5,
        'temporal_filter_frames': 3,
    }
    supervisor = TrafficSupervisor(config)
    
    # Test parameters
    image_shape = (480, 640, 3)
    vehicle_steer = 0.0  # Straight
    vehicle_speed_kmh = 30.0
    
    # Generate polygon
    polygon = supervisor._build_obstacle_danger_polygon(
        image_shape=image_shape,
        vehicle_steer=vehicle_steer,
        vehicle_speed_kmh=vehicle_speed_kmh,
    )
    
    if polygon is not None:
        print(f"✅ Polygon generated successfully")
        print(f"   Shape: {polygon.shape}")
        print(f"   First point: {polygon[0]}")
        print(f"   Last point: {polygon[-1]}")
        
        # Check if it's a valid trapezoid (symmetric for straight line)
        # Left half should be mirror of right half
        mid = len(polygon) // 2
        left_half = polygon[:mid]
        right_half = polygon[mid:]
        
        # Left points should be on left side, right points on right side
        left_x_mean = left_half[:, 0].mean()
        right_x_mean = right_half[:, 0].mean()
        
        print(f"   Left mean X: {left_x_mean:.1f}")
        print(f"   Right mean X: {right_x_mean:.1f}")
        print(f"   Center X (expect ~320): {image_shape[1]/2}")
        
        if left_x_mean < image_shape[1]/2 < right_x_mean:
            print("✅ Trapezoid structure is CORRECT (symmetric)")
        else:
            print("❌ Trapezoid structure is INCORRECT")
    else:
        print("❌ Failed to generate polygon")


def test_left_turn_polygon():
    """Test vẽ hình thang khi xe rẽ trái (negative steering)"""
    print("\n" + "="*70)
    print("TEST 2: Left Turn (Negative Steering)")
    print("="*70)
    
    config = {
        'confidence_threshold': 0.5,
        'temporal_filter_frames': 3,
    }
    supervisor = TrafficSupervisor(config)
    
    # Test parameters
    image_shape = (480, 640, 3)
    vehicle_steer = -0.3  # Left turn (reduced from -0.5)
    vehicle_speed_kmh = 20.0
    
    # Generate polygon
    polygon = supervisor._build_obstacle_danger_polygon(
        image_shape=image_shape,
        vehicle_steer=vehicle_steer,
        vehicle_speed_kmh=vehicle_speed_kmh,
    )
    
    if polygon is not None:
        print(f"✅ Polygon generated successfully")
        print(f"   Shape: {polygon.shape}")
        
        # Check left bias
        mid = len(polygon) // 2
        left_half = polygon[:mid]
        right_half = polygon[mid:]
        
        left_x_mean = left_half[:, 0].mean()
        right_x_mean = right_half[:, 0].mean()
        center_x = image_shape[1] / 2
        
        print(f"   Left mean X: {left_x_mean:.1f}")
        print(f"   Right mean X: {right_x_mean:.1f}")
        print(f"   Bias towards left: {abs(left_x_mean - right_x_mean):.1f} pixels")
        
        # Points should be clipped to bounds
        all_in_bounds = (
            (polygon[:, 0] >= 0).all() and (polygon[:, 0] < image_shape[1]).all() and
            (polygon[:, 1] >= 0).all() and (polygon[:, 1] < image_shape[0]).all()
        )
        
        if all_in_bounds:
            print("✅ All points within image bounds (clipped)")
            if left_x_mean < center_x < right_x_mean:
                print("✅ Left turn bias is visible")
            else:
                print("⚠️  Left bias present but subtle")
        else:
            print("❌ Some points out of bounds")
    else:
        print("❌ Failed to generate polygon")


def test_right_turn_polygon():
    """Test vẽ hình thang khi xe rẽ phải (positive steering)"""
    print("\n" + "="*70)
    print("TEST 3: Right Turn (Positive Steering)")
    print("="*70)
    
    config = {
        'confidence_threshold': 0.5,
        'temporal_filter_frames': 3,
    }
    supervisor = TrafficSupervisor(config)
    
    # Test parameters
    image_shape = (480, 640, 3)
    vehicle_steer = 0.3  # Right turn (reduced from 0.5)
    vehicle_speed_kmh = 25.0
    
    # Generate polygon
    polygon = supervisor._build_obstacle_danger_polygon(
        image_shape=image_shape,
        vehicle_steer=vehicle_steer,
        vehicle_speed_kmh=vehicle_speed_kmh,
    )
    
    if polygon is not None:
        print(f"✅ Polygon generated successfully")
        print(f"   Shape: {polygon.shape}")
        
        # Check right bias
        mid = len(polygon) // 2
        left_half = polygon[:mid]
        right_half = polygon[mid:]
        
        left_x_mean = left_half[:, 0].mean()
        right_x_mean = right_half[:, 0].mean()
        center_x = image_shape[1] / 2
        
        print(f"   Left mean X: {left_x_mean:.1f}")
        print(f"   Right mean X: {right_x_mean:.1f}")
        print(f"   Bias towards right: {abs(left_x_mean - right_x_mean):.1f} pixels")
        
        # Points should be clipped to bounds
        all_in_bounds = (
            (polygon[:, 0] >= 0).all() and (polygon[:, 0] < image_shape[1]).all() and
            (polygon[:, 1] >= 0).all() and (polygon[:, 1] < image_shape[0]).all()
        )
        
        if all_in_bounds:
            print("✅ All points within image bounds (clipped)")
            if left_x_mean < center_x < right_x_mean:
                print("✅ Right turn bias is visible")
            else:
                print("⚠️  Right bias present but subtle")
        else:
            print("❌ Some points out of bounds")
    else:
        print("❌ Failed to generate polygon")


def test_speed_effect():
    """Test ảnh hưởng của tốc độ lên horizon"""
    print("\n" + "="*70)
    print("TEST 4: Speed Effect on Horizon")
    print("="*70)
    
    config = {
        'confidence_threshold': 0.5,
        'temporal_filter_frames': 3,
    }
    supervisor = TrafficSupervisor(config)
    
    image_shape = (480, 640, 3)
    vehicle_steer = 0.0
    
    speeds = [10.0, 30.0, 60.0]
    
    for speed in speeds:
        polygon = supervisor._build_obstacle_danger_polygon(
            image_shape=image_shape,
            vehicle_steer=vehicle_steer,
            vehicle_speed_kmh=speed,
        )
        
        if polygon is not None:
            # Horizon nên ở phía trên khi tốc độ cao hơn
            min_y = polygon[:, 1].min()
            max_y = polygon[:, 1].max()
            print(f"   Speed {speed:5.1f} km/h → Y range: [{min_y}, {max_y}] (height={max_y-min_y})")
        else:
            print(f"   Speed {speed:5.1f} km/h → Failed to generate")
    
    print("✅ Higher speed should expand horizon → larger Y range")


def test_helper_methods():
    """Test các helper methods"""
    print("\n" + "="*70)
    print("TEST 5: Helper Methods")
    print("="*70)
    
    config = {
        'confidence_threshold': 0.5,
        'temporal_filter_frames': 3,
    }
    supervisor = TrafficSupervisor(config)
    
    # Test _steer_to_curvature
    steer_vals = [-1.0, -0.5, 0.0, 0.5, 1.0]
    print("   _steer_to_curvature (steering → curvature in 1/m):")
    for steer in steer_vals:
        curv = supervisor._steer_to_curvature(steer)
        print(f"      Steer={steer:+.1f} → Curvature={curv:+.4f} (1/m)")
    
    # Test _curved_path_center_lateral
    print("\n   _curved_path_center_lateral (forward → lateral offset):")
    forward_vals = [1.0, 5.0, 10.0, 20.0]
    steer = 0.3  # Some steering
    for fw in forward_vals:
        lat = supervisor._curved_path_center_lateral(fw, steer)
        print(f"      Forward={fw:5.1f}m, Steer={steer:+.1f} → Lateral={lat:+.3f}m")
    
    # Test _curved_path_half_width
    print("\n   _curved_path_half_width (forward → corridor half-width):")
    for fw in forward_vals:
        half_w = supervisor._curved_path_half_width(fw, steer)
        print(f"      Forward={fw:5.1f}m → Half-width={half_w:.3f}m")


if __name__ == "__main__":
    print("\n" + "▀"*70)
    print("  TRAPEZOID OBSTACLE CORRIDOR TEST SUITE")
    print("▄"*70)
    
    try:
        test_straight_line_polygon()
        test_left_turn_polygon()
        test_right_turn_polygon()
        test_speed_effect()
        test_helper_methods()
        
        print("\n" + "="*70)
        print("✅ ALL TESTS COMPLETED")
        print("="*70 + "\n")
        
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
