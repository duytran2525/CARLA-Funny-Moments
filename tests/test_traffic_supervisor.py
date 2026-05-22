import unittest

from core_control.traffic_supervisor import TrafficSupervisor


IMAGE_SHAPE = (480, 640, 3)


def red_light(distance_m=20.0, confidence=0.9):
    return {
        "class_name": "traffic_light_red",
        "confidence": confidence,
        "bbox": (300, 50, 40, 60),
        "distance_m": distance_m,
    }


def green_light(distance_m=20.0, confidence=0.95):
    return {
        "class_name": "traffic_light_green",
        "confidence": confidence,
        "bbox": (300, 50, 40, 60),
        "distance_m": distance_m,
    }


def stop_line(distance_m=17.0, confidence=0.85):
    return {
        "class_name": "stop_line",
        "confidence": confidence,
        "bbox": (250, 350, 120, 12),
        "distance_m": distance_m,
    }


class TrafficSupervisorRedLightTests(unittest.TestCase):
    def test_stop_line_detection_creates_limited_crawl_brake_without_light(self):
        supervisor = TrafficSupervisor({})

        brake = supervisor.compute(
            detections=[stop_line(17.0)],
            current_speed=30.0 / 3.6,
            image_shape=IMAGE_SHAPE,
            dt=0.1,
        )

        self.assertGreater(brake, 0.0)
        self.assertLessEqual(brake, 0.45)
        debug = supervisor.get_debug_info()
        self.assertEqual(debug["selected_target_type"], "stop_line_crawl")
        self.assertEqual(debug["stop_line_crawl_mode"], "active")

    def test_green_light_releases_stop_line_crawl(self):
        supervisor = TrafficSupervisor({})

        brake = supervisor.compute(
            detections=[green_light(), stop_line(17.0)],
            current_speed=30.0 / 3.6,
            image_shape=IMAGE_SHAPE,
            dt=0.1,
        )

        self.assertEqual(brake, 0.0)
        debug = supervisor.get_debug_info()
        self.assertTrue(debug["green_release_active"])
        self.assertEqual(debug["selected_target_type"], "none")
        self.assertEqual(debug["stop_line_crawl_mode"], "disabled_by_green")

    def test_red_light_approach_brakes_even_when_already_slow(self):
        supervisor = TrafficSupervisor({})

        brake = supervisor.compute(
            detections=[red_light(), stop_line(10.0)],
            current_speed=12.0 / 3.6,
            image_shape=IMAGE_SHAPE,
            dt=0.1,
        )

        self.assertGreater(brake, 0.0)
        debug = supervisor.get_debug_info()
        self.assertEqual(debug["selected_target_type"], "stop_line")
        self.assertGreater(debug["red_stopline_approach_brake"], 0.0)
        self.assertNotEqual(debug["stop_line_crawl_mode"], "below_target_speed")

    def test_red_light_hard_stops_at_tracked_seven_meter_gate(self):
        supervisor = TrafficSupervisor({})

        first_brake = supervisor.compute(
            detections=[red_light(), stop_line(8.0)],
            current_speed=20.0 / 3.6,
            image_shape=IMAGE_SHAPE,
            dt=0.1,
        )
        self.assertGreater(first_brake, 0.0)
        self.assertLess(first_brake, 1.0)

        second_brake = supervisor.compute(
            detections=[red_light()],
            current_speed=20.0 / 3.6,
            image_shape=IMAGE_SHAPE,
            dt=0.25,
        )

        self.assertEqual(second_brake, 1.0)
        debug = supervisor.get_debug_info()
        self.assertEqual(debug["selected_target_type"], "stop_line")
        self.assertTrue(debug["red_hard_stop_locked"])
        self.assertFalse(debug["stop_line_observed_this_frame"])

    def test_urban_green_releases_red_hard_stop(self):
        supervisor = TrafficSupervisor({})
        supervisor.compute(
            detections=[red_light(), stop_line(6.5)],
            current_speed=5.0 / 3.6,
            image_shape=IMAGE_SHAPE,
            dt=0.1,
        )
        self.assertTrue(supervisor.get_debug_info()["red_hard_stop_locked"])

        released_brake = supervisor.compute(
            detections=[green_light(), stop_line(6.0)],
            current_speed=0.0,
            image_shape=IMAGE_SHAPE,
            dt=0.1,
        )

        self.assertEqual(released_brake, 0.0)
        debug = supervisor.get_debug_info()
        self.assertTrue(debug["green_release_active"])
        self.assertFalse(debug["red_hard_stop_locked"])
        self.assertFalse(debug["red_hard_stop_active"])

    def test_red_light_does_not_late_lock_after_stop_line_already_passed(self):
        supervisor = TrafficSupervisor({})

        brake = supervisor.compute(
            detections=[red_light(), stop_line(0.1)],
            current_speed=12.0 / 3.6,
            image_shape=IMAGE_SHAPE,
            dt=0.1,
        )

        self.assertEqual(brake, 0.0)
        debug = supervisor.get_debug_info()
        self.assertFalse(debug["red_hard_stop_locked"])
        self.assertEqual(debug["selected_target_type"], "none")

    def test_red_light_near_stop_line_overrides_stale_green_immunity(self):
        supervisor = TrafficSupervisor({})
        supervisor.compute(
            detections=[green_light(), stop_line(16.0)],
            current_speed=20.0 / 3.6,
            image_shape=IMAGE_SHAPE,
            dt=0.1,
        )
        self.assertGreater(supervisor.get_debug_info()["green_immunity_counter"], 0)

        brake = supervisor.compute(
            detections=[red_light(), stop_line(12.0)],
            current_speed=20.0 / 3.6,
            image_shape=IMAGE_SHAPE,
            dt=0.1,
        )

        self.assertGreater(brake, 0.0)
        debug = supervisor.get_debug_info()
        self.assertTrue(debug["red_green_immunity_overridden"])
        self.assertFalse(debug["red_suppressed_by_green_immunity"])
        self.assertEqual(debug["selected_target_type"], "stop_line")


if __name__ == "__main__":
    unittest.main()
