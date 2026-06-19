"""
Unit tests for collect_multi_agent_data.py
"""

import csv
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, Mock, patch

import numpy as np

# Mock carla module before importing
import sys
sys.modules['carla'] = MagicMock()

from collect_multi_agent_data import (
    ActorState,
    FrameData,
    MultiAgentCSVWriter,
    SUPPORTED_TOWNS,
    MIN_NPC_VEHICLES,
    MAX_NPC_VEHICLES,
    COLLECTION_FPS,
)


class TestActorState(unittest.TestCase):
    """Test ActorState dataclass."""
    
    def test_actor_state_creation(self):
        """Test creating ActorState instance."""
        state = ActorState(
            actor_id=123,
            actor_type="vehicle.tesla.model3",
            x=100.5,
            y=200.3,
            z=0.5,
            vx=10.0,
            vy=5.0,
            yaw=1.57,
        )
        
        self.assertEqual(state.actor_id, 123)
        self.assertEqual(state.actor_type, "vehicle.tesla.model3")
        self.assertAlmostEqual(state.x, 100.5)
        self.assertAlmostEqual(state.y, 200.3)
        self.assertAlmostEqual(state.z, 0.5)
        self.assertAlmostEqual(state.vx, 10.0)
        self.assertAlmostEqual(state.vy, 5.0)
        self.assertAlmostEqual(state.yaw, 1.57)


class TestFrameData(unittest.TestCase):
    """Test FrameData dataclass."""
    
    def test_frame_data_creation(self):
        """Test creating FrameData instance."""
        ego_state = ActorState(1, "vehicle.tesla.model3", 0, 0, 0, 0, 0, 0)
        npc_states = [
            ActorState(2, "vehicle.audi.a2", 10, 10, 0, 5, 5, 0),
            ActorState(3, "vehicle.bmw.grandtourer", 20, 20, 0, 8, 8, 0),
        ]
        
        frame_data = FrameData(
            frame=100,
            timestamp=10.0,
            ego_state=ego_state,
            npc_states=npc_states,
        )
        
        self.assertEqual(frame_data.frame, 100)
        self.assertAlmostEqual(frame_data.timestamp, 10.0)
        self.assertEqual(frame_data.ego_state.actor_id, 1)
        self.assertEqual(len(frame_data.npc_states), 2)


class TestMultiAgentCSVWriter(unittest.TestCase):
    """Test MultiAgentCSVWriter class."""
    
    def setUp(self):
        """Create temporary directory for test outputs."""
        self.temp_dir = tempfile.mkdtemp()
        self.output_path = Path(self.temp_dir) / "test_output.csv"
        
    def test_csv_writer_initialization(self):
        """Test CSV writer initialization."""
        writer = MultiAgentCSVWriter(
            output_path=self.output_path,
            run_id="test_run_123",
            town="Town01",
        )
        
        self.assertEqual(writer.output_path, self.output_path)
        self.assertEqual(writer.run_id, "test_run_123")
        self.assertEqual(writer.town, "Town01")
        self.assertEqual(writer.rows_written, 0)
        
    def test_csv_writer_start_creates_file(self):
        """Test that start() creates CSV file with header."""
        writer = MultiAgentCSVWriter(
            output_path=self.output_path,
            run_id="test_run_123",
            town="Town01",
        )
        
        writer.start()
        
        # Check file exists
        self.assertTrue(self.output_path.exists())
        
        # Check header
        with self.output_path.open("r") as f:
            reader = csv.DictReader(f)
            self.assertEqual(list(reader.fieldnames), writer.FIELDNAMES)
            
        writer.close()
        
    def test_csv_writer_write_frame(self):
        """Test writing frame data to CSV."""
        writer = MultiAgentCSVWriter(
            output_path=self.output_path,
            run_id="test_run_123",
            town="Town01",
        )
        
        writer.start()
        
        # Create test frame data
        ego_state = ActorState(
            actor_id=1,
            actor_type="vehicle.tesla.model3",
            x=0.0,
            y=0.0,
            z=0.0,
            vx=10.0,
            vy=0.0,
            yaw=0.0,
        )
        
        npc_states = [
            ActorState(
                actor_id=2,
                actor_type="vehicle.audi.a2",
                x=10.0,
                y=5.0,
                z=0.0,
                vx=8.0,
                vy=2.0,
                yaw=0.5,
            ),
            ActorState(
                actor_id=3,
                actor_type="vehicle.bmw.grandtourer",
                x=20.0,
                y=10.0,
                z=0.0,
                vx=12.0,
                vy=3.0,
                yaw=1.0,
            ),
        ]
        
        frame_data = FrameData(
            frame=1,
            timestamp=0.1,
            ego_state=ego_state,
            npc_states=npc_states,
        )
        
        writer.write_frame(frame_data)
        writer.close()
        
        # Verify CSV content
        with self.output_path.open("r") as f:
            reader = csv.DictReader(f)
            rows = list(reader)
            
        # Should have 2 rows (one per NPC)
        self.assertEqual(len(rows), 2)
        
        # Check first row
        row1 = rows[0]
        self.assertEqual(row1["run_id"], "test_run_123")
        self.assertEqual(row1["town"], "Town01")
        self.assertEqual(row1["frame"], "1")
        self.assertEqual(row1["ego_id"], "1")
        self.assertEqual(row1["actor_id"], "2")
        self.assertEqual(row1["actor_type"], "vehicle.audi.a2")
        
        # Check distance calculation
        distance = float(row1["distance_m"])
        expected_distance = np.sqrt(10.0**2 + 5.0**2)
        self.assertAlmostEqual(distance, expected_distance, places=2)
        
    def test_csv_writer_rows_written_counter(self):
        """Test that rows_written counter is accurate."""
        writer = MultiAgentCSVWriter(
            output_path=self.output_path,
            run_id="test_run_123",
            town="Town01",
        )
        
        writer.start()
        
        ego_state = ActorState(1, "vehicle.tesla.model3", 0, 0, 0, 0, 0, 0)
        
        # Write 3 frames with 2 NPCs each
        for frame_num in range(3):
            npc_states = [
                ActorState(2, "vehicle.audi.a2", 10, 10, 0, 5, 5, 0),
                ActorState(3, "vehicle.bmw.grandtourer", 20, 20, 0, 8, 8, 0),
            ]
            
            frame_data = FrameData(
                frame=frame_num,
                timestamp=frame_num * 0.1,
                ego_state=ego_state,
                npc_states=npc_states,
            )
            
            writer.write_frame(frame_data)
            
        writer.close()
        
        # Should have written 6 rows (3 frames * 2 NPCs)
        self.assertEqual(writer.rows_written, 6)


class TestConstants(unittest.TestCase):
    """Test module constants."""
    
    def test_supported_towns(self):
        """Test that all required towns are supported."""
        required_towns = [
            "Town01", "Town02", "Town03", "Town04", "Town05",
            "Town06", "Town07", "Town10HD"
        ]
        
        for town in required_towns:
            self.assertIn(town, SUPPORTED_TOWNS)
            
    def test_npc_vehicle_range(self):
        """Test NPC vehicle count range."""
        self.assertEqual(MIN_NPC_VEHICLES, 30)
        self.assertEqual(MAX_NPC_VEHICLES, 100)
        
    def test_collection_fps(self):
        """Test collection FPS is 10."""
        self.assertEqual(COLLECTION_FPS, 10)


if __name__ == "__main__":
    unittest.main()
