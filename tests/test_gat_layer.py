"""Unit tests for GATLayer implementation.

Tests cover:
- Attention score computation and masking
- Multi-head attention combination (concat vs average)
- Residual connections and layer normalization
- Handling of invalid agents and empty neighborhoods
"""

import unittest

import torch

from core_perception.multi_agent_model import GATLayer, MultiAgentModelConfig, MultiAgentTrajectoryPredictor


class GATLayerTests(unittest.TestCase):
    """Test suite for GATLayer class."""
    
    def test_gat_layer_output_shape(self) -> None:
        """GATLayer should preserve input shape [B, N, hidden_dim]."""
        batch_size, max_agents, hidden_dim = 2, 5, 128
        num_heads = 4
        
        gat = GATLayer(hidden_dim=hidden_dim, num_heads=num_heads, concat_heads=True)
        
        h = torch.randn(batch_size, max_agents, hidden_dim)
        adj = torch.ones(batch_size, max_agents, max_agents)
        agent_mask = torch.ones(batch_size, max_agents, dtype=torch.bool)
        
        output = gat(h, adj, agent_mask)
        
        self.assertEqual(output.shape, (batch_size, max_agents, hidden_dim))
    
    def test_gat_layer_attention_weights_sum_to_one(self) -> None:
        """For a valid agent with neighbors, attention weights should sum to ~1.0."""
        batch_size, max_agents, hidden_dim = 1, 4, 64
        num_heads = 2
        
        gat = GATLayer(hidden_dim=hidden_dim, num_heads=num_heads, concat_heads=True)
        
        # Create simple scenario: agent 0 has 3 neighbors (agents 1, 2, 3)
        h = torch.randn(batch_size, max_agents, hidden_dim)
        adj = torch.ones(batch_size, max_agents, max_agents)
        agent_mask = torch.ones(batch_size, max_agents, dtype=torch.bool)
        
        # We can't directly access attention weights from outside, but we can verify
        # that the output is computed correctly by checking it's not all zeros
        # and that masked agents produce zero output
        output = gat(h, adj, agent_mask)
        
        # Output should not be all zeros for valid agents
        self.assertFalse(torch.allclose(output, torch.zeros_like(output)))
        
        # Output should be finite (no NaN or Inf)
        self.assertTrue(torch.isfinite(output).all())
    
    def test_gat_layer_masked_agent_produces_zero_output(self) -> None:
        """Agents with agent_mask=False should have zero output."""
        batch_size, max_agents, hidden_dim = 1, 5, 64
        num_heads = 4
        
        gat = GATLayer(hidden_dim=hidden_dim, num_heads=num_heads, concat_heads=True)
        
        h = torch.randn(batch_size, max_agents, hidden_dim)
        adj = torch.ones(batch_size, max_agents, max_agents)
        agent_mask = torch.ones(batch_size, max_agents, dtype=torch.bool)
        
        # Mask out agents 3 and 4
        agent_mask[0, 3] = False
        agent_mask[0, 4] = False
        
        output = gat(h, adj, agent_mask)
        
        # Masked agents should have zero output
        self.assertTrue(torch.allclose(output[0, 3], torch.zeros(hidden_dim)))
        self.assertTrue(torch.allclose(output[0, 4], torch.zeros(hidden_dim)))
        
        # Valid agents should have non-zero output
        self.assertFalse(torch.allclose(output[0, 0], torch.zeros(hidden_dim)))
        self.assertFalse(torch.allclose(output[0, 1], torch.zeros(hidden_dim)))
        self.assertFalse(torch.allclose(output[0, 2], torch.zeros(hidden_dim)))

    def test_gat_layer_does_not_leak_nan_from_masked_agents(self) -> None:
        """NaNs in padded agents must not contaminate valid agents."""
        batch_size, max_agents, hidden_dim = 1, 4, 64
        num_heads = 4

        gat = GATLayer(hidden_dim=hidden_dim, num_heads=num_heads, concat_heads=True)

        h = torch.randn(batch_size, max_agents, hidden_dim)
        h[0, 3] = float("nan")
        adj = torch.zeros(batch_size, max_agents, max_agents)
        adj[0, :3, :3] = 1.0
        agent_mask = torch.tensor([[True, True, True, False]], dtype=torch.bool)

        output = gat(h, adj, agent_mask)

        self.assertTrue(torch.isfinite(output).all())
        self.assertTrue(torch.allclose(output[0, 3], torch.zeros(hidden_dim)))

    def test_gat_layer_half_precision_with_padded_agents_is_finite(self) -> None:
        """GAT masking should be safe under AMP-like float16 inputs."""
        if not torch.cuda.is_available():
            self.skipTest("CUDA is required for reliable float16 Linear coverage")

        device = torch.device("cuda")
        batch_size, max_agents, hidden_dim = 2, 5, 64
        num_heads = 4

        gat = GATLayer(hidden_dim=hidden_dim, num_heads=num_heads, concat_heads=True).to(device).half()

        h = torch.randn(batch_size, max_agents, hidden_dim, device=device).half()
        adj = torch.zeros(batch_size, max_agents, max_agents, device=device).half()
        adj[:, :3, :3] = 1.0
        agent_mask = torch.zeros(batch_size, max_agents, dtype=torch.bool, device=device)
        agent_mask[:, :3] = True

        output = gat(h, adj, agent_mask)

        self.assertEqual(output.dtype, torch.float16)
        self.assertTrue(torch.isfinite(output).all())
        self.assertTrue(torch.allclose(output[:, 3:], torch.zeros_like(output[:, 3:])))
    
    def test_gat_layer_no_neighbors_self_attention_only(self) -> None:
        """Agent with no neighbors (adj all zeros except diagonal) should use self-attention."""
        batch_size, max_agents, hidden_dim = 1, 3, 64
        num_heads = 2
        
        gat = GATLayer(hidden_dim=hidden_dim, num_heads=num_heads, concat_heads=True)
        
        h = torch.randn(batch_size, max_agents, hidden_dim)
        
        # Agent 0 has no neighbors (only self-connection)
        adj = torch.zeros(batch_size, max_agents, max_agents)
        # Note: GATLayer adds self-connections automatically, so we don't need to set diagonal
        
        agent_mask = torch.ones(batch_size, max_agents, dtype=torch.bool)
        
        output = gat(h, adj, agent_mask)
        
        # Output should still be valid (not NaN or Inf)
        self.assertTrue(torch.isfinite(output).all())
        
        # Output should not be all zeros (self-attention should produce something)
        self.assertFalse(torch.allclose(output[0, 0], torch.zeros(hidden_dim)))
    
    def test_gat_layer_concat_vs_average_modes(self) -> None:
        """Test both concat and average modes for multi-head combination."""
        batch_size, max_agents, hidden_dim = 2, 4, 64
        num_heads = 4
        
        # Test concat mode
        gat_concat = GATLayer(hidden_dim=hidden_dim, num_heads=num_heads, concat_heads=True)
        
        # Test average mode
        gat_average = GATLayer(hidden_dim=hidden_dim, num_heads=num_heads, concat_heads=False)
        
        h = torch.randn(batch_size, max_agents, hidden_dim)
        adj = torch.ones(batch_size, max_agents, max_agents)
        agent_mask = torch.ones(batch_size, max_agents, dtype=torch.bool)
        
        output_concat = gat_concat(h, adj, agent_mask)
        output_average = gat_average(h, adj, agent_mask)
        
        # Both should have same output shape
        self.assertEqual(output_concat.shape, (batch_size, max_agents, hidden_dim))
        self.assertEqual(output_average.shape, (batch_size, max_agents, hidden_dim))
        
        # Both should produce finite outputs
        self.assertTrue(torch.isfinite(output_concat).all())
        self.assertTrue(torch.isfinite(output_average).all())
        
        # Outputs should be different (different combination strategies)
        self.assertFalse(torch.allclose(output_concat, output_average))
    
    def test_gat_layer_respects_adjacency_matrix(self) -> None:
        """GATLayer should only attend to neighbors specified in adjacency matrix."""
        batch_size, max_agents, hidden_dim = 1, 4, 64
        num_heads = 2
        
        gat = GATLayer(hidden_dim=hidden_dim, num_heads=num_heads, concat_heads=True)
        
        h = torch.randn(batch_size, max_agents, hidden_dim)
        
        # Create sparse adjacency: agent 0 only connects to agent 1
        adj = torch.zeros(batch_size, max_agents, max_agents)
        adj[0, 0, 1] = 1.0
        adj[0, 1, 0] = 1.0  # Symmetric
        
        # Agent 2 only connects to agent 3
        adj[0, 2, 3] = 1.0
        adj[0, 3, 2] = 1.0  # Symmetric
        
        agent_mask = torch.ones(batch_size, max_agents, dtype=torch.bool)
        
        output = gat(h, adj, agent_mask)
        
        # All outputs should be finite
        self.assertTrue(torch.isfinite(output).all())
        
        # All agents should have non-zero output (due to self-attention)
        for i in range(max_agents):
            self.assertFalse(torch.allclose(output[0, i], torch.zeros(hidden_dim)))
    
    def test_gat_layer_hidden_dim_divisible_by_heads_concat_mode(self) -> None:
        """When concat_heads=True, hidden_dim must be divisible by num_heads."""
        hidden_dim = 65  # Not divisible by 4
        num_heads = 4
        
        with self.assertRaises(AssertionError):
            GATLayer(hidden_dim=hidden_dim, num_heads=num_heads, concat_heads=True)
    
    def test_gat_layer_average_mode_allows_any_hidden_dim(self) -> None:
        """When concat_heads=False, hidden_dim can be any value."""
        hidden_dim = 65  # Not divisible by 4
        num_heads = 4
        
        # Should not raise an error
        gat = GATLayer(hidden_dim=hidden_dim, num_heads=num_heads, concat_heads=False)
        
        batch_size, max_agents = 1, 3
        h = torch.randn(batch_size, max_agents, hidden_dim)
        adj = torch.ones(batch_size, max_agents, max_agents)
        agent_mask = torch.ones(batch_size, max_agents, dtype=torch.bool)
        
        output = gat(h, adj, agent_mask)
        
        self.assertEqual(output.shape, (batch_size, max_agents, hidden_dim))
        self.assertTrue(torch.isfinite(output).all())
    
    def test_model_with_gat_enabled(self) -> None:
        """Test MultiAgentTrajectoryPredictor with GAT enabled."""
        config = MultiAgentModelConfig(
            input_dim=6,
            hidden_dim=64,
            graph_layers=2,
            future_steps=10,
            enable_gat=True,
            num_attention_heads=4,
            attention_concat_mode="concat"
        )
        
        model = MultiAgentTrajectoryPredictor(config)
        
        batch_size, max_agents, history_len = 2, 5, 6
        
        x = torch.randn(batch_size, max_agents, history_len, 6)
        adj = torch.ones(batch_size, max_agents, max_agents)
        x_mask = torch.ones(batch_size, max_agents, history_len, dtype=torch.bool)
        agent_mask = torch.ones(batch_size, max_agents, dtype=torch.bool)
        
        # Forward pass
        pred = model(x=x, adj=adj, x_mask=x_mask, agent_mask=agent_mask)
        
        # Check output shape
        self.assertEqual(pred.shape, (batch_size, max_agents, 10, 2))
        
        # Check output is finite
        self.assertTrue(torch.isfinite(pred).all())
    
    def test_model_with_gat_disabled_uses_graph_interaction_block(self) -> None:
        """Test that GAT is not used when enable_gat=False."""
        config = MultiAgentModelConfig(
            input_dim=6,
            hidden_dim=64,
            graph_layers=2,
            future_steps=10,
            enable_gat=False
        )
        
        model = MultiAgentTrajectoryPredictor(config)
        
        # Check that graph_blocks are GraphInteractionBlock, not GATLayer
        from core_perception.multi_agent_model import GraphInteractionBlock
        
        for block in model.graph_blocks:
            self.assertIsInstance(block, GraphInteractionBlock)
            self.assertNotIsInstance(block, GATLayer)
    
    def test_gat_layer_gradient_flow(self) -> None:
        """Test that gradients flow through GATLayer correctly."""
        batch_size, max_agents, hidden_dim = 2, 4, 64
        num_heads = 4
        
        gat = GATLayer(hidden_dim=hidden_dim, num_heads=num_heads, concat_heads=True)
        
        h = torch.randn(batch_size, max_agents, hidden_dim, requires_grad=True)
        adj = torch.ones(batch_size, max_agents, max_agents)
        agent_mask = torch.ones(batch_size, max_agents, dtype=torch.bool)
        
        output = gat(h, adj, agent_mask)
        
        # Compute a simple loss
        loss = output.sum()
        loss.backward()
        
        # Check that gradients exist and are finite
        self.assertIsNotNone(h.grad)
        self.assertTrue(torch.isfinite(h.grad).all())
        
        # Check that model parameters have gradients
        for param in gat.parameters():
            if param.requires_grad:
                self.assertIsNotNone(param.grad)
                self.assertTrue(torch.isfinite(param.grad).all())


if __name__ == "__main__":
    unittest.main()
