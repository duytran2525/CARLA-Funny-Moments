# GTNet Visualization Notebooks

This directory contains Jupyter notebooks for visualizing and analyzing the GTNet multi-agent trajectory prediction model.

## Available Notebooks

### `gtnet_visualization.ipynb`

Comprehensive visualization notebook demonstrating:

1. **Model Loading**: Load trained models with GAT and multimodal prediction
2. **Inference**: Run predictions to get K=3 trajectory hypotheses
3. **Trajectory Visualization**: Plot ground truth vs. predicted trajectories
4. **Attention Weights**: Visualize which neighbors are most important (GAT)
5. **Mode Specialization**: Analyze how modes specialize to different patterns (left/straight/right)
6. **Scene Overview**: Multi-agent scene visualization with all predictions

**Requirements:** Requirement 12.10

## Setup

### Prerequisites

```bash
# Install Jupyter
pip install jupyter notebook

# Install visualization dependencies
pip install matplotlib seaborn

# Ensure project dependencies are installed
pip install torch numpy
```

### Running the Notebook

1. **Start Jupyter Notebook:**
   ```bash
   cd notebooks
   jupyter notebook
   ```

2. **Open `gtnet_visualization.ipynb`** in your browser

3. **Run all cells** (Cell → Run All)

## Using with Trained Models

### Training a Model

Before using the visualization notebook with real predictions, train a model:

```bash
# Collect data for Town01
python collect_multi_agent_data.py --town Town01 --duration 600

# Build dataset
python scripts/build_multi_agent_dataset.py \
  --csv data/multi_agent/raw/Town01_*.csv \
  --output data/multi_agent/Town01 \
  --adaptive-radius

# Train model
python scripts/train_multi_agent_trajectory.py \
  --data-dir data/multi_agent/Town01 \
  --town-filter Town01 \
  --enable-gat \
  --enable-multimodal \
  --enable-adaptive-radius \
  --epochs 50
```

### Using the Notebook

Once you have a trained model:

1. Update `MODEL_PATH` in the notebook to point to your checkpoint:
   ```python
   MODEL_PATH = project_root / "models" / "multi_agent" / "Town01" / "best_checkpoint.pt"
   ```

2. Update `SAMPLE_PATH` to point to a sample from your dataset:
   ```python
   SAMPLE_PATH = project_root / "data" / "multi_agent" / "Town01" / "samples" / "sample_0000.pt"
   ```

3. Run the notebook cells to visualize predictions

## Visualization Examples

### Trajectory Plot

Shows:
- Gray lines: Historical trajectories (observed)
- Green lines: Ground truth future trajectories
- Colored dashed lines: Predicted trajectories (K=3 modes)
- Blue dotted lines: Adjacency connections between agents

### Attention Heatmap

Shows:
- Rows: Query agents (who is attending)
- Columns: Key agents (who is being attended to)
- Color intensity: Attention weight (0 = no attention, 1 = high attention)
- Highlights which neighbors are most important for each agent

### Mode Specialization Analysis

Shows:
- **Lateral Deviation**: How modes differ in left/right movement
- **Longitudinal Progress**: How modes differ in forward movement
- **Final Position Distribution**: Where each mode predicts the agent will end up
- **Mode Statistics**: Quantitative analysis of mode characteristics

Expected patterns:
- **Mode 0**: Left turns (positive lateral deviation)
- **Mode 1**: Straight trajectories (near-zero lateral deviation)
- **Mode 2**: Right turns (negative lateral deviation)

### Scene Overview

Shows:
- All agents in the scene simultaneously
- Their historical trajectories
- Ground truth futures
- Predicted futures (all modes or best mode only)
- Adjacency connections showing which agents interact

## Demo Mode

The notebook includes a **demo mode** that works without trained models or datasets:

- Creates synthetic sample data
- Initializes a model with random weights
- Demonstrates all visualization functions

This is useful for:
- Understanding the visualization capabilities
- Testing the notebook setup
- Learning how to use the API

## Customization

### Changing the Focus Agent

To visualize a different agent:

```python
# Change agent_idx parameter
plot_trajectories(sample, predictions, agent_idx=2)
analyze_mode_specialization(predictions, sample, agent_idx=2)
```

### Showing Only Best Mode

To reduce clutter in multi-agent scenes:

```python
plot_scene_overview(sample, predictions, show_best_mode_only=True)
```

### Adjusting Colors

Modify the `mode_colors` list to change mode colors:

```python
mode_colors = ['#FF6B6B', '#4ECDC4', '#45B7D1']  # Red, Teal, Blue
# Or use your own colors:
mode_colors = ['red', 'blue', 'green']
```

## Troubleshooting

### "Model not found" Error

**Solution**: Either train a model or use demo mode (notebook will automatically create synthetic data)

### "Sample data not found" Error

**Solution**: Either build a dataset or use demo mode (notebook will automatically create synthetic data)

### Import Errors

**Solution**: Ensure you're running from the notebooks directory and the project root is in the Python path:

```python
import sys
from pathlib import Path
project_root = Path.cwd().parent
sys.path.insert(0, str(project_root))
```

### Visualization Issues

**Solution**: Ensure matplotlib backend is configured correctly:

```python
%matplotlib inline  # For inline plots in Jupyter
# or
%matplotlib notebook  # For interactive plots
```

## Advanced Usage

### Extracting Attention Weights

To extract actual learned attention weights (requires modifying GAT layer):

```python
# Modify GATLayer to store attention weights
class GATLayer(nn.Module):
    def forward(self, h, adj, agent_mask):
        # ... existing code ...
        self._last_attention_weights = alpha  # Store attention weights
        # ... rest of forward pass ...
```

Then in the notebook:

```python
attention_weights = extract_attention_weights(model, sample)
# attention_weights is a list of tensors, one per GAT layer
```

### Batch Processing

To visualize multiple samples:

```python
sample_paths = list((project_root / "data" / "multi_agent" / "Town01" / "samples").glob("sample_*.pt"))

for sample_path in sample_paths[:10]:  # First 10 samples
    sample = torch.load(sample_path)
    # ... process and visualize ...
```

### Exporting Visualizations

To save figures:

```python
fig = plot_trajectories(sample, predictions, agent_idx=0)
fig.savefig('trajectory_visualization.png', dpi=300, bbox_inches='tight')
```

## References

- **GTNet Improvements Spec**: `.kiro/specs/gtnet-improvements/`
- **Model Implementation**: `core_perception/multi_agent_model.py`
- **Training Scripts**: `scripts/train_multi_agent_trajectory.py`
- **Dataset Builder**: `scripts/build_multi_agent_dataset.py`

## Contributing

To add new visualizations:

1. Create a new cell in the notebook
2. Define a visualization function
3. Call it with sample data
4. Document the visualization in this README

Example:

```python
def plot_velocity_profiles(sample, predictions):
    """Plot velocity profiles over time for each mode."""
    # ... implementation ...
    
# Use it
fig = plot_velocity_profiles(sample, predictions)
plt.show()
```
