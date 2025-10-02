# CNN-based PPO for Timetable Scheduling

This directory contains a CNN (Convolutional Neural Network) implementation for the PPO (Proximal Policy Optimization) algorithm, specifically designed for timetable scheduling tasks.

## Overview

The CNN architecture is designed to better capture spatial relationships in the timetable grid compared to traditional MLP (Multi-Layer Perceptron) networks. This can be particularly beneficial for scheduling problems where the spatial arrangement of classes matters.

## Files

### Core CNN Implementation
- **`cnn_model.py`** - Main CNN architecture implementation
  - `TimetableCNNExtractor` - CNN feature extractor for timetable data
  - `TimetableCNNActorCritic` - Actor-critic network using CNN
  - `create_cnn_model()` - Function to create CNN-based PPO model

### Configuration and Hyperparameters
- **`cnn_hyperparams.py`** - CNN-optimized hyperparameters
  - CNN-specific learning rates, batch sizes, and training configurations
  - Comparison configurations for MLP vs CNN
  - Architecture size options (small, medium, large)

### Training Scripts
- **`cnn_train.py`** - CNN-specific training script
  - Async training function for CNN models
  - Progress tracking and logging
  - Integration with job management system

- **`compare_models.py`** - Model comparison script
  - Compare MLP vs CNN performance
  - Training and evaluation metrics
  - Results logging and analysis

### Testing and Utilities
- **`test_cnn.py`** - Simple test script for CNN implementation
- **`model.py`** - Updated to support both MLP and CNN models

## CNN Architecture Details

### Feature Extractor
The `TimetableCNNExtractor` treats the timetable as a 2D grid:
- **Input**: Flattened timetable state (num_slots × num_classrooms)
- **Reshape**: Convert to 2D grid (height × width)
- **Convolutional Layers**: 3 conv layers with batch normalization
  - Conv1: 1 → 32 channels, 3×3 kernel
  - Conv2: 32 → 64 channels, 3×3 kernel  
  - Conv3: 64 → 128 channels, 3×3 kernel
- **Fully Connected**: 512 → 256 features
- **Regularization**: Dropout layers for overfitting prevention

### Key Advantages
1. **Spatial Awareness**: Captures relationships between adjacent time slots and classrooms
2. **Translation Invariance**: Can recognize patterns regardless of exact position
3. **Parameter Efficiency**: Fewer parameters than equivalent MLP for large state spaces
4. **Better Generalization**: Often performs better on structured data like grids

## Usage

### Basic CNN Training
```python
from ppo.cnn_train import run_cnn_training
from ppo.cnn_hyperparams import get_config

# Get CNN configuration
config = get_config("cnn")

# Run training
await run_cnn_training(timetable_data, job_id)
```

### Model Comparison
```python
from ppo.compare_models import compare_models

# Compare MLP vs CNN performance
results = compare_models()
```

### Custom CNN Model Creation
```python
from ppo.model import create_cnn_ppo_model
from ppo.cnn_hyperparams import get_config

# Create CNN model with custom parameters
config = get_config("cnn")
model = create_cnn_ppo_model(env, **config["ppo_kwargs"])
```

## Hyperparameter Tuning

### CNN-Optimized Settings
- **Learning Rate**: 1e-4 (lower than MLP for stability)
- **Batch Size**: 128 (larger for better gradient estimates)
- **N Steps**: 2048 (larger buffer for CNN training)
- **N Epochs**: 10 (more epochs for CNN convergence)

### Architecture Sizes
- **Small**: 128 features, [16,32,64] conv channels
- **Medium**: 256 features, [32,64,128] conv channels (default)
- **Large**: 512 features, [64,128,256] conv channels

## Performance Considerations

### Memory Usage
- CNN models use more memory than MLP due to convolutional operations
- Recommended to use fewer parallel environments (n_envs=2 vs 4)
- Consider reducing batch size if memory issues occur

### Training Time
- CNN training is typically slower per step but may converge faster
- Use larger log intervals to reduce overhead
- Monitor GPU memory usage if available

### CPU Optimization
- Current implementation forces CPU usage to avoid GPU compatibility issues
- CNN operations are still efficient on modern CPUs
- Consider using PyTorch's CPU optimizations

## Integration with Existing System

### Job Management
The CNN training integrates with the existing job management system:
- Progress tracking via `update_progress()`
- Logging via `add_log()`
- Async training support

### Environment Compatibility
- Works with existing `TimetableEnv`
- Supports action masking for constraint satisfaction
- Compatible with vectorized environments

## Testing

Run the test script to verify implementation:
```bash
cd Scheduler/ppo
python test_cnn.py
```

Run model comparison:
```bash
python compare_models.py
```

## Expected Performance

### Advantages of CNN
- Better handling of spatial constraints
- Improved generalization to different timetable sizes
- More stable training on structured data
- Better performance on complex scheduling scenarios

### When to Use CNN vs MLP
- **Use CNN when**:
  - Timetable has clear spatial structure
  - You have sufficient training data
  - Spatial relationships are important
  - You want better generalization

- **Use MLP when**:
  - Simple scheduling problems
  - Limited computational resources
  - Quick prototyping needed
  - Small state spaces

## Troubleshooting

### Common Issues
1. **Memory errors**: Reduce batch size or number of environments
2. **Slow training**: Increase log interval, reduce evaluation frequency
3. **Import errors**: Ensure all dependencies are installed
4. **Model not loading**: Check file paths and model compatibility

### Performance Tips
1. Start with medium architecture size
2. Use default hyperparameters initially
3. Monitor training metrics closely
4. Compare with MLP baseline

## Future Improvements

1. **Attention Mechanisms**: Add attention layers for better focus
2. **Residual Connections**: Improve gradient flow in deep networks
3. **Multi-scale Features**: Capture patterns at different scales
4. **GPU Support**: Add proper GPU utilization when available
5. **Architecture Search**: Automated hyperparameter optimization
