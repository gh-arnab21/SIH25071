# Training Script Documentation

## Overview

The enhanced training script (`scripts/train_model.py`) provides a comprehensive command-line interface for training the Rockfall Prediction Model ensemble. It includes experiment management, configuration validation, advanced logging, and model comparison capabilities.

## Features

- **Comprehensive CLI**: Grouped command-line arguments for different aspects of training
- **Configuration Management**: YAML-based configuration with validation and CLI overrides
- **Experiment Tracking**: Automatic experiment directory creation and metadata logging
- **Enhanced Logging**: Training metrics, progress tracking, and comprehensive logging system
- **Model Comparison**: Built-in utilities for comparing different model configurations
- **Cross-Validation**: Built-in support for k-fold cross-validation
- **Hyperparameter Optimization**: Optional hyperparameter tuning capabilities
- **Resume Training**: Support for resuming training from checkpoints

## Quick Start

### Basic Training

```bash
# Train with default configuration
python scripts/train_model.py --experiment-name my_experiment

# Train with custom epochs and batch size
python scripts/train_model.py \
    --experiment-name custom_training \
    --epochs 50 \
    --batch-size 64
```

### Dry Run

```bash
# Test configuration without running actual training
python scripts/train_model.py --dry-run --experiment-name test_run
```

## Command Line Arguments

### Configuration Arguments

- `--config, -c`: Path to configuration file (default: `config/default_config.yaml`)
- `--data-dir, -d`: Override data directory from config
- `--output-dir, -o`: Override output directory from config
- `--dataset-type`: Dataset type (`open_pit_mine`, `rocknet_seismic`, `brazilian_rockfall`, `auto`)

### Experiment Management

- `--experiment-name, -n`: Name for the training experiment
- `--tags`: Tags to associate with the experiment
- `--description`: Description of the experiment

### Training Parameters

- `--epochs, -e`: Number of training epochs
- `--batch-size, -b`: Batch size for training
- `--learning-rate, -lr`: Learning rate

### Cross-Validation and Optimization

- `--cv-folds, -k`: Number of cross-validation folds
- `--optimize-hyperparams`: Enable hyperparameter optimization
- `--optimization-trials`: Number of optimization trials (default: 100)

### Model Management

- `--resume-from`: Path to checkpoint to resume training from
- `--save-best-only`: Save only the best model during training

### Evaluation and Comparison

- `--evaluate-only`: Only run evaluation on existing model
- `--compare-models`: Paths to models to compare

### Logging and Monitoring

- `--log-level`: Logging level (`DEBUG`, `INFO`, `WARNING`, `ERROR`)
- `--debug`: Enable debug mode with additional logging

### Miscellaneous

- `--seed`: Random seed for reproducibility (default: 42)
- `--dry-run`: Perform a dry run without actual training
- `--force`: Force overwrite existing experiment directory

## Configuration System

### Configuration File Structure

The training script uses YAML configuration files with the following structure:

```yaml
# Data configuration
data:
  raw_data_dir: "data/raw"
  processed_data_dir: "data/processed"
  models_dir: "models/saved"

# Model configuration
model:
  image:
    input_size: [224, 224]
    batch_size: 32
    augmentation: true
    normalization:
      mean: [0.485, 0.456, 0.406]
      std: [0.229, 0.224, 0.225]
  
  cnn:
    backbone: "resnet50"
    pretrained: true
    freeze_backbone: false
    feature_dim: 2048
  
  lstm:
    hidden_size: 128
    num_layers: 2
    dropout: 0.2
    sequence_length: 50
  
  ensemble:
    algorithms: ["random_forest", "xgboost", "neural_network"]
    voting_method: "soft"
    use_stacking: true

# Training configuration
training:
  validation_split: 0.15
  test_split: 0.15
  cross_validation_folds: 5
  early_stopping_patience: 10
  max_epochs: 100
  batch_size: 32

# Preprocessing configuration
preprocessing:
  imputation:
    numerical_strategy: "median"
    categorical_strategy: "mode"
  
  outlier_detection:
    method: "iqr"
    threshold: 1.5
  
  class_balance:
    method: "smote"
    sampling_strategy: "auto"
  
  scaling:
    method: "standard"

# Evaluation configuration
evaluation:
  metrics: ["precision", "recall", "f1_score", "roc_auc"]
  average: "weighted"

# Logging configuration
logging:
  level: "INFO"
  format: "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
  file: "logs/rockfall_prediction.log"
```

### Configuration Validation

The script automatically validates configuration files against a predefined schema:

```bash
# Validate configuration manually
python -m src.utils.config_validation config/default_config.yaml
```

### Command Line Overrides

Configuration values can be overridden via command line:

```bash
python scripts/train_model.py \
    --config config/custom_config.yaml \
    --epochs 100 \
    --batch-size 64 \
    --learning-rate 0.001
```

## Experiment Management

### Experiment Directory Structure

Each training run creates an organized experiment directory:

```
experiments/
└── my_experiment_20250101_120000/
    ├── metadata.json          # Experiment metadata
    ├── config.yaml           # Final configuration used
    ├── logs/                 # Training logs
    │   ├── training.log
    │   ├── training_metrics.jsonl
    │   └── checkpoints.jsonl
    ├── models/               # Saved models
    │   ├── best_model.pkl
    │   └── checkpoints/
    ├── results/              # Results and visualizations
    │   ├── model_comparison.csv
    │   ├── confusion_matrices.png
    │   └── training_curves.png
    └── artifacts/            # Additional artifacts
```

### Experiment Metadata

Each experiment automatically saves comprehensive metadata:

```json
{
  "experiment_name": "my_experiment",
  "timestamp": "2025-01-01T12:00:00",
  "tags": ["baseline", "production"],
  "description": "Baseline model for production deployment",
  "command_line_args": {...},
  "python_version": "3.10.0",
  "config": {...}
}
```

## Advanced Features

### Cross-Validation

```bash
# Run 5-fold cross-validation
python scripts/train_model.py \
    --experiment-name cv_experiment \
    --cv-folds 5
```

### Hyperparameter Optimization

```bash
# Enable hyperparameter optimization with 50 trials
python scripts/train_model.py \
    --experiment-name hp_optimization \
    --optimize-hyperparams \
    --optimization-trials 50
```

### Model Comparison

```bash
# Compare multiple trained models
python scripts/train_model.py \
    --experiment-name model_comparison \
    --compare-models models/model1.pkl models/model2.pkl models/model3.pkl
```

### Resume Training

```bash
# Resume training from checkpoint
python scripts/train_model.py \
    --experiment-name resumed_training \
    --resume-from experiments/previous_experiment/models/checkpoints/epoch_50.pkl
```

## Logging System

### Training Metrics

The enhanced logging system tracks comprehensive training metrics:

- Train/validation loss and accuracy
- F1 score, precision, recall
- Learning rate changes
- Epoch timing
- Checkpoint information

### Progress Tracking

Real-time progress bars for different training phases:

```
Training: |████████████████████████████████████████████████████| 100/100 [100.00%] ETA: 0.0s
```

### Log Files

Multiple log files are generated:

- `training.log`: Main training log with detailed information
- `training_metrics.jsonl`: Machine-readable metrics for analysis
- `checkpoints.jsonl`: Checkpoint metadata and metrics

## Model Comparison

### Automatic Comparison

The training script includes built-in model comparison capabilities:

```python
from src.utils.model_comparison import ModelComparisonSuite

# Create comparison suite
suite = ModelComparisonSuite(output_dir)

# Add model evaluations
suite.add_model_evaluation(model1, X_test, y_test, "Random Forest")
suite.add_model_evaluation(model2, X_test, y_test, "XGBoost")

# Generate comprehensive report
comparison_df = suite.generate_comparison_report()
suite.plot_comparison_metrics()
suite.plot_confusion_matrices()
```

### Comparison Metrics

- Accuracy, Precision, Recall, F1-score
- ROC AUC score
- Training and inference time
- Model size
- Confusion matrices
- Classification reports

## Usage Examples

### Example 1: Basic Training

```bash
python scripts/train_model.py \
    --experiment-name basic_training \
    --description "Basic training run with default parameters" \
    --tags baseline initial
```

### Example 2: Custom Configuration

```bash
python scripts/train_model.py \
    --config config/production_config.yaml \
    --experiment-name production_training \
    --epochs 200 \
    --batch-size 128 \
    --learning-rate 0.0001 \
    --description "Production model training with optimized parameters"
```

### Example 3: Cross-Validation with Optimization

```bash
python scripts/train_model.py \
    --experiment-name cv_with_optimization \
    --cv-folds 10 \
    --optimize-hyperparams \
    --optimization-trials 100 \
    --description "10-fold CV with hyperparameter optimization"
```

### Example 4: Model Evaluation Only

```bash
python scripts/train_model.py \
    --experiment-name model_evaluation \
    --evaluate-only \
    --compare-models \
        models/random_forest.pkl \
        models/xgboost.pkl \
        models/neural_network.pkl \
    --description "Comparison of trained models"
```

### Example 5: Debug Mode

```bash
python scripts/train_model.py \
    --experiment-name debug_run \
    --debug \
    --log-level DEBUG \
    --epochs 5 \
    --description "Debug run with verbose logging"
```

## Integration with Existing Components

### Training Orchestrator

The script integrates with the existing `TrainingOrchestrator`:

```python
from src.training.training_orchestrator import TrainingOrchestrator

# Initialize orchestrator with configuration
orchestrator = TrainingOrchestrator(config)

# Run training
results = orchestrator.train_models(
    dataset_name=args.dataset_type,
    cross_validation_folds=args.cv_folds
)
```

### Data Loader Registry

Automatic dataset detection and loading:

```python
from src.data.data_loader import DataLoaderRegistry

# Auto-detect dataset type
registry = DataLoaderRegistry()
data_loader = registry.get_loader(dataset_type="auto", data_dir=data_dir)

# Load data
X, y, metadata = data_loader.load_data()
```

## Troubleshooting

### Common Issues

1. **Configuration Validation Errors**
   ```bash
   # Check configuration validity
   python -m src.utils.config_validation config/your_config.yaml
   ```

2. **Memory Issues**
   ```bash
   # Reduce batch size
   python scripts/train_model.py --batch-size 16
   ```

3. **Permission Errors**
   ```bash
   # Use force flag to overwrite existing experiments
   python scripts/train_model.py --force --experiment-name existing_name
   ```

### Debug Mode

Enable debug mode for detailed troubleshooting:

```bash
python scripts/train_model.py \
    --debug \
    --log-level DEBUG \
    --experiment-name debug_session
```

## Best Practices

### Experiment Naming

Use descriptive experiment names:

```bash
# Good
--experiment-name resnet50_aug_lr001_batch64

# Better
--experiment-name "ResNet50 with augmentation, lr=0.001, batch=64"
```

### Configuration Management

1. Keep configuration files in version control
2. Use different configs for different environments (dev, staging, prod)
3. Always validate configurations before training

### Resource Management

1. Monitor system resources during training
2. Use appropriate batch sizes for your hardware
3. Enable checkpointing for long training runs

### Reproducibility

1. Always set random seeds
2. Save complete experiment metadata
3. Use version control for code and configurations

## Advanced Configuration

### Custom Loss Functions

```yaml
training:
  loss_function: "focal_loss"  # or "weighted_cross_entropy"
  loss_params:
    alpha: 0.25
    gamma: 2.0
```

### Early Stopping

```yaml
training:
  early_stopping:
    patience: 10
    min_delta: 0.001
    monitor: "val_f1_score"
```

### Learning Rate Scheduling

```yaml
training:
  lr_scheduler:
    type: "step"
    step_size: 30
    gamma: 0.1
```

## API Reference

### Main Functions

- `parse_arguments()`: Parse command line arguments
- `setup_experiment_directory()`: Create experiment directory structure
- `load_and_merge_config()`: Load and validate configuration
- `setup_training_logger()`: Initialize enhanced logging
- `main()`: Main training function

### Utility Classes

- `ConfigValidator`: Configuration validation
- `TrainingLogger`: Enhanced logging with metrics
- `ModelComparisonSuite`: Model comparison and evaluation
- `ProgressTracker`: Training progress tracking

## Contributing

When contributing to the training script:

1. Add new CLI arguments to appropriate argument groups
2. Update configuration schema for new parameters
3. Add validation for new configuration options
4. Include comprehensive logging for new features
5. Update this documentation

## Support

For issues and questions:

1. Check the troubleshooting section
2. Review log files in the experiment directory
3. Use debug mode for detailed information
4. Check configuration validation results