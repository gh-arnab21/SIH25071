# Training Script Usage Examples

This document provides practical examples for using the enhanced training script.

## Basic Examples

### 1. Simple Training Run

```bash
python scripts/train_model.py \
    --experiment-name my_first_experiment \
    --description "My first training run"
```

### 2. Custom Parameters

```bash
python scripts/train_model.py \
    --experiment-name custom_params \
    --epochs 50 \
    --batch-size 64 \
    --learning-rate 0.001 \
    --description "Training with custom parameters"
```

### 3. Cross-Validation

```bash
python scripts/train_model.py \
    --experiment-name cross_validation \
    --cv-folds 5 \
    --description "5-fold cross-validation"
```

## Advanced Examples

### 4. Hyperparameter Optimization

```bash
python scripts/train_model.py \
    --experiment-name hyperopt \
    --optimize-hyperparams \
    --optimization-trials 50 \
    --description "Hyperparameter optimization with 50 trials"
```

### 5. Model Comparison

```bash
python scripts/train_model.py \
    --experiment-name model_comparison \
    --compare-models \
        models/random_forest.pkl \
        models/xgboost.pkl \
        models/neural_network.pkl \
    --description "Compare three different models"
```

### 6. Debug Mode

```bash
python scripts/train_model.py \
    --experiment-name debug_run \
    --debug \
    --log-level DEBUG \
    --epochs 2 \
    --description "Debug run with verbose logging"
```

## Configuration Examples

### 7. Custom Configuration File

```bash
python scripts/train_model.py \
    --config config/production_config.yaml \
    --experiment-name production_model \
    --description "Production model training"
```

### 8. Data Directory Override

```bash
python scripts/train_model.py \
    --experiment-name external_data \
    --data-dir /path/to/external/data \
    --dataset-type brazilian_rockfall \
    --description "Training with external Brazilian rockfall dataset"
```

## Validation and Testing

### 9. Dry Run

```bash
python scripts/train_model.py \
    --dry-run \
    --experiment-name test_config \
    --epochs 100 \
    --batch-size 128
```

### 10. Configuration Validation

```bash
# Validate configuration file
python -m src.utils.config_validation config/default_config.yaml

# Validate custom configuration
python -m src.utils.config_validation config/my_config.yaml
```

## Output Examples

### Experiment Directory Structure

After running an experiment, you'll get:

```
experiments/my_experiment_20250119_152030/
├── metadata.json
├── config.yaml
├── logs/
│   ├── training.log
│   ├── training_metrics.jsonl
│   └── checkpoints.jsonl
├── models/
│   └── best_model.pkl
└── results/
    ├── model_comparison.csv
    ├── confusion_matrices.png
    └── model_comparison_metrics.png
```

### Sample Log Output

```
2025-01-19 15:20:30,123 - training - INFO - ============================================================
2025-01-19 15:20:30,123 - training - INFO - TRAINING EXPERIMENT STARTED
2025-01-19 15:20:30,123 - training - INFO - ============================================================
2025-01-19 15:20:30,124 - training - INFO - Experiment: my_experiment
2025-01-19 15:20:30,124 - training - INFO - Description: My first training run
2025-01-19 15:20:30,124 - training - INFO - Tags: 
2025-01-19 15:20:30,125 - training - INFO - 
Key Configuration:
2025-01-19 15:20:30,125 - training - INFO -   Max Epochs: 100
2025-01-19 15:20:30,125 - training - INFO -   Batch Size: 32
2025-01-19 15:20:30,125 - training - INFO -   Validation Split: 0.15
```

### Model Comparison Output

```
         model_name  accuracy  precision   recall  f1_score  roc_auc
0        Random Forest  0.856667   0.858939 0.856667  0.856649 0.924449
1  Logistic Regression  0.850000   0.851561 0.850000  0.850015 0.914171
```

## Tips and Best Practices

- Use descriptive experiment names
- Always include a description for your experiments
- Use tags to organize related experiments
- Run dry runs to test configurations
- Enable debug mode when troubleshooting
- Use cross-validation for robust evaluation
- Save important configurations as separate files