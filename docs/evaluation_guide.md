# Model Evaluation Guide

This guide explains how to use the evaluation utilities in the rockfall prediction system.

## Overview

The evaluation module (`src/utils/evaluation.py`) provides comprehensive tools for assessing model performance including:

- Classification metrics (precision, recall, F1-score)
- ROC curve generation and AUC calculation
- Confusion matrix visualization
- Model performance reporting and comparison
- Performance tracking over time

## Quick Start

```python
from utils.evaluation import ModelEvaluator, quick_evaluate
import numpy as np

# Your model predictions
y_true = np.array([0, 1, 2, 0, 1])  # True risk levels
y_pred = np.array([0, 1, 1, 0, 2])  # Predicted risk levels
y_prob = np.array([[0.9, 0.1, 0.0],  # Prediction probabilities
                   [0.2, 0.7, 0.1],
                   [0.1, 0.3, 0.6],
                   [0.8, 0.2, 0.0],
                   [0.3, 0.6, 0.1]])

# Quick evaluation
results = quick_evaluate(y_true, y_pred, y_prob, ['Low', 'Medium', 'High'])
print(f"Accuracy: {results['classification_metrics']['accuracy']:.3f}")
```

## Detailed Usage

### 1. ModelEvaluator Class

```python
from utils.evaluation import ModelEvaluator

# Initialize with custom class names
evaluator = ModelEvaluator(['Low', 'Medium', 'High'])

# Calculate classification metrics
metrics = evaluator.calculate_classification_metrics(y_true, y_pred)
print(f"F1-Score: {metrics['f1_weighted']:.3f}")

# Calculate ROC-AUC metrics
roc_metrics = evaluator.calculate_roc_auc_metrics(y_true, y_prob)
print(f"ROC-AUC: {roc_metrics['roc_auc_macro']:.3f}")

# Generate confusion matrix
cm = evaluator.plot_confusion_matrix(y_true, y_pred, save_path='confusion_matrix.png')

# Comprehensive evaluation
results = evaluator.comprehensive_evaluation(y_true, y_pred, y_prob, save_dir='results/')
```

### 2. Model Comparison

```python
# Compare multiple models
model_results = {
    'Random Forest': {
        'classification_metrics': {'accuracy': 0.85, 'f1_weighted': 0.83},
        'roc_auc_metrics': {'roc_auc_macro': 0.82}
    },
    'XGBoost': {
        'classification_metrics': {'accuracy': 0.88, 'f1_weighted': 0.86},
        'roc_auc_metrics': {'roc_auc_macro': 0.85}
    }
}

comparison_df = evaluator.compare_models(model_results)
print(comparison_df)
```

### 3. Performance Tracking

```python
from utils.evaluation import ModelPerformanceTracker

# Track model performance over time
tracker = ModelPerformanceTracker('performance_history.json')

# Add evaluation results
tracker.add_evaluation(
    model_name='RandomForest_v1',
    dataset_name='validation_set',
    metrics={'accuracy': 0.85, 'f1_weighted': 0.83},
    metadata={'learning_rate': 0.1, 'n_estimators': 100}
)

# Get best performing model
best_model = tracker.get_best_model('f1_weighted')
print(f"Best model: {best_model['model_name']}")

# Get performance trends
trends = tracker.get_performance_trends('RandomForest_v1')
```

### 4. Utility Functions

```python
from utils.evaluation import evaluate_and_save

# Evaluate and save all results
results = evaluate_and_save(
    y_true, y_pred, 'MyModel',
    y_prob=y_prob,
    class_names=['Low', 'Medium', 'High'],
    save_dir='evaluation_results/'
)
```

## Available Metrics

### Classification Metrics
- **Accuracy**: Overall correctness
- **Precision**: True positives / (True positives + False positives)
- **Recall**: True positives / (True positives + False negatives)
- **F1-Score**: Harmonic mean of precision and recall
- **Support**: Number of samples per class

### ROC Metrics
- **ROC-AUC**: Area under the ROC curve
- **Macro Average**: Unweighted mean of per-class metrics
- **Weighted Average**: Weighted by class support

### Visualizations
- **Confusion Matrix**: Shows prediction vs actual classifications
- **ROC Curves**: True positive rate vs false positive rate
- **Precision-Recall Curves**: Precision vs recall trade-off

## Best Practices

1. **Use appropriate metrics**: For imbalanced datasets, focus on F1-score and AUC rather than accuracy
2. **Cross-validation**: Always validate on multiple data splits
3. **Class-specific analysis**: Check per-class metrics to identify problematic classes
4. **Confidence scores**: Use prediction probabilities to assess model uncertainty
5. **Track performance**: Monitor model performance over time and across datasets

## Example Output

```
Classification Metrics:
- Accuracy: 0.850
- Precision (Weighted): 0.847
- Recall (Weighted): 0.850
- F1-Score (Weighted): 0.848

ROC-AUC Metrics:
- ROC-AUC (Macro): 0.823
- ROC-AUC (Weighted): 0.831

Per-Class Performance:
- Low Risk: Precision=0.92, Recall=0.88, F1=0.90
- Medium Risk: Precision=0.78, Recall=0.82, F1=0.80
- High Risk: Precision=0.85, Recall=0.85, F1=0.85
```

This evaluation framework ensures comprehensive assessment of rockfall prediction models across all relevant metrics and provides tools for ongoing performance monitoring.