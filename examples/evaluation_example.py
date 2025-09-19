"""
Example script demonstrating model evaluation and metrics functionality.

This script shows how to use the evaluation utilities to assess
rockfall prediction model performance.
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

import numpy as np
from utils.evaluation import ModelEvaluator, quick_evaluate, evaluate_and_save

def main():
    """Demonstrate evaluation functionality."""
    print("Rockfall Prediction Model - Evaluation Example")
    print("=" * 50)
    
    # Generate sample data for demonstration
    np.random.seed(42)
    n_samples = 200
    
    # Simulate multi-class rockfall risk prediction
    # 0: Low risk, 1: Medium risk, 2: High risk
    y_true = np.random.choice([0, 1, 2], size=n_samples, p=[0.6, 0.3, 0.1])
    
    # Simulate model predictions (with some correlation to true labels)
    y_pred = y_true.copy()
    # Add some noise to predictions
    noise_indices = np.random.choice(n_samples, size=int(0.2 * n_samples), replace=False)
    y_pred[noise_indices] = np.random.choice([0, 1, 2], size=len(noise_indices))
    
    # Simulate prediction probabilities
    y_prob = np.random.rand(n_samples, 3)
    # Make probabilities somewhat realistic
    for i in range(n_samples):
        y_prob[i, y_true[i]] += 0.5  # Boost true class probability
    y_prob = y_prob / y_prob.sum(axis=1, keepdims=True)  # Normalize
    
    print(f"Generated {n_samples} samples with {len(np.unique(y_true))} risk classes")
    print(f"True class distribution: {np.bincount(y_true)}")
    print(f"Predicted class distribution: {np.bincount(y_pred)}")
    print()
    
    # Initialize evaluator
    evaluator = ModelEvaluator(['Low', 'Medium', 'High'])
    
    # 1. Calculate basic classification metrics
    print("1. Classification Metrics:")
    print("-" * 25)
    metrics = evaluator.calculate_classification_metrics(y_true, y_pred)
    
    key_metrics = ['accuracy', 'precision_weighted', 'recall_weighted', 'f1_weighted']
    for metric in key_metrics:
        print(f"{metric.replace('_', ' ').title()}: {metrics[metric]:.3f}")
    
    print()
    
    # 2. Calculate ROC-AUC metrics
    print("2. ROC-AUC Metrics:")
    print("-" * 18)
    roc_metrics = evaluator.calculate_roc_auc_metrics(y_true, y_prob)
    
    for metric, value in roc_metrics.items():
        if 'roc_auc' in metric:
            print(f"{metric.replace('_', ' ').title()}: {value:.3f}")
    
    print()
    
    # 3. Generate classification report
    print("3. Classification Report:")
    print("-" * 23)
    report = evaluator.generate_classification_report(y_true, y_pred)
    print(report)
    
    # 4. Comprehensive evaluation (without plots for this example)
    print("4. Comprehensive Evaluation Summary:")
    print("-" * 35)
    
    # Mock the plotting functions to avoid display issues
    import matplotlib
    matplotlib.use('Agg')  # Use non-interactive backend
    
    results = evaluator.comprehensive_evaluation(y_true, y_pred, y_prob)
    
    print(f"Confusion Matrix Shape: {results['confusion_matrix'].shape}")
    print(f"Number of ROC Curves Generated: {len(results['roc_curves'])}")
    print(f"Number of PR Curves Generated: {len(results['precision_recall_curves'])}")
    
    # 5. Model comparison example
    print("\n5. Model Comparison Example:")
    print("-" * 28)
    
    # Simulate results from two different models
    model1_results = {
        'classification_metrics': {
            'accuracy': 0.85,
            'f1_weighted': 0.83,
            'precision_weighted': 0.84
        },
        'roc_auc_metrics': {
            'roc_auc_macro': 0.82
        }
    }
    
    model2_results = {
        'classification_metrics': {
            'accuracy': 0.88,
            'f1_weighted': 0.86,
            'precision_weighted': 0.87
        },
        'roc_auc_metrics': {
            'roc_auc_macro': 0.85
        }
    }
    
    comparison = evaluator.compare_models({
        'Random Forest': model1_results,
        'XGBoost': model2_results
    })
    
    print("Model Comparison:")
    print(comparison[['Model', 'accuracy', 'f1_weighted', 'roc_auc_macro']].to_string(index=False))
    
    # 6. Quick evaluation utility
    print("\n6. Quick Evaluation Utility:")
    print("-" * 28)
    
    quick_results = quick_evaluate(y_true, y_pred, y_prob, ['Low', 'Medium', 'High'])
    quick_metrics = quick_results['classification_metrics']
    
    print(f"Quick Accuracy: {quick_metrics['accuracy']:.3f}")
    print(f"Quick F1-Score: {quick_metrics['f1_weighted']:.3f}")
    
    print("\nEvaluation example completed successfully!")
    print("All evaluation functions are working correctly.")


if __name__ == "__main__":
    main()