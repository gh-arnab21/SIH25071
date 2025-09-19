#!/usr/bin/env python3
"""
Simple training script for mining datasets.
"""

import sys
import os
import argparse
import logging
import numpy as np
from pathlib import Path
from datetime import datetime
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
import joblib

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

from src.data.mining_loader import load_mining_dataset_for_training
from src.utils.logging import setup_logging


def setup_experiment_directory(experiment_name: str) -> Path:
    """Create experiment directory."""
    if not experiment_name:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        experiment_name = f"mining_experiment_{timestamp}"
    
    exp_dir = Path("experiments") / experiment_name
    exp_dir.mkdir(parents=True, exist_ok=True)
    return exp_dir


def train_single_model(X_train, y_train, X_test, y_test, model, model_name: str, exp_dir: Path):
    """Train and evaluate a single model."""
    logger = logging.getLogger(__name__)
    
    logger.info(f"Training {model_name}...")
    
    # Train the model
    model.fit(X_train, y_train)
    
    # Make predictions
    y_pred = model.predict(X_test)
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    
    # Cross-validation score
    cv_scores = cross_val_score(model, X_train, y_train, cv=5)
    
    logger.info(f"{model_name} Results:")
    logger.info(f"  Test Accuracy: {accuracy:.4f}")
    logger.info(f"  CV Mean: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
    
    # Save model
    model_path = exp_dir / f"{model_name.lower().replace(' ', '_')}_model.joblib"
    joblib.dump(model, model_path)
    logger.info(f"  Model saved to: {model_path}")
    
    # Save detailed results
    results = {
        'model_name': model_name,
        'test_accuracy': accuracy,
        'cv_mean': cv_scores.mean(),
        'cv_std': cv_scores.std(),
        'cv_scores': cv_scores.tolist(),
        'classification_report': classification_report(y_test, y_pred, output_dict=True),
        'confusion_matrix': confusion_matrix(y_test, y_pred).tolist()
    }
    
    # Save classification report
    report_path = exp_dir / f"{model_name.lower().replace(' ', '_')}_report.txt"
    with open(report_path, 'w') as f:
        f.write(f"{model_name} Classification Report\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"Test Accuracy: {accuracy:.4f}\n")
        f.write(f"Cross-validation: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})\n\n")
        f.write("Classification Report:\n")
        f.write(classification_report(y_test, y_pred))
        f.write("\n\nConfusion Matrix:\n")
        f.write(str(confusion_matrix(y_test, y_pred)))
    
    return results


def main():
    """Main training function."""
    parser = argparse.ArgumentParser(description="Train models on mining datasets")
    parser.add_argument('--dataset-type', choices=['object_detection', 'segmentation', 'combined'], 
                       default='combined', help='Dataset to use')
    parser.add_argument('--experiment-name', help='Experiment name')
    parser.add_argument('--test-size', type=float, default=0.2, help='Test set size')
    parser.add_argument('--random-state', type=int, default=42, help='Random state')
    parser.add_argument('--models', nargs='+', 
                       choices=['random_forest', 'gradient_boosting', 'logistic_regression', 'neural_network'],
                       default=['random_forest', 'gradient_boosting'],
                       help='Models to train')
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging()
    logger = logging.getLogger(__name__)
    
    # Create experiment directory
    exp_dir = setup_experiment_directory(args.experiment_name)
    logger.info(f"Experiment directory: {exp_dir}")
    
    try:
        # Load dataset
        logger.info(f"Loading {args.dataset_type} dataset...")
        X, y, metadata = load_mining_dataset_for_training(args.dataset_type)
        
        logger.info(f"Dataset loaded successfully:")
        logger.info(f"  Samples: {metadata['num_samples']}")
        logger.info(f"  Features: {metadata['num_features']}")
        logger.info(f"  Classes: {metadata['num_classes']}")
        logger.info(f"  Class distribution: {metadata['class_distribution']}")
        
        # Split the data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=args.test_size, random_state=args.random_state, stratify=y
        )
        
        logger.info(f"Data split: {len(X_train)} train, {len(X_test)} test")
        
        # Define models
        models = {
            'random_forest': RandomForestClassifier(n_estimators=100, random_state=args.random_state),
            'gradient_boosting': GradientBoostingClassifier(n_estimators=100, random_state=args.random_state),
            'logistic_regression': LogisticRegression(random_state=args.random_state, max_iter=1000),
            'neural_network': MLPClassifier(hidden_layer_sizes=(100, 50), random_state=args.random_state, max_iter=500)
        }
        
        # Train selected models
        results = {}
        best_model = None
        best_accuracy = 0
        
        for model_name in args.models:
            if model_name in models:
                model = models[model_name]
                result = train_single_model(X_train, y_train, X_test, y_test, 
                                          model, model_name.replace('_', ' ').title(), exp_dir)
                results[model_name] = result
                
                # Track best model
                if result['test_accuracy'] > best_accuracy:
                    best_accuracy = result['test_accuracy']
                    best_model = model_name
        
        # Save experiment summary
        summary_path = exp_dir / "experiment_summary.txt"
        with open(summary_path, 'w') as f:
            f.write(f"Mining Dataset Training Experiment\n")
            f.write("=" * 50 + "\n\n")
            f.write(f"Dataset: {args.dataset_type}\n")
            f.write(f"Samples: {metadata['num_samples']}\n")
            f.write(f"Features: {metadata['num_features']}\n")
            f.write(f"Classes: {metadata['num_classes']}\n")
            f.write(f"Test size: {args.test_size}\n")
            f.write(f"Random state: {args.random_state}\n\n")
            
            f.write("Results Summary:\n")
            f.write("-" * 30 + "\n")
            for model_name, result in results.items():
                f.write(f"{result['model_name']}: {result['test_accuracy']:.4f}\n")
            
            f.write(f"\nBest Model: {best_model.replace('_', ' ').title()} ({best_accuracy:.4f})\n")
        
        logger.info("\n" + "="*50)
        logger.info("TRAINING COMPLETED")
        logger.info("="*50)
        logger.info(f"Best model: {best_model.replace('_', ' ').title()} (accuracy: {best_accuracy:.4f})")
        logger.info(f"Results saved to: {exp_dir}")
        
        # Print final summary
        print("\nTraining Summary:")
        print("-" * 30)
        for model_name, result in results.items():
            print(f"{result['model_name']}: {result['test_accuracy']:.4f}")
        print(f"\nBest Model: {best_model.replace('_', ' ').title()} ({best_accuracy:.4f})")
        print(f"Experiment saved to: {exp_dir}")
        
    except Exception as e:
        logger.error(f"Training failed: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return 1
    
    return 0


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)