"""
Model evaluation and metrics utilities for rockfall prediction system.

This module provides comprehensive evaluation functions including:
- Classification metrics (precision, recall, F1-score)
- ROC curve generation and AUC calculation
- Confusion matrix visualization
- Model performance reporting and comparison
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional, Any, Union
from sklearn.metrics import (
    precision_score, recall_score, f1_score, accuracy_score,
    confusion_matrix, classification_report, roc_curve, auc,
    roc_auc_score, precision_recall_curve, average_precision_score
)
from sklearn.preprocessing import label_binarize
import pandas as pd
import json
from pathlib import Path


class ModelEvaluator:
    """
    Comprehensive model evaluation class for rockfall prediction models.
    
    Provides methods for calculating various metrics, generating visualizations,
    and creating performance reports.
    """
    
    def __init__(self, class_names: List[str] = None):
        """
        Initialize the evaluator.
        
        Args:
            class_names: List of class names for multi-class classification
                        Default: ['Low', 'Medium', 'High'] for rockfall risk levels
        """
        self.class_names = class_names or ['Low', 'Medium', 'High']
        self.n_classes = len(self.class_names)
    
    def calculate_classification_metrics(
        self, 
        y_true: np.ndarray, 
        y_pred: np.ndarray,
        average: str = 'weighted'
    ) -> Dict[str, float]:
        """
        Calculate comprehensive classification metrics.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            average: Averaging strategy for multi-class metrics
        
        Returns:
            Dictionary containing all classification metrics
        """
        metrics = {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision_macro': precision_score(y_true, y_pred, average='macro', zero_division=0),
            'precision_micro': precision_score(y_true, y_pred, average='micro', zero_division=0),
            'precision_weighted': precision_score(y_true, y_pred, average='weighted', zero_division=0),
            'recall_macro': recall_score(y_true, y_pred, average='macro', zero_division=0),
            'recall_micro': recall_score(y_true, y_pred, average='micro', zero_division=0),
            'recall_weighted': recall_score(y_true, y_pred, average='weighted', zero_division=0),
            'f1_macro': f1_score(y_true, y_pred, average='macro', zero_division=0),
            'f1_micro': f1_score(y_true, y_pred, average='micro', zero_division=0),
            'f1_weighted': f1_score(y_true, y_pred, average='weighted', zero_division=0)
        }
        
        # Per-class metrics
        precision_per_class = precision_score(y_true, y_pred, average=None, zero_division=0)
        recall_per_class = recall_score(y_true, y_pred, average=None, zero_division=0)
        f1_per_class = f1_score(y_true, y_pred, average=None, zero_division=0)
        
        for i, class_name in enumerate(self.class_names):
            if i < len(precision_per_class):
                metrics[f'precision_{class_name.lower()}'] = precision_per_class[i]
                metrics[f'recall_{class_name.lower()}'] = recall_per_class[i]
                metrics[f'f1_{class_name.lower()}'] = f1_per_class[i]
        
        return metrics
    
    def calculate_roc_auc_metrics(
        self, 
        y_true: np.ndarray, 
        y_prob: np.ndarray,
        multi_class: str = 'ovr'
    ) -> Dict[str, float]:
        """
        Calculate ROC-AUC metrics for binary and multi-class classification.
        
        Args:
            y_true: True labels
            y_prob: Predicted probabilities (shape: [n_samples, n_classes])
            multi_class: Strategy for multi-class ROC-AUC ('ovr' or 'ovo')
        
        Returns:
            Dictionary containing ROC-AUC metrics
        """
        metrics = {}
        
        try:
            if self.n_classes == 2:
                # Binary classification
                if y_prob.ndim == 2:
                    y_prob_binary = y_prob[:, 1]  # Probability of positive class
                else:
                    y_prob_binary = y_prob
                
                metrics['roc_auc'] = roc_auc_score(y_true, y_prob_binary)
                
            else:
                # Multi-class classification
                metrics['roc_auc_macro'] = roc_auc_score(
                    y_true, y_prob, multi_class=multi_class, average='macro'
                )
                metrics['roc_auc_weighted'] = roc_auc_score(
                    y_true, y_prob, multi_class=multi_class, average='weighted'
                )
                
                # Per-class AUC (one-vs-rest)
                y_true_binarized = label_binarize(y_true, classes=range(self.n_classes))
                if y_true_binarized.shape[1] == 1:
                    # Handle case where only one class is present
                    y_true_binarized = np.hstack([1 - y_true_binarized, y_true_binarized])
                
                for i, class_name in enumerate(self.class_names):
                    if i < y_prob.shape[1] and i < y_true_binarized.shape[1]:
                        try:
                            class_auc = roc_auc_score(y_true_binarized[:, i], y_prob[:, i])
                            metrics[f'roc_auc_{class_name.lower()}'] = class_auc
                        except ValueError:
                            # Handle case where class is not present in y_true
                            metrics[f'roc_auc_{class_name.lower()}'] = 0.0
                            
        except ValueError as e:
            print(f"Warning: Could not calculate ROC-AUC metrics: {e}")
            metrics['roc_auc'] = 0.0
        
        return metrics
    
    def generate_roc_curves(
        self, 
        y_true: np.ndarray, 
        y_prob: np.ndarray,
        save_path: Optional[str] = None,
        figsize: Tuple[int, int] = (10, 8)
    ) -> Dict[str, Tuple[np.ndarray, np.ndarray, float]]:
        """
        Generate ROC curves for each class and overall.
        
        Args:
            y_true: True labels
            y_prob: Predicted probabilities
            save_path: Path to save the plot (optional)
            figsize: Figure size for the plot
        
        Returns:
            Dictionary containing FPR, TPR, and AUC for each class
        """
        plt.figure(figsize=figsize)
        roc_data = {}
        
        if self.n_classes == 2:
            # Binary classification
            y_prob_binary = y_prob[:, 1] if y_prob.ndim == 2 else y_prob
            fpr, tpr, _ = roc_curve(y_true, y_prob_binary)
            roc_auc = auc(fpr, tpr)
            
            plt.plot(fpr, tpr, linewidth=2, 
                    label=f'ROC Curve (AUC = {roc_auc:.3f})')
            roc_data['binary'] = (fpr, tpr, roc_auc)
            
        else:
            # Multi-class classification
            y_true_binarized = label_binarize(y_true, classes=range(self.n_classes))
            if y_true_binarized.shape[1] == 1:
                y_true_binarized = np.hstack([1 - y_true_binarized, y_true_binarized])
            
            colors = plt.cm.Set1(np.linspace(0, 1, self.n_classes))
            
            for i, (class_name, color) in enumerate(zip(self.class_names, colors)):
                if i < y_prob.shape[1] and i < y_true_binarized.shape[1]:
                    try:
                        fpr, tpr, _ = roc_curve(y_true_binarized[:, i], y_prob[:, i])
                        roc_auc = auc(fpr, tpr)
                        
                        plt.plot(fpr, tpr, color=color, linewidth=2,
                                label=f'{class_name} (AUC = {roc_auc:.3f})')
                        roc_data[class_name.lower()] = (fpr, tpr, roc_auc)
                    except ValueError:
                        continue
        
        # Plot diagonal line
        plt.plot([0, 1], [0, 1], 'k--', linewidth=1, alpha=0.8)
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic (ROC) Curves')
        plt.legend(loc="lower right")
        plt.grid(True, alpha=0.3)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"ROC curves saved to: {save_path}")
        
        plt.show()
        return roc_data
    
    def plot_confusion_matrix(
        self, 
        y_true: np.ndarray, 
        y_pred: np.ndarray,
        normalize: Optional[str] = None,
        save_path: Optional[str] = None,
        figsize: Tuple[int, int] = (8, 6)
    ) -> np.ndarray:
        """
        Plot confusion matrix with visualization.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            normalize: Normalization method ('true', 'pred', 'all', or None)
            save_path: Path to save the plot (optional)
            figsize: Figure size for the plot
        
        Returns:
            Confusion matrix array
        """
        cm = confusion_matrix(y_true, y_pred)
        
        if normalize:
            if normalize == 'true':
                cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            elif normalize == 'pred':
                cm = cm.astype('float') / cm.sum(axis=0)
            elif normalize == 'all':
                cm = cm.astype('float') / cm.sum()
        
        plt.figure(figsize=figsize)
        sns.heatmap(cm, annot=True, fmt='.2f' if normalize else 'd',
                   cmap='Blues', xticklabels=self.class_names, 
                   yticklabels=self.class_names)
        
        plt.title(f'Confusion Matrix{" (Normalized)" if normalize else ""}')
        plt.xlabel('Predicted Label')
        plt.ylabel('True Label')
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Confusion matrix saved to: {save_path}")
        
        plt.show()
        return cm
    
    def generate_classification_report(
        self, 
        y_true: np.ndarray, 
        y_pred: np.ndarray,
        output_dict: bool = False
    ) -> Union[str, Dict]:
        """
        Generate detailed classification report.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            output_dict: Return as dictionary instead of string
        
        Returns:
            Classification report as string or dictionary
        """
        return classification_report(
            y_true, y_pred, 
            target_names=self.class_names,
            output_dict=output_dict,
            zero_division=0
        )
    
    def calculate_precision_recall_curves(
        self, 
        y_true: np.ndarray, 
        y_prob: np.ndarray,
        save_path: Optional[str] = None,
        figsize: Tuple[int, int] = (10, 8)
    ) -> Dict[str, Tuple[np.ndarray, np.ndarray, float]]:
        """
        Generate precision-recall curves for each class.
        
        Args:
            y_true: True labels
            y_prob: Predicted probabilities
            save_path: Path to save the plot (optional)
            figsize: Figure size for the plot
        
        Returns:
            Dictionary containing precision, recall, and AP for each class
        """
        plt.figure(figsize=figsize)
        pr_data = {}
        
        if self.n_classes == 2:
            # Binary classification
            y_prob_binary = y_prob[:, 1] if y_prob.ndim == 2 else y_prob
            precision, recall, _ = precision_recall_curve(y_true, y_prob_binary)
            ap = average_precision_score(y_true, y_prob_binary)
            
            plt.plot(recall, precision, linewidth=2,
                    label=f'PR Curve (AP = {ap:.3f})')
            pr_data['binary'] = (precision, recall, ap)
            
        else:
            # Multi-class classification
            y_true_binarized = label_binarize(y_true, classes=range(self.n_classes))
            if y_true_binarized.shape[1] == 1:
                y_true_binarized = np.hstack([1 - y_true_binarized, y_true_binarized])
            
            colors = plt.cm.Set1(np.linspace(0, 1, self.n_classes))
            
            for i, (class_name, color) in enumerate(zip(self.class_names, colors)):
                if i < y_prob.shape[1] and i < y_true_binarized.shape[1]:
                    try:
                        precision, recall, _ = precision_recall_curve(
                            y_true_binarized[:, i], y_prob[:, i]
                        )
                        ap = average_precision_score(y_true_binarized[:, i], y_prob[:, i])
                        
                        plt.plot(recall, precision, color=color, linewidth=2,
                                label=f'{class_name} (AP = {ap:.3f})')
                        pr_data[class_name.lower()] = (precision, recall, ap)
                    except ValueError:
                        continue
        
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision-Recall Curves')
        plt.legend(loc="lower left")
        plt.grid(True, alpha=0.3)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Precision-Recall curves saved to: {save_path}")
        
        plt.show()
        return pr_data
    
    def comprehensive_evaluation(
        self, 
        y_true: np.ndarray, 
        y_pred: np.ndarray,
        y_prob: Optional[np.ndarray] = None,
        save_dir: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Perform comprehensive model evaluation with all metrics and visualizations.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            y_prob: Predicted probabilities (optional, for ROC/PR curves)
            save_dir: Directory to save plots and reports (optional)
        
        Returns:
            Dictionary containing all evaluation results
        """
        results = {}
        
        # Basic classification metrics
        results['classification_metrics'] = self.calculate_classification_metrics(y_true, y_pred)
        
        # Classification report
        results['classification_report'] = self.generate_classification_report(
            y_true, y_pred, output_dict=True
        )
        
        # Confusion matrix
        cm_path = None
        if save_dir:
            Path(save_dir).mkdir(parents=True, exist_ok=True)
            cm_path = f"{save_dir}/confusion_matrix.png"
        
        results['confusion_matrix'] = self.plot_confusion_matrix(
            y_true, y_pred, save_path=cm_path
        )
        
        # ROC curves and metrics (if probabilities provided)
        if y_prob is not None:
            results['roc_auc_metrics'] = self.calculate_roc_auc_metrics(y_true, y_prob)
            
            roc_path = None
            pr_path = None
            if save_dir:
                roc_path = f"{save_dir}/roc_curves.png"
                pr_path = f"{save_dir}/precision_recall_curves.png"
            
            results['roc_curves'] = self.generate_roc_curves(
                y_true, y_prob, save_path=roc_path
            )
            results['precision_recall_curves'] = self.calculate_precision_recall_curves(
                y_true, y_prob, save_path=pr_path
            )
        
        return results
    
    def compare_models(
        self, 
        model_results: Dict[str, Dict[str, Any]],
        save_path: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Compare multiple model evaluation results.
        
        Args:
            model_results: Dictionary with model names as keys and evaluation results as values
            save_path: Path to save comparison table (optional)
        
        Returns:
            DataFrame containing model comparison
        """
        comparison_data = []
        
        for model_name, results in model_results.items():
            metrics = results.get('classification_metrics', {})
            roc_metrics = results.get('roc_auc_metrics', {})
            
            row = {'Model': model_name}
            row.update(metrics)
            row.update(roc_metrics)
            comparison_data.append(row)
        
        comparison_df = pd.DataFrame(comparison_data)
        
        if save_path:
            comparison_df.to_csv(save_path, index=False)
            print(f"Model comparison saved to: {save_path}")
        
        return comparison_df
    
    def save_evaluation_report(
        self, 
        results: Dict[str, Any], 
        model_name: str,
        save_path: str
    ) -> None:
        """
        Save comprehensive evaluation report to JSON file.
        
        Args:
            results: Evaluation results dictionary
            model_name: Name of the evaluated model
            save_path: Path to save the report
        """
        report = {
            'model_name': model_name,
            'evaluation_timestamp': pd.Timestamp.now().isoformat(),
            'class_names': self.class_names,
            'metrics': {}
        }
        
        # Add serializable metrics
        for key, value in results.items():
            if key in ['classification_metrics', 'roc_auc_metrics']:
                report['metrics'][key] = value
            elif key == 'classification_report':
                report['metrics'][key] = value
            elif key == 'confusion_matrix':
                report['metrics'][key] = value.tolist()
        
        with open(save_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        print(f"Evaluation report saved to: {save_path}")


class ModelPerformanceTracker:
    """
    Track model performance over time and across different datasets.
    """
    
    def __init__(self, tracking_file: str = "model_performance_history.json"):
        """
        Initialize performance tracker.
        
        Args:
            tracking_file: Path to file for storing performance history
        """
        self.tracking_file = tracking_file
        self.history = self._load_history()
    
    def _load_history(self) -> List[Dict]:
        """Load performance history from file."""
        try:
            with open(self.tracking_file, 'r') as f:
                content = f.read().strip()
                if not content:
                    return []
                return json.loads(content)
        except (FileNotFoundError, json.JSONDecodeError):
            return []
    
    def _save_history(self) -> None:
        """Save performance history to file."""
        with open(self.tracking_file, 'w') as f:
            json.dump(self.history, f, indent=2)
    
    def add_evaluation(
        self, 
        model_name: str, 
        dataset_name: str,
        metrics: Dict[str, float],
        metadata: Optional[Dict] = None
    ) -> None:
        """
        Add evaluation results to performance history.
        
        Args:
            model_name: Name of the evaluated model
            dataset_name: Name of the dataset used for evaluation
            metrics: Dictionary of evaluation metrics
            metadata: Additional metadata (hyperparameters, etc.)
        """
        entry = {
            'timestamp': pd.Timestamp.now().isoformat(),
            'model_name': model_name,
            'dataset_name': dataset_name,
            'metrics': metrics,
            'metadata': metadata or {}
        }
        
        self.history.append(entry)
        self._save_history()
    
    def get_best_model(self, metric: str = 'f1_weighted') -> Optional[Dict]:
        """
        Get the best performing model based on specified metric.
        
        Args:
            metric: Metric to use for comparison
        
        Returns:
            Dictionary containing best model information
        """
        if not self.history:
            return None
        
        best_entry = max(
            self.history, 
            key=lambda x: x['metrics'].get(metric, 0)
        )
        
        return best_entry
    
    def get_performance_trends(self, model_name: str) -> pd.DataFrame:
        """
        Get performance trends for a specific model over time.
        
        Args:
            model_name: Name of the model to analyze
        
        Returns:
            DataFrame with performance trends
        """
        model_history = [
            entry for entry in self.history 
            if entry['model_name'] == model_name
        ]
        
        if not model_history:
            return pd.DataFrame()
        
        # Flatten metrics for DataFrame creation
        flattened_data = []
        for entry in model_history:
            row = {
                'timestamp': entry['timestamp'],
                'dataset_name': entry['dataset_name']
            }
            row.update(entry['metrics'])
            flattened_data.append(row)
        
        df = pd.DataFrame(flattened_data)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        return df.sort_values('timestamp')


# Utility functions for quick evaluation
def quick_evaluate(
    y_true: np.ndarray, 
    y_pred: np.ndarray, 
    y_prob: Optional[np.ndarray] = None,
    class_names: Optional[List[str]] = None
) -> Dict[str, Any]:
    """
    Quick evaluation function for immediate results.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        y_prob: Predicted probabilities (optional)
        class_names: List of class names (optional)
    
    Returns:
        Dictionary containing evaluation results
    """
    evaluator = ModelEvaluator(class_names)
    return evaluator.comprehensive_evaluation(y_true, y_pred, y_prob)


def evaluate_and_save(
    y_true: np.ndarray, 
    y_pred: np.ndarray,
    model_name: str,
    y_prob: Optional[np.ndarray] = None,
    class_names: Optional[List[str]] = None,
    save_dir: str = "evaluation_results"
) -> Dict[str, Any]:
    """
    Evaluate model and save all results to specified directory.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        model_name: Name of the model being evaluated
        y_prob: Predicted probabilities (optional)
        class_names: List of class names (optional)
        save_dir: Directory to save results
    
    Returns:
        Dictionary containing evaluation results
    """
    evaluator = ModelEvaluator(class_names)
    results = evaluator.comprehensive_evaluation(y_true, y_pred, y_prob, save_dir)
    
    # Save evaluation report
    report_path = f"{save_dir}/{model_name}_evaluation_report.json"
    evaluator.save_evaluation_report(results, model_name, report_path)
    
    return results