"""
Model comparison utilities for the Rockfall Prediction System.
Provides comprehensive model evaluation, comparison, and selection capabilities.
"""

import json
import pickle
import logging
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple, Union
from dataclasses import dataclass, field
from datetime import datetime
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report
)
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns


@dataclass
class ModelMetrics:
    """Data class for storing model evaluation metrics."""
    model_name: str
    accuracy: float = 0.0
    precision: float = 0.0
    recall: float = 0.0
    f1_score: float = 0.0
    roc_auc: float = 0.0
    confusion_matrix: Optional[np.ndarray] = None
    classification_report: Optional[str] = None
    training_time: float = 0.0
    inference_time: float = 0.0
    model_size_mb: float = 0.0
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert metrics to dictionary for serialization."""
        result = {
            'model_name': self.model_name,
            'accuracy': float(self.accuracy),
            'precision': float(self.precision),
            'recall': float(self.recall),
            'f1_score': float(self.f1_score),
            'roc_auc': float(self.roc_auc),
            'training_time': float(self.training_time),
            'inference_time': float(self.inference_time),
            'model_size_mb': float(self.model_size_mb),
            'timestamp': self.timestamp.isoformat(),
            'metadata': self.metadata
        }
        
        if self.confusion_matrix is not None:
            result['confusion_matrix'] = self.confusion_matrix.tolist()
        
        if self.classification_report is not None:
            result['classification_report'] = self.classification_report
            
        return result
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ModelMetrics':
        """Create ModelMetrics from dictionary."""
        metrics = cls(
            model_name=data['model_name'],
            accuracy=data.get('accuracy', 0.0),
            precision=data.get('precision', 0.0),
            recall=data.get('recall', 0.0),
            f1_score=data.get('f1_score', 0.0),
            roc_auc=data.get('roc_auc', 0.0),
            training_time=data.get('training_time', 0.0),
            inference_time=data.get('inference_time', 0.0),
            model_size_mb=data.get('model_size_mb', 0.0),
            timestamp=datetime.fromisoformat(data.get('timestamp', datetime.now().isoformat())),
            metadata=data.get('metadata', {})
        )
        
        if 'confusion_matrix' in data:
            metrics.confusion_matrix = np.array(data['confusion_matrix'])
        
        if 'classification_report' in data:
            metrics.classification_report = data['classification_report']
            
        return metrics


class ModelEvaluator:
    """
    Comprehensive model evaluator with multiple metrics and comparison capabilities.
    """
    
    def __init__(self, class_names: Optional[List[str]] = None):
        self.class_names = class_names or ['No Rockfall', 'Rockfall']
        self.logger = logging.getLogger(__name__)
    
    def evaluate_model(self, model: Any, X_test: np.ndarray, y_test: np.ndarray,
                      model_name: str, metadata: Optional[Dict[str, Any]] = None) -> ModelMetrics:
        """
        Evaluate a single model and return comprehensive metrics.
        
        Args:
            model: Trained model with predict and predict_proba methods
            X_test: Test features
            y_test: Test labels
            model_name: Name of the model
            metadata: Additional metadata about the model
            
        Returns:
            ModelMetrics object with evaluation results
        """
        import time
        
        # Time inference
        start_time = time.time()
        y_pred = model.predict(X_test)
        inference_time = time.time() - start_time
        
        # Get probabilities if available
        try:
            y_proba = model.predict_proba(X_test)[:, 1]
        except (AttributeError, IndexError):
            y_proba = None
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
        recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
        f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
        
        # ROC AUC (only for binary classification with probabilities)
        roc_auc = 0.0
        if y_proba is not None and len(np.unique(y_test)) == 2:
            try:
                roc_auc = roc_auc_score(y_test, y_proba)
            except ValueError:
                roc_auc = 0.0
        
        # Confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        
        # Classification report
        class_report = classification_report(y_test, y_pred, target_names=self.class_names)
        
        # Model size (estimate)
        model_size = self._estimate_model_size(model)
        
        # Create metrics object
        metrics = ModelMetrics(
            model_name=model_name,
            accuracy=accuracy,
            precision=precision,
            recall=recall,
            f1_score=f1,
            roc_auc=roc_auc,
            confusion_matrix=cm,
            classification_report=class_report,
            inference_time=inference_time / len(X_test),  # Per sample
            model_size_mb=model_size,
            metadata=metadata or {}
        )
        
        return metrics
    
    def _estimate_model_size(self, model: Any) -> float:
        """Estimate model size in MB."""
        try:
            # Try to pickle the model to estimate size
            import sys
            import pickle
            import io
            
            buffer = io.BytesIO()
            pickle.dump(model, buffer)
            size_bytes = buffer.tell()
            return size_bytes / (1024 * 1024)  # Convert to MB
        except Exception:
            return 0.0
    
    def compare_models(self, model_metrics: List[ModelMetrics],
                      comparison_metrics: Optional[List[str]] = None) -> pd.DataFrame:
        """
        Compare multiple models and return comparison results.
        
        Args:
            model_metrics: List of ModelMetrics objects
            comparison_metrics: Metrics to include in comparison
            
        Returns:
            DataFrame with comparison results
        """
        if not comparison_metrics:
            comparison_metrics = ['accuracy', 'precision', 'recall', 'f1_score', 'roc_auc']
        
        comparison_data = []
        for metrics in model_metrics:
            row = {'model_name': metrics.model_name}
            for metric in comparison_metrics:
                row[metric] = getattr(metrics, metric, 0.0)
            
            # Add additional useful metrics
            row['training_time'] = metrics.training_time
            row['inference_time_per_sample'] = metrics.inference_time
            row['model_size_mb'] = metrics.model_size_mb
            
            comparison_data.append(row)
        
        df = pd.DataFrame(comparison_data)
        
        # Sort by F1 score (or primary metric)
        primary_metric = 'f1_score' if 'f1_score' in df.columns else comparison_metrics[0]
        df = df.sort_values(primary_metric, ascending=False).reset_index(drop=True)
        
        return df
    
    def rank_models(self, model_metrics: List[ModelMetrics],
                   weights: Optional[Dict[str, float]] = None) -> List[Tuple[str, float]]:
        """
        Rank models based on weighted scoring.
        
        Args:
            model_metrics: List of ModelMetrics objects
            weights: Weights for different metrics
            
        Returns:
            List of tuples (model_name, weighted_score) sorted by score
        """
        if not weights:
            weights = {
                'accuracy': 0.2,
                'precision': 0.2,
                'recall': 0.2,
                'f1_score': 0.3,
                'roc_auc': 0.1
            }
        
        # Normalize weights
        total_weight = sum(weights.values())
        weights = {k: v / total_weight for k, v in weights.items()}
        
        rankings = []
        for metrics in model_metrics:
            score = 0.0
            for metric, weight in weights.items():
                value = getattr(metrics, metric, 0.0)
                score += value * weight
            
            rankings.append((metrics.model_name, score))
        
        # Sort by score (descending)
        rankings.sort(key=lambda x: x[1], reverse=True)
        return rankings
    
    def select_best_model(self, model_metrics: List[ModelMetrics],
                         selection_criteria: str = 'f1_score') -> ModelMetrics:
        """
        Select the best model based on specified criteria.
        
        Args:
            model_metrics: List of ModelMetrics objects
            selection_criteria: Metric to use for selection
            
        Returns:
            Best ModelMetrics object
        """
        if not model_metrics:
            raise ValueError("No model metrics provided")
        
        best_model = max(model_metrics, key=lambda x: getattr(x, selection_criteria, 0.0))
        return best_model


class ModelComparisonSuite:
    """
    Complete suite for model comparison with visualization and reporting.
    """
    
    def __init__(self, output_dir: Path, class_names: Optional[List[str]] = None):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.evaluator = ModelEvaluator(class_names)
        self.logger = logging.getLogger(__name__)
        
        # Storage for results
        self.model_metrics: List[ModelMetrics] = []
        self.comparison_results: Optional[pd.DataFrame] = None
    
    def add_model_evaluation(self, model: Any, X_test: np.ndarray, y_test: np.ndarray,
                           model_name: str, metadata: Optional[Dict[str, Any]] = None):
        """Add a model evaluation to the comparison suite."""
        metrics = self.evaluator.evaluate_model(model, X_test, y_test, model_name, metadata)
        self.model_metrics.append(metrics)
        self.logger.info(f"Added evaluation for model: {model_name}")
    
    def load_model_metrics(self, metrics_file: Path):
        """Load model metrics from file."""
        with open(metrics_file, 'r') as f:
            data = json.load(f)
        
        for item in data:
            metrics = ModelMetrics.from_dict(item)
            self.model_metrics.append(metrics)
    
    def save_model_metrics(self, metrics_file: Optional[Path] = None):
        """Save model metrics to file."""
        if metrics_file is None:
            metrics_file = self.output_dir / "model_metrics.json"
        
        data = [metrics.to_dict() for metrics in self.model_metrics]
        with open(metrics_file, 'w') as f:
            json.dump(data, f, indent=2)
    
    def generate_comparison_report(self) -> pd.DataFrame:
        """Generate comprehensive comparison report."""
        if not self.model_metrics:
            raise ValueError("No model metrics available for comparison")
        
        self.comparison_results = self.evaluator.compare_models(self.model_metrics)
        
        # Save comparison results
        comparison_file = self.output_dir / "model_comparison.csv"
        self.comparison_results.to_csv(comparison_file, index=False)
        
        # Generate rankings
        rankings = self.evaluator.rank_models(self.model_metrics)
        
        # Save rankings
        rankings_file = self.output_dir / "model_rankings.json"
        with open(rankings_file, 'w') as f:
            json.dump(rankings, f, indent=2)
        
        return self.comparison_results
    
    def plot_comparison_metrics(self, metrics: Optional[List[str]] = None,
                               figsize: Tuple[int, int] = (12, 8)):
        """Plot comparison metrics across models."""
        if self.comparison_results is None:
            self.generate_comparison_report()
        
        if metrics is None:
            metrics = ['accuracy', 'precision', 'recall', 'f1_score']
        
        # Create subplots
        fig, axes = plt.subplots(2, 2, figsize=figsize)
        axes = axes.flatten()
        
        for i, metric in enumerate(metrics[:4]):
            if metric in self.comparison_results.columns:
                ax = axes[i]
                self.comparison_results.plot(x='model_name', y=metric, kind='bar', ax=ax)
                ax.set_title(f'{metric.replace("_", " ").title()} Comparison')
                ax.set_xlabel('Model')
                ax.set_ylabel(metric.replace("_", " ").title())
                ax.tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        
        # Save plot
        plot_file = self.output_dir / "model_comparison_metrics.png"
        plt.savefig(plot_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        self.logger.info(f"Comparison plot saved to: {plot_file}")
    
    def plot_confusion_matrices(self, figsize: Tuple[int, int] = (15, 10)):
        """Plot confusion matrices for all models."""
        if not self.model_metrics:
            return
        
        n_models = len(self.model_metrics)
        cols = min(3, n_models)
        rows = (n_models + cols - 1) // cols
        
        fig, axes = plt.subplots(rows, cols, figsize=figsize)
        
        # Handle different subplot configurations
        if n_models == 1:
            axes = [axes]
        elif rows == 1 and cols > 1:
            axes = list(axes)
        elif rows > 1 and cols > 1:
            axes = axes.flatten()
        else:
            axes = [axes]
        
        for i, metrics in enumerate(self.model_metrics):
            if metrics.confusion_matrix is not None and i < len(axes):
                ax = axes[i]
                
                sns.heatmap(metrics.confusion_matrix, annot=True, fmt='d', 
                           xticklabels=self.evaluator.class_names,
                           yticklabels=self.evaluator.class_names, ax=ax)
                ax.set_title(f'{metrics.model_name}')
                ax.set_xlabel('Predicted')
                ax.set_ylabel('Actual')
        
        # Hide empty subplots
        for i in range(n_models, len(axes)):
            if i < len(axes):
                axes[i].set_visible(False)
        
        plt.tight_layout()
        
        # Save plot
        plot_file = self.output_dir / "confusion_matrices.png"
        plt.savefig(plot_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        self.logger.info(f"Confusion matrices saved to: {plot_file}")
    
    def generate_summary_report(self) -> str:
        """Generate a text summary report."""
        if not self.model_metrics:
            return "No model metrics available for summary."
        
        report = []
        report.append("="*60)
        report.append("MODEL COMPARISON SUMMARY REPORT")
        report.append("="*60)
        report.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append(f"Number of models compared: {len(self.model_metrics)}")
        report.append("")
        
        # Best model by different criteria
        criteria = ['accuracy', 'precision', 'recall', 'f1_score', 'roc_auc']
        report.append("BEST MODELS BY CRITERIA:")
        report.append("-" * 40)
        
        for criterion in criteria:
            try:
                best = self.evaluator.select_best_model(self.model_metrics, criterion)
                value = getattr(best, criterion, 0.0)
                report.append(f"Best {criterion}: {best.model_name} ({value:.4f})")
            except Exception:
                continue
        
        report.append("")
        
        # Rankings
        rankings = self.evaluator.rank_models(self.model_metrics)
        report.append("OVERALL RANKINGS (Weighted Score):")
        report.append("-" * 40)
        
        for i, (model_name, score) in enumerate(rankings):
            report.append(f"{i+1:2d}. {model_name:20s} - {score:.4f}")
        
        report.append("")
        
        # Performance summary table
        if self.comparison_results is not None:
            report.append("DETAILED COMPARISON:")
            report.append("-" * 40)
            report.append(self.comparison_results.to_string(index=False))
        
        report_text = "\n".join(report)
        
        # Save report
        report_file = self.output_dir / "comparison_summary.txt"
        with open(report_file, 'w') as f:
            f.write(report_text)
        
        self.logger.info(f"Summary report saved to: {report_file}")
        return report_text


# Example usage and testing
if __name__ == "__main__":
    import tempfile
    import shutil
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.datasets import make_classification
    from sklearn.model_selection import train_test_split
    
    # Create sample data
    X, y = make_classification(n_samples=1000, n_features=20, n_classes=2, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    # Create temporary directory
    temp_dir = Path(tempfile.mkdtemp())
    
    try:
        # Initialize comparison suite
        suite = ModelComparisonSuite(temp_dir)
        
        # Train and evaluate models
        models = {
            'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
            'Logistic Regression': LogisticRegression(random_state=42)
        }
        
        for name, model in models.items():
            print(f"Training {name}...")
            model.fit(X_train, y_train)
            suite.add_model_evaluation(model, X_test, y_test, name)
        
        # Generate reports
        print("\nGenerating comparison report...")
        comparison_df = suite.generate_comparison_report()
        print(comparison_df)
        
        # Generate visualizations
        print("\nGenerating visualizations...")
        suite.plot_comparison_metrics()
        suite.plot_confusion_matrices()
        
        # Generate summary
        print("\nGenerating summary report...")
        summary = suite.generate_summary_report()
        print(summary)
        
        print(f"\nAll outputs saved to: {temp_dir}")
        
    finally:
        # Cleanup
        # shutil.rmtree(temp_dir, ignore_errors=True)
        print("Test completed successfully!")