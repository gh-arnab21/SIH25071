"""
Unit tests for model evaluation and metrics utilities.

Tests cover:
- Classification metrics calculation
- ROC curve generation and AUC calculation
- Confusion matrix visualization and analysis
- Model performance reporting and comparison
"""

import unittest
import numpy as np
import pandas as pd
import tempfile
import os
import json
from unittest.mock import patch, MagicMock
from pathlib import Path

# Import the modules to test
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

from utils.evaluation import (
    ModelEvaluator, ModelPerformanceTracker, 
    quick_evaluate, evaluate_and_save
)


class TestModelEvaluator(unittest.TestCase):
    """Test cases for ModelEvaluator class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.evaluator = ModelEvaluator(['Low', 'Medium', 'High'])
        
        # Create sample data for testing
        np.random.seed(42)
        self.n_samples = 100
        
        # Binary classification data
        self.y_true_binary = np.random.randint(0, 2, self.n_samples)
        self.y_pred_binary = np.random.randint(0, 2, self.n_samples)
        self.y_prob_binary = np.random.rand(self.n_samples, 2)
        
        # Multi-class classification data
        self.y_true_multi = np.random.randint(0, 3, self.n_samples)
        self.y_pred_multi = np.random.randint(0, 3, self.n_samples)
        self.y_prob_multi = np.random.rand(self.n_samples, 3)
        
        # Normalize probabilities
        self.y_prob_binary = self.y_prob_binary / self.y_prob_binary.sum(axis=1, keepdims=True)
        self.y_prob_multi = self.y_prob_multi / self.y_prob_multi.sum(axis=1, keepdims=True)
    
    def test_initialization(self):
        """Test ModelEvaluator initialization."""
        # Test default initialization
        evaluator_default = ModelEvaluator()
        self.assertEqual(evaluator_default.class_names, ['Low', 'Medium', 'High'])
        self.assertEqual(evaluator_default.n_classes, 3)
        
        # Test custom initialization
        custom_classes = ['Class1', 'Class2']
        evaluator_custom = ModelEvaluator(custom_classes)
        self.assertEqual(evaluator_custom.class_names, custom_classes)
        self.assertEqual(evaluator_custom.n_classes, 2)
    
    def test_calculate_classification_metrics_binary(self):
        """Test classification metrics calculation for binary classification."""
        # Use binary evaluator
        binary_evaluator = ModelEvaluator(['Negative', 'Positive'])
        
        metrics = binary_evaluator.calculate_classification_metrics(
            self.y_true_binary, self.y_pred_binary
        )
        
        # Check that all expected metrics are present
        expected_metrics = [
            'accuracy', 'precision_macro', 'precision_micro', 'precision_weighted',
            'recall_macro', 'recall_micro', 'recall_weighted',
            'f1_macro', 'f1_micro', 'f1_weighted'
        ]
        
        for metric in expected_metrics:
            self.assertIn(metric, metrics)
            self.assertIsInstance(metrics[metric], (int, float))
            self.assertGreaterEqual(metrics[metric], 0.0)
            self.assertLessEqual(metrics[metric], 1.0)
        
        # Check per-class metrics
        self.assertIn('precision_negative', metrics)
        self.assertIn('precision_positive', metrics)
        self.assertIn('recall_negative', metrics)
        self.assertIn('recall_positive', metrics)
        self.assertIn('f1_negative', metrics)
        self.assertIn('f1_positive', metrics)
    
    def test_calculate_classification_metrics_multiclass(self):
        """Test classification metrics calculation for multi-class classification."""
        metrics = self.evaluator.calculate_classification_metrics(
            self.y_true_multi, self.y_pred_multi
        )
        
        # Check that all expected metrics are present
        expected_metrics = [
            'accuracy', 'precision_macro', 'precision_micro', 'precision_weighted',
            'recall_macro', 'recall_micro', 'recall_weighted',
            'f1_macro', 'f1_micro', 'f1_weighted'
        ]
        
        for metric in expected_metrics:
            self.assertIn(metric, metrics)
            self.assertIsInstance(metrics[metric], (int, float))
            self.assertGreaterEqual(metrics[metric], 0.0)
            self.assertLessEqual(metrics[metric], 1.0)
        
        # Check per-class metrics
        for class_name in ['low', 'medium', 'high']:
            self.assertIn(f'precision_{class_name}', metrics)
            self.assertIn(f'recall_{class_name}', metrics)
            self.assertIn(f'f1_{class_name}', metrics)
    
    def test_calculate_roc_auc_metrics_binary(self):
        """Test ROC-AUC metrics calculation for binary classification."""
        binary_evaluator = ModelEvaluator(['Negative', 'Positive'])
        
        metrics = binary_evaluator.calculate_roc_auc_metrics(
            self.y_true_binary, self.y_prob_binary
        )
        
        self.assertIn('roc_auc', metrics)
        self.assertIsInstance(metrics['roc_auc'], (int, float))
        self.assertGreaterEqual(metrics['roc_auc'], 0.0)
        self.assertLessEqual(metrics['roc_auc'], 1.0)
    
    def test_calculate_roc_auc_metrics_multiclass(self):
        """Test ROC-AUC metrics calculation for multi-class classification."""
        metrics = self.evaluator.calculate_roc_auc_metrics(
            self.y_true_multi, self.y_prob_multi
        )
        
        # Check macro and weighted AUC
        self.assertIn('roc_auc_macro', metrics)
        self.assertIn('roc_auc_weighted', metrics)
        
        for metric in ['roc_auc_macro', 'roc_auc_weighted']:
            self.assertIsInstance(metrics[metric], (int, float))
            self.assertGreaterEqual(metrics[metric], 0.0)
            self.assertLessEqual(metrics[metric], 1.0)
        
        # Check per-class AUC
        for class_name in ['low', 'medium', 'high']:
            auc_key = f'roc_auc_{class_name}'
            self.assertIn(auc_key, metrics)
            self.assertIsInstance(metrics[auc_key], (int, float))
    
    @patch('matplotlib.pyplot.show')
    @patch('matplotlib.pyplot.savefig')
    def test_generate_roc_curves(self, mock_savefig, mock_show):
        """Test ROC curve generation."""
        roc_data = self.evaluator.generate_roc_curves(
            self.y_true_multi, self.y_prob_multi
        )
        
        self.assertIsInstance(roc_data, dict)
        
        # Check that data is returned for each class
        for class_name in ['low', 'medium', 'high']:
            if class_name in roc_data:
                fpr, tpr, auc_score = roc_data[class_name]
                self.assertIsInstance(fpr, np.ndarray)
                self.assertIsInstance(tpr, np.ndarray)
                self.assertIsInstance(auc_score, (int, float))
                self.assertEqual(len(fpr), len(tpr))
    
    @patch('matplotlib.pyplot.show')
    @patch('matplotlib.pyplot.savefig')
    def test_plot_confusion_matrix(self, mock_savefig, mock_show):
        """Test confusion matrix plotting."""
        cm = self.evaluator.plot_confusion_matrix(
            self.y_true_multi, self.y_pred_multi
        )
        
        self.assertIsInstance(cm, np.ndarray)
        self.assertEqual(cm.shape, (3, 3))  # 3x3 for 3 classes
        
        # Test normalized confusion matrix
        cm_normalized = self.evaluator.plot_confusion_matrix(
            self.y_true_multi, self.y_pred_multi, normalize='true'
        )
        
        self.assertIsInstance(cm_normalized, np.ndarray)
        # Check that rows sum to approximately 1 (allowing for floating point errors)
        row_sums = cm_normalized.sum(axis=1)
        for row_sum in row_sums:
            if row_sum > 0:  # Only check non-zero rows
                self.assertAlmostEqual(row_sum, 1.0, places=5)
    
    def test_generate_classification_report(self):
        """Test classification report generation."""
        # Test string output
        report_str = self.evaluator.generate_classification_report(
            self.y_true_multi, self.y_pred_multi
        )
        self.assertIsInstance(report_str, str)
        self.assertIn('precision', report_str)
        self.assertIn('recall', report_str)
        self.assertIn('f1-score', report_str)
        
        # Test dictionary output
        report_dict = self.evaluator.generate_classification_report(
            self.y_true_multi, self.y_pred_multi, output_dict=True
        )
        self.assertIsInstance(report_dict, dict)
        self.assertIn('accuracy', report_dict)
        self.assertIn('macro avg', report_dict)
        self.assertIn('weighted avg', report_dict)
    
    @patch('matplotlib.pyplot.show')
    @patch('matplotlib.pyplot.savefig')
    def test_calculate_precision_recall_curves(self, mock_savefig, mock_show):
        """Test precision-recall curve calculation."""
        pr_data = self.evaluator.calculate_precision_recall_curves(
            self.y_true_multi, self.y_prob_multi
        )
        
        self.assertIsInstance(pr_data, dict)
        
        # Check that data is returned for each class
        for class_name in ['low', 'medium', 'high']:
            if class_name in pr_data:
                precision, recall, ap_score = pr_data[class_name]
                self.assertIsInstance(precision, np.ndarray)
                self.assertIsInstance(recall, np.ndarray)
                self.assertIsInstance(ap_score, (int, float))
                self.assertEqual(len(precision), len(recall))
    
    @patch('matplotlib.pyplot.show')
    @patch('matplotlib.pyplot.savefig')
    def test_comprehensive_evaluation(self, mock_savefig, mock_show):
        """Test comprehensive evaluation function."""
        results = self.evaluator.comprehensive_evaluation(
            self.y_true_multi, self.y_pred_multi, self.y_prob_multi
        )
        
        # Check that all expected result keys are present
        expected_keys = [
            'classification_metrics', 'classification_report', 
            'confusion_matrix', 'roc_auc_metrics', 'roc_curves',
            'precision_recall_curves'
        ]
        
        for key in expected_keys:
            self.assertIn(key, results)
        
        # Check types
        self.assertIsInstance(results['classification_metrics'], dict)
        self.assertIsInstance(results['classification_report'], dict)
        self.assertIsInstance(results['confusion_matrix'], np.ndarray)
        self.assertIsInstance(results['roc_auc_metrics'], dict)
        self.assertIsInstance(results['roc_curves'], dict)
        self.assertIsInstance(results['precision_recall_curves'], dict)
    
    def test_compare_models(self):
        """Test model comparison functionality."""
        # Create mock results for two models
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
        
        model_results = {
            'Model1': model1_results,
            'Model2': model2_results
        }
        
        comparison_df = self.evaluator.compare_models(model_results)
        
        self.assertIsInstance(comparison_df, pd.DataFrame)
        self.assertEqual(len(comparison_df), 2)
        self.assertIn('Model', comparison_df.columns)
        self.assertIn('accuracy', comparison_df.columns)
        self.assertIn('f1_weighted', comparison_df.columns)
    
    def test_save_evaluation_report(self):
        """Test evaluation report saving."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create mock results
            results = {
                'classification_metrics': {'accuracy': 0.85},
                'roc_auc_metrics': {'roc_auc_macro': 0.82},
                'classification_report': {'accuracy': 0.85},
                'confusion_matrix': np.array([[10, 2], [3, 15]])
            }
            
            report_path = os.path.join(temp_dir, 'test_report.json')
            
            self.evaluator.save_evaluation_report(
                results, 'TestModel', report_path
            )
            
            # Check that file was created
            self.assertTrue(os.path.exists(report_path))
            
            # Check file contents
            with open(report_path, 'r') as f:
                saved_report = json.load(f)
            
            self.assertEqual(saved_report['model_name'], 'TestModel')
            self.assertIn('evaluation_timestamp', saved_report)
            self.assertIn('metrics', saved_report)


class TestModelPerformanceTracker(unittest.TestCase):
    """Test cases for ModelPerformanceTracker class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.json')
        self.temp_file.close()
        self.tracker = ModelPerformanceTracker(self.temp_file.name)
    
    def tearDown(self):
        """Clean up test fixtures."""
        if os.path.exists(self.temp_file.name):
            os.unlink(self.temp_file.name)
    
    def test_initialization(self):
        """Test ModelPerformanceTracker initialization."""
        self.assertIsInstance(self.tracker.history, list)
        self.assertEqual(len(self.tracker.history), 0)
    
    def test_add_evaluation(self):
        """Test adding evaluation results."""
        metrics = {
            'accuracy': 0.85,
            'f1_weighted': 0.83,
            'precision_weighted': 0.84
        }
        
        metadata = {
            'learning_rate': 0.001,
            'batch_size': 32
        }
        
        self.tracker.add_evaluation(
            'TestModel', 'TestDataset', metrics, metadata
        )
        
        self.assertEqual(len(self.tracker.history), 1)
        
        entry = self.tracker.history[0]
        self.assertEqual(entry['model_name'], 'TestModel')
        self.assertEqual(entry['dataset_name'], 'TestDataset')
        self.assertEqual(entry['metrics'], metrics)
        self.assertEqual(entry['metadata'], metadata)
        self.assertIn('timestamp', entry)
    
    def test_get_best_model(self):
        """Test getting best model functionality."""
        # Add multiple evaluations
        self.tracker.add_evaluation(
            'Model1', 'Dataset1', {'f1_weighted': 0.80}
        )
        self.tracker.add_evaluation(
            'Model2', 'Dataset1', {'f1_weighted': 0.85}
        )
        self.tracker.add_evaluation(
            'Model3', 'Dataset1', {'f1_weighted': 0.82}
        )
        
        best_model = self.tracker.get_best_model('f1_weighted')
        
        self.assertIsNotNone(best_model)
        self.assertEqual(best_model['model_name'], 'Model2')
        self.assertEqual(best_model['metrics']['f1_weighted'], 0.85)
    
    def test_get_performance_trends(self):
        """Test performance trends functionality."""
        # Add multiple evaluations for the same model
        self.tracker.add_evaluation(
            'TestModel', 'Dataset1', {'accuracy': 0.80}
        )
        self.tracker.add_evaluation(
            'TestModel', 'Dataset2', {'accuracy': 0.85}
        )
        self.tracker.add_evaluation(
            'OtherModel', 'Dataset1', {'accuracy': 0.75}
        )
        
        trends_df = self.tracker.get_performance_trends('TestModel')
        
        self.assertIsInstance(trends_df, pd.DataFrame)
        self.assertEqual(len(trends_df), 2)
        self.assertIn('timestamp', trends_df.columns)
        self.assertIn('accuracy', trends_df.columns)
        self.assertIn('dataset_name', trends_df.columns)
    
    def test_persistence(self):
        """Test that data persists across instances."""
        # Add data to first instance
        metrics = {'accuracy': 0.85}
        self.tracker.add_evaluation('TestModel', 'TestDataset', metrics)
        
        # Create new instance with same file
        new_tracker = ModelPerformanceTracker(self.temp_file.name)
        
        # Check that data was loaded
        self.assertEqual(len(new_tracker.history), 1)
        self.assertEqual(new_tracker.history[0]['model_name'], 'TestModel')


class TestUtilityFunctions(unittest.TestCase):
    """Test cases for utility functions."""
    
    def setUp(self):
        """Set up test fixtures."""
        np.random.seed(42)
        self.n_samples = 50
        self.y_true = np.random.randint(0, 3, self.n_samples)
        self.y_pred = np.random.randint(0, 3, self.n_samples)
        self.y_prob = np.random.rand(self.n_samples, 3)
        self.y_prob = self.y_prob / self.y_prob.sum(axis=1, keepdims=True)
    
    @patch('matplotlib.pyplot.show')
    def test_quick_evaluate(self, mock_show):
        """Test quick_evaluate function."""
        results = quick_evaluate(
            self.y_true, self.y_pred, self.y_prob
        )
        
        self.assertIsInstance(results, dict)
        self.assertIn('classification_metrics', results)
        self.assertIn('confusion_matrix', results)
    
    @patch('matplotlib.pyplot.show')
    @patch('matplotlib.pyplot.savefig')
    def test_evaluate_and_save(self, mock_savefig, mock_show):
        """Test evaluate_and_save function."""
        with tempfile.TemporaryDirectory() as temp_dir:
            results = evaluate_and_save(
                self.y_true, self.y_pred, 'TestModel',
                self.y_prob, save_dir=temp_dir
            )
            
            self.assertIsInstance(results, dict)
            
            # Check that report file was created
            report_file = os.path.join(temp_dir, 'TestModel_evaluation_report.json')
            self.assertTrue(os.path.exists(report_file))


if __name__ == '__main__':
    unittest.main()