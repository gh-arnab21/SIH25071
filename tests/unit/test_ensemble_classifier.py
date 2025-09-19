"""
Unit tests for the Ensemble Classifier module.

This module tests the EnsembleClassifier, EnsembleConfig, and NeuralNetworkClassifier
classes to ensure proper functionality for rockfall risk prediction.
"""

import pytest
import numpy as np
import pandas as pd
from unittest.mock import patch, MagicMock
import tempfile
import os
from pathlib import Path

# Import the classes to test
from src.models.classifiers.ensemble_classifier import (
    EnsembleClassifier, 
    EnsembleConfig, 
    NeuralNetworkClassifier
)
from src.data.schema import RiskLevel, RockfallPrediction


class TestEnsembleConfig:
    """Test cases for EnsembleConfig class."""
    
    def test_default_config(self):
        """Test default configuration values."""
        config = EnsembleConfig()
        
        # Test Random Forest defaults
        assert config.rf_n_estimators == 100
        assert config.rf_max_depth is None
        assert config.rf_min_samples_split == 2
        assert config.rf_min_samples_leaf == 1
        assert config.rf_random_state == 42
        
        # Test XGBoost defaults
        assert config.xgb_n_estimators == 100
        assert config.xgb_max_depth == 6
        assert config.xgb_learning_rate == 0.1
        assert config.xgb_subsample == 0.8
        assert config.xgb_colsample_bytree == 0.8
        assert config.xgb_random_state == 42
        
        # Test Neural Network defaults
        assert config.nn_hidden_layers == [128, 64, 32]
        assert config.nn_dropout_rate == 0.3
        assert config.nn_learning_rate == 0.001
        assert config.nn_batch_size == 32
        assert config.nn_epochs == 100
        assert config.nn_early_stopping_patience == 10
        
        # Test Ensemble defaults
        assert config.ensemble_method == "voting"
        assert config.voting_type == "soft"
        assert config.cv_folds == 5
        assert config.random_state == 42
    
    def test_custom_config(self):
        """Test custom configuration values."""
        config = EnsembleConfig(
            rf_n_estimators=200,
            xgb_learning_rate=0.05,
            nn_hidden_layers=[64, 32],
            ensemble_method="stacking"
        )
        
        assert config.rf_n_estimators == 200
        assert config.xgb_learning_rate == 0.05
        assert config.nn_hidden_layers == [64, 32]
        assert config.ensemble_method == "stacking"


class TestNeuralNetworkClassifier:
    """Test cases for NeuralNetworkClassifier class."""
    
    @pytest.fixture
    def sample_data(self):
        """Create sample data for testing."""
        np.random.seed(42)
        X = np.random.randn(100, 10)
        y = np.random.randint(0, 3, 100)
        return X, y
    
    def test_initialization(self):
        """Test neural network classifier initialization."""
        nn = NeuralNetworkClassifier(
            hidden_layers=[64, 32],
            dropout_rate=0.2,
            learning_rate=0.01,
            batch_size=16,
            epochs=50,
            random_state=123
        )
        
        assert nn.hidden_layers == [64, 32]
        assert nn.dropout_rate == 0.2
        assert nn.learning_rate == 0.01
        assert nn.batch_size == 16
        assert nn.epochs == 50
        assert nn.random_state == 123
        assert nn.model is None
        assert nn.classes_ is None
    
    def test_fit_and_predict(self, sample_data):
        """Test fitting and prediction."""
        X, y = sample_data
        
        nn = NeuralNetworkClassifier(
            hidden_layers=[32, 16],
            epochs=5,  # Reduced for faster testing
            early_stopping_patience=2
        )
        
        # Test fitting
        nn.fit(X, y)
        
        assert nn.model is not None
        assert nn.classes_ is not None
        assert nn.n_classes_ == 3
        assert len(nn.classes_) == 3
        
        # Test prediction
        predictions = nn.predict(X)
        assert len(predictions) == len(X)
        assert all(pred in [0, 1, 2] for pred in predictions)
        
        # Test probability prediction
        probabilities = nn.predict_proba(X)
        assert probabilities.shape == (len(X), 3)
        assert np.allclose(probabilities.sum(axis=1), 1.0, atol=1e-6)
    
    def test_binary_classification(self):
        """Test binary classification scenario."""
        np.random.seed(42)
        X = np.random.randn(50, 5)
        y = np.random.randint(0, 2, 50)
        
        nn = NeuralNetworkClassifier(
            hidden_layers=[16],
            epochs=3,
            early_stopping_patience=1
        )
        
        nn.fit(X, y)
        
        assert nn.n_classes_ == 2
        
        predictions = nn.predict(X)
        probabilities = nn.predict_proba(X)
        
        assert len(predictions) == len(X)
        assert probabilities.shape == (len(X), 2)
        assert all(pred in [0, 1] for pred in predictions)
    
    def test_predict_before_fit_raises_error(self, sample_data):
        """Test that prediction before fitting raises error."""
        X, _ = sample_data
        nn = NeuralNetworkClassifier()
        
        with pytest.raises(ValueError, match="Model not fitted yet"):
            nn.predict(X)
        
        with pytest.raises(ValueError, match="Model not fitted yet"):
            nn.predict_proba(X)


class TestEnsembleClassifier:
    """Test cases for EnsembleClassifier class."""
    
    @pytest.fixture
    def sample_data(self):
        """Create sample data for testing."""
        np.random.seed(42)
        X = np.random.randn(100, 15)
        y = np.random.randint(0, 3, 100)
        feature_names = [f"feature_{i}" for i in range(15)]
        return X, y, feature_names
    
    @pytest.fixture
    def risk_level_data(self):
        """Create sample data with RiskLevel enums."""
        np.random.seed(42)
        X = np.random.randn(50, 10)
        y = [RiskLevel(np.random.randint(0, 3)) for _ in range(50)]
        return X, y
    
    def test_initialization(self):
        """Test ensemble classifier initialization."""
        config = EnsembleConfig(rf_n_estimators=50, xgb_n_estimators=50)
        ensemble = EnsembleClassifier(config)
        
        assert ensemble.config == config
        assert ensemble.rf_classifier is not None
        assert ensemble.xgb_classifier is not None
        assert ensemble.nn_classifier is not None
        assert not ensemble.is_fitted
        assert ensemble.feature_names is None
    
    def test_initialization_with_default_config(self):
        """Test initialization with default configuration."""
        ensemble = EnsembleClassifier()
        
        assert isinstance(ensemble.config, EnsembleConfig)
        assert ensemble.config.rf_n_estimators == 100
        assert ensemble.config.ensemble_method == "voting"
    
    @patch('src.models.classifiers.ensemble_classifier.logger')
    def test_fit_and_predict(self, mock_logger, sample_data):
        """Test fitting and prediction with ensemble classifier."""
        X, y, feature_names = sample_data
        
        # Use smaller models for faster testing
        config = EnsembleConfig(
            rf_n_estimators=10,
            xgb_n_estimators=10,
            nn_epochs=3,
            nn_early_stopping_patience=1
        )
        ensemble = EnsembleClassifier(config)
        
        # Test fitting
        ensemble.fit(X, y, feature_names)
        
        assert ensemble.is_fitted
        assert ensemble.feature_names == feature_names
        assert ensemble.classes_ is not None
        assert len(ensemble.classes_) == 3
        
        # Test prediction
        predictions = ensemble.predict(X)
        assert len(predictions) == len(X)
        assert all(pred in [0, 1, 2] for pred in predictions)
        
        # Test probability prediction
        probabilities = ensemble.predict_proba(X)
        assert probabilities.shape == (len(X), 3)
        assert np.allclose(probabilities.sum(axis=1), 1.0, atol=1e-6)
    
    def test_fit_with_risk_level_enums(self, risk_level_data):
        """Test fitting with RiskLevel enum targets."""
        X, y = risk_level_data
        
        config = EnsembleConfig(
            rf_n_estimators=5,
            xgb_n_estimators=5,
            nn_epochs=2,
            nn_early_stopping_patience=1
        )
        ensemble = EnsembleClassifier(config)
        
        ensemble.fit(X, y)
        
        assert ensemble.is_fitted
        predictions = ensemble.predict(X)
        assert all(pred in [0, 1, 2] for pred in predictions)
    
    def test_predict_with_confidence(self, sample_data):
        """Test prediction with confidence scores and uncertainty estimates."""
        X, y, feature_names = sample_data
        
        config = EnsembleConfig(
            rf_n_estimators=5,
            xgb_n_estimators=5,
            nn_epochs=2,
            nn_early_stopping_patience=1
        )
        ensemble = EnsembleClassifier(config)
        ensemble.fit(X, y, feature_names)
        
        # Test prediction with confidence
        predictions = ensemble.predict_with_confidence(X[:5])  # Test with subset
        
        assert len(predictions) == 5
        for pred in predictions:
            assert isinstance(pred, RockfallPrediction)
            assert isinstance(pred.risk_level, RiskLevel)
            assert 0 <= pred.confidence_score <= 1
            assert pred.uncertainty_estimate is not None
            assert pred.uncertainty_estimate >= 0
            assert isinstance(pred.contributing_factors, dict)
            assert pred.model_version == "ensemble_v1.0"
    
    def test_get_feature_importance(self, sample_data):
        """Test feature importance extraction."""
        X, y, feature_names = sample_data
        
        config = EnsembleConfig(
            rf_n_estimators=5,
            xgb_n_estimators=5,
            nn_epochs=2
        )
        ensemble = EnsembleClassifier(config)
        ensemble.fit(X, y, feature_names)
        
        importance = ensemble.get_feature_importance()
        
        assert isinstance(importance, dict)
        assert len(importance) == len(feature_names)
        assert all(name in importance for name in feature_names)
        assert all(isinstance(score, float) for score in importance.values())
        assert all(score >= 0 for score in importance.values())
    
    def test_evaluate(self, sample_data):
        """Test model evaluation."""
        X, y, feature_names = sample_data
        
        config = EnsembleConfig(
            rf_n_estimators=5,
            xgb_n_estimators=5,
            nn_epochs=2
        )
        ensemble = EnsembleClassifier(config)
        ensemble.fit(X, y, feature_names)
        
        # Evaluate on training data (just for testing)
        results = ensemble.evaluate(X, y)
        
        assert isinstance(results, dict)
        assert 'accuracy' in results
        assert 'classification_report' in results
        assert 'confusion_matrix' in results
        assert 0 <= results['accuracy'] <= 1
        assert isinstance(results['classification_report'], dict)
        assert isinstance(results['confusion_matrix'], list)
    
    def test_cross_validate(self, sample_data):
        """Test cross-validation."""
        X, y, _ = sample_data
        
        config = EnsembleConfig(
            rf_n_estimators=5,
            xgb_n_estimators=5,
            nn_epochs=2,
            cv_folds=3  # Reduced for faster testing
        )
        ensemble = EnsembleClassifier(config)
        
        cv_results = ensemble.cross_validate(X, y)
        
        assert isinstance(cv_results, dict)
        expected_keys = [
            'rf_mean_accuracy', 'rf_std_accuracy',
            'xgb_mean_accuracy', 'xgb_std_accuracy',
            'nn_mean_accuracy', 'nn_std_accuracy'
        ]
        
        for key in expected_keys:
            assert key in cv_results
            assert isinstance(cv_results[key], float)
            assert 0 <= cv_results[key] <= 1 if 'mean' in key else cv_results[key] >= 0
    
    def test_stacking_ensemble_method(self, sample_data):
        """Test stacking ensemble method."""
        X, y, feature_names = sample_data
        
        config = EnsembleConfig(
            rf_n_estimators=5,
            xgb_n_estimators=5,
            nn_epochs=2,
            ensemble_method="stacking"
        )
        ensemble = EnsembleClassifier(config)
        ensemble.fit(X, y, feature_names)
        
        predictions = ensemble.predict(X)
        probabilities = ensemble.predict_proba(X)
        
        assert len(predictions) == len(X)
        assert probabilities.shape == (len(X), 3)
        assert all(pred in [0, 1, 2] for pred in predictions)
    
    def test_predict_before_fit_raises_error(self, sample_data):
        """Test that prediction before fitting raises error."""
        X, _, _ = sample_data
        ensemble = EnsembleClassifier()
        
        with pytest.raises(ValueError, match="Ensemble classifier not fitted yet"):
            ensemble.predict(X)
        
        with pytest.raises(ValueError, match="Ensemble classifier not fitted yet"):
            ensemble.predict_proba(X)
        
        with pytest.raises(ValueError, match="Ensemble classifier not fitted yet"):
            ensemble.predict_with_confidence(X)
        
        with pytest.raises(ValueError, match="Ensemble classifier not fitted yet"):
            ensemble.get_feature_importance()
        
        with pytest.raises(ValueError, match="Ensemble classifier not fitted yet"):
            ensemble.evaluate(X, [0] * len(X))
    
    def test_save_and_load(self, sample_data):
        """Test saving and loading ensemble classifier."""
        X, y, feature_names = sample_data
        
        config = EnsembleConfig(
            rf_n_estimators=5,
            xgb_n_estimators=5,
            nn_epochs=2
        )
        ensemble = EnsembleClassifier(config)
        ensemble.fit(X, y, feature_names)
        
        # Test saving
        with tempfile.TemporaryDirectory() as temp_dir:
            filepath = Path(temp_dir) / "test_ensemble.pkl"
            
            ensemble.save(filepath)
            assert filepath.exists()
            
            # Test loading
            loaded_ensemble = EnsembleClassifier.load(filepath)
            
            assert loaded_ensemble.is_fitted
            assert loaded_ensemble.feature_names == feature_names
            assert np.array_equal(loaded_ensemble.classes_, ensemble.classes_)
            
            # Test that loaded model makes same predictions
            original_pred = ensemble.predict(X[:10])
            loaded_pred = loaded_ensemble.predict(X[:10])
            assert np.array_equal(original_pred, loaded_pred)
    
    def test_save_unfitted_model_raises_error(self):
        """Test that saving unfitted model raises error."""
        ensemble = EnsembleClassifier()
        
        with tempfile.TemporaryDirectory() as temp_dir:
            filepath = Path(temp_dir) / "test_ensemble.pkl"
            
            with pytest.raises(ValueError, match="Cannot save unfitted ensemble classifier"):
                ensemble.save(filepath)
    
    def test_load_nonexistent_file_raises_error(self):
        """Test that loading nonexistent file raises error."""
        with pytest.raises(FileNotFoundError, match="Model file not found"):
            EnsembleClassifier.load("nonexistent_file.pkl")
    
    def test_pandas_dataframe_input(self, sample_data):
        """Test that pandas DataFrame input works correctly."""
        X, y, feature_names = sample_data
        
        # Convert to DataFrame
        X_df = pd.DataFrame(X, columns=feature_names)
        y_series = pd.Series(y)
        
        config = EnsembleConfig(
            rf_n_estimators=5,
            xgb_n_estimators=5,
            nn_epochs=2
        )
        ensemble = EnsembleClassifier(config)
        ensemble.fit(X_df, y_series, feature_names)
        
        predictions = ensemble.predict(X_df)
        probabilities = ensemble.predict_proba(X_df)
        
        assert len(predictions) == len(X_df)
        assert probabilities.shape == (len(X_df), 3)
    
    def test_empty_input_handling(self):
        """Test handling of empty inputs."""
        ensemble = EnsembleClassifier()
        
        # Test with empty arrays
        X_empty = np.array([]).reshape(0, 5)
        y_empty = np.array([])
        
        # This should not crash but may not be very useful
        try:
            ensemble.fit(X_empty, y_empty)
        except ValueError:
            # It's acceptable for this to raise a ValueError
            pass


if __name__ == "__main__":
    pytest.main([__file__])