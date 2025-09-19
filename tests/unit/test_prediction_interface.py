"""
Unit tests for the Prediction Interface module.

This module tests the PredictionInterface class to ensure proper functionality
for making predictions, handling various input formats, and providing accurate results.
"""

import pytest
import numpy as np
import pandas as pd
import tempfile
import os
from unittest.mock import patch, MagicMock, Mock
from datetime import datetime
from pathlib import Path

# Import the classes to test
from src.models.prediction_interface import PredictionInterface, PredictionError
from src.models.classifiers.ensemble_classifier import EnsembleClassifier, EnsembleConfig
from src.models.persistence import ModelPersistence
from src.data.schema import (
    RockfallDataPoint, RockfallPrediction, RiskLevel, 
    GeoCoordinate, EnvironmentalData, SensorData, TimeSeries
)
from src.data.quality import RobustDataProcessor


class TestPredictionInterface:
    """Test cases for PredictionInterface class."""
    
    @pytest.fixture
    def mock_ensemble_model(self):
        """Create a mock ensemble classifier."""
        mock_model = Mock(spec=EnsembleClassifier)
        mock_model.predict_with_confidence.return_value = [
            RockfallPrediction(
                risk_level=RiskLevel.HIGH,
                confidence_score=0.85,
                contributing_factors={"feature_0": 0.3, "feature_1": 0.7},
                uncertainty_estimate=0.15,
                model_version="test_v1.0"
            )
        ]
        mock_model.predict.return_value = np.array([2])  # HIGH risk
        mock_model.predict_proba.return_value = np.array([[0.1, 0.05, 0.85]])
        return mock_model
    
    @pytest.fixture
    def mock_generic_model(self):
        """Create a mock generic classifier."""
        mock_model = Mock()
        mock_model.predict.return_value = np.array([1])  # MEDIUM risk
        mock_model.predict_proba.return_value = np.array([[0.2, 0.7, 0.1]])
        return mock_model
    
    @pytest.fixture
    def sample_feature_matrix(self):
        """Create sample feature matrix."""
        np.random.seed(42)
        return np.random.randn(5, 15)
    
    @pytest.fixture
    def sample_dataframe(self):
        """Create sample DataFrame."""
        np.random.seed(42)
        data = np.random.randn(3, 10)
        columns = [f"feature_{i}" for i in range(10)]
        return pd.DataFrame(data, columns=columns)
    
    @pytest.fixture
    def sample_rockfall_datapoint(self):
        """Create sample RockfallDataPoint."""
        return RockfallDataPoint(
            timestamp=datetime.now(),
            location=GeoCoordinate(latitude=45.0, longitude=-120.0, elevation=1000.0),
            environmental=EnvironmentalData(
                rainfall=10.5,
                temperature=15.2,
                vibrations=0.1,
                wind_speed=5.0,
                humidity=65.0
            ),
            sensor_readings=SensorData(
                displacement=TimeSeries(
                    timestamps=[datetime.now()],
                    values=[0.05],
                    unit="mm"
                )
            )
        )
    
    def test_initialization_empty(self):
        """Test initialization without model or preprocessor paths."""
        interface = PredictionInterface()
        
        assert interface.model_path is None
        assert interface.preprocessor_path is None
        assert interface.model is None
        assert interface.preprocessor is None
        assert not interface.is_loaded
    
    def test_initialization_with_paths(self):
        """Test initialization with model and preprocessor paths."""
        with patch.object(PredictionInterface, 'load_model') as mock_load_model, \
             patch.object(PredictionInterface, 'load_preprocessor') as mock_load_preprocessor:
            
            interface = PredictionInterface(
                model_path="/path/to/model.pkl",
                preprocessor_path="/path/to/preprocessor.pkl"
            )
            
            mock_load_model.assert_called_once_with("/path/to/model.pkl")
            mock_load_preprocessor.assert_called_once_with("/path/to/preprocessor.pkl")
    
    @patch('src.models.prediction_interface.ModelPersistence')
    def test_load_model_success(self, mock_persistence_class):
        """Test successful model loading."""
        mock_persistence = mock_persistence_class.return_value
        mock_model = MagicMock()
        mock_model.predict = MagicMock()
        
        mock_persistence.load_model.return_value = {
            'model': mock_model,
            'feature_names': ['feature_0', 'feature_1']
        }
        
        interface = PredictionInterface()
        interface.load_model("/path/to/model.pkl")
        
        assert interface.model == mock_model
        assert interface.feature_names == ['feature_0', 'feature_1']
        assert interface.is_loaded
        assert interface.model_path == "/path/to/model.pkl"
    
    @patch('src.models.prediction_interface.ModelPersistence')
    def test_load_model_failure(self, mock_persistence_class):
        """Test model loading failure."""
        mock_persistence = mock_persistence_class.return_value
        mock_persistence.load_model.side_effect = Exception("File not found")
        
        interface = PredictionInterface()
        
        with pytest.raises(PredictionError, match="Failed to load model"):
            interface.load_model("/path/to/nonexistent.pkl")
    
    @patch('src.models.prediction_interface.ModelPersistence')
    def test_load_preprocessor_success(self, mock_persistence_class):
        """Test successful preprocessor loading."""
        mock_persistence = mock_persistence_class.return_value
        mock_preprocessor = MagicMock()
        
        mock_persistence.load_preprocessing_pipeline.return_value = {
            'pipeline': mock_preprocessor
        }
        
        interface = PredictionInterface()
        interface.load_preprocessor("/path/to/preprocessor.pkl")
        
        assert interface.preprocessor == mock_preprocessor
        assert interface.preprocessor_path == "/path/to/preprocessor.pkl"
    
    def test_validate_inputs_not_loaded(self):
        """Test input validation when model not loaded."""
        interface = PredictionInterface()
        
        with pytest.raises(PredictionError, match="Model not loaded"):
            interface._validate_inputs(np.array([[1, 2, 3]]))
    
    def test_validate_inputs_numpy_array(self, mock_ensemble_model):
        """Test input validation with numpy array."""
        interface = PredictionInterface()
        interface.model = mock_ensemble_model
        interface.is_loaded = True
        interface.feature_names = ['f1', 'f2', 'f3']
        
        # 2D array
        X = np.array([[1, 2, 3], [4, 5, 6]])
        result = interface._validate_inputs(X)
        assert result.shape == (2, 3)
        
        # 1D array (should be reshaped)
        X = np.array([1, 2, 3])
        result = interface._validate_inputs(X)
        assert result.shape == (1, 3)
    
    def test_validate_inputs_dataframe(self, mock_ensemble_model):
        """Test input validation with pandas DataFrame."""
        interface = PredictionInterface()
        interface.model = mock_ensemble_model
        interface.is_loaded = True
        interface.feature_names = ['f1', 'f2', 'f3']
        
        df = pd.DataFrame([[1, 2, 3], [4, 5, 6]], columns=['f1', 'f2', 'f3'])
        result = interface._validate_inputs(df)
        assert result.shape == (2, 3)
    
    def test_validate_inputs_wrong_dimensions(self, mock_ensemble_model):
        """Test input validation with wrong number of features."""
        interface = PredictionInterface()
        interface.model = mock_ensemble_model
        interface.is_loaded = True
        interface.feature_names = ['f1', 'f2', 'f3']
        
        X = np.array([[1, 2]])  # Only 2 features, expected 3
        
        with pytest.raises(PredictionError, match="Input has 2 features, expected 3"):
            interface._validate_inputs(X)
    
    def test_extract_features_from_datapoints(self, mock_ensemble_model, sample_rockfall_datapoint):
        """Test feature extraction from RockfallDataPoint objects."""
        interface = PredictionInterface()
        interface.model = mock_ensemble_model
        interface.is_loaded = True
        interface.feature_names = [f'feature_{i}' for i in range(15)]
        
        datapoints = [sample_rockfall_datapoint]
        result = interface._extract_features_from_datapoints(datapoints)
        
        assert result.shape == (1, 15)
        assert isinstance(result, np.ndarray)
        
        # Check that location features are included
        assert result[0, 0] == 45.0  # latitude
        assert result[0, 1] == -120.0  # longitude
        assert result[0, 2] == 1000.0  # elevation
    
    @patch('src.models.prediction_interface.RobustDataProcessor')
    def test_preprocess_data_with_default_processor(self, mock_processor_class, mock_ensemble_model):
        """Test data preprocessing with default processor."""
        mock_processor = mock_processor_class.return_value
        mock_processor.process_data.return_value = {'processed_data': np.array([[1, 2, 3]])}
        
        interface = PredictionInterface()
        interface.model = mock_ensemble_model
        interface.preprocessor = None
        
        X = np.array([[4, 5, 6]])
        result = interface._preprocess_data(X)
        
        mock_processor.process_data.assert_called_once_with(X)
        assert np.array_equal(result, np.array([[1, 2, 3]]))
    
    def test_preprocess_data_with_loaded_processor(self, mock_ensemble_model):
        """Test data preprocessing with loaded processor."""
        mock_preprocessor = MagicMock()
        mock_preprocessor.transform.return_value = np.array([[7, 8, 9]])
        
        interface = PredictionInterface()
        interface.model = mock_ensemble_model
        interface.preprocessor = mock_preprocessor
        
        X = np.array([[1, 2, 3]])
        result = interface._preprocess_data(X)
        
        mock_preprocessor.transform.assert_called_once_with(X)
        assert np.array_equal(result, np.array([[7, 8, 9]]))
    
    def test_predict_with_ensemble_model(self, mock_ensemble_model, sample_feature_matrix):
        """Test prediction with ensemble classifier."""
        interface = PredictionInterface()
        interface.model = mock_ensemble_model
        interface.is_loaded = True
        interface.feature_names = [f'feature_{i}' for i in range(15)]
        
        # Mock preprocessing to return the input unchanged
        with patch.object(interface, '_preprocess_data', return_value=sample_feature_matrix):
            predictions = interface.predict(sample_feature_matrix)
        
        assert len(predictions) == 1  # Mock returns 1 prediction
        assert isinstance(predictions[0], RockfallPrediction)
        assert predictions[0].risk_level == RiskLevel.HIGH
        assert predictions[0].confidence_score == 0.85
    
    def test_predict_with_generic_model(self, mock_generic_model, sample_feature_matrix):
        """Test prediction with generic classifier."""
        interface = PredictionInterface()
        interface.model = mock_generic_model
        interface.is_loaded = True
        interface.feature_names = [f'feature_{i}' for i in range(15)]
        
        # Mock preprocessing to return the input unchanged
        with patch.object(interface, '_preprocess_data', return_value=sample_feature_matrix[:1]):
            predictions = interface.predict(sample_feature_matrix[:1])
        
        assert len(predictions) == 1
        assert isinstance(predictions[0], RockfallPrediction)
        assert predictions[0].risk_level == RiskLevel.MEDIUM
        assert predictions[0].confidence_score == 0.7
    
    def test_predict_single_numpy_array(self, mock_ensemble_model):
        """Test single prediction with numpy array."""
        interface = PredictionInterface()
        interface.model = mock_ensemble_model
        interface.is_loaded = True
        interface.feature_names = [f'feature_{i}' for i in range(5)]
        
        X = np.array([1, 2, 3, 4, 5])
        
        with patch.object(interface, 'predict', return_value=[
            RockfallPrediction(risk_level=RiskLevel.LOW, confidence_score=0.9)
        ]) as mock_predict:
            prediction = interface.predict_single(X)
        
        assert isinstance(prediction, RockfallPrediction)
        mock_predict.assert_called_once()
    
    def test_predict_single_datapoint(self, mock_ensemble_model, sample_rockfall_datapoint):
        """Test single prediction with RockfallDataPoint."""
        interface = PredictionInterface()
        interface.model = mock_ensemble_model
        interface.is_loaded = True
        
        with patch.object(interface, 'predict', return_value=[
            RockfallPrediction(risk_level=RiskLevel.MEDIUM, confidence_score=0.8)
        ]) as mock_predict:
            prediction = interface.predict_single(sample_rockfall_datapoint)
        
        assert isinstance(prediction, RockfallPrediction)
        mock_predict.assert_called_once()
    
    def test_predict_batch(self, mock_ensemble_model, sample_feature_matrix):
        """Test batch prediction."""
        interface = PredictionInterface()
        interface.model = mock_ensemble_model
        interface.is_loaded = True
        interface.feature_names = [f'feature_{i}' for i in range(15)]
        
        # Mock the predict method to return different predictions for each call
        def mock_predict_side_effect(X):
            return [
                RockfallPrediction(risk_level=RiskLevel.LOW, confidence_score=0.9)
                for _ in range(len(X))
            ]
        
        with patch.object(interface, 'predict', side_effect=mock_predict_side_effect) as mock_predict:
            predictions = interface.predict_batch(sample_feature_matrix, batch_size=2)
        
        assert len(predictions) == 5  # sample_feature_matrix has 5 rows
        assert all(isinstance(p, RockfallPrediction) for p in predictions)
        
        # Should be called 3 times (batches of 2, 2, 1)
        assert mock_predict.call_count == 3
    
    def test_predict_batch_with_progress_callback(self, mock_ensemble_model, sample_feature_matrix):
        """Test batch prediction with progress callback."""
        interface = PredictionInterface()
        interface.model = mock_ensemble_model
        interface.is_loaded = True
        interface.feature_names = [f'feature_{i}' for i in range(15)]
        
        progress_calls = []
        
        def progress_callback(progress, current, total):
            progress_calls.append((progress, current, total))
        
        with patch.object(interface, 'predict', return_value=[
            RockfallPrediction(risk_level=RiskLevel.LOW, confidence_score=0.9)
        ]):
            interface.predict_batch(sample_feature_matrix, batch_size=2, 
                                  progress_callback=progress_callback)
        
        assert len(progress_calls) == 3  # 3 batches
        assert progress_calls[-1][1] == 5  # Final current should be 5 (total samples)
    
    def test_get_prediction_summary_empty(self):
        """Test prediction summary with empty list."""
        interface = PredictionInterface()
        summary = interface.get_prediction_summary([])
        
        assert "error" in summary
        assert summary["error"] == "No predictions provided"
    
    def test_get_prediction_summary_valid(self):
        """Test prediction summary with valid predictions."""
        predictions = [
            RockfallPrediction(risk_level=RiskLevel.LOW, confidence_score=0.9, uncertainty_estimate=0.1),
            RockfallPrediction(risk_level=RiskLevel.HIGH, confidence_score=0.8, uncertainty_estimate=0.2),
            RockfallPrediction(risk_level=RiskLevel.HIGH, confidence_score=0.7, uncertainty_estimate=0.3),
        ]
        
        interface = PredictionInterface()
        summary = interface.get_prediction_summary(predictions)
        
        assert summary["total_predictions"] == 3
        assert summary["risk_distribution"]["LOW"] == 1
        assert summary["risk_distribution"]["HIGH"] == 2
        assert summary["risk_distribution"]["MEDIUM"] == 0
        
        assert summary["confidence_stats"]["mean"] == pytest.approx(0.8, abs=1e-3)
        assert summary["uncertainty_stats"]["mean"] == pytest.approx(0.2, abs=1e-3)
    
    def test_get_model_info_not_loaded(self):
        """Test model info when no model is loaded."""
        interface = PredictionInterface()
        info = interface.get_model_info()
        
        assert "error" in info
        assert info["error"] == "No model loaded"
    
    def test_get_model_info_loaded(self, mock_ensemble_model):
        """Test model info when model is loaded."""
        interface = PredictionInterface()
        interface.model = mock_ensemble_model
        interface.model_path = "/path/to/model.pkl"
        interface.preprocessor_path = "/path/to/preprocessor.pkl"
        interface.feature_names = ['f1', 'f2', 'f3']
        interface.is_loaded = True
        
        # Add ensemble-specific attributes
        mock_ensemble_model.is_fitted = True
        mock_ensemble_model.config = MagicMock()
        mock_ensemble_model.config.ensemble_method = "voting"
        
        info = interface.get_model_info()
        
        assert info["model_path"] == "/path/to/model.pkl"
        assert info["preprocessor_path"] == "/path/to/preprocessor.pkl"
        assert info["model_type"] == "Mock"
        assert info["feature_count"] == 3
        assert info["feature_names"] == ['f1', 'f2', 'f3']
        assert info["model_details"]["is_ensemble"] == True
        assert info["model_details"]["ensemble_method"] == "voting"
    
    def test_prediction_error_handling(self, mock_ensemble_model):
        """Test error handling during prediction."""
        interface = PredictionInterface()
        interface.model = mock_ensemble_model
        interface.is_loaded = True
        
        # Mock predict_with_confidence to raise an exception
        mock_ensemble_model.predict_with_confidence.side_effect = Exception("Prediction failed")
        
        with pytest.raises(PredictionError, match="Prediction failed"):
            interface.predict(np.array([[1, 2, 3]]))
    
    def test_unsupported_input_type(self, mock_ensemble_model):
        """Test handling of unsupported input types."""
        interface = PredictionInterface()
        interface.model = mock_ensemble_model
        interface.is_loaded = True
        
        with pytest.raises(PredictionError, match="Unsupported input type"):
            interface._validate_inputs("invalid_input")


if __name__ == "__main__":
    pytest.main([__file__])