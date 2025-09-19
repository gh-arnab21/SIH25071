"""Unit tests for LSTM Temporal Feature Extractor."""

import pytest
import numpy as np
import torch
from datetime import datetime
from unittest.mock import patch, MagicMock

from src.models.extractors.lstm_temporal_extractor import LSTMTemporalExtractor, TemporalFeatureNetwork
from src.data.schemas import SensorData, TimeSeries


class TestLSTMTemporalExtractor:
    """Test cases for LSTMTemporalExtractor class."""
    
    @pytest.fixture
    def sample_config(self):
        """Sample configuration for testing."""
        return {
            'sequence_length': 20,
            'hidden_size': 64,
            'num_layers': 2,
            'feature_dim': 32,
            'dropout': 0.1,
            'bidirectional': True,
            'network_type': 'lstm',
            'device': 'cpu',
            'scaler_type': 'standard'
        }
    
    @pytest.fixture
    def sample_sensor_data(self):
        """Create sample sensor data for testing."""
        n_points = 100
        timestamps = np.arange(n_points)
        
        # Create synthetic time series with some patterns
        displacement_values = np.sin(np.linspace(0, 4*np.pi, n_points)) + np.random.normal(0, 0.1, n_points)
        strain_values = np.cos(np.linspace(0, 4*np.pi, n_points)) + np.random.normal(0, 0.1, n_points)
        pressure_values = np.linspace(0, 1, n_points) + np.random.normal(0, 0.05, n_points)
        
        displacement = TimeSeries(
            timestamps=timestamps,
            values=displacement_values,
            unit='mm',
            sampling_rate=1.0
        )
        
        strain = TimeSeries(
            timestamps=timestamps,
            values=strain_values,
            unit='microstrain',
            sampling_rate=1.0
        )
        
        pore_pressure = TimeSeries(
            timestamps=timestamps,
            values=pressure_values,
            unit='kPa',
            sampling_rate=1.0
        )
        
        return SensorData(
            displacement=displacement,
            strain=strain,
            pore_pressure=pore_pressure
        )
    
    @pytest.fixture
    def extractor(self, sample_config):
        """Create extractor instance for testing."""
        return LSTMTemporalExtractor(sample_config)
    
    def test_initialization_default_config(self):
        """Test extractor initialization with default configuration."""
        extractor = LSTMTemporalExtractor()
        
        assert extractor.config['sequence_length'] == 50
        assert extractor.config['hidden_size'] == 128
        assert extractor.config['num_layers'] == 2
        assert extractor.config['feature_dim'] == 64
        assert extractor.config['network_type'] == 'lstm'
        assert extractor.feature_dim == 64
        assert extractor.model is not None
        assert extractor.scaler is not None
    
    def test_initialization_custom_config(self, sample_config):
        """Test extractor initialization with custom configuration."""
        extractor = LSTMTemporalExtractor(sample_config)
        
        assert extractor.config['sequence_length'] == 20
        assert extractor.config['hidden_size'] == 64
        assert extractor.config['feature_dim'] == 32
        assert extractor.feature_dim == 32
    
    def test_prepare_sensor_data(self, extractor, sample_sensor_data):
        """Test sensor data preparation."""
        processed_data = extractor._prepare_sensor_data(sample_sensor_data)
        
        assert isinstance(processed_data, np.ndarray)
        assert processed_data.shape[1] == 3  # displacement, strain, pore_pressure
        assert processed_data.shape[0] == 100  # number of time points
        assert not np.isnan(processed_data).any()
    
    def test_prepare_sensor_data_missing_modalities(self, extractor):
        """Test sensor data preparation with missing modalities."""
        # Test with only displacement
        displacement = TimeSeries(
            timestamps=np.arange(50),
            values=np.random.randn(50),
            unit='mm'
        )
        sensor_data = SensorData(displacement=displacement)
        
        processed_data = extractor._prepare_sensor_data(sensor_data)
        assert processed_data.shape == (50, 3)
        assert not np.isnan(processed_data[:, 0]).any()  # displacement should be filled
        assert np.all(processed_data[:, 1] == 0)  # strain should be zero
        assert np.all(processed_data[:, 2] == 0)  # pressure should be zero
    
    def test_prepare_sensor_data_no_data(self, extractor):
        """Test sensor data preparation with no data."""
        sensor_data = SensorData()
        
        with pytest.raises(ValueError, match="No sensor data available"):
            extractor._prepare_sensor_data(sensor_data)
    
    def test_create_sequences(self, extractor):
        """Test sequence creation from time series data."""
        data = np.random.randn(100, 3)
        sequences = extractor._create_sequences(data)
        
        expected_num_sequences = 100 - extractor.config['sequence_length'] + 1
        assert sequences.shape == (expected_num_sequences, extractor.config['sequence_length'], 3)
    
    def test_create_sequences_short_data(self, extractor):
        """Test sequence creation with data shorter than sequence length."""
        data = np.random.randn(10, 3)  # Shorter than sequence_length (20)
        sequences = extractor._create_sequences(data)
        
        assert sequences.shape == (1, extractor.config['sequence_length'], 3)
        # Check that padding was applied
        assert np.all(sequences[0, :10, :] == 0)  # First 10 should be padding
    
    def test_fit_scaler(self, extractor, sample_sensor_data):
        """Test scaler fitting."""
        sensor_data_list = [sample_sensor_data]
        
        extractor.fit_scaler(sensor_data_list)
        
        assert extractor.scaler is not None
        # Check that scaler has been fitted
        assert hasattr(extractor.scaler, 'mean_') or hasattr(extractor.scaler, 'data_min_')
    
    def test_fit_scaler_empty_list(self, extractor):
        """Test scaler fitting with empty list."""
        with pytest.raises(ValueError, match="No valid sensor data found"):
            extractor.fit_scaler([])
    
    def test_extract_features(self, extractor, sample_sensor_data):
        """Test feature extraction from sensor data."""
        # First fit the scaler
        extractor.fit_scaler([sample_sensor_data])
        
        features = extractor.extract_features(sample_sensor_data)
        
        assert isinstance(features, np.ndarray)
        assert features.shape == (extractor.config['feature_dim'],)
        assert not np.isnan(features).any()
        assert not np.isinf(features).any()
    
    def test_extract_features_numpy_input(self, extractor):
        """Test feature extraction with numpy array input."""
        # Create sample data and fit scaler
        data = np.random.randn(100, 3)
        sample_sensor_data = SensorData(
            displacement=TimeSeries(np.arange(100), data[:, 0], 'mm'),
            strain=TimeSeries(np.arange(100), data[:, 1], 'microstrain'),
            pore_pressure=TimeSeries(np.arange(100), data[:, 2], 'kPa')
        )
        extractor.fit_scaler([sample_sensor_data])
        
        features = extractor.extract_features(data)
        
        assert isinstance(features, np.ndarray)
        assert features.shape == (extractor.config['feature_dim'],)
    
    def test_extract_features_without_fitted_scaler(self, extractor, sample_sensor_data):
        """Test feature extraction without fitted scaler."""
        with pytest.raises(ValueError, match="Scaler not fitted"):
            extractor.extract_features(sample_sensor_data)
    
    def test_extract_batch_features(self, extractor, sample_sensor_data):
        """Test batch feature extraction."""
        sensor_data_list = [sample_sensor_data, sample_sensor_data]
        
        # Fit scaler first
        extractor.fit_scaler(sensor_data_list)
        
        batch_features = extractor.extract_batch_features(sensor_data_list)
        
        assert isinstance(batch_features, np.ndarray)
        assert batch_features.shape == (2, extractor.config['feature_dim'])
        assert not np.isnan(batch_features).any()
    
    def test_identify_precursor_patterns(self, extractor, sample_sensor_data):
        """Test precursor pattern identification."""
        # Fit scaler first
        extractor.fit_scaler([sample_sensor_data])
        
        patterns = extractor.identify_precursor_patterns(sample_sensor_data)
        
        assert isinstance(patterns, dict)
        assert 'pattern_scores' in patterns
        assert 'anomaly_indices' in patterns
        assert 'displacement_trend' in patterns
        assert 'strain_trend' in patterns
        assert 'pressure_trend' in patterns
        assert 'max_pattern_score' in patterns
        assert 'mean_pattern_score' in patterns
        
        # Check trend analysis structure
        for trend_key in ['displacement_trend', 'strain_trend', 'pressure_trend']:
            trend = patterns[trend_key]
            assert 'slope' in trend
            assert 'acceleration' in trend
            assert 'volatility' in trend
    
    def test_calculate_trend(self, extractor):
        """Test trend calculation."""
        # Test with linear increasing data
        data = np.linspace(0, 10, 100)
        trend = extractor._calculate_trend(data)
        
        assert trend['slope'] > 0  # Should be positive slope
        assert isinstance(trend['acceleration'], float)
        assert isinstance(trend['volatility'], float)
        
        # Test with constant data
        constant_data = np.ones(50)
        trend = extractor._calculate_trend(constant_data)
        
        assert abs(trend['slope']) < 1e-10  # Should be near zero
        assert trend['volatility'] == 0.0  # No variation
    
    def test_calculate_trend_short_data(self, extractor):
        """Test trend calculation with short data."""
        data = np.array([1.0])
        trend = extractor._calculate_trend(data)
        
        assert trend['slope'] == 0.0
        assert trend['acceleration'] == 0.0
        assert trend['volatility'] == 0.0
    
    def test_save_and_load_model(self, extractor, tmp_path):
        """Test model saving and loading."""
        model_path = tmp_path / "test_model.pth"
        
        # Fit scaler to have something to save
        sample_data = np.random.randn(100, 3)
        sensor_data = SensorData(
            displacement=TimeSeries(np.arange(100), sample_data[:, 0], 'mm'),
            strain=TimeSeries(np.arange(100), sample_data[:, 1], 'microstrain'),
            pore_pressure=TimeSeries(np.arange(100), sample_data[:, 2], 'kPa')
        )
        extractor.fit_scaler([sensor_data])
        
        # Save model
        extractor.save_model(str(model_path))
        assert model_path.exists()
        
        # Create new extractor and load model
        new_extractor = LSTMTemporalExtractor()
        new_extractor.load_model(str(model_path))
        
        # Check that configuration is preserved
        assert new_extractor.config == extractor.config
        assert new_extractor.feature_dim == extractor.feature_dim
        
        # Check that scaler is preserved
        test_features_1 = extractor.extract_features(sensor_data)
        test_features_2 = new_extractor.extract_features(sensor_data)
        np.testing.assert_array_almost_equal(test_features_1, test_features_2)
    
    def test_save_and_load_features(self, extractor, tmp_path):
        """Test feature saving and loading."""
        features = np.random.randn(10, extractor.config['feature_dim'])
        features_path = tmp_path / "test_features.joblib"
        
        # Save features
        extractor.save_features(features, str(features_path))
        assert features_path.exists()
        
        # Load features
        loaded_features = extractor.load_features(str(features_path))
        np.testing.assert_array_equal(features, loaded_features)
    
    def test_invalid_input_type(self, extractor):
        """Test with invalid input type."""
        with pytest.raises(ValueError, match="Unsupported input type"):
            extractor.extract_features("invalid_input")
    
    def test_gru_network_type(self):
        """Test with GRU network type."""
        config = {'network_type': 'gru', 'device': 'cpu'}
        extractor = LSTMTemporalExtractor(config)
        
        assert extractor.config['network_type'] == 'gru'
        assert isinstance(extractor.model.rnn, torch.nn.GRU)
    
    def test_invalid_network_type(self):
        """Test with invalid network type."""
        config = {'network_type': 'invalid', 'device': 'cpu'}
        
        with pytest.raises(ValueError, match="Unsupported network type"):
            LSTMTemporalExtractor(config)
    
    def test_invalid_scaler_type(self):
        """Test with invalid scaler type."""
        config = {'scaler_type': 'invalid', 'device': 'cpu'}
        
        with pytest.raises(ValueError, match="Unsupported scaler type"):
            LSTMTemporalExtractor(config)


class TestTemporalFeatureNetwork:
    """Test cases for TemporalFeatureNetwork class."""
    
    @pytest.fixture
    def network_params(self):
        """Network parameters for testing."""
        return {
            'input_size': 3,
            'hidden_size': 32,
            'num_layers': 2,
            'feature_dim': 16,
            'dropout': 0.1,
            'bidirectional': True,
            'network_type': 'lstm'
        }
    
    def test_lstm_network_initialization(self, network_params):
        """Test LSTM network initialization."""
        network = TemporalFeatureNetwork(**network_params)
        
        assert isinstance(network.rnn, torch.nn.LSTM)
        assert network.rnn.input_size == 3
        assert network.rnn.hidden_size == 32
        assert network.rnn.num_layers == 2
        assert network.rnn.bidirectional == True
    
    def test_gru_network_initialization(self, network_params):
        """Test GRU network initialization."""
        network_params['network_type'] = 'gru'
        network = TemporalFeatureNetwork(**network_params)
        
        assert isinstance(network.rnn, torch.nn.GRU)
    
    def test_invalid_network_type_initialization(self, network_params):
        """Test network initialization with invalid type."""
        network_params['network_type'] = 'invalid'
        
        with pytest.raises(ValueError, match="Unsupported network type"):
            TemporalFeatureNetwork(**network_params)
    
    def test_forward_pass(self, network_params):
        """Test forward pass through the network."""
        network = TemporalFeatureNetwork(**network_params)
        
        # Create sample input
        batch_size, seq_length, input_size = 2, 10, 3
        x = torch.randn(batch_size, seq_length, input_size)
        
        # Forward pass
        output = network(x)
        
        assert output.shape == (batch_size, network_params['feature_dim'])
        assert not torch.isnan(output).any()
        assert not torch.isinf(output).any()
    
    def test_forward_pass_single_layer(self, network_params):
        """Test forward pass with single layer (no dropout in RNN)."""
        network_params['num_layers'] = 1
        network = TemporalFeatureNetwork(**network_params)
        
        batch_size, seq_length, input_size = 1, 5, 3
        x = torch.randn(batch_size, seq_length, input_size)
        
        output = network(x)
        assert output.shape == (batch_size, network_params['feature_dim'])
    
    def test_unidirectional_network(self, network_params):
        """Test unidirectional network."""
        network_params['bidirectional'] = False
        network = TemporalFeatureNetwork(**network_params)
        
        batch_size, seq_length, input_size = 2, 8, 3
        x = torch.randn(batch_size, seq_length, input_size)
        
        output = network(x)
        assert output.shape == (batch_size, network_params['feature_dim'])
    
    def test_attention_mechanism(self, network_params):
        """Test that attention mechanism works."""
        network = TemporalFeatureNetwork(**network_params)
        
        # Create input with different patterns
        batch_size, seq_length, input_size = 1, 10, 3
        x = torch.randn(batch_size, seq_length, input_size)
        
        # Get RNN output and attention weights
        rnn_output, _ = network.rnn(x)
        attention_weights = network.attention(rnn_output)
        
        # Check attention weights sum to 1
        assert torch.allclose(attention_weights.sum(dim=1), torch.ones(batch_size))
        assert attention_weights.shape == (batch_size, seq_length, 1)


if __name__ == "__main__":
    pytest.main([__file__])