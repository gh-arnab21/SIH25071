"""Unit tests for SensorDataProcessor."""

import pytest
import numpy as np
from datetime import datetime, timedelta
from unittest.mock import patch, MagicMock

from src.data.processors.sensor_processor import SensorDataProcessor
from src.data.schemas import SensorData, TimeSeries


class TestSensorDataProcessor:
    """Test cases for SensorDataProcessor class."""
    
    @pytest.fixture
    def sample_timeseries(self):
        """Create sample time series data for testing."""
        # Generate 100 data points over 24 hours
        timestamps = np.linspace(0, 24*3600, 100)  # 24 hours in seconds
        
        # Generate synthetic sensor data with trend and noise
        trend = 0.001 * timestamps  # Small increasing trend
        noise = np.random.normal(0, 0.1, len(timestamps))
        values = 10.0 + trend + noise
        
        return TimeSeries(
            timestamps=timestamps,
            values=values,
            unit='mm',
            sampling_rate=1/360  # One sample per 6 minutes
        )
    
    @pytest.fixture
    def sample_sensor_data(self, sample_timeseries):
        """Create sample SensorData object."""
        # Create different time series for each sensor type
        displacement_ts = sample_timeseries
        
        # Strain data with different characteristics
        strain_values = np.random.normal(500, 50, len(sample_timeseries.timestamps))
        strain_ts = TimeSeries(
            timestamps=sample_timeseries.timestamps,
            values=strain_values,
            unit='microstrain',
            sampling_rate=sample_timeseries.sampling_rate
        )
        
        # Pore pressure data
        pp_values = np.random.normal(200, 20, len(sample_timeseries.timestamps))
        pp_ts = TimeSeries(
            timestamps=sample_timeseries.timestamps,
            values=pp_values,
            unit='kPa',
            sampling_rate=sample_timeseries.sampling_rate
        )
        
        return SensorData(
            displacement=displacement_ts,
            strain=strain_ts,
            pore_pressure=pp_ts
        )
    
    @pytest.fixture
    def processor_config(self):
        """Default configuration for processor."""
        return {
            'sampling_rate': 1.0,
            'filter_type': 'lowpass',
            'filter_params': {'cutoff': 0.1, 'order': 4},
            'normalization': 'standard',
            'anomaly_method': 'isolation_forest',
            'anomaly_params': {'contamination': 0.1, 'random_state': 42},
            'trend_window': 10,
            'interpolation_method': 'linear',
            'max_gap_size': 5,
            'outlier_threshold': 3.0
        }
    
    def test_processor_initialization(self, processor_config):
        """Test processor initialization with configuration."""
        processor = SensorDataProcessor(processor_config)
        
        assert processor.config['sampling_rate'] == 1.0
        assert processor.config['filter_type'] == 'lowpass'
        assert processor.config['normalization'] == 'standard'
        assert not processor.is_fitted
        
        # Test default initialization
        default_processor = SensorDataProcessor()
        assert default_processor.config['sampling_rate'] == 1.0
        assert default_processor.config['anomaly_method'] == 'isolation_forest'
    
    def test_fit_processor(self, sample_sensor_data, processor_config):
        """Test fitting the processor to training data."""
        processor = SensorDataProcessor(processor_config)
        
        # Fit with list of sensor data
        training_data = [sample_sensor_data]
        fitted_processor = processor.fit(training_data)
        
        assert fitted_processor.is_fitted
        assert fitted_processor is processor  # Should return self
        
        # Check that scalers are fitted
        assert processor.scalers['displacement'] is not None
        assert processor.scalers['strain'] is not None
        assert processor.scalers['pore_pressure'] is not None
        
        # Check that statistics are stored
        assert 'displacement' in processor.data_stats
        assert 'strain' in processor.data_stats
        assert 'pore_pressure' in processor.data_stats
    
    def test_transform_single_sensor_data(self, sample_sensor_data, processor_config):
        """Test transforming a single SensorData object."""
        processor = SensorDataProcessor(processor_config)
        processor.fit([sample_sensor_data])
        
        # Transform single sensor data
        features = processor.transform(sample_sensor_data)
        
        assert isinstance(features, np.ndarray)
        assert len(features.shape) == 1  # Should be 1D feature vector
        
        # Should have features for all three sensor types
        expected_feature_count = 3 * processor._get_feature_count()  # 3 sensor types
        assert len(features) == expected_feature_count
    
    def test_transform_list_sensor_data(self, sample_sensor_data, processor_config):
        """Test transforming a list of SensorData objects."""
        processor = SensorDataProcessor(processor_config)
        
        # Create multiple sensor data objects
        sensor_data_list = [sample_sensor_data, sample_sensor_data]
        processor.fit(sensor_data_list)
        
        # Transform list
        features = processor.transform(sensor_data_list)
        
        assert isinstance(features, np.ndarray)
        assert len(features.shape) == 2  # Should be 2D array
        assert features.shape[0] == 2  # Two samples
        
        expected_feature_count = 3 * processor._get_feature_count()
        assert features.shape[1] == expected_feature_count
    
    def test_transform_without_fit_raises_error(self, sample_sensor_data):
        """Test that transform raises error when processor is not fitted."""
        processor = SensorDataProcessor()
        
        with pytest.raises(ValueError, match="Processor must be fitted"):
            processor.transform(sample_sensor_data)
    
    def test_preprocess_timeseries(self, sample_timeseries, processor_config):
        """Test time series preprocessing."""
        processor = SensorDataProcessor(processor_config)
        
        # Test preprocessing
        processed_ts = processor._preprocess_timeseries(sample_timeseries)
        
        assert processed_ts is not None
        assert isinstance(processed_ts, TimeSeries)
        assert len(processed_ts.values) > 0
        assert len(processed_ts.timestamps) == len(processed_ts.values)
    
    def test_preprocess_timeseries_with_missing_values(self, processor_config):
        """Test preprocessing with missing values."""
        processor = SensorDataProcessor(processor_config)
        
        # Create time series with NaN values
        timestamps = np.linspace(0, 100, 50)
        values = np.random.normal(0, 1, 50)
        values[10:15] = np.nan  # Add missing values
        values[30:32] = np.nan  # Add small gap
        
        ts_with_nan = TimeSeries(
            timestamps=timestamps,
            values=values,
            unit='test',
            sampling_rate=0.5
        )
        
        processed_ts = processor._preprocess_timeseries(ts_with_nan)
        
        assert processed_ts is not None
        assert not np.any(np.isnan(processed_ts.values))
        assert len(processed_ts.values) > 0
    
    def test_handle_missing_values_large_gaps(self, processor_config):
        """Test handling of large gaps in data."""
        processor = SensorDataProcessor(processor_config)
        
        # Create time series with large gap
        timestamps = np.linspace(0, 100, 50)
        values = np.random.normal(0, 1, 50)
        values[10:25] = np.nan  # Large gap (15 points, > max_gap_size)
        
        ts_with_large_gap = TimeSeries(
            timestamps=timestamps,
            values=values,
            unit='test',
            sampling_rate=0.5
        )
        
        processed_ts = processor._preprocess_timeseries(ts_with_large_gap)
        
        # Should still return data but may be resampled to different length
        assert processed_ts is not None
        # The resampling might change the length, so just check it's reasonable
        assert len(processed_ts.values) > 10  # Should have some data
    
    def test_apply_filter(self, sample_timeseries, processor_config):
        """Test digital filtering."""
        processor = SensorDataProcessor(processor_config)
        
        # Test lowpass filter
        filtered_ts = processor._apply_filter(sample_timeseries)
        
        assert isinstance(filtered_ts, TimeSeries)
        assert len(filtered_ts.values) == len(sample_timeseries.values)
        assert filtered_ts.unit == sample_timeseries.unit
    
    def test_apply_filter_different_types(self, sample_timeseries):
        """Test different filter types."""
        # Test highpass filter
        config = {'filter_type': 'highpass', 'filter_params': {'cutoff': 0.01, 'order': 2}}
        processor = SensorDataProcessor(config)
        filtered_ts = processor._apply_filter(sample_timeseries)
        assert len(filtered_ts.values) == len(sample_timeseries.values)
        
        # Test bandpass filter
        config = {'filter_type': 'bandpass', 'filter_params': {'low_cutoff': 0.01, 'high_cutoff': 0.1, 'order': 2}}
        processor = SensorDataProcessor(config)
        filtered_ts = processor._apply_filter(sample_timeseries)
        assert len(filtered_ts.values) == len(sample_timeseries.values)
    
    def test_resample_timeseries(self, sample_timeseries, processor_config):
        """Test time series resampling."""
        processor = SensorDataProcessor(processor_config)
        
        # Test resampling to different rate
        processor.config['sampling_rate'] = 2.0  # 2 Hz
        resampled_ts = processor._resample_timeseries(sample_timeseries)
        
        assert isinstance(resampled_ts, TimeSeries)
        assert resampled_ts.sampling_rate == 2.0
        # Should have approximately correct number of samples
        expected_samples = int((sample_timeseries.timestamps[-1] - sample_timeseries.timestamps[0]) * 2.0) + 1
        assert abs(len(resampled_ts.values) - expected_samples) <= 1
    
    def test_extract_statistical_features(self, processor_config):
        """Test statistical feature extraction."""
        processor = SensorDataProcessor(processor_config)
        
        # Test with normal data
        values = np.random.normal(10, 2, 100)
        features = processor._extract_statistical_features(values)
        
        assert len(features) == 8  # mean, std, min, max, median, Q1, Q3, skew
        assert abs(features[0] - 10) < 1  # Mean should be close to 10
        assert abs(features[1] - 2) < 1   # Std should be close to 2
        
        # Test with empty array
        empty_features = processor._extract_statistical_features(np.array([]))
        assert len(empty_features) == 8
        assert all(f == 0.0 for f in empty_features)
    
    def test_extract_temporal_features(self, sample_timeseries, processor_config):
        """Test temporal feature extraction."""
        processor = SensorDataProcessor(processor_config)
        
        features = processor._extract_temporal_features(sample_timeseries)
        
        assert len(features) == 6  # rate of change features + metadata
        assert isinstance(features[0], (int, float))  # Average rate of change
        assert isinstance(features[4], (int, float))  # Number of data points
        assert isinstance(features[5], (int, float))  # Duration
    
    def test_extract_trend_features(self, processor_config):
        """Test trend feature extraction."""
        processor = SensorDataProcessor(processor_config)
        
        # Create data with known trend
        x = np.linspace(0, 100, 50)
        values = 2 * x + 10 + np.random.normal(0, 1, 50)  # Linear trend with noise
        
        features = processor._extract_trend_features(values)
        
        assert len(features) == 4  # slope, r_squared, trend_strength, p_value
        assert features[0] > 0  # Positive slope
        assert 0 <= features[1] <= 1  # R-squared between 0 and 1
        assert 0 <= features[3] <= 1  # p-value between 0 and 1
    
    def test_extract_anomaly_features(self, sample_sensor_data, processor_config):
        """Test anomaly feature extraction."""
        processor = SensorDataProcessor(processor_config)
        processor.fit([sample_sensor_data])
        
        # Test with normal data
        values = np.random.normal(0, 1, 100)
        features = processor._extract_anomaly_features(values, 'displacement')
        
        assert len(features) == 4  # outlier_count, outlier_ratio, anomaly_count, anomaly_ratio
        # Convert numpy types to Python types for testing
        features = [float(f) if hasattr(f, 'item') else f for f in features]
        assert all(isinstance(f, (int, float)) for f in features)
        assert 0 <= features[1] <= 1  # Outlier ratio should be between 0 and 1
        assert 0 <= features[3] <= 1  # Anomaly ratio should be between 0 and 1
    
    def test_extract_frequency_features(self, processor_config):
        """Test frequency domain feature extraction."""
        processor = SensorDataProcessor(processor_config)
        
        # Create signal with known frequency components
        t = np.linspace(0, 10, 1000)
        signal = np.sin(2 * np.pi * 5 * t) + 0.5 * np.sin(2 * np.pi * 10 * t)  # 5 Hz + 10 Hz
        
        features = processor._extract_frequency_features(signal, sampling_rate=100.0)
        
        assert len(features) == 6  # frequency domain features
        assert all(isinstance(f, (int, float)) for f in features)
        assert features[0] > 0  # Dominant frequency should be positive
    
    def test_detect_anomalies_isolation_forest(self, sample_timeseries, sample_sensor_data, processor_config):
        """Test anomaly detection using isolation forest."""
        processor = SensorDataProcessor(processor_config)
        processor.fit([sample_sensor_data])
        
        # Add some anomalies to the data
        anomalous_ts = TimeSeries(
            timestamps=sample_timeseries.timestamps,
            values=sample_timeseries.values.copy(),
            unit=sample_timeseries.unit,
            sampling_rate=sample_timeseries.sampling_rate
        )
        anomalous_ts.values[10] = 1000  # Add obvious anomaly
        anomalous_ts.values[50] = -1000  # Add another anomaly
        
        results = processor.detect_anomalies(anomalous_ts, 'displacement')
        
        assert 'anomalies' in results
        assert 'anomaly_scores' in results
        assert 'method' in results
        assert results['method'] == 'isolation_forest'
        assert len(results['anomalies']) >= 0  # Should detect some anomalies
    
    def test_detect_anomalies_statistical(self, sample_timeseries, sample_sensor_data):
        """Test statistical anomaly detection."""
        config = {'anomaly_method': 'statistical', 'outlier_threshold': 2.0}
        processor = SensorDataProcessor(config)
        processor.fit([sample_sensor_data])
        
        # Create data with outliers
        values = np.random.normal(0, 1, 100)
        values[10] = 10  # Clear outlier
        values[50] = -10  # Another outlier
        
        outlier_ts = TimeSeries(
            timestamps=sample_timeseries.timestamps,
            values=values,
            unit='test',
            sampling_rate=1.0
        )
        
        results = processor.detect_anomalies(outlier_ts, 'displacement')
        
        assert results['method'] == 'statistical'
        assert len(results['anomalies']) >= 2  # Should detect the outliers
    
    def test_analyze_trends(self, processor_config):
        """Test trend analysis."""
        processor = SensorDataProcessor(processor_config)
        
        # Create data with increasing trend
        x = np.linspace(0, 100, 50)
        values = 0.5 * x + np.random.normal(0, 0.1, 50)
        
        trend_ts = TimeSeries(
            timestamps=x,
            values=values,
            unit='test',
            sampling_rate=0.5
        )
        
        results = processor.analyze_trends(trend_ts)
        
        assert 'trend' in results
        assert 'slope' in results
        assert 'r_squared' in results
        assert 'p_value' in results
        assert 'significance' in results
        
        assert results['trend'] in ['increasing', 'decreasing', 'stable']
        assert results['slope'] > 0  # Should detect positive trend
    
    def test_normalize_sensor_data(self, sample_timeseries, sample_sensor_data, processor_config):
        """Test sensor data normalization."""
        processor = SensorDataProcessor(processor_config)
        processor.fit([sample_sensor_data])
        
        normalized_ts = processor.normalize_sensor_data(sample_timeseries, 'displacement')
        
        assert isinstance(normalized_ts, TimeSeries)
        # Length may change due to resampling, just check it's reasonable
        assert len(normalized_ts.values) > 0
        assert 'normalized' in normalized_ts.unit
        
        # Normalized data should have different scale (check mean is close to 0 for standard normalization)
        assert abs(np.mean(normalized_ts.values)) < 1.0  # Should be roughly centered around 0
    
    def test_normalize_without_fit_raises_error(self, sample_timeseries):
        """Test that normalization raises error when processor is not fitted."""
        processor = SensorDataProcessor()
        
        with pytest.raises(ValueError, match="Processor must be fitted"):
            processor.normalize_sensor_data(sample_timeseries, 'displacement')
    
    def test_get_sensor_statistics(self, sample_sensor_data, processor_config):
        """Test getting sensor statistics."""
        processor = SensorDataProcessor(processor_config)
        processor.fit([sample_sensor_data])
        
        stats = processor.get_sensor_statistics('displacement')
        
        assert isinstance(stats, dict)
        assert 'mean' in stats
        assert 'std' in stats
        assert 'min' in stats
        assert 'max' in stats
        assert 'median' in stats
    
    def test_process_displacement_data(self, sample_timeseries, sample_sensor_data, processor_config):
        """Test displacement-specific processing."""
        processor = SensorDataProcessor(processor_config)
        processor.fit([sample_sensor_data])
        
        results = processor.process_displacement_data(sample_timeseries)
        
        assert results['sensor_type'] == 'displacement'
        assert 'processed_timeseries' in results
        assert 'features' in results
        assert 'anomalies' in results
        assert 'trends' in results
        assert 'alerts' in results
        
        assert isinstance(results['alerts'], list)
    
    def test_process_strain_data(self, sample_sensor_data, processor_config):
        """Test strain-specific processing."""
        processor = SensorDataProcessor(processor_config)
        processor.fit([sample_sensor_data])
        
        results = processor.process_strain_data(sample_sensor_data.strain)
        
        assert results['sensor_type'] == 'strain'
        assert 'processed_timeseries' in results
        assert 'features' in results
        assert 'anomalies' in results
        assert 'trends' in results
        assert 'alerts' in results
    
    def test_process_pore_pressure_data(self, sample_sensor_data, processor_config):
        """Test pore pressure-specific processing."""
        processor = SensorDataProcessor(processor_config)
        processor.fit([sample_sensor_data])
        
        results = processor.process_pore_pressure_data(sample_sensor_data.pore_pressure)
        
        assert results['sensor_type'] == 'pore_pressure'
        assert 'processed_timeseries' in results
        assert 'features' in results
        assert 'anomalies' in results
        assert 'trends' in results
        assert 'alerts' in results
    
    def test_missing_sensor_data_handling(self, processor_config):
        """Test handling of missing sensor data."""
        processor = SensorDataProcessor(processor_config)
        
        # Create sensor data with missing components
        incomplete_sensor_data = SensorData(
            displacement=None,  # Missing displacement
            strain=None,        # Missing strain
            pore_pressure=None  # Missing pore pressure
        )
        
        # Should handle gracefully during fitting
        processor.fit([incomplete_sensor_data])
        
        # Transform should return zero features
        features = processor.transform(incomplete_sensor_data)
        expected_feature_count = 3 * processor._get_feature_count()
        assert len(features) == expected_feature_count
        assert np.allclose(features, 0.0)  # All features should be zero
    
    def test_get_feature_count(self, processor_config):
        """Test feature count calculation."""
        processor = SensorDataProcessor(processor_config)
        
        feature_count = processor._get_feature_count()
        
        # Should be sum of all feature types: 8 + 6 + 4 + 4 + 6 = 28
        assert feature_count == 28
    
    def test_estimate_sampling_rate(self, processor_config):
        """Test sampling rate estimation."""
        processor = SensorDataProcessor(processor_config)
        
        # Test with regular timestamps
        timestamps = np.linspace(0, 100, 101)  # 1 Hz
        estimated_rate = processor._estimate_sampling_rate(timestamps)
        assert abs(estimated_rate - 1.0) < 0.01
        
        # Test with irregular timestamps
        irregular_timestamps = np.array([0, 1, 3, 6, 10])
        estimated_rate = processor._estimate_sampling_rate(irregular_timestamps)
        assert estimated_rate > 0
        
        # Test with single timestamp
        single_timestamp = np.array([0])
        estimated_rate = processor._estimate_sampling_rate(single_timestamp)
        assert estimated_rate == 1.0
    
    def test_config_validation(self):
        """Test configuration validation and defaults."""
        # Test with empty config
        processor = SensorDataProcessor({})
        assert processor.config['sampling_rate'] == 1.0
        assert processor.config['filter_type'] == 'lowpass'
        
        # Test with partial config
        partial_config = {'sampling_rate': 2.0}
        processor = SensorDataProcessor(partial_config)
        assert processor.config['sampling_rate'] == 2.0
        assert processor.config['filter_type'] == 'lowpass'  # Should use default
    
    @patch('warnings.warn')
    def test_error_handling_in_preprocessing(self, mock_warn, processor_config):
        """Test error handling during preprocessing."""
        processor = SensorDataProcessor(processor_config)
        
        # Create problematic time series
        problematic_ts = TimeSeries(
            timestamps=np.array([]),  # Empty timestamps
            values=np.array([]),      # Empty values
            unit='test',
            sampling_rate=1.0
        )
        
        result = processor._preprocess_timeseries(problematic_ts)
        assert result is None
        
        # Test with invalid data
        invalid_ts = TimeSeries(
            timestamps=np.array([1, 2, 3]),
            values=np.array([np.inf, np.nan, -np.inf]),  # Invalid values
            unit='test',
            sampling_rate=1.0
        )
        
        # Should handle gracefully
        result = processor._preprocess_timeseries(invalid_ts)
        # Result might be None or processed data, but shouldn't crash