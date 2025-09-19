"""Unit tests for EnvironmentalProcessor."""

import pytest
import numpy as np
from unittest.mock import patch, MagicMock

from src.data.processors.environmental_processor import EnvironmentalProcessor
from src.data.schemas import EnvironmentalData


class TestEnvironmentalProcessor:
    """Test cases for EnvironmentalProcessor class."""
    
    @pytest.fixture
    def sample_environmental_data(self):
        """Create sample environmental data for testing."""
        return EnvironmentalData(
            rainfall=15.5,
            temperature=22.3,
            vibrations=2.1,
            wind_speed=8.7
        )
    
    @pytest.fixture
    def sample_environmental_data_list(self):
        """Create a list of sample environmental data for training."""
        data_list = []
        
        # Generate diverse environmental conditions
        for i in range(50):
            data_list.append(EnvironmentalData(
                rainfall=np.random.exponential(5.0),  # Exponential distribution for rainfall
                temperature=np.random.normal(20.0, 10.0),  # Normal distribution for temperature
                vibrations=np.random.exponential(1.0),  # Exponential for vibrations
                wind_speed=np.random.exponential(8.0)  # Exponential for wind speed
            ))
        
        return data_list
    
    @pytest.fixture
    def processor(self):
        """Create a basic EnvironmentalProcessor instance."""
        return EnvironmentalProcessor()
    
    @pytest.fixture
    def fitted_processor(self, sample_environmental_data_list):
        """Create a fitted EnvironmentalProcessor instance."""
        processor = EnvironmentalProcessor()
        processor.fit(sample_environmental_data_list)
        return processor
    
    def test_initialization_default_config(self):
        """Test processor initialization with default configuration."""
        processor = EnvironmentalProcessor()
        
        assert processor.config['normalization'] == 'standard'
        assert processor.config['rainfall_cumulative_window'] == 24
        assert processor.config['temperature_smoothing_window'] == 6
        assert processor.config['vibration_threshold'] == 5.0
        assert processor.config['rainfall_trigger_threshold'] == 50.0
        assert not processor.is_fitted
    
    def test_initialization_custom_config(self):
        """Test processor initialization with custom configuration."""
        custom_config = {
            'normalization': 'minmax',
            'rainfall_trigger_threshold': 75.0,
            'vibration_threshold': 10.0
        }
        
        processor = EnvironmentalProcessor(config=custom_config)
        
        assert processor.config['normalization'] == 'minmax'
        assert processor.config['rainfall_trigger_threshold'] == 75.0
        assert processor.config['vibration_threshold'] == 10.0
        # Default values should still be present
        assert processor.config['temperature_smoothing_window'] == 6
    
    def test_fit_with_valid_data(self, sample_environmental_data_list):
        """Test fitting the processor with valid environmental data."""
        processor = EnvironmentalProcessor()
        
        # Fit the processor
        fitted_processor = processor.fit(sample_environmental_data_list)
        
        # Check that fitting returns self
        assert fitted_processor is processor
        assert processor.is_fitted
        
        # Check that scalers are fitted
        assert processor.scalers['rainfall'] is not None
        assert processor.scalers['temperature'] is not None
        assert processor.scalers['vibrations'] is not None
        assert processor.scalers['wind_speed'] is not None
        
        # Check that statistics are calculated
        assert 'rainfall' in processor.data_stats
        assert 'temperature' in processor.data_stats
        assert 'vibrations' in processor.data_stats
        assert 'wind_speed' in processor.data_stats
        
        # Check that trigger thresholds are set
        assert 'rainfall' in processor.trigger_thresholds
        assert 'vibrations' in processor.trigger_thresholds
    
    def test_fit_with_missing_data(self):
        """Test fitting with data containing missing values."""
        data_with_missing = [
            EnvironmentalData(rainfall=10.0, temperature=None, vibrations=2.0, wind_speed=5.0),
            EnvironmentalData(rainfall=None, temperature=15.0, vibrations=None, wind_speed=7.0),
            EnvironmentalData(rainfall=5.0, temperature=20.0, vibrations=1.5, wind_speed=None)
        ]
        
        processor = EnvironmentalProcessor()
        processor.fit(data_with_missing)
        
        assert processor.is_fitted
        # Should handle missing data gracefully
        assert processor.scalers['rainfall'] is not None
        assert processor.scalers['temperature'] is not None
    
    def test_transform_single_data_point(self, fitted_processor, sample_environmental_data):
        """Test transforming a single environmental data point."""
        features = fitted_processor.transform(sample_environmental_data)
        
        # Check feature vector shape
        expected_feature_count = 6 + 8 + 6 + 6 + 4  # rainfall + temperature + vibrations + wind + combined
        assert isinstance(features, np.ndarray)
        assert len(features) == expected_feature_count
        
        # Check that features are numeric
        assert np.all(np.isfinite(features))
    
    def test_transform_multiple_data_points(self, fitted_processor, sample_environmental_data_list):
        """Test transforming multiple environmental data points."""
        # Take first 5 data points for testing
        test_data = sample_environmental_data_list[:5]
        features = fitted_processor.transform(test_data)
        
        # Check shape
        assert features.shape[0] == 5
        assert features.shape[1] == 30  # Expected feature count
        
        # Check that all features are numeric
        assert np.all(np.isfinite(features))
    
    def test_transform_without_fitting(self, processor, sample_environmental_data):
        """Test that transform raises error when processor is not fitted."""
        with pytest.raises(ValueError, match="Processor must be fitted before transform"):
            processor.transform(sample_environmental_data)
    
    def test_extract_rainfall_features(self, fitted_processor):
        """Test rainfall feature extraction."""
        # Test with normal rainfall
        features = fitted_processor._extract_rainfall_features(15.0)
        assert len(features) == 6
        assert features[0] == 15.0  # Raw rainfall
        
        # Test with zero rainfall
        features_zero = fitted_processor._extract_rainfall_features(0.0)
        assert features_zero[2] == 0.0  # Light rain indicator should be 0
        
        # Test with heavy rainfall
        features_heavy = fitted_processor._extract_rainfall_features(25.0)
        assert features_heavy[4] == 1.0  # Heavy rain indicator should be 1
        
        # Test with None rainfall
        features_none = fitted_processor._extract_rainfall_features(None)
        assert all(f == 0.0 for f in features_none)
    
    def test_extract_temperature_features(self, fitted_processor):
        """Test temperature feature extraction."""
        # Test with normal temperature
        features = fitted_processor._extract_temperature_features(22.0)
        assert len(features) == 8
        assert features[0] == 22.0  # Raw temperature
        
        # Test with freezing temperature
        features_freeze = fitted_processor._extract_temperature_features(-2.0)
        assert features_freeze[2] == 1.0  # Freezing indicator should be 1
        
        # Test with extreme hot temperature
        features_hot = fitted_processor._extract_temperature_features(45.0)
        assert features_hot[3] == 1.0  # Extreme hot indicator should be 1
        
        # Test with None temperature
        features_none = fitted_processor._extract_temperature_features(None)
        assert all(f == 0.0 for f in features_none)
    
    def test_extract_vibration_features(self, fitted_processor):
        """Test vibration feature extraction."""
        # Test with normal vibrations
        features = fitted_processor._extract_vibration_features(3.0)
        assert len(features) == 6
        assert features[0] == 3.0  # Raw vibrations
        
        # Test with high vibrations
        features_high = fitted_processor._extract_vibration_features(10.0)
        assert features_high[3] == 1.0  # Significant vibrations indicator
        
        # Test with negative vibrations (should use absolute value)
        features_neg = fitted_processor._extract_vibration_features(-5.0)
        assert features_neg[0] == 5.0  # Should be converted to positive
        
        # Test with None vibrations
        features_none = fitted_processor._extract_vibration_features(None)
        assert all(f == 0.0 for f in features_none)
    
    def test_extract_wind_features(self, fitted_processor):
        """Test wind speed feature extraction."""
        # Test with normal wind speed
        features = fitted_processor._extract_wind_features(12.0)
        assert len(features) == 6
        assert features[0] == 12.0  # Raw wind speed
        
        # Test with high wind speed
        features_high = fitted_processor._extract_wind_features(25.0)
        assert features_high[4] == 1.0  # Near gale indicator should be 1
        
        # Test with None wind speed
        features_none = fitted_processor._extract_wind_features(None)
        assert all(f == 0.0 for f in features_none)
    
    def test_calculate_combined_risk_indicators(self, fitted_processor):
        """Test combined risk indicator calculations."""
        # Test freeze-thaw risk scenario
        env_data_freeze_thaw = EnvironmentalData(
            rainfall=5.0, temperature=1.0, vibrations=2.0, wind_speed=8.0
        )
        features = fitted_processor._calculate_combined_risk_indicators(env_data_freeze_thaw)
        assert len(features) == 4
        assert features[0] == 1.0  # Freeze-thaw risk should be detected
        
        # Test rain + vibration risk scenario
        env_data_rain_vib = EnvironmentalData(
            rainfall=10.0, temperature=20.0, vibrations=8.0, wind_speed=5.0
        )
        features_rain_vib = fitted_processor._calculate_combined_risk_indicators(env_data_rain_vib)
        assert features_rain_vib[1] == 1.0  # Rain-vibration risk should be detected
    
    def test_validate_environmental_data(self, processor):
        """Test environmental data validation."""
        # Test valid data
        valid_data = EnvironmentalData(rainfall=10.0, temperature=20.0, vibrations=2.0, wind_speed=8.0)
        result = processor.validate_environmental_data(valid_data)
        assert result['is_valid'] is True
        assert len(result['errors']) == 0
        
        # Test invalid data (negative rainfall)
        invalid_data = EnvironmentalData(rainfall=-5.0, temperature=20.0, vibrations=2.0, wind_speed=8.0)
        result_invalid = processor.validate_environmental_data(invalid_data)
        assert result_invalid['is_valid'] is False
        assert len(result_invalid['errors']) > 0
        assert result_invalid['corrected_data'].rainfall == 0.0
        
        # Test extreme values (should generate warnings)
        extreme_data = EnvironmentalData(rainfall=600.0, temperature=70.0, vibrations=150.0, wind_speed=250.0)
        result_extreme = processor.validate_environmental_data(extreme_data)
        assert len(result_extreme['warnings']) > 0
    
    def test_calculate_cumulative_effects(self, fitted_processor):
        """Test cumulative effects calculation."""
        # Create time series of environmental data
        environmental_series = []
        for i in range(48):  # 48 hours of data
            timestamp = float(i)
            env_data = EnvironmentalData(
                rainfall=np.random.exponential(2.0),
                temperature=20.0 + 5.0 * np.sin(i * 0.1),  # Sinusoidal temperature
                vibrations=np.random.exponential(1.0),
                wind_speed=np.random.exponential(5.0)
            )
            environmental_series.append((timestamp, env_data))
        
        cumulative_effects = fitted_processor.calculate_cumulative_effects(environmental_series)
        
        # Check that cumulative effects are calculated
        assert 'cumulative_rainfall' in cumulative_effects
        assert 'temperature_effects' in cumulative_effects
        assert 'vibration_effects' in cumulative_effects
        assert 'overall_risk_assessment' in cumulative_effects
        
        # Check cumulative rainfall structure
        rainfall_effects = cumulative_effects['cumulative_rainfall']
        assert 'value' in rainfall_effects
        assert 'window_hours' in rainfall_effects
        assert 'trigger_exceeded' in rainfall_effects
        assert 'intensity' in rainfall_effects
        
        # Check temperature effects structure
        temp_effects = cumulative_effects['temperature_effects']
        assert 'freeze_thaw_cycles' in temp_effects
        assert 'trend' in temp_effects
        assert 'current_temperature' in temp_effects
    
    def test_detect_trigger_conditions(self, fitted_processor):
        """Test trigger condition detection."""
        # Test normal conditions (no triggers)
        normal_data = EnvironmentalData(rainfall=5.0, temperature=20.0, vibrations=1.0, wind_speed=8.0)
        triggers = fitted_processor.detect_trigger_conditions(normal_data)
        assert triggers['overall_trigger_status'] is False
        assert triggers['risk_level'] == 'minimal'
        
        # Test high rainfall trigger
        high_rain_data = EnvironmentalData(rainfall=60.0, temperature=20.0, vibrations=1.0, wind_speed=8.0)
        triggers_rain = fitted_processor.detect_trigger_conditions(high_rain_data)
        assert 'rainfall' in triggers_rain['triggered_conditions']
        assert triggers_rain['overall_trigger_status'] is True
        
        # Test extreme temperature trigger
        extreme_temp_data = EnvironmentalData(rainfall=5.0, temperature=45.0, vibrations=1.0, wind_speed=8.0)
        triggers_temp = fitted_processor.detect_trigger_conditions(extreme_temp_data)
        assert 'temperature_extreme_high' in triggers_temp['triggered_conditions']
        
        # Test multiple triggers (high risk)
        multi_trigger_data = EnvironmentalData(rainfall=70.0, temperature=45.0, vibrations=10.0, wind_speed=25.0)
        triggers_multi = fitted_processor.detect_trigger_conditions(multi_trigger_data)
        assert len(triggers_multi['triggered_conditions']) >= 3
        assert triggers_multi['risk_level'] == 'high'
    
    def test_get_feature_names(self, processor):
        """Test feature names retrieval."""
        feature_names = processor.get_feature_names()
        
        # Check total number of features
        assert len(feature_names) == 30  # 6+8+6+6+4
        
        # Check that all names are strings
        assert all(isinstance(name, str) for name in feature_names)
        
        # Check for expected feature categories
        rainfall_features = [name for name in feature_names if name.startswith('rainfall')]
        temperature_features = [name for name in feature_names if name.startswith('temperature')]
        vibration_features = [name for name in feature_names if name.startswith('vibrations')]
        wind_features = [name for name in feature_names if name.startswith('wind')]
        
        assert len(rainfall_features) == 6
        assert len(temperature_features) == 8
        assert len(vibration_features) == 6
        assert len(wind_features) == 6
    
    def test_get_parameter_statistics(self, fitted_processor):
        """Test parameter statistics retrieval."""
        # Test getting statistics for fitted parameters
        rainfall_stats = fitted_processor.get_parameter_statistics('rainfall')
        assert 'mean' in rainfall_stats
        assert 'std' in rainfall_stats
        assert 'min' in rainfall_stats
        assert 'max' in rainfall_stats
        
        # Test getting statistics for non-existent parameter
        empty_stats = fitted_processor.get_parameter_statistics('non_existent')
        assert empty_stats == {}
    
    def test_get_parameter_statistics_not_fitted(self, processor):
        """Test that getting statistics raises error when not fitted."""
        with pytest.raises(ValueError, match="Processor must be fitted before getting statistics"):
            processor.get_parameter_statistics('rainfall')
    
    def test_get_trigger_thresholds(self, fitted_processor):
        """Test trigger thresholds retrieval."""
        thresholds = fitted_processor.get_trigger_thresholds()
        
        # Check that all expected thresholds are present
        assert 'rainfall' in thresholds
        assert 'temperature_freeze' in thresholds
        assert 'temperature_extreme_high' in thresholds
        assert 'temperature_extreme_low' in thresholds
        assert 'vibrations' in thresholds
        assert 'wind_speed' in thresholds
        
        # Check that thresholds are numeric
        assert all(isinstance(threshold, (int, float)) for threshold in thresholds.values())
    
    def test_edge_cases_with_nan_values(self, fitted_processor):
        """Test handling of NaN values in environmental data."""
        # Test with NaN values
        nan_data = EnvironmentalData(
            rainfall=float('nan'),
            temperature=float('nan'),
            vibrations=float('nan'),
            wind_speed=float('nan')
        )
        
        features = fitted_processor.transform(nan_data)
        
        # Should handle NaN values gracefully (convert to zeros)
        assert np.all(np.isfinite(features))
        
        # Most features should be zero when all inputs are NaN
        non_zero_features = np.count_nonzero(features)
        assert non_zero_features <= len(features) * 0.1  # Allow some non-zero due to normalization
    
    def test_extreme_values_handling(self, fitted_processor):
        """Test handling of extreme values."""
        # Test with very large values
        extreme_data = EnvironmentalData(
            rainfall=1000.0,
            temperature=100.0,
            vibrations=1000.0,
            wind_speed=500.0
        )
        
        features = fitted_processor.transform(extreme_data)
        
        # Should handle extreme values without errors
        assert np.all(np.isfinite(features))
        assert len(features) == 30
    
    def test_different_normalization_methods(self, sample_environmental_data_list):
        """Test different normalization methods."""
        # Test standard normalization
        processor_std = EnvironmentalProcessor(config={'normalization': 'standard'})
        processor_std.fit(sample_environmental_data_list)
        
        # Test minmax normalization
        processor_minmax = EnvironmentalProcessor(config={'normalization': 'minmax'})
        processor_minmax.fit(sample_environmental_data_list)
        
        # Test robust normalization
        processor_robust = EnvironmentalProcessor(config={'normalization': 'robust'})
        processor_robust.fit(sample_environmental_data_list)
        
        # All should fit successfully
        assert processor_std.is_fitted
        assert processor_minmax.is_fitted
        assert processor_robust.is_fitted
        
        # Test transformation with each
        test_data = sample_environmental_data_list[0]
        features_std = processor_std.transform(test_data)
        features_minmax = processor_minmax.transform(test_data)
        features_robust = processor_robust.transform(test_data)
        
        # All should produce valid features
        assert np.all(np.isfinite(features_std))
        assert np.all(np.isfinite(features_minmax))
        assert np.all(np.isfinite(features_robust))