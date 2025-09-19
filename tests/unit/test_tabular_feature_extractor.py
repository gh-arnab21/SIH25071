"""Unit tests for TabularFeatureExtractor."""

import pytest
import numpy as np
import pandas as pd
from datetime import datetime
import tempfile
import os

from src.models.extractors.tabular_feature_extractor import TabularFeatureExtractor
from src.data.schemas import RockfallDataPoint, EnvironmentalData, GeoCoordinate, RiskLevel


class TestTabularFeatureExtractor:
    """Test cases for TabularFeatureExtractor."""
    
    @pytest.fixture
    def sample_environmental_data(self):
        """Create sample EnvironmentalData objects for testing."""
        return [
            EnvironmentalData(rainfall=10.5, temperature=15.2, vibrations=0.3, wind_speed=5.1),
            EnvironmentalData(rainfall=25.0, temperature=12.8, vibrations=0.8, wind_speed=8.2),
            EnvironmentalData(rainfall=5.2, temperature=18.5, vibrations=0.1, wind_speed=3.5),
            EnvironmentalData(rainfall=0.0, temperature=22.1, vibrations=0.0, wind_speed=2.1),
            EnvironmentalData(rainfall=35.8, temperature=8.9, vibrations=1.2, wind_speed=12.3)
        ]
    
    @pytest.fixture
    def sample_rockfall_datapoints(self, sample_environmental_data):
        """Create sample RockfallDataPoint objects for testing."""
        datapoints = []
        for i, env_data in enumerate(sample_environmental_data):
            datapoint = RockfallDataPoint(
                timestamp=datetime(2023, 1, 1 + i),
                location=GeoCoordinate(latitude=45.0 + i * 0.1, longitude=-120.0 + i * 0.1, elevation=1000 + i * 10),
                environmental=env_data,
                ground_truth=RiskLevel.LOW if i < 3 else RiskLevel.HIGH
            )
            datapoints.append(datapoint)
        return datapoints
    
    @pytest.fixture
    def sample_dict_data(self):
        """Create sample dictionary data for testing."""
        return [
            {'rainfall': 10.5, 'temperature': 15.2, 'vibrations': 0.3, 'wind_speed': 5.1, 'category': 'A'},
            {'rainfall': 25.0, 'temperature': 12.8, 'vibrations': 0.8, 'wind_speed': 8.2, 'category': 'B'},
            {'rainfall': 5.2, 'temperature': 18.5, 'vibrations': 0.1, 'wind_speed': 3.5, 'category': 'A'},
            {'rainfall': 0.0, 'temperature': 22.1, 'vibrations': 0.0, 'wind_speed': 2.1, 'category': 'C'},
            {'rainfall': 35.8, 'temperature': 8.9, 'vibrations': 1.2, 'wind_speed': 12.3, 'category': 'B'}
        ]
    
    def test_initialization_default_config(self):
        """Test TabularFeatureExtractor initialization with default configuration."""
        extractor = TabularFeatureExtractor()
        
        assert extractor.config['scaling_method'] == 'standard'
        assert extractor.config['encoding_method'] == 'onehot'
        assert extractor.config['feature_selection_method'] == 'kbest'
        assert extractor.config['n_features'] == 50
        assert extractor.config['imputation_strategy'] == 'median'
        assert extractor.config['handle_outliers'] is True
        assert extractor.config['outlier_method'] == 'iqr'
        assert extractor.config['outlier_threshold'] == 3.0
        assert extractor.config['domain_features'] is True
        assert not extractor._is_fitted
    
    def test_initialization_custom_config(self):
        """Test TabularFeatureExtractor initialization with custom configuration."""
        config = {
            'scaling_method': 'minmax',
            'encoding_method': 'label',
            'feature_selection_method': 'rfe',
            'n_features': 20,
            'imputation_strategy': 'mean',
            'handle_outliers': False,
            'domain_features': False
        }
        extractor = TabularFeatureExtractor(config)
        
        assert extractor.config['scaling_method'] == 'minmax'
        assert extractor.config['encoding_method'] == 'label'
        assert extractor.config['feature_selection_method'] == 'rfe'
        assert extractor.config['n_features'] == 20
        assert extractor.config['imputation_strategy'] == 'mean'
        assert extractor.config['handle_outliers'] is False
        assert extractor.config['domain_features'] is False
    
    def test_convert_to_dataframe_rockfall_datapoints(self, sample_rockfall_datapoints):
        """Test conversion of RockfallDataPoint objects to DataFrame."""
        extractor = TabularFeatureExtractor()
        df = extractor._convert_to_dataframe(sample_rockfall_datapoints)
        
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 5
        assert 'rainfall' in df.columns
        assert 'temperature' in df.columns
        assert 'vibrations' in df.columns
        assert 'wind_speed' in df.columns
        assert 'latitude' in df.columns
        assert 'longitude' in df.columns
        assert 'elevation' in df.columns
        assert 'timestamp' in df.columns
        
        # Check values
        assert df['rainfall'].iloc[0] == 10.5
        assert df['latitude'].iloc[0] == 45.0
        assert df['elevation'].iloc[0] == 1000
    
    def test_convert_to_dataframe_dict_data(self, sample_dict_data):
        """Test conversion of dictionary data to DataFrame."""
        extractor = TabularFeatureExtractor()
        df = extractor._convert_to_dataframe(sample_dict_data)
        
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 5
        assert 'rainfall' in df.columns
        assert 'category' in df.columns
        assert df['rainfall'].iloc[0] == 10.5
        assert df['category'].iloc[0] == 'A'
    
    def test_identify_feature_types(self, sample_dict_data):
        """Test identification of categorical and numerical features."""
        extractor = TabularFeatureExtractor()
        df = extractor._convert_to_dataframe(sample_dict_data)
        
        categorical_features, numerical_features = extractor._identify_feature_types(df)
        
        assert 'category' in categorical_features
        assert 'rainfall' in numerical_features
        assert 'temperature' in numerical_features
        assert 'vibrations' in numerical_features
        assert 'wind_speed' in numerical_features
    
    def test_create_domain_features(self, sample_dict_data):
        """Test creation of domain-specific features."""
        extractor = TabularFeatureExtractor()
        df = extractor._convert_to_dataframe(sample_dict_data)
        
        # Add timestamp column for temporal features
        df['timestamp'] = pd.date_range('2023-01-01', periods=len(df), freq='H')
        
        df_enhanced = extractor._create_domain_features(df)
        
        # Check rainfall features
        assert 'rainfall_cumulative_24h' in df_enhanced.columns
        assert 'rainfall_cumulative_7d' in df_enhanced.columns
        assert 'rainfall_intensity' in df_enhanced.columns
        assert 'antecedent_precipitation_index' in df_enhanced.columns
        
        # Check temperature features
        assert 'freeze_thaw_cycles' in df_enhanced.columns
        assert 'temp_daily_range' in df_enhanced.columns
        
        # Check vibration features
        assert 'vibration_peak' in df_enhanced.columns
        assert 'vibration_rms' in df_enhanced.columns
        assert 'vibration_exceedance' in df_enhanced.columns
        
        # Check wind features
        assert 'wind_gust_factor' in df_enhanced.columns
        
        # Check combined stress index
        assert 'environmental_stress_index' in df_enhanced.columns
        
        # Check temporal features
        assert 'hour' in df_enhanced.columns
        assert 'day_of_week' in df_enhanced.columns
        assert 'month' in df_enhanced.columns
        assert 'season' in df_enhanced.columns
    
    def test_detect_outliers_iqr_method(self):
        """Test outlier detection using IQR method."""
        extractor = TabularFeatureExtractor()
        
        # Create data with outliers
        data = pd.DataFrame({
            'feature1': [1, 2, 3, 4, 5, 100],  # 100 is an outlier
            'feature2': [10, 20, 30, 40, 50, 60]
        })
        
        df_clean = extractor._detect_outliers(data, method='iqr')
        
        # Check that outlier was capped
        assert df_clean['feature1'].max() < 100
        assert df_clean['feature2'].equals(data['feature2'])  # No outliers in feature2
    
    def test_detect_outliers_zscore_method(self):
        """Test outlier detection using Z-score method."""
        extractor = TabularFeatureExtractor()
        
        # Create data with outliers
        data = pd.DataFrame({
            'feature1': [1, 2, 3, 4, 5, 100],  # 100 is an outlier
            'feature2': [10, 20, 30, 40, 50, 60]
        })
        
        df_clean = extractor._detect_outliers(data, method='zscore', threshold=2.0)
        
        # Check that outlier was replaced with median
        assert df_clean['feature1'].iloc[-1] == data['feature1'].median()
    
    def test_fit_transform_dict_data(self, sample_dict_data):
        """Test fitting and transforming dictionary data."""
        extractor = TabularFeatureExtractor({'domain_features': False, 'handle_outliers': False})
        
        # Create target labels
        y = np.array([0, 1, 0, 0, 1])
        
        # Fit and transform
        X = extractor.fit(sample_dict_data, y).transform(sample_dict_data)
        
        assert isinstance(X, np.ndarray)
        assert X.shape[0] == 5  # 5 samples
        assert X.shape[1] > 0   # Should have features
        assert extractor._is_fitted
        assert extractor.feature_dim == X.shape[1]
    
    def test_fit_transform_rockfall_datapoints(self, sample_rockfall_datapoints):
        """Test fitting and transforming RockfallDataPoint objects."""
        extractor = TabularFeatureExtractor({'domain_features': True, 'handle_outliers': True})
        
        # Create target labels
        y = np.array([0, 0, 0, 1, 1])
        
        # Fit and transform
        X = extractor.fit(sample_rockfall_datapoints, y).transform(sample_rockfall_datapoints)
        
        assert isinstance(X, np.ndarray)
        assert X.shape[0] == 5  # 5 samples
        assert X.shape[1] > 0   # Should have features
        assert extractor._is_fitted
    
    def test_extract_features_alias(self, sample_dict_data):
        """Test that extract_features is an alias for transform."""
        extractor = TabularFeatureExtractor({'domain_features': False})
        extractor.fit(sample_dict_data)
        
        X1 = extractor.transform(sample_dict_data)
        X2 = extractor.extract_features(sample_dict_data)
        
        np.testing.assert_array_equal(X1, X2)
    
    def test_transform_single_datapoint(self, sample_dict_data):
        """Test transforming a single data point."""
        extractor = TabularFeatureExtractor({'domain_features': False})
        extractor.fit(sample_dict_data)
        
        # Transform single data point
        single_point = sample_dict_data[0]
        X = extractor.transform(single_point)
        
        assert isinstance(X, np.ndarray)
        assert X.shape[0] == 1  # Single sample
        assert X.shape[1] == extractor.feature_dim
    
    def test_transform_before_fit_raises_error(self, sample_dict_data):
        """Test that transform raises error when called before fit."""
        extractor = TabularFeatureExtractor()
        
        with pytest.raises(ValueError, match="must be fitted before transform"):
            extractor.transform(sample_dict_data)
    
    def test_get_feature_names(self, sample_dict_data):
        """Test getting feature names after fitting."""
        extractor = TabularFeatureExtractor({'domain_features': False, 'feature_selection_method': None})
        extractor.fit(sample_dict_data)
        
        feature_names = extractor.get_feature_names()
        
        assert isinstance(feature_names, list)
        assert len(feature_names) > 0
        assert len(feature_names) == extractor.feature_dim
    
    def test_save_and_load(self, sample_dict_data):
        """Test saving and loading the fitted extractor."""
        extractor = TabularFeatureExtractor({'domain_features': False})
        extractor.fit(sample_dict_data)
        
        # Save to temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pkl') as tmp_file:
            filepath = tmp_file.name
        
        try:
            extractor.save(filepath)
            
            # Load the extractor
            loaded_extractor = TabularFeatureExtractor.load(filepath)
            
            # Test that loaded extractor works the same
            X_original = extractor.transform(sample_dict_data)
            X_loaded = loaded_extractor.transform(sample_dict_data)
            
            np.testing.assert_array_equal(X_original, X_loaded)
            assert loaded_extractor._is_fitted
            assert loaded_extractor.feature_dim == extractor.feature_dim
            
        finally:
            # Clean up
            if os.path.exists(filepath):
                os.unlink(filepath)
    
    def test_save_unfitted_raises_error(self):
        """Test that saving unfitted extractor raises error."""
        extractor = TabularFeatureExtractor()
        
        with pytest.raises(ValueError, match="Cannot save unfitted extractor"):
            extractor.save("test.pkl")
    
    def test_different_scaling_methods(self, sample_dict_data):
        """Test different scaling methods."""
        scaling_methods = ['standard', 'minmax', 'robust']
        
        for method in scaling_methods:
            extractor = TabularFeatureExtractor({
                'scaling_method': method,
                'domain_features': False
            })
            
            X = extractor.fit(sample_dict_data).transform(sample_dict_data)
            
            assert isinstance(X, np.ndarray)
            assert X.shape[0] == 5
            assert X.shape[1] > 0
    
    def test_different_encoding_methods(self, sample_dict_data):
        """Test different encoding methods."""
        encoding_methods = ['onehot', 'label']
        
        for method in encoding_methods:
            extractor = TabularFeatureExtractor({
                'encoding_method': method,
                'domain_features': False
            })
            
            X = extractor.fit(sample_dict_data).transform(sample_dict_data)
            
            assert isinstance(X, np.ndarray)
            assert X.shape[0] == 5
            assert X.shape[1] > 0
    
    def test_different_imputation_strategies(self):
        """Test different imputation strategies."""
        # Create data with missing values
        data_with_missing = [
            {'rainfall': 10.5, 'temperature': None, 'category': 'A'},
            {'rainfall': None, 'temperature': 12.8, 'category': 'B'},
            {'rainfall': 5.2, 'temperature': 18.5, 'category': None},
            {'rainfall': 0.0, 'temperature': 22.1, 'category': 'A'}
        ]
        
        strategies = ['mean', 'median', 'most_frequent']
        
        for strategy in strategies:
            extractor = TabularFeatureExtractor({
                'imputation_strategy': strategy,
                'domain_features': False
            })
            
            X = extractor.fit(data_with_missing).transform(data_with_missing)
            
            assert isinstance(X, np.ndarray)
            assert X.shape[0] == 4
            assert not np.isnan(X).any()  # No NaN values after imputation
    
    def test_feature_selection_methods(self, sample_dict_data):
        """Test different feature selection methods."""
        selection_methods = ['kbest', 'mutual_info', 'rfe']
        y = np.array([0, 1, 0, 0, 1])
        
        for method in selection_methods:
            extractor = TabularFeatureExtractor({
                'feature_selection_method': method,
                'n_features': 3,
                'domain_features': False
            })
            
            X = extractor.fit(sample_dict_data, y).transform(sample_dict_data)
            
            assert isinstance(X, np.ndarray)
            assert X.shape[0] == 5
            assert X.shape[1] <= 3  # Should select at most 3 features
    
    def test_empty_data_handling(self):
        """Test handling of empty data."""
        extractor = TabularFeatureExtractor()
        
        # Test with empty list
        df = extractor._convert_to_dataframe([])
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 0
    
    def test_missing_environmental_data(self):
        """Test handling of RockfallDataPoint with missing environmental data."""
        datapoint = RockfallDataPoint(
            timestamp=datetime(2023, 1, 1),
            location=GeoCoordinate(latitude=45.0, longitude=-120.0),
            environmental=None  # Missing environmental data
        )
        
        extractor = TabularFeatureExtractor()
        df = extractor._convert_to_dataframe([datapoint])
        
        # Should still create DataFrame with location data
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 1
        assert 'latitude' in df.columns
        assert 'longitude' in df.columns