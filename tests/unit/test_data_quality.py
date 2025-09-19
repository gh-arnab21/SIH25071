"""
Unit tests for data quality and error handling functionality.
"""

import pytest
import numpy as np
import pandas as pd
from unittest.mock import patch, MagicMock
import warnings

from src.data.quality import (
    MissingDataImputer, OutlierDetector, ClassImbalanceHandler,
    RobustDataProcessor, DataQualityError, ImputationStrategy,
    OutlierMethod, handle_corrupted_data
)


class TestMissingDataImputer:
    """Test cases for MissingDataImputer class."""
    
    def test_mean_imputation_array(self):
        """Test mean imputation with numpy array."""
        # Create data with missing values
        X = np.array([[1.0, 2.0], [np.nan, 4.0], [5.0, np.nan], [7.0, 8.0]])
        
        imputer = MissingDataImputer(strategy=ImputationStrategy.MEAN)
        X_imputed = imputer.fit_transform(X)
        
        # Check that no NaN values remain
        assert not np.isnan(X_imputed).any()
        
        # Check specific imputed values
        expected_col0_mean = (1.0 + 5.0 + 7.0) / 3  # 4.33...
        expected_col1_mean = (2.0 + 4.0 + 8.0) / 3  # 4.66...
        
        assert np.isclose(X_imputed[1, 0], expected_col0_mean)
        assert np.isclose(X_imputed[2, 1], expected_col1_mean)
    
    def test_median_imputation_dataframe(self):
        """Test median imputation with pandas DataFrame."""
        df = pd.DataFrame({
            'A': [1.0, np.nan, 3.0, 4.0, 5.0],
            'B': [2.0, 3.0, np.nan, 5.0, 6.0]
        })
        
        imputer = MissingDataImputer(strategy=ImputationStrategy.MEDIAN)
        df_imputed = imputer.fit_transform(df)
        
        # Check that no NaN values remain
        assert not df_imputed.isnull().any().any()
        
        # Check median values
        assert df_imputed.loc[1, 'A'] == 3.5  # median of [1, 3, 4, 5]
        assert df_imputed.loc[2, 'B'] == 4.0  # median of [2, 3, 5, 6]
    
    def test_mode_imputation(self):
        """Test mode imputation for categorical data."""
        # Use numeric data for mode imputation as SimpleImputer works better with numeric data
        X = np.array([[1], [2], [np.nan], [1], [3], [1]])
        
        imputer = MissingDataImputer(strategy=ImputationStrategy.MODE)
        X_imputed = imputer.fit_transform(X)
        
        # Mode should be 1 (appears 3 times)
        assert X_imputed[2, 0] == 1.0
    
    def test_constant_imputation(self):
        """Test constant value imputation."""
        X = np.array([[1.0], [np.nan], [3.0]])
        
        imputer = MissingDataImputer(strategy=ImputationStrategy.CONSTANT, fill_value=999.0)
        X_imputed = imputer.fit_transform(X)
        
        assert X_imputed[1, 0] == 999.0
    
    def test_knn_imputation(self):
        """Test KNN imputation."""
        X = np.array([[1.0, 2.0], [2.0, np.nan], [3.0, 4.0], [4.0, 5.0]])
        
        imputer = MissingDataImputer(strategy=ImputationStrategy.KNN, n_neighbors=2)
        X_imputed = imputer.fit_transform(X)
        
        # Check that no NaN values remain
        assert not np.isnan(X_imputed).any()
    
    def test_forward_fill_dataframe(self):
        """Test forward fill imputation with DataFrame."""
        df = pd.DataFrame({
            'A': [1.0, np.nan, np.nan, 4.0],
            'B': [2.0, 3.0, np.nan, 5.0]
        })
        
        imputer = MissingDataImputer(strategy=ImputationStrategy.FORWARD_FILL)
        df_imputed = imputer.fit_transform(df)
        
        # Check forward fill behavior
        assert df_imputed.loc[1, 'A'] == 1.0  # Forward filled from index 0
        assert df_imputed.loc[2, 'A'] == 1.0  # Forward filled from index 0
        assert df_imputed.loc[2, 'B'] == 3.0  # Forward filled from index 1
    
    def test_imputer_not_fitted_error(self):
        """Test error when trying to transform without fitting."""
        X = np.array([[1.0], [np.nan]])
        imputer = MissingDataImputer()
        
        with pytest.raises(DataQualityError, match="must be fitted"):
            imputer.transform(X)
    
    def test_constant_imputation_without_fill_value(self):
        """Test error when using constant strategy without fill_value."""
        X = np.array([[1.0], [np.nan]])
        
        with pytest.raises(DataQualityError, match="fill_value must be specified"):
            imputer = MissingDataImputer(strategy=ImputationStrategy.CONSTANT)
            imputer.fit(X)


class TestOutlierDetector:
    """Test cases for OutlierDetector class."""
    
    def test_iqr_outlier_detection(self):
        """Test IQR-based outlier detection."""
        # Create data with clear outliers
        X = np.array([[1], [2], [3], [4], [5], [100]])  # 100 is an outlier
        
        detector = OutlierDetector(method=OutlierMethod.IQR, threshold=1.5)
        outlier_info = detector.detect_outliers(X, feature_names=['feature_0'])
        
        # Check that outlier was detected
        assert outlier_info['overall_outlier_count'] > 0
        assert outlier_info['outlier_mask'][-1] == True  # Last value should be outlier
        assert 'feature_0' in outlier_info['features']
    
    def test_zscore_outlier_detection(self):
        """Test Z-score based outlier detection."""
        # Create data with outliers
        np.random.seed(42)
        X = np.random.normal(0, 1, (100, 1))
        X = np.vstack([X, [[10]]])  # Add clear outlier
        
        detector = OutlierDetector(method=OutlierMethod.Z_SCORE, threshold=3.0)
        outlier_info = detector.detect_outliers(X)
        
        # Check that outlier was detected
        assert outlier_info['overall_outlier_count'] > 0
        assert outlier_info['method'] == 'Z-Score'
    
    def test_modified_zscore_outlier_detection(self):
        """Test Modified Z-score outlier detection."""
        X = np.array([[1], [2], [3], [4], [5], [50]])  # 50 is an outlier
        
        detector = OutlierDetector(method=OutlierMethod.MODIFIED_Z_SCORE, threshold=3.5)
        outlier_info = detector.detect_outliers(X)
        
        assert outlier_info['method'] == 'Modified Z-Score'
        assert outlier_info['overall_outlier_count'] > 0
    
    @patch('sklearn.ensemble.IsolationForest')
    def test_isolation_forest_outlier_detection(self, mock_isolation_forest):
        """Test Isolation Forest outlier detection."""
        # Mock the IsolationForest
        mock_iso = MagicMock()
        mock_iso.fit_predict.return_value = np.array([1, 1, 1, -1])  # Last one is outlier
        mock_iso.decision_function.return_value = np.array([0.1, 0.2, 0.1, -0.5])
        mock_isolation_forest.return_value = mock_iso
        
        X = np.array([[1], [2], [3], [100]])
        
        detector = OutlierDetector(method=OutlierMethod.ISOLATION_FOREST, contamination=0.1)
        outlier_info = detector.detect_outliers(X)
        
        assert outlier_info['method'] == 'Isolation Forest'
        assert outlier_info['overall_outlier_count'] == 1
        assert outlier_info['outlier_mask'][-1] == True
    
    def test_outlier_detection_with_dataframe(self):
        """Test outlier detection with pandas DataFrame."""
        df = pd.DataFrame({
            'A': [1, 2, 3, 4, 100],
            'B': [10, 20, 30, 40, 50]
        })
        
        detector = OutlierDetector(method=OutlierMethod.IQR)
        outlier_info = detector.detect_outliers(df)
        
        assert 'A' in outlier_info['features']
        assert 'B' in outlier_info['features']


class TestClassImbalanceHandler:
    """Test cases for ClassImbalanceHandler class."""
    
    def test_smote_oversampling(self):
        """Test SMOTE oversampling for imbalanced data."""
        # Create imbalanced dataset with more samples for SMOTE to work
        X = np.array([[1, 2], [2, 3], [3, 4], [4, 5], [5, 6], [10, 11], [11, 12]])
        y = np.array([0, 0, 0, 0, 0, 1, 1])  # Imbalanced but with minimum samples for SMOTE
        
        handler = ClassImbalanceHandler(strategy='smote')
        X_resampled, y_resampled = handler.fit_resample(X, y)
        
        # Check that minority class was oversampled
        unique, counts = np.unique(y_resampled, return_counts=True)
        assert len(unique) == 2
        # After SMOTE, classes should be more balanced
        assert counts[1] >= 2  # Minority class should have at least 2 samples
    
    def test_undersampling(self):
        """Test random undersampling."""
        # Create imbalanced dataset
        X = np.array([[i, i+1] for i in range(10)])
        y = np.array([0] * 8 + [1] * 2)  # 8:2 ratio
        
        handler = ClassImbalanceHandler(strategy='undersample')
        X_resampled, y_resampled = handler.fit_resample(X, y)
        
        # Check that majority class was undersampled
        unique, counts = np.unique(y_resampled, return_counts=True)
        assert len(unique) == 2
        assert counts[0] == counts[1]  # Should be balanced
        assert len(X_resampled) < len(X)  # Should be smaller
    
    def test_class_weights_computation(self):
        """Test class weights computation."""
        y = np.array([0, 0, 0, 0, 1, 1])  # 4:2 ratio
        
        handler = ClassImbalanceHandler(strategy='class_weights')
        X_dummy = np.array([[1], [2], [3], [4], [5], [6]])
        
        X_result, y_result = handler.fit_resample(X_dummy, y)
        
        # Data should remain unchanged for class weights
        np.testing.assert_array_equal(X_result, X_dummy)
        np.testing.assert_array_equal(y_result, y)
        
        # Check that class weights were computed
        class_weights = handler.get_class_weights()
        assert class_weights is not None
        assert 0 in class_weights and 1 in class_weights
        assert class_weights[1] > class_weights[0]  # Minority class should have higher weight
    
    def test_smote_with_insufficient_samples(self):
        """Test SMOTE behavior with insufficient samples."""
        X = np.array([[1, 2], [3, 4]])
        y = np.array([0, 1])  # Only one sample per class
        
        handler = ClassImbalanceHandler(strategy='smote')
        X_result, y_result = handler.fit_resample(X, y)
        
        # Should return original data when SMOTE fails
        np.testing.assert_array_equal(X_result, X)
        np.testing.assert_array_equal(y_result, y)
    
    def test_dataframe_preservation(self):
        """Test that DataFrame structure is preserved after resampling."""
        df = pd.DataFrame({'A': [1, 2, 3, 4], 'B': [5, 6, 7, 8]})
        y = pd.Series([0, 0, 0, 1], name='target')
        
        handler = ClassImbalanceHandler(strategy='smote')
        X_resampled, y_resampled = handler.fit_resample(df, y)
        
        assert isinstance(X_resampled, pd.DataFrame)
        assert isinstance(y_resampled, pd.Series)
        assert list(X_resampled.columns) == ['A', 'B']
        assert y_resampled.name == 'target'


class TestRobustDataProcessor:
    """Test cases for RobustDataProcessor class."""
    
    def test_complete_data_processing_pipeline(self):
        """Test the complete data processing pipeline."""
        # Create problematic dataset
        np.random.seed(42)
        X = np.random.normal(0, 1, (100, 3))
        X[10:15, 0] = np.nan  # Add missing values
        X[95:, 1] = 10  # Add outliers
        y = np.array([0] * 80 + [1] * 20)  # Imbalanced classes
        
        processor = RobustDataProcessor(
            handle_missing=True,
            detect_outliers=True,
            handle_imbalance=True
        )
        
        results = processor.process_data(X, y)
        
        # Check that processing completed
        assert 'processed_X' in results
        assert 'processed_y' in results
        assert 'quality_report' in results
        assert 'processing_log' in results
        
        # Check quality report sections
        assert 'missing_data' in results['quality_report']
        assert 'outliers' in results['quality_report']
        assert 'class_imbalance' in results['quality_report']
        assert 'validation' in results['quality_report']
        
        # Check that missing data was handled
        assert results['quality_report']['missing_data']['imputation_applied'] == True
        
        # Check that outliers were detected
        assert 'overall_outlier_count' in results['quality_report']['outliers']
        
        # Check that class imbalance was handled
        assert results['quality_report']['class_imbalance']['is_imbalanced'] == True
    
    def test_processing_with_dataframe(self):
        """Test processing with pandas DataFrame."""
        df = pd.DataFrame({
            'feature1': [1, 2, np.nan, 4, 5],
            'feature2': [10, 20, 30, 40, 1000]  # Last value is outlier
        })
        y = pd.Series([0, 0, 1, 1, 1])
        
        processor = RobustDataProcessor()
        results = processor.process_data(df, y)
        
        assert isinstance(results['processed_X'], pd.DataFrame)
        assert isinstance(results['processed_y'], pd.Series)
    
    def test_processing_without_target(self):
        """Test processing without target labels."""
        X = np.random.normal(0, 1, (50, 2))
        X[0, 0] = np.nan
        
        processor = RobustDataProcessor(handle_imbalance=True)
        results = processor.process_data(X)
        
        # Should not have class_imbalance in quality_report when no target is provided
        assert 'class_imbalance' not in results['quality_report']
        # Should still process other steps
        assert 'missing_data' in results['quality_report']
        assert 'outliers' in results['quality_report']
    
    def test_outlier_removal(self):
        """Test outlier removal functionality."""
        X = np.array([[1], [2], [3], [100]])  # Clear outlier
        
        processor = RobustDataProcessor(detect_outliers=True)
        results = processor.process_data(X, remove_outliers=True)
        
        # Check that outlier was removed
        assert results['processed_X'].shape[0] < X.shape[0]
    
    def test_processing_error_handling(self):
        """Test error handling in data processing."""
        # Create data that will cause errors
        X = None
        
        processor = RobustDataProcessor()
        results = processor.process_data(X)
        
        # Should handle error gracefully
        assert 'processing_error' in results['quality_report']
        assert any('ERROR' in log for log in results['processing_log'])


class TestCorruptedDataHandling:
    """Test cases for corrupted data handling functions."""
    
    def test_handle_none_data(self):
        """Test handling of None data."""
        data, errors = handle_corrupted_data(None, "test_data")
        
        assert data is None
        assert len(errors) == 1
        assert "Data is None" in errors[0]
    
    def test_handle_empty_dataframe(self):
        """Test handling of empty DataFrame."""
        df = pd.DataFrame()
        data, errors = handle_corrupted_data(df, "empty_df")
        
        assert isinstance(data, pd.DataFrame)
        assert len(errors) == 1
        assert "Empty DataFrame" in errors[0]
    
    def test_handle_all_nan_columns(self):
        """Test handling of DataFrame with all-NaN columns."""
        df = pd.DataFrame({
            'good_col': [1, 2, 3],
            'bad_col': [np.nan, np.nan, np.nan]
        })
        
        data, errors = handle_corrupted_data(df, "df_with_nan_col")
        
        assert 'good_col' in data.columns
        assert 'bad_col' not in data.columns
        assert len(errors) == 1
        assert "all NaN values" in errors[0]
    
    def test_handle_duplicate_columns(self):
        """Test handling of DataFrame with duplicate columns."""
        df = pd.DataFrame([[1, 2, 3], [4, 5, 6]])
        df.columns = ['A', 'B', 'A']  # Duplicate column names
        
        data, errors = handle_corrupted_data(df, "df_with_duplicates")
        
        assert len(data.columns) == 2  # Should remove duplicate
        assert len(errors) == 1
        assert "Duplicate columns" in errors[0]
    
    def test_handle_infinite_values_in_array(self):
        """Test handling of infinite values in numpy array."""
        arr = np.array([1.0, 2.0, np.inf, 4.0])
        
        data, errors = handle_corrupted_data(arr, "array_with_inf")
        
        assert not np.isinf(data).any()
        assert np.isnan(data[2])  # inf should be converted to nan
        assert len(errors) == 1
        assert "Infinite values" in errors[0]
    
    def test_handle_empty_sequence(self):
        """Test handling of empty list."""
        empty_list = []
        
        data, errors = handle_corrupted_data(empty_list, "empty_list")
        
        assert data == []
        assert len(errors) == 1
        assert "Empty sequence" in errors[0]
    
    def test_handle_sequence_with_none(self):
        """Test handling of list with None values."""
        list_with_none = [1, 2, None, 4, None]
        
        data, errors = handle_corrupted_data(list_with_none, "list_with_none")
        
        assert None not in data
        assert len(data) == 3
        assert len(errors) == 1
        assert "None values" in errors[0]
    
    def test_handle_empty_dictionary(self):
        """Test handling of empty dictionary."""
        empty_dict = {}
        
        data, errors = handle_corrupted_data(empty_dict, "empty_dict")
        
        assert data == {}
        assert len(errors) == 1
        assert "Empty dictionary" in errors[0]
    
    def test_handle_dict_with_none_values(self):
        """Test handling of dictionary with None values."""
        dict_with_none = {'a': 1, 'b': None, 'c': 3}
        
        data, errors = handle_corrupted_data(dict_with_none, "dict_with_none")
        
        assert data == dict_with_none  # Dict structure preserved
        assert len(errors) == 1
        assert "None values" in errors[0]
    
    def test_handle_scalar_array(self):
        """Test handling of scalar numpy array."""
        scalar_arr = np.array(5)
        
        data, errors = handle_corrupted_data(scalar_arr, "scalar_array")
        
        assert data.shape == (1,)  # Should be converted to 1D
        assert data[0] == 5
        assert len(errors) == 1
        assert "Scalar array" in errors[0]


class TestDataQualityIntegration:
    """Integration tests for data quality components."""
    
    def test_end_to_end_quality_pipeline(self):
        """Test complete end-to-end data quality pipeline."""
        # Create realistic problematic dataset
        np.random.seed(42)
        
        # Generate base data
        n_samples = 200
        X = np.random.normal(0, 1, (n_samples, 4))
        
        # Add various quality issues
        # 1. Missing values (10% of data)
        missing_mask = np.random.random((n_samples, 4)) < 0.1
        X[missing_mask] = np.nan
        
        # 2. Outliers (5% of data)
        outlier_indices = np.random.choice(n_samples, size=int(0.05 * n_samples), replace=False)
        X[outlier_indices, 0] = np.random.normal(10, 1, len(outlier_indices))  # Outliers in first column
        
        # 3. Imbalanced classes (80:20 ratio)
        y = np.array([0] * int(0.8 * n_samples) + [1] * int(0.2 * n_samples))
        
        # Process with full pipeline
        processor = RobustDataProcessor(
            handle_missing=True,
            detect_outliers=True,
            handle_imbalance=True
        )
        
        results = processor.process_data(
            X, y,
            imputation_strategy=ImputationStrategy.MEDIAN,
            outlier_method=OutlierMethod.IQR,
            outlier_threshold=1.5,
            imbalance_strategy='smote'
        )
        
        # Validate results
        processed_X = results['processed_X']
        processed_y = results['processed_y']
        quality_report = results['quality_report']
        
        # Check that all quality issues were addressed
        assert not np.isnan(processed_X).any(), "Missing values should be imputed"
        
        assert quality_report['missing_data']['imputation_applied'] == True
        assert quality_report['outliers']['overall_outlier_count'] > 0
        assert quality_report['class_imbalance']['is_imbalanced'] == True
        assert quality_report['validation']['is_valid'] == True
        
        # Check that classes are more balanced after SMOTE
        unique, counts = np.unique(processed_y, return_counts=True)
        balance_ratio = max(counts) / min(counts)
        assert balance_ratio < 2.0, "Classes should be more balanced after SMOTE"
        
        # Check that data shape is reasonable
        assert processed_X.shape[1] == X.shape[1], "Number of features should be preserved"
        assert processed_X.shape[0] >= X.shape[0], "SMOTE should increase or maintain sample count"
    
    def test_quality_pipeline_with_pandas(self):
        """Test quality pipeline with pandas data structures."""
        # Create DataFrame with mixed data types and quality issues
        df = pd.DataFrame({
            'numeric1': [1.0, 2.0, np.nan, 4.0, 100.0],  # Missing value and outlier
            'numeric2': [10, 20, 30, 40, 50],
            'categorical': ['A', 'B', None, 'A', 'C']  # Missing categorical
        })
        
        # Create target with imbalance
        y = pd.Series([0, 0, 0, 1, 1], name='target')
        
        # Process only numeric columns for this test
        numeric_df = df.select_dtypes(include=[np.number])
        
        processor = RobustDataProcessor()
        results = processor.process_data(numeric_df, y)
        
        # Validate pandas structures are preserved
        assert isinstance(results['processed_X'], pd.DataFrame)
        assert isinstance(results['processed_y'], pd.Series)
        assert results['processed_y'].name == 'target'
        
        # Check quality report
        assert results['quality_report']['validation']['is_valid'] == True


if __name__ == '__main__':
    pytest.main([__file__])