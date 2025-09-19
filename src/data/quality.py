"""
Data quality and error handling module for the Rockfall Prediction System.

This module provides comprehensive data quality functions including missing data
imputation, outlier detection, class imbalance handling, and robust error handling
for corrupted or invalid data.
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Any, Optional, Union, Tuple
from enum import Enum
import logging
from scipy import stats
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer, KNNImputer
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from sklearn.utils.class_weight import compute_class_weight
import warnings

logger = logging.getLogger(__name__)


class ImputationStrategy(Enum):
    """Enumeration of available imputation strategies."""
    MEAN = "mean"
    MEDIAN = "median"
    MODE = "most_frequent"
    KNN = "knn"
    FORWARD_FILL = "ffill"
    BACKWARD_FILL = "bfill"
    CONSTANT = "constant"


class OutlierMethod(Enum):
    """Enumeration of available outlier detection methods."""
    IQR = "iqr"
    Z_SCORE = "z_score"
    MODIFIED_Z_SCORE = "modified_z_score"
    ISOLATION_FOREST = "isolation_forest"


class DataQualityError(Exception):
    """Custom exception for data quality issues."""
    pass


class MissingDataImputer:
    """
    Handles missing data imputation using various strategies.
    """
    
    def __init__(self, strategy: ImputationStrategy = ImputationStrategy.MEDIAN, 
                 fill_value: Optional[Union[str, int, float]] = None,
                 n_neighbors: int = 5):
        """
        Initialize the imputer with specified strategy.
        
        Args:
            strategy: Imputation strategy to use
            fill_value: Value to use for constant imputation
            n_neighbors: Number of neighbors for KNN imputation
        """
        self.strategy = strategy
        self.fill_value = fill_value
        self.n_neighbors = n_neighbors
        self.imputer = None
        self.fitted = False
    
    def fit(self, X: Union[np.ndarray, pd.DataFrame]) -> 'MissingDataImputer':
        """
        Fit the imputer on training data.
        
        Args:
            X: Training data to fit imputer on
            
        Returns:
            Self for method chaining
        """
        try:
            if self.strategy == ImputationStrategy.KNN:
                self.imputer = KNNImputer(n_neighbors=self.n_neighbors)
            elif self.strategy == ImputationStrategy.CONSTANT:
                if self.fill_value is None:
                    raise DataQualityError("fill_value must be specified for constant imputation")
                self.imputer = SimpleImputer(strategy="constant", fill_value=self.fill_value)
            elif self.strategy in [ImputationStrategy.FORWARD_FILL, ImputationStrategy.BACKWARD_FILL]:
                # For pandas-specific methods, we'll handle these in transform
                self.imputer = None
            else:
                self.imputer = SimpleImputer(strategy=self.strategy.value)
            
            if self.imputer is not None:
                self.imputer.fit(X)
            
            self.fitted = True
            logger.info(f"Imputer fitted with strategy: {self.strategy.value}")
            return self
            
        except Exception as e:
            raise DataQualityError(f"Failed to fit imputer: {str(e)}")
    
    def transform(self, X: Union[np.ndarray, pd.DataFrame]) -> Union[np.ndarray, pd.DataFrame]:
        """
        Transform data using fitted imputer.
        
        Args:
            X: Data to impute missing values for
            
        Returns:
            Data with missing values imputed
        """
        if not self.fitted:
            raise DataQualityError("Imputer must be fitted before transform")
        
        try:
            if isinstance(X, pd.DataFrame):
                return self._transform_dataframe(X)
            else:
                return self._transform_array(X)
                
        except Exception as e:
            raise DataQualityError(f"Failed to transform data: {str(e)}")
    
    def _transform_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """Transform pandas DataFrame."""
        df_copy = df.copy()
        
        if self.strategy == ImputationStrategy.FORWARD_FILL:
            return df_copy.ffill()
        elif self.strategy == ImputationStrategy.BACKWARD_FILL:
            return df_copy.bfill()
        elif self.imputer is not None:
            imputed_values = self.imputer.transform(df_copy)
            return pd.DataFrame(imputed_values, columns=df.columns, index=df.index)
        else:
            return df_copy
    
    def _transform_array(self, X: np.ndarray) -> np.ndarray:
        """Transform numpy array."""
        if self.strategy in [ImputationStrategy.FORWARD_FILL, ImputationStrategy.BACKWARD_FILL]:
            # Convert to DataFrame for pandas methods, then back to array
            df = pd.DataFrame(X)
            if self.strategy == ImputationStrategy.FORWARD_FILL:
                return df.ffill().values
            else:
                return df.bfill().values
        elif self.imputer is not None:
            return self.imputer.transform(X)
        else:
            return X
    
    def fit_transform(self, X: Union[np.ndarray, pd.DataFrame]) -> Union[np.ndarray, pd.DataFrame]:
        """
        Fit imputer and transform data in one step.
        
        Args:
            X: Data to fit imputer on and transform
            
        Returns:
            Data with missing values imputed
        """
        return self.fit(X).transform(X)


class OutlierDetector:
    """
    Detects outliers using various statistical methods.
    """
    
    def __init__(self, method: OutlierMethod = OutlierMethod.IQR, 
                 threshold: float = 1.5, contamination: float = 0.1):
        """
        Initialize outlier detector.
        
        Args:
            method: Outlier detection method to use
            threshold: Threshold for outlier detection (method-specific)
            contamination: Expected proportion of outliers (for isolation forest)
        """
        self.method = method
        self.threshold = threshold
        self.contamination = contamination
        self.fitted_params = {}
    
    def detect_outliers(self, X: Union[np.ndarray, pd.DataFrame], 
                       feature_names: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Detect outliers in the data.
        
        Args:
            X: Data to detect outliers in
            feature_names: Names of features (for reporting)
            
        Returns:
            Dictionary containing outlier information
        """
        try:
            if isinstance(X, pd.DataFrame):
                feature_names = feature_names or X.columns.tolist()
                X_array = X.values
            else:
                X_array = X
                feature_names = feature_names or [f"feature_{i}" for i in range(X_array.shape[1])]
            
            if self.method == OutlierMethod.IQR:
                return self._detect_iqr_outliers(X_array, feature_names)
            elif self.method == OutlierMethod.Z_SCORE:
                return self._detect_zscore_outliers(X_array, feature_names)
            elif self.method == OutlierMethod.MODIFIED_Z_SCORE:
                return self._detect_modified_zscore_outliers(X_array, feature_names)
            elif self.method == OutlierMethod.ISOLATION_FOREST:
                return self._detect_isolation_forest_outliers(X_array, feature_names)
            else:
                raise DataQualityError(f"Unknown outlier detection method: {self.method}")
                
        except Exception as e:
            raise DataQualityError(f"Failed to detect outliers: {str(e)}")
    
    def _detect_iqr_outliers(self, X: np.ndarray, feature_names: List[str]) -> Dict[str, Any]:
        """Detect outliers using Interquartile Range method."""
        outlier_mask = np.zeros(X.shape, dtype=bool)
        outlier_info = {'method': 'IQR', 'threshold': self.threshold, 'features': {}}
        
        for i, feature_name in enumerate(feature_names):
            feature_data = X[:, i]
            q1 = np.percentile(feature_data, 25)
            q3 = np.percentile(feature_data, 75)
            iqr = q3 - q1
            
            lower_bound = q1 - self.threshold * iqr
            upper_bound = q3 + self.threshold * iqr
            
            feature_outliers = (feature_data < lower_bound) | (feature_data > upper_bound)
            outlier_mask[:, i] = feature_outliers
            
            outlier_info['features'][feature_name] = {
                'outlier_count': np.sum(feature_outliers),
                'outlier_percentage': (np.sum(feature_outliers) / len(feature_data)) * 100,
                'bounds': {'lower': lower_bound, 'upper': upper_bound},
                'q1': q1, 'q3': q3, 'iqr': iqr
            }
        
        # Overall outlier mask (any feature is outlier)
        overall_outliers = np.any(outlier_mask, axis=1)
        outlier_info['overall_outlier_count'] = np.sum(overall_outliers)
        outlier_info['overall_outlier_percentage'] = (np.sum(overall_outliers) / X.shape[0]) * 100
        outlier_info['outlier_mask'] = overall_outliers
        outlier_info['feature_outlier_mask'] = outlier_mask
        
        return outlier_info
    
    def _detect_zscore_outliers(self, X: np.ndarray, feature_names: List[str]) -> Dict[str, Any]:
        """Detect outliers using Z-score method."""
        outlier_mask = np.zeros(X.shape, dtype=bool)
        outlier_info = {'method': 'Z-Score', 'threshold': self.threshold, 'features': {}}
        
        for i, feature_name in enumerate(feature_names):
            feature_data = X[:, i]
            z_scores = np.abs(stats.zscore(feature_data))
            feature_outliers = z_scores > self.threshold
            outlier_mask[:, i] = feature_outliers
            
            outlier_info['features'][feature_name] = {
                'outlier_count': np.sum(feature_outliers),
                'outlier_percentage': (np.sum(feature_outliers) / len(feature_data)) * 100,
                'max_z_score': np.max(z_scores),
                'mean_z_score': np.mean(z_scores)
            }
        
        overall_outliers = np.any(outlier_mask, axis=1)
        outlier_info['overall_outlier_count'] = np.sum(overall_outliers)
        outlier_info['overall_outlier_percentage'] = (np.sum(overall_outliers) / X.shape[0]) * 100
        outlier_info['outlier_mask'] = overall_outliers
        outlier_info['feature_outlier_mask'] = outlier_mask
        
        return outlier_info
    
    def _detect_modified_zscore_outliers(self, X: np.ndarray, feature_names: List[str]) -> Dict[str, Any]:
        """Detect outliers using Modified Z-score method (using median)."""
        outlier_mask = np.zeros(X.shape, dtype=bool)
        outlier_info = {'method': 'Modified Z-Score', 'threshold': self.threshold, 'features': {}}
        
        for i, feature_name in enumerate(feature_names):
            feature_data = X[:, i]
            median = np.median(feature_data)
            mad = np.median(np.abs(feature_data - median))
            
            # Avoid division by zero
            if mad == 0:
                mad = np.mean(np.abs(feature_data - median))
            
            modified_z_scores = 0.6745 * (feature_data - median) / mad
            feature_outliers = np.abs(modified_z_scores) > self.threshold
            outlier_mask[:, i] = feature_outliers
            
            outlier_info['features'][feature_name] = {
                'outlier_count': np.sum(feature_outliers),
                'outlier_percentage': (np.sum(feature_outliers) / len(feature_data)) * 100,
                'max_modified_z_score': np.max(np.abs(modified_z_scores)),
                'median': median,
                'mad': mad
            }
        
        overall_outliers = np.any(outlier_mask, axis=1)
        outlier_info['overall_outlier_count'] = np.sum(overall_outliers)
        outlier_info['overall_outlier_percentage'] = (np.sum(overall_outliers) / X.shape[0]) * 100
        outlier_info['outlier_mask'] = overall_outliers
        outlier_info['feature_outlier_mask'] = outlier_mask
        
        return outlier_info
    
    def _detect_isolation_forest_outliers(self, X: np.ndarray, feature_names: List[str]) -> Dict[str, Any]:
        """Detect outliers using Isolation Forest method."""
        try:
            from sklearn.ensemble import IsolationForest
            
            iso_forest = IsolationForest(contamination=self.contamination, random_state=42)
            outlier_predictions = iso_forest.fit_predict(X)
            outlier_mask = outlier_predictions == -1
            
            outlier_info = {
                'method': 'Isolation Forest',
                'contamination': self.contamination,
                'overall_outlier_count': np.sum(outlier_mask),
                'overall_outlier_percentage': (np.sum(outlier_mask) / X.shape[0]) * 100,
                'outlier_mask': outlier_mask,
                'anomaly_scores': iso_forest.decision_function(X)
            }
            
            return outlier_info
            
        except ImportError:
            raise DataQualityError("scikit-learn is required for Isolation Forest outlier detection")


class ClassImbalanceHandler:
    """
    Handles class imbalance using various techniques including SMOTE and class weighting.
    """
    
    def __init__(self, strategy: str = 'smote', random_state: int = 42):
        """
        Initialize class imbalance handler.
        
        Args:
            strategy: Strategy to handle imbalance ('smote', 'undersample', 'class_weights')
            random_state: Random state for reproducibility
        """
        self.strategy = strategy
        self.random_state = random_state
        self.sampler = None
        self.class_weights = None
    
    def fit_resample(self, X: Union[np.ndarray, pd.DataFrame], 
                    y: Union[np.ndarray, pd.Series]) -> Tuple[Union[np.ndarray, pd.DataFrame], 
                                                             Union[np.ndarray, pd.Series]]:
        """
        Apply resampling strategy to balance classes.
        
        Args:
            X: Feature data
            y: Target labels
            
        Returns:
            Tuple of (resampled_X, resampled_y)
        """
        try:
            if self.strategy == 'smote':
                return self._apply_smote(X, y)
            elif self.strategy == 'undersample':
                return self._apply_undersampling(X, y)
            elif self.strategy == 'class_weights':
                # For class weights, we don't resample but compute weights
                self._compute_class_weights(y)
                return X, y
            else:
                raise DataQualityError(f"Unknown imbalance strategy: {self.strategy}")
                
        except Exception as e:
            raise DataQualityError(f"Failed to handle class imbalance: {str(e)}")
    
    def _apply_smote(self, X: Union[np.ndarray, pd.DataFrame], 
                    y: Union[np.ndarray, pd.Series]) -> Tuple[Union[np.ndarray, pd.DataFrame], 
                                                             Union[np.ndarray, pd.Series]]:
        """Apply SMOTE oversampling."""
        try:
            # Check if we have enough samples for SMOTE
            unique_classes, class_counts = np.unique(y, return_counts=True)
            min_samples = np.min(class_counts)
            
            if min_samples < 2:
                logger.warning("Not enough samples for SMOTE. Returning original data.")
                return X, y
            
            # Adjust k_neighbors if necessary
            k_neighbors = min(5, min_samples - 1)
            
            smote = SMOTE(random_state=self.random_state, k_neighbors=k_neighbors)
            X_resampled, y_resampled = smote.fit_resample(X, y)
            
            logger.info(f"SMOTE applied. Original shape: {X.shape}, New shape: {X_resampled.shape}")
            
            # Preserve DataFrame structure if input was DataFrame
            if isinstance(X, pd.DataFrame):
                X_resampled = pd.DataFrame(X_resampled, columns=X.columns)
            if isinstance(y, pd.Series):
                y_resampled = pd.Series(y_resampled, name=y.name)
            
            return X_resampled, y_resampled
            
        except Exception as e:
            logger.warning(f"SMOTE failed: {str(e)}. Returning original data.")
            return X, y
    
    def _apply_undersampling(self, X: Union[np.ndarray, pd.DataFrame], 
                           y: Union[np.ndarray, pd.Series]) -> Tuple[Union[np.ndarray, pd.DataFrame], 
                                                                   Union[np.ndarray, pd.Series]]:
        """Apply random undersampling."""
        undersampler = RandomUnderSampler(random_state=self.random_state)
        X_resampled, y_resampled = undersampler.fit_resample(X, y)
        
        logger.info(f"Undersampling applied. Original shape: {X.shape}, New shape: {X_resampled.shape}")
        
        # Preserve DataFrame structure if input was DataFrame
        if isinstance(X, pd.DataFrame):
            X_resampled = pd.DataFrame(X_resampled, columns=X.columns)
        if isinstance(y, pd.Series):
            y_resampled = pd.Series(y_resampled, name=y.name)
        
        return X_resampled, y_resampled
    
    def _compute_class_weights(self, y: Union[np.ndarray, pd.Series]) -> Dict[int, float]:
        """Compute class weights for imbalanced data."""
        unique_classes = np.unique(y)
        class_weights = compute_class_weight('balanced', classes=unique_classes, y=y)
        self.class_weights = dict(zip(unique_classes, class_weights))
        
        logger.info(f"Class weights computed: {self.class_weights}")
        return self.class_weights
    
    def get_class_weights(self) -> Optional[Dict[int, float]]:
        """Get computed class weights."""
        return self.class_weights


class RobustDataProcessor:
    """
    Provides robust data processing with comprehensive error handling.
    """
    
    def __init__(self, handle_missing: bool = True, 
                 detect_outliers: bool = True,
                 handle_imbalance: bool = True):
        """
        Initialize robust data processor.
        
        Args:
            handle_missing: Whether to handle missing data
            detect_outliers: Whether to detect and report outliers
            handle_imbalance: Whether to handle class imbalance
        """
        self.handle_missing = handle_missing
        self.detect_outliers = detect_outliers
        self.handle_imbalance = handle_imbalance
        
        self.imputer = None
        self.outlier_detector = None
        self.imbalance_handler = None
        self.processing_log = []
    
    def process_data(self, X: Union[np.ndarray, pd.DataFrame], 
                    y: Optional[Union[np.ndarray, pd.Series]] = None,
                    **kwargs) -> Dict[str, Any]:
        """
        Process data with comprehensive quality checks and corrections.
        
        Args:
            X: Feature data
            y: Target labels (optional)
            **kwargs: Additional parameters for processing steps
            
        Returns:
            Dictionary containing processed data and quality report
        """
        results = {
            'processed_X': X.copy() if hasattr(X, 'copy') else X,
            'processed_y': y.copy() if y is not None and hasattr(y, 'copy') else y,
            'quality_report': {},
            'processing_log': []
        }
        
        try:
            # Step 1: Handle missing data
            if self.handle_missing:
                results = self._process_missing_data(results, **kwargs)
            
            # Step 2: Detect outliers
            if self.detect_outliers:
                results = self._process_outliers(results, **kwargs)
            
            # Step 3: Handle class imbalance
            if self.handle_imbalance and y is not None:
                results = self._process_class_imbalance(results, **kwargs)
            
            # Step 4: Final validation
            results = self._validate_processed_data(results)
            
            logger.info("Data processing completed successfully")
            return results
            
        except Exception as e:
            error_msg = f"Data processing failed: {str(e)}"
            logger.error(error_msg)
            results['processing_log'].append(f"ERROR: {error_msg}")
            results['quality_report']['processing_error'] = error_msg
            return results
    
    def _process_missing_data(self, results: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        """Process missing data in the dataset."""
        X = results['processed_X']
        
        # Check for missing data
        if isinstance(X, pd.DataFrame):
            missing_info = X.isnull().sum()
            total_missing = missing_info.sum()
        else:
            missing_mask = np.isnan(X) if X.dtype.kind in 'fc' else np.equal(X, None)
            total_missing = np.sum(missing_mask)
            missing_info = np.sum(missing_mask, axis=0)
        
        results['quality_report']['missing_data'] = {
            'total_missing_values': int(total_missing),
            'missing_percentage': (total_missing / X.size) * 100 if X.size > 0 else 0
        }
        
        if total_missing > 0:
            # Apply imputation
            imputation_strategy = kwargs.get('imputation_strategy', ImputationStrategy.MEDIAN)
            self.imputer = MissingDataImputer(strategy=imputation_strategy)
            
            try:
                results['processed_X'] = self.imputer.fit_transform(X)
                results['processing_log'].append(f"Missing data imputed using {imputation_strategy.value}")
                results['quality_report']['missing_data']['imputation_applied'] = True
                results['quality_report']['missing_data']['imputation_strategy'] = imputation_strategy.value
            except Exception as e:
                results['processing_log'].append(f"WARNING: Imputation failed: {str(e)}")
                results['quality_report']['missing_data']['imputation_applied'] = False
        else:
            results['processing_log'].append("No missing data detected")
            results['quality_report']['missing_data']['imputation_applied'] = False
        
        return results
    
    def _process_outliers(self, results: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        """Detect and report outliers in the dataset."""
        X = results['processed_X']
        
        outlier_method = kwargs.get('outlier_method', OutlierMethod.IQR)
        outlier_threshold = kwargs.get('outlier_threshold', 1.5)
        
        self.outlier_detector = OutlierDetector(method=outlier_method, threshold=outlier_threshold)
        
        try:
            outlier_info = self.outlier_detector.detect_outliers(X)
            results['quality_report']['outliers'] = outlier_info
            results['processing_log'].append(f"Outlier detection completed using {outlier_method.value}")
            
            # Optionally remove outliers if requested
            if kwargs.get('remove_outliers', False):
                outlier_mask = outlier_info['outlier_mask']
                if isinstance(X, pd.DataFrame):
                    results['processed_X'] = X[~outlier_mask]
                    if results['processed_y'] is not None:
                        if isinstance(results['processed_y'], pd.Series):
                            results['processed_y'] = results['processed_y'][~outlier_mask]
                        else:
                            results['processed_y'] = results['processed_y'][~outlier_mask]
                else:
                    results['processed_X'] = X[~outlier_mask]
                    if results['processed_y'] is not None:
                        results['processed_y'] = results['processed_y'][~outlier_mask]
                
                results['processing_log'].append(f"Removed {np.sum(outlier_mask)} outlier samples")
                
        except Exception as e:
            results['processing_log'].append(f"WARNING: Outlier detection failed: {str(e)}")
            results['quality_report']['outliers'] = {'error': str(e)}
        
        return results
    
    def _process_class_imbalance(self, results: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        """Handle class imbalance in the dataset."""
        X = results['processed_X']
        y = results['processed_y']
        
        if y is None:
            results['processing_log'].append("No target labels provided, skipping imbalance handling")
            return results
        
        # Check class distribution
        unique_classes, class_counts = np.unique(y, return_counts=True)
        class_distribution = dict(zip(unique_classes, class_counts))
        
        # Calculate imbalance ratio
        max_count = np.max(class_counts)
        min_count = np.min(class_counts)
        imbalance_ratio = max_count / min_count if min_count > 0 else float('inf')
        
        results['quality_report']['class_imbalance'] = {
            'class_distribution': {str(k): int(v) for k, v in class_distribution.items()},
            'imbalance_ratio': float(imbalance_ratio),
            'is_imbalanced': imbalance_ratio > 1.5
        }
        
        if imbalance_ratio > 1.5:  # Consider imbalanced if ratio > 1.5
            imbalance_strategy = kwargs.get('imbalance_strategy', 'smote')
            self.imbalance_handler = ClassImbalanceHandler(strategy=imbalance_strategy)
            
            try:
                X_resampled, y_resampled = self.imbalance_handler.fit_resample(X, y)
                results['processed_X'] = X_resampled
                results['processed_y'] = y_resampled
                
                # Update class distribution after resampling
                if imbalance_strategy != 'class_weights':
                    new_unique, new_counts = np.unique(y_resampled, return_counts=True)
                    new_distribution = dict(zip(new_unique, new_counts))
                    results['quality_report']['class_imbalance']['resampled_distribution'] = {
                        str(k): int(v) for k, v in new_distribution.items()
                    }
                
                results['processing_log'].append(f"Class imbalance handled using {imbalance_strategy}")
                results['quality_report']['class_imbalance']['strategy_applied'] = imbalance_strategy
                
                if imbalance_strategy == 'class_weights':
                    results['quality_report']['class_imbalance']['class_weights'] = self.imbalance_handler.get_class_weights()
                
            except Exception as e:
                results['processing_log'].append(f"WARNING: Imbalance handling failed: {str(e)}")
                results['quality_report']['class_imbalance']['strategy_applied'] = None
        else:
            results['processing_log'].append("Dataset is reasonably balanced, no resampling applied")
            results['quality_report']['class_imbalance']['strategy_applied'] = None
        
        return results
    
    def _validate_processed_data(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Validate the processed data for quality and consistency."""
        X = results['processed_X']
        y = results['processed_y']
        
        validation_results = {
            'data_shape': X.shape if hasattr(X, 'shape') else (len(X),),
            'data_type': str(type(X)),
            'has_infinite_values': False,
            'has_nan_values': False,
            'is_valid': True
        }
        
        try:
            # Check for infinite values
            if isinstance(X, (np.ndarray, pd.DataFrame)):
                if isinstance(X, pd.DataFrame):
                    has_inf = np.isinf(X.select_dtypes(include=[np.number])).any().any()
                    has_nan = X.isnull().any().any()
                else:
                    has_inf = np.isinf(X).any() if X.dtype.kind in 'fc' else False
                    has_nan = np.isnan(X).any() if X.dtype.kind in 'fc' else False
                
                validation_results['has_infinite_values'] = bool(has_inf)
                validation_results['has_nan_values'] = bool(has_nan)
                
                if has_inf or has_nan:
                    validation_results['is_valid'] = False
                    results['processing_log'].append("WARNING: Processed data contains invalid values")
            
            # Validate target labels if present
            if y is not None:
                validation_results['target_shape'] = y.shape if hasattr(y, 'shape') else (len(y),)
                validation_results['target_type'] = str(type(y))
                
                # Check if X and y have consistent sample counts
                x_samples = X.shape[0] if hasattr(X, 'shape') else len(X)
                y_samples = y.shape[0] if hasattr(y, 'shape') else len(y)
                
                if x_samples != y_samples:
                    validation_results['is_valid'] = False
                    results['processing_log'].append(f"ERROR: Sample count mismatch - X: {x_samples}, y: {y_samples}")
            
            results['quality_report']['validation'] = validation_results
            
            if validation_results['is_valid']:
                results['processing_log'].append("Data validation passed")
            else:
                results['processing_log'].append("Data validation failed")
            
        except Exception as e:
            results['processing_log'].append(f"ERROR: Data validation failed: {str(e)}")
            validation_results['is_valid'] = False
            validation_results['validation_error'] = str(e)
            results['quality_report']['validation'] = validation_results
        
        return results


def handle_corrupted_data(data: Any, data_type: str = "unknown") -> Tuple[Any, List[str]]:
    """
    Handle corrupted or invalid data with robust error recovery.
    
    Args:
        data: Data to validate and clean
        data_type: Type of data for context in error messages
        
    Returns:
        Tuple of (cleaned_data, error_messages)
    """
    error_messages = []
    
    try:
        if data is None:
            error_messages.append(f"Data is None for type: {data_type}")
            return None, error_messages
        
        # Handle different data types
        if isinstance(data, (np.ndarray, pd.DataFrame, pd.Series)):
            return _handle_array_like_corruption(data, data_type, error_messages)
        elif isinstance(data, (list, tuple)):
            return _handle_sequence_corruption(data, data_type, error_messages)
        elif isinstance(data, dict):
            return _handle_dict_corruption(data, data_type, error_messages)
        else:
            # For other types, just validate basic properties
            return data, error_messages
            
    except Exception as e:
        error_messages.append(f"Critical error handling {data_type}: {str(e)}")
        return None, error_messages


def _handle_array_like_corruption(data: Union[np.ndarray, pd.DataFrame, pd.Series], 
                                 data_type: str, error_messages: List[str]) -> Tuple[Any, List[str]]:
    """Handle corruption in array-like data structures."""
    try:
        if isinstance(data, pd.DataFrame):
            # Check for completely empty DataFrame
            if data.empty:
                error_messages.append(f"Empty DataFrame for {data_type}")
                return data, error_messages
            
            # Check for columns with all NaN values
            all_nan_cols = data.columns[data.isnull().all()].tolist()
            if all_nan_cols:
                error_messages.append(f"Columns with all NaN values in {data_type}: {all_nan_cols}")
                data = data.drop(columns=all_nan_cols)
            
            # Check for duplicate columns
            duplicate_cols = data.columns[data.columns.duplicated()].tolist()
            if duplicate_cols:
                error_messages.append(f"Duplicate columns in {data_type}: {duplicate_cols}")
                data = data.loc[:, ~data.columns.duplicated()]
            
        elif isinstance(data, np.ndarray):
            # Check for empty array
            if data.size == 0:
                error_messages.append(f"Empty array for {data_type}")
                return data, error_messages
            
            # Check for invalid shape
            if len(data.shape) == 0:
                error_messages.append(f"Scalar array for {data_type}, converting to 1D")
                data = np.array([data])
            
            # Check for infinite values
            if np.isinf(data).any():
                error_messages.append(f"Infinite values detected in {data_type}")
                data = np.where(np.isinf(data), np.nan, data)
        
        return data, error_messages
        
    except Exception as e:
        error_messages.append(f"Error processing array-like data for {data_type}: {str(e)}")
        return data, error_messages


def _handle_sequence_corruption(data: Union[list, tuple], 
                               data_type: str, error_messages: List[str]) -> Tuple[Any, List[str]]:
    """Handle corruption in sequence data structures."""
    try:
        if len(data) == 0:
            error_messages.append(f"Empty sequence for {data_type}")
            return data, error_messages
        
        # Check for None values
        none_count = sum(1 for item in data if item is None)
        if none_count > 0:
            error_messages.append(f"Found {none_count} None values in {data_type}")
            # Filter out None values
            data = [item for item in data if item is not None]
        
        # Convert back to original type
        if isinstance(data, tuple):
            data = tuple(data)
        
        return data, error_messages
        
    except Exception as e:
        error_messages.append(f"Error processing sequence data for {data_type}: {str(e)}")
        return data, error_messages


def _handle_dict_corruption(data: dict, data_type: str, error_messages: List[str]) -> Tuple[Any, List[str]]:
    """Handle corruption in dictionary data structures."""
    try:
        if not data:
            error_messages.append(f"Empty dictionary for {data_type}")
            return data, error_messages
        
        # Check for None values
        none_keys = [k for k, v in data.items() if v is None]
        if none_keys:
            error_messages.append(f"Keys with None values in {data_type}: {none_keys}")
        
        # Check for duplicate values that might indicate corruption
        values = list(data.values())
        if len(values) != len(set(str(v) for v in values)):
            error_messages.append(f"Potential duplicate values detected in {data_type}")
        
        return data, error_messages
        
    except Exception as e:
        error_messages.append(f"Error processing dictionary data for {data_type}: {str(e)}")
        return data, error_messages