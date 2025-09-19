"""Tabular Feature Extractor for structured environmental and safety data."""

import numpy as np
import pandas as pd
from typing import Dict, Any, Optional, List, Union, Tuple
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, LabelEncoder, OneHotEncoder
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif, RFE
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, TransformerMixin
import joblib
import os
from pathlib import Path

from ...data.base import BaseFeatureExtractor
from ...data.schemas import EnvironmentalData, RockfallDataPoint


class ColumnLabelEncoder(BaseEstimator, TransformerMixin):
    """Label encoder that works with ColumnTransformer by handling 2D input."""
    
    def __init__(self):
        self.label_encoder = LabelEncoder()
    
    def fit(self, X, y=None):
        # Convert 2D to 1D for LabelEncoder
        X_1d = X.ravel() if X.ndim > 1 else X
        self.label_encoder.fit(X_1d)
        return self
    
    def transform(self, X):
        # Convert 2D to 1D, transform, then back to 2D
        X_1d = X.ravel() if X.ndim > 1 else X
        transformed = self.label_encoder.transform(X_1d)
        return transformed.reshape(-1, 1)
    
    def fit_transform(self, X, y=None):
        return self.fit(X, y).transform(X)


class TabularFeatureExtractor(BaseFeatureExtractor):
    """Feature extractor for structured tabular data including environmental and safety metrics."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize tabular feature extractor.
        
        Args:
            config: Configuration dictionary with parameters:
                - scaling_method: 'standard', 'minmax', 'robust' (default: 'standard')
                - encoding_method: 'onehot', 'label' (default: 'onehot')
                - feature_selection_method: 'kbest', 'rfe', 'mutual_info', None (default: 'kbest')
                - n_features: Number of features to select (default: 50)
                - imputation_strategy: 'mean', 'median', 'mode', 'knn' (default: 'median')
                - handle_outliers: Whether to handle outliers (default: True)
                - outlier_method: 'iqr', 'zscore' (default: 'iqr')
                - outlier_threshold: Threshold for outlier detection (default: 3.0)
                - domain_features: Whether to include domain-specific features (default: True)
        """
        super().__init__(config)
        
        # Default configuration
        default_config = {
            'scaling_method': 'standard',
            'encoding_method': 'onehot',
            'feature_selection_method': 'kbest',
            'n_features': 50,
            'imputation_strategy': 'median',
            'handle_outliers': True,
            'outlier_method': 'iqr',
            'outlier_threshold': 3.0,
            'domain_features': True
        }
        self.config = {**default_config, **self.config}
        
        # Initialize components
        self.scaler = None
        self.encoder = None
        self.feature_selector = None
        self.imputer = None
        self.preprocessing_pipeline = None
        self._feature_names = []
        self._categorical_features = []
        self._numerical_features = []
        self._is_fitted = False
    
    def _create_domain_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create domain-specific features for mining safety and environmental data.
        
        Args:
            df: Input dataframe with environmental and safety data
            
        Returns:
            DataFrame with additional domain-specific features
        """
        df_enhanced = df.copy()
        
        # Environmental risk indicators
        if 'rainfall' in df.columns:
            # Cumulative rainfall features
            df_enhanced['rainfall_cumulative_24h'] = df['rainfall'].rolling(window=24, min_periods=1).sum()
            df_enhanced['rainfall_cumulative_7d'] = df['rainfall'].rolling(window=168, min_periods=1).sum()
            
            # Rainfall intensity categories
            df_enhanced['rainfall_intensity'] = pd.cut(
                df['rainfall'], 
                bins=[0, 5, 15, 30, float('inf')], 
                labels=['light', 'moderate', 'heavy', 'extreme']
            )
            
            # Antecedent precipitation index (API)
            if len(df) > 1:
                api = np.zeros(len(df))
                k = 0.9  # decay constant
                for i in range(1, len(df)):
                    api[i] = k * api[i-1] + df['rainfall'].iloc[i]
                df_enhanced['antecedent_precipitation_index'] = api
        
        if 'temperature' in df.columns:
            # Temperature-based features
            df_enhanced['freeze_thaw_cycles'] = (
                (df['temperature'].shift(1) <= 0) & (df['temperature'] > 0)
            ).astype(int)
            
            # Temperature variability
            df_enhanced['temp_daily_range'] = df['temperature'].rolling(window=24).max() - df['temperature'].rolling(window=24).min()
        
        if 'vibrations' in df.columns:
            # Vibration-based features
            df_enhanced['vibration_peak'] = df['vibrations'].rolling(window=24).max()
            df_enhanced['vibration_rms'] = np.sqrt(df['vibrations'].rolling(window=24).apply(lambda x: (x**2).mean()))
            
            # Vibration threshold exceedances
            vibration_threshold = df['vibrations'].quantile(0.95)
            df_enhanced['vibration_exceedance'] = (df['vibrations'] > vibration_threshold).astype(int)
        
        if 'wind_speed' in df.columns:
            # Wind-based features
            df_enhanced['wind_gust_factor'] = df['wind_speed'].rolling(window=6).max() / (df['wind_speed'].rolling(window=6).mean() + 1e-6)
        
        # Combined environmental stress indicators
        if all(col in df.columns for col in ['rainfall', 'temperature', 'vibrations']):
            # Normalize each factor to 0-1 scale for combination
            rainfall_norm = (df['rainfall'] - df['rainfall'].min()) / (df['rainfall'].max() - df['rainfall'].min() + 1e-6)
            temp_stress = np.abs(df['temperature']) / (np.abs(df['temperature']).max() + 1e-6)
            vibration_norm = (df['vibrations'] - df['vibrations'].min()) / (df['vibrations'].max() - df['vibrations'].min() + 1e-6)
            
            df_enhanced['environmental_stress_index'] = (rainfall_norm + temp_stress + vibration_norm) / 3
        
        # Temporal features
        if 'timestamp' in df.columns:
            df_enhanced['hour'] = pd.to_datetime(df['timestamp']).dt.hour
            df_enhanced['day_of_week'] = pd.to_datetime(df['timestamp']).dt.dayofweek
            df_enhanced['month'] = pd.to_datetime(df['timestamp']).dt.month
            df_enhanced['season'] = pd.to_datetime(df['timestamp']).dt.month % 12 // 3 + 1
        
        return df_enhanced
    
    def _detect_outliers(self, df: pd.DataFrame, method: str = 'iqr', threshold: float = 3.0) -> pd.DataFrame:
        """Detect and handle outliers in numerical data.
        
        Args:
            df: Input dataframe
            method: Outlier detection method ('iqr' or 'zscore')
            threshold: Threshold for outlier detection
            
        Returns:
            DataFrame with outliers handled
        """
        df_clean = df.copy()
        numerical_cols = df.select_dtypes(include=[np.number]).columns
        
        for col in numerical_cols:
            if method == 'iqr':
                Q1 = df[col].quantile(0.25)
                Q3 = df[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                
                # Cap outliers to bounds
                df_clean[col] = df_clean[col].clip(lower=lower_bound, upper=upper_bound)
                
            elif method == 'zscore':
                z_scores = np.abs((df[col] - df[col].mean()) / df[col].std())
                outlier_mask = z_scores > threshold
                
                # Replace outliers with median
                df_clean.loc[outlier_mask, col] = df[col].median()
        
        return df_clean
    
    def _identify_feature_types(self, df: pd.DataFrame) -> Tuple[List[str], List[str]]:
        """Identify categorical and numerical features in the dataframe.
        
        Args:
            df: Input dataframe
            
        Returns:
            Tuple of (categorical_features, numerical_features)
        """
        categorical_features = []
        numerical_features = []
        
        for col in df.columns:
            if df[col].dtype == 'object' or df[col].dtype.name == 'category':
                categorical_features.append(col)
            elif pd.api.types.is_numeric_dtype(df[col]):
                numerical_features.append(col)
        
        return categorical_features, numerical_features
    
    def _create_preprocessing_pipeline(self, categorical_features: List[str], numerical_features: List[str]) -> Pipeline:
        """Create preprocessing pipeline for categorical and numerical features.
        
        Args:
            categorical_features: List of categorical feature names
            numerical_features: List of numerical feature names
            
        Returns:
            Sklearn preprocessing pipeline
        """
        # Numerical preprocessing
        numerical_steps = []
        
        # Imputation
        if self.config['imputation_strategy'] == 'knn':
            numerical_steps.append(('imputer', KNNImputer(n_neighbors=5)))
        else:
            numerical_steps.append(('imputer', SimpleImputer(strategy=self.config['imputation_strategy'])))
        
        # Scaling
        if self.config['scaling_method'] == 'standard':
            numerical_steps.append(('scaler', StandardScaler()))
        elif self.config['scaling_method'] == 'minmax':
            numerical_steps.append(('scaler', MinMaxScaler()))
        elif self.config['scaling_method'] == 'robust':
            numerical_steps.append(('scaler', RobustScaler()))
        
        numerical_pipeline = Pipeline(numerical_steps)
        
        # Categorical preprocessing
        categorical_steps = [('imputer', SimpleImputer(strategy='most_frequent'))]
        
        if self.config['encoding_method'] == 'onehot':
            categorical_steps.append(('encoder', OneHotEncoder(drop='first', sparse_output=False, handle_unknown='ignore')))
        elif self.config['encoding_method'] == 'label':
            categorical_steps.append(('encoder', ColumnLabelEncoder()))
        
        categorical_pipeline = Pipeline(categorical_steps)
        
        # Combine pipelines
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', numerical_pipeline, numerical_features),
                ('cat', categorical_pipeline, categorical_features)
            ],
            remainder='drop'
        )
        
        return Pipeline([('preprocessor', preprocessor)])
    
    def _create_feature_selector(self, method: str, n_features: int) -> Optional[Any]:
        """Create feature selection object.
        
        Args:
            method: Feature selection method
            n_features: Number of features to select
            
        Returns:
            Feature selector object or None
        """
        if method == 'kbest':
            return SelectKBest(score_func=f_classif, k=n_features)
        elif method == 'mutual_info':
            return SelectKBest(score_func=mutual_info_classif, k=n_features)
        elif method == 'rfe':
            return RFE(estimator=RandomForestClassifier(n_estimators=50, random_state=42), n_features_to_select=n_features)
        else:
            return None
    
    def _convert_to_dataframe(self, data: List[Union[RockfallDataPoint, Dict[str, Any]]]) -> pd.DataFrame:
        """Convert input data to pandas DataFrame.
        
        Args:
            data: List of RockfallDataPoint objects or dictionaries
            
        Returns:
            Pandas DataFrame with tabular features
        """
        rows = []
        
        for item in data:
            row = {}
            
            if isinstance(item, RockfallDataPoint):
                # Extract environmental data
                if item.environmental:
                    env_data = item.environmental
                    if env_data.rainfall is not None:
                        row['rainfall'] = env_data.rainfall
                    if env_data.temperature is not None:
                        row['temperature'] = env_data.temperature
                    if env_data.vibrations is not None:
                        row['vibrations'] = env_data.vibrations
                    if env_data.wind_speed is not None:
                        row['wind_speed'] = env_data.wind_speed
                
                # Extract location data
                if item.location:
                    row['latitude'] = item.location.latitude
                    row['longitude'] = item.location.longitude
                    if item.location.elevation is not None:
                        row['elevation'] = item.location.elevation
                
                # Add timestamp
                if item.timestamp:
                    row['timestamp'] = item.timestamp
                
            elif isinstance(item, dict):
                # Handle dictionary input
                row.update(item)
            
            if row:  # Only add non-empty rows
                rows.append(row)
        
        return pd.DataFrame(rows)
    
    def _create_feature_names(self):
        """Create feature names after preprocessing."""
        self._feature_names = []
        
        # Get feature names from preprocessing pipeline
        if hasattr(self.preprocessing_pipeline, 'named_steps'):
            preprocessor = self.preprocessing_pipeline.named_steps['preprocessor']
            
            # Numerical feature names
            if self._numerical_features:
                self._feature_names.extend(self._numerical_features)
            
            # Categorical feature names (after encoding)
            if self._categorical_features and hasattr(preprocessor, 'named_transformers_'):
                cat_transformer = preprocessor.named_transformers_.get('cat')
                if cat_transformer and hasattr(cat_transformer, 'named_steps'):
                    encoder = cat_transformer.named_steps.get('encoder')
                    if hasattr(encoder, 'get_feature_names_out'):
                        cat_names = encoder.get_feature_names_out(self._categorical_features)
                        self._feature_names.extend(cat_names)
                    else:
                        # Fallback for LabelEncoder or other encoders
                        self._feature_names.extend(self._categorical_features)
    
    def fit(self, data: List[Union[RockfallDataPoint, Dict[str, Any]]], y: Optional[np.ndarray] = None) -> 'TabularFeatureExtractor':
        """Fit the tabular feature extractor to training data.
        
        Args:
            data: List of RockfallDataPoint objects or dictionaries with tabular data
            y: Optional target labels for supervised feature selection
            
        Returns:
            Self for method chaining
        """
        # Convert data to DataFrame
        df = self._convert_to_dataframe(data)
        
        # Add domain-specific features if enabled
        if self.config['domain_features']:
            df = self._create_domain_features(df)
        
        # Handle outliers if enabled
        if self.config['handle_outliers']:
            df = self._detect_outliers(df, self.config['outlier_method'], self.config['outlier_threshold'])
        
        # Identify feature types
        self._categorical_features, self._numerical_features = self._identify_feature_types(df)
        
        # Create and fit preprocessing pipeline
        self.preprocessing_pipeline = self._create_preprocessing_pipeline(
            self._categorical_features, self._numerical_features
        )
        
        # Fit preprocessing pipeline
        X_processed = self.preprocessing_pipeline.fit_transform(df)
        
        # Create feature names after preprocessing
        self._create_feature_names()
        
        # Fit feature selector if specified
        if self.config['feature_selection_method'] and y is not None:
            self.feature_selector = self._create_feature_selector(
                self.config['feature_selection_method'], 
                min(self.config['n_features'], X_processed.shape[1])
            )
            X_processed = self.feature_selector.fit_transform(X_processed, y)
            
            # Update feature names after selection
            if hasattr(self.feature_selector, 'get_support'):
                selected_mask = self.feature_selector.get_support()
                self._feature_names = [name for name, selected in zip(self._feature_names, selected_mask) if selected]
        
        self._feature_dim = X_processed.shape[1]
        self._is_fitted = True
        
        return self
    
    def transform(self, data: Union[RockfallDataPoint, Dict[str, Any], List[Union[RockfallDataPoint, Dict[str, Any]]]]) -> np.ndarray:
        """Transform input data to feature vectors.
        
        Args:
            data: Single data point or list of data points
            
        Returns:
            Feature array of shape (n_samples, n_features)
        """
        if not self._is_fitted:
            raise ValueError("TabularFeatureExtractor must be fitted before transform")
        
        # Handle single data point
        if not isinstance(data, list):
            data = [data]
        
        # Convert to DataFrame
        df = self._convert_to_dataframe(data)
        
        # Add domain-specific features if enabled
        if self.config['domain_features']:
            df = self._create_domain_features(df)
        
        # Handle outliers if enabled
        if self.config['handle_outliers']:
            df = self._detect_outliers(df, self.config['outlier_method'], self.config['outlier_threshold'])
        
        # Apply preprocessing pipeline
        X_processed = self.preprocessing_pipeline.transform(df)
        
        # Apply feature selection if fitted
        if self.feature_selector is not None:
            X_processed = self.feature_selector.transform(X_processed)
        
        return X_processed
    
    def extract_features(self, data: Union[RockfallDataPoint, Dict[str, Any], List[Union[RockfallDataPoint, Dict[str, Any]]]]) -> np.ndarray:
        """Extract features from tabular data (alias for transform).
        
        Args:
            data: Input data to extract features from
            
        Returns:
            Feature array of shape (n_samples, n_features)
        """
        return self.transform(data)
    
    def get_feature_names(self) -> List[str]:
        """Get names of extracted features.
        
        Returns:
            List of feature names
        """
        return self._feature_names.copy()
    
    def get_feature_importance(self, method: str = 'selector') -> Optional[np.ndarray]:
        """Get feature importance scores.
        
        Args:
            method: Method to get importance ('selector' or 'random_forest')
            
        Returns:
            Feature importance scores or None if not available
        """
        if method == 'selector' and self.feature_selector is not None:
            if hasattr(self.feature_selector, 'scores_'):
                return self.feature_selector.scores_
            elif hasattr(self.feature_selector, 'ranking_'):
                # For RFE, convert ranking to importance (lower rank = higher importance)
                return 1.0 / self.feature_selector.ranking_
        
        return None
    
    def save(self, filepath: str):
        """Save the fitted extractor to disk.
        
        Args:
            filepath: Path to save the extractor
        """
        if not self._is_fitted:
            raise ValueError("Cannot save unfitted extractor")
        
        save_data = {
            'config': self.config,
            'preprocessing_pipeline': self.preprocessing_pipeline,
            'feature_selector': self.feature_selector,
            'feature_names': self._feature_names,
            'categorical_features': self._categorical_features,
            'numerical_features': self._numerical_features,
            'feature_dim': self._feature_dim,
            'is_fitted': self._is_fitted
        }
        
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        joblib.dump(save_data, filepath)
    
    @classmethod
    def load(cls, filepath: str) -> 'TabularFeatureExtractor':
        """Load a fitted extractor from disk.
        
        Args:
            filepath: Path to load the extractor from
            
        Returns:
            Loaded TabularFeatureExtractor instance
        """
        save_data = joblib.load(filepath)
        
        extractor = cls(save_data['config'])
        extractor.preprocessing_pipeline = save_data['preprocessing_pipeline']
        extractor.feature_selector = save_data['feature_selector']
        extractor._feature_names = save_data['feature_names']
        extractor._categorical_features = save_data['categorical_features']
        extractor._numerical_features = save_data['numerical_features']
        extractor._feature_dim = save_data['feature_dim']
        extractor._is_fitted = save_data['is_fitted']
        
        return extractor