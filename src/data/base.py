"""Base classes and interfaces for data processing."""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Union
import numpy as np
from .schemas import RockfallDataPoint


class BaseDataProcessor(ABC):
    """Abstract base class for all data processors."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize the processor with configuration."""
        self.config = config or {}
        self._is_fitted = False
    
    @abstractmethod
    def fit(self, data: List[Any]) -> 'BaseDataProcessor':
        """Fit the processor to training data."""
        pass
    
    @abstractmethod
    def transform(self, data: Any) -> np.ndarray:
        """Transform input data to processed features."""
        pass
    
    def fit_transform(self, data: List[Any]) -> np.ndarray:
        """Fit the processor and transform the data."""
        return self.fit(data).transform(data)
    
    @property
    def is_fitted(self) -> bool:
        """Check if the processor has been fitted."""
        return self._is_fitted
    
    def get_config(self) -> Dict[str, Any]:
        """Get processor configuration."""
        return self.config.copy()


class BaseFeatureExtractor(ABC):
    """Abstract base class for feature extractors."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize the feature extractor."""
        self.config = config or {}
        self._feature_dim = None
    
    @abstractmethod
    def extract_features(self, data: Any) -> np.ndarray:
        """Extract features from input data."""
        pass
    
    @property
    def feature_dim(self) -> Optional[int]:
        """Get the dimensionality of extracted features."""
        return self._feature_dim
    
    def get_config(self) -> Dict[str, Any]:
        """Get extractor configuration."""
        return self.config.copy()


class BaseClassifier(ABC):
    """Abstract base class for classifiers."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize the classifier."""
        self.config = config or {}
        self._is_trained = False
    
    @abstractmethod
    def train(self, X: np.ndarray, y: np.ndarray) -> 'BaseClassifier':
        """Train the classifier on features and labels."""
        pass
    
    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions on input features."""
        pass
    
    @abstractmethod
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Get prediction probabilities."""
        pass
    
    @property
    def is_trained(self) -> bool:
        """Check if the classifier has been trained."""
        return self._is_trained
    
    def get_config(self) -> Dict[str, Any]:
        """Get classifier configuration."""
        return self.config.copy()


class DataValidator:
    """Utility class for data validation."""
    
    @staticmethod
    def validate_rockfall_datapoint(datapoint: RockfallDataPoint) -> bool:
        """Validate a RockfallDataPoint instance."""
        if not isinstance(datapoint, RockfallDataPoint):
            return False
        
        # Check required fields
        if not datapoint.timestamp or not datapoint.location:
            return False
        
        # Validate at least one data modality is present
        modalities = [
            datapoint.imagery,
            datapoint.dem_data,
            datapoint.sensor_readings,
            datapoint.environmental,
            datapoint.seismic
        ]
        
        return any(modality is not None for modality in modalities)
    
    @staticmethod
    def validate_array_shape(array: np.ndarray, expected_shape: tuple) -> bool:
        """Validate numpy array shape."""
        return array.shape == expected_shape
    
    @staticmethod
    def validate_feature_vector(features: np.ndarray, min_dim: int = 1) -> bool:
        """Validate feature vector dimensions."""
        return len(features.shape) >= 1 and features.shape[-1] >= min_dim