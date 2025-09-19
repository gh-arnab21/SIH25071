"""
Prediction Interface for Rockfall Prediction System.

This module provides a high-level interface for making predictions using trained
models, handling preprocessing, batch processing, and result formatting.
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Union, Optional, Any
from pathlib import Path
import logging
import warnings
from datetime import datetime

# Local imports
from ..data.schema import RockfallDataPoint, RockfallPrediction, RiskLevel
from ..data.validation import validate_rockfall_datapoint, ValidationError
from ..data.quality import RobustDataProcessor
from ..models.persistence import ModelPersistence, ModelPersistenceError
from ..models.classifiers.ensemble_classifier import EnsembleClassifier


logger = logging.getLogger(__name__)


class PredictionError(Exception):
    """Exception raised during prediction process."""
    pass


class PredictionInterface:
    """
    High-level interface for making rockfall risk predictions.
    
    This class provides a unified interface for loading models, preprocessing data,
    and making predictions with confidence scores and uncertainty estimates.
    """
    
    def __init__(self, model_path: Optional[str] = None, 
                 preprocessor_path: Optional[str] = None):
        """
        Initialize prediction interface.
        
        Args:
            model_path: Path to saved model file
            preprocessor_path: Path to saved preprocessor file
        """
        self.model_path = model_path
        self.preprocessor_path = preprocessor_path
        self.model = None
        self.preprocessor = None
        self.feature_names = None
        self.is_loaded = False
        
        # Initialize components if paths provided
        if model_path:
            self.load_model(model_path)
        if preprocessor_path:
            self.load_preprocessor(preprocessor_path)
    
    def load_model(self, model_path: str) -> None:
        """
        Load a trained model from disk.
        
        Args:
            model_path: Path to the saved model file
            
        Raises:
            PredictionError: If model loading fails
        """
        try:
            logger.info(f"Loading model from {model_path}")
            
            # Try direct joblib loading first (for simple demos)
            import joblib
            try:
                model_data = joblib.load(model_path)
                if isinstance(model_data, dict) and 'model' in model_data:
                    # Handle dictionary format
                    self.model = model_data['model']
                    self.feature_names = model_data.get('feature_names', [])
                else:
                    # Handle direct model format
                    self.model = model_data
                    self.feature_names = []
            except:
                # Fall back to ModelPersistence
                persistence = ModelPersistence()
                model_data = persistence.load_model(model_path)
                self.model = model_data['model']
                self.feature_names = model_data.get('feature_names', [])
            
            # Validate loaded model
            if not hasattr(self.model, 'predict'):
                raise PredictionError("Loaded model does not have predict method")
            
            self.model_path = model_path
            self.is_loaded = True
            
            logger.info("Model loaded successfully")
            
        except Exception as e:
            raise PredictionError(f"Failed to load model: {str(e)}")
    
    def load_preprocessor(self, preprocessor_path: str) -> None:
        """
        Load a trained preprocessor from disk.
        
        Args:
            preprocessor_path: Path to the saved preprocessor file
            
        Raises:
            PredictionError: If preprocessor loading fails
        """
        try:
            logger.info(f"Loading preprocessor from {preprocessor_path}")
            
            persistence = ModelPersistence()
            preprocessor_data = persistence.load_preprocessing_pipeline(preprocessor_path)
            
            self.preprocessor = preprocessor_data['pipeline']
            self.preprocessor_path = preprocessor_path
            
            logger.info("Preprocessor loaded successfully")
            
        except Exception as e:
            raise PredictionError(f"Failed to load preprocessor: {str(e)}")
    
    def _validate_inputs(self, X: Union[np.ndarray, pd.DataFrame, List[RockfallDataPoint]]) -> np.ndarray:
        """
        Validate and convert inputs to proper format.
        
        Args:
            X: Input data in various formats
            
        Returns:
            Validated numpy array
            
        Raises:
            PredictionError: If input validation fails
        """
        if not self.is_loaded:
            raise PredictionError("Model not loaded. Call load_model() first.")
        
        try:
            # Handle different input types
            if isinstance(X, list) and len(X) > 0 and isinstance(X[0], RockfallDataPoint):
                # Convert RockfallDataPoint objects to feature matrix
                X = self._extract_features_from_datapoints(X)
            elif isinstance(X, pd.DataFrame):
                X = X.values
            elif isinstance(X, np.ndarray):
                pass  # Already in correct format
            else:
                raise PredictionError(f"Unsupported input type: {type(X)}")
            
            # Ensure 2D array
            if X.ndim == 1:
                X = X.reshape(1, -1)
            
            # Validate dimensions
            expected_features = len(self.feature_names) if self.feature_names else None
            if expected_features and X.shape[1] != expected_features:
                raise PredictionError(
                    f"Input has {X.shape[1]} features, expected {expected_features}"
                )
            
            return X
            
        except Exception as e:
            raise PredictionError(f"Input validation failed: {str(e)}")
    
    def _extract_features_from_datapoints(self, datapoints: List[RockfallDataPoint]) -> np.ndarray:
        """
        Extract feature matrix from RockfallDataPoint objects.
        
        Args:
            datapoints: List of RockfallDataPoint objects
            
        Returns:
            Feature matrix as numpy array
            
        Note:
            This is a simplified feature extraction. In practice, this would
            involve complex processing of imagery, sensor data, etc.
        """
        features = []
        
        for dp in datapoints:
            # Validate data point
            try:
                validate_rockfall_datapoint(dp)
            except ValidationError as e:
                logger.warning(f"Data point validation warning: {e}")
            
            # Extract features (simplified example)
            feature_vector = []
            
            # Location features
            feature_vector.extend([dp.location.latitude, dp.location.longitude])
            if dp.location.elevation is not None:
                feature_vector.append(dp.location.elevation)
            else:
                feature_vector.append(0.0)  # Default elevation
            
            # Environmental features
            if dp.environmental:
                env = dp.environmental
                feature_vector.extend([
                    env.rainfall or 0.0,
                    env.temperature or 0.0,
                    env.vibrations or 0.0,
                    env.wind_speed or 0.0,
                    env.humidity or 0.0
                ])
            else:
                feature_vector.extend([0.0] * 5)  # Default environmental values
            
            # Sensor features (simplified)
            if dp.sensor_readings:
                # Use mean of time series data
                displacement = np.mean(dp.sensor_readings.displacement.values) if dp.sensor_readings.displacement else 0.0
                strain = np.mean(dp.sensor_readings.strain.values) if dp.sensor_readings.strain else 0.0
                pore_pressure = np.mean(dp.sensor_readings.pore_pressure.values) if dp.sensor_readings.pore_pressure else 0.0
                feature_vector.extend([displacement, strain, pore_pressure])
            else:
                feature_vector.extend([0.0] * 3)
            
            # Pad or truncate to expected length
            if self.feature_names:
                expected_length = len(self.feature_names)
                if len(feature_vector) < expected_length:
                    feature_vector.extend([0.0] * (expected_length - len(feature_vector)))
                elif len(feature_vector) > expected_length:
                    feature_vector = feature_vector[:expected_length]
            
            features.append(feature_vector)
        
        return np.array(features)
    
    def _preprocess_data(self, X: np.ndarray) -> np.ndarray:
        """
        Apply preprocessing pipeline to input data.
        
        Args:
            X: Raw feature matrix
            
        Returns:
            Preprocessed feature matrix
        """
        try:
            if self.preprocessor is not None:
                # Use loaded preprocessor
                if hasattr(self.preprocessor, 'transform'):
                    X = self.preprocessor.transform(X)
                elif hasattr(self.preprocessor, 'process_data'):
                    processed_result = self.preprocessor.process_data(X)
                    X = processed_result.get('processed_data', X)
                else:
                    logger.warning("Preprocessor has no transform or process method")
            else:
                # Use default robust preprocessing
                logger.info("Using default robust preprocessing")
                default_processor = RobustDataProcessor()
                processed_result = default_processor.process_data(X)
                X = processed_result.get('processed_data', X)
            
            return X
            
        except Exception as e:
            logger.error(f"Preprocessing failed: {e}")
            return X  # Return unprocessed data as fallback
    
    def predict(self, X: Union[np.ndarray, pd.DataFrame, List[RockfallDataPoint]]) -> List[RockfallPrediction]:
        """
        Make predictions for input data.
        
        Args:
            X: Input data (feature matrix, DataFrame, or list of RockfallDataPoint objects)
            
        Returns:
            List of RockfallPrediction objects with confidence scores
            
        Raises:
            PredictionError: If prediction fails
        """
        try:
            # Validate and convert inputs
            X_validated = self._validate_inputs(X)
            
            # Apply preprocessing
            X_processed = self._preprocess_data(X_validated)
            
            # Make predictions
            if isinstance(self.model, EnsembleClassifier):
                # Use ensemble classifier's predict_with_confidence method
                predictions = self.model.predict_with_confidence(X_processed)
            else:
                # Handle other model types
                predictions = self._predict_with_generic_model(X_processed)
            
            logger.info(f"Generated {len(predictions)} predictions")
            return predictions
            
        except Exception as e:
            raise PredictionError(f"Prediction failed: {str(e)}")
    
    def _predict_with_generic_model(self, X: np.ndarray) -> List[RockfallPrediction]:
        """
        Make predictions with generic model (non-ensemble).
        
        Args:
            X: Preprocessed feature matrix
            
        Returns:
            List of RockfallPrediction objects
        """
        # Get predictions
        y_pred = self.model.predict(X)
        
        # Get probabilities if available
        if hasattr(self.model, 'predict_proba'):
            y_proba = self.model.predict_proba(X)
        else:
            # Create dummy probabilities
            y_proba = np.zeros((len(y_pred), 3))
            for i, pred in enumerate(y_pred):
                y_proba[i, pred] = 0.8  # Default confidence
        
        predictions = []
        for i in range(len(y_pred)):
            risk_level = RiskLevel(y_pred[i])
            confidence = float(y_proba[i, y_pred[i]])
            
            # Calculate simple uncertainty (1 - confidence)
            uncertainty = 1.0 - confidence
            
            # Basic feature importance (placeholder)
            contributing_factors = {}
            if self.feature_names:
                # Assign equal importance to all features (placeholder)
                importance = 1.0 / len(self.feature_names)
                contributing_factors = {name: importance for name in self.feature_names}
            
            prediction = RockfallPrediction(
                risk_level=risk_level,
                confidence_score=confidence,
                contributing_factors=contributing_factors,
                uncertainty_estimate=uncertainty,
                model_version="generic_v1.0"
            )
            
            predictions.append(prediction)
        
        return predictions
    
    def predict_single(self, X: Union[np.ndarray, pd.DataFrame, RockfallDataPoint]) -> RockfallPrediction:
        """
        Make prediction for a single data point.
        
        Args:
            X: Single data point
            
        Returns:
            Single RockfallPrediction object
        """
        if isinstance(X, RockfallDataPoint):
            X = [X]
        elif isinstance(X, (np.ndarray, pd.DataFrame)):
            if isinstance(X, np.ndarray) and X.ndim == 1:
                X = X.reshape(1, -1)
            elif isinstance(X, pd.DataFrame) and len(X) > 1:
                X = X.iloc[0:1]  # Take first row
        
        predictions = self.predict(X)
        return predictions[0]
    
    def predict_batch(self, X: Union[np.ndarray, pd.DataFrame, List[RockfallDataPoint]], 
                     batch_size: int = 32, progress_callback: Optional[callable] = None) -> List[RockfallPrediction]:
        """
        Make predictions for large datasets in batches.
        
        Args:
            X: Input data
            batch_size: Number of samples per batch
            progress_callback: Optional callback function for progress updates
            
        Returns:
            List of RockfallPrediction objects
        """
        # Convert to proper format
        X_validated = self._validate_inputs(X)
        
        predictions = []
        n_samples = X_validated.shape[0]
        n_batches = (n_samples + batch_size - 1) // batch_size
        
        logger.info(f"Processing {n_samples} samples in {n_batches} batches")
        
        for i in range(0, n_samples, batch_size):
            batch_start = i
            batch_end = min(i + batch_size, n_samples)
            batch_X = X_validated[batch_start:batch_end]
            
            # Make predictions for batch
            batch_predictions = self.predict(batch_X)
            predictions.extend(batch_predictions)
            
            # Update progress
            if progress_callback:
                progress = (batch_end / n_samples) * 100
                progress_callback(progress, batch_end, n_samples)
            
            if i % (batch_size * 10) == 0:  # Log every 10 batches
                logger.info(f"Processed {batch_end}/{n_samples} samples")
        
        return predictions
    
    def get_prediction_summary(self, predictions: List[RockfallPrediction]) -> Dict[str, Any]:
        """
        Generate summary statistics for a list of predictions.
        
        Args:
            predictions: List of RockfallPrediction objects
            
        Returns:
            Dictionary with summary statistics
        """
        if not predictions:
            return {"error": "No predictions provided"}
        
        # Count risk levels
        risk_counts = {level.name: 0 for level in RiskLevel}
        confidence_scores = []
        uncertainty_estimates = []
        
        for pred in predictions:
            risk_counts[pred.risk_level.name] += 1
            confidence_scores.append(pred.confidence_score)
            if pred.uncertainty_estimate is not None:
                uncertainty_estimates.append(pred.uncertainty_estimate)
        
        # Calculate statistics
        summary = {
            "total_predictions": len(predictions),
            "risk_distribution": risk_counts,
            "risk_percentages": {
                level: (count / len(predictions)) * 100 
                for level, count in risk_counts.items()
            },
            "confidence_stats": {
                "mean": np.mean(confidence_scores),
                "std": np.std(confidence_scores),
                "min": np.min(confidence_scores),
                "max": np.max(confidence_scores)
            }
        }
        
        if uncertainty_estimates:
            summary["uncertainty_stats"] = {
                "mean": np.mean(uncertainty_estimates),
                "std": np.std(uncertainty_estimates),
                "min": np.min(uncertainty_estimates),
                "max": np.max(uncertainty_estimates)
            }
        
        return summary
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        Get information about the loaded model.
        
        Returns:
            Dictionary with model information
        """
        if not self.is_loaded:
            return {"error": "No model loaded"}
        
        info = {
            "model_path": self.model_path,
            "preprocessor_path": self.preprocessor_path,
            "model_type": type(self.model).__name__,
            "feature_count": len(self.feature_names) if self.feature_names else "unknown",
            "feature_names": self.feature_names
        }
        
        # Add model-specific information
        if isinstance(self.model, EnsembleClassifier):
            info["model_details"] = {
                "is_ensemble": True,
                "is_fitted": getattr(self.model, 'is_fitted', False),
                "ensemble_method": getattr(self.model.config, 'ensemble_method', 'unknown')
            }
        
        return info