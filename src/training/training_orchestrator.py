"""
Training Orchestrator for Rockfall Prediction System.

This module coordinates the training of all model components including
data preprocessing, feature extraction, fusion, and ensemble classification.
"""

import logging
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union, Any
from dataclasses import dataclass, field
from datetime import datetime
import yaml
import json

# Local imports
from ..data.schema import RockfallDataPoint, RiskLevel
from ..data.processors import (
    ImagePreprocessor, TerrainProcessor, SensorDataProcessor,
    EnvironmentalProcessor, SeismicProcessor
)
from ..models.extractors import CNNFeatureExtractor, LSTMTemporalExtractor, TabularFeatureExtractor
from ..models.fusion import MultiModalFusion
from ..models.classifiers import EnsembleClassifier, EnsembleConfig
from .data_loader import DataLoader, BatchProcessor
from .cross_validation import CrossValidator, HyperparameterOptimizer
from .checkpointing import ModelCheckpoint, EarlyStopping

logger = logging.getLogger(__name__)


@dataclass
class TrainingConfig:
    """Configuration for training pipeline."""
    
    # Data configuration
    data_dir: str = "data/processed"
    train_split: float = 0.7
    val_split: float = 0.15
    test_split: float = 0.15
    batch_size: int = 32
    shuffle: bool = True
    random_state: int = 42
    
    # Model configuration
    ensemble_config: Optional[EnsembleConfig] = None
    
    # Training configuration
    max_epochs: int = 100
    early_stopping_patience: int = 10
    checkpoint_frequency: int = 5
    validation_frequency: int = 1
    
    # Cross-validation configuration
    cv_folds: int = 5
    hyperparameter_optimization: bool = True
    n_trials: int = 50
    
    # Output configuration
    output_dir: str = "models/saved"
    experiment_name: str = "rockfall_prediction"
    save_intermediate_models: bool = True
    
    # Feature extraction configuration
    use_image_features: bool = True
    use_terrain_features: bool = True
    use_sensor_features: bool = True
    use_environmental_features: bool = True
    use_seismic_features: bool = True
    
    def __post_init__(self):
        """Validate configuration and set defaults."""
        if abs(self.train_split + self.val_split + self.test_split - 1.0) > 1e-6:
            raise ValueError("Train, validation, and test splits must sum to 1.0")
        
        if self.ensemble_config is None:
            self.ensemble_config = EnsembleConfig()
    
    @classmethod
    def from_yaml(cls, config_path: Union[str, Path]) -> 'TrainingConfig':
        """Load configuration from YAML file."""
        with open(config_path, 'r') as f:
            config_dict = yaml.safe_load(f)
        
        # Extract training-specific configuration
        training_config = config_dict.get('training', {})
        model_config = config_dict.get('model', {})
        
        # Create ensemble config from model configuration
        ensemble_config = EnsembleConfig(
            rf_n_estimators=model_config.get('random_forest', {}).get('n_estimators', 100),
            rf_max_depth=model_config.get('random_forest', {}).get('max_depth'),
            rf_random_state=model_config.get('random_forest', {}).get('random_state', 42),
            xgb_n_estimators=model_config.get('xgboost', {}).get('n_estimators', 100),
            xgb_max_depth=model_config.get('xgboost', {}).get('max_depth', 6),
            xgb_learning_rate=model_config.get('xgboost', {}).get('learning_rate', 0.1),
            nn_hidden_layers=model_config.get('neural_network', {}).get('hidden_layers', [256, 128, 64]),
            nn_dropout_rate=model_config.get('neural_network', {}).get('dropout', 0.3),
            nn_learning_rate=model_config.get('neural_network', {}).get('learning_rate', 0.001),
            nn_batch_size=training_config.get('batch_size', 32),
            nn_epochs=training_config.get('max_epochs', 100),
            nn_early_stopping_patience=training_config.get('early_stopping_patience', 10),
            cv_folds=training_config.get('cross_validation_folds', 5)
        )
        
        return cls(
            train_split=1.0 - training_config.get('validation_split', 0.15) - training_config.get('test_split', 0.15),
            val_split=training_config.get('validation_split', 0.15),
            test_split=training_config.get('test_split', 0.15),
            batch_size=training_config.get('batch_size', 32),
            max_epochs=training_config.get('max_epochs', 100),
            early_stopping_patience=training_config.get('early_stopping_patience', 10),
            cv_folds=training_config.get('cross_validation_folds', 5),
            ensemble_config=ensemble_config
        )


class TrainingOrchestrator:
    """
    Orchestrates the complete training pipeline for the rockfall prediction system.
    
    Coordinates data loading, preprocessing, feature extraction, model training,
    validation, and checkpointing.
    """
    
    def __init__(self, config: TrainingConfig):
        """Initialize training orchestrator."""
        self.config = config
        self.experiment_dir = Path(config.output_dir) / config.experiment_name
        self.experiment_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize components
        self.data_loader = DataLoader(config.data_dir)
        self.batch_processor = BatchProcessor(config.batch_size, config.shuffle, config.random_state)
        
        # Initialize processors
        self.image_processor = ImagePreprocessor() if config.use_image_features else None
        self.terrain_processor = TerrainProcessor() if config.use_terrain_features else None
        self.sensor_processor = SensorDataProcessor() if config.use_sensor_features else None
        self.environmental_processor = EnvironmentalProcessor() if config.use_environmental_features else None
        self.seismic_processor = SeismicProcessor() if config.use_seismic_features else None
        
        # Initialize feature extractors
        self.cnn_extractor = CNNFeatureExtractor() if config.use_image_features else None
        self.lstm_extractor = LSTMTemporalExtractor() if config.use_sensor_features else None
        self.tabular_extractor = TabularFeatureExtractor() if config.use_environmental_features else None
        
        # Initialize fusion and classifier
        self.fusion_model = MultiModalFusion()
        self.ensemble_classifier = EnsembleClassifier(config.ensemble_config)
        
        # Initialize training utilities
        self.cross_validator = CrossValidator(config.cv_folds, config.random_state)
        self.hyperparameter_optimizer = HyperparameterOptimizer(config.n_trials) if config.hyperparameter_optimization else None
        self.checkpoint_manager = ModelCheckpoint(self.experiment_dir, config.checkpoint_frequency)
        self.early_stopping = EarlyStopping(config.early_stopping_patience)
        
        # Training state
        self.training_history = []
        self.best_model = None
        self.best_score = -np.inf
        
        # Setup logging
        self._setup_logging()
    
    def _setup_logging(self):
        """Setup experiment logging."""
        log_file = self.experiment_dir / "training.log"
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.INFO)
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
        
        logger.info(f"Training orchestrator initialized for experiment: {self.config.experiment_name}")
    
    def load_and_split_data(self) -> Tuple[List[RockfallDataPoint], List[RockfallDataPoint], List[RockfallDataPoint]]:
        """
        Load and split data into train, validation, and test sets.
        
        Returns:
            Tuple of (train_data, val_data, test_data)
        """
        logger.info("Loading and splitting data...")
        
        # Load all data
        all_data = self.data_loader.load_all_data()
        logger.info(f"Loaded {len(all_data)} data points")
        
        # Split data
        np.random.seed(self.config.random_state)
        indices = np.random.permutation(len(all_data))
        
        train_end = int(len(all_data) * self.config.train_split)
        val_end = train_end + int(len(all_data) * self.config.val_split)
        
        train_indices = indices[:train_end]
        val_indices = indices[train_end:val_end]
        test_indices = indices[val_end:]
        
        train_data = [all_data[i] for i in train_indices]
        val_data = [all_data[i] for i in val_indices]
        test_data = [all_data[i] for i in test_indices]
        
        logger.info(f"Data split - Train: {len(train_data)}, Val: {len(val_data)}, Test: {len(test_data)}")
        
        return train_data, val_data, test_data
    
    def extract_features(self, data_points: List[RockfallDataPoint]) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """
        Extract features from data points using all available modalities.
        
        Args:
            data_points: List of RockfallDataPoint objects
            
        Returns:
            Tuple of (features, labels, feature_names)
        """
        logger.info(f"Extracting features from {len(data_points)} data points...")
        
        all_features = []
        labels = []
        feature_names = []
        
        for data_point in data_points:
            point_features = []
            
            # Extract image features
            if self.cnn_extractor and data_point.imagery:
                try:
                    processed_image = self.image_processor.process_image(data_point.imagery)
                    image_features = self.cnn_extractor.extract_features(processed_image)
                    point_features.extend(image_features.flatten())
                    if not feature_names or len([n for n in feature_names if n.startswith('image_')]) == 0:
                        feature_names.extend([f'image_feature_{i}' for i in range(len(image_features.flatten()))])
                except Exception as e:
                    logger.warning(f"Failed to extract image features: {e}")
                    # Add zeros for missing image features
                    if feature_names and any(n.startswith('image_') for n in feature_names):
                        img_feature_count = len([n for n in feature_names if n.startswith('image_')])
                        point_features.extend([0.0] * img_feature_count)
            
            # Extract terrain features
            if self.terrain_processor and data_point.dem_data:
                try:
                    terrain_features = self.terrain_processor.extract_features(data_point.dem_data)
                    point_features.extend(terrain_features)
                    if not any(n.startswith('terrain_') for n in feature_names):
                        feature_names.extend([f'terrain_feature_{i}' for i in range(len(terrain_features))])
                except Exception as e:
                    logger.warning(f"Failed to extract terrain features: {e}")
                    # Add zeros for missing terrain features
                    if feature_names and any(n.startswith('terrain_') for n in feature_names):
                        terrain_feature_count = len([n for n in feature_names if n.startswith('terrain_')])
                        point_features.extend([0.0] * terrain_feature_count)
            
            # Extract sensor features
            if self.lstm_extractor and data_point.sensor_readings:
                try:
                    processed_sensor = self.sensor_processor.process_sensor_data(data_point.sensor_readings)
                    sensor_features = self.lstm_extractor.extract_features(processed_sensor)
                    point_features.extend(sensor_features.flatten())
                    if not any(n.startswith('sensor_') for n in feature_names):
                        feature_names.extend([f'sensor_feature_{i}' for i in range(len(sensor_features.flatten()))])
                except Exception as e:
                    logger.warning(f"Failed to extract sensor features: {e}")
                    # Add zeros for missing sensor features
                    if feature_names and any(n.startswith('sensor_') for n in feature_names):
                        sensor_feature_count = len([n for n in feature_names if n.startswith('sensor_')])
                        point_features.extend([0.0] * sensor_feature_count)
            
            # Extract environmental features
            if self.tabular_extractor and data_point.environmental:
                try:
                    env_features = self.tabular_extractor.extract_features(data_point.environmental)
                    point_features.extend(env_features)
                    if not any(n.startswith('env_') for n in feature_names):
                        feature_names.extend([f'env_feature_{i}' for i in range(len(env_features))])
                except Exception as e:
                    logger.warning(f"Failed to extract environmental features: {e}")
                    # Add zeros for missing environmental features
                    if feature_names and any(n.startswith('env_') for n in feature_names):
                        env_feature_count = len([n for n in feature_names if n.startswith('env_')])
                        point_features.extend([0.0] * env_feature_count)
            
            # Extract seismic features
            if self.seismic_processor and data_point.seismic:
                try:
                    seismic_features = self.seismic_processor.extract_features(data_point.seismic)
                    point_features.extend(seismic_features)
                    if not any(n.startswith('seismic_') for n in feature_names):
                        feature_names.extend([f'seismic_feature_{i}' for i in range(len(seismic_features))])
                except Exception as e:
                    logger.warning(f"Failed to extract seismic features: {e}")
                    # Add zeros for missing seismic features
                    if feature_names and any(n.startswith('seismic_') for n in feature_names):
                        seismic_feature_count = len([n for n in feature_names if n.startswith('seismic_')])
                        point_features.extend([0.0] * seismic_feature_count)
            
            # Ensure consistent feature length
            if len(all_features) > 0 and len(point_features) != len(all_features[0]):
                # Pad or truncate to match expected length
                expected_length = len(all_features[0])
                if len(point_features) < expected_length:
                    point_features.extend([0.0] * (expected_length - len(point_features)))
                else:
                    point_features = point_features[:expected_length]
            
            all_features.append(point_features)
            
            # Extract label
            if data_point.ground_truth is not None:
                labels.append(data_point.ground_truth.value)
            else:
                labels.append(0)  # Default to LOW risk if no ground truth
        
        features = np.array(all_features)
        labels = np.array(labels)
        
        logger.info(f"Extracted features shape: {features.shape}, Labels shape: {labels.shape}")
        
        return features, labels, feature_names
    
    def train_model(self, train_data: List[RockfallDataPoint], val_data: List[RockfallDataPoint]) -> Dict[str, Any]:
        """
        Train the ensemble model with the provided data.
        
        Args:
            train_data: Training data points
            val_data: Validation data points
            
        Returns:
            Training results and metrics
        """
        logger.info("Starting model training...")
        
        # Extract features
        X_train, y_train, feature_names = self.extract_features(train_data)
        X_val, y_val, _ = self.extract_features(val_data)
        
        # Apply feature fusion if multiple modalities are present
        if self.fusion_model and X_train.shape[1] > 100:  # Arbitrary threshold for fusion
            logger.info("Applying multi-modal feature fusion...")
            X_train = self.fusion_model.fit_transform(X_train)
            X_val = self.fusion_model.transform(X_val)
        
        # Train ensemble classifier
        logger.info("Training ensemble classifier...")
        self.ensemble_classifier.fit(X_train, y_train, feature_names)
        
        # Evaluate on validation set
        val_results = self.ensemble_classifier.evaluate(X_val, y_val)
        val_accuracy = val_results['accuracy']
        
        logger.info(f"Validation accuracy: {val_accuracy:.4f}")
        
        # Update best model if this is better
        if val_accuracy > self.best_score:
            self.best_score = val_accuracy
            self.best_model = self.ensemble_classifier
            logger.info(f"New best model with validation accuracy: {val_accuracy:.4f}")
        
        # Save checkpoint
        if self.config.save_intermediate_models:
            checkpoint_path = self.experiment_dir / f"model_checkpoint_acc_{val_accuracy:.4f}.pkl"
            self.ensemble_classifier.save(checkpoint_path)
        
        # Record training history
        training_record = {
            'timestamp': datetime.now().isoformat(),
            'validation_accuracy': val_accuracy,
            'validation_results': val_results,
            'feature_count': X_train.shape[1],
            'training_samples': len(train_data),
            'validation_samples': len(val_data)
        }
        self.training_history.append(training_record)
        
        return training_record
    
    def perform_cross_validation(self, data_points: List[RockfallDataPoint]) -> Dict[str, Any]:
        """
        Perform cross-validation on the model.
        
        Args:
            data_points: All available data points
            
        Returns:
            Cross-validation results
        """
        logger.info("Performing cross-validation...")
        
        # Extract features
        X, y, feature_names = self.extract_features(data_points)
        
        # Apply feature fusion if needed
        if self.fusion_model and X.shape[1] > 100:
            X = self.fusion_model.fit_transform(X)
        
        # Perform cross-validation
        cv_results = self.cross_validator.cross_validate(
            self.ensemble_classifier, X, y, feature_names
        )
        
        logger.info(f"Cross-validation results: {cv_results}")
        
        return cv_results
    
    def optimize_hyperparameters(self, train_data: List[RockfallDataPoint], val_data: List[RockfallDataPoint]) -> Dict[str, Any]:
        """
        Optimize hyperparameters using the validation set.
        
        Args:
            train_data: Training data points
            val_data: Validation data points
            
        Returns:
            Optimization results and best parameters
        """
        if not self.hyperparameter_optimizer:
            logger.info("Hyperparameter optimization disabled")
            return {}
        
        logger.info("Starting hyperparameter optimization...")
        
        # Extract features
        X_train, y_train, feature_names = self.extract_features(train_data)
        X_val, y_val, _ = self.extract_features(val_data)
        
        # Apply feature fusion if needed
        if self.fusion_model and X_train.shape[1] > 100:
            X_train = self.fusion_model.fit_transform(X_train)
            X_val = self.fusion_model.transform(X_val)
        
        # Optimize hyperparameters
        best_params, optimization_history = self.hyperparameter_optimizer.optimize(
            X_train, y_train, X_val, y_val, feature_names
        )
        
        # Update ensemble config with best parameters
        if best_params:
            self.config.ensemble_config = EnsembleConfig(**best_params)
            self.ensemble_classifier = EnsembleClassifier(self.config.ensemble_config)
            logger.info(f"Updated ensemble config with best parameters: {best_params}")
        
        return {
            'best_parameters': best_params,
            'optimization_history': optimization_history
        }
    
    def run_full_training(self) -> Dict[str, Any]:
        """
        Run the complete training pipeline.
        
        Returns:
            Complete training results
        """
        logger.info("Starting full training pipeline...")
        start_time = datetime.now()
        
        try:
            # Load and split data
            train_data, val_data, test_data = self.load_and_split_data()
            
            # Optimize hyperparameters if enabled
            optimization_results = {}
            if self.config.hyperparameter_optimization:
                optimization_results = self.optimize_hyperparameters(train_data, val_data)
            
            # Train model
            training_results = self.train_model(train_data, val_data)
            
            # Perform cross-validation
            all_data = train_data + val_data  # Use train+val for CV
            cv_results = self.perform_cross_validation(all_data)
            
            # Final evaluation on test set
            test_results = {}
            if test_data:
                X_test, y_test, _ = self.extract_features(test_data)
                if self.fusion_model and X_test.shape[1] > 100:
                    X_test = self.fusion_model.transform(X_test)
                test_results = self.ensemble_classifier.evaluate(X_test, y_test)
                logger.info(f"Test accuracy: {test_results['accuracy']:.4f}")
            
            # Save final model
            final_model_path = self.experiment_dir / "final_model.pkl"
            self.best_model.save(final_model_path)
            
            # Save training configuration and results
            results_summary = {
                'experiment_name': self.config.experiment_name,
                'start_time': start_time.isoformat(),
                'end_time': datetime.now().isoformat(),
                'training_config': self.config.__dict__,
                'data_splits': {
                    'train_size': len(train_data),
                    'val_size': len(val_data),
                    'test_size': len(test_data)
                },
                'optimization_results': optimization_results,
                'training_results': training_results,
                'cross_validation_results': cv_results,
                'test_results': test_results,
                'training_history': self.training_history,
                'best_validation_score': self.best_score,
                'final_model_path': str(final_model_path)
            }
            
            # Save results to JSON
            results_file = self.experiment_dir / "training_results.json"
            with open(results_file, 'w') as f:
                json.dump(results_summary, f, indent=2, default=str)
            
            logger.info(f"Training completed successfully. Results saved to {results_file}")
            
            return results_summary
            
        except Exception as e:
            logger.error(f"Training failed with error: {e}")
            raise
    
    def get_training_summary(self) -> Dict[str, Any]:
        """Get a summary of the training process."""
        return {
            'experiment_name': self.config.experiment_name,
            'training_history': self.training_history,
            'best_score': self.best_score,
            'total_epochs': len(self.training_history)
        }