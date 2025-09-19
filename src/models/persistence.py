"""
Model persistence and loading module for the Rockfall Prediction System.

This module provides comprehensive functionality for saving and loading machine learning
models with metadata, versioning, and preprocessing pipeline preservation.
"""

import pickle
import joblib
import json
import torch
import tensorflow as tf
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional, Union, List, Tuple
from dataclasses import dataclass, asdict
import hashlib
import numpy as np
import pandas as pd
import logging
from packaging import version

from ..data.quality import MissingDataImputer, OutlierDetector, ClassImbalanceHandler

logger = logging.getLogger(__name__)


@dataclass
class ModelMetadata:
    """Metadata for saved models."""
    model_name: str
    model_type: str
    version: str
    creation_date: str
    framework: str  # 'sklearn', 'pytorch', 'tensorflow', 'xgboost'
    feature_names: List[str]
    feature_count: int
    target_classes: List[str]
    model_parameters: Dict[str, Any]
    training_info: Dict[str, Any]
    preprocessing_info: Dict[str, Any]
    performance_metrics: Dict[str, float]
    file_hash: str
    dependencies: Dict[str, str]  # package versions
    description: Optional[str] = None
    tags: List[str] = None
    
    def __post_init__(self):
        """Initialize mutable fields."""
        if self.tags is None:
            self.tags = []


class ModelPersistenceError(Exception):
    """Custom exception for model persistence operations."""
    pass


class ModelVersionManager:
    """Manages model versioning and naming conventions."""
    
    def __init__(self, base_path: Union[str, Path]):
        """Initialize version manager.
        
        Args:
            base_path: Base directory for model storage
        """
        self.base_path = Path(base_path)
        self.base_path.mkdir(parents=True, exist_ok=True)
        
    def generate_version(self, model_name: str, version_type: str = "auto") -> str:
        """Generate version number for a model.
        
        Args:
            model_name: Name of the model
            version_type: Type of versioning ("auto", "major", "minor", "patch")
            
        Returns:
            Version string in format "v1.0.0"
        """
        try:
            existing_versions = self._get_existing_versions(model_name)
            
            if not existing_versions:
                return "v1.0.0"
            
            # Parse latest version
            latest_version = max(existing_versions, key=self._parse_version)
            major, minor, patch = self._parse_version(latest_version)
            
            if version_type == "major":
                return f"v{major + 1}.0.0"
            elif version_type == "minor":
                return f"v{major}.{minor + 1}.0"
            elif version_type == "patch":
                return f"v{major}.{minor}.{patch + 1}"
            else:  # auto
                return f"v{major}.{minor}.{patch + 1}"
                
        except Exception as e:
            logger.warning(f"Error generating version for {model_name}: {e}")
            return "v1.0.0"
    
    def _get_existing_versions(self, model_name: str) -> List[str]:
        """Get existing versions for a model."""
        versions = []
        for path in self.base_path.glob(f"{model_name}_v*"):
            if path.is_dir():
                version_str = path.name.split('_')[-1]
                if version_str.startswith('v') and self._is_valid_version(version_str):
                    versions.append(version_str)
        return versions
    
    def _parse_version(self, version_str: str) -> Tuple[int, int, int]:
        """Parse version string into tuple of integers."""
        try:
            version_str = version_str.lstrip('v')
            parts = version_str.split('.')
            return int(parts[0]), int(parts[1]), int(parts[2])
        except (ValueError, IndexError):
            return 0, 0, 0
    
    def _is_valid_version(self, version_str: str) -> bool:
        """Check if version string is valid."""
        try:
            self._parse_version(version_str)
            return True
        except:
            return False
    
    def get_model_path(self, model_name: str, model_version: str) -> Path:
        """Get path for a specific model version."""
        return self.base_path / f"{model_name}_{model_version}"
    
    def list_model_versions(self, model_name: str) -> List[str]:
        """List all versions for a specific model."""
        return sorted(self._get_existing_versions(model_name), 
                     key=self._parse_version, reverse=True)


class ModelPersistence:
    """Main class for model persistence operations."""
    
    def __init__(self, base_path: Union[str, Path] = None):
        """Initialize model persistence.
        
        Args:
            base_path: Base directory for model storage. Defaults to models/saved/
        """
        if base_path is None:
            base_path = Path(__file__).parent.parent.parent / "models" / "saved"
        
        self.base_path = Path(base_path)
        self.base_path.mkdir(parents=True, exist_ok=True)
        self.version_manager = ModelVersionManager(self.base_path)
        
    def save_model(self, 
                   model: Any,
                   model_name: str,
                   model_type: str,
                   feature_names: List[str],
                   target_classes: List[str],
                   preprocessing_pipeline: Optional[Dict[str, Any]] = None,
                   performance_metrics: Optional[Dict[str, float]] = None,
                   version_type: str = "auto",
                   description: Optional[str] = None,
                   tags: Optional[List[str]] = None,
                   **kwargs) -> Tuple[str, Path]:
        """Save a model with full metadata.
        
        Args:
            model: The model object to save
            model_name: Name of the model
            model_type: Type of model (e.g., 'ensemble_classifier', 'cnn_extractor')
            feature_names: List of feature names
            target_classes: List of target class names
            preprocessing_pipeline: Preprocessing steps and their configurations
            performance_metrics: Model performance metrics
            version_type: Version increment type ("auto", "major", "minor", "patch")
            description: Model description
            tags: Model tags for categorization
            **kwargs: Additional metadata
            
        Returns:
            Tuple of (version, model_path)
        """
        try:
            # Generate version and create model directory
            model_version = self.version_manager.generate_version(model_name, version_type)
            model_dir = self.version_manager.get_model_path(model_name, model_version)
            model_dir.mkdir(parents=True, exist_ok=True)
            
            # Determine framework and save model
            framework = self._detect_framework(model)
            model_file_path = self._save_model_file(model, model_dir, framework)
            
            # Calculate file hash
            file_hash = self._calculate_file_hash(model_file_path)
            
            # Create metadata
            metadata = ModelMetadata(
                model_name=model_name,
                model_type=model_type,
                version=model_version,
                creation_date=datetime.now().isoformat(),
                framework=framework,
                feature_names=feature_names,
                feature_count=len(feature_names),
                target_classes=target_classes,
                model_parameters=self._extract_model_parameters(model),
                training_info=kwargs.get('training_info', {}),
                preprocessing_info=preprocessing_pipeline or {},
                performance_metrics=performance_metrics or {},
                file_hash=file_hash,
                dependencies=self._get_dependencies(),
                description=description,
                tags=tags or []
            )
            
            # Save metadata
            metadata_path = model_dir / "metadata.json"
            with open(metadata_path, 'w') as f:
                json.dump(asdict(metadata), f, indent=2, default=str)
            
            # Save preprocessing pipeline separately if provided
            if preprocessing_pipeline:
                self._save_preprocessing_pipeline(preprocessing_pipeline, model_dir)
            
            logger.info(f"Model saved successfully: {model_name} {model_version}")
            logger.info(f"Model path: {model_dir}")
            
            return model_version, model_dir
            
        except Exception as e:
            logger.error(f"Failed to save model {model_name}: {e}")
            raise ModelPersistenceError(f"Failed to save model: {e}")
    
    def load_model(self, 
                   model_name: str, 
                   version: str = "latest",
                   load_preprocessing: bool = True) -> Tuple[Any, ModelMetadata, Optional[Dict[str, Any]]]:
        """Load a model with metadata and preprocessing pipeline.
        
        Args:
            model_name: Name of the model to load
            version: Model version ("latest" or specific version like "v1.0.0")
            load_preprocessing: Whether to load preprocessing pipeline
            
        Returns:
            Tuple of (model, metadata, preprocessing_pipeline)
        """
        try:
            # Resolve version
            if version == "latest":
                versions = self.version_manager.list_model_versions(model_name)
                if not versions:
                    raise ModelPersistenceError(f"No versions found for model: {model_name}")
                version = versions[0]
            
            model_dir = self.version_manager.get_model_path(model_name, version)
            if not model_dir.exists():
                raise ModelPersistenceError(f"Model not found: {model_name} {version}")
            
            # Load metadata
            metadata_path = model_dir / "metadata.json"
            if not metadata_path.exists():
                raise ModelPersistenceError(f"Metadata not found for model: {model_name} {version}")
            
            with open(metadata_path, 'r') as f:
                metadata_dict = json.load(f)
            metadata = ModelMetadata(**metadata_dict)
            
            # Load model file
            model = self._load_model_file(model_dir, metadata.framework)
            
            # Load preprocessing pipeline if requested
            preprocessing_pipeline = None
            if load_preprocessing:
                preprocessing_pipeline = self._load_preprocessing_pipeline(model_dir)
            
            # Verify file integrity
            model_files = list(model_dir.glob("model.*"))
            if model_files:
                current_hash = self._calculate_file_hash(model_files[0])
                if current_hash != metadata.file_hash:
                    logger.warning(f"File hash mismatch for {model_name} {version}")
            
            logger.info(f"Model loaded successfully: {model_name} {version}")
            
            return model, metadata, preprocessing_pipeline
            
        except Exception as e:
            logger.error(f"Failed to load model {model_name} {version}: {e}")
            raise ModelPersistenceError(f"Failed to load model: {e}")
    
    def list_models(self) -> Dict[str, List[str]]:
        """List all available models and their versions.
        
        Returns:
            Dictionary mapping model names to list of versions
        """
        models = {}
        for path in self.base_path.iterdir():
            if path.is_dir() and '_v' in path.name:
                model_name = '_'.join(path.name.split('_')[:-1])
                version = path.name.split('_')[-1]
                
                if model_name not in models:
                    models[model_name] = []
                models[model_name].append(version)
        
        # Sort versions for each model
        for model_name in models:
            models[model_name] = sorted(models[model_name], 
                                      key=self.version_manager._parse_version, 
                                      reverse=True)
        
        return models
    
    def get_model_info(self, model_name: str, version: str = "latest") -> ModelMetadata:
        """Get metadata for a specific model version.
        
        Args:
            model_name: Name of the model
            version: Model version
            
        Returns:
            Model metadata
        """
        _, metadata, _ = self.load_model(model_name, version, load_preprocessing=False)
        return metadata
    
    def delete_model(self, model_name: str, version: str) -> bool:
        """Delete a specific model version.
        
        Args:
            model_name: Name of the model
            version: Model version to delete
            
        Returns:
            True if deleted successfully
        """
        try:
            model_dir = self.version_manager.get_model_path(model_name, version)
            if model_dir.exists():
                import shutil
                shutil.rmtree(model_dir)
                logger.info(f"Deleted model: {model_name} {version}")
                return True
            else:
                logger.warning(f"Model not found: {model_name} {version}")
                return False
        except Exception as e:
            logger.error(f"Failed to delete model {model_name} {version}: {e}")
            return False
    
    def _detect_framework(self, model: Any) -> str:
        """Detect the ML framework of the model."""
        if hasattr(model, 'state_dict') or isinstance(model, torch.nn.Module):
            return 'pytorch'
        elif hasattr(model, 'save_weights') or 'tensorflow' in str(type(model)).lower():
            return 'tensorflow'
        elif hasattr(model, 'save_model') and 'xgboost' in str(type(model)).lower():
            return 'xgboost'
        else:
            return 'sklearn'  # Default to sklearn/joblib
    
    def _save_model_file(self, model: Any, model_dir: Path, framework: str) -> Path:
        """Save the model file based on framework."""
        if framework == 'pytorch':
            model_path = model_dir / "model.pth"
            if hasattr(model, 'state_dict'):
                torch.save(model.state_dict(), model_path)
            else:
                torch.save(model, model_path)
        elif framework == 'tensorflow':
            model_path = model_dir / "model.h5"
            model.save(model_path)
        elif framework == 'xgboost':
            model_path = model_dir / "model.json"
            model.save_model(model_path)
        else:  # sklearn/joblib
            model_path = model_dir / "model.pkl"
            joblib.dump(model, model_path)
        
        return model_path
    
    def _load_model_file(self, model_dir: Path, framework: str) -> Any:
        """Load the model file based on framework."""
        if framework == 'pytorch':
            model_path = model_dir / "model.pth"
            return torch.load(model_path, map_location='cpu')
        elif framework == 'tensorflow':
            model_path = model_dir / "model.h5"
            return tf.keras.models.load_model(model_path)
        elif framework == 'xgboost':
            import xgboost as xgb
            model_path = model_dir / "model.json"
            model = xgb.XGBClassifier()
            model.load_model(model_path)
            return model
        else:  # sklearn/joblib
            model_path = model_dir / "model.pkl"
            return joblib.load(model_path)
    
    def _save_preprocessing_pipeline(self, pipeline: Dict[str, Any], model_dir: Path):
        """Save preprocessing pipeline."""
        pipeline_path = model_dir / "preprocessing.pkl"
        joblib.dump(pipeline, pipeline_path)
    
    def _load_preprocessing_pipeline(self, model_dir: Path) -> Optional[Dict[str, Any]]:
        """Load preprocessing pipeline."""
        pipeline_path = model_dir / "preprocessing.pkl"
        if pipeline_path.exists():
            return joblib.load(pipeline_path)
        return None
    
    def _extract_model_parameters(self, model: Any) -> Dict[str, Any]:
        """Extract model parameters for metadata."""
        try:
            if hasattr(model, 'get_params'):
                return model.get_params()
            elif hasattr(model, 'get_config'):
                return model.get_config()
            else:
                return {}
        except Exception as e:
            logger.warning(f"Could not extract model parameters: {e}")
            return {}
    
    def _calculate_file_hash(self, file_path: Path) -> str:
        """Calculate SHA256 hash of a file."""
        hash_sha256 = hashlib.sha256()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_sha256.update(chunk)
        return hash_sha256.hexdigest()
    
    def _get_dependencies(self) -> Dict[str, str]:
        """Get current package versions."""
        dependencies = {}
        try:
            import sklearn
            dependencies['scikit-learn'] = sklearn.__version__
        except ImportError:
            pass
        
        try:
            import torch
            dependencies['torch'] = torch.__version__
        except ImportError:
            pass
        
        try:
            import tensorflow as tf
            dependencies['tensorflow'] = tf.__version__
        except ImportError:
            pass
        
        try:
            import xgboost as xgb
            dependencies['xgboost'] = xgb.__version__
        except ImportError:
            pass
        
        try:
            import numpy as np
            dependencies['numpy'] = np.__version__
        except ImportError:
            pass
        
        try:
            import pandas as pd
            dependencies['pandas'] = pd.__version__
        except ImportError:
            pass
        
        return dependencies


class PreprocessingPipelineManager:
    """Manages preprocessing pipelines for model inference."""
    
    def __init__(self):
        """Initialize preprocessing pipeline manager."""
        self.pipeline_components = {}
    
    def create_pipeline(self, 
                       steps: List[Tuple[str, Any, Dict[str, Any]]]) -> Dict[str, Any]:
        """Create a preprocessing pipeline.
        
        Args:
            steps: List of (step_name, processor_class, parameters) tuples
            
        Returns:
            Preprocessing pipeline dictionary
        """
        pipeline = {
            'steps': [],
            'fitted_processors': {},
            'metadata': {
                'creation_date': datetime.now().isoformat(),
                'step_count': len(steps)
            }
        }
        
        for step_name, processor_class, parameters in steps:
            step_info = {
                'name': step_name,
                'processor_class': processor_class.__name__,
                'parameters': parameters
            }
            pipeline['steps'].append(step_info)
        
        return pipeline
    
    def fit_pipeline(self, 
                    pipeline: Dict[str, Any], 
                    X: Union[np.ndarray, pd.DataFrame],
                    y: Optional[Union[np.ndarray, pd.Series]] = None) -> Dict[str, Any]:
        """Fit preprocessing pipeline on data.
        
        Args:
            pipeline: Preprocessing pipeline
            X: Feature data
            y: Target data (optional)
            
        Returns:
            Fitted pipeline
        """
        fitted_processors = {}
        
        for step in pipeline['steps']:
            step_name = step['name']
            processor_class_name = step['processor_class']
            parameters = step['parameters']
            
            # Create processor instance
            if processor_class_name == 'MissingDataImputer':
                processor = MissingDataImputer(**parameters)
            elif processor_class_name == 'OutlierDetector':
                processor = OutlierDetector(**parameters)
            elif processor_class_name == 'ClassImbalanceHandler':
                processor = ClassImbalanceHandler(**parameters)
            else:
                logger.warning(f"Unknown processor class: {processor_class_name}")
                continue
            
            # Fit processor
            try:
                if hasattr(processor, 'fit'):
                    if processor_class_name == 'MissingDataImputer':
                        processor.fit(X)  # MissingDataImputer.fit() takes only X
                    else:
                        processor.fit(X, y)
                fitted_processors[step_name] = processor
            except Exception as e:
                logger.error(f"Failed to fit processor {step_name}: {e}")
        
        pipeline['fitted_processors'] = fitted_processors
        return pipeline
    
    def transform_data(self, 
                      pipeline: Dict[str, Any],
                      X: Union[np.ndarray, pd.DataFrame]) -> Union[np.ndarray, pd.DataFrame]:
        """Apply preprocessing pipeline to data.
        
        Args:
            pipeline: Fitted preprocessing pipeline
            X: Data to transform
            
        Returns:
            Transformed data
        """
        transformed_X = X.copy() if hasattr(X, 'copy') else X
        
        for step in pipeline['steps']:
            step_name = step['name']
            if step_name in pipeline['fitted_processors']:
                processor = pipeline['fitted_processors'][step_name]
                try:
                    if hasattr(processor, 'transform'):
                        transformed_X = processor.transform(transformed_X)
                    elif hasattr(processor, 'detect_outliers'):
                        # For outlier detector, we might want to flag outliers
                        outlier_info = processor.detect_outliers(transformed_X)
                        # Store outlier information but don't modify data
                        logger.info(f"Outliers detected: {outlier_info['overall_outlier_count']}")
                except Exception as e:
                    logger.error(f"Failed to apply processor {step_name}: {e}")
        
        return transformed_X


# Convenience functions for common operations
def save_ensemble_classifier(classifier, 
                           model_name: str,
                           feature_names: List[str],
                           target_classes: List[str],
                           performance_metrics: Optional[Dict[str, float]] = None,
                           base_path: Optional[str] = None) -> Tuple[str, Path]:
    """Save an ensemble classifier with metadata.
    
    Args:
        classifier: Ensemble classifier instance
        model_name: Name for the saved model
        feature_names: List of feature names
        target_classes: List of target class names
        performance_metrics: Model performance metrics
        base_path: Base path for saving models
        
    Returns:
        Tuple of (version, model_path)
    """
    persistence = ModelPersistence(base_path)
    return persistence.save_model(
        model=classifier,
        model_name=model_name,
        model_type="ensemble_classifier",
        feature_names=feature_names,
        target_classes=target_classes,
        performance_metrics=performance_metrics,
        description="Ensemble classifier for rockfall prediction"
    )


def load_ensemble_classifier(model_name: str,
                           version: str = "latest",
                           base_path: Optional[str] = None) -> Tuple[Any, ModelMetadata]:
    """Load an ensemble classifier.
    
    Args:
        model_name: Name of the model to load
        version: Version to load
        base_path: Base path for loading models
        
    Returns:
        Tuple of (classifier, metadata)
    """
    persistence = ModelPersistence(base_path)
    model, metadata, _ = persistence.load_model(model_name, version, load_preprocessing=False)
    return model, metadata


def save_feature_extractor(extractor,
                          model_name: str,
                          extractor_type: str,
                          feature_names: List[str],
                          performance_metrics: Optional[Dict[str, float]] = None,
                          base_path: Optional[str] = None) -> Tuple[str, Path]:
    """Save a feature extractor with metadata.
    
    Args:
        extractor: Feature extractor instance
        model_name: Name for the saved model
        extractor_type: Type of extractor (e.g., 'cnn', 'lstm', 'tabular')
        feature_names: List of feature names
        performance_metrics: Model performance metrics
        base_path: Base path for saving models
        
    Returns:
        Tuple of (version, model_path)
    """
    persistence = ModelPersistence(base_path)
    return persistence.save_model(
        model=extractor,
        model_name=model_name,
        model_type=f"{extractor_type}_feature_extractor",
        feature_names=feature_names,
        target_classes=[],  # Feature extractors don't have target classes
        performance_metrics=performance_metrics,
        description=f"{extractor_type.upper()} feature extractor for rockfall prediction"
    )


def load_feature_extractor(model_name: str,
                          version: str = "latest",
                          base_path: Optional[str] = None) -> Tuple[Any, ModelMetadata]:
    """Load a feature extractor.
    
    Args:
        model_name: Name of the model to load
        version: Version to load
        base_path: Base path for loading models
        
    Returns:
        Tuple of (extractor, metadata)
    """
    persistence = ModelPersistence(base_path)
    model, metadata, _ = persistence.load_model(model_name, version, load_preprocessing=False)
    return model, metadata