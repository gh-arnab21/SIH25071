"""Machine learning models and feature extractors."""

from .fusion import MultiModalFusion
from .persistence import (
    ModelPersistence, ModelMetadata, ModelVersionManager,
    PreprocessingPipelineManager, save_ensemble_classifier,
    load_ensemble_classifier, save_feature_extractor, load_feature_extractor
)

__all__ = [
    'MultiModalFusion',
    'ModelPersistence', 'ModelMetadata', 'ModelVersionManager',
    'PreprocessingPipelineManager', 'save_ensemble_classifier',
    'load_ensemble_classifier', 'save_feature_extractor', 'load_feature_extractor'
]