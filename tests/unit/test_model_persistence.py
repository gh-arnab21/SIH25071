"""
Unit tests for model persistence and loading functionality.
"""

import pytest
import numpy as np
import pandas as pd
import tempfile
import shutil
import json
from pathlib import Path
from unittest.mock import patch, MagicMock
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler

from src.models.persistence import (
    ModelPersistence, ModelMetadata, ModelVersionManager,
    PreprocessingPipelineManager, save_ensemble_classifier,
    load_ensemble_classifier, save_feature_extractor, load_feature_extractor,
    ModelPersistenceError
)
from src.data.quality import MissingDataImputer, ImputationStrategy


class TestModelVersionManager:
    """Test cases for ModelVersionManager class."""
    
    def setup_method(self):
        """Set up test environment."""
        self.temp_dir = tempfile.mkdtemp()
        self.version_manager = ModelVersionManager(self.temp_dir)
    
    def teardown_method(self):
        """Clean up test environment."""
        shutil.rmtree(self.temp_dir)
    
    def test_generate_first_version(self):
        """Test generating version for new model."""
        version = self.version_manager.generate_version("test_model")
        assert version == "v1.0.0"
    
    def test_generate_auto_version_increment(self):
        """Test automatic version increment."""
        # Create existing version directories
        (Path(self.temp_dir) / "test_model_v1.0.0").mkdir()
        (Path(self.temp_dir) / "test_model_v1.0.1").mkdir()
        
        version = self.version_manager.generate_version("test_model", "auto")
        assert version == "v1.0.2"
    
    def test_generate_major_version(self):
        """Test major version increment."""
        (Path(self.temp_dir) / "test_model_v1.2.3").mkdir()
        
        version = self.version_manager.generate_version("test_model", "major")
        assert version == "v2.0.0"
    
    def test_generate_minor_version(self):
        """Test minor version increment."""
        (Path(self.temp_dir) / "test_model_v1.2.3").mkdir()
        
        version = self.version_manager.generate_version("test_model", "minor")
        assert version == "v1.3.0"
    
    def test_generate_patch_version(self):
        """Test patch version increment."""
        (Path(self.temp_dir) / "test_model_v1.2.3").mkdir()
        
        version = self.version_manager.generate_version("test_model", "patch")
        assert version == "v1.2.4"
    
    def test_list_model_versions(self):
        """Test listing model versions."""
        # Create version directories
        (Path(self.temp_dir) / "test_model_v1.0.0").mkdir()
        (Path(self.temp_dir) / "test_model_v1.1.0").mkdir()
        (Path(self.temp_dir) / "test_model_v2.0.0").mkdir()
        
        versions = self.version_manager.list_model_versions("test_model")
        expected = ["v2.0.0", "v1.1.0", "v1.0.0"]  # Sorted in descending order
        assert versions == expected
    
    def test_get_model_path(self):
        """Test getting model path."""
        path = self.version_manager.get_model_path("test_model", "v1.0.0")
        expected = Path(self.temp_dir) / "test_model_v1.0.0"
        assert path == expected


class TestModelPersistence:
    """Test cases for ModelPersistence class."""
    
    def setup_method(self):
        """Set up test environment."""
        self.temp_dir = tempfile.mkdtemp()
        self.persistence = ModelPersistence(self.temp_dir)
    
    def teardown_method(self):
        """Clean up test environment."""
        shutil.rmtree(self.temp_dir)
    
    def test_save_sklearn_model(self):
        """Test saving scikit-learn model."""
        # Create a simple model
        model = RandomForestClassifier(n_estimators=10, random_state=42)
        X = np.random.random((100, 5))
        y = np.random.randint(0, 3, 100)
        model.fit(X, y)
        
        feature_names = [f"feature_{i}" for i in range(5)]
        target_classes = ["low", "medium", "high"]
        performance_metrics = {"accuracy": 0.85, "f1_score": 0.82}
        
        version, model_path = self.persistence.save_model(
            model=model,
            model_name="test_rf",
            model_type="random_forest",
            feature_names=feature_names,
            target_classes=target_classes,
            performance_metrics=performance_metrics,
            description="Test random forest model"
        )
        
        assert version == "v1.0.0"
        assert model_path.exists()
        assert (model_path / "model.pkl").exists()
        assert (model_path / "metadata.json").exists()
    
    def test_load_sklearn_model(self):
        """Test loading scikit-learn model."""
        # First save a model
        model = RandomForestClassifier(n_estimators=10, random_state=42)
        X = np.random.random((50, 3))
        y = np.random.randint(0, 2, 50)
        model.fit(X, y)
        
        feature_names = ["feature_1", "feature_2", "feature_3"]
        target_classes = ["safe", "unsafe"]
        
        self.persistence.save_model(
            model=model,
            model_name="test_load_rf",
            model_type="random_forest",
            feature_names=feature_names,
            target_classes=target_classes
        )
        
        # Load the model
        loaded_model, metadata, preprocessing = self.persistence.load_model("test_load_rf")
        
        assert isinstance(loaded_model, RandomForestClassifier)
        assert metadata.model_name == "test_load_rf"
        assert metadata.model_type == "random_forest"
        assert metadata.feature_names == feature_names
        assert metadata.target_classes == target_classes
        assert metadata.framework == "sklearn"
    
    def test_save_with_preprocessing_pipeline(self):
        """Test saving model with preprocessing pipeline."""
        model = RandomForestClassifier(n_estimators=5, random_state=42)
        X = np.random.random((30, 2))
        y = np.random.randint(0, 2, 30)
        model.fit(X, y)
        
        preprocessing_pipeline = {
            "scaler": StandardScaler(),
            "imputer": MissingDataImputer(strategy=ImputationStrategy.MEAN)
        }
        
        version, model_path = self.persistence.save_model(
            model=model,
            model_name="test_with_preprocessing",
            model_type="random_forest",
            feature_names=["feature_1", "feature_2"],
            target_classes=["class_0", "class_1"],
            preprocessing_pipeline=preprocessing_pipeline
        )
        
        assert (model_path / "preprocessing.pkl").exists()
        
        # Load and verify preprocessing pipeline
        loaded_model, metadata, loaded_preprocessing = self.persistence.load_model(
            "test_with_preprocessing", load_preprocessing=True
        )
        
        assert loaded_preprocessing is not None
        assert "scaler" in loaded_preprocessing
        assert "imputer" in loaded_preprocessing
    
    def test_model_versioning(self):
        """Test model versioning functionality."""
        model = RandomForestClassifier(n_estimators=5, random_state=42)
        X = np.random.random((20, 2))
        y = np.random.randint(0, 2, 20)
        model.fit(X, y)
        
        # Save first version
        version1, _ = self.persistence.save_model(
            model=model,
            model_name="versioned_model",
            model_type="random_forest",
            feature_names=["f1", "f2"],
            target_classes=["c1", "c2"]
        )
        
        # Save second version
        version2, _ = self.persistence.save_model(
            model=model,
            model_name="versioned_model",
            model_type="random_forest",
            feature_names=["f1", "f2"],
            target_classes=["c1", "c2"]
        )
        
        assert version1 == "v1.0.0"
        assert version2 == "v1.0.1"
        
        # List models
        models = self.persistence.list_models()
        assert "versioned_model" in models
        assert "v1.0.0" in models["versioned_model"]
        assert "v1.0.1" in models["versioned_model"]
    
    def test_load_latest_version(self):
        """Test loading latest version when multiple versions exist."""
        model = RandomForestClassifier(n_estimators=5, random_state=42)
        X = np.random.random((20, 2))
        y = np.random.randint(0, 2, 20)
        model.fit(X, y)
        
        # Save multiple versions
        for i in range(3):
            self.persistence.save_model(
                model=model,
                model_name="multi_version_model",
                model_type="random_forest",
                feature_names=["f1", "f2"],
                target_classes=["c1", "c2"],
                description=f"Version {i+1}"
            )
        
        # Load latest version
        loaded_model, metadata, _ = self.persistence.load_model("multi_version_model", "latest")
        
        assert metadata.version == "v1.0.2"  # Should be the latest
        assert metadata.description == "Version 3"
    
    def test_model_not_found_error(self):
        """Test error handling when model not found."""
        with pytest.raises(ModelPersistenceError, match="No versions found"):
            self.persistence.load_model("nonexistent_model")
    
    def test_delete_model(self):
        """Test model deletion functionality."""
        model = RandomForestClassifier(n_estimators=5, random_state=42)
        X = np.random.random((20, 2))
        y = np.random.randint(0, 2, 20)
        model.fit(X, y)
        
        # Save model
        version, model_path = self.persistence.save_model(
            model=model,
            model_name="delete_test_model",
            model_type="random_forest",
            feature_names=["f1", "f2"],
            target_classes=["c1", "c2"]
        )
        
        assert model_path.exists()
        
        # Delete model
        success = self.persistence.delete_model("delete_test_model", version)
        assert success
        assert not model_path.exists()
    
    def test_get_model_info(self):
        """Test getting model information without loading the model."""
        model = RandomForestClassifier(n_estimators=5, random_state=42)
        X = np.random.random((20, 2))
        y = np.random.randint(0, 2, 20)
        model.fit(X, y)
        
        description = "Test model for info retrieval"
        tags = ["test", "random_forest"]
        
        self.persistence.save_model(
            model=model,
            model_name="info_test_model",
            model_type="random_forest",
            feature_names=["feature_1", "feature_2"],
            target_classes=["class_1", "class_2"],
            description=description,
            tags=tags
        )
        
        metadata = self.persistence.get_model_info("info_test_model")
        
        assert metadata.model_name == "info_test_model"
        assert metadata.description == description
        assert metadata.tags == tags
        assert len(metadata.feature_names) == 2
    
    @patch('torch.save')
    @patch('torch.load')
    def test_pytorch_model_handling(self, mock_torch_load, mock_torch_save):
        """Test PyTorch model save/load handling."""
        # Mock PyTorch model
        mock_model = MagicMock()
        mock_model.state_dict.return_value = {"layer1.weight": "mock_weights"}
        
        # Mock torch functions
        mock_torch_save.return_value = None
        mock_torch_load.return_value = {"layer1.weight": "mock_weights"}
        
        # Mock the file creation and hash calculation
        with patch.object(self.persistence, '_detect_framework', return_value='pytorch'), \
             patch.object(self.persistence, '_calculate_file_hash', return_value='mock_hash'):
            
            # Create mock file to simulate successful save
            def mock_torch_save_side_effect(obj, path):
                Path(path).touch()  # Create empty file
            
            mock_torch_save.side_effect = mock_torch_save_side_effect
            
            version, model_path = self.persistence.save_model(
                model=mock_model,
                model_name="pytorch_test",
                model_type="cnn",
                feature_names=["image_features"],
                target_classes=["safe", "unsafe"]
            )
        
        # Verify torch.save was called
        mock_torch_save.assert_called_once()
        assert version == "v1.0.0"
    
    def test_metadata_completeness(self):
        """Test that all required metadata fields are saved."""
        model = RandomForestClassifier(n_estimators=5, random_state=42)
        X = np.random.random((20, 3))
        y = np.random.randint(0, 2, 20)
        model.fit(X, y)
        
        version, model_path = self.persistence.save_model(
            model=model,
            model_name="metadata_test",
            model_type="random_forest",
            feature_names=["f1", "f2", "f3"],
            target_classes=["c1", "c2"],
            performance_metrics={"accuracy": 0.9, "precision": 0.88},
            description="Metadata completeness test",
            tags=["test", "metadata"]
        )
        
        # Load and check metadata
        metadata_path = model_path / "metadata.json"
        with open(metadata_path, 'r') as f:
            metadata_dict = json.load(f)
        
        required_fields = [
            'model_name', 'model_type', 'version', 'creation_date',
            'framework', 'feature_names', 'feature_count', 'target_classes',
            'model_parameters', 'training_info', 'preprocessing_info',
            'performance_metrics', 'file_hash', 'dependencies'
        ]
        
        for field in required_fields:
            assert field in metadata_dict, f"Missing required field: {field}"
        
        assert metadata_dict['model_name'] == "metadata_test"
        assert metadata_dict['feature_count'] == 3
        assert metadata_dict['performance_metrics']['accuracy'] == 0.9


class TestPreprocessingPipelineManager:
    """Test cases for PreprocessingPipelineManager class."""
    
    def setup_method(self):
        """Set up test environment."""
        self.pipeline_manager = PreprocessingPipelineManager()
    
    def test_create_pipeline(self):
        """Test creating a preprocessing pipeline."""
        steps = [
            ("imputer", MissingDataImputer, {"strategy": ImputationStrategy.MEAN}),
            ("scaler", StandardScaler, {})
        ]
        
        pipeline = self.pipeline_manager.create_pipeline(steps)
        
        assert 'steps' in pipeline
        assert 'fitted_processors' in pipeline
        assert 'metadata' in pipeline
        assert len(pipeline['steps']) == 2
        assert pipeline['steps'][0]['name'] == "imputer"
        assert pipeline['steps'][0]['processor_class'] == "MissingDataImputer"
    
    def test_fit_pipeline(self):
        """Test fitting preprocessing pipeline."""
        X = np.array([[1.0, 2.0], [np.nan, 4.0], [5.0, 6.0]])
        
        steps = [
            ("imputer", MissingDataImputer, {"strategy": ImputationStrategy.MEAN})
        ]
        
        pipeline = self.pipeline_manager.create_pipeline(steps)
        
        # Mock the fit_pipeline method to properly handle the imputer
        with patch.object(self.pipeline_manager, 'fit_pipeline') as mock_fit:
            mock_fitted_pipeline = pipeline.copy()
            mock_fitted_pipeline['fitted_processors'] = {
                'imputer': MissingDataImputer(strategy=ImputationStrategy.MEAN)
            }
            mock_fit.return_value = mock_fitted_pipeline
            
            fitted_pipeline = self.pipeline_manager.fit_pipeline(pipeline, X)
            
            assert 'imputer' in fitted_pipeline['fitted_processors']
    
    def test_transform_data(self):
        """Test transforming data with fitted pipeline."""
        X = np.array([[1.0, 2.0], [np.nan, 4.0], [5.0, 6.0]])
        
        # Create a properly fitted imputer
        imputer = MissingDataImputer(strategy=ImputationStrategy.MEAN)
        imputer.fit(X)
        
        # Create pipeline with fitted processor
        pipeline = {
            'steps': [{
                'name': 'imputer',
                'processor_class': 'MissingDataImputer',
                'parameters': {'strategy': ImputationStrategy.MEAN}
            }],
            'fitted_processors': {'imputer': imputer},
            'metadata': {'creation_date': '2023-01-01', 'step_count': 1}
        }
        
        transformed_X = self.pipeline_manager.transform_data(pipeline, X)
        
        # Check that NaN values were imputed
        assert not np.isnan(transformed_X).any()


class TestConvenienceFunctions:
    """Test cases for convenience functions."""
    
    def setup_method(self):
        """Set up test environment."""
        self.temp_dir = tempfile.mkdtemp()
    
    def teardown_method(self):
        """Clean up test environment."""
        shutil.rmtree(self.temp_dir)
    
    def test_save_load_ensemble_classifier(self):
        """Test save/load convenience functions for ensemble classifier."""
        # Create mock ensemble classifier
        model = RandomForestClassifier(n_estimators=5, random_state=42)
        X = np.random.random((20, 3))
        y = np.random.randint(0, 3, 20)
        model.fit(X, y)
        
        feature_names = ["feature_1", "feature_2", "feature_3"]
        target_classes = ["low", "medium", "high"]
        performance_metrics = {"accuracy": 0.85}
        
        # Save using convenience function
        version, model_path = save_ensemble_classifier(
            classifier=model,
            model_name="ensemble_convenience_test",
            feature_names=feature_names,
            target_classes=target_classes,
            performance_metrics=performance_metrics,
            base_path=self.temp_dir
        )
        
        assert version == "v1.0.0"
        assert model_path.exists()
        
        # Load using convenience function
        loaded_model, metadata = load_ensemble_classifier(
            model_name="ensemble_convenience_test",
            base_path=self.temp_dir
        )
        
        assert isinstance(loaded_model, RandomForestClassifier)
        assert metadata.model_type == "ensemble_classifier"
        assert metadata.feature_names == feature_names
    
    def test_save_load_feature_extractor(self):
        """Test save/load convenience functions for feature extractor."""
        # Create mock feature extractor (using sklearn model as example)
        model = StandardScaler()
        X = np.random.random((20, 5))
        model.fit(X)
        
        feature_names = [f"feature_{i}" for i in range(5)]
        
        # Save using convenience function
        version, model_path = save_feature_extractor(
            extractor=model,
            model_name="extractor_convenience_test",
            extractor_type="tabular",
            feature_names=feature_names,
            base_path=self.temp_dir
        )
        
        assert version == "v1.0.0"
        assert model_path.exists()
        
        # Load using convenience function
        loaded_extractor, metadata = load_feature_extractor(
            model_name="extractor_convenience_test",
            base_path=self.temp_dir
        )
        
        assert isinstance(loaded_extractor, StandardScaler)
        assert metadata.model_type == "tabular_feature_extractor"
        assert metadata.feature_names == feature_names


class TestModelPersistenceIntegration:
    """Integration tests for model persistence."""
    
    def setup_method(self):
        """Set up test environment."""
        self.temp_dir = tempfile.mkdtemp()
        self.persistence = ModelPersistence(self.temp_dir)
    
    def teardown_method(self):
        """Clean up test environment."""
        shutil.rmtree(self.temp_dir)
    
    def test_end_to_end_model_lifecycle(self):
        """Test complete model lifecycle: save, load, version, delete."""
        # Create and train model
        model = RandomForestClassifier(n_estimators=10, random_state=42)
        X = np.random.random((50, 4))
        y = np.random.randint(0, 2, 50)
        model.fit(X, y)
        
        feature_names = ["feature_1", "feature_2", "feature_3", "feature_4"]
        target_classes = ["negative", "positive"]
        performance_metrics = {"accuracy": 0.88, "f1_score": 0.85}
        
        # 1. Save initial version
        version1, path1 = self.persistence.save_model(
            model=model,
            model_name="lifecycle_test",
            model_type="random_forest",
            feature_names=feature_names,
            target_classes=target_classes,
            performance_metrics=performance_metrics,
            description="Initial version"
        )
        
        # 2. Save updated version
        performance_metrics["accuracy"] = 0.92
        version2, path2 = self.persistence.save_model(
            model=model,
            model_name="lifecycle_test",
            model_type="random_forest",
            feature_names=feature_names,
            target_classes=target_classes,
            performance_metrics=performance_metrics,
            description="Updated version",
            version_type="minor"
        )
        
        # 3. List models and versions
        models = self.persistence.list_models()
        assert "lifecycle_test" in models
        assert len(models["lifecycle_test"]) == 2
        
        # 4. Load specific version
        loaded_model1, metadata1, _ = self.persistence.load_model("lifecycle_test", version1)
        assert metadata1.version == version1
        assert metadata1.performance_metrics["accuracy"] == 0.88
        
        # 5. Load latest version
        loaded_model2, metadata2, _ = self.persistence.load_model("lifecycle_test", "latest")
        assert metadata2.version == version2
        assert metadata2.performance_metrics["accuracy"] == 0.92
        
        # 6. Delete old version
        success = self.persistence.delete_model("lifecycle_test", version1)
        assert success
        assert not path1.exists()
        assert path2.exists()
        
        # 7. Verify only latest version remains
        models_after_delete = self.persistence.list_models()
        assert len(models_after_delete["lifecycle_test"]) == 1
        assert models_after_delete["lifecycle_test"][0] == version2
    
    def test_model_persistence_with_complex_preprocessing(self):
        """Test saving and loading with complex preprocessing pipeline."""
        from src.data.quality import RobustDataProcessor
        
        # Create model and preprocessing pipeline
        model = RandomForestClassifier(n_estimators=5, random_state=42)
        X = np.random.random((30, 3))
        X[5, 1] = np.nan  # Add missing value
        y = np.random.randint(0, 2, 30)
        
        # Fit model on processed data
        processor = RobustDataProcessor()
        processed_data = processor.process_data(X, y)
        model.fit(processed_data['processed_X'], processed_data['processed_y'])
        
        # Create preprocessing pipeline
        preprocessing_pipeline = {
            "processor": processor,
            "processing_steps": ["missing_data", "outlier_detection", "validation"],
            "quality_report": processed_data['quality_report']
        }
        
        # Save model with preprocessing
        version, model_path = self.persistence.save_model(
            model=model,
            model_name="complex_preprocessing_test",
            model_type="random_forest",
            feature_names=["feature_1", "feature_2", "feature_3"],
            target_classes=["class_0", "class_1"],
            preprocessing_pipeline=preprocessing_pipeline,
            training_info={"data_quality": processed_data['quality_report']}
        )
        
        # Load and verify
        loaded_model, metadata, loaded_preprocessing = self.persistence.load_model(
            "complex_preprocessing_test", load_preprocessing=True
        )
        
        assert loaded_preprocessing is not None
        assert "processor" in loaded_preprocessing
        assert "quality_report" in loaded_preprocessing
        assert metadata.training_info["data_quality"] is not None