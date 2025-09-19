"""
Integration tests for the training pipeline.

Tests the end-to-end training workflow including data loading,
feature extraction, model training, cross-validation, and checkpointing.
"""

import pytest
import numpy as np
import tempfile
import shutil
from pathlib import Path
from datetime import datetime
from unittest.mock import Mock, patch

# Local imports
from src.training import (
    TrainingOrchestrator, TrainingConfig, DataLoader, BatchProcessor,
    CrossValidator, HyperparameterOptimizer, ModelCheckpoint, EarlyStopping
)
from src.data.schema import RockfallDataPoint, RiskLevel, GeoCoordinate
from src.models.classifiers import EnsembleClassifier, EnsembleConfig
from src.data.utils import create_sample_datapoint


class TestTrainingPipeline:
    """Integration tests for the complete training pipeline."""
    
    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory for tests."""
        temp_dir = tempfile.mkdtemp()
        yield Path(temp_dir)
        shutil.rmtree(temp_dir)
    
    @pytest.fixture
    def sample_data(self):
        """Create sample data for testing."""
        data_points = []
        for i in range(50):
            data_point = create_sample_datapoint()
            # Ensure we have ground truth labels
            data_point.ground_truth = RiskLevel(i % 3)  # Distribute across LOW, MEDIUM, HIGH
            data_points.append(data_point)
        return data_points
    
    @pytest.fixture
    def training_config(self, temp_dir):
        """Create training configuration for tests."""
        return TrainingConfig(
            data_dir=str(temp_dir / "data"),
            output_dir=str(temp_dir / "models"),
            experiment_name="test_experiment",
            max_epochs=5,  # Reduced for testing
            early_stopping_patience=3,
            cv_folds=3,  # Reduced for testing
            hyperparameter_optimization=False,  # Disabled for faster tests
            batch_size=8,
            use_image_features=False,  # Simplified for testing
            use_terrain_features=False,
            use_sensor_features=False,
            use_environmental_features=True,
            use_seismic_features=False
        )
    
    def test_data_loader_integration(self, temp_dir, sample_data):
        """Test data loading functionality."""
        # Save sample data
        data_dir = temp_dir / "data"
        data_dir.mkdir(parents=True)
        
        # Save as pickle file
        import pickle
        with open(data_dir / "sample_data.pkl", 'wb') as f:
            pickle.dump(sample_data, f)
        
        # Test loading
        data_loader = DataLoader(data_dir)
        loaded_data = data_loader.load_all_data()
        
        assert len(loaded_data) == len(sample_data)
        assert all(isinstance(dp, RockfallDataPoint) for dp in loaded_data)
    
    def test_batch_processor_integration(self, sample_data):
        """Test batch processing functionality."""
        batch_processor = BatchProcessor(batch_size=10, shuffle=True)
        
        # Test batch creation
        batches = list(batch_processor.create_batches(sample_data))
        
        assert len(batches) == 5  # 50 samples / 10 batch_size
        assert all(len(batch) <= 10 for batch in batches)
        
        # Test class balancing
        balanced_data = batch_processor.balance_classes(sample_data)
        
        # Check that classes are balanced
        class_counts = {}
        for dp in balanced_data:
            if dp.ground_truth:
                class_counts[dp.ground_truth] = class_counts.get(dp.ground_truth, 0) + 1
        
        # All classes should have similar counts
        counts = list(class_counts.values())
        assert max(counts) - min(counts) <= 1
    
    def test_cross_validator_integration(self, sample_data):
        """Test cross-validation functionality."""
        # Create simple feature matrix and labels
        X = np.random.rand(len(sample_data), 10)
        y = np.array([dp.ground_truth.value for dp in sample_data])
        
        # Test cross-validation
        cv = CrossValidator(n_folds=3, random_state=42)
        ensemble = EnsembleClassifier(EnsembleConfig())
        
        results = cv.cross_validate(ensemble, X, y)
        
        assert hasattr(results, 'mean_accuracy')
        assert hasattr(results, 'std_accuracy')
        assert len(results.fold_scores) == 3
        assert 0 <= results.mean_accuracy <= 1
    
    def test_hyperparameter_optimizer_integration(self, sample_data):
        """Test hyperparameter optimization functionality."""
        # Create simple feature matrix and labels
        X = np.random.rand(len(sample_data), 10)
        y = np.array([dp.ground_truth.value for dp in sample_data])
        
        # Split into train/val
        split_idx = len(X) // 2
        X_train, X_val = X[:split_idx], X[split_idx:]
        y_train, y_val = y[:split_idx], y[split_idx:]
        
        # Test optimization with reduced trials
        optimizer = HyperparameterOptimizer(n_trials=3)
        best_params, history = optimizer.optimize(X_train, y_train, X_val, y_val)
        
        assert isinstance(best_params, dict)
        assert len(history) <= 3
        assert all('params' in trial for trial in history)
    
    def test_checkpoint_manager_integration(self, temp_dir):
        """Test model checkpointing functionality."""
        checkpoint_dir = temp_dir / "checkpoints"
        checkpoint_manager = ModelCheckpoint(checkpoint_dir, save_frequency=1)
        
        # Create a simple model
        ensemble = EnsembleClassifier(EnsembleConfig())
        
        # Mock fit method to avoid actual training
        with patch.object(ensemble, 'fit'):
            # Save a checkpoint
            checkpoint_id = checkpoint_manager.save_checkpoint(
                ensemble, epoch=1, validation_score=0.85
            )
            
            assert checkpoint_id is not None
            assert len(checkpoint_manager.checkpoints) == 1
            
            # Save another checkpoint with better score
            checkpoint_id_2 = checkpoint_manager.save_checkpoint(
                ensemble, epoch=2, validation_score=0.90
            )
            
            assert checkpoint_manager.best_checkpoint == checkpoint_id_2
            assert checkpoint_manager.best_score == 0.90
    
    def test_early_stopping_integration(self):
        """Test early stopping functionality."""
        early_stopping = EarlyStopping(patience=3, min_delta=0.001)
        
        # Simulate improving scores
        assert not early_stopping(0.80, 1)  # First score
        assert not early_stopping(0.85, 2)  # Improvement
        assert not early_stopping(0.87, 3)  # Improvement
        assert not early_stopping(0.86, 4)  # No improvement but within patience
        assert not early_stopping(0.865, 5)  # Small improvement
        assert not early_stopping(0.86, 6)  # No improvement
        assert not early_stopping(0.855, 7)  # No improvement
        assert early_stopping(0.85, 8)  # Should trigger early stopping
        
        assert early_stopping.best_score == 0.87
        assert early_stopping.best_epoch == 3
    
    @patch('src.training.training_orchestrator.ImagePreprocessor')
    @patch('src.training.training_orchestrator.TerrainProcessor')
    @patch('src.training.training_orchestrator.SensorDataProcessor')
    @patch('src.training.training_orchestrator.EnvironmentalProcessor')
    @patch('src.training.training_orchestrator.SeismicProcessor')
    def test_training_orchestrator_integration(self, mock_seismic, mock_env, mock_sensor, 
                                             mock_terrain, mock_image, temp_dir, 
                                             training_config, sample_data):
        """Test the complete training orchestrator."""
        # Setup data directory
        data_dir = Path(training_config.data_dir)
        data_dir.mkdir(parents=True)
        
        # Save sample data
        import pickle
        with open(data_dir / "sample_data.pkl", 'wb') as f:
            pickle.dump(sample_data, f)
        
        # Mock processors to avoid complex setup
        mock_env.return_value.extract_features.return_value = np.random.rand(5)
        
        # Create orchestrator
        orchestrator = TrainingOrchestrator(training_config)
        
        # Mock feature extraction to return simple features
        with patch.object(orchestrator, 'extract_features') as mock_extract:
            # Return simple features and labels
            n_samples = len(sample_data)
            mock_extract.return_value = (
                np.random.rand(n_samples, 10),  # Features
                np.array([dp.ground_truth.value for dp in sample_data]),  # Labels
                [f'feature_{i}' for i in range(10)]  # Feature names
            )
            
            # Run training
            results = orchestrator.run_full_training()
            
            # Verify results
            assert 'experiment_name' in results
            assert 'training_results' in results
            assert 'cross_validation_results' in results
            assert 'final_model_path' in results
            
            # Check that model was saved
            model_path = Path(results['final_model_path'])
            assert model_path.exists()
    
    def test_end_to_end_training_pipeline(self, temp_dir, sample_data):
        """Test complete end-to-end training pipeline with minimal setup."""
        # Create configuration with minimal features
        config = TrainingConfig(
            data_dir=str(temp_dir / "data"),
            output_dir=str(temp_dir / "models"),
            experiment_name="e2e_test",
            max_epochs=2,
            early_stopping_patience=2,
            cv_folds=2,
            hyperparameter_optimization=False,
            batch_size=10,
            # Disable complex features for testing
            use_image_features=False,
            use_terrain_features=False,
            use_sensor_features=False,
            use_environmental_features=False,
            use_seismic_features=False
        )
        
        # Setup data
        data_dir = Path(config.data_dir)
        data_dir.mkdir(parents=True)
        
        import pickle
        with open(data_dir / "sample_data.pkl", 'wb') as f:
            pickle.dump(sample_data, f)
        
        # Create orchestrator
        orchestrator = TrainingOrchestrator(config)
        
        # Mock feature extraction to return dummy features
        def mock_extract_features(data_points):
            n_samples = len(data_points)
            features = np.random.rand(n_samples, 5)  # Simple 5-dimensional features
            labels = np.array([dp.ground_truth.value for dp in data_points])
            feature_names = [f'dummy_feature_{i}' for i in range(5)]
            return features, labels, feature_names
        
        with patch.object(orchestrator, 'extract_features', side_effect=mock_extract_features):
            # Run complete pipeline
            results = orchestrator.run_full_training()
            
            # Verify pipeline completed successfully
            assert results is not None
            assert 'experiment_name' in results
            assert results['experiment_name'] == 'e2e_test'
            
            # Check data splits
            assert 'data_splits' in results
            splits = results['data_splits']
            assert splits['train_size'] > 0
            assert splits['val_size'] > 0
            assert splits['test_size'] > 0
            
            # Check training results
            assert 'training_results' in results
            training_results = results['training_results']
            assert 'validation_accuracy' in training_results
            assert 0 <= training_results['validation_accuracy'] <= 1
            
            # Check cross-validation results
            assert 'cross_validation_results' in results
            cv_results = results['cross_validation_results']
            assert hasattr(cv_results, 'mean_accuracy') or 'mean_accuracy' in cv_results
            
            # Check that final model was saved
            assert 'final_model_path' in results
            model_path = Path(results['final_model_path'])
            assert model_path.exists()
            
            # Verify experiment directory structure
            experiment_dir = Path(config.output_dir) / config.experiment_name
            assert experiment_dir.exists()
            assert (experiment_dir / "training_results.json").exists()
            assert (experiment_dir / "final_model.pkl").exists()
    
    def test_training_with_different_configurations(self, temp_dir, sample_data):
        """Test training with different configuration options."""
        configurations = [
            # Configuration 1: With hyperparameter optimization
            TrainingConfig(
                data_dir=str(temp_dir / "data"),
                output_dir=str(temp_dir / "models1"),
                experiment_name="config1",
                max_epochs=2,
                hyperparameter_optimization=True,
                n_trials=2,  # Minimal for testing
                use_image_features=False,
                use_terrain_features=False,
                use_sensor_features=False,
                use_environmental_features=False,
                use_seismic_features=False
            ),
            # Configuration 2: Different ensemble settings
            TrainingConfig(
                data_dir=str(temp_dir / "data"),
                output_dir=str(temp_dir / "models2"),
                experiment_name="config2",
                max_epochs=2,
                hyperparameter_optimization=False,
                ensemble_config=EnsembleConfig(
                    ensemble_method="stacking",
                    rf_n_estimators=50,
                    xgb_n_estimators=50
                ),
                use_image_features=False,
                use_terrain_features=False,
                use_sensor_features=False,
                use_environmental_features=False,
                use_seismic_features=False
            )
        ]
        
        # Setup data
        data_dir = Path(temp_dir / "data")
        data_dir.mkdir(parents=True)
        
        import pickle
        with open(data_dir / "sample_data.pkl", 'wb') as f:
            pickle.dump(sample_data, f)
        
        # Test each configuration
        for config in configurations:
            orchestrator = TrainingOrchestrator(config)
            
            # Mock feature extraction
            def mock_extract_features(data_points):
                n_samples = len(data_points)
                features = np.random.rand(n_samples, 5)
                labels = np.array([dp.ground_truth.value for dp in data_points])
                feature_names = [f'feature_{i}' for i in range(5)]
                return features, labels, feature_names
            
            with patch.object(orchestrator, 'extract_features', side_effect=mock_extract_features):
                results = orchestrator.run_full_training()
                
                # Verify each configuration produces valid results
                assert results is not None
                assert results['experiment_name'] == config.experiment_name
                
                # Check configuration-specific results
                if config.hyperparameter_optimization:
                    assert 'optimization_results' in results
                
                # Verify model was saved
                model_path = Path(results['final_model_path'])
                assert model_path.exists()


class TestTrainingComponents:
    """Unit tests for individual training components."""
    
    def test_training_config_validation(self):
        """Test training configuration validation."""
        # Valid configuration
        config = TrainingConfig(
            train_split=0.7,
            val_split=0.15,
            test_split=0.15
        )
        assert config.train_split + config.val_split + config.test_split == 1.0
        
        # Invalid configuration (splits don't sum to 1.0)
        with pytest.raises(ValueError):
            TrainingConfig(
                train_split=0.8,
                val_split=0.15,
                test_split=0.15
            )
    
    def test_training_config_from_yaml(self, temp_dir):
        """Test loading configuration from YAML file."""
        # Create sample YAML config
        config_content = """
training:
  validation_split: 0.2
  test_split: 0.1
  batch_size: 64
  max_epochs: 50
  early_stopping_patience: 5
  cross_validation_folds: 5

model:
  random_forest:
    n_estimators: 200
    max_depth: 15
  xgboost:
    n_estimators: 150
    learning_rate: 0.05
  neural_network:
    hidden_layers: [512, 256, 128]
    dropout: 0.4
"""
        
        config_path = temp_dir / "config.yaml"
        with open(config_path, 'w') as f:
            f.write(config_content)
        
        # Load configuration
        config = TrainingConfig.from_yaml(config_path)
        
        assert config.val_split == 0.2
        assert config.test_split == 0.1
        assert config.batch_size == 64
        assert config.max_epochs == 50
        assert config.ensemble_config.rf_n_estimators == 200
        assert config.ensemble_config.nn_hidden_layers == [512, 256, 128]
    
    def test_batch_processor_edge_cases(self):
        """Test batch processor with edge cases."""
        # Empty data
        batch_processor = BatchProcessor(batch_size=10)
        batches = list(batch_processor.create_batches([]))
        assert len(batches) == 0
        
        # Single item
        single_item = [create_sample_datapoint()]
        batches = list(batch_processor.create_batches(single_item))
        assert len(batches) == 1
        assert len(batches[0]) == 1
        
        # Batch size larger than data
        small_data = [create_sample_datapoint() for _ in range(3)]
        batch_processor = BatchProcessor(batch_size=10)
        batches = list(batch_processor.create_batches(small_data))
        assert len(batches) == 1
        assert len(batches[0]) == 3
    
    def test_early_stopping_edge_cases(self):
        """Test early stopping with edge cases."""
        # Test with mode='min'
        early_stopping = EarlyStopping(patience=2, mode='min')
        
        assert not early_stopping(1.0, 1)  # First score
        assert not early_stopping(0.8, 2)  # Improvement (lower is better)
        assert not early_stopping(0.9, 3)  # No improvement
        assert early_stopping(0.95, 4)  # Should trigger early stopping
        
        # Test reset functionality
        early_stopping.reset()
        assert early_stopping.best_score is None
        assert early_stopping.wait == 0
        assert not early_stopping.should_stop


if __name__ == "__main__":
    pytest.main([__file__])