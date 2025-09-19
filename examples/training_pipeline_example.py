"""
Example demonstrating the training pipeline for the Rockfall Prediction System.

This example shows how to use the training orchestrator to train the ensemble model
with sample data and various configuration options.
"""

import tempfile
import shutil
from pathlib import Path
import numpy as np
import pickle

from src.training import TrainingOrchestrator, TrainingConfig
from src.data.utils import create_sample_datapoint
from src.data.schema import RiskLevel


def create_sample_dataset(n_samples: int = 100):
    """Create a sample dataset for demonstration."""
    print(f"Creating sample dataset with {n_samples} data points...")
    
    sample_data = []
    for i in range(n_samples):
        data_point = create_sample_datapoint()
        # Distribute across risk levels with some pattern
        if i < n_samples // 3:
            data_point.ground_truth = RiskLevel.LOW
        elif i < 2 * n_samples // 3:
            data_point.ground_truth = RiskLevel.MEDIUM
        else:
            data_point.ground_truth = RiskLevel.HIGH
        
        sample_data.append(data_point)
    
    return sample_data


def demonstrate_basic_training():
    """Demonstrate basic training pipeline."""
    print("\n" + "="*60)
    print("BASIC TRAINING PIPELINE DEMONSTRATION")
    print("="*60)
    
    # Create temporary directory
    temp_dir = Path(tempfile.mkdtemp())
    print(f"Working directory: {temp_dir}")
    
    try:
        # Create and save sample data
        sample_data = create_sample_dataset(60)
        
        data_dir = temp_dir / "data"
        data_dir.mkdir(parents=True)
        
        with open(data_dir / "sample_data.pkl", 'wb') as f:
            pickle.dump(sample_data, f)
        
        print(f"Sample data saved to: {data_dir}")
        
        # Create basic configuration
        config = TrainingConfig(
            data_dir=str(data_dir),
            output_dir=str(temp_dir / "models"),
            experiment_name="basic_demo",
            max_epochs=3,
            early_stopping_patience=2,
            cv_folds=3,
            hyperparameter_optimization=False,
            batch_size=16,
            # Use minimal features for demo
            use_image_features=False,
            use_terrain_features=False,
            use_sensor_features=False,
            use_environmental_features=False,
            use_seismic_features=False
        )
        
        print("\nTraining Configuration:")
        print(f"  Max Epochs: {config.max_epochs}")
        print(f"  Batch Size: {config.batch_size}")
        print(f"  CV Folds: {config.cv_folds}")
        print(f"  Early Stopping Patience: {config.early_stopping_patience}")
        
        # Create orchestrator
        orchestrator = TrainingOrchestrator(config)
        
        # Mock feature extraction for demo
        def mock_extract_features(data_points):
            n_samples = len(data_points)
            # Create simple synthetic features
            features = np.random.rand(n_samples, 8)
            labels = np.array([dp.ground_truth.value for dp in data_points])
            feature_names = [f'synthetic_feature_{i}' for i in range(8)]
            return features, labels, feature_names
        
        orchestrator.extract_features = mock_extract_features
        
        # Run training
        print("\nStarting training...")
        results = orchestrator.run_full_training()
        
        # Display results
        print("\nTraining Results:")
        print(f"  Experiment: {results['experiment_name']}")
        print(f"  Training Samples: {results['data_splits']['train_size']}")
        print(f"  Validation Samples: {results['data_splits']['val_size']}")
        print(f"  Test Samples: {results['data_splits']['test_size']}")
        
        if 'training_results' in results:
            training_results = results['training_results']
            print(f"  Final Validation Accuracy: {training_results['validation_accuracy']:.4f}")
        
        if 'cross_validation_results' in results:
            cv_results = results['cross_validation_results']
            if hasattr(cv_results, 'mean_accuracy'):
                print(f"  CV Mean Accuracy: {cv_results.mean_accuracy:.4f} ± {cv_results.std_accuracy:.4f}")
        
        print(f"  Model saved to: {results['final_model_path']}")
        
        return results
        
    finally:
        # Note: Cleanup might fail on Windows due to file locking
        try:
            shutil.rmtree(temp_dir)
        except:
            print(f"Note: Could not clean up {temp_dir} (Windows file locking)")


def demonstrate_hyperparameter_optimization():
    """Demonstrate training with hyperparameter optimization."""
    print("\n" + "="*60)
    print("HYPERPARAMETER OPTIMIZATION DEMONSTRATION")
    print("="*60)
    
    temp_dir = Path(tempfile.mkdtemp())
    print(f"Working directory: {temp_dir}")
    
    try:
        # Create sample data
        sample_data = create_sample_dataset(80)
        
        data_dir = temp_dir / "data"
        data_dir.mkdir(parents=True)
        
        with open(data_dir / "sample_data.pkl", 'wb') as f:
            pickle.dump(sample_data, f)
        
        # Configuration with hyperparameter optimization
        config = TrainingConfig(
            data_dir=str(data_dir),
            output_dir=str(temp_dir / "models"),
            experiment_name="hyperopt_demo",
            max_epochs=3,
            hyperparameter_optimization=True,
            n_trials=5,  # Small number for demo
            cv_folds=3,
            batch_size=16,
            use_image_features=False,
            use_terrain_features=False,
            use_sensor_features=False,
            use_environmental_features=False,
            use_seismic_features=False
        )
        
        print(f"\nHyperparameter Optimization:")
        print(f"  Number of Trials: {config.n_trials}")
        print(f"  Optimization Enabled: {config.hyperparameter_optimization}")
        
        orchestrator = TrainingOrchestrator(config)
        
        # Mock feature extraction
        def mock_extract_features(data_points):
            n_samples = len(data_points)
            features = np.random.rand(n_samples, 10)
            labels = np.array([dp.ground_truth.value for dp in data_points])
            feature_names = [f'feature_{i}' for i in range(10)]
            return features, labels, feature_names
        
        orchestrator.extract_features = mock_extract_features
        
        print("\nStarting training with hyperparameter optimization...")
        results = orchestrator.run_full_training()
        
        print("\nOptimization Results:")
        if 'optimization_results' in results:
            opt_results = results['optimization_results']
            if 'best_parameters' in opt_results:
                print("  Best Parameters Found:")
                for param, value in opt_results['best_parameters'].items():
                    print(f"    {param}: {value}")
        
        return results
        
    finally:
        try:
            shutil.rmtree(temp_dir)
        except:
            print(f"Note: Could not clean up {temp_dir}")


def demonstrate_cross_validation():
    """Demonstrate cross-validation functionality."""
    print("\n" + "="*60)
    print("CROSS-VALIDATION DEMONSTRATION")
    print("="*60)
    
    temp_dir = Path(tempfile.mkdtemp())
    
    try:
        # Create sample data
        sample_data = create_sample_dataset(90)
        
        data_dir = temp_dir / "data"
        data_dir.mkdir(parents=True)
        
        with open(data_dir / "sample_data.pkl", 'wb') as f:
            pickle.dump(sample_data, f)
        
        config = TrainingConfig(
            data_dir=str(data_dir),
            output_dir=str(temp_dir / "models"),
            experiment_name="cv_demo",
            cv_folds=5,  # More folds for better CV
            hyperparameter_optimization=False,
            use_image_features=False,
            use_terrain_features=False,
            use_sensor_features=False,
            use_environmental_features=False,
            use_seismic_features=False
        )
        
        print(f"Cross-Validation with {config.cv_folds} folds")
        
        orchestrator = TrainingOrchestrator(config)
        
        # Mock feature extraction
        def mock_extract_features(data_points):
            n_samples = len(data_points)
            features = np.random.rand(n_samples, 6)
            labels = np.array([dp.ground_truth.value for dp in data_points])
            feature_names = [f'cv_feature_{i}' for i in range(6)]
            return features, labels, feature_names
        
        orchestrator.extract_features = mock_extract_features
        
        print("\nRunning cross-validation...")
        results = orchestrator.run_full_training()
        
        print("\nCross-Validation Results:")
        if 'cross_validation_results' in results:
            cv_results = results['cross_validation_results']
            if hasattr(cv_results, 'fold_scores'):
                print("  Per-Fold Results:")
                for fold_score in cv_results.fold_scores:
                    print(f"    Fold {fold_score['fold']}: {fold_score['accuracy']:.4f}")
                print(f"  Mean ± Std: {cv_results.mean_accuracy:.4f} ± {cv_results.std_accuracy:.4f}")
        
        return results
        
    finally:
        try:
            shutil.rmtree(temp_dir)
        except:
            pass


def main():
    """Run all demonstrations."""
    print("ROCKFALL PREDICTION SYSTEM - TRAINING PIPELINE EXAMPLES")
    print("="*80)
    
    try:
        # Run demonstrations
        demonstrate_basic_training()
        demonstrate_hyperparameter_optimization()
        demonstrate_cross_validation()
        
        print("\n" + "="*80)
        print("ALL DEMONSTRATIONS COMPLETED SUCCESSFULLY!")
        print("="*80)
        
    except Exception as e:
        print(f"\nError during demonstration: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()