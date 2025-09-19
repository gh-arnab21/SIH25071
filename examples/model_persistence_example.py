"""
Example script demonstrating model persistence functionality.

This script shows how to save and load different types of models
with metadata, versioning, and preprocessing pipelines.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, classification_report

# Import model persistence utilities
from src.models.persistence import (
    ModelPersistence, save_ensemble_classifier, load_ensemble_classifier,
    PreprocessingPipelineManager
)
from src.data.quality import MissingDataImputer, OutlierDetector, ImputationStrategy


def create_sample_data():
    """Create sample rockfall prediction data."""
    # Generate synthetic features
    np.random.seed(42)
    n_samples = 1000
    
    # Features: slope_angle, rainfall, vibration, soil_moisture, rock_hardness
    X = np.random.randn(n_samples, 5)
    X[:, 0] = np.abs(X[:, 0]) * 30 + 10  # slope_angle (10-70 degrees)
    X[:, 1] = np.abs(X[:, 1]) * 50 + 5   # rainfall (5-105 mm)
    X[:, 2] = np.abs(X[:, 2]) * 10 + 1   # vibration (1-21 units)
    X[:, 3] = np.abs(X[:, 3]) * 0.3 + 0.1  # soil_moisture (0.1-0.7)
    X[:, 4] = np.abs(X[:, 4]) * 20 + 30   # rock_hardness (30-70 MPa)
    
    # Add some missing values to demonstrate preprocessing
    missing_indices = np.random.choice(n_samples, size=50, replace=False)
    feature_indices = np.random.choice(5, size=50, replace=True)
    X[missing_indices, feature_indices] = np.nan
    
    # Generate risk labels based on features
    risk_score = (X[:, 0] / 70) * 0.3 + (X[:, 1] / 105) * 0.4 + (X[:, 2] / 21) * 0.3
    y = np.where(risk_score < 0.3, 0, np.where(risk_score < 0.7, 1, 2))  # 0=Low, 1=Medium, 2=High
    
    feature_names = ['slope_angle', 'rainfall', 'vibration', 'soil_moisture', 'rock_hardness']
    target_classes = ['Low Risk', 'Medium Risk', 'High Risk']
    
    return X, y, feature_names, target_classes


def demonstrate_basic_model_persistence():
    """Demonstrate basic model saving and loading."""
    print("=" * 60)
    print("BASIC MODEL PERSISTENCE EXAMPLE")
    print("=" * 60)
    
    # Create and train a model
    X, y, feature_names, target_classes = create_sample_data()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Handle missing values first
    imputer = MissingDataImputer(strategy=ImputationStrategy.MEDIAN)
    X_train_clean = imputer.fit_transform(X_train)
    X_test_clean = imputer.transform(X_test)
    
    # Train model
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train_clean, y_train)
    
    # Evaluate model
    y_pred = model.predict(X_test_clean)
    performance_metrics = {
        'accuracy': accuracy_score(y_test, y_pred),
        'f1_score': f1_score(y_test, y_pred, average='weighted')
    }
    
    print(f"Model Performance:")
    print(f"  Accuracy: {performance_metrics['accuracy']:.3f}")
    print(f"  F1 Score: {performance_metrics['f1_score']:.3f}")
    
    # Save model using convenience function
    version, model_path = save_ensemble_classifier(
        classifier=model,
        model_name="rockfall_risk_classifier",
        feature_names=feature_names,
        target_classes=target_classes,
        performance_metrics=performance_metrics
    )
    
    print(f"\\nModel saved:")
    print(f"  Version: {version}")
    print(f"  Path: {model_path}")
    
    # Load model
    loaded_model, metadata = load_ensemble_classifier("rockfall_risk_classifier")
    
    print(f"\\nModel loaded:")
    print(f"  Name: {metadata.model_name}")
    print(f"  Type: {metadata.model_type}")
    print(f"  Version: {metadata.version}")
    print(f"  Features: {len(metadata.feature_names)}")
    print(f"  Classes: {metadata.target_classes}")
    print(f"  Framework: {metadata.framework}")
    
    # Test loaded model
    y_pred_loaded = loaded_model.predict(X_test_clean)
    loaded_accuracy = accuracy_score(y_test, y_pred_loaded)
    print(f"  Loaded model accuracy: {loaded_accuracy:.3f}")
    
    return model_path


def demonstrate_advanced_persistence():
    """Demonstrate advanced persistence with preprocessing pipelines."""
    print("\\n" + "=" * 60)
    print("ADVANCED PERSISTENCE WITH PREPROCESSING")
    print("=" * 60)
    
    # Create data
    X, y, feature_names, target_classes = create_sample_data()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Create preprocessing pipeline
    pipeline_manager = PreprocessingPipelineManager()
    
    # Define preprocessing steps
    preprocessing_steps = [
        ("imputer", MissingDataImputer, {"strategy": ImputationStrategy.MEDIAN}),
        ("outlier_detector", OutlierDetector, {"method": "iqr", "threshold": 1.5})
    ]
    
    # Create and fit pipeline
    pipeline = pipeline_manager.create_pipeline(preprocessing_steps)
    fitted_pipeline = pipeline_manager.fit_pipeline(pipeline, X_train, y_train)
    
    # Apply preprocessing
    X_train_processed = pipeline_manager.transform_data(fitted_pipeline, X_train)
    X_test_processed = pipeline_manager.transform_data(fitted_pipeline, X_test)
    
    # Train model
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train_processed, y_train)
    
    # Evaluate
    y_pred = model.predict(X_test_processed)
    performance_metrics = {
        'accuracy': accuracy_score(y_test, y_pred),
        'f1_score': f1_score(y_test, y_pred, average='weighted'),
        'train_samples': len(X_train),
        'test_samples': len(X_test)
    }
    
    print(f"Model Performance (with preprocessing):")
    print(f"  Accuracy: {performance_metrics['accuracy']:.3f}")
    print(f"  F1 Score: {performance_metrics['f1_score']:.3f}")
    
    # Initialize persistence manager
    persistence = ModelPersistence()
    
    # Save model with preprocessing pipeline
    version, model_path = persistence.save_model(
        model=model,
        model_name="rockfall_classifier_advanced",
        model_type="ensemble_classifier",
        feature_names=feature_names,
        target_classes=target_classes,
        preprocessing_pipeline=fitted_pipeline,
        performance_metrics=performance_metrics,
        description="Advanced rockfall classifier with preprocessing pipeline",
        tags=["production", "preprocessing", "v2"],
        training_info={
            "training_samples": len(X_train),
            "features_used": len(feature_names),
            "cross_validation": "5-fold",
            "hyperparameters": model.get_params()
        }
    )
    
    print(f"\\nAdvanced model saved:")
    print(f"  Version: {version}")
    print(f"  Path: {model_path}")
    
    # Load model with preprocessing
    loaded_model, metadata, loaded_preprocessing = persistence.load_model(
        "rockfall_classifier_advanced", 
        load_preprocessing=True
    )
    
    print(f"\\nAdvanced model loaded:")
    print(f"  Version: {metadata.version}")
    print(f"  Description: {metadata.description}")
    print(f"  Tags: {metadata.tags}")
    print(f"  Preprocessing steps: {len(loaded_preprocessing['steps'])}") 
    print(f"  Training info: {list(metadata.training_info.keys())}")
    
    # Test with new data using loaded preprocessing
    X_new_processed = pipeline_manager.transform_data(loaded_preprocessing, X_test)
    y_pred_new = loaded_model.predict(X_new_processed)
    new_accuracy = accuracy_score(y_test, y_pred_new)
    print(f"  New prediction accuracy: {new_accuracy:.3f}")


def demonstrate_model_versioning():
    """Demonstrate model versioning capabilities."""
    print("\\n" + "=" * 60)
    print("MODEL VERSIONING EXAMPLE")
    print("=" * 60)
    
    # Create base data
    X, y, feature_names, target_classes = create_sample_data()
    
    persistence = ModelPersistence()
    
    # Save multiple versions with different configurations
    versions_info = []
    
    for i, n_estimators in enumerate([50, 100, 200]):
        # Handle missing values
        imputer = MissingDataImputer(strategy=ImputationStrategy.MEDIAN)
        X_clean = imputer.fit_transform(X)
        
        # Train model with different configurations
        model = RandomForestClassifier(n_estimators=n_estimators, random_state=42)
        X_train, X_test, y_train, y_test = train_test_split(X_clean, y, test_size=0.2, random_state=42)
        model.fit(X_train, y_train)
        
        # Evaluate
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        
        # Determine version type
        if i == 0:
            version_type = "auto"  # v1.0.0
        elif i == 1:
            version_type = "minor"  # v1.1.0
        else:
            version_type = "major"  # v2.0.0
        
        # Save version
        version, model_path = persistence.save_model(
            model=model,
            model_name="versioned_rockfall_classifier",
            model_type="random_forest",
            feature_names=feature_names,
            target_classes=target_classes,
            performance_metrics={"accuracy": accuracy, "n_estimators": n_estimators},
            version_type=version_type,
            description=f"Random Forest with {n_estimators} estimators"
        )
        
        versions_info.append({
            'version': version,
            'n_estimators': n_estimators,
            'accuracy': accuracy
        })
        
        print(f"Saved version {version}: {n_estimators} estimators, accuracy={accuracy:.3f}")
    
    # List all models and versions
    print("\\nAll available models:")
    all_models = persistence.list_models()
    for model_name, versions in all_models.items():
        print(f"  {model_name}: {versions}")
    
    # Load and compare different versions
    print("\\nComparing model versions:")
    for version_info in versions_info:
        model, metadata, _ = persistence.load_model(
            "versioned_rockfall_classifier", 
            version_info['version']
        )
        print(f"  {metadata.version}: {metadata.description} -> accuracy={metadata.performance_metrics['accuracy']:.3f}")
    
    # Load latest version
    latest_model, latest_metadata, _ = persistence.load_model("versioned_rockfall_classifier", "latest")
    print(f"\\nLatest version: {latest_metadata.version}")
    print(f"  Performance: {latest_metadata.performance_metrics}")


def main():
    """Run all demonstration examples."""
    print("Rockfall Prediction Model Persistence Examples")
    print("=" * 60)
    
    try:
        # Run examples
        demonstrate_basic_model_persistence()
        demonstrate_advanced_persistence()
        demonstrate_model_versioning()
        
        print("\\n" + "=" * 60)
        print("ALL EXAMPLES COMPLETED SUCCESSFULLY!")
        print("=" * 60)
        
    except Exception as e:
        print(f"\\nError running examples: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()