"""
Example usage of the Ensemble Classifier for Rockfall Prediction.

This script demonstrates how to use the EnsembleClassifier to train and make
predictions on synthetic rockfall data.
"""

import numpy as np
import pandas as pd
from pathlib import Path
import sys

# Add project root to path for imports
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.models.classifiers.ensemble_classifier import EnsembleClassifier, EnsembleConfig
from src.data.schema import RiskLevel, RockfallPrediction


def generate_synthetic_data(n_samples: int = 1000, n_features: int = 20, random_state: int = 42):
    """Generate synthetic rockfall prediction data."""
    np.random.seed(random_state)
    
    # Generate features representing different data modalities
    # Features 0-4: Terrain features (slope, aspect, etc.)
    terrain_features = np.random.randn(n_samples, 5)
    
    # Features 5-9: Environmental features (rainfall, temperature, etc.)
    environmental_features = np.random.randn(n_samples, 5)
    
    # Features 10-14: Sensor features (displacement, strain, etc.)
    sensor_features = np.random.randn(n_samples, 5)
    
    # Features 15-19: Image features (CNN extracted features)
    image_features = np.random.randn(n_samples, 5)
    
    # Combine all features
    X = np.column_stack([
        terrain_features,
        environmental_features,
        sensor_features,
        image_features
    ])
    
    # Generate target labels with some logic
    # Higher risk when multiple factors are elevated
    risk_score = (
        np.mean(np.abs(terrain_features), axis=1) * 0.3 +
        np.mean(np.abs(environmental_features), axis=1) * 0.2 +
        np.mean(np.abs(sensor_features), axis=1) * 0.3 +
        np.mean(np.abs(image_features), axis=1) * 0.2
    )
    
    # Convert to risk levels
    y = np.zeros(n_samples, dtype=int)
    y[risk_score > 1.2] = 2  # High risk
    y[(risk_score > 0.8) & (risk_score <= 1.2)] = 1  # Medium risk
    # Rest remain 0 (Low risk)
    
    # Create feature names
    feature_names = (
        [f"terrain_{i}" for i in range(5)] +
        [f"environmental_{i}" for i in range(5)] +
        [f"sensor_{i}" for i in range(5)] +
        [f"image_{i}" for i in range(5)]
    )
    
    return X, y, feature_names


def main():
    """Main example function."""
    print("Ensemble Classifier Example for Rockfall Prediction")
    print("=" * 55)
    
    # Generate synthetic data
    print("\n1. Generating synthetic data...")
    X, y, feature_names = generate_synthetic_data(n_samples=500, n_features=20)
    
    print(f"   - Dataset shape: {X.shape}")
    print(f"   - Risk level distribution:")
    unique, counts = np.unique(y, return_counts=True)
    for risk_level, count in zip(unique, counts):
        risk_name = RiskLevel(risk_level).name
        print(f"     {risk_name}: {count} samples ({count/len(y)*100:.1f}%)")
    
    # Split data
    split_idx = int(0.8 * len(X))
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]
    
    print(f"\n   - Training set: {X_train.shape[0]} samples")
    print(f"   - Test set: {X_test.shape[0]} samples")
    
    # Create ensemble configuration
    print("\n2. Configuring ensemble classifier...")
    config = EnsembleConfig(
        rf_n_estimators=50,  # Reduced for faster training
        xgb_n_estimators=50,
        nn_epochs=20,
        nn_early_stopping_patience=5,
        ensemble_method="voting",
        voting_type="soft"
    )
    
    print(f"   - Random Forest: {config.rf_n_estimators} estimators")
    print(f"   - XGBoost: {config.xgb_n_estimators} estimators")
    print(f"   - Neural Network: {config.nn_epochs} epochs max")
    print(f"   - Ensemble method: {config.ensemble_method}")
    
    # Train ensemble classifier
    print("\n3. Training ensemble classifier...")
    ensemble = EnsembleClassifier(config)
    ensemble.fit(X_train, y_train, feature_names)
    
    print("   - Training completed successfully!")
    
    # Make predictions
    print("\n4. Making predictions...")
    predictions = ensemble.predict(X_test)
    probabilities = ensemble.predict_proba(X_test)
    
    # Calculate accuracy
    accuracy = np.mean(predictions == y_test)
    print(f"   - Test accuracy: {accuracy:.3f}")
    
    # Get predictions with confidence
    print("\n5. Detailed predictions with confidence...")
    detailed_predictions = ensemble.predict_with_confidence(X_test[:5])
    
    for i, pred in enumerate(detailed_predictions):
        print(f"   Sample {i+1}:")
        print(f"     Risk Level: {pred.risk_level.name}")
        print(f"     Confidence: {pred.confidence_score:.3f}")
        print(f"     Uncertainty: {pred.uncertainty_estimate:.3f}")
        print(f"     True Label: {RiskLevel(y_test[i]).name}")
        print()
    
    # Feature importance
    print("6. Feature importance analysis...")
    importance = ensemble.get_feature_importance()
    
    # Sort by importance
    sorted_features = sorted(importance.items(), key=lambda x: x[1], reverse=True)
    
    print("   Top 10 most important features:")
    for i, (feature, score) in enumerate(sorted_features[:10]):
        print(f"     {i+1:2d}. {feature:15s}: {score:.4f}")
    
    # Cross-validation
    print("\n7. Cross-validation performance...")
    cv_results = ensemble.cross_validate(X_train, y_train)
    
    print("   Individual model performance:")
    print(f"     Random Forest: {cv_results['rf_mean_accuracy']:.3f} ± {cv_results['rf_std_accuracy']:.3f}")
    print(f"     XGBoost:       {cv_results['xgb_mean_accuracy']:.3f} ± {cv_results['xgb_std_accuracy']:.3f}")
    print(f"     Neural Net:    {cv_results['nn_mean_accuracy']:.3f} ± {cv_results['nn_std_accuracy']:.3f}")
    
    # Evaluation metrics
    print("\n8. Detailed evaluation...")
    eval_results = ensemble.evaluate(X_test, y_test)
    
    print(f"   Overall accuracy: {eval_results['accuracy']:.3f}")
    print("\n   Per-class metrics:")
    
    class_report = eval_results['classification_report']
    for class_idx in ['0', '1', '2']:
        if class_idx in class_report:
            metrics = class_report[class_idx]
            risk_name = RiskLevel(int(class_idx)).name
            print(f"     {risk_name:6s}: Precision={metrics['precision']:.3f}, "
                  f"Recall={metrics['recall']:.3f}, F1={metrics['f1-score']:.3f}")
    
    # Save model
    print("\n9. Saving model...")
    model_path = Path("models/saved/ensemble_classifier_example.pkl")
    model_path.parent.mkdir(parents=True, exist_ok=True)
    ensemble.save(model_path)
    print(f"   Model saved to: {model_path}")
    
    # Load and test
    print("\n10. Loading and testing saved model...")
    loaded_ensemble = EnsembleClassifier.load(model_path)
    loaded_predictions = loaded_ensemble.predict(X_test[:10])
    original_predictions = ensemble.predict(X_test[:10])
    
    if np.array_equal(loaded_predictions, original_predictions):
        print("    ✓ Loaded model produces identical predictions")
    else:
        print("    ✗ Loaded model predictions differ from original")
    
    print("\nExample completed successfully!")


if __name__ == "__main__":
    main()