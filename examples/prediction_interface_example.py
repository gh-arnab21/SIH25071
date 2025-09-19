"""
Example usage of the Prediction Interface for Rockfall Prediction.

This script demonstrates how to use the PredictionInterface to make predictions
on new data using trained models and preprocessors.
"""

import numpy as np
import pandas as pd
from pathlib import Path
import sys
from datetime import datetime

# Add project root to path for imports
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.models.prediction_interface import PredictionInterface, PredictionError
from src.models.classifiers.ensemble_classifier import EnsembleClassifier, EnsembleConfig
from src.models.persistence import ModelPersistence
from src.data.schema import (
    RockfallDataPoint, RiskLevel, GeoCoordinate, 
    EnvironmentalData, SensorData, TimeSeries
)


def create_sample_trained_model():
    """Create and train a sample model for demonstration."""
    print("Creating and training sample model...")
    
    # Generate synthetic training data
    np.random.seed(42)
    X_train = np.random.randn(200, 15)
    y_train = np.random.randint(0, 3, 200)
    feature_names = [f"feature_{i}" for i in range(15)]
    
    # Create and train ensemble classifier
    config = EnsembleConfig(
        rf_n_estimators=10,  # Small for demo
        xgb_n_estimators=10,
        nn_epochs=5,
        nn_early_stopping_patience=2
    )
    
    ensemble = EnsembleClassifier(config)
    ensemble.fit(X_train, y_train, feature_names)
    
    return ensemble, feature_names


def save_model_and_preprocessor(model, feature_names):
    """Save model for later use."""
    print("Saving model...")
    
    # Create a simple demonstration by directly saving with joblib
    import joblib
    import os
    
    # Ensure directory exists
    os.makedirs("models/saved", exist_ok=True)
    
    model_path = "models/saved/prediction_interface_demo.pkl"
    model_data = {
        'model': model,
        'feature_names': feature_names,
        'metadata': {
            'model_type': 'ensemble_classifier',
            'created_date': datetime.now().isoformat(),
            'description': 'Demo model for prediction interface'
        }
    }
    
    joblib.dump(model_data, model_path)
    print(f"Model saved to: {model_path}")
    return model_path, None


def create_sample_prediction_data():
    """Create sample data for prediction."""
    print("Creating sample prediction data...")
    
    # Create feature matrix
    np.random.seed(123)
    feature_matrix = np.random.randn(10, 15)
    
    # Create DataFrame
    feature_df = pd.DataFrame(
        feature_matrix,
        columns=[f"feature_{i}" for i in range(15)]
    )
    
    # Create RockfallDataPoint objects
    datapoints = []
    for i in range(3):
        datapoint = RockfallDataPoint(
            timestamp=datetime.now(),
            location=GeoCoordinate(
                latitude=45.0 + i * 0.1,
                longitude=-120.0 + i * 0.1,
                elevation=1000.0 + i * 50
            ),
            environmental=EnvironmentalData(
                rainfall=10.0 + i * 2,
                temperature=15.0 + i,
                vibrations=0.1 + i * 0.05,
                wind_speed=5.0 + i,
                humidity=60.0 + i * 5
            ),
            sensor_readings=SensorData(
                displacement=TimeSeries(
                    timestamps=[datetime.now()],
                    values=[0.05 + i * 0.01],
                    unit="mm"
                ),
                strain=TimeSeries(
                    timestamps=[datetime.now()],
                    values=[0.02 + i * 0.005],
                    unit="microstrains"
                ),
                pore_pressure=TimeSeries(
                    timestamps=[datetime.now()],
                    values=[100.0 + i * 10],
                    unit="kPa"
                )
            )
        )
        datapoints.append(datapoint)
    
    return feature_matrix, feature_df, datapoints


def demonstrate_basic_prediction():
    """Demonstrate basic prediction functionality."""
    print("\n" + "="*60)
    print("BASIC PREDICTION DEMONSTRATION")
    print("="*60)
    
    # Create and save model
    model, feature_names = create_sample_trained_model()
    model_path, preprocessor_path = save_model_and_preprocessor(model, feature_names)
    
    # Create prediction interface
    print(f"\nLoading model from: {model_path}")
    
    interface = PredictionInterface(model_path=model_path)
    
    # Get model info
    model_info = interface.get_model_info()
    print(f"\nModel Information:")
    print(f"  Type: {model_info['model_type']}")
    print(f"  Features: {model_info['feature_count']}")
    print(f"  Is Ensemble: {model_info.get('model_details', {}).get('is_ensemble', False)}")
    
    # Create sample data
    feature_matrix, feature_df, datapoints = create_sample_prediction_data()
    
    # Single prediction with numpy array
    print(f"\n1. Single Prediction (Numpy Array):")
    single_prediction = interface.predict_single(feature_matrix[0])
    print(f"   Risk Level: {single_prediction.risk_level.name}")
    print(f"   Confidence: {single_prediction.confidence_score:.3f}")
    print(f"   Uncertainty: {single_prediction.uncertainty_estimate:.3f}")
    
    # Single prediction with DataFrame
    print(f"\n2. Single Prediction (DataFrame):")
    single_prediction_df = interface.predict_single(feature_df.iloc[0:1])
    print(f"   Risk Level: {single_prediction_df.risk_level.name}")
    print(f"   Confidence: {single_prediction_df.confidence_score:.3f}")
    
    # Single prediction with RockfallDataPoint
    print(f"\n3. Single Prediction (RockfallDataPoint):")
    single_prediction_dp = interface.predict_single(datapoints[0])
    print(f"   Risk Level: {single_prediction_dp.risk_level.name}")
    print(f"   Confidence: {single_prediction_dp.confidence_score:.3f}")
    
    return interface, feature_matrix, feature_df, datapoints


def demonstrate_batch_prediction(interface, feature_matrix, feature_df, datapoints):
    """Demonstrate batch prediction functionality."""
    print("\n" + "="*60)
    print("BATCH PREDICTION DEMONSTRATION")
    print("="*60)
    
    # Batch prediction with feature matrix
    print(f"\n1. Batch Prediction (Feature Matrix - {feature_matrix.shape[0]} samples):")
    
    def progress_callback(progress, current, total):
        if current % 5 == 0 or current == total:  # Report every 5 samples or at the end
            print(f"   Progress: {current}/{total} ({progress:.1f}%)")
    
    batch_predictions = interface.predict_batch(
        feature_matrix,
        batch_size=3,
        progress_callback=progress_callback
    )
    
    print(f"   Completed {len(batch_predictions)} predictions")
    
    # Show risk distribution
    risk_counts = {}
    for pred in batch_predictions:
        risk_name = pred.risk_level.name
        risk_counts[risk_name] = risk_counts.get(risk_name, 0) + 1
    
    print(f"   Risk Distribution:")
    for risk, count in risk_counts.items():
        percentage = (count / len(batch_predictions)) * 100
        print(f"     {risk}: {count} ({percentage:.1f}%)")
    
    # Batch prediction with DataFrame
    print(f"\n2. Batch Prediction (DataFrame - {len(feature_df)} samples):")
    df_predictions = interface.predict_batch(feature_df, batch_size=2)
    print(f"   Completed {len(df_predictions)} predictions")
    
    # Batch prediction with RockfallDataPoint objects
    print(f"\n3. Batch Prediction (RockfallDataPoints - {len(datapoints)} samples):")
    dp_predictions = interface.predict_batch(datapoints, batch_size=1)
    print(f"   Completed {len(dp_predictions)} predictions")
    
    for i, pred in enumerate(dp_predictions):
        print(f"   Sample {i+1}: {pred.risk_level.name} (confidence: {pred.confidence_score:.3f})")
    
    return batch_predictions


def demonstrate_prediction_analysis(interface, predictions):
    """Demonstrate prediction analysis and summary statistics."""
    print("\n" + "="*60)
    print("PREDICTION ANALYSIS DEMONSTRATION")
    print("="*60)
    
    # Get prediction summary
    summary = interface.get_prediction_summary(predictions)
    
    print(f"\nPrediction Summary:")
    print(f"  Total Predictions: {summary['total_predictions']}")
    
    print(f"\n  Risk Distribution:")
    for risk, count in summary['risk_distribution'].items():
        percentage = summary['risk_percentages'][risk]
        print(f"    {risk}: {count} samples ({percentage:.1f}%)")
    
    print(f"\n  Confidence Statistics:")
    conf_stats = summary['confidence_stats']
    print(f"    Mean: {conf_stats['mean']:.3f}")
    print(f"    Std:  {conf_stats['std']:.3f}")
    print(f"    Min:  {conf_stats['min']:.3f}")
    print(f"    Max:  {conf_stats['max']:.3f}")
    
    if 'uncertainty_stats' in summary:
        print(f"\n  Uncertainty Statistics:")
        unc_stats = summary['uncertainty_stats']
        print(f"    Mean: {unc_stats['mean']:.3f}")
        print(f"    Std:  {unc_stats['std']:.3f}")
        print(f"    Min:  {unc_stats['min']:.3f}")
        print(f"    Max:  {unc_stats['max']:.3f}")
    
    # Analyze contributing factors
    print(f"\n  Top Contributing Factors:")
    all_factors = {}
    for pred in predictions:
        for factor, importance in pred.contributing_factors.items():
            if factor not in all_factors:
                all_factors[factor] = []
            all_factors[factor].append(importance)
    
    # Calculate average importance for each factor
    avg_factors = {
        factor: np.mean(importances) 
        for factor, importances in all_factors.items()
    }
    
    # Sort by importance
    sorted_factors = sorted(avg_factors.items(), key=lambda x: x[1], reverse=True)
    
    for i, (factor, importance) in enumerate(sorted_factors[:5]):  # Top 5
        print(f"    {i+1}. {factor}: {importance:.4f}")


def demonstrate_error_handling():
    """Demonstrate error handling scenarios."""
    print("\n" + "="*60)
    print("ERROR HANDLING DEMONSTRATION")
    print("="*60)
    
    # Test with unloaded model
    print("\n1. Prediction without loaded model:")
    interface = PredictionInterface()
    try:
        interface.predict(np.array([[1, 2, 3]]))
    except PredictionError as e:
        print(f"   Expected error: {e}")
    
    # Test with invalid model path
    print("\n2. Loading non-existent model:")
    try:
        interface.load_model("non_existent_model.pkl")
    except PredictionError as e:
        print(f"   Expected error: {e}")
    
    # Test with wrong input dimensions (using a loaded model)
    print("\n3. Wrong input dimensions:")
    try:
        # First create a working interface
        model, feature_names = create_sample_trained_model()
        interface.model = model
        interface.is_loaded = True
        interface.feature_names = feature_names
        
        # Try to predict with wrong number of features
        wrong_input = np.array([[1, 2, 3]])  # Only 3 features, expected 15
        interface.predict(wrong_input)
    except PredictionError as e:
        print(f"   Expected error: {e}")


def main():
    """Main demonstration function."""
    print("Prediction Interface Example for Rockfall Prediction")
    print("=" * 65)
    
    try:
        # Basic prediction demonstration
        interface, feature_matrix, feature_df, datapoints = demonstrate_basic_prediction()
        
        # Batch prediction demonstration
        predictions = demonstrate_batch_prediction(interface, feature_matrix, feature_df, datapoints)
        
        # Prediction analysis demonstration
        demonstrate_prediction_analysis(interface, predictions)
        
        # Error handling demonstration
        demonstrate_error_handling()
        
        print("\n" + "="*60)
        print("DEMONSTRATION COMPLETED SUCCESSFULLY")
        print("="*60)
        print("\nKey Features Demonstrated:")
        print("  ✓ Model and preprocessor loading")
        print("  ✓ Single predictions (multiple input formats)")
        print("  ✓ Batch predictions with progress tracking")
        print("  ✓ Prediction analysis and statistics")
        print("  ✓ Error handling and validation")
        print("  ✓ Integration with existing components")
        
    except Exception as e:
        print(f"\nError during demonstration: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()