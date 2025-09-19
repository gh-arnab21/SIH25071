"""Example usage of LSTM Temporal Feature Extractor for sensor data processing."""

import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import sys
import os

# Add src to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.models.extractors.lstm_temporal_extractor import LSTMTemporalExtractor
from src.data.schemas import SensorData, TimeSeries


def create_synthetic_sensor_data(n_samples=5, n_timesteps=200):
    """Create synthetic sensor data with different patterns."""
    sensor_data_list = []
    
    for i in range(n_samples):
        # Create timestamps
        timestamps = np.arange(n_timesteps)
        
        # Create synthetic patterns
        # Pattern 1: Gradual increase with noise (potential precursor)
        if i < 2:
            displacement_values = np.linspace(0, 5, n_timesteps) + np.random.normal(0, 0.2, n_timesteps)
            strain_values = np.linspace(0, 3, n_timesteps) + np.random.normal(0, 0.1, n_timesteps)
            pressure_values = np.linspace(1, 4, n_timesteps) + np.random.normal(0, 0.15, n_timesteps)
        
        # Pattern 2: Oscillating with increasing amplitude
        elif i < 4:
            t = np.linspace(0, 4*np.pi, n_timesteps)
            displacement_values = (1 + 0.5*t/np.max(t)) * np.sin(t) + np.random.normal(0, 0.1, n_timesteps)
            strain_values = (0.5 + 0.3*t/np.max(t)) * np.cos(t) + np.random.normal(0, 0.05, n_timesteps)
            pressure_values = 2 + 0.5*np.sin(2*t) + np.random.normal(0, 0.1, n_timesteps)
        
        # Pattern 3: Stable with sudden spike (anomaly)
        else:
            displacement_values = np.random.normal(1, 0.1, n_timesteps)
            strain_values = np.random.normal(0.5, 0.05, n_timesteps)
            pressure_values = np.random.normal(2, 0.1, n_timesteps)
            
            # Add sudden spike
            spike_start = n_timesteps // 2
            spike_end = spike_start + 20
            displacement_values[spike_start:spike_end] += 3
            strain_values[spike_start:spike_end] += 2
            pressure_values[spike_start:spike_end] += 1.5
        
        # Create TimeSeries objects
        displacement = TimeSeries(
            timestamps=timestamps,
            values=displacement_values,
            unit='mm',
            sampling_rate=1.0
        )
        
        strain = TimeSeries(
            timestamps=timestamps,
            values=strain_values,
            unit='microstrain',
            sampling_rate=1.0
        )
        
        pore_pressure = TimeSeries(
            timestamps=timestamps,
            values=pressure_values,
            unit='kPa',
            sampling_rate=1.0
        )
        
        sensor_data = SensorData(
            displacement=displacement,
            strain=strain,
            pore_pressure=pore_pressure
        )
        
        sensor_data_list.append(sensor_data)
    
    return sensor_data_list


def demonstrate_temporal_feature_extraction():
    """Demonstrate the LSTM temporal feature extractor."""
    print("LSTM Temporal Feature Extractor Example")
    print("=" * 50)
    
    # Create synthetic sensor data
    print("1. Creating synthetic sensor data...")
    sensor_data_list = create_synthetic_sensor_data(n_samples=5, n_timesteps=150)
    print(f"   Created {len(sensor_data_list)} sensor data samples")
    
    # Initialize the extractor
    print("\n2. Initializing LSTM Temporal Feature Extractor...")
    config = {
        'sequence_length': 30,
        'hidden_size': 64,
        'num_layers': 2,
        'feature_dim': 32,
        'dropout': 0.2,
        'bidirectional': True,
        'network_type': 'lstm',
        'device': 'cpu',  # Use CPU for example
        'scaler_type': 'standard'
    }
    
    extractor = LSTMTemporalExtractor(config)
    print(f"   Extractor initialized with feature dimension: {extractor.feature_dim}")
    
    # Fit the scaler
    print("\n3. Fitting data scaler...")
    extractor.fit_scaler(sensor_data_list)
    print("   Scaler fitted successfully")
    
    # Extract features from individual samples
    print("\n4. Extracting features from individual samples...")
    individual_features = []
    for i, sensor_data in enumerate(sensor_data_list):
        features = extractor.extract_features(sensor_data)
        individual_features.append(features)
        print(f"   Sample {i+1}: Feature vector shape = {features.shape}, "
              f"Mean = {np.mean(features):.4f}, Std = {np.std(features):.4f}")
    
    # Extract features in batch
    print("\n5. Extracting features in batch...")
    batch_features = extractor.extract_batch_features(sensor_data_list)
    print(f"   Batch features shape: {batch_features.shape}")
    
    # Identify precursor patterns
    print("\n6. Analyzing precursor patterns...")
    for i, sensor_data in enumerate(sensor_data_list):
        patterns = extractor.identify_precursor_patterns(sensor_data, threshold=0.5)
        
        print(f"\n   Sample {i+1} Pattern Analysis:")
        print(f"     Max pattern score: {patterns['max_pattern_score']:.4f}")
        print(f"     Mean pattern score: {patterns['mean_pattern_score']:.4f}")
        print(f"     Number of anomalies: {len(patterns['anomaly_indices'])}")
        
        # Print trend analysis
        disp_trend = patterns['displacement_trend']
        print(f"     Displacement trend - Slope: {disp_trend['slope']:.4f}, "
              f"Volatility: {disp_trend['volatility']:.4f}")
        
        strain_trend = patterns['strain_trend']
        print(f"     Strain trend - Slope: {strain_trend['slope']:.4f}, "
              f"Volatility: {strain_trend['volatility']:.4f}")
    
    # Visualize some results
    print("\n7. Creating visualizations...")
    create_visualizations(sensor_data_list, batch_features, extractor)
    
    # Test model saving and loading
    print("\n8. Testing model persistence...")
    model_path = "models/saved/lstm_temporal_extractor_example.pth"
    
    # Save model
    extractor.save_model(model_path)
    print(f"   Model saved to: {model_path}")
    
    # Load model
    new_extractor = LSTMTemporalExtractor()
    new_extractor.load_model(model_path)
    print("   Model loaded successfully")
    
    # Verify loaded model works
    test_features = new_extractor.extract_features(sensor_data_list[0])
    original_features = extractor.extract_features(sensor_data_list[0])
    
    if np.allclose(test_features, original_features):
        print("   ✓ Loaded model produces identical results")
    else:
        print("   ✗ Loaded model results differ from original")
    
    print("\n" + "=" * 50)
    print("LSTM Temporal Feature Extraction Example Complete!")


def create_visualizations(sensor_data_list, batch_features, extractor):
    """Create visualizations of the results."""
    try:
        # Plot 1: Raw sensor data
        fig, axes = plt.subplots(3, 1, figsize=(12, 10))
        
        for i, sensor_data in enumerate(sensor_data_list[:3]):  # Plot first 3 samples
            # Displacement
            axes[0].plot(sensor_data.displacement.timestamps, 
                        sensor_data.displacement.values, 
                        label=f'Sample {i+1}', alpha=0.7)
            axes[0].set_title('Displacement Over Time')
            axes[0].set_ylabel('Displacement (mm)')
            axes[0].legend()
            axes[0].grid(True, alpha=0.3)
            
            # Strain
            axes[1].plot(sensor_data.strain.timestamps, 
                        sensor_data.strain.values, 
                        label=f'Sample {i+1}', alpha=0.7)
            axes[1].set_title('Strain Over Time')
            axes[1].set_ylabel('Strain (microstrain)')
            axes[1].legend()
            axes[1].grid(True, alpha=0.3)
            
            # Pore pressure
            axes[2].plot(sensor_data.pore_pressure.timestamps, 
                        sensor_data.pore_pressure.values, 
                        label=f'Sample {i+1}', alpha=0.7)
            axes[2].set_title('Pore Pressure Over Time')
            axes[2].set_ylabel('Pressure (kPa)')
            axes[2].set_xlabel('Time')
            axes[2].legend()
            axes[2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('sensor_data_visualization.png', dpi=150, bbox_inches='tight')
        print("   Sensor data visualization saved as 'sensor_data_visualization.png'")
        
        # Plot 2: Feature space visualization (PCA)
        from sklearn.decomposition import PCA
        
        if batch_features.shape[1] > 2:
            pca = PCA(n_components=2)
            features_2d = pca.fit_transform(batch_features)
            
            plt.figure(figsize=(10, 6))
            scatter = plt.scatter(features_2d[:, 0], features_2d[:, 1], 
                                c=range(len(features_2d)), cmap='viridis', s=100)
            plt.colorbar(scatter, label='Sample Index')
            plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2%} variance)')
            plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2%} variance)')
            plt.title('LSTM Features in 2D Space (PCA)')
            plt.grid(True, alpha=0.3)
            
            # Annotate points
            for i, (x, y) in enumerate(features_2d):
                plt.annotate(f'S{i+1}', (x, y), xytext=(5, 5), 
                           textcoords='offset points', fontsize=10)
            
            plt.savefig('lstm_features_pca.png', dpi=150, bbox_inches='tight')
            print("   Feature PCA visualization saved as 'lstm_features_pca.png'")
        
        # Plot 3: Pattern scores
        plt.figure(figsize=(12, 6))
        
        pattern_scores_all = []
        sample_labels = []
        
        for i, sensor_data in enumerate(sensor_data_list):
            patterns = extractor.identify_precursor_patterns(sensor_data)
            pattern_scores = patterns['pattern_scores']
            
            plt.subplot(1, 2, 1)
            plt.plot(pattern_scores, label=f'Sample {i+1}', alpha=0.7)
            
            pattern_scores_all.extend(pattern_scores)
            sample_labels.extend([i] * len(pattern_scores))
        
        plt.subplot(1, 2, 1)
        plt.title('Pattern Scores Over Sequences')
        plt.xlabel('Sequence Index')
        plt.ylabel('Pattern Score')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Box plot of pattern scores by sample
        plt.subplot(1, 2, 2)
        sample_scores = [extractor.identify_precursor_patterns(sd)['pattern_scores'] 
                        for sd in sensor_data_list]
        plt.boxplot(sample_scores, labels=[f'S{i+1}' for i in range(len(sample_scores))])
        plt.title('Pattern Score Distribution by Sample')
        plt.xlabel('Sample')
        plt.ylabel('Pattern Score')
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('pattern_analysis.png', dpi=150, bbox_inches='tight')
        print("   Pattern analysis visualization saved as 'pattern_analysis.png'")
        
        plt.close('all')  # Close all figures to free memory
        
    except ImportError:
        print("   Matplotlib not available for visualization")
    except Exception as e:
        print(f"   Visualization error: {e}")


if __name__ == "__main__":
    demonstrate_temporal_feature_extraction()