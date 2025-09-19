"""Example demonstrating multi-modal feature fusion for rockfall prediction."""

import numpy as np
import sys
import os

# Add src to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.models.fusion.multimodal_fusion import MultiModalFusion


def create_sample_features():
    """Create sample features for different modalities."""
    # Simulate features from different extractors
    sample_features = []
    
    for i in range(20):  # Create 20 samples
        features = {
            'image': np.random.randn(512),      # CNN features from imagery
            'temporal': np.random.randn(64),    # LSTM features from sensor data
            'tabular': np.random.randn(50),     # Tabular features from environmental data
            'terrain': np.random.randn(30),     # Terrain features from DEM
            'seismic': np.random.randn(40)      # Seismic signal features
        }
        sample_features.append(features)
    
    return sample_features


def demonstrate_fusion_methods():
    """Demonstrate different fusion methods."""
    print("Multi-Modal Feature Fusion Example")
    print("=" * 50)
    
    # Create sample training data
    training_data = create_sample_features()
    print(f"Created {len(training_data)} training samples")
    
    # Test different fusion methods
    fusion_methods = ['attention', 'concatenation', 'weighted_sum']
    
    for method in fusion_methods:
        print(f"\n--- Testing {method.upper()} fusion method ---")
        
        # Configure fusion module
        config = {
            'fusion_method': method,
            'output_dim': 128,
            'normalize_features': True,
            'handle_missing': 'zero'
        }
        
        # Create and fit fusion module
        fusion = MultiModalFusion(config)
        fusion.fit(training_data)
        
        print(f"Fitted fusion module with feature dimensions:")
        for modality, dim in fusion.feature_dims.items():
            print(f"  {modality}: {dim}")
        
        # Test with complete features
        test_sample = {
            'image': np.random.randn(512),
            'temporal': np.random.randn(64),
            'tabular': np.random.randn(50),
            'terrain': np.random.randn(30),
            'seismic': np.random.randn(40)
        }
        
        fused_features = fusion.extract_features(test_sample)
        print(f"Fused features shape: {fused_features.shape}")
        print(f"Feature range: [{fused_features.min():.3f}, {fused_features.max():.3f}]")
        
        # Test with missing modalities
        partial_sample = {
            'image': np.random.randn(512),
            'temporal': np.random.randn(64)
            # Missing tabular, terrain, and seismic
        }
        
        partial_features = fusion.extract_features(partial_sample)
        print(f"Partial features shape: {partial_features.shape}")
        
        # Get attention weights (if using attention method)
        if method == 'attention':
            weights = fusion.get_attention_weights(test_sample)
            if weights:
                print("Attention weights:")
                for modality, weight in weights.items():
                    print(f"  {modality}: {weight:.3f}")
        
        # Get modality contributions
        contributions = fusion.get_modality_contributions(test_sample)
        print("Modality contributions:")
        for modality, contrib in contributions.items():
            print(f"  {modality}: {contrib:.3f}")


def demonstrate_missing_modality_handling():
    """Demonstrate handling of missing modalities."""
    print("\n" + "=" * 50)
    print("Missing Modality Handling Example")
    print("=" * 50)
    
    # Create training data
    training_data = create_sample_features()
    
    # Test different missing modality strategies
    strategies = ['zero', 'mean', 'drop']
    
    for strategy in strategies:
        print(f"\n--- Testing '{strategy}' missing modality strategy ---")
        
        config = {
            'fusion_method': 'attention',
            'output_dim': 64,
            'handle_missing': strategy
        }
        
        fusion = MultiModalFusion(config)
        fusion.fit(training_data)
        
        # Test with progressively fewer modalities
        test_cases = [
            {'image': np.random.randn(512), 'temporal': np.random.randn(64), 'tabular': np.random.randn(50)},
            {'image': np.random.randn(512), 'temporal': np.random.randn(64)},
            {'image': np.random.randn(512)},
            {}  # No modalities
        ]
        
        for i, test_case in enumerate(test_cases):
            try:
                features = fusion.extract_features(test_case)
                available_modalities = list(test_case.keys())
                print(f"  {len(available_modalities)} modalities {available_modalities}: "
                      f"shape {features.shape}, mean {features.mean():.3f}")
            except Exception as e:
                print(f"  {len(test_case)} modalities: Error - {e}")


def demonstrate_batch_processing():
    """Demonstrate batch feature extraction."""
    print("\n" + "=" * 50)
    print("Batch Processing Example")
    print("=" * 50)
    
    # Create training data
    training_data = create_sample_features()
    
    # Create fusion module
    fusion = MultiModalFusion({
        'fusion_method': 'attention',
        'output_dim': 128,
        'normalize_features': True
    })
    fusion.fit(training_data)
    
    # Create batch of test samples with varying completeness
    batch_samples = [
        {'image': np.random.randn(512), 'temporal': np.random.randn(64), 'tabular': np.random.randn(50)},
        {'image': np.random.randn(512), 'temporal': np.random.randn(64)},
        {'image': np.random.randn(512), 'tabular': np.random.randn(50), 'terrain': np.random.randn(30)},
        {'temporal': np.random.randn(64), 'seismic': np.random.randn(40)},
        {'image': np.random.randn(512)}
    ]
    
    # Process batch
    batch_features = fusion.extract_batch_features(batch_samples)
    
    print(f"Processed batch of {len(batch_samples)} samples")
    print(f"Output shape: {batch_features.shape}")
    print(f"Feature statistics:")
    print(f"  Mean: {batch_features.mean():.3f}")
    print(f"  Std:  {batch_features.std():.3f}")
    print(f"  Min:  {batch_features.min():.3f}")
    print(f"  Max:  {batch_features.max():.3f}")


def demonstrate_model_persistence():
    """Demonstrate saving and loading fusion models."""
    print("\n" + "=" * 50)
    print("Model Persistence Example")
    print("=" * 50)
    
    # Create and train fusion module
    training_data = create_sample_features()
    
    fusion = MultiModalFusion({
        'fusion_method': 'attention',
        'output_dim': 64,
        'normalize_features': True
    })
    fusion.fit(training_data)
    
    # Test sample
    test_sample = {
        'image': np.random.randn(512),
        'temporal': np.random.randn(64),
        'tabular': np.random.randn(50)
    }
    
    original_features = fusion.extract_features(test_sample)
    print(f"Original features shape: {original_features.shape}")
    
    # Save model
    model_path = 'fusion_model_example.pth'
    fusion.save_model(model_path)
    print(f"Model saved to: {model_path}")
    
    # Load model
    new_fusion = MultiModalFusion()
    new_fusion.load_model(model_path)
    print("Model loaded successfully")
    
    # Test loaded model
    loaded_features = new_fusion.extract_features(test_sample)
    print(f"Loaded features shape: {loaded_features.shape}")
    
    # Verify consistency
    print(f"Features are consistent: {np.allclose(original_features, loaded_features, rtol=1e-5)}")
    
    # Clean up
    if os.path.exists(model_path):
        os.remove(model_path)
        print("Cleaned up model file")


if __name__ == "__main__":
    # Set random seed for reproducible results
    np.random.seed(42)
    
    try:
        demonstrate_fusion_methods()
        demonstrate_missing_modality_handling()
        demonstrate_batch_processing()
        demonstrate_model_persistence()
        
        print("\n" + "=" * 50)
        print("Multi-Modal Fusion Example Completed Successfully!")
        print("=" * 50)
        
    except Exception as e:
        print(f"Error running example: {e}")
        import traceback
        traceback.print_exc()