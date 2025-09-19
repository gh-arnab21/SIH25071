"""Example usage of CNN Feature Extractor for mining imagery analysis."""

import numpy as np
from PIL import Image
import os
import sys

# Add src to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.models.extractors.cnn_feature_extractor import CNNFeatureExtractor
from src.data.schemas import ImageData, ImageMetadata, BoundingBox


def create_sample_image(width=224, height=224):
    """Create a sample mining-related image for demonstration."""
    # Create a synthetic image representing a mining slope
    image_array = np.random.randint(50, 200, (height, width, 3), dtype=np.uint8)
    
    # Add some patterns that might represent geological features
    # Horizontal layers (sedimentary rock)
    for i in range(0, height, 20):
        image_array[i:i+2, :, :] = [120, 100, 80]  # Brown layers
    
    # Add some vertical fractures
    for j in range(0, width, 50):
        image_array[:, j:j+1, :] = [60, 60, 60]  # Dark fractures
    
    return Image.fromarray(image_array)


def main():
    """Demonstrate CNN feature extraction capabilities."""
    print("CNN Feature Extractor Example")
    print("=" * 40)
    
    # 1. Initialize feature extractor with different configurations
    print("\n1. Initializing CNN Feature Extractors...")
    
    # ResNet-based extractor
    resnet_config = {
        'backbone': 'resnet50',
        'feature_dim': 256,
        'pretrained': True,
        'freeze_backbone': True,
        'device': 'cpu'  # Use CPU for demo
    }
    resnet_extractor = CNNFeatureExtractor(resnet_config)
    print(f"   ResNet50 extractor initialized - Feature dim: {resnet_extractor.feature_dim}")
    
    # EfficientNet-based extractor
    efficientnet_config = {
        'backbone': 'efficientnet_b0',
        'feature_dim': 128,
        'pretrained': True,
        'freeze_backbone': True,
        'device': 'cpu'
    }
    efficientnet_extractor = CNNFeatureExtractor(efficientnet_config)
    print(f"   EfficientNet-B0 extractor initialized - Feature dim: {efficientnet_extractor.feature_dim}")
    
    # 2. Create sample images
    print("\n2. Creating sample mining images...")
    sample_images = []
    for i in range(3):
        img = create_sample_image()
        sample_images.append(img)
        print(f"   Created sample image {i+1}: {img.size}")
    
    # 3. Extract features from individual images
    print("\n3. Extracting features from individual images...")
    for i, img in enumerate(sample_images):
        # ResNet features
        resnet_features = resnet_extractor.extract_features(img)
        print(f"   Image {i+1} - ResNet features shape: {resnet_features.shape}")
        print(f"   Image {i+1} - ResNet features range: [{resnet_features.min():.3f}, {resnet_features.max():.3f}]")
        
        # EfficientNet features
        efficientnet_features = efficientnet_extractor.extract_features(img)
        print(f"   Image {i+1} - EfficientNet features shape: {efficientnet_features.shape}")
        print(f"   Image {i+1} - EfficientNet features range: [{efficientnet_features.min():.3f}, {efficientnet_features.max():.3f}]")
    
    # 4. Demonstrate batch processing
    print("\n4. Demonstrating batch feature extraction...")
    
    # Save sample images temporarily
    temp_paths = []
    for i, img in enumerate(sample_images):
        temp_path = f"temp_image_{i}.jpg"
        img.save(temp_path)
        temp_paths.append(temp_path)
    
    # Initialize file paths
    feature_path = "sample_features.pkl"
    model_path = "sample_cnn_model.pth"
    
    try:
        # Extract features in batch
        batch_features = resnet_extractor.extract_batch_features(temp_paths)
        print(f"   Batch features shape: {batch_features.shape}")
        print(f"   Average feature magnitude: {np.mean(np.abs(batch_features)):.3f}")
        
        # 5. Demonstrate feature saving and loading
        print("\n5. Demonstrating feature persistence...")
        resnet_extractor.save_features(batch_features, feature_path)
        print(f"   Features saved to: {feature_path}")
        
        loaded_features = resnet_extractor.load_features(feature_path)
        print(f"   Loaded features shape: {loaded_features.shape}")
        print(f"   Features match: {np.allclose(batch_features, loaded_features)}")
        
        # 6. Demonstrate model saving and loading
        print("\n6. Demonstrating model persistence...")
        resnet_extractor.save_model(model_path)
        print(f"   Model saved to: {model_path}")
        
        # Create new extractor and load model
        new_extractor = CNNFeatureExtractor()
        new_extractor.load_model(model_path)
        print(f"   Model loaded - Feature dim: {new_extractor.feature_dim}")
        
        # Verify loaded model produces same features
        test_features_original = resnet_extractor.extract_features(sample_images[0])
        test_features_loaded = new_extractor.extract_features(sample_images[0])
        print(f"   Feature consistency: {np.allclose(test_features_original, test_features_loaded)}")
        
        # 7. Demonstrate feature importance analysis
        print("\n7. Demonstrating feature importance analysis...")
        importance_scores = resnet_extractor.get_feature_importance(sample_images[0])
        print(f"   Feature importance shape: {importance_scores.shape}")
        print(f"   Channel importance: R={importance_scores[0]:.3f}, G={importance_scores[1]:.3f}, B={importance_scores[2]:.3f}")
        
        # 8. Demonstrate ImageData integration
        print("\n8. Demonstrating ImageData integration...")
        metadata = ImageMetadata(width=224, height=224, channels=3)
        annotations = [
            BoundingBox(x=50, y=50, width=100, height=100, class_id=1, confidence=0.9)
        ]
        image_data = ImageData(
            image_path=temp_paths[0],
            annotations=annotations,
            metadata=metadata
        )
        
        imagedata_features = resnet_extractor.extract_features(image_data)
        print(f"   ImageData features shape: {imagedata_features.shape}")
        print(f"   Features extracted from ImageData object successfully")
        
    finally:
        # Clean up temporary files
        for path in temp_paths + [feature_path, model_path]:
            if os.path.exists(path):
                os.remove(path)
        print("\n   Temporary files cleaned up")
    
    print("\n" + "=" * 40)
    print("CNN Feature Extraction Example Complete!")
    print("\nKey capabilities demonstrated:")
    print("- Multiple backbone architectures (ResNet, EfficientNet)")
    print("- Transfer learning with pre-trained weights")
    print("- Batch processing for multiple images")
    print("- Feature persistence (save/load)")
    print("- Model persistence (save/load)")
    print("- Feature importance analysis")
    print("- Integration with data schemas")
    print("- Robust error handling")


if __name__ == "__main__":
    main()