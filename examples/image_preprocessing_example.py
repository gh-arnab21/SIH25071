"""Example usage of ImagePreprocessor for rockfall prediction system."""

import os
import sys
import numpy as np
from PIL import Image

# Add src to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from data.processors.image_processor import ImagePreprocessor
from data.schemas import ImageData, BoundingBox, ImageMetadata


def create_sample_data():
    """Create sample image data for demonstration."""
    # Create a sample image
    sample_image = Image.new('RGB', (200, 150), color=(100, 150, 200))
    image_path = 'sample_image.jpg'
    sample_image.save(image_path)
    
    # Create sample annotations
    annotations = [
        BoundingBox(x=20, y=30, width=40, height=35, class_id=1, confidence=0.95),
        BoundingBox(x=100, y=80, width=50, height=40, class_id=2, confidence=0.87),
        BoundingBox(x=150, y=20, width=30, height=25, class_id=1, confidence=0.92)
    ]
    
    # Create metadata
    metadata = ImageMetadata(width=200, height=150, channels=3)
    
    # Create ImageData object
    image_data = ImageData(
        image_path=image_path,
        annotations=annotations,
        metadata=metadata
    )
    
    return image_data, image_path


def main():
    """Demonstrate ImagePreprocessor functionality."""
    print("ImagePreprocessor Example for Rockfall Prediction System")
    print("=" * 60)
    
    # Create sample data
    image_data, image_path = create_sample_data()
    
    # Initialize ImagePreprocessor with custom configuration
    config = {
        'target_size': (128, 128),
        'normalize': True,
        'augment': False,
        'mean': [0.485, 0.456, 0.406],
        'std': [0.229, 0.224, 0.225]
    }
    
    processor = ImagePreprocessor(config)
    
    print(f"1. Created ImagePreprocessor with config:")
    print(f"   - Target size: {processor.target_size}")
    print(f"   - Normalization: {processor.config['normalize']}")
    print(f"   - Augmentation: {processor.config['augment']}")
    print()
    
    # Process single image
    print("2. Processing single image...")
    processed_image = processor.transform(image_data)
    print(f"   - Input image size: {image_data.metadata.width}x{image_data.metadata.height}")
    print(f"   - Processed image shape: {processed_image.shape}")
    print(f"   - Value range: [{processed_image.min():.3f}, {processed_image.max():.3f}]")
    print()
    
    # Extract object features
    print("3. Extracting object features...")
    features = processor.extract_object_features(image_data)
    print(f"   - Object count: {features['object_count'][0]}")
    print(f"   - Object areas: {features['object_areas']}")
    print(f"   - Class distribution: {features['class_distribution']}")
    print(f"   - Position features length: {len(features['object_positions'])}")
    print()
    
    # Demonstrate annotation parsing
    print("4. Creating sample YOLO annotation file...")
    yolo_content = "1 0.5 0.6 0.2 0.3 0.95\n2 0.3 0.4 0.15 0.25 0.87\n"
    with open('sample_yolo.txt', 'w') as f:
        f.write(yolo_content)
    
    yolo_bboxes = processor.parse_yolo_annotations('sample_yolo.txt', 200, 150)
    print(f"   - Parsed {len(yolo_bboxes)} YOLO annotations")
    for i, bbox in enumerate(yolo_bboxes):
        print(f"     Bbox {i+1}: x={bbox.x:.1f}, y={bbox.y:.1f}, "
              f"w={bbox.width:.1f}, h={bbox.height:.1f}, class={bbox.class_id}")
    print()
    
    # Demonstrate batch processing
    print("5. Batch processing multiple images...")
    # Create additional sample images
    image_list = [image_data]
    for i in range(2):
        img = Image.new('RGB', (180 + i*20, 140 + i*10), color=(80 + i*30, 120 + i*20, 180 + i*15))
        img_path = f'sample_image_{i+2}.jpg'
        img.save(img_path)
        
        img_data = ImageData(
            image_path=img_path,
            annotations=[BoundingBox(x=10+i*5, y=15+i*3, width=25, height=20, class_id=i+1, confidence=0.9)],
            metadata=ImageMetadata(width=180+i*20, height=140+i*10, channels=3)
        )
        image_list.append(img_data)
    
    # Fit and transform batch
    batch_processed = processor.fit_transform(image_list)
    print(f"   - Processed {len(image_list)} images")
    print(f"   - Batch output shape: {batch_processed.shape}")
    print(f"   - Processor fitted: {processor.is_fitted}")
    print()
    
    # Demonstrate image utilities
    print("6. Testing image utility functions...")
    sample_array = np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)
    
    # Resize
    resized = processor.resize_image(sample_array, (96, 96))
    print(f"   - Resized {sample_array.shape} -> {resized.shape}")
    
    # Normalize
    normalized = processor.normalize_image(sample_array)
    print(f"   - Normalized range: [{normalized.min():.3f}, {normalized.max():.3f}]")
    
    # Augment
    augmented = processor.augment_image(sample_array)
    print(f"   - Augmented shape: {augmented.shape}")
    print()
    
    print("7. Cleanup...")
    # Clean up created files
    cleanup_files = ['sample_image.jpg', 'sample_yolo.txt', 'sample_image_2.jpg', 'sample_image_3.jpg']
    for file in cleanup_files:
        if os.path.exists(file):
            os.remove(file)
            print(f"   - Removed {file}")
    
    print("\nImagePreprocessor demonstration completed successfully!")


if __name__ == '__main__':
    main()