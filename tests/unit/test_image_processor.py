"""Unit tests for image preprocessing functionality."""

import json
import os
import tempfile
import unittest
from unittest.mock import patch, MagicMock
import numpy as np
from PIL import Image
import pytest

from src.data.processors.image_processor import ImagePreprocessor
from src.data.schemas import ImageData, BoundingBox, ImageMetadata


class TestImagePreprocessor(unittest.TestCase):
    """Test cases for ImagePreprocessor class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.processor = ImagePreprocessor()
        
        # Create test image
        self.test_image = Image.new('RGB', (100, 100), color='red')
        self.test_image_path = os.path.join(self.temp_dir, 'test_image.jpg')
        self.test_image.save(self.test_image_path)
        
        # Create test annotations
        self.test_annotations = [
            BoundingBox(x=10, y=10, width=20, height=20, class_id=1, confidence=0.9),
            BoundingBox(x=50, y=50, width=30, height=25, class_id=2, confidence=0.8)
        ]
        
        # Create test ImageData
        self.test_metadata = ImageMetadata(width=100, height=100, channels=3)
        self.test_image_data = ImageData(
            image_path=self.test_image_path,
            annotations=self.test_annotations,
            metadata=self.test_metadata
        )
    
    def tearDown(self):
        """Clean up test fixtures."""
        import shutil
        shutil.rmtree(self.temp_dir)
    
    def test_init_default_config(self):
        """Test ImagePreprocessor initialization with default config."""
        processor = ImagePreprocessor()
        
        self.assertEqual(processor.target_size, (224, 224))
        self.assertTrue(processor.config['normalize'])
        self.assertFalse(processor.config['augment'])
        self.assertEqual(processor.config['mean'], [0.485, 0.456, 0.406])
        self.assertEqual(processor.config['std'], [0.229, 0.224, 0.225])
    
    def test_init_custom_config(self):
        """Test ImagePreprocessor initialization with custom config."""
        config = {
            'target_size': (128, 128),
            'normalize': False,
            'augment': True
        }
        processor = ImagePreprocessor(config)
        
        self.assertEqual(processor.target_size, (128, 128))
        self.assertFalse(processor.config['normalize'])
        self.assertTrue(processor.config['augment'])
    
    def test_load_image_success(self):
        """Test successful image loading."""
        image = self.processor._load_image(self.test_image_path)
        
        self.assertIsInstance(image, Image.Image)
        self.assertEqual(image.mode, 'RGB')
        self.assertEqual(image.size, (100, 100))
    
    def test_load_image_file_not_found(self):
        """Test image loading with non-existent file."""
        with self.assertRaises(FileNotFoundError):
            self.processor._load_image('non_existent_image.jpg')
    
    def test_process_single_image(self):
        """Test processing a single image."""
        result = self.processor._process_single_image(self.test_image_data)
        
        self.assertIsInstance(result, np.ndarray)
        self.assertEqual(result.shape, (3, 224, 224))  # CHW format
        self.assertTrue(np.all(result >= -3))  # Normalized values should be reasonable
        self.assertTrue(np.all(result <= 3))
    
    def test_transform_single_image(self):
        """Test transform method with single image."""
        result = self.processor.transform(self.test_image_data)
        
        self.assertIsInstance(result, np.ndarray)
        self.assertEqual(result.shape, (3, 224, 224))
    
    def test_transform_multiple_images(self):
        """Test transform method with multiple images."""
        # Create second test image
        test_image2_path = os.path.join(self.temp_dir, 'test_image2.jpg')
        self.test_image.save(test_image2_path)
        
        test_image_data2 = ImageData(
            image_path=test_image2_path,
            annotations=[],
            metadata=self.test_metadata
        )
        
        image_list = [self.test_image_data, test_image_data2]
        result = self.processor.transform(image_list)
        
        self.assertIsInstance(result, np.ndarray)
        self.assertEqual(result.shape, (2, 3, 224, 224))
    
    def test_fit_method(self):
        """Test fit method."""
        image_list = [self.test_image_data]
        result = self.processor.fit(image_list)
        
        self.assertIs(result, self.processor)
        self.assertTrue(self.processor.is_fitted)
    
    def test_fit_transform(self):
        """Test fit_transform method."""
        image_list = [self.test_image_data]
        result = self.processor.fit_transform(image_list)
        
        self.assertIsInstance(result, np.ndarray)
        self.assertTrue(self.processor.is_fitted)
    
    def test_parse_json_annotations_coco_format(self):
        """Test parsing COCO format JSON annotations."""
        # Create COCO format annotation file
        coco_annotations = {
            "annotations": [
                {
                    "bbox": [10, 20, 30, 40],
                    "category_id": 1,
                    "score": 0.95
                },
                {
                    "bbox": [50, 60, 25, 35],
                    "category_id": 2,
                    "score": 0.87
                }
            ]
        }
        
        annotation_path = os.path.join(self.temp_dir, 'coco_annotations.json')
        with open(annotation_path, 'w') as f:
            json.dump(coco_annotations, f)
        
        bboxes = self.processor.parse_json_annotations(annotation_path)
        
        self.assertEqual(len(bboxes), 2)
        self.assertEqual(bboxes[0].x, 10)
        self.assertEqual(bboxes[0].y, 20)
        self.assertEqual(bboxes[0].width, 30)
        self.assertEqual(bboxes[0].height, 40)
        self.assertEqual(bboxes[0].class_id, 1)
        self.assertEqual(bboxes[0].confidence, 0.95)
    
    def test_parse_json_annotations_custom_format(self):
        """Test parsing custom format JSON annotations."""
        # Create custom format annotation file
        custom_annotations = {
            "objects": [
                {
                    "bbox": {"x": 15, "y": 25, "width": 35, "height": 45},
                    "class_id": 3,
                    "confidence": 0.92
                }
            ]
        }
        
        annotation_path = os.path.join(self.temp_dir, 'custom_annotations.json')
        with open(annotation_path, 'w') as f:
            json.dump(custom_annotations, f)
        
        bboxes = self.processor.parse_json_annotations(annotation_path)
        
        self.assertEqual(len(bboxes), 1)
        self.assertEqual(bboxes[0].x, 15)
        self.assertEqual(bboxes[0].y, 25)
        self.assertEqual(bboxes[0].width, 35)
        self.assertEqual(bboxes[0].height, 45)
        self.assertEqual(bboxes[0].class_id, 3)
        self.assertEqual(bboxes[0].confidence, 0.92)
    
    def test_parse_json_annotations_file_not_found(self):
        """Test JSON annotation parsing with non-existent file."""
        with self.assertRaises(FileNotFoundError):
            self.processor.parse_json_annotations('non_existent.json')
    
    def test_parse_yolo_annotations(self):
        """Test parsing YOLO format annotations."""
        # Create YOLO format annotation file
        yolo_content = "1 0.5 0.6 0.2 0.3 0.95\n2 0.3 0.4 0.15 0.25 0.87\n"
        
        annotation_path = os.path.join(self.temp_dir, 'yolo_annotations.txt')
        with open(annotation_path, 'w') as f:
            f.write(yolo_content)
        
        bboxes = self.processor.parse_yolo_annotations(annotation_path, 200, 150)
        
        self.assertEqual(len(bboxes), 2)
        
        # First bbox: center (0.5, 0.6), size (0.2, 0.3) in 200x150 image
        # x = (0.5 - 0.2/2) * 200 = 80, y = (0.6 - 0.3/2) * 150 = 67.5
        self.assertEqual(bboxes[0].x, 80.0)
        self.assertEqual(bboxes[0].y, 67.5)
        self.assertEqual(bboxes[0].width, 40.0)  # 0.2 * 200
        self.assertEqual(bboxes[0].height, 45.0)  # 0.3 * 150
        self.assertEqual(bboxes[0].class_id, 1)
        self.assertEqual(bboxes[0].confidence, 0.95)
    
    def test_parse_yolo_annotations_file_not_found(self):
        """Test YOLO annotation parsing with non-existent file."""
        with self.assertRaises(FileNotFoundError):
            self.processor.parse_yolo_annotations('non_existent.txt', 100, 100)
    
    def test_extract_object_features_with_annotations(self):
        """Test object feature extraction with annotations."""
        features = self.processor.extract_object_features(self.test_image_data)
        
        self.assertIn('object_count', features)
        self.assertIn('object_areas', features)
        self.assertIn('object_positions', features)
        self.assertIn('class_distribution', features)
        
        self.assertEqual(features['object_count'][0], 2)
        self.assertEqual(len(features['object_areas']), 2)
        
        # Check relative areas (bbox area / image area)
        expected_area1 = (20 * 20) / (100 * 100)  # 0.04
        expected_area2 = (30 * 25) / (100 * 100)  # 0.075
        np.testing.assert_almost_equal(features['object_areas'][0], expected_area1)
        np.testing.assert_almost_equal(features['object_areas'][1], expected_area2)
        
        # Check class distribution
        self.assertEqual(features['class_distribution'][1], 1)  # One object of class 1
        self.assertEqual(features['class_distribution'][2], 1)  # One object of class 2
    
    def test_extract_object_features_no_annotations(self):
        """Test object feature extraction with no annotations."""
        image_data_no_ann = ImageData(
            image_path=self.test_image_path,
            annotations=[],
            metadata=self.test_metadata
        )
        
        features = self.processor.extract_object_features(image_data_no_ann)
        
        self.assertEqual(features['object_count'][0], 0)
        self.assertEqual(len(features['object_areas']), 0)
        self.assertEqual(len(features['class_distribution']), 10)  # Default 10 classes
    
    def test_resize_image_numpy(self):
        """Test image resizing with numpy array input."""
        image_array = np.random.randint(0, 255, (50, 60, 3), dtype=np.uint8)
        resized = self.processor.resize_image(image_array, (128, 96))
        
        self.assertEqual(resized.shape, (96, 128, 3))  # Height, Width, Channels
    
    def test_resize_image_pil(self):
        """Test image resizing with PIL Image input."""
        pil_image = Image.new('RGB', (80, 70), color='blue')
        resized = self.processor.resize_image(pil_image, (64, 48))
        
        self.assertEqual(resized.shape, (48, 64, 3))
    
    def test_normalize_image(self):
        """Test image normalization."""
        image_array = np.random.randint(0, 255, (32, 32, 3), dtype=np.uint8)
        normalized = self.processor.normalize_image(image_array)
        
        self.assertEqual(normalized.dtype, np.float32)
        self.assertTrue(np.all(normalized >= -3))  # Should be within reasonable range
        self.assertTrue(np.all(normalized <= 3))
    
    def test_normalize_image_no_normalization(self):
        """Test image normalization when normalization is disabled."""
        config = {'normalize': False}
        processor = ImagePreprocessor(config)
        
        image_array = np.random.randint(0, 255, (32, 32, 3), dtype=np.uint8)
        normalized = processor.normalize_image(image_array)
        
        self.assertTrue(np.all(normalized >= 0))
        self.assertTrue(np.all(normalized <= 1))
    
    def test_augment_image(self):
        """Test image augmentation."""
        image_array = np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)
        
        # Test multiple times to ensure randomness works
        augmented_images = []
        for _ in range(5):
            augmented = self.processor.augment_image(image_array.copy())
            augmented_images.append(augmented)
        
        # Check that augmented images have correct shape
        for aug_img in augmented_images:
            self.assertEqual(aug_img.shape, (64, 64, 3))
            self.assertEqual(aug_img.dtype, np.uint8)
    
    def test_calculate_dataset_stats(self):
        """Test dataset statistics calculation."""
        # Create multiple test images
        image_paths = []
        for i in range(3):
            img_path = os.path.join(self.temp_dir, f'test_img_{i}.jpg')
            test_img = Image.new('RGB', (50, 50), color=(i*50, i*60, i*70))
            test_img.save(img_path)
            image_paths.append(img_path)
        
        image_data_list = [
            ImageData(path, [], ImageMetadata(50, 50, 3)) 
            for path in image_paths
        ]
        
        # Test with config that doesn't have mean/std
        config = {'normalize': True}
        processor = ImagePreprocessor(config)
        processor._calculate_dataset_stats(image_data_list)
        
        self.assertIn('mean', processor.config)
        self.assertIn('std', processor.config)
        self.assertEqual(len(processor.config['mean']), 3)
        self.assertEqual(len(processor.config['std']), 3)


class TestImageProcessorIntegration(unittest.TestCase):
    """Integration tests for ImagePreprocessor."""
    
    def setUp(self):
        """Set up integration test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        
        # Create test images with different sizes and colors
        self.test_images = []
        for i in range(3):
            img = Image.new('RGB', (100 + i*10, 80 + i*5), color=(i*80, i*60, i*40))
            img_path = os.path.join(self.temp_dir, f'test_image_{i}.jpg')
            img.save(img_path)
            
            # Create corresponding annotations
            annotations = [
                BoundingBox(x=10+i*5, y=10+i*3, width=20, height=15, class_id=i, confidence=0.9-i*0.1)
            ]
            
            metadata = ImageMetadata(width=100+i*10, height=80+i*5, channels=3)
            image_data = ImageData(image_path=img_path, annotations=annotations, metadata=metadata)
            self.test_images.append(image_data)
    
    def tearDown(self):
        """Clean up integration test fixtures."""
        import shutil
        shutil.rmtree(self.temp_dir)
    
    def test_end_to_end_processing(self):
        """Test complete end-to-end image processing pipeline."""
        processor = ImagePreprocessor({
            'target_size': (128, 128),
            'normalize': True,
            'augment': False
        })
        
        # Fit and transform
        processed_images = processor.fit_transform(self.test_images)
        
        # Verify output
        self.assertEqual(processed_images.shape, (3, 3, 128, 128))
        self.assertTrue(processor.is_fitted)
        
        # Test individual feature extraction
        for image_data in self.test_images:
            features = processor.extract_object_features(image_data)
            self.assertGreater(features['object_count'][0], 0)
    
    def test_batch_processing_with_augmentation(self):
        """Test batch processing with augmentation enabled."""
        processor = ImagePreprocessor({
            'target_size': (96, 96),
            'normalize': True,
            'augment': True
        })
        
        # Process images
        processed_images = processor.fit_transform(self.test_images)
        
        # Verify output shape and properties
        self.assertEqual(processed_images.shape, (3, 3, 96, 96))
        self.assertTrue(np.all(np.isfinite(processed_images)))


if __name__ == '__main__':
    unittest.main()