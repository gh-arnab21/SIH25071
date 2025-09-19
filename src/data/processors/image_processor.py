"""Image preprocessing pipeline for satellite and drone imagery."""

import json
import os
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import cv2
from PIL import Image
import torch
import torchvision.transforms as transforms
from torchvision.transforms import functional as F

from ..base import BaseDataProcessor
from ..schemas import ImageData, BoundingBox, ImageMetadata


class ImagePreprocessor(BaseDataProcessor):
    """Preprocessor for satellite and drone imagery with annotation support."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize the image preprocessor.
        
        Args:
            config: Configuration dictionary with preprocessing parameters
                - target_size: Tuple of (width, height) for resizing
                - normalize: Whether to normalize pixel values
                - augment: Whether to apply data augmentation
                - mean: RGB mean values for normalization
                - std: RGB standard deviation values for normalization
        """
        super().__init__(config)
        
        # Default configuration
        default_config = {
            'target_size': (224, 224),
            'normalize': True,
            'augment': False,
            'mean': [0.485, 0.456, 0.406],  # ImageNet means
            'std': [0.229, 0.224, 0.225],   # ImageNet stds
            'max_objects': 50,  # Maximum objects to extract from annotations
            'min_object_size': 0.01,  # Minimum object size (relative to image)
        }
        
        self.config = {**default_config, **self.config}
        self.target_size = tuple(self.config['target_size'])
        
        # Initialize transforms
        self._setup_transforms()
    
    def _setup_transforms(self):
        """Setup image transformation pipelines."""
        # Basic transforms
        basic_transforms = [
            transforms.Resize(self.target_size),
            transforms.ToTensor(),
        ]
        
        if self.config['normalize']:
            basic_transforms.append(
                transforms.Normalize(
                    mean=self.config['mean'],
                    std=self.config['std']
                )
            )
        
        self.basic_transform = transforms.Compose(basic_transforms)
        
        # Augmentation transforms
        if self.config['augment']:
            augment_transforms = [
                transforms.Resize(self.target_size),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomRotation(degrees=10),
                transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
                transforms.ToTensor(),
            ]
            
            if self.config['normalize']:
                augment_transforms.append(
                    transforms.Normalize(
                        mean=self.config['mean'],
                        std=self.config['std']
                    )
                )
            
            self.augment_transform = transforms.Compose(augment_transforms)
        else:
            self.augment_transform = self.basic_transform
    
    def fit(self, data: List[ImageData]) -> 'ImagePreprocessor':
        """Fit the processor to training data.
        
        Args:
            data: List of ImageData objects
            
        Returns:
            Self for method chaining
        """
        # Calculate dataset statistics if needed
        if not self.config['normalize'] or 'mean' not in self.config:
            self._calculate_dataset_stats(data)
        
        self._is_fitted = True
        return self
    
    def transform(self, data: Union[ImageData, List[ImageData]]) -> np.ndarray:
        """Transform image data to processed features.
        
        Args:
            data: ImageData object or list of ImageData objects
            
        Returns:
            Processed image features as numpy array
        """
        if isinstance(data, list):
            return np.array([self._process_single_image(img_data) for img_data in data])
        else:
            return self._process_single_image(data)
    
    def _process_single_image(self, image_data: ImageData) -> np.ndarray:
        """Process a single image with its annotations.
        
        Args:
            image_data: ImageData object containing image path and annotations
            
        Returns:
            Processed image tensor as numpy array
        """
        # Load image
        image = self._load_image(image_data.image_path)
        
        # Apply transforms
        if self.config['augment'] and not self._is_fitted:
            processed_image = self.augment_transform(image)
        else:
            processed_image = self.basic_transform(image)
        
        return processed_image.numpy()
    
    def _load_image(self, image_path: str) -> Image.Image:
        """Load image from file path.
        
        Args:
            image_path: Path to image file
            
        Returns:
            PIL Image object
        """
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image file not found: {image_path}")
        
        image = Image.open(image_path).convert('RGB')
        return image
    
    def _calculate_dataset_stats(self, data: List[ImageData]):
        """Calculate mean and std statistics for the dataset.
        
        Args:
            data: List of ImageData objects
        """
        # Sample a subset for efficiency
        sample_size = min(100, len(data))
        sample_indices = np.random.choice(len(data), sample_size, replace=False)
        
        pixel_values = []
        
        for idx in sample_indices:
            try:
                image = self._load_image(data[idx].image_path)
                image_array = np.array(image) / 255.0  # Normalize to [0, 1]
                pixel_values.append(image_array.reshape(-1, 3))
            except Exception:
                continue
        
        if pixel_values:
            all_pixels = np.vstack(pixel_values)
            mean = np.mean(all_pixels, axis=0).tolist()
            std = np.std(all_pixels, axis=0).tolist()
            
            self.config['mean'] = mean
            self.config['std'] = std
            self._setup_transforms()
    
    def parse_json_annotations(self, annotation_path: str) -> List[BoundingBox]:
        """Parse JSON format annotations.
        
        Args:
            annotation_path: Path to JSON annotation file
            
        Returns:
            List of BoundingBox objects
        """
        if not os.path.exists(annotation_path):
            raise FileNotFoundError(f"Annotation file not found: {annotation_path}")
        
        with open(annotation_path, 'r') as f:
            annotations = json.load(f)
        
        bboxes = []
        
        # Handle different JSON annotation formats
        if 'annotations' in annotations:
            # COCO format
            for ann in annotations['annotations']:
                bbox = ann['bbox']  # [x, y, width, height]
                bboxes.append(BoundingBox(
                    x=bbox[0],
                    y=bbox[1],
                    width=bbox[2],
                    height=bbox[3],
                    class_id=ann.get('category_id', 0),
                    confidence=ann.get('score', 1.0)
                ))
        elif 'objects' in annotations:
            # Custom format with objects list
            for obj in annotations['objects']:
                bbox = obj['bbox']
                bboxes.append(BoundingBox(
                    x=bbox['x'],
                    y=bbox['y'],
                    width=bbox['width'],
                    height=bbox['height'],
                    class_id=obj.get('class_id', 0),
                    confidence=obj.get('confidence', 1.0)
                ))
        
        return bboxes
    
    def parse_yolo_annotations(self, annotation_path: str, image_width: int, image_height: int) -> List[BoundingBox]:
        """Parse YOLO format annotations.
        
        Args:
            annotation_path: Path to YOLO annotation file (.txt)
            image_width: Width of the corresponding image
            image_height: Height of the corresponding image
            
        Returns:
            List of BoundingBox objects
        """
        if not os.path.exists(annotation_path):
            raise FileNotFoundError(f"Annotation file not found: {annotation_path}")
        
        bboxes = []
        
        with open(annotation_path, 'r') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                
                parts = line.split()
                if len(parts) < 5:
                    continue
                
                class_id = int(parts[0])
                x_center = float(parts[1])
                y_center = float(parts[2])
                width = float(parts[3])
                height = float(parts[4])
                confidence = float(parts[5]) if len(parts) > 5 else 1.0
                
                # Convert from YOLO format (normalized center coordinates) to absolute coordinates
                x = (x_center - width / 2) * image_width
                y = (y_center - height / 2) * image_height
                abs_width = width * image_width
                abs_height = height * image_height
                
                bboxes.append(BoundingBox(
                    x=x,
                    y=y,
                    width=abs_width,
                    height=abs_height,
                    class_id=class_id,
                    confidence=confidence
                ))
        
        return bboxes
    
    def extract_object_features(self, image_data: ImageData) -> Dict[str, np.ndarray]:
        """Extract features from object detection annotations.
        
        Args:
            image_data: ImageData object with annotations
            
        Returns:
            Dictionary containing extracted object features
        """
        if not image_data.annotations:
            # Pad position features to fixed size
            max_positions = self.config['max_objects'] * 2
            return {
                'object_count': np.array([0]),
                'object_areas': np.array([]),
                'object_positions': np.array([0.0] * max_positions),
                'class_distribution': np.zeros(10, dtype=int)
            }
        
        # Load image to get dimensions
        image = self._load_image(image_data.image_path)
        img_width, img_height = image.size
        
        # Extract features from annotations
        object_count = len(image_data.annotations)
        object_areas = []
        object_positions = []
        class_ids = []
        
        for bbox in image_data.annotations:
            # Calculate relative area
            area = (bbox.width * bbox.height) / (img_width * img_height)
            object_areas.append(area)
            
            # Calculate relative center position
            center_x = (bbox.x + bbox.width / 2) / img_width
            center_y = (bbox.y + bbox.height / 2) / img_height
            object_positions.extend([center_x, center_y])
            
            class_ids.append(bbox.class_id)
        
        # Create class distribution (assuming max 10 classes)
        if class_ids:
            class_distribution = np.bincount(class_ids, minlength=10)
        else:
            class_distribution = np.zeros(10, dtype=int)
        
        # Pad or truncate position features to fixed size
        max_positions = self.config['max_objects'] * 2  # x, y for each object
        if len(object_positions) > max_positions:
            object_positions = object_positions[:max_positions]
        else:
            object_positions.extend([0.0] * (max_positions - len(object_positions)))
        
        return {
            'object_count': np.array([object_count]),
            'object_areas': np.array(object_areas),
            'object_positions': np.array(object_positions),
            'class_distribution': class_distribution
        }
    
    def resize_image(self, image: Union[np.ndarray, Image.Image], target_size: Optional[Tuple[int, int]] = None) -> np.ndarray:
        """Resize image to target dimensions.
        
        Args:
            image: Input image as numpy array or PIL Image
            target_size: Target size as (width, height). Uses config if None.
            
        Returns:
            Resized image as numpy array
        """
        if target_size is None:
            target_size = self.target_size
        
        if isinstance(image, np.ndarray):
            # Convert numpy array to PIL for consistent resizing
            if image.dtype != np.uint8:
                image = (image * 255).astype(np.uint8)
            image = Image.fromarray(image)
        
        resized = image.resize(target_size, Image.Resampling.LANCZOS)
        return np.array(resized)
    
    def normalize_image(self, image: np.ndarray) -> np.ndarray:
        """Normalize image pixel values.
        
        Args:
            image: Input image as numpy array
            
        Returns:
            Normalized image
        """
        # Convert to float and normalize to [0, 1]
        normalized = image.astype(np.float32) / 255.0
        
        if self.config['normalize']:
            mean = np.array(self.config['mean'], dtype=np.float32)
            std = np.array(self.config['std'], dtype=np.float32)
            normalized = (normalized - mean) / std
        
        return normalized
    
    def augment_image(self, image: np.ndarray) -> np.ndarray:
        """Apply data augmentation to image.
        
        Args:
            image: Input image as numpy array
            
        Returns:
            Augmented image
        """
        # Convert to PIL for augmentation
        if image.dtype != np.uint8:
            image = (image * 255).astype(np.uint8)
        
        pil_image = Image.fromarray(image)
        
        # Apply random augmentations
        if np.random.random() > 0.5:
            pil_image = F.hflip(pil_image)
        
        if np.random.random() > 0.5:
            angle = np.random.uniform(-10, 10)
            pil_image = F.rotate(pil_image, angle)
        
        if np.random.random() > 0.5:
            brightness = np.random.uniform(0.8, 1.2)
            pil_image = F.adjust_brightness(pil_image, brightness)
        
        return np.array(pil_image)