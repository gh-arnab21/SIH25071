#!/usr/bin/env python3
"""
Mining datasets preprocessing script for rockfall prediction system.
Handles Open Pit Mine Object Detection and Segmentation datasets.
"""

import os
import sys
import json
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional
import numpy as np
import cv2
from PIL import Image
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
import matplotlib.pyplot as plt

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

from src.utils.logging import setup_logging


class MiningDatasetPreprocessor:
    """Preprocesses mining datasets for rockfall prediction."""
    
    def __init__(self, raw_data_dir: str = "data/raw", processed_data_dir: str = "data/processed"):
        self.raw_data_dir = Path(raw_data_dir)
        self.processed_data_dir = Path(processed_data_dir)
        self.processed_data_dir.mkdir(parents=True, exist_ok=True)
        
        # Setup logging
        setup_logging()
        self.logger = logging.getLogger(__name__)
        
        # Dataset paths
        self.object_detection_path = self.raw_data_dir / "Open-Pit-Mine-Object-Detection-Dataset" / "Open-Pit-Mine-Object-Detection-Dataset"
        self.segmentation_path = self.raw_data_dir / "An open-pit mine segmentation dataset for deep learning" / "An open-pit mine segmentation dataset for deep learning"
        
        # Image processing parameters
        self.target_image_size = (224, 224)  # Standard CNN input size
        self.max_samples_per_dataset = 1000  # Limit for processing speed
        
    def preprocess_object_detection_dataset(self) -> Dict[str, Any]:
        """Preprocess the Object Detection Dataset."""
        self.logger.info("🔍 Preprocessing Open Pit Mine Object Detection Dataset...")
        
        images_dir = self.object_detection_path / "images"
        annotations_dir = self.object_detection_path / "annotation"
        
        # Get list of images and annotations
        image_files = sorted(list(images_dir.glob("*.jpg")))
        annotation_files = sorted(list(annotations_dir.glob("*.json")))
        
        # Limit samples for processing
        if len(image_files) > self.max_samples_per_dataset:
            image_files = image_files[:self.max_samples_per_dataset]
            annotation_files = annotation_files[:self.max_samples_per_dataset]
        
        self.logger.info(f"Processing {len(image_files)} object detection samples...")
        
        # Extract features and labels
        features = []
        labels = []
        metadata = []
        
        for img_file, ann_file in zip(image_files, annotation_files):
            try:
                # Load and process image
                image = cv2.imread(str(img_file))
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                image_resized = cv2.resize(image, self.target_image_size)
                
                # Load annotation
                with open(ann_file, 'r') as f:
                    annotation = json.load(f)
                
                # Extract image features
                img_features = self._extract_image_features(image_resized)
                
                # Extract annotation features
                ann_features = self._extract_annotation_features(annotation)
                
                # Combine features
                combined_features = np.concatenate([img_features, ann_features])
                features.append(combined_features)
                
                # Create label (number of objects detected)
                num_objects = len(annotation.get('shapes', []))
                risk_level = self._classify_risk_level(num_objects)
                labels.append(risk_level)
                
                # Store metadata
                metadata.append({
                    'image_file': img_file.name,
                    'annotation_file': ann_file.name,
                    'num_objects': num_objects,
                    'image_size': (annotation.get('imageWidth', 0), annotation.get('imageHeight', 0))
                })
                
            except Exception as e:
                self.logger.warning(f"Error processing {img_file.name}: {e}")
                continue
        
        features = np.array(features)
        labels = np.array(labels)
        
        # Save processed data
        output_path = self.processed_data_dir / "object_detection_processed.npz"
        np.savez(output_path, 
                 features=features, 
                 labels=labels, 
                 metadata=metadata)
        
        self.logger.info(f"✅ Object detection preprocessing complete: {len(features)} samples")
        self.logger.info(f"   Features shape: {features.shape}")
        self.logger.info(f"   Saved to: {output_path}")
        
        return {
            'dataset_type': 'object_detection',
            'features': features,
            'labels': labels,
            'metadata': metadata,
            'num_samples': len(features),
            'num_features': features.shape[1],
            'output_path': str(output_path)
        }
    
    def preprocess_segmentation_dataset(self) -> Dict[str, Any]:
        """Preprocess the Segmentation Dataset."""
        self.logger.info("🔍 Preprocessing Open Pit Mine Segmentation Dataset...")
        
        images_dir = self.segmentation_path / "images"
        labels_dir = self.segmentation_path / "labels"
        
        # Get list of images and labels
        image_files = sorted(list(images_dir.glob("*.jpg")))
        label_files = sorted(list(labels_dir.glob("*.txt")))
        
        # Limit samples for processing
        if len(image_files) > self.max_samples_per_dataset:
            image_files = image_files[:self.max_samples_per_dataset]
            label_files = label_files[:self.max_samples_per_dataset]
        
        self.logger.info(f"Processing {len(image_files)} segmentation samples...")
        
        # Extract features and labels
        features = []
        labels = []
        metadata = []
        
        for img_file, label_file in zip(image_files, label_files):
            try:
                # Load and process image
                image = cv2.imread(str(img_file))
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                image_resized = cv2.resize(image, self.target_image_size)
                
                # Load segmentation labels
                with open(label_file, 'r') as f:
                    label_data = f.read().strip().split('\n')
                
                # Extract image features
                img_features = self._extract_image_features(image_resized)
                
                # Extract segmentation features
                seg_features = self._extract_segmentation_features(label_data)
                
                # Combine features
                combined_features = np.concatenate([img_features, seg_features])
                features.append(combined_features)
                
                # Create risk label based on segmentation complexity
                num_segments = len(label_data)
                risk_level = self._classify_segmentation_risk(num_segments, label_data)
                labels.append(risk_level)
                
                # Store metadata
                metadata.append({
                    'image_file': img_file.name,
                    'label_file': label_file.name,
                    'num_segments': num_segments,
                    'image_shape': image.shape
                })
                
            except Exception as e:
                self.logger.warning(f"Error processing {img_file.name}: {e}")
                continue
        
        features = np.array(features)
        labels = np.array(labels)
        
        # Save processed data
        output_path = self.processed_data_dir / "segmentation_processed.npz"
        np.savez(output_path, 
                 features=features, 
                 labels=labels, 
                 metadata=metadata)
        
        self.logger.info(f"✅ Segmentation preprocessing complete: {len(features)} samples")
        self.logger.info(f"   Features shape: {features.shape}")
        self.logger.info(f"   Saved to: {output_path}")
        
        return {
            'dataset_type': 'segmentation',
            'features': features,
            'labels': labels,
            'metadata': metadata,
            'num_samples': len(features),
            'num_features': features.shape[1],
            'output_path': str(output_path)
        }
    
    def _extract_image_features(self, image: np.ndarray) -> np.ndarray:
        """Extract statistical features from images."""
        # Convert to different color spaces
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
        
        features = []
        
        # Basic statistics (10 features)
        features.extend([
            np.mean(gray), np.std(gray), np.min(gray), np.max(gray),
            np.mean(image[:,:,0]), np.mean(image[:,:,1]), np.mean(image[:,:,2]),  # RGB means
            np.std(image[:,:,0]), np.std(image[:,:,1]), np.std(image[:,:,2])      # RGB stds
        ])
        
        # Texture features (4 features)
        grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        gradient_magnitude = np.sqrt(grad_x**2 + grad_y**2)
        
        features.extend([
            np.mean(gradient_magnitude), np.std(gradient_magnitude),
            np.mean(np.abs(grad_x)), np.mean(np.abs(grad_y))
        ])
        
        # Edge density (1 feature)
        edges = cv2.Canny(gray, 50, 150)
        edge_density = np.sum(edges > 0) / (edges.shape[0] * edges.shape[1])
        features.append(edge_density)
        
        # HSV statistics (6 features)
        features.extend([
            np.mean(hsv[:,:,0]), np.mean(hsv[:,:,1]), np.mean(hsv[:,:,2]),  # HSV means
            np.std(hsv[:,:,0]), np.std(hsv[:,:,1]), np.std(hsv[:,:,2])      # HSV stds
        ])
        
        # Ensure exactly 21 features
        features = features[:21]
        while len(features) < 21:
            features.append(0.0)
        
        return np.array(features)
    
    def _extract_annotation_features(self, annotation: Dict) -> np.ndarray:
        """Extract features from object detection annotations."""
        shapes = annotation.get('shapes', [])
        
        features = []
        
        # Basic counts
        features.append(len(shapes))  # Number of objects
        
        if len(shapes) == 0:
            # No objects detected - add zeros
            features.extend([0] * 15)  # Padding for consistent feature size
        else:
            # Object size statistics
            areas = []
            aspect_ratios = []
            
            for shape in shapes:
                if shape['shape_type'] == 'rectangle' and len(shape['points']) == 2:
                    x1, y1 = shape['points'][0]
                    x2, y2 = shape['points'][1]
                    
                    width = abs(x2 - x1)
                    height = abs(y2 - y1)
                    area = width * height
                    aspect_ratio = width / height if height > 0 else 1.0
                    
                    areas.append(area)
                    aspect_ratios.append(aspect_ratio)
            
            if areas:
                features.extend([
                    np.mean(areas), np.std(areas), np.min(areas), np.max(areas),
                    np.mean(aspect_ratios), np.std(aspect_ratios),
                    np.min(aspect_ratios), np.max(aspect_ratios)
                ])
            else:
                features.extend([0] * 8)
            
            # Object distribution features
            if len(shapes) > 1:
                centers = []
                for shape in shapes:
                    if shape['shape_type'] == 'rectangle' and len(shape['points']) == 2:
                        x1, y1 = shape['points'][0]
                        x2, y2 = shape['points'][1]
                        center_x = (x1 + x2) / 2
                        center_y = (y1 + y2) / 2
                        centers.append([center_x, center_y])
                
                if len(centers) > 1:
                    centers = np.array(centers)
                    features.extend([
                        np.std(centers[:, 0]),  # X distribution
                        np.std(centers[:, 1]),  # Y distribution
                        np.mean(centers[:, 0]), # Mean X
                        np.mean(centers[:, 1]), # Mean Y
                        len(set([shape['label'] for shape in shapes]))  # Unique labels
                    ])
                else:
                    features.extend([0] * 5)
            else:
                features.extend([0] * 5)
            
            # Padding to ensure consistent size
            while len(features) < 16:
                features.append(0)
        
        return np.array(features[:16])  # Ensure exactly 16 features
    
    def _extract_segmentation_features(self, label_data: List[str]) -> np.ndarray:
        """Extract features from segmentation labels."""
        features = []
        
        # Basic counts
        features.append(len(label_data))  # Number of segments
        
        if not label_data or label_data == ['']:
            # No segmentation data
            features.extend([0] * 19)
        else:
            segment_complexities = []
            class_counts = {}
            
            for line in label_data:
                if line.strip():
                    parts = line.strip().split()
                    if len(parts) > 1:
                        class_id = int(parts[0])
                        coordinates = [float(x) for x in parts[1:]]
                        
                        # Count points in polygon
                        num_points = len(coordinates) // 2
                        segment_complexities.append(num_points)
                        
                        # Count classes
                        class_counts[class_id] = class_counts.get(class_id, 0) + 1
            
            if segment_complexities:
                features.extend([
                    np.mean(segment_complexities), np.std(segment_complexities),
                    np.min(segment_complexities), np.max(segment_complexities),
                    len(class_counts),  # Number of unique classes
                    max(class_counts.values()) if class_counts else 0,  # Max class frequency
                    np.sum(segment_complexities)  # Total complexity
                ])
            else:
                features.extend([0] * 7)
            
            # Analyze coordinate distributions
            all_x_coords = []
            all_y_coords = []
            
            for line in label_data:
                if line.strip():
                    parts = line.strip().split()
                    if len(parts) > 1:
                        coordinates = [float(x) for x in parts[1:]]
                        x_coords = coordinates[::2]  # Even indices are x
                        y_coords = coordinates[1::2]  # Odd indices are y
                        all_x_coords.extend(x_coords)
                        all_y_coords.extend(y_coords)
            
            if all_x_coords and all_y_coords:
                features.extend([
                    np.mean(all_x_coords), np.std(all_x_coords),
                    np.mean(all_y_coords), np.std(all_y_coords),
                    np.min(all_x_coords), np.max(all_x_coords),
                    np.min(all_y_coords), np.max(all_y_coords),
                    len(all_x_coords)  # Total coordinate points
                ])
            else:
                features.extend([0] * 9)
            
            # Coverage analysis
            if all_x_coords and all_y_coords:
                x_range = np.max(all_x_coords) - np.min(all_x_coords)
                y_range = np.max(all_y_coords) - np.min(all_y_coords)
                coverage_area = x_range * y_range
                features.extend([coverage_area, x_range, y_range])
            else:
                features.extend([0, 0, 0])
        
        return np.array(features[:20])  # Ensure exactly 20 features
    
    def _classify_risk_level(self, num_objects: int) -> int:
        """Classify risk level based on number of detected objects."""
        if num_objects == 0:
            return 0  # Low risk - no objects
        elif num_objects <= 2:
            return 1  # Medium risk - few objects
        else:
            return 2  # High risk - many objects
    
    def _classify_segmentation_risk(self, num_segments: int, label_data: List[str]) -> int:
        """Classify risk level based on segmentation complexity."""
        if num_segments == 0:
            return 0  # Low risk - no segments
        
        # Calculate total complexity
        total_complexity = 0
        for line in label_data:
            if line.strip():
                parts = line.strip().split()
                if len(parts) > 1:
                    total_complexity += len(parts) - 1  # Exclude class ID
        
        if total_complexity < 20:
            return 0  # Low risk - simple segmentation
        elif total_complexity < 100:
            return 1  # Medium risk - moderate complexity
        else:
            return 2  # High risk - high complexity
    
    def create_combined_dataset(self) -> Dict[str, Any]:
        """Combine both datasets into a unified dataset."""
        self.logger.info("🔗 Creating combined mining dataset...")
        
        # Load processed datasets
        obj_det_file = self.processed_data_dir / "object_detection_processed.npz"
        seg_file = self.processed_data_dir / "segmentation_processed.npz"
        
        datasets = []
        
        if obj_det_file.exists():
            obj_data = np.load(obj_det_file, allow_pickle=True)
            datasets.append({
                'features': obj_data['features'],
                'labels': obj_data['labels'],
                'metadata': obj_data['metadata'],
                'source': 'object_detection'
            })
            self.logger.info(f"   Loaded object detection: {len(obj_data['features'])} samples, {obj_data['features'].shape[1]} features")
        
        if seg_file.exists():
            seg_data = np.load(seg_file, allow_pickle=True)
            datasets.append({
                'features': seg_data['features'],
                'labels': seg_data['labels'],
                'metadata': seg_data['metadata'],
                'source': 'segmentation'
            })
            self.logger.info(f"   Loaded segmentation: {len(seg_data['features'])} samples, {seg_data['features'].shape[1]} features")
        
        if not datasets:
            raise ValueError("No processed datasets found. Run preprocessing first.")
        
        # Find the maximum number of features across datasets
        max_features = max(d['features'].shape[1] for d in datasets)
        self.logger.info(f"   Standardizing to {max_features} features")
        
        # Pad features to match dimensions
        padded_datasets = []
        for dataset in datasets:
            features = dataset['features']
            if features.shape[1] < max_features:
                # Pad with zeros
                padding = np.zeros((features.shape[0], max_features - features.shape[1]))
                padded_features = np.hstack([features, padding])
            else:
                padded_features = features
            
            padded_datasets.append({
                'features': padded_features,
                'labels': dataset['labels'],
                'metadata': dataset['metadata'],
                'source': dataset['source']
            })
        
        # Combine datasets
        all_features = np.vstack([d['features'] for d in padded_datasets])
        all_labels = np.hstack([d['labels'] for d in padded_datasets])
        
        # Create combined metadata
        all_metadata = []
        for i, dataset in enumerate(padded_datasets):
            for j, meta in enumerate(dataset['metadata']):
                combined_meta = dict(meta)
                combined_meta['source_dataset'] = dataset['source']
                combined_meta['original_index'] = j
                all_metadata.append(combined_meta)
        
        # Normalize features
        scaler = StandardScaler()
        all_features_scaled = scaler.fit_transform(all_features)
        
        # Save combined dataset
        output_path = self.processed_data_dir / "mining_combined_processed.npz"
        np.savez(output_path,
                 features=all_features_scaled,
                 labels=all_labels,
                 metadata=all_metadata,
                 scaler_mean=scaler.mean_,
                 scaler_scale=scaler.scale_)
        
        self.logger.info(f"✅ Combined dataset created: {len(all_features)} samples")
        self.logger.info(f"   Features shape: {all_features_scaled.shape}")
        self.logger.info(f"   Class distribution: {np.bincount(all_labels)}")
        self.logger.info(f"   Saved to: {output_path}")
        
        return {
            'dataset_type': 'combined_mining',
            'features': all_features_scaled,
            'labels': all_labels,
            'metadata': all_metadata,
            'num_samples': len(all_features),
            'num_features': all_features_scaled.shape[1],
            'class_distribution': dict(zip(*np.unique(all_labels, return_counts=True))),
            'output_path': str(output_path)
        }
    
    def generate_preprocessing_report(self) -> str:
        """Generate a comprehensive preprocessing report."""
        report_path = self.processed_data_dir / "preprocessing_report.md"
        
        with open(report_path, 'w') as f:
            f.write("# Mining Datasets Preprocessing Report\n\n")
            f.write(f"**Generated:** {pd.Timestamp.now()}\n\n")
            
            # Check processed files
            processed_files = list(self.processed_data_dir.glob("*.npz"))
            
            f.write("## Processed Datasets\n\n")
            
            for file in processed_files:
                f.write(f"### {file.stem}\n\n")
                
                try:
                    data = np.load(file, allow_pickle=True)
                    features = data['features']
                    labels = data['labels']
                    
                    f.write(f"- **Samples:** {len(features)}\n")
                    f.write(f"- **Features:** {features.shape[1]}\n")
                    f.write(f"- **Classes:** {len(np.unique(labels))}\n")
                    f.write(f"- **Class Distribution:** {dict(zip(*np.unique(labels, return_counts=True)))}\n")
                    f.write(f"- **File Size:** {file.stat().st_size / 1024 / 1024:.2f} MB\n\n")
                    
                except Exception as e:
                    f.write(f"- **Error:** Could not load file: {e}\n\n")
            
            f.write("## Feature Descriptions\n\n")
            f.write("### Image Features (21 features)\n")
            f.write("1. Gray statistics: mean, std, min, max\n")
            f.write("2. RGB statistics: means and stds for R, G, B channels\n")
            f.write("3. Texture features: gradient magnitude and directional gradients\n")
            f.write("4. Edge density\n")
            f.write("5. HSV color space statistics\n\n")
            
            f.write("### Object Detection Features (16 features)\n")
            f.write("1. Number of detected objects\n")
            f.write("2. Object size statistics: area mean, std, min, max\n")
            f.write("3. Aspect ratio statistics\n")
            f.write("4. Object distribution: spatial spread and centers\n")
            f.write("5. Unique object classes\n\n")
            
            f.write("### Segmentation Features (20 features)\n")
            f.write("1. Number of segments\n")
            f.write("2. Segment complexity statistics\n")
            f.write("3. Class diversity metrics\n")
            f.write("4. Coordinate distribution analysis\n")
            f.write("5. Coverage area metrics\n\n")
            
            f.write("## Risk Classification\n\n")
            f.write("**Object Detection Risk Levels:**\n")
            f.write("- Level 0 (Low): No objects detected\n")
            f.write("- Level 1 (Medium): 1-2 objects detected\n")
            f.write("- Level 2 (High): 3+ objects detected\n\n")
            
            f.write("**Segmentation Risk Levels:**\n")
            f.write("- Level 0 (Low): Simple segmentation (complexity < 20)\n")
            f.write("- Level 1 (Medium): Moderate complexity (20-100)\n")
            f.write("- Level 2 (High): High complexity (100+)\n\n")
        
        self.logger.info(f"📄 Preprocessing report saved to: {report_path}")
        return str(report_path)
    
    def run_full_preprocessing(self) -> Dict[str, Any]:
        """Run complete preprocessing pipeline."""
        self.logger.info("🚀 Starting full mining datasets preprocessing...")
        
        results = {}
        
        # Process object detection dataset
        if self.object_detection_path.exists():
            results['object_detection'] = self.preprocess_object_detection_dataset()
        else:
            self.logger.warning("Object detection dataset not found")
        
        # Process segmentation dataset
        if self.segmentation_path.exists():
            results['segmentation'] = self.preprocess_segmentation_dataset()
        else:
            self.logger.warning("Segmentation dataset not found")
        
        # Create combined dataset if both exist
        if len(results) >= 1:
            results['combined'] = self.create_combined_dataset()
        
        # Generate report
        report_path = self.generate_preprocessing_report()
        results['report_path'] = report_path
        
        self.logger.info("✅ Full preprocessing completed!")
        
        return results


def main():
    """Main function."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Preprocess mining datasets")
    parser.add_argument('--raw-dir', default='data/raw', help='Raw data directory')
    parser.add_argument('--processed-dir', default='data/processed', help='Processed data directory')
    parser.add_argument('--max-samples', type=int, default=1000, help='Maximum samples per dataset')
    parser.add_argument('--dataset', choices=['object_detection', 'segmentation', 'combined', 'all'], 
                       default='all', help='Dataset to process')
    
    args = parser.parse_args()
    
    # Create preprocessor
    preprocessor = MiningDatasetPreprocessor(args.raw_dir, args.processed_dir)
    preprocessor.max_samples_per_dataset = args.max_samples
    
    try:
        if args.dataset == 'all':
            results = preprocessor.run_full_preprocessing()
        elif args.dataset == 'object_detection':
            results = preprocessor.preprocess_object_detection_dataset()
        elif args.dataset == 'segmentation':
            results = preprocessor.preprocess_segmentation_dataset()
        elif args.dataset == 'combined':
            results = preprocessor.create_combined_dataset()
        
        print("\n" + "="*80)
        print("PREPROCESSING SUMMARY")
        print("="*80)
        
        for dataset_type, result in results.items():
            if isinstance(result, dict) and 'num_samples' in result:
                print(f"✅ {dataset_type}: {result['num_samples']} samples, {result['num_features']} features")
        
        print("="*80)
        
    except Exception as e:
        logging.error(f"Preprocessing failed: {e}")
        raise


if __name__ == "__main__":
    main()