#!/usr/bin/env python3
"""
Mining dataset loader for the rockfall prediction system.
Loads preprocessed mining datasets for training.
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Tuple, Any, Optional
import logging


class MiningDatasetLoader:
    """Data loader for preprocessed mining datasets."""
    
    def __init__(self, data_dir: str, dataset_type: str = "combined"):
        """
        Initialize the mining dataset loader.
        
        Args:
            data_dir: Directory containing preprocessed data
            dataset_type: Type of dataset to load ('object_detection', 'segmentation', 'combined')
        """
        self.data_dir = data_dir
        self.dataset_type = dataset_type
        self.logger = logging.getLogger(__name__)
        
        # Map dataset types to file names
        self.dataset_files = {
            'object_detection': 'object_detection_processed.npz',
            'segmentation': 'segmentation_processed.npz',
            'combined': 'mining_combined_processed.npz'
        }
        
        self.class_names = {
            0: 'Low Risk',
            1: 'Medium Risk', 
            2: 'High Risk'
        }
    
    def load_data(self, **kwargs) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
        """
        Load the mining dataset.
        
        Returns:
            Tuple of (features, labels, metadata)
        """
        if self.dataset_type not in self.dataset_files:
            raise ValueError(f"Unknown dataset type: {self.dataset_type}. "
                           f"Available: {list(self.dataset_files.keys())}")
        
        file_path = Path(self.data_dir) / self.dataset_files[self.dataset_type]
        
        if not file_path.exists():
            raise FileNotFoundError(f"Dataset file not found: {file_path}. "
                                  f"Run preprocessing first.")
        
        self.logger.info(f"Loading mining dataset: {self.dataset_type}")
        
        # Load the data
        data = np.load(file_path, allow_pickle=True)
        
        features = data['features']
        labels = data['labels']
        metadata = data['metadata']
        
        # Create comprehensive metadata
        result_metadata = {
            'dataset_type': self.dataset_type,
            'num_samples': len(features),
            'num_features': features.shape[1],
            'num_classes': len(np.unique(labels)),
            'class_names': self.class_names,
            'class_distribution': dict(zip(*np.unique(labels, return_counts=True))),
            'feature_names': self._get_feature_names(),
            'file_path': str(file_path)
        }
        
        # Add scaling info if available
        if 'scaler_mean' in data and 'scaler_scale' in data:
            result_metadata['scaler_mean'] = data['scaler_mean']
            result_metadata['scaler_scale'] = data['scaler_scale']
            result_metadata['is_scaled'] = True
        else:
            result_metadata['is_scaled'] = False
        
        # Add original metadata if available
        if len(metadata) > 0:
            result_metadata['sample_metadata'] = metadata
        
        self.logger.info(f"Loaded {len(features)} samples with {features.shape[1]} features")
        self.logger.info(f"Class distribution: {result_metadata['class_distribution']}")
        
        return features, labels, result_metadata
    
    def _get_feature_names(self) -> list:
        """Get descriptive names for features."""
        # Base image features (21)
        image_features = [
            'gray_mean', 'gray_std', 'gray_min', 'gray_max',
            'red_mean', 'green_mean', 'blue_mean',
            'red_std', 'green_std', 'blue_std',
            'gradient_magnitude_mean', 'gradient_magnitude_std',
            'gradient_x_mean', 'gradient_y_mean',
            'edge_density',
            'hue_mean', 'saturation_mean', 'value_mean',
            'hue_std', 'saturation_std', 'value_std'
        ]
        
        if self.dataset_type == 'object_detection':
            # Object detection features (16)
            od_features = [
                'num_objects',
                'area_mean', 'area_std', 'area_min', 'area_max',
                'aspect_ratio_mean', 'aspect_ratio_std', 'aspect_ratio_min', 'aspect_ratio_max',
                'center_x_std', 'center_y_std', 'center_x_mean', 'center_y_mean',
                'unique_labels', 'od_feature_14', 'od_feature_15'
            ]
            return image_features + od_features
        
        elif self.dataset_type == 'segmentation':
            # Segmentation features (20)
            seg_features = [
                'num_segments',
                'complexity_mean', 'complexity_std', 'complexity_min', 'complexity_max',
                'num_classes', 'max_class_freq', 'total_complexity',
                'coord_x_mean', 'coord_x_std', 'coord_y_mean', 'coord_y_std',
                'coord_x_min', 'coord_x_max', 'coord_y_min', 'coord_y_max',
                'total_points', 'coverage_area', 'x_range', 'y_range'
            ]
            return image_features + seg_features
        
        elif self.dataset_type == 'combined':
            # Combined features (padded to max)
            od_features = [f'od_feature_{i}' for i in range(16)]
            seg_features = [f'seg_feature_{i}' for i in range(20)]
            # Return the longer feature set (segmentation has more features)
            return image_features + seg_features
        
        else:
            # Generic feature names
            return [f'feature_{i}' for i in range(41)]  # Max observed features
    
    def get_class_info(self) -> Dict[str, Any]:
        """Get information about the classes."""
        return {
            'class_names': self.class_names,
            'num_classes': len(self.class_names),
            'description': {
                0: 'Low risk - minimal objects/complexity detected',
                1: 'Medium risk - moderate objects/complexity detected', 
                2: 'High risk - many objects/high complexity detected'
            }
        }
    
    def split_by_source(self, features: np.ndarray, labels: np.ndarray, metadata: Dict[str, Any]) -> Dict[str, Dict]:
        """
        Split combined dataset by source (object_detection vs segmentation).
        
        Args:
            features: Feature array
            labels: Label array
            metadata: Dataset metadata
            
        Returns:
            Dictionary with separate datasets by source
        """
        if self.dataset_type != 'combined':
            raise ValueError("Source splitting only available for combined dataset")
        
        if 'sample_metadata' not in metadata:
            raise ValueError("Sample metadata not available for source splitting")
        
        sample_metadata = metadata['sample_metadata']
        
        # Group by source
        sources = {}
        for i, meta in enumerate(sample_metadata):
            source = meta.get('source_dataset', 'unknown')
            if source not in sources:
                sources[source] = {'indices': [], 'metadata': []}
            sources[source]['indices'].append(i)
            sources[source]['metadata'].append(meta)
        
        # Create separate datasets
        result = {}
        for source, data in sources.items():
            indices = np.array(data['indices'])
            result[source] = {
                'features': features[indices],
                'labels': labels[indices],
                'metadata': data['metadata'],
                'num_samples': len(indices)
            }
        
        return result


def create_mining_data_loader(processed_data_dir: str = "data/processed", 
                            dataset_type: str = "combined") -> MiningDatasetLoader:
    """
    Create a mining dataset loader.
    
    Args:
        processed_data_dir: Directory containing processed data
        dataset_type: Type of dataset ('object_detection', 'segmentation', 'combined')
        
    Returns:
        MiningDatasetLoader instance
    """
    return MiningDatasetLoader(processed_data_dir, dataset_type)


# Integration with existing training system
def load_mining_dataset_for_training(dataset_type: str = "combined", 
                                    data_dir: str = "data/processed") -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    """
    Load mining dataset in format compatible with existing training scripts.
    
    Args:
        dataset_type: Type of dataset to load
        data_dir: Directory containing processed data
        
    Returns:
        Tuple of (features, labels, metadata)
    """
    loader = MiningDatasetLoader(data_dir, dataset_type)
    return loader.load_data()