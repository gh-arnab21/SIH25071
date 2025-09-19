"""
Enhanced data loaders for specialized rockfall and mining datasets.
Supports the 10 datasets collected for comprehensive training.
"""

import os
import json
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional, Union
import logging
from abc import ABC, abstractmethod
import yaml
import cv2
from sklearn.preprocessing import StandardScaler, LabelEncoder
import obspy  # For seismic data processing

from .base import BaseDataLoader
from ..schema import RockfallDataPoint, ImageData, SensorData, EnvironmentalData


class BrazilianRockfallSlopeLoader(BaseDataLoader):
    """
    Data loader for Brazilian Rockfall Slope Dataset.
    
    Dataset: 220 slope samples with 8 variables (scored 1-4)
    Classes: Low (8-16), Medium (17-20), High (21-32) risk scores
    Format: CSV with structured features
    """
    
    def __init__(self, data_dir: str):
        super().__init__(data_dir)
        self.data_file = Path(data_dir) / "slope_data.csv"
        self.logger = logging.getLogger(__name__)
        
    def load_data(self) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
        """Load Brazilian rockfall slope data."""
        if not self.data_file.exists():
            raise FileNotFoundError(f"Dataset not found: {self.data_file}")
        
        # Load CSV data
        df = pd.read_csv(self.data_file)
        
        # Expected columns (may need adjustment based on actual dataset)
        feature_columns = [
            'rock_type', 'slope_angle', 'joint_spacing', 'water_presence',
            'weathering_degree', 'slope_height', 'discontinuity_orientation',
            'rock_mass_quality'
        ]
        
        # Extract features
        if all(col in df.columns for col in feature_columns):
            X = df[feature_columns].values
        else:
            # Use all columns except the target
            target_cols = ['risk_score', 'risk_class', 'sample_id']
            X = df.drop(columns=[col for col in target_cols if col in df.columns]).values
        
        # Extract target (risk classification)
        if 'risk_score' in df.columns:
            risk_scores = df['risk_score'].values
            # Convert scores to classes: Low (8-16), Medium (17-20), High (21-32)
            y = np.where(risk_scores <= 16, 0,  # Low
                        np.where(risk_scores <= 20, 1, 2))  # Medium, High
        elif 'risk_class' in df.columns:
            le = LabelEncoder()
            y = le.fit_transform(df['risk_class'].values)
        else:
            # Default: use last column as target
            y = df.iloc[:, -1].values
        
        metadata = {
            'dataset_name': 'brazilian_rockfall_slope',
            'num_samples': len(df),
            'num_features': X.shape[1],
            'num_classes': len(np.unique(y)),
            'feature_names': feature_columns if all(col in df.columns for col in feature_columns) else df.columns[:-1].tolist(),
            'class_names': ['Low Risk', 'Medium Risk', 'High Risk'],
            'description': 'Brazilian Rockfall Slope Stability Dataset with 8 engineered features'
        }
        
        self.logger.info(f"Loaded Brazilian slope dataset: {metadata['num_samples']} samples, {metadata['num_features']} features")
        return X, y, metadata


class RockNetSeismicLoader(BaseDataLoader):
    """
    Data loader for RockNet Seismic Dataset.
    
    Dataset: Seismic data for rockfall and earthquake detection from Taiwan
    Format: SAC files + Python processing tools
    Classes: Rockfall, Earthquake, Noise
    """
    
    def __init__(self, data_dir: str):
        super().__init__(data_dir)
        self.data_dir = Path(data_dir)
        self.logger = logging.getLogger(__name__)
        
        # Expected subdirectories
        self.earthquake_dir = self.data_dir / "data" / "earthquake"
        self.rockfall_dir = self.data_dir / "data" / "rockfall"
        self.noise_dir = self.data_dir / "data" / "noise"
        
    def load_data(self) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
        """Load RockNet seismic data."""
        # Check if required directories exist
        required_dirs = [self.earthquake_dir, self.rockfall_dir, self.noise_dir]
        for dir_path in required_dirs:
            if not dir_path.exists():
                raise FileNotFoundError(f"Required directory not found: {dir_path}")
        
        X_list = []
        y_list = []
        
        # Load each class
        classes = [
            (self.earthquake_dir, 0, "earthquake"),
            (self.rockfall_dir, 1, "rockfall"),
            (self.noise_dir, 2, "noise")
        ]
        
        for data_dir, label, class_name in classes:
            # Find SAC files
            sac_files = list(data_dir.glob("*.SAC")) + list(data_dir.glob("*.sac"))
            
            self.logger.info(f"Loading {len(sac_files)} {class_name} files")
            
            for sac_file in sac_files:
                try:
                    # Read SAC file using obspy
                    st = obspy.read(str(sac_file))
                    
                    # Extract features from seismic trace
                    trace = st[0]
                    features = self._extract_seismic_features(trace)
                    
                    X_list.append(features)
                    y_list.append(label)
                    
                except Exception as e:
                    self.logger.warning(f"Failed to process {sac_file}: {e}")
                    continue
        
        X = np.array(X_list)
        y = np.array(y_list)
        
        metadata = {
            'dataset_name': 'rocknet_seismic',
            'num_samples': len(X),
            'num_features': X.shape[1] if X.ndim > 1 else 1,
            'num_classes': 3,
            'class_names': ['Earthquake', 'Rockfall', 'Noise'],
            'sampling_rate': 'Variable (from SAC headers)',
            'description': 'RockNet seismic dataset for rockfall detection from Taiwan'
        }
        
        self.logger.info(f"Loaded RockNet dataset: {metadata['num_samples']} samples, {metadata['num_features']} features")
        return X, y, metadata
    
    def _extract_seismic_features(self, trace) -> np.ndarray:
        """Extract features from seismic trace."""
        data = trace.data
        
        # Time domain features
        features = []
        
        # Statistical features
        features.extend([
            np.mean(data),
            np.std(data),
            np.max(data),
            np.min(data),
            np.median(data),
            np.percentile(data, 25),
            np.percentile(data, 75)
        ])
        
        # Energy features
        features.extend([
            np.sum(data**2),  # Total energy
            np.sum(np.abs(data)),  # Total absolute energy
            np.sqrt(np.mean(data**2))  # RMS
        ])
        
        # Frequency domain features (simple)
        fft = np.fft.fft(data)
        fft_mag = np.abs(fft)
        
        features.extend([
            np.argmax(fft_mag),  # Dominant frequency index
            np.sum(fft_mag),  # Total spectral energy
            np.mean(fft_mag),  # Mean spectral magnitude
            np.std(fft_mag)   # Spectral standard deviation
        ])
        
        # Zero crossing rate
        zero_crossings = np.sum(np.diff(np.sign(data)) != 0)
        features.append(zero_crossings / len(data))
        
        return np.array(features)


class OpenPitMineDetectionLoader(BaseDataLoader):
    """
    Data loader for Open Pit Mine Object Detection Dataset.
    
    Dataset: 1.06 GB of remote sensing images with JSON annotations
    Format: Images + JSON bounding box annotations
    Content: Mining equipment and structures
    """
    
    def __init__(self, data_dir: str):
        super().__init__(data_dir)
        self.data_dir = Path(data_dir)
        self.images_dir = self.data_dir / "images"
        self.annotations_dir = self.data_dir / "annotations"
        self.logger = logging.getLogger(__name__)
        
    def load_data(self) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
        """Load open pit mine detection data."""
        if not self.images_dir.exists() or not self.annotations_dir.exists():
            raise FileNotFoundError("Images or annotations directory not found")
        
        # Load annotations
        annotation_files = list(self.annotations_dir.glob("*.json"))
        if not annotation_files:
            raise FileNotFoundError("No JSON annotation files found")
        
        # Load all annotations
        all_annotations = []
        for ann_file in annotation_files:
            with open(ann_file, 'r') as f:
                annotations = json.load(f)
                if isinstance(annotations, list):
                    all_annotations.extend(annotations)
                else:
                    all_annotations.append(annotations)
        
        # Process images and extract features
        X_list = []
        y_list = []
        
        for annotation in all_annotations:
            # Get image path
            image_filename = annotation.get('filename', annotation.get('image_name'))
            if not image_filename:
                continue
                
            image_path = self.images_dir / image_filename
            if not image_path.exists():
                continue
            
            try:
                # Load and process image
                image = cv2.imread(str(image_path))
                if image is None:
                    continue
                
                # Extract image features
                features = self._extract_image_features(image)
                
                # Extract object detection labels (simplified to binary: has_objects)
                has_objects = len(annotation.get('objects', annotation.get('annotations', []))) > 0
                
                X_list.append(features)
                y_list.append(int(has_objects))
                
            except Exception as e:
                self.logger.warning(f"Failed to process {image_path}: {e}")
                continue
        
        X = np.array(X_list)
        y = np.array(y_list)
        
        metadata = {
            'dataset_name': 'open_pit_mine_detection',
            'num_samples': len(X),
            'num_features': X.shape[1] if X.ndim > 1 else 1,
            'num_classes': 2,
            'class_names': ['No Objects', 'Has Objects'],
            'description': 'Open pit mine object detection dataset with mining equipment',
            'original_task': 'object_detection'
        }
        
        self.logger.info(f"Loaded open pit mine dataset: {metadata['num_samples']} samples")
        return X, y, metadata
    
    def _extract_image_features(self, image: np.ndarray) -> np.ndarray:
        """Extract features from mining images."""
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        features = []
        
        # Basic statistical features
        features.extend([
            np.mean(gray),
            np.std(gray),
            np.min(gray),
            np.max(gray)
        ])
        
        # Texture features (simplified)
        # Calculate local binary pattern approximation
        # Sobel edge detection
        sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        edge_magnitude = np.sqrt(sobel_x**2 + sobel_y**2)
        
        features.extend([
            np.mean(edge_magnitude),
            np.std(edge_magnitude),
            np.sum(edge_magnitude > np.mean(edge_magnitude))  # Edge pixel count
        ])
        
        # Color features (from original BGR image)
        for channel in range(3):
            channel_data = image[:, :, channel]
            features.extend([
                np.mean(channel_data),
                np.std(channel_data)
            ])
        
        return np.array(features)


class RailwayRockfallLoader(BaseDataLoader):
    """
    Data loader for Railway Rockfall Detection Dataset.
    
    Dataset: 1,733 samples with 14 engineered features from fiber optic sensors
    Classes: Event_RF_Medium, Event_RF_Small, Trigger_RF_Small
    Format: CSV with preprocessed features
    """
    
    def __init__(self, data_dir: str):
        super().__init__(data_dir)
        self.data_dir = Path(data_dir)
        self.features_file = self.data_dir / "features.csv"
        self.labels_file = self.data_dir / "labels.csv"
        self.logger = logging.getLogger(__name__)
        
    def load_data(self) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
        """Load railway rockfall detection data."""
        # Try to load features and labels separately
        if self.features_file.exists() and self.labels_file.exists():
            X = pd.read_csv(self.features_file).values
            y_df = pd.read_csv(self.labels_file)
            if 'label' in y_df.columns:
                y = y_df['label'].values
            else:
                y = y_df.values.flatten()
        else:
            # Try to load from a single file
            data_file = self.data_dir / "railway_rockfall_data.csv"
            if not data_file.exists():
                # Look for any CSV file
                csv_files = list(self.data_dir.glob("*.csv"))
                if not csv_files:
                    raise FileNotFoundError("No CSV data files found")
                data_file = csv_files[0]
            
            df = pd.read_csv(data_file)
            
            # Assume last column is the target
            X = df.iloc[:, :-1].values
            y = df.iloc[:, -1].values
        
        # Encode string labels if necessary
        if y.dtype == 'O':  # Object type (strings)
            le = LabelEncoder()
            y = le.fit_transform(y)
            class_names = le.classes_.tolist()
        else:
            class_names = [f'Class_{i}' for i in range(len(np.unique(y)))]
        
        metadata = {
            'dataset_name': 'railway_rockfall',
            'num_samples': len(X),
            'num_features': X.shape[1],
            'num_classes': len(np.unique(y)),
            'class_names': class_names,
            'description': 'Railway rockfall detection from fiber optic sensor data',
            'sensor_type': 'fiber_optic_das'
        }
        
        self.logger.info(f"Loaded railway rockfall dataset: {metadata['num_samples']} samples, {metadata['num_features']} features")
        return X, y, metadata


class MultiDatasetLoader(BaseDataLoader):
    """
    Multi-dataset loader that can combine multiple datasets for comprehensive training.
    """
    
    def __init__(self, data_dir: str, datasets: Optional[List[str]] = None):
        super().__init__(data_dir)
        self.data_dir = Path(data_dir)
        self.datasets = datasets or ['brazilian_rockfall_slope', 'railway_rockfall']
        self.logger = logging.getLogger(__name__)
        
        # Initialize individual loaders
        self.loaders = {}
        self._initialize_loaders()
    
    def _initialize_loaders(self):
        """Initialize individual dataset loaders."""
        loader_mapping = {
            'brazilian_rockfall_slope': BrazilianRockfallSlopeLoader,
            'rocknet_seismic': RockNetSeismicLoader,
            'open_pit_mine_detection': OpenPitMineDetectionLoader,
            'railway_rockfall': RailwayRockfallLoader
        }
        
        for dataset_name in self.datasets:
            if dataset_name in loader_mapping:
                dataset_dir = self.data_dir / dataset_name
                if dataset_dir.exists():
                    self.loaders[dataset_name] = loader_mapping[dataset_name](str(dataset_dir))
                else:
                    self.logger.warning(f"Dataset directory not found: {dataset_dir}")
    
    def load_data(self) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
        """Load and combine multiple datasets."""
        X_combined = []
        y_combined = []
        dataset_metadata = {}
        
        for dataset_name, loader in self.loaders.items():
            try:
                X, y, metadata = loader.load_data()
                
                # Standardize features
                scaler = StandardScaler()
                X_scaled = scaler.fit_transform(X)
                
                X_combined.append(X_scaled)
                y_combined.append(y)
                dataset_metadata[dataset_name] = metadata
                
                self.logger.info(f"Added {dataset_name}: {len(X)} samples")
                
            except Exception as e:
                self.logger.error(f"Failed to load {dataset_name}: {e}")
                continue
        
        if not X_combined:
            raise ValueError("No datasets could be loaded successfully")
        
        # Combine datasets
        X_final = np.vstack(X_combined)
        y_final = np.hstack(y_combined)
        
        # Adjust labels to be consistent across datasets
        y_final = self._harmonize_labels(y_combined, dataset_metadata)
        
        combined_metadata = {
            'dataset_name': 'multimodal_combined',
            'num_samples': len(X_final),
            'num_features': X_final.shape[1],
            'num_classes': len(np.unique(y_final)),
            'datasets_included': list(self.loaders.keys()),
            'individual_metadata': dataset_metadata,
            'description': f'Combined dataset from {len(self.loaders)} sources'
        }
        
        self.logger.info(f"Combined dataset: {combined_metadata['num_samples']} samples from {len(self.loaders)} datasets")
        return X_final, y_final, combined_metadata
    
    def _harmonize_labels(self, y_list: List[np.ndarray], metadata: Dict[str, Any]) -> np.ndarray:
        """Harmonize labels across different datasets."""
        # Simple approach: map all to binary risk classification
        # 0 = Low/Safe, 1 = High/Dangerous
        
        harmonized_labels = []
        
        for i, y in enumerate(y_list):
            dataset_name = list(self.loaders.keys())[i]
            
            if dataset_name == 'brazilian_rockfall_slope':
                # 0=Low, 1=Medium, 2=High -> 0=Safe (Low/Medium), 1=Dangerous (High)
                y_harmonized = (y == 2).astype(int)
            elif dataset_name == 'rocknet_seismic':
                # 0=Earthquake, 1=Rockfall, 2=Noise -> 1=Dangerous (Rockfall), 0=Safe (others)
                y_harmonized = (y == 1).astype(int)
            elif dataset_name == 'railway_rockfall':
                # Assume binary or map largest class as dangerous
                y_harmonized = (y > 0).astype(int)
            else:
                # Default: keep as binary
                y_harmonized = (y > 0).astype(int)
            
            harmonized_labels.append(y_harmonized)
        
        return np.hstack(harmonized_labels)


# Enhanced DataLoaderRegistry with new datasets
def create_enhanced_data_loader_registry():
    """Create enhanced data loader registry with all specialized datasets."""
    from .data_loader import DataLoaderRegistry
    
    registry = DataLoaderRegistry()
    
    # Register new specialized loaders
    registry.register_loader('brazilian_rockfall', BrazilianRockfallSlopeLoader)
    registry.register_loader('rocknet_seismic', RockNetSeismicLoader)
    registry.register_loader('open_pit_mine_detection', OpenPitMineDetectionLoader)
    registry.register_loader('railway_rockfall', RailwayRockfallLoader)
    registry.register_loader('multimodal', MultiDatasetLoader)
    
    return registry