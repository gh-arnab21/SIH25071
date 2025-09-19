"""
Data Loading and Batch Processing for Rockfall Prediction System.

This module provides functionality for loading multi-modal data and
processing it in batches for efficient training.
"""

import logging
import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Iterator, Optional, Tuple, Union, Dict, Any
import json
import pickle
from datetime import datetime
import random

# Local imports
from ..data.schema import RockfallDataPoint, RiskLevel, ImageData, SensorData, EnvironmentalData, DEMData, SeismicData
from ..data.validation import validate_rockfall_datapoint, ValidationError
from ..data.utils import create_sample_datapoint

logger = logging.getLogger(__name__)


class DataLoader:
    """
    Loads multi-modal rockfall data from various sources and formats.
    
    Supports loading from:
    - JSON files with structured data
    - CSV files with tabular data
    - Directory structures with organized datasets
    - Pickle files with serialized data points
    """
    
    def __init__(self, data_dir: Union[str, Path]):
        """
        Initialize data loader.
        
        Args:
            data_dir: Directory containing the data files
        """
        self.data_dir = Path(data_dir)
        self.supported_formats = ['.json', '.csv', '.pkl', '.pickle']
        
        if not self.data_dir.exists():
            logger.warning(f"Data directory does not exist: {self.data_dir}")
            self.data_dir.mkdir(parents=True, exist_ok=True)
    
    def load_all_data(self) -> List[RockfallDataPoint]:
        """
        Load all available data from the data directory.
        
        Returns:
            List of RockfallDataPoint objects
        """
        logger.info(f"Loading data from {self.data_dir}")
        
        all_data = []
        
        # Search for data files
        for file_path in self.data_dir.rglob("*"):
            if file_path.is_file() and file_path.suffix.lower() in self.supported_formats:
                try:
                    data_points = self._load_file(file_path)
                    all_data.extend(data_points)
                    logger.info(f"Loaded {len(data_points)} data points from {file_path}")
                except Exception as e:
                    logger.error(f"Failed to load data from {file_path}: {e}")
        
        # If no data found, create sample data for testing
        if not all_data:
            logger.warning("No data files found. Creating sample data for testing.")
            all_data = self._create_sample_data(100)
        
        logger.info(f"Total data points loaded: {len(all_data)}")
        return all_data
    
    def _load_file(self, file_path: Path) -> List[RockfallDataPoint]:
        """
        Load data from a single file.
        
        Args:
            file_path: Path to the data file
            
        Returns:
            List of RockfallDataPoint objects
        """
        suffix = file_path.suffix.lower()
        
        if suffix == '.json':
            return self._load_json(file_path)
        elif suffix == '.csv':
            return self._load_csv(file_path)
        elif suffix in ['.pkl', '.pickle']:
            return self._load_pickle(file_path)
        else:
            raise ValueError(f"Unsupported file format: {suffix}")
    
    def _load_json(self, file_path: Path) -> List[RockfallDataPoint]:
        """Load data from JSON file."""
        with open(file_path, 'r') as f:
            data = json.load(f)
        
        data_points = []
        
        # Handle different JSON structures
        if isinstance(data, list):
            # List of data points
            for item in data:
                try:
                    data_point = self._dict_to_datapoint(item)
                    validate_rockfall_datapoint(data_point)
                    data_points.append(data_point)
                except (ValidationError, ValueError) as e:
                    logger.warning(f"Skipping invalid data point: {e}")
        elif isinstance(data, dict):
            # Single data point or structured format
            if 'data_points' in data:
                # Structured format with metadata
                for item in data['data_points']:
                    try:
                        data_point = self._dict_to_datapoint(item)
                        validate_rockfall_datapoint(data_point)
                        data_points.append(data_point)
                    except (ValidationError, ValueError) as e:
                        logger.warning(f"Skipping invalid data point: {e}")
            else:
                # Single data point
                try:
                    data_point = self._dict_to_datapoint(data)
                    validate_rockfall_datapoint(data_point)
                    data_points.append(data_point)
                except (ValidationError, ValueError) as e:
                    logger.warning(f"Skipping invalid data point: {e}")
        
        return data_points
    
    def _load_csv(self, file_path: Path) -> List[RockfallDataPoint]:
        """Load data from CSV file."""
        df = pd.read_csv(file_path)
        data_points = []
        
        for _, row in df.iterrows():
            try:
                data_point = self._row_to_datapoint(row)
                validate_rockfall_datapoint(data_point)
                data_points.append(data_point)
            except (ValidationError, ValueError) as e:
                logger.warning(f"Skipping invalid data point: {e}")
        
        return data_points
    
    def _load_pickle(self, file_path: Path) -> List[RockfallDataPoint]:
        """Load data from pickle file."""
        with open(file_path, 'rb') as f:
            data = pickle.load(f)
        
        if isinstance(data, list):
            # Validate each data point
            valid_data = []
            for item in data:
                try:
                    if isinstance(item, RockfallDataPoint):
                        validate_rockfall_datapoint(item)
                        valid_data.append(item)
                    else:
                        # Try to convert dict to datapoint
                        data_point = self._dict_to_datapoint(item)
                        validate_rockfall_datapoint(data_point)
                        valid_data.append(data_point)
                except (ValidationError, ValueError) as e:
                    logger.warning(f"Skipping invalid data point: {e}")
            return valid_data
        elif isinstance(data, RockfallDataPoint):
            validate_rockfall_datapoint(data)
            return [data]
        else:
            raise ValueError("Pickle file must contain RockfallDataPoint or list of RockfallDataPoint objects")
    
    def _dict_to_datapoint(self, data_dict: Dict[str, Any]) -> RockfallDataPoint:
        """Convert dictionary to RockfallDataPoint."""
        # This is a simplified conversion - in practice, you'd need more sophisticated parsing
        # based on your actual data format
        
        # Parse timestamp
        timestamp_str = data_dict.get('timestamp', datetime.now().isoformat())
        if isinstance(timestamp_str, str):
            timestamp = datetime.fromisoformat(timestamp_str.replace('Z', '+00:00'))
        else:
            timestamp = timestamp_str
        
        # Parse location
        location_data = data_dict.get('location', {})
        from ..data.schema import GeoCoordinate
        location = GeoCoordinate(
            latitude=location_data.get('latitude', 0.0),
            longitude=location_data.get('longitude', 0.0),
            elevation=location_data.get('elevation')
        )
        
        # Parse ground truth
        ground_truth = None
        if 'ground_truth' in data_dict:
            gt_value = data_dict['ground_truth']
            if isinstance(gt_value, str):
                ground_truth = RiskLevel[gt_value.upper()]
            elif isinstance(gt_value, int):
                ground_truth = RiskLevel(gt_value)
            elif isinstance(gt_value, RiskLevel):
                ground_truth = gt_value
        
        # Create data point (other fields would be parsed similarly)
        return RockfallDataPoint(
            timestamp=timestamp,
            location=location,
            ground_truth=ground_truth
        )
    
    def _row_to_datapoint(self, row: pd.Series) -> RockfallDataPoint:
        """Convert pandas row to RockfallDataPoint."""
        # Parse timestamp
        timestamp = pd.to_datetime(row.get('timestamp', datetime.now())).to_pydatetime()
        
        # Parse location
        from ..data.schema import GeoCoordinate
        location = GeoCoordinate(
            latitude=float(row.get('latitude', 0.0)),
            longitude=float(row.get('longitude', 0.0)),
            elevation=row.get('elevation') if pd.notna(row.get('elevation')) else None
        )
        
        # Parse ground truth
        ground_truth = None
        if 'ground_truth' in row and pd.notna(row['ground_truth']):
            gt_value = row['ground_truth']
            if isinstance(gt_value, str):
                ground_truth = RiskLevel[gt_value.upper()]
            elif isinstance(gt_value, (int, float)):
                ground_truth = RiskLevel(int(gt_value))
        
        return RockfallDataPoint(
            timestamp=timestamp,
            location=location,
            ground_truth=ground_truth
        )
    
    def _create_sample_data(self, n_samples: int) -> List[RockfallDataPoint]:
        """Create sample data for testing purposes."""
        logger.info(f"Creating {n_samples} sample data points")
        
        sample_data = []
        for i in range(n_samples):
            data_point = create_sample_datapoint()
            sample_data.append(data_point)
        
        return sample_data
    
    def save_data(self, data_points: List[RockfallDataPoint], file_path: Union[str, Path], format: str = 'pickle'):
        """
        Save data points to file.
        
        Args:
            data_points: List of data points to save
            file_path: Output file path
            format: Output format ('pickle', 'json')
        """
        file_path = Path(file_path)
        file_path.parent.mkdir(parents=True, exist_ok=True)
        
        if format == 'pickle':
            with open(file_path, 'wb') as f:
                pickle.dump(data_points, f)
        elif format == 'json':
            # Convert to serializable format
            serializable_data = []
            for dp in data_points:
                dp_dict = {
                    'timestamp': dp.timestamp.isoformat(),
                    'location': {
                        'latitude': dp.location.latitude,
                        'longitude': dp.location.longitude,
                        'elevation': dp.location.elevation
                    },
                    'ground_truth': dp.ground_truth.name if dp.ground_truth else None
                }
                serializable_data.append(dp_dict)
            
            with open(file_path, 'w') as f:
                json.dump(serializable_data, f, indent=2)
        else:
            raise ValueError(f"Unsupported format: {format}")
        
        logger.info(f"Saved {len(data_points)} data points to {file_path}")


class BatchProcessor:
    """
    Processes data in batches for efficient training and inference.
    
    Handles batching, shuffling, and data augmentation during training.
    """
    
    def __init__(self, batch_size: int = 32, shuffle: bool = True, random_state: int = 42):
        """
        Initialize batch processor.
        
        Args:
            batch_size: Size of each batch
            shuffle: Whether to shuffle data between epochs
            random_state: Random seed for reproducibility
        """
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.random_state = random_state
        
        # Set random seeds
        random.seed(random_state)
        np.random.seed(random_state)
    
    def create_batches(self, data_points: List[RockfallDataPoint]) -> Iterator[List[RockfallDataPoint]]:
        """
        Create batches from data points.
        
        Args:
            data_points: List of data points to batch
            
        Yields:
            Batches of data points
        """
        # Shuffle if requested
        if self.shuffle:
            data_points = data_points.copy()
            random.shuffle(data_points)
        
        # Create batches
        for i in range(0, len(data_points), self.batch_size):
            batch = data_points[i:i + self.batch_size]
            yield batch
    
    def create_feature_batches(self, features: np.ndarray, labels: np.ndarray) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
        """
        Create batches from feature arrays.
        
        Args:
            features: Feature matrix
            labels: Label array
            
        Yields:
            Batches of (features, labels)
        """
        n_samples = features.shape[0]
        
        # Create indices
        indices = np.arange(n_samples)
        if self.shuffle:
            np.random.shuffle(indices)
        
        # Create batches
        for i in range(0, n_samples, self.batch_size):
            batch_indices = indices[i:i + self.batch_size]
            batch_features = features[batch_indices]
            batch_labels = labels[batch_indices]
            yield batch_features, batch_labels
    
    def get_batch_count(self, n_samples: int) -> int:
        """
        Get the number of batches for a given number of samples.
        
        Args:
            n_samples: Total number of samples
            
        Returns:
            Number of batches
        """
        return (n_samples + self.batch_size - 1) // self.batch_size
    
    def balance_classes(self, data_points: List[RockfallDataPoint]) -> List[RockfallDataPoint]:
        """
        Balance classes in the dataset using oversampling.
        
        Args:
            data_points: Original data points
            
        Returns:
            Balanced data points
        """
        # Group by risk level
        class_groups = {risk_level: [] for risk_level in RiskLevel}
        
        for dp in data_points:
            if dp.ground_truth is not None:
                class_groups[dp.ground_truth].append(dp)
        
        # Find the maximum class size
        max_size = max(len(group) for group in class_groups.values() if group)
        
        if max_size == 0:
            logger.warning("No labeled data found for class balancing")
            return data_points
        
        # Oversample minority classes
        balanced_data = []
        for risk_level, group in class_groups.items():
            if not group:
                continue
            
            # Oversample to match max_size
            while len(group) < max_size:
                group.extend(random.choices(group, k=min(len(group), max_size - len(group))))
            
            balanced_data.extend(group[:max_size])
        
        # Shuffle the balanced data
        random.shuffle(balanced_data)
        
        logger.info(f"Balanced dataset: {len(balanced_data)} samples")
        for risk_level in RiskLevel:
            count = sum(1 for dp in balanced_data if dp.ground_truth == risk_level)
            logger.info(f"  {risk_level.name}: {count} samples")
        
        return balanced_data
    
    def apply_data_augmentation(self, data_points: List[RockfallDataPoint]) -> List[RockfallDataPoint]:
        """
        Apply data augmentation techniques to increase dataset size.
        
        Args:
            data_points: Original data points
            
        Returns:
            Augmented data points
        """
        # This is a placeholder for data augmentation
        # In practice, you would implement specific augmentation techniques
        # for each data modality (image rotation, sensor noise injection, etc.)
        
        augmented_data = data_points.copy()
        
        # Simple augmentation: add noise to numerical values
        for dp in data_points:
            if dp.environmental and random.random() < 0.3:  # 30% chance of augmentation
                # Create augmented version with slight noise
                augmented_dp = RockfallDataPoint(
                    timestamp=dp.timestamp,
                    location=dp.location,
                    imagery=dp.imagery,
                    dem_data=dp.dem_data,
                    sensor_readings=dp.sensor_readings,
                    environmental=dp.environmental,  # Could add noise here
                    seismic=dp.seismic,
                    ground_truth=dp.ground_truth
                )
                augmented_data.append(augmented_dp)
        
        logger.info(f"Data augmentation: {len(data_points)} -> {len(augmented_data)} samples")
        
        return augmented_data