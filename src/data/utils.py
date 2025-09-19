"""
Utility functions for data type conversion and normalization.

This module provides functions for converting between different data formats,
normalizing data values, and handling common data preprocessing tasks.
"""

import numpy as np
import pandas as pd
from datetime import datetime, timezone
from typing import List, Dict, Any, Optional, Union, Tuple
import json
import logging

from .schema import (
    RockfallDataPoint, ImageData, SensorData, EnvironmentalData,
    DEMData, SeismicData, TimeSeries, GeoCoordinate, BoundingBox,
    RiskLevel, ImageMetadata
)

logger = logging.getLogger(__name__)


def normalize_coordinates(latitude: float, longitude: float) -> Tuple[float, float]:
    """
    Normalize geographic coordinates to standard ranges.
    
    Args:
        latitude: Latitude in degrees
        longitude: Longitude in degrees
        
    Returns:
        Tuple of normalized (latitude, longitude)
    """
    # Normalize latitude to [-90, 90]
    lat = max(-90.0, min(90.0, latitude))
    
    # Normalize longitude to [-180, 180]
    lon = longitude % 360
    if lon > 180:
        lon -= 360
    
    return lat, lon


def convert_timestamp_to_datetime(timestamp: Union[str, int, float, datetime]) -> datetime:
    """
    Convert various timestamp formats to datetime object.
    
    Args:
        timestamp: Timestamp in various formats (string, unix timestamp, datetime)
        
    Returns:
        datetime object
        
    Raises:
        ValueError: If timestamp format is not supported
    """
    if isinstance(timestamp, datetime):
        return timestamp
    
    if isinstance(timestamp, (int, float)):
        # Assume unix timestamp
        try:
            return datetime.fromtimestamp(timestamp, tz=timezone.utc)
        except (ValueError, OSError) as e:
            raise ValueError(f"Invalid unix timestamp: {timestamp}") from e
    
    if isinstance(timestamp, str):
        # Try common datetime formats
        formats = [
            '%Y-%m-%d %H:%M:%S',
            '%Y-%m-%dT%H:%M:%S',
            '%Y-%m-%dT%H:%M:%SZ',
            '%Y-%m-%dT%H:%M:%S.%f',
            '%Y-%m-%dT%H:%M:%S.%fZ',
            '%Y-%m-%d',
            '%d/%m/%Y %H:%M:%S',
            '%m/%d/%Y %H:%M:%S'
        ]
        
        for fmt in formats:
            try:
                return datetime.strptime(timestamp, fmt)
            except ValueError:
                continue
        
        raise ValueError(f"Unable to parse timestamp: {timestamp}")
    
    raise ValueError(f"Unsupported timestamp type: {type(timestamp)}")


def normalize_array(arr: np.ndarray, method: str = 'minmax') -> np.ndarray:
    """
    Normalize numpy array using specified method.
    
    Args:
        arr: Input array to normalize
        method: Normalization method ('minmax', 'zscore', 'robust')
        
    Returns:
        Normalized array
        
    Raises:
        ValueError: If method is not supported
    """
    if method == 'minmax':
        min_val = np.min(arr)
        max_val = np.max(arr)
        if max_val == min_val:
            return np.zeros_like(arr)
        return (arr - min_val) / (max_val - min_val)
    
    elif method == 'zscore':
        mean_val = np.mean(arr)
        std_val = np.std(arr)
        if std_val == 0:
            return np.zeros_like(arr)
        return (arr - mean_val) / std_val
    
    elif method == 'robust':
        median_val = np.median(arr)
        mad = np.median(np.abs(arr - median_val))
        if mad == 0:
            return np.zeros_like(arr)
        return (arr - median_val) / (1.4826 * mad)  # 1.4826 for normal distribution
    
    else:
        raise ValueError(f"Unsupported normalization method: {method}")


def convert_dict_to_timeseries(data_dict: Dict[str, Any], 
                              timestamp_key: str = 'timestamp',
                              value_key: str = 'value',
                              unit: str = 'unknown') -> TimeSeries:
    """
    Convert dictionary data to TimeSeries object.
    
    Args:
        data_dict: Dictionary containing timestamp and value data
        timestamp_key: Key for timestamp data
        value_key: Key for value data
        unit: Unit of measurement
        
    Returns:
        TimeSeries object
    """
    if isinstance(data_dict, list):
        # Handle list of dictionaries
        timestamps = []
        values = []
        for item in data_dict:
            timestamps.append(convert_timestamp_to_datetime(item[timestamp_key]))
            values.append(float(item[value_key]))
    else:
        # Handle dictionary with arrays
        timestamps = [convert_timestamp_to_datetime(ts) for ts in data_dict[timestamp_key]]
        values = [float(v) for v in data_dict[value_key]]
    
    return TimeSeries(timestamps=timestamps, values=values, unit=unit)


def convert_json_to_bounding_boxes(json_data: Union[str, Dict, List]) -> List[BoundingBox]:
    """
    Convert JSON annotation data to BoundingBox objects.
    
    Args:
        json_data: JSON data containing bounding box annotations
        
    Returns:
        List of BoundingBox objects
    """
    if isinstance(json_data, str):
        json_data = json.loads(json_data)
    
    bboxes = []
    
    if isinstance(json_data, list):
        # Handle list of annotations
        for annotation in json_data:
            bbox = BoundingBox(
                x=float(annotation.get('x', annotation.get('left', 0))),
                y=float(annotation.get('y', annotation.get('top', 0))),
                width=float(annotation.get('width', annotation.get('w', 0))),
                height=float(annotation.get('height', annotation.get('h', 0))),
                class_name=str(annotation.get('class', annotation.get('label', 'unknown'))),
                confidence=annotation.get('confidence', annotation.get('score'))
            )
            bboxes.append(bbox)
    
    elif isinstance(json_data, dict):
        # Handle single annotation or nested structure
        if 'annotations' in json_data:
            return convert_json_to_bounding_boxes(json_data['annotations'])
        elif 'objects' in json_data:
            return convert_json_to_bounding_boxes(json_data['objects'])
        else:
            # Single annotation
            bbox = BoundingBox(
                x=float(json_data.get('x', json_data.get('left', 0))),
                y=float(json_data.get('y', json_data.get('top', 0))),
                width=float(json_data.get('width', json_data.get('w', 0))),
                height=float(json_data.get('height', json_data.get('h', 0))),
                class_name=str(json_data.get('class', json_data.get('label', 'unknown'))),
                confidence=json_data.get('confidence', json_data.get('score'))
            )
            bboxes.append(bbox)
    
    return bboxes


def convert_yolo_to_bounding_box(yolo_line: str, image_width: int, image_height: int, 
                                class_names: Optional[List[str]] = None) -> BoundingBox:
    """
    Convert YOLO format annotation to BoundingBox object.
    
    Args:
        yolo_line: YOLO format line (class_id center_x center_y width height)
        image_width: Width of the image
        image_height: Height of the image
        class_names: Optional list of class names
        
    Returns:
        BoundingBox object
    """
    parts = yolo_line.strip().split()
    if len(parts) < 5:
        raise ValueError(f"Invalid YOLO format: {yolo_line}")
    
    class_id = int(parts[0])
    center_x = float(parts[1]) * image_width
    center_y = float(parts[2]) * image_height
    width = float(parts[3]) * image_width
    height = float(parts[4]) * image_height
    
    # Convert center coordinates to top-left coordinates
    x = center_x - width / 2
    y = center_y - height / 2
    
    class_name = class_names[class_id] if class_names and class_id < len(class_names) else f"class_{class_id}"
    confidence = float(parts[5]) if len(parts) > 5 else None
    
    return BoundingBox(x=x, y=y, width=width, height=height, 
                      class_name=class_name, confidence=confidence)


def convert_dataframe_to_environmental_data(df: pd.DataFrame, 
                                          column_mapping: Optional[Dict[str, str]] = None) -> EnvironmentalData:
    """
    Convert pandas DataFrame to EnvironmentalData object.
    
    Args:
        df: DataFrame containing environmental measurements
        column_mapping: Optional mapping of DataFrame columns to EnvironmentalData fields
        
    Returns:
        EnvironmentalData object with aggregated values
    """
    if column_mapping is None:
        column_mapping = {
            'rainfall': 'rainfall',
            'temperature': 'temperature', 
            'vibrations': 'vibrations',
            'wind_speed': 'wind_speed',
            'humidity': 'humidity'
        }
    
    env_data = {}
    
    for field, column in column_mapping.items():
        if column in df.columns:
            # Use mean value if multiple measurements
            values = df[column].dropna()
            if len(values) > 0:
                env_data[field] = float(values.mean())
    
    return EnvironmentalData(**env_data)


def impute_missing_values(arr: np.ndarray, method: str = 'mean') -> np.ndarray:
    """
    Impute missing values in numpy array.
    
    Args:
        arr: Input array with potential NaN values
        method: Imputation method ('mean', 'median', 'mode', 'forward_fill', 'backward_fill')
        
    Returns:
        Array with imputed values
    """
    if not np.any(np.isnan(arr)):
        return arr.copy()
    
    result = arr.copy()
    mask = np.isnan(result)
    
    if method == 'mean':
        fill_value = np.nanmean(result)
    elif method == 'median':
        fill_value = np.nanmedian(result)
    elif method == 'mode':
        # For mode, use most frequent non-NaN value
        valid_values = result[~mask]
        if len(valid_values) > 0:
            unique, counts = np.unique(valid_values, return_counts=True)
            fill_value = unique[np.argmax(counts)]
        else:
            fill_value = 0
    elif method == 'forward_fill':
        # Forward fill
        for i in range(len(result)):
            if mask[i] and i > 0:
                result[i] = result[i-1]
        return result
    elif method == 'backward_fill':
        # Backward fill
        for i in range(len(result)-2, -1, -1):
            if mask[i] and i < len(result)-1:
                result[i] = result[i+1]
        return result
    else:
        raise ValueError(f"Unsupported imputation method: {method}")
    
    result[mask] = fill_value
    return result


def detect_outliers(arr: np.ndarray, method: str = 'iqr', threshold: float = 1.5) -> np.ndarray:
    """
    Detect outliers in numpy array.
    
    Args:
        arr: Input array
        method: Detection method ('iqr', 'zscore', 'modified_zscore')
        threshold: Threshold for outlier detection
        
    Returns:
        Boolean array indicating outliers
    """
    if method == 'iqr':
        q1 = np.percentile(arr, 25)
        q3 = np.percentile(arr, 75)
        iqr = q3 - q1
        lower_bound = q1 - threshold * iqr
        upper_bound = q3 + threshold * iqr
        return (arr < lower_bound) | (arr > upper_bound)
    
    elif method == 'zscore':
        z_scores = np.abs((arr - np.mean(arr)) / np.std(arr))
        return z_scores > threshold
    
    elif method == 'modified_zscore':
        median = np.median(arr)
        mad = np.median(np.abs(arr - median))
        modified_z_scores = 0.6745 * (arr - median) / mad
        return np.abs(modified_z_scores) > threshold
    
    else:
        raise ValueError(f"Unsupported outlier detection method: {method}")


def convert_risk_level_string(risk_str: str) -> RiskLevel:
    """
    Convert string representation to RiskLevel enum.
    
    Args:
        risk_str: String representation of risk level
        
    Returns:
        RiskLevel enum value
        
    Raises:
        ValueError: If string cannot be converted
    """
    risk_str = risk_str.upper().strip()
    
    risk_mapping = {
        'LOW': RiskLevel.LOW,
        'MEDIUM': RiskLevel.MEDIUM,
        'MED': RiskLevel.MEDIUM,
        'HIGH': RiskLevel.HIGH,
        '0': RiskLevel.LOW,
        '1': RiskLevel.MEDIUM,
        '2': RiskLevel.HIGH
    }
    
    if risk_str in risk_mapping:
        return risk_mapping[risk_str]
    
    raise ValueError(f"Cannot convert '{risk_str}' to RiskLevel")


def create_sample_datapoint() -> RockfallDataPoint:
    """
    Create a sample RockfallDataPoint for testing purposes.
    
    Returns:
        Sample RockfallDataPoint object
    """
    # Create sample timestamp and location
    base_time = datetime.now()
    location = GeoCoordinate(latitude=45.0, longitude=-120.0, elevation=1500.0)
    
    # Create sample sensor data with proper ascending timestamps
    timestamps = [base_time.replace(second=i) for i in range(10)]
    values = np.random.normal(0, 1, 10).tolist()
    sensor_data = SensorData(
        displacement=TimeSeries(timestamps=timestamps, values=values, unit='mm'),
        strain=TimeSeries(timestamps=timestamps, values=values, unit='microstrain')
    )
    
    # Create sample environmental data
    env_data = EnvironmentalData(
        rainfall=5.2,
        temperature=15.5,
        wind_speed=12.0,
        humidity=65.0
    )
    
    return RockfallDataPoint(
        timestamp=base_time,
        location=location,
        sensor_readings=sensor_data,
        environmental=env_data,
        ground_truth=RiskLevel.MEDIUM
    )