"""
Data validation functions for the Rockfall Prediction System.

This module provides comprehensive validation functions to ensure data quality
and schema compliance across all data types used in the system.
"""

import os
from typing import List, Dict, Any, Optional, Union
import numpy as np
from datetime import datetime
import logging

from .schema import (
    RockfallDataPoint, ImageData, SensorData, EnvironmentalData,
    DEMData, SeismicData, TimeSeries, GeoCoordinate, BoundingBox,
    RiskLevel, RockfallPrediction
)

logger = logging.getLogger(__name__)


class ValidationError(Exception):
    """Custom exception for data validation errors."""
    pass


def validate_file_exists(file_path: str) -> bool:
    """
    Validate that a file exists and is accessible.
    
    Args:
        file_path: Path to the file to validate
        
    Returns:
        True if file exists and is readable
        
    Raises:
        ValidationError: If file doesn't exist or isn't readable
    """
    if not os.path.exists(file_path):
        raise ValidationError(f"File does not exist: {file_path}")
    if not os.path.isfile(file_path):
        raise ValidationError(f"Path is not a file: {file_path}")
    if not os.access(file_path, os.R_OK):
        raise ValidationError(f"File is not readable: {file_path}")
    return True


def validate_image_data(image_data: ImageData) -> bool:
    """
    Validate ImageData object for completeness and correctness.
    
    Args:
        image_data: ImageData object to validate
        
    Returns:
        True if validation passes
        
    Raises:
        ValidationError: If validation fails
    """
    if not isinstance(image_data, ImageData):
        raise ValidationError("Input must be an ImageData object")
    
    # Validate image file exists
    validate_file_exists(image_data.image_path)
    
    # Validate image file extension
    valid_extensions = {'.jpg', '.jpeg', '.png', '.tiff', '.tif'}
    file_ext = os.path.splitext(image_data.image_path)[1].lower()
    if file_ext not in valid_extensions:
        raise ValidationError(f"Invalid image format: {file_ext}. Must be one of {valid_extensions}")
    
    # Validate annotations
    for i, bbox in enumerate(image_data.annotations):
        if not isinstance(bbox, BoundingBox):
            raise ValidationError(f"Annotation {i} must be a BoundingBox object")
    
    return True


def validate_sensor_data(sensor_data: SensorData) -> bool:
    """
    Validate SensorData object for completeness and correctness.
    
    Args:
        sensor_data: SensorData object to validate
        
    Returns:
        True if validation passes
        
    Raises:
        ValidationError: If validation fails
    """
    if not isinstance(sensor_data, SensorData):
        raise ValidationError("Input must be a SensorData object")
    
    # Validate at least one measurement exists
    measurements = [sensor_data.displacement, sensor_data.strain, sensor_data.pore_pressure]
    if all(m is None for m in measurements):
        raise ValidationError("At least one sensor measurement must be provided")
    
    # Validate each time series
    for name, ts in [("displacement", sensor_data.displacement), 
                     ("strain", sensor_data.strain), 
                     ("pore_pressure", sensor_data.pore_pressure)]:
        if ts is not None:
            validate_time_series(ts, name)
    
    return True


def validate_time_series(time_series: TimeSeries, series_name: str = "time_series") -> bool:
    """
    Validate TimeSeries object for completeness and correctness.
    
    Args:
        time_series: TimeSeries object to validate
        series_name: Name of the series for error messages
        
    Returns:
        True if validation passes
        
    Raises:
        ValidationError: If validation fails
    """
    if not isinstance(time_series, TimeSeries):
        raise ValidationError(f"{series_name} must be a TimeSeries object")
    
    # Check for monotonic timestamps
    timestamps = time_series.timestamps
    for i in range(1, len(timestamps)):
        if timestamps[i] <= timestamps[i-1]:
            raise ValidationError(f"{series_name} timestamps must be in ascending order")
    
    # Check for reasonable value ranges
    values = time_series.values
    if any(not np.isfinite(v) for v in values):
        raise ValidationError(f"{series_name} contains non-finite values (NaN or Inf)")
    
    # Check for extreme outliers (values beyond 6 standard deviations)
    if len(values) > 1:
        mean_val = np.mean(values)
        std_val = np.std(values)
        if std_val > 0:
            outliers = [v for v in values if abs(v - mean_val) > 6 * std_val]
            if outliers:
                logger.warning(f"{series_name} contains {len(outliers)} potential outliers")
    
    return True


def validate_environmental_data(env_data: EnvironmentalData) -> bool:
    """
    Validate EnvironmentalData object for completeness and correctness.
    
    Args:
        env_data: EnvironmentalData object to validate
        
    Returns:
        True if validation passes
        
    Raises:
        ValidationError: If validation fails
    """
    if not isinstance(env_data, EnvironmentalData):
        raise ValidationError("Input must be an EnvironmentalData object")
    
    # Check for reasonable value ranges
    if env_data.temperature is not None:
        if not -50 <= env_data.temperature <= 60:  # Celsius
            logger.warning(f"Temperature {env_data.temperature}°C is outside typical range (-50 to 60°C)")
    
    if env_data.wind_speed is not None:
        if env_data.wind_speed > 200:  # km/h
            logger.warning(f"Wind speed {env_data.wind_speed} km/h is extremely high")
    
    if env_data.vibrations is not None:
        if env_data.vibrations < 0:
            raise ValidationError("Vibration measurements cannot be negative")
    
    return True


def validate_dem_data(dem_data: DEMData) -> bool:
    """
    Validate DEMData object for completeness and correctness.
    
    Args:
        dem_data: DEMData object to validate
        
    Returns:
        True if validation passes
        
    Raises:
        ValidationError: If validation fails
    """
    if not isinstance(dem_data, DEMData):
        raise ValidationError("Input must be a DEMData object")
    
    # Validate elevation matrix
    if dem_data.elevation_matrix.size == 0:
        raise ValidationError("DEM elevation matrix cannot be empty")
    
    # Check for reasonable elevation values
    min_elev = np.min(dem_data.elevation_matrix)
    max_elev = np.max(dem_data.elevation_matrix)
    
    if min_elev < -500 or max_elev > 9000:  # meters
        logger.warning(f"DEM elevations ({min_elev:.1f} to {max_elev:.1f}m) are outside typical range")
    
    # Check for NaN values
    if np.any(np.isnan(dem_data.elevation_matrix)):
        raise ValidationError("DEM elevation matrix contains NaN values")
    
    return True


def validate_seismic_data(seismic_data: SeismicData) -> bool:
    """
    Validate SeismicData object for completeness and correctness.
    
    Args:
        seismic_data: SeismicData object to validate
        
    Returns:
        True if validation passes
        
    Raises:
        ValidationError: If validation fails
    """
    if not isinstance(seismic_data, SeismicData):
        raise ValidationError("Input must be a SeismicData object")
    
    # Validate signal data
    if seismic_data.signal.size == 0:
        raise ValidationError("Seismic signal cannot be empty")
    
    # Check for reasonable sampling rate
    if not 1 <= seismic_data.sampling_rate <= 10000:  # Hz
        logger.warning(f"Sampling rate {seismic_data.sampling_rate} Hz is outside typical range (1-10000 Hz)")
    
    # Check for NaN or infinite values
    if np.any(~np.isfinite(seismic_data.signal)):
        raise ValidationError("Seismic signal contains non-finite values")
    
    return True


def validate_rockfall_datapoint(datapoint: RockfallDataPoint) -> bool:
    """
    Validate complete RockfallDataPoint for schema compliance.
    
    Args:
        datapoint: RockfallDataPoint object to validate
        
    Returns:
        True if validation passes
        
    Raises:
        ValidationError: If validation fails
    """
    if not isinstance(datapoint, RockfallDataPoint):
        raise ValidationError("Input must be a RockfallDataPoint object")
    
    # Validate required fields
    if not isinstance(datapoint.timestamp, datetime):
        raise ValidationError("Timestamp must be a datetime object")
    
    if not isinstance(datapoint.location, GeoCoordinate):
        raise ValidationError("Location must be a GeoCoordinate object")
    
    # Validate optional data components
    if datapoint.imagery is not None:
        validate_image_data(datapoint.imagery)
    
    if datapoint.sensor_readings is not None:
        validate_sensor_data(datapoint.sensor_readings)
    
    if datapoint.environmental is not None:
        validate_environmental_data(datapoint.environmental)
    
    if datapoint.dem_data is not None:
        validate_dem_data(datapoint.dem_data)
    
    if datapoint.seismic is not None:
        validate_seismic_data(datapoint.seismic)
    
    # Check that at least some data is provided
    data_sources = [
        datapoint.imagery, datapoint.sensor_readings, datapoint.environmental,
        datapoint.dem_data, datapoint.seismic
    ]
    if all(source is None for source in data_sources):
        raise ValidationError("At least one data source must be provided")
    
    return True


def validate_prediction_output(prediction: RockfallPrediction) -> bool:
    """
    Validate RockfallPrediction output for correctness.
    
    Args:
        prediction: RockfallPrediction object to validate
        
    Returns:
        True if validation passes
        
    Raises:
        ValidationError: If validation fails
    """
    if not isinstance(prediction, RockfallPrediction):
        raise ValidationError("Input must be a RockfallPrediction object")
    
    # Validate contributing factors sum to reasonable value
    if prediction.contributing_factors:
        factor_sum = sum(abs(v) for v in prediction.contributing_factors.values())
        if factor_sum > 2.0:  # Allow some flexibility for feature importance
            logger.warning(f"Contributing factors sum to {factor_sum:.2f}, which seems high")
    
    return True


def validate_batch_data(datapoints: List[RockfallDataPoint]) -> Dict[str, Any]:
    """
    Validate a batch of RockfallDataPoint objects and return summary statistics.
    
    Args:
        datapoints: List of RockfallDataPoint objects to validate
        
    Returns:
        Dictionary containing validation summary and statistics
        
    Raises:
        ValidationError: If critical validation errors are found
    """
    if not datapoints:
        raise ValidationError("Batch cannot be empty")
    
    validation_summary = {
        'total_points': len(datapoints),
        'valid_points': 0,
        'errors': [],
        'warnings': [],
        'data_coverage': {
            'imagery': 0,
            'sensor_readings': 0,
            'environmental': 0,
            'dem_data': 0,
            'seismic': 0
        }
    }
    
    for i, datapoint in enumerate(datapoints):
        try:
            validate_rockfall_datapoint(datapoint)
            validation_summary['valid_points'] += 1
            
            # Count data coverage
            if datapoint.imagery is not None:
                validation_summary['data_coverage']['imagery'] += 1
            if datapoint.sensor_readings is not None:
                validation_summary['data_coverage']['sensor_readings'] += 1
            if datapoint.environmental is not None:
                validation_summary['data_coverage']['environmental'] += 1
            if datapoint.dem_data is not None:
                validation_summary['data_coverage']['dem_data'] += 1
            if datapoint.seismic is not None:
                validation_summary['data_coverage']['seismic'] += 1
                
        except ValidationError as e:
            validation_summary['errors'].append(f"Point {i}: {str(e)}")
        except Exception as e:
            validation_summary['errors'].append(f"Point {i}: Unexpected error - {str(e)}")
    
    # Calculate coverage percentages
    total = validation_summary['total_points']
    for key in validation_summary['data_coverage']:
        count = validation_summary['data_coverage'][key]
        validation_summary['data_coverage'][key] = {
            'count': count,
            'percentage': (count / total) * 100 if total > 0 else 0
        }
    
    return validation_summary