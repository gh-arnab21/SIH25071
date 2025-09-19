"""
Data module for the Rockfall Prediction System.

This module provides data schema definitions, validation functions,
and utility functions for handling multi-modal mining data.
"""

from .schema import (
    RockfallDataPoint,
    ImageData,
    SensorData,
    EnvironmentalData,
    DEMData,
    SeismicData,
    TimeSeries,
    GeoCoordinate,
    BoundingBox,
    RiskLevel,
    RockfallPrediction,
    ImageMetadata
)

from .validation import (
    ValidationError,
    validate_file_exists,
    validate_image_data,
    validate_sensor_data,
    validate_time_series,
    validate_environmental_data,
    validate_dem_data,
    validate_seismic_data,
    validate_rockfall_datapoint,
    validate_prediction_output,
    validate_batch_data
)

from .utils import (
    normalize_coordinates,
    convert_timestamp_to_datetime,
    normalize_array,
    convert_dict_to_timeseries,
    convert_json_to_bounding_boxes,
    convert_yolo_to_bounding_box,
    convert_dataframe_to_environmental_data,
    impute_missing_values,
    detect_outliers,
    convert_risk_level_string,
    create_sample_datapoint
)

__all__ = [
    # Schema classes
    'RockfallDataPoint',
    'ImageData',
    'SensorData',
    'EnvironmentalData',
    'DEMData',
    'SeismicData',
    'TimeSeries',
    'GeoCoordinate',
    'BoundingBox',
    'RiskLevel',
    'RockfallPrediction',
    'ImageMetadata',
    
    # Validation functions
    'ValidationError',
    'validate_file_exists',
    'validate_image_data',
    'validate_sensor_data',
    'validate_time_series',
    'validate_environmental_data',
    'validate_dem_data',
    'validate_seismic_data',
    'validate_rockfall_datapoint',
    'validate_prediction_output',
    'validate_batch_data',
    
    # Utility functions
    'normalize_coordinates',
    'convert_timestamp_to_datetime',
    'normalize_array',
    'convert_dict_to_timeseries',
    'convert_json_to_bounding_boxes',
    'convert_yolo_to_bounding_box',
    'convert_dataframe_to_environmental_data',
    'impute_missing_values',
    'detect_outliers',
    'convert_risk_level_string',
    'create_sample_datapoint'
]