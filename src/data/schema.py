"""
Data schema definitions for the Rockfall Prediction System.

This module defines the core data structures used throughout the system
for handling multi-modal mining data including imagery, sensor readings,
environmental data, and seismic information.
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import List, Optional, Dict, Any, Union
import numpy as np


class RiskLevel(Enum):
    """Risk level classification for rockfall predictions."""
    LOW = 0
    MEDIUM = 1
    HIGH = 2


@dataclass
class GeoCoordinate:
    """Geographic coordinate representation."""
    latitude: float
    longitude: float
    elevation: Optional[float] = None
    
    def __post_init__(self):
        """Validate coordinate values."""
        if not -90 <= self.latitude <= 90:
            raise ValueError(f"Invalid latitude: {self.latitude}. Must be between -90 and 90.")
        if not -180 <= self.longitude <= 180:
            raise ValueError(f"Invalid longitude: {self.longitude}. Must be between -180 and 180.")


@dataclass
class BoundingBox:
    """Bounding box for object detection annotations."""
    x: float
    y: float
    width: float
    height: float
    class_name: str
    confidence: Optional[float] = None
    
    def __post_init__(self):
        """Validate bounding box parameters."""
        if self.width <= 0 or self.height <= 0:
            raise ValueError("Width and height must be positive values.")
        if self.confidence is not None and not 0 <= self.confidence <= 1:
            raise ValueError("Confidence must be between 0 and 1.")


@dataclass
class ImageMetadata:
    """Metadata for image data."""
    resolution: tuple
    capture_date: datetime
    sensor_type: str
    weather_conditions: Optional[str] = None
    
    def __post_init__(self):
        """Validate image metadata."""
        if len(self.resolution) != 2 or any(r <= 0 for r in self.resolution):
            raise ValueError("Resolution must be a tuple of two positive integers.")


@dataclass
class ImageData:
    """Container for image data and annotations."""
    image_path: str
    annotations: List[BoundingBox] = field(default_factory=list)
    metadata: Optional[ImageMetadata] = None
    
    def __post_init__(self):
        """Validate image data."""
        if not self.image_path:
            raise ValueError("Image path cannot be empty.")


@dataclass
class TimeSeries:
    """Time series data container."""
    timestamps: List[datetime]
    values: List[float]
    unit: str
    
    def __post_init__(self):
        """Validate time series data."""
        if len(self.timestamps) != len(self.values):
            raise ValueError("Timestamps and values must have the same length.")
        if len(self.timestamps) == 0:
            raise ValueError("Time series cannot be empty.")
        if not all(isinstance(ts, datetime) for ts in self.timestamps):
            raise ValueError("All timestamps must be datetime objects.")


@dataclass
class SensorData:
    """Geotechnical sensor measurements."""
    displacement: Optional[TimeSeries] = None
    strain: Optional[TimeSeries] = None
    pore_pressure: Optional[TimeSeries] = None
    
    def __post_init__(self):
        """Validate sensor data."""
        if all(data is None for data in [self.displacement, self.strain, self.pore_pressure]):
            raise ValueError("At least one sensor measurement must be provided.")


@dataclass
class EnvironmentalData:
    """Environmental conditions data."""
    rainfall: Optional[float] = None
    temperature: Optional[float] = None
    vibrations: Optional[float] = None
    wind_speed: Optional[float] = None
    humidity: Optional[float] = None
    
    def __post_init__(self):
        """Validate environmental data."""
        if self.rainfall is not None and self.rainfall < 0:
            raise ValueError("Rainfall cannot be negative.")
        if self.temperature is not None and self.temperature < -273.15:
            raise ValueError("Temperature cannot be below absolute zero.")
        if self.wind_speed is not None and self.wind_speed < 0:
            raise ValueError("Wind speed cannot be negative.")
        if self.humidity is not None and not 0 <= self.humidity <= 100:
            raise ValueError("Humidity must be between 0 and 100 percent.")


@dataclass
class DEMData:
    """Digital Elevation Model data."""
    elevation_matrix: np.ndarray
    resolution: float  # meters per pixel
    bounds: tuple  # (min_lat, min_lon, max_lat, max_lon)
    
    def __post_init__(self):
        """Validate DEM data."""
        if self.elevation_matrix.ndim != 2:
            raise ValueError("Elevation matrix must be 2-dimensional.")
        if self.resolution <= 0:
            raise ValueError("Resolution must be positive.")
        if len(self.bounds) != 4:
            raise ValueError("Bounds must contain exactly 4 values (min_lat, min_lon, max_lat, max_lon).")


@dataclass
class SeismicData:
    """Seismic signal data."""
    signal: np.ndarray
    sampling_rate: float  # Hz
    start_time: datetime
    station_id: str
    
    def __post_init__(self):
        """Validate seismic data."""
        if self.signal.ndim != 1:
            raise ValueError("Seismic signal must be 1-dimensional.")
        if self.sampling_rate <= 0:
            raise ValueError("Sampling rate must be positive.")
        if not self.station_id:
            raise ValueError("Station ID cannot be empty.")


@dataclass
class RockfallDataPoint:
    """Complete data point for rockfall prediction."""
    timestamp: datetime
    location: GeoCoordinate
    imagery: Optional[ImageData] = None
    dem_data: Optional[DEMData] = None
    sensor_readings: Optional[SensorData] = None
    environmental: Optional[EnvironmentalData] = None
    seismic: Optional[SeismicData] = None
    ground_truth: Optional[RiskLevel] = None
    
    def __post_init__(self):
        """Validate rockfall data point."""
        if not isinstance(self.timestamp, datetime):
            raise ValueError("Timestamp must be a datetime object.")
        if not isinstance(self.location, GeoCoordinate):
            raise ValueError("Location must be a GeoCoordinate object.")


@dataclass
class RockfallPrediction:
    """Model prediction output."""
    risk_level: RiskLevel
    confidence_score: float
    contributing_factors: Dict[str, float] = field(default_factory=dict)
    uncertainty_estimate: Optional[float] = None
    model_version: str = "1.0.0"
    
    def __post_init__(self):
        """Validate prediction output."""
        if not 0 <= self.confidence_score <= 1:
            raise ValueError("Confidence score must be between 0 and 1.")
        if self.uncertainty_estimate is not None and not 0 <= self.uncertainty_estimate <= 1:
            raise ValueError("Uncertainty estimate must be between 0 and 1.")
        if not isinstance(self.risk_level, RiskLevel):
            raise ValueError("Risk level must be a RiskLevel enum value.")