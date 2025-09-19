"""Data schemas and models for the rockfall prediction system."""

from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import List, Optional, Dict, Any
import numpy as np


class RiskLevel(Enum):
    """Risk level classifications for rockfall predictions."""
    LOW = 0
    MEDIUM = 1
    HIGH = 2


@dataclass
class GeoCoordinate:
    """Geographic coordinate representation."""
    latitude: float
    longitude: float
    elevation: Optional[float] = None


@dataclass
class BoundingBox:
    """Bounding box for object detection annotations."""
    x: float
    y: float
    width: float
    height: float
    class_id: int
    confidence: Optional[float] = None


@dataclass
class ImageMetadata:
    """Metadata for image data."""
    width: int
    height: int
    channels: int
    resolution: Optional[float] = None
    capture_date: Optional[datetime] = None


@dataclass
class ImageData:
    """Image data with annotations and metadata."""
    image_path: str
    annotations: List[BoundingBox]
    metadata: ImageMetadata


@dataclass
class TimeSeries:
    """Time series data representation."""
    timestamps: np.ndarray
    values: np.ndarray
    unit: str
    sampling_rate: Optional[float] = None


@dataclass
class SensorData:
    """Geotechnical sensor measurements."""
    displacement: Optional[TimeSeries] = None
    strain: Optional[TimeSeries] = None
    pore_pressure: Optional[TimeSeries] = None


@dataclass
class EnvironmentalData:
    """Environmental measurements."""
    rainfall: Optional[float] = None
    temperature: Optional[float] = None
    vibrations: Optional[float] = None
    wind_speed: Optional[float] = None


@dataclass
class SeismicData:
    """Seismic signal data."""
    signal: np.ndarray
    sampling_rate: float
    start_time: datetime
    station_id: str


@dataclass
class DEMData:
    """Digital Elevation Model data."""
    elevation_matrix: np.ndarray
    resolution: float
    bounds: Dict[str, float]  # {'north', 'south', 'east', 'west'}


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


@dataclass
class RockfallPrediction:
    """Model prediction output."""
    risk_level: RiskLevel
    confidence_score: float
    contributing_factors: Dict[str, float]
    uncertainty_estimate: float
    model_version: str
    prediction_timestamp: datetime