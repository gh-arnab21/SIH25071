"""
Dataset Integration and Loading System for Rockfall Prediction.

This module provides specialized data loaders for different dataset formats
including Open Pit Mine Object Detection Dataset, RockNet Seismic Dataset,
and Brazilian Rockfall Slope Dataset.
"""

import os
import json
import csv
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional, Union, Tuple
import numpy as np
import pandas as pd
from datetime import datetime
import warnings

# Local imports
from ..data.schema import (
    RockfallDataPoint, ImageData, SensorData, EnvironmentalData,
    DEMData, SeismicData, TimeSeries, GeoCoordinate, BoundingBox,
    ImageMetadata, RiskLevel
)
from ..data.validation import validate_rockfall_datapoint, ValidationError

# Optional imports (might not be available in test environment)
try:
    from ..data.processors.seismic_processor import SeismicProcessor
    SEISMIC_AVAILABLE = True
except ImportError:
    SeismicProcessor = None
    SEISMIC_AVAILABLE = False

try:
    from ..data.processors.image_processor import ImagePreprocessor
    IMAGE_AVAILABLE = True
except ImportError:
    ImagePreprocessor = None
    IMAGE_AVAILABLE = False

logger = logging.getLogger(__name__)


class DatasetIntegrationError(Exception):
    """Exception raised during dataset integration process."""
    pass


class BaseDatasetLoader:
    """Base class for dataset loaders."""
    
    def __init__(self, dataset_path: Union[str, Path]):
        """Initialize base dataset loader."""
        self.dataset_path = Path(dataset_path)
        self.supported_extensions = []
        self.dataset_info = {}
        
    def validate_dataset(self) -> bool:
        """Validate dataset structure and contents."""
        if not self.dataset_path.exists():
            raise DatasetIntegrationError(f"Dataset path does not exist: {self.dataset_path}")
        return True
    
    def get_dataset_info(self) -> Dict[str, Any]:
        """Get information about the loaded dataset."""
        return self.dataset_info
    
    def load_data(self) -> List[RockfallDataPoint]:
        """Load data points from the dataset."""
        raise NotImplementedError("Subclasses must implement load_data method")


class OpenPitMineDatasetLoader(BaseDatasetLoader):
    """
    Loader for Open Pit Mine Object Detection Dataset.
    
    This dataset typically contains:
    - Satellite imagery (JPG/PNG)
    - JSON annotations with object detection bounding boxes
    - Metadata about mining operations and locations
    """
    
    def __init__(self, dataset_path: Union[str, Path]):
        """Initialize Open Pit Mine dataset loader."""
        super().__init__(dataset_path)
        self.supported_extensions = ['.jpg', '.jpeg', '.png', '.json']
        self.images_dir = self.dataset_path / "images"
        self.annotations_dir = self.dataset_path / "annotations"
        
    def validate_dataset(self) -> bool:
        """Validate Open Pit Mine dataset structure."""
        super().validate_dataset()
        
        if not self.images_dir.exists():
            raise DatasetIntegrationError(f"Images directory not found: {self.images_dir}")
        
        if not self.annotations_dir.exists():
            raise DatasetIntegrationError(f"Annotations directory not found: {self.annotations_dir}")
        
        return True
    
    def load_data(self) -> List[RockfallDataPoint]:
        """Load Open Pit Mine dataset."""
        logger.info(f"Loading Open Pit Mine dataset from {self.dataset_path}")
        
        self.validate_dataset()
        data_points = []
        
        # Find all annotation files
        annotation_files = list(self.annotations_dir.glob("*.json"))
        
        for annotation_file in annotation_files:
            try:
                data_point = self._load_annotation_file(annotation_file)
                if data_point:
                    data_points.append(data_point)
            except Exception as e:
                logger.warning(f"Failed to load annotation {annotation_file}: {e}")
        
        self.dataset_info = {
            'dataset_type': 'open_pit_mine_object_detection',
            'total_samples': len(data_points),
            'images_directory': str(self.images_dir),
            'annotations_directory': str(self.annotations_dir),
            'annotation_files_processed': len(annotation_files)
        }
        
        logger.info(f"Loaded {len(data_points)} data points from Open Pit Mine dataset")
        return data_points
    
    def _load_annotation_file(self, annotation_file: Path) -> Optional[RockfallDataPoint]:
        """Load a single annotation file and create RockfallDataPoint."""
        try:
            with open(annotation_file, 'r') as f:
                annotation_data = json.load(f)
            
            # Extract image information
            image_filename = annotation_data.get('filename', annotation_file.stem + '.jpg')
            image_path = self.images_dir / image_filename
            
            if not image_path.exists():
                # Try different extensions
                for ext in ['.jpg', '.jpeg', '.png']:
                    alt_path = self.images_dir / (annotation_file.stem + ext)
                    if alt_path.exists():
                        image_path = alt_path
                        break
                else:
                    logger.warning(f"Image file not found for annotation: {annotation_file}")
                    return None
            
            # Parse annotations
            annotations = []
            for obj in annotation_data.get('objects', []):
                bbox = BoundingBox(
                    x=float(obj.get('x', 0)),
                    y=float(obj.get('y', 0)),
                    width=float(obj.get('width', 0)),
                    height=float(obj.get('height', 0)),
                    class_name=obj.get('class', 'unknown'),
                    confidence=obj.get('confidence')
                )
                annotations.append(bbox)
            
            # Create image metadata
            metadata = ImageMetadata(
                resolution=annotation_data.get('size', (1024, 1024)),
                capture_date=datetime.now(),  # Use current time if not provided
                sensor_type=annotation_data.get('sensor_type', 'satellite'),
                weather_conditions=annotation_data.get('weather_conditions')
            )
            
            # Create ImageData
            image_data = ImageData(
                image_path=str(image_path),
                annotations=annotations,
                metadata=metadata
            )
            
            # Extract location information
            location_data = annotation_data.get('location', {})
            location = GeoCoordinate(
                latitude=float(location_data.get('latitude', 0.0)),
                longitude=float(location_data.get('longitude', 0.0)),
                elevation=location_data.get('elevation')
            )
            
            # Create RockfallDataPoint
            data_point = RockfallDataPoint(
                timestamp=datetime.now(),
                location=location,
                imagery=image_data,
                ground_truth=self._parse_risk_level(annotation_data.get('risk_level'))
            )
            
            return data_point
            
        except Exception as e:
            logger.error(f"Error loading annotation file {annotation_file}: {e}")
            return None
    
    def _parse_risk_level(self, risk_value: Any) -> Optional[RiskLevel]:
        """Parse risk level from annotation data."""
        if risk_value is None:
            return None
        
        if isinstance(risk_value, str):
            risk_value = risk_value.upper()
            if risk_value in ['LOW', 'L', '0']:
                return RiskLevel.LOW
            elif risk_value in ['MEDIUM', 'MED', 'M', '1']:
                return RiskLevel.MEDIUM
            elif risk_value in ['HIGH', 'H', '2']:
                return RiskLevel.HIGH
        elif isinstance(risk_value, (int, float)):
            if risk_value == 0:
                return RiskLevel.LOW
            elif risk_value == 1:
                return RiskLevel.MEDIUM
            elif risk_value == 2:
                return RiskLevel.HIGH
        
        return None


class RockNetSeismicDatasetLoader(BaseDatasetLoader):
    """
    Loader for RockNet Seismic Dataset.
    
    This dataset typically contains:
    - SAC seismic files with waveform data
    - Metadata files with station information
    - Event catalogs with rockfall occurrence times
    """
    
    def __init__(self, dataset_path: Union[str, Path]):
        """Initialize RockNet Seismic dataset loader."""
        super().__init__(dataset_path)
        self.supported_extensions = ['.sac', '.SAC']
        
        # Initialize seismic processor if available
        if SEISMIC_AVAILABLE:
            self.seismic_processor = SeismicProcessor()
        else:
            self.seismic_processor = None
            logger.warning("SeismicProcessor not available. Seismic processing will be limited.")
        
    def validate_dataset(self) -> bool:
        """Validate RockNet Seismic dataset structure."""
        super().validate_dataset()
        
        # Check for SAC files
        sac_files = list(self.dataset_path.rglob("*.sac")) + list(self.dataset_path.rglob("*.SAC"))
        if not sac_files:
            raise DatasetIntegrationError(f"No SAC files found in dataset: {self.dataset_path}")
        
        return True
    
    def load_data(self) -> List[RockfallDataPoint]:
        """Load RockNet Seismic dataset."""
        logger.info(f"Loading RockNet Seismic dataset from {self.dataset_path}")
        
        self.validate_dataset()
        data_points = []
        
        # Find all SAC files
        sac_files = list(self.dataset_path.rglob("*.sac")) + list(self.dataset_path.rglob("*.SAC"))
        
        # Load event catalog if available
        event_catalog = self._load_event_catalog()
        
        for sac_file in sac_files:
            try:
                data_point = self._load_sac_file(sac_file, event_catalog)
                if data_point:
                    data_points.append(data_point)
            except Exception as e:
                logger.warning(f"Failed to load SAC file {sac_file}: {e}")
        
        self.dataset_info = {
            'dataset_type': 'rocknet_seismic',
            'total_samples': len(data_points),
            'sac_files_processed': len(sac_files),
            'event_catalog_loaded': event_catalog is not None
        }
        
        logger.info(f"Loaded {len(data_points)} data points from RockNet Seismic dataset")
        return data_points
    
    def _load_event_catalog(self) -> Optional[Dict[str, Any]]:
        """Load event catalog if available."""
        catalog_files = list(self.dataset_path.glob("*catalog*.csv")) + \
                      list(self.dataset_path.glob("*events*.csv"))
        
        if not catalog_files:
            return None
        
        try:
            catalog_df = pd.read_csv(catalog_files[0])
            catalog = {}
            
            for _, row in catalog_df.iterrows():
                timestamp = pd.to_datetime(row.get('timestamp', row.get('time', '')))
                station_id = row.get('station_id', row.get('station', ''))
                
                if timestamp and station_id:
                    key = f"{station_id}_{timestamp.strftime('%Y%m%d_%H%M%S')}"
                    catalog[key] = {
                        'timestamp': timestamp,
                        'magnitude': row.get('magnitude'),
                        'event_type': row.get('event_type', 'rockfall'),
                        'confidence': row.get('confidence')
                    }
            
            logger.info(f"Loaded event catalog with {len(catalog)} events")
            return catalog
            
        except Exception as e:
            logger.warning(f"Failed to load event catalog: {e}")
            return None
    
    def _load_sac_file(self, sac_file: Path, event_catalog: Optional[Dict[str, Any]] = None) -> Optional[RockfallDataPoint]:
        """Load a single SAC file and create RockfallDataPoint."""
        try:
            # Read SAC file using SeismicProcessor if available
            if self.seismic_processor is None:
                logger.warning(f"SeismicProcessor not available, skipping {sac_file}")
                return None
                
            seismic_data = self.seismic_processor.read_sac_file(str(sac_file))
            
            if seismic_data is None:
                return None
            
            # Extract location from file path or metadata
            location = self._extract_location_from_path(sac_file)
            
            # Check event catalog for ground truth
            ground_truth = None
            if event_catalog:
                file_key = self._generate_file_key(sac_file, seismic_data)
                if file_key in event_catalog:
                    event_info = event_catalog[file_key]
                    # Determine risk level based on event type or magnitude
                    if event_info.get('event_type') == 'rockfall':
                        magnitude = event_info.get('magnitude', 0)
                        if magnitude < 1.0:
                            ground_truth = RiskLevel.LOW
                        elif magnitude < 2.0:
                            ground_truth = RiskLevel.MEDIUM
                        else:
                            ground_truth = RiskLevel.HIGH
            
            # Create RockfallDataPoint
            data_point = RockfallDataPoint(
                timestamp=seismic_data.start_time,
                location=location,
                seismic=seismic_data,
                ground_truth=ground_truth
            )
            
            return data_point
            
        except Exception as e:
            logger.error(f"Error loading SAC file {sac_file}: {e}")
            return None
    
    def _extract_location_from_path(self, sac_file: Path) -> GeoCoordinate:
        """Extract location information from file path or name."""
        # Try to parse location from filename
        filename = sac_file.stem
        
        # Look for coordinate patterns in filename
        parts = filename.split('_')
        lat, lon = 0.0, 0.0
        
        for part in parts:
            if part.startswith('lat'):
                try:
                    lat = float(part[3:])
                except ValueError:
                    pass
            elif part.startswith('lon'):
                try:
                    lon = float(part[3:])
                except ValueError:
                    pass
        
        return GeoCoordinate(latitude=lat, longitude=lon)
    
    def _generate_file_key(self, sac_file: Path, seismic_data: SeismicData) -> str:
        """Generate key for event catalog lookup."""
        timestamp = seismic_data.start_time.strftime('%Y%m%d_%H%M%S')
        station_parts = seismic_data.station_id.split('.')
        station = station_parts[1] if len(station_parts) > 1 else seismic_data.station_id
        return f"{station}_{timestamp}"


class BrazilianRockfallDatasetLoader(BaseDatasetLoader):
    """
    Loader for Brazilian Rockfall Slope Dataset.
    
    This dataset typically contains:
    - CSV files with structured slope stability data
    - Environmental measurements
    - Geological survey data
    - Risk assessment records
    """
    
    def __init__(self, dataset_path: Union[str, Path]):
        """Initialize Brazilian Rockfall dataset loader."""
        super().__init__(dataset_path)
        self.supported_extensions = ['.csv', '.xlsx']
        
    def validate_dataset(self) -> bool:
        """Validate Brazilian Rockfall dataset structure."""
        super().validate_dataset()
        
        # Check for CSV files
        csv_files = list(self.dataset_path.glob("*.csv")) + list(self.dataset_path.glob("*.xlsx"))
        if not csv_files:
            raise DatasetIntegrationError(f"No CSV/Excel files found in dataset: {self.dataset_path}")
        
        return True
    
    def load_data(self) -> List[RockfallDataPoint]:
        """Load Brazilian Rockfall dataset."""
        logger.info(f"Loading Brazilian Rockfall dataset from {self.dataset_path}")
        
        self.validate_dataset()
        data_points = []
        
        # Find all data files
        data_files = list(self.dataset_path.glob("*.csv")) + list(self.dataset_path.glob("*.xlsx"))
        
        for data_file in data_files:
            try:
                file_data_points = self._load_data_file(data_file)
                data_points.extend(file_data_points)
            except Exception as e:
                logger.warning(f"Failed to load data file {data_file}: {e}")
        
        self.dataset_info = {
            'dataset_type': 'brazilian_rockfall_slope',
            'total_samples': len(data_points),
            'data_files_processed': len(data_files)
        }
        
        logger.info(f"Loaded {len(data_points)} data points from Brazilian Rockfall dataset")
        return data_points
    
    def _load_data_file(self, data_file: Path) -> List[RockfallDataPoint]:
        """Load a single data file and create RockfallDataPoints."""
        try:
            # Load data based on file extension
            if data_file.suffix.lower() == '.csv':
                df = pd.read_csv(data_file)
            else:  # Excel file
                df = pd.read_excel(data_file)
            
            data_points = []
            
            for _, row in df.iterrows():
                try:
                    data_point = self._row_to_datapoint(row)
                    if data_point:
                        data_points.append(data_point)
                except Exception as e:
                    logger.warning(f"Failed to process row: {e}")
            
            return data_points
            
        except Exception as e:
            logger.error(f"Error loading data file {data_file}: {e}")
            return []
    
    def _row_to_datapoint(self, row: pd.Series) -> Optional[RockfallDataPoint]:
        """Convert DataFrame row to RockfallDataPoint."""
        try:
            # Extract location
            latitude = float(row.get('latitude', row.get('lat', 0.0)))
            longitude = float(row.get('longitude', row.get('lon', row.get('lng', 0.0))))
            elevation = row.get('elevation', row.get('alt'))
            
            location = GeoCoordinate(
                latitude=latitude,
                longitude=longitude,
                elevation=float(elevation) if elevation is not None else None
            )
            
            # Extract timestamp
            timestamp_col = None
            for col in ['timestamp', 'date', 'time', 'datetime']:
                if col in row.index:
                    timestamp_col = col
                    break
            
            if timestamp_col:
                timestamp = pd.to_datetime(row[timestamp_col])
            else:
                timestamp = datetime.now()
            
            # Extract environmental data
            environmental = EnvironmentalData(
                rainfall=self._safe_float(row.get('rainfall', row.get('precipitation'))),
                temperature=self._safe_float(row.get('temperature', row.get('temp'))),
                vibrations=self._safe_float(row.get('vibrations', row.get('vibration'))),
                wind_speed=self._safe_float(row.get('wind_speed', row.get('wind'))),
                humidity=self._safe_float(row.get('humidity'))
            )
            
            # Extract sensor data if available
            sensor_data = None
            displacement_val = self._safe_float(row.get('displacement'))
            strain_val = self._safe_float(row.get('strain'))
            pore_pressure_val = self._safe_float(row.get('pore_pressure'))
            
            if any([displacement_val, strain_val, pore_pressure_val]):
                # Create simple time series with single values
                ts_timestamp = [timestamp]
                
                sensor_data = SensorData(
                    displacement=TimeSeries(
                        timestamps=ts_timestamp,
                        values=[displacement_val] if displacement_val is not None else [],
                        unit="mm"
                    ) if displacement_val is not None else None,
                    strain=TimeSeries(
                        timestamps=ts_timestamp,
                        values=[strain_val] if strain_val is not None else [],
                        unit="microstrains"
                    ) if strain_val is not None else None,
                    pore_pressure=TimeSeries(
                        timestamps=ts_timestamp,
                        values=[pore_pressure_val] if pore_pressure_val is not None else [],
                        unit="kPa"
                    ) if pore_pressure_val is not None else None
                )
            
            # Extract ground truth risk level
            ground_truth = None
            risk_col = None
            for col in ['risk_level', 'risk', 'classification', 'label']:
                if col in row.index:
                    risk_col = col
                    break
            
            if risk_col:
                risk_value = row[risk_col]
                if isinstance(risk_value, str):
                    risk_value = risk_value.upper()
                    if risk_value in ['LOW', 'L', '0', 'STABLE']:
                        ground_truth = RiskLevel.LOW
                    elif risk_value in ['MEDIUM', 'MED', 'M', '1', 'MODERATE']:
                        ground_truth = RiskLevel.MEDIUM
                    elif risk_value in ['HIGH', 'H', '2', 'UNSTABLE']:
                        ground_truth = RiskLevel.HIGH
                elif isinstance(risk_value, (int, float)):
                    if risk_value == 0:
                        ground_truth = RiskLevel.LOW
                    elif risk_value == 1:
                        ground_truth = RiskLevel.MEDIUM
                    elif risk_value == 2:
                        ground_truth = RiskLevel.HIGH
            
            # Create RockfallDataPoint
            data_point = RockfallDataPoint(
                timestamp=timestamp,
                location=location,
                environmental=environmental,
                sensor_readings=sensor_data,
                ground_truth=ground_truth
            )
            
            return data_point
            
        except Exception as e:
            logger.error(f"Error converting row to data point: {e}")
            return None
    
    def _safe_float(self, value: Any) -> Optional[float]:
        """Safely convert value to float."""
        if value is None or pd.isna(value):
            return None
        try:
            return float(value)
        except (ValueError, TypeError):
            return None


class DataLoaderRegistry:
    """Registry for managing different dataset loaders."""
    
    def __init__(self):
        """Initialize dataset loader registry."""
        self.loaders = {
            'open_pit_mine': OpenPitMineDatasetLoader,
            'rocknet_seismic': RockNetSeismicDatasetLoader,
            'brazilian_rockfall': BrazilianRockfallDatasetLoader
        }
        
    def register_loader(self, dataset_type: str, loader_class: type):
        """Register a new dataset loader."""
        self.loaders[dataset_type] = loader_class
        
    def get_loader(self, dataset_type: str, dataset_path: Union[str, Path]) -> BaseDatasetLoader:
        """Get a dataset loader for the specified type."""
        if dataset_type not in self.loaders:
            raise DatasetIntegrationError(f"Unsupported dataset type: {dataset_type}")
        
        loader_class = self.loaders[dataset_type]
        return loader_class(dataset_path)
    
    def auto_detect_dataset_type(self, dataset_path: Union[str, Path]) -> Optional[str]:
        """Auto-detect dataset type based on directory structure."""
        dataset_path = Path(dataset_path)
        
        if not dataset_path.exists():
            return None
        
        # Check for Open Pit Mine dataset structure
        if (dataset_path / "images").exists() and (dataset_path / "annotations").exists():
            return 'open_pit_mine'
        
        # Check for seismic dataset
        sac_files = list(dataset_path.rglob("*.sac")) + list(dataset_path.rglob("*.SAC"))
        if sac_files:
            return 'rocknet_seismic'
        
        # Check for Brazilian dataset (CSV files)
        csv_files = list(dataset_path.glob("*.csv")) + list(dataset_path.glob("*.xlsx"))
        if csv_files:
            return 'brazilian_rockfall'
        
        return None
    
    def load_dataset(self, dataset_path: Union[str, Path], dataset_type: Optional[str] = None) -> Tuple[List[RockfallDataPoint], Dict[str, Any]]:
        """Load dataset with automatic type detection if not specified."""
        if dataset_type is None:
            dataset_type = self.auto_detect_dataset_type(dataset_path)
            if dataset_type is None:
                raise DatasetIntegrationError(f"Could not auto-detect dataset type for: {dataset_path}")
        
        loader = self.get_loader(dataset_type, dataset_path)
        data_points = loader.load_data()
        dataset_info = loader.get_dataset_info()
        
        return data_points, dataset_info
    
    def get_supported_types(self) -> List[str]:
        """Get list of supported dataset types."""
        return list(self.loaders.keys())