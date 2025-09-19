"""
Unit tests for Dataset Integration and Loading System.

This module tests the specialized data loaders for different dataset formats
including Open Pit Mine Object Detection Dataset, RockNet Seismic Dataset,
and Brazilian Rockfall Slope Dataset.
"""

import pytest
import tempfile
import json
import csv
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
from unittest.mock import patch, MagicMock, mock_open
import warnings

# Import the classes to test
from src.data.dataset_integration import (
    DataLoaderRegistry, BaseDatasetLoader, OpenPitMineDatasetLoader,
    RockNetSeismicDatasetLoader, BrazilianRockfallDatasetLoader,
    DatasetIntegrationError
)
from src.data.schema import (
    RockfallDataPoint, RiskLevel, ImageData, SeismicData, 
    GeoCoordinate, BoundingBox, ImageMetadata, SensorData,
    EnvironmentalData, TimeSeries
)


class TestDataLoaderRegistry:
    """Test cases for DataLoaderRegistry class."""
    
    def test_initialization(self):
        """Test registry initialization."""
        registry = DataLoaderRegistry()
        
        assert 'open_pit_mine' in registry.loaders
        assert 'rocknet_seismic' in registry.loaders
        assert 'brazilian_rockfall' in registry.loaders
        
        expected_types = ['open_pit_mine', 'rocknet_seismic', 'brazilian_rockfall']
        assert set(registry.get_supported_types()) == set(expected_types)
    
    def test_register_loader(self):
        """Test registering a new loader."""
        registry = DataLoaderRegistry()
        
        class TestLoader(BaseDatasetLoader):
            def load_data(self):
                return []
        
        registry.register_loader('test_dataset', TestLoader)
        assert 'test_dataset' in registry.loaders
        assert registry.loaders['test_dataset'] == TestLoader
    
    def test_get_loader(self):
        """Test getting a loader instance."""
        registry = DataLoaderRegistry()
        
        with tempfile.TemporaryDirectory() as temp_dir:
            loader = registry.get_loader('open_pit_mine', temp_dir)
            assert isinstance(loader, OpenPitMineDatasetLoader)
            assert loader.dataset_path == Path(temp_dir)
    
    def test_get_loader_unsupported_type(self):
        """Test getting loader for unsupported type."""
        registry = DataLoaderRegistry()
        
        with pytest.raises(DatasetIntegrationError, match="Unsupported dataset type"):
            registry.get_loader('unsupported_type', '/tmp')
    
    def test_auto_detect_dataset_type(self):
        """Test automatic dataset type detection."""
        registry = DataLoaderRegistry()
        
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            
            # Test Open Pit Mine detection
            (temp_path / "images").mkdir()
            (temp_path / "annotations").mkdir()
            assert registry.auto_detect_dataset_type(temp_path) == 'open_pit_mine'
            
            # Clean up
            (temp_path / "images").rmdir()
            (temp_path / "annotations").rmdir()
            
            # Test seismic dataset detection
            (temp_path / "test.sac").touch()
            assert registry.auto_detect_dataset_type(temp_path) == 'rocknet_seismic'
            
            # Clean up
            (temp_path / "test.sac").unlink()
            
            # Test Brazilian dataset detection
            (temp_path / "data.csv").touch()
            assert registry.auto_detect_dataset_type(temp_path) == 'brazilian_rockfall'
    
    def test_auto_detect_nonexistent_path(self):
        """Test auto-detection with non-existent path."""
        registry = DataLoaderRegistry()
        result = registry.auto_detect_dataset_type("/nonexistent/path")
        assert result is None


class TestOpenPitMineDatasetLoader:
    """Test cases for OpenPitMineDatasetLoader class."""
    
    @pytest.fixture
    def temp_dataset(self):
        """Create temporary Open Pit Mine dataset structure."""
        temp_dir = tempfile.mkdtemp()
        dataset_path = Path(temp_dir)
        
        # Create directory structure
        images_dir = dataset_path / "images"
        annotations_dir = dataset_path / "annotations"
        images_dir.mkdir()
        annotations_dir.mkdir()
        
        # Create sample image file
        (images_dir / "sample_001.jpg").touch()
        
        # Create sample annotation file
        annotation_data = {
            "filename": "sample_001.jpg",
            "size": [1024, 768],
            "objects": [
                {
                    "class": "rockfall_zone",
                    "x": 100,
                    "y": 200,
                    "width": 300,
                    "height": 150,
                    "confidence": 0.85
                }
            ],
            "location": {
                "latitude": 45.123,
                "longitude": -122.456,
                "elevation": 1200.0
            },
            "sensor_type": "satellite",
            "risk_level": "high"
        }
        
        with open(annotations_dir / "sample_001.json", 'w') as f:
            json.dump(annotation_data, f)
        
        yield dataset_path
        
        # Cleanup
        import shutil
        shutil.rmtree(temp_dir)
    
    def test_initialization(self, temp_dataset):
        """Test loader initialization."""
        loader = OpenPitMineDatasetLoader(temp_dataset)
        
        assert loader.dataset_path == temp_dataset
        assert loader.images_dir == temp_dataset / "images"
        assert loader.annotations_dir == temp_dataset / "annotations"
        assert '.jpg' in loader.supported_extensions
        assert '.json' in loader.supported_extensions
    
    def test_validate_dataset_success(self, temp_dataset):
        """Test successful dataset validation."""
        loader = OpenPitMineDatasetLoader(temp_dataset)
        assert loader.validate_dataset() == True
    
    def test_validate_dataset_missing_images(self, temp_dataset):
        """Test validation with missing images directory."""
        (temp_dataset / "images").rmdir()
        loader = OpenPitMineDatasetLoader(temp_dataset)
        
        with pytest.raises(DatasetIntegrationError, match="Images directory not found"):
            loader.validate_dataset()
    
    def test_validate_dataset_missing_annotations(self, temp_dataset):
        """Test validation with missing annotations directory."""
        (temp_dataset / "annotations").rmdir()
        loader = OpenPitMineDatasetLoader(temp_dataset)
        
        with pytest.raises(DatasetIntegrationError, match="Annotations directory not found"):
            loader.validate_dataset()
    
    def test_load_data_success(self, temp_dataset):
        """Test successful data loading."""
        loader = OpenPitMineDatasetLoader(temp_dataset)
        data_points = loader.load_data()
        
        assert len(data_points) == 1
        
        data_point = data_points[0]
        assert isinstance(data_point, RockfallDataPoint)
        assert data_point.imagery is not None
        assert data_point.location.latitude == 45.123
        assert data_point.location.longitude == -122.456
        assert data_point.ground_truth == RiskLevel.HIGH
        
        # Check image data
        assert len(data_point.imagery.annotations) == 1
        bbox = data_point.imagery.annotations[0]
        assert bbox.class_name == "rockfall_zone"
        assert bbox.confidence == 0.85
        
        # Check dataset info
        info = loader.get_dataset_info()
        assert info['dataset_type'] == 'open_pit_mine_object_detection'
        assert info['total_samples'] == 1
    
    def test_parse_risk_level(self, temp_dataset):
        """Test risk level parsing."""
        loader = OpenPitMineDatasetLoader(temp_dataset)
        
        # Test string values
        assert loader._parse_risk_level('LOW') == RiskLevel.LOW
        assert loader._parse_risk_level('MEDIUM') == RiskLevel.MEDIUM
        assert loader._parse_risk_level('HIGH') == RiskLevel.HIGH
        assert loader._parse_risk_level('L') == RiskLevel.LOW
        assert loader._parse_risk_level('M') == RiskLevel.MEDIUM
        assert loader._parse_risk_level('H') == RiskLevel.HIGH
        
        # Test numeric values
        assert loader._parse_risk_level(0) == RiskLevel.LOW
        assert loader._parse_risk_level(1) == RiskLevel.MEDIUM
        assert loader._parse_risk_level(2) == RiskLevel.HIGH
        
        # Test None and invalid values
        assert loader._parse_risk_level(None) is None
        assert loader._parse_risk_level('invalid') is None


class TestRockNetSeismicDatasetLoader:
    """Test cases for RockNetSeismicDatasetLoader class."""
    
    @pytest.fixture
    def temp_seismic_dataset(self):
        """Create temporary seismic dataset structure."""
        temp_dir = tempfile.mkdtemp()
        dataset_path = Path(temp_dir)
        
        # Create sample SAC files
        (dataset_path / "station1_20230101_120000.sac").touch()
        (dataset_path / "station2_20230101_130000.SAC").touch()
        
        # Create event catalog
        catalog_data = [
            {
                'timestamp': '2023-01-01 12:00:00',
                'station': 'station1',
                'magnitude': 1.5,
                'event_type': 'rockfall',
                'confidence': 0.9
            },
            {
                'timestamp': '2023-01-01 13:00:00',
                'station': 'station2',
                'magnitude': 2.2,
                'event_type': 'rockfall',
                'confidence': 0.8
            }
        ]
        
        catalog_df = pd.DataFrame(catalog_data)
        catalog_df.to_csv(dataset_path / "event_catalog.csv", index=False)
        
        yield dataset_path
        
        # Cleanup
        import shutil
        shutil.rmtree(temp_dir)
    
    def test_initialization(self, temp_seismic_dataset):
        """Test seismic loader initialization."""
        loader = RockNetSeismicDatasetLoader(temp_seismic_dataset)
        
        assert loader.dataset_path == temp_seismic_dataset
        assert '.sac' in loader.supported_extensions
        assert '.SAC' in loader.supported_extensions
        assert loader.seismic_processor is not None
    
    def test_validate_dataset_success(self, temp_seismic_dataset):
        """Test successful seismic dataset validation."""
        loader = RockNetSeismicDatasetLoader(temp_seismic_dataset)
        assert loader.validate_dataset() == True
    
    def test_validate_dataset_no_sac_files(self):
        """Test validation with no SAC files."""
        with tempfile.TemporaryDirectory() as temp_dir:
            loader = RockNetSeismicDatasetLoader(temp_dir)
            
            with pytest.raises(DatasetIntegrationError, match="No SAC files found"):
                loader.validate_dataset()
    
    def test_load_event_catalog(self, temp_seismic_dataset):
        """Test event catalog loading."""
        loader = RockNetSeismicDatasetLoader(temp_seismic_dataset)
        catalog = loader._load_event_catalog()
        
        assert catalog is not None
        assert len(catalog) == 2
        assert 'station1_20230101_120000' in catalog
        assert 'station2_20230101_130000' in catalog
    
    def test_extract_location_from_path(self, temp_seismic_dataset):
        """Test location extraction from file path."""
        loader = RockNetSeismicDatasetLoader(temp_seismic_dataset)
        
        # Test with coordinates in filename
        file_path = Path("data_lat45.123_lon-122.456_station1.sac")
        location = loader._extract_location_from_path(file_path)
        
        assert location.latitude == 45.123
        assert location.longitude == -122.456
    
    @patch('src.data.dataset_integration.SeismicProcessor')
    def test_load_data_with_mock(self, mock_processor_class, temp_seismic_dataset):
        """Test data loading with mocked seismic processor."""
        # Setup mock
        mock_processor = mock_processor_class.return_value
        mock_seismic_data = SeismicData(
            signal=np.random.randn(1000),
            sampling_rate=100.0,
            start_time=datetime(2023, 1, 1, 12, 0, 0),
            station_id="TEST.STATION.00.HHZ"
        )
        mock_processor.read_sac_file.return_value = mock_seismic_data
        
        loader = RockNetSeismicDatasetLoader(temp_seismic_dataset)
        data_points = loader.load_data()
        
        assert len(data_points) == 2  # Two SAC files
        
        for data_point in data_points:
            assert isinstance(data_point, RockfallDataPoint)
            assert data_point.seismic is not None
            assert data_point.location is not None
    
    def test_generate_file_key(self, temp_seismic_dataset):
        """Test file key generation for catalog lookup."""
        loader = RockNetSeismicDatasetLoader(temp_seismic_dataset)
        
        file_path = Path("station1_20230101_120000.sac")
        seismic_data = SeismicData(
            signal=np.random.randn(100),
            sampling_rate=100.0,
            start_time=datetime(2023, 1, 1, 12, 0, 0),
            station_id="NET.STATION1.00.HHZ"
        )
        
        key = loader._generate_file_key(file_path, seismic_data)
        assert key == "STATION1_20230101_120000"


class TestBrazilianRockfallDatasetLoader:
    """Test cases for BrazilianRockfallDatasetLoader class."""
    
    @pytest.fixture
    def temp_brazilian_dataset(self):
        """Create temporary Brazilian dataset structure."""
        temp_dir = tempfile.mkdtemp()
        dataset_path = Path(temp_dir)
        
        # Create sample CSV data
        sample_data = [
            {
                'latitude': 45.123,
                'longitude': -122.456,
                'elevation': 1200.0,
                'timestamp': '2023-01-01 12:00:00',
                'rainfall': 25.5,
                'temperature': 18.2,
                'vibrations': 0.15,
                'wind_speed': 8.3,
                'humidity': 65.0,
                'displacement': 0.05,
                'strain': 120.0,
                'pore_pressure': 85.5,
                'risk_level': 'HIGH'
            },
            {
                'latitude': 45.234,
                'longitude': -122.567,
                'elevation': 1150.0,
                'timestamp': '2023-01-01 13:00:00',
                'rainfall': 5.2,
                'temperature': 22.1,
                'vibrations': 0.02,
                'wind_speed': 3.1,
                'humidity': 45.0,
                'displacement': 0.01,
                'strain': 45.0,
                'pore_pressure': 62.3,
                'risk_level': 'LOW'
            }
        ]
        
        df = pd.DataFrame(sample_data)
        df.to_csv(dataset_path / "rockfall_data.csv", index=False)
        
        yield dataset_path
        
        # Cleanup
        import shutil
        shutil.rmtree(temp_dir)
    
    def test_initialization(self, temp_brazilian_dataset):
        """Test Brazilian dataset loader initialization."""
        loader = BrazilianRockfallDatasetLoader(temp_brazilian_dataset)
        
        assert loader.dataset_path == temp_brazilian_dataset
        assert '.csv' in loader.supported_extensions
        assert '.xlsx' in loader.supported_extensions
    
    def test_validate_dataset_success(self, temp_brazilian_dataset):
        """Test successful dataset validation."""
        loader = BrazilianRockfallDatasetLoader(temp_brazilian_dataset)
        assert loader.validate_dataset() == True
    
    def test_validate_dataset_no_files(self):
        """Test validation with no data files."""
        with tempfile.TemporaryDirectory() as temp_dir:
            loader = BrazilianRockfallDatasetLoader(temp_dir)
            
            with pytest.raises(DatasetIntegrationError, match="No CSV/Excel files found"):
                loader.validate_dataset()
    
    def test_load_data_success(self, temp_brazilian_dataset):
        """Test successful Brazilian dataset loading."""
        loader = BrazilianRockfallDatasetLoader(temp_brazilian_dataset)
        data_points = loader.load_data()
        
        assert len(data_points) == 2
        
        # Check first data point
        data_point = data_points[0]
        assert isinstance(data_point, RockfallDataPoint)
        assert data_point.location.latitude == 45.123
        assert data_point.location.longitude == -122.456
        assert data_point.location.elevation == 1200.0
        assert data_point.ground_truth == RiskLevel.HIGH
        
        # Check environmental data
        assert data_point.environmental.rainfall == 25.5
        assert data_point.environmental.temperature == 18.2
        assert data_point.environmental.humidity == 65.0
        
        # Check sensor data
        assert data_point.sensor_readings is not None
        assert data_point.sensor_readings.displacement is not None
        assert data_point.sensor_readings.strain is not None
        assert data_point.sensor_readings.pore_pressure is not None
        
        # Check second data point
        data_point = data_points[1]
        assert data_point.ground_truth == RiskLevel.LOW
        
        # Check dataset info
        info = loader.get_dataset_info()
        assert info['dataset_type'] == 'brazilian_rockfall_slope'
        assert info['total_samples'] == 2
    
    def test_safe_float_conversion(self, temp_brazilian_dataset):
        """Test safe float conversion."""
        loader = BrazilianRockfallDatasetLoader(temp_brazilian_dataset)
        
        assert loader._safe_float(123.45) == 123.45
        assert loader._safe_float("123.45") == 123.45
        assert loader._safe_float(None) is None
        assert loader._safe_float("invalid") is None
        assert loader._safe_float(np.nan) is None


class TestIntegrationScenarios:
    """Integration tests for dataset loading scenarios."""
    
    def test_cross_dataset_compatibility(self):
        """Test loading multiple dataset types."""
        registry = DataLoaderRegistry()
        
        with tempfile.TemporaryDirectory() as temp_dir:
            base_path = Path(temp_dir)
            
            # Create Open Pit Mine dataset
            mine_path = base_path / "mine_dataset"
            mine_path.mkdir()
            (mine_path / "images").mkdir()
            (mine_path / "annotations").mkdir()
            
            # Create Brazilian dataset
            brazilian_path = base_path / "brazilian_dataset"
            brazilian_path.mkdir()
            (brazilian_path / "data.csv").touch()
            
            # Test auto-detection
            assert registry.auto_detect_dataset_type(mine_path) == 'open_pit_mine'
            assert registry.auto_detect_dataset_type(brazilian_path) == 'brazilian_rockfall'
    
    def test_missing_data_scenarios(self):
        """Test handling of missing data scenarios."""
        with tempfile.TemporaryDirectory() as temp_dir:
            dataset_path = Path(temp_dir)
            
            # Create CSV with missing values
            sample_data = pd.DataFrame([
                {
                    'latitude': 45.123,
                    'longitude': -122.456,
                    'rainfall': None,
                    'temperature': np.nan,
                    'risk_level': 'LOW'
                }
            ])
            sample_data.to_csv(dataset_path / "data_with_missing.csv", index=False)
            
            loader = BrazilianRockfallDatasetLoader(dataset_path)
            data_points = loader.load_data()
            
            assert len(data_points) == 1
            data_point = data_points[0]
            assert data_point.environmental.rainfall is None
            assert data_point.environmental.temperature is None
    
    @patch('src.data.dataset_integration.logger')
    def test_error_handling_and_logging(self, mock_logger):
        """Test error handling and logging functionality."""
        with tempfile.TemporaryDirectory() as temp_dir:
            dataset_path = Path(temp_dir)
            
            # Create invalid JSON file
            invalid_json = dataset_path / "annotations"
            invalid_json.mkdir()
            with open(invalid_json / "invalid.json", 'w') as f:
                f.write("{ invalid json")
            
            # Create images directory
            (dataset_path / "images").mkdir()
            
            loader = OpenPitMineDatasetLoader(dataset_path)
            data_points = loader.load_data()
            
            # Should handle invalid JSON gracefully
            assert len(data_points) == 0
            
            # Check that warning was logged
            mock_logger.warning.assert_called()
    
    def test_end_to_end_pipeline_compatibility(self):
        """Test that loaded data is compatible with processing pipeline."""
        with tempfile.TemporaryDirectory() as temp_dir:
            dataset_path = Path(temp_dir)
            
            # Create valid Brazilian dataset
            sample_data = pd.DataFrame([
                {
                    'latitude': 45.123,
                    'longitude': -122.456,
                    'elevation': 1200.0,
                    'timestamp': '2023-01-01 12:00:00',
                    'rainfall': 25.5,
                    'temperature': 18.2,
                    'risk_level': 'HIGH'
                }
            ])
            sample_data.to_csv(dataset_path / "test_data.csv", index=False)
            
            loader = BrazilianRockfallDatasetLoader(dataset_path)
            data_points = loader.load_data()
            
            # Validate compatibility with validation system
            from src.data.validation import validate_rockfall_datapoint
            
            for data_point in data_points:
                # Should not raise validation errors
                validate_rockfall_datapoint(data_point)


if __name__ == "__main__":
    pytest.main([__file__])