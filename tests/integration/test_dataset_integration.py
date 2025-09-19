"""
Integration tests for Dataset Integration system.

This module tests the complete dataset loading pipeline with realistic
scenarios and data patterns from actual rockfall prediction datasets.
"""

import pytest
import tempfile
import json
import csv
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime, timedelta
import shutil
from PIL import Image
import struct

from src.data.dataset_integration import (
    DataLoaderRegistry, OpenPitMineDatasetLoader,
    RockNetSeismicDatasetLoader, BrazilianRockfallDatasetLoader
)
from src.data.schema import RiskLevel


class TestCompleteDatasetIntegration:
    """End-to-end integration tests for dataset loading."""
    
    @pytest.fixture
    def realistic_open_pit_dataset(self):
        """Create realistic Open Pit Mine dataset with multiple samples."""
        temp_dir = tempfile.mkdtemp()
        dataset_path = Path(temp_dir)
        
        # Create directory structure
        images_dir = dataset_path / "images"
        annotations_dir = dataset_path / "annotations"
        images_dir.mkdir()
        annotations_dir.mkdir()
        
        # Create multiple sample images and annotations
        samples = [
            {
                "filename": "mine_section_001.jpg",
                "location": {"latitude": 45.123, "longitude": -122.456, "elevation": 1200.0},
                "objects": [
                    {"class": "rockfall_zone", "x": 100, "y": 200, "width": 300, "height": 150, "confidence": 0.85},
                    {"class": "unstable_slope", "x": 450, "y": 100, "width": 200, "height": 250, "confidence": 0.72}
                ],
                "risk_level": "HIGH",
                "sensor_type": "satellite"
            },
            {
                "filename": "mine_section_002.jpg",
                "location": {"latitude": 45.234, "longitude": -122.567, "elevation": 1150.0},
                "objects": [
                    {"class": "stable_slope", "x": 200, "y": 300, "width": 400, "height": 200, "confidence": 0.91}
                ],
                "risk_level": "LOW",
                "sensor_type": "drone"
            },
            {
                "filename": "mine_section_003.jpg",
                "location": {"latitude": 45.345, "longitude": -122.678, "elevation": 1300.0},
                "objects": [
                    {"class": "monitoring_zone", "x": 50, "y": 50, "width": 500, "height": 400, "confidence": 0.78}
                ],
                "risk_level": "MEDIUM",
                "sensor_type": "satellite"
            }
        ]
        
        for sample in samples:
            # Create dummy image file
            image_path = images_dir / sample["filename"]
            # Create minimal valid JPEG file
            img = Image.new('RGB', (800, 600), color='red')
            img.save(image_path, 'JPEG')
            
            # Create annotation file
            annotation_data = {
                "filename": sample["filename"],
                "size": [800, 600],
                "objects": sample["objects"],
                "location": sample["location"],
                "sensor_type": sample["sensor_type"],
                "risk_level": sample["risk_level"],
                "timestamp": "2023-01-01T12:00:00Z",
                "weather_conditions": "clear",
                "image_quality": "high"
            }
            
            annotation_path = annotations_dir / (Path(sample["filename"]).stem + ".json")
            with open(annotation_path, 'w') as f:
                json.dump(annotation_data, f, indent=2)
        
        yield dataset_path
        
        # Cleanup
        shutil.rmtree(temp_dir)
    
    @pytest.fixture
    def realistic_seismic_dataset(self):
        """Create realistic seismic dataset with SAC files and catalog."""
        temp_dir = tempfile.mkdtemp()
        dataset_path = Path(temp_dir)
        
        # Create SAC file structure
        stations = ['SITE01', 'SITE02', 'SITE03']
        dates = ['20230101', '20230102', '20230103']
        times = ['120000', '130000', '140000']
        
        sac_files = []
        catalog_entries = []
        
        for i, (station, date, time) in enumerate(zip(stations, dates, times)):
            # Create SAC filename
            sac_filename = f"{station}_{date}_{time}.sac"
            sac_path = dataset_path / sac_filename
            
            # Create dummy SAC file (minimal binary structure)
            # SAC header is 158 32-bit words (632 bytes)
            header = bytearray(632)
            # Set required header fields (simplified)
            struct.pack_into('f', header, 0, 100.0)  # sampling rate
            struct.pack_into('f', header, 4, 1000)   # number of points
            
            # Create some dummy seismic data
            data = np.random.randn(1000).astype(np.float32)
            
            with open(sac_path, 'wb') as f:
                f.write(header)
                f.write(data.tobytes())
            
            sac_files.append(sac_filename)
            
            # Create corresponding catalog entry
            catalog_entries.append({
                'timestamp': f"2023-01-0{i+1} 12:00:00",
                'station': station,
                'magnitude': 1.5 + i * 0.3,
                'event_type': 'rockfall' if i % 2 == 0 else 'earthquake',
                'confidence': 0.8 + i * 0.05,
                'latitude': 45.0 + i * 0.1,
                'longitude': -122.0 - i * 0.1,
                'depth': 5.0 + i * 2.0,
                'duration': 15.5 + i * 5.0
            })
        
        # Create event catalog
        catalog_df = pd.DataFrame(catalog_entries)
        catalog_df.to_csv(dataset_path / "event_catalog.csv", index=False)
        
        # Create metadata file
        metadata = {
            "network": "ROCKNET",
            "deployment_date": "2023-01-01",
            "sampling_rate": 100.0,
            "instrument_type": "broadband_seismometer",
            "station_locations": {
                "SITE01": {"lat": 45.0, "lon": -122.0, "elev": 1200},
                "SITE02": {"lat": 45.1, "lon": -122.1, "elev": 1150},
                "SITE03": {"lat": 45.2, "lon": -122.2, "elev": 1300}
            }
        }
        
        with open(dataset_path / "metadata.json", 'w') as f:
            json.dump(metadata, f, indent=2)
        
        yield dataset_path
        
        # Cleanup
        shutil.rmtree(temp_dir)
    
    @pytest.fixture
    def realistic_brazilian_dataset(self):
        """Create realistic Brazilian rockfall dataset with time series data."""
        temp_dir = tempfile.mkdtemp()
        dataset_path = Path(temp_dir)
        
        # Generate time series data for multiple locations
        base_time = datetime(2023, 1, 1, 0, 0, 0)
        time_points = [base_time + timedelta(hours=i) for i in range(48)]  # 48 hours
        
        locations = [
            {"lat": -23.123, "lon": -46.456, "elev": 800.0, "site": "Site_A"},
            {"lat": -23.234, "lon": -46.567, "elev": 750.0, "site": "Site_B"},
            {"lat": -23.345, "lon": -46.678, "elev": 900.0, "site": "Site_C"}
        ]
        
        all_data = []
        
        for location in locations:
            for time_point in time_points:
                # Simulate realistic patterns
                hour = time_point.hour
                day_factor = np.sin(2 * np.pi * hour / 24)  # Daily cycle
                noise = np.random.normal(0, 0.1)
                
                # Weather patterns
                base_temp = 20 + 5 * day_factor + noise
                base_humidity = 60 + 20 * day_factor + noise * 5
                rainfall = max(0, 10 * np.random.exponential(0.1) if np.random.random() < 0.2 else 0)
                
                # Sensor readings with some correlation to weather
                vibrations = max(0, 0.05 + 0.02 * (rainfall / 10) + noise * 0.01)
                displacement = max(0, 0.01 + 0.005 * (rainfall / 10) + noise * 0.002)
                strain = 50 + 20 * (rainfall / 10) + noise * 10
                pore_pressure = 60 + 15 * (rainfall / 10) + noise * 5
                
                # Risk assessment based on multiple factors
                risk_score = (
                    0.3 * (rainfall / 20) +
                    0.2 * (vibrations / 0.1) +
                    0.2 * (displacement / 0.02) +
                    0.3 * (strain / 100)
                )
                
                if risk_score < 0.3:
                    risk_level = 'LOW'
                elif risk_score < 0.7:
                    risk_level = 'MEDIUM'
                else:
                    risk_level = 'HIGH'
                
                data_row = {
                    'timestamp': time_point.strftime('%Y-%m-%d %H:%M:%S'),
                    'latitude': location['lat'],
                    'longitude': location['lon'],
                    'elevation': location['elev'],
                    'site_id': location['site'],
                    'temperature': round(base_temp, 1),
                    'humidity': round(max(0, min(100, base_humidity)), 1),
                    'rainfall': round(rainfall, 2),
                    'wind_speed': round(max(0, 5 + 3 * day_factor + noise), 1),
                    'vibrations': round(vibrations, 4),
                    'displacement': round(displacement, 4),
                    'strain': round(strain, 1),
                    'pore_pressure': round(pore_pressure, 1),
                    'risk_level': risk_level,
                    'soil_moisture': round(30 + 20 * (rainfall / 20) + noise * 5, 1),
                    'slope_angle': location['elev'] / 10,  # Simplified slope calculation
                    'rock_type': 'granite' if location['site'] == 'Site_A' else 'sandstone'
                }
                
                all_data.append(data_row)
        
        # Create main dataset file
        df = pd.DataFrame(all_data)
        df.to_csv(dataset_path / "rockfall_monitoring_data.csv", index=False)
        
        # Create summary statistics file
        summary_data = []
        for location in locations:
            site_data = df[df['site_id'] == location['site']]
            summary = {
                'site_id': location['site'],
                'latitude': location['lat'],
                'longitude': location['lon'],
                'elevation': location['elev'],
                'total_records': len(site_data),
                'avg_temperature': site_data['temperature'].mean(),
                'total_rainfall': site_data['rainfall'].sum(),
                'max_vibrations': site_data['vibrations'].max(),
                'max_displacement': site_data['displacement'].max(),
                'high_risk_events': len(site_data[site_data['risk_level'] == 'HIGH']),
                'medium_risk_events': len(site_data[site_data['risk_level'] == 'MEDIUM']),
                'low_risk_events': len(site_data[site_data['risk_level'] == 'LOW'])
            }
            summary_data.append(summary)
        
        summary_df = pd.DataFrame(summary_data)
        summary_df.to_csv(dataset_path / "site_summary.csv", index=False)
        
        # Create metadata
        metadata = {
            "dataset_name": "Brazilian Rockfall Slope Monitoring",
            "collection_period": "2023-01-01 to 2023-01-02",
            "total_sites": len(locations),
            "total_records": len(df),
            "sensors": ["vibration", "displacement", "strain", "pore_pressure"],
            "weather_params": ["temperature", "humidity", "rainfall", "wind_speed"],
            "sampling_frequency": "hourly",
            "coordinate_system": "WGS84"
        }
        
        with open(dataset_path / "metadata.json", 'w') as f:
            json.dump(metadata, f, indent=2)
        
        yield dataset_path
        
        # Cleanup
        shutil.rmtree(temp_dir)
    
    def test_open_pit_mine_complete_loading(self, realistic_open_pit_dataset):
        """Test complete loading of Open Pit Mine dataset."""
        loader = OpenPitMineDatasetLoader(realistic_open_pit_dataset)
        
        # Validate dataset structure
        assert loader.validate_dataset()
        
        # Load data
        data_points = loader.load_data()
        assert len(data_points) == 3
        
        # Verify data quality
        risk_levels = [dp.ground_truth for dp in data_points]
        assert RiskLevel.HIGH in risk_levels
        assert RiskLevel.MEDIUM in risk_levels
        assert RiskLevel.LOW in risk_levels
        
        # Check image annotations
        for data_point in data_points:
            assert data_point.imagery is not None
            assert len(data_point.imagery.annotations) > 0
            
            for annotation in data_point.imagery.annotations:
                assert annotation.confidence > 0.0
                assert annotation.confidence <= 1.0
                assert annotation.class_name in ['rockfall_zone', 'unstable_slope', 'stable_slope', 'monitoring_zone']
        
        # Verify dataset info
        info = loader.get_dataset_info()
        assert info['dataset_type'] == 'open_pit_mine_object_detection'
        assert info['total_samples'] == 3
        assert 'satellite' in str(info['sensor_types'])
        assert 'drone' in str(info['sensor_types'])
    
    def test_seismic_dataset_complete_loading(self, realistic_seismic_dataset):
        """Test complete loading of seismic dataset."""
        loader = RockNetSeismicDatasetLoader(realistic_seismic_dataset)
        
        # Validate dataset structure
        assert loader.validate_dataset()
        
        # Load event catalog
        catalog = loader._load_event_catalog()
        assert catalog is not None
        assert len(catalog) == 3
        
        # Check catalog content
        for entry in catalog.values():
            assert 'magnitude' in entry
            assert 'event_type' in entry
            assert 'confidence' in entry
            assert entry['confidence'] >= 0.8
        
        # Verify dataset info
        info = loader.get_dataset_info()
        assert info['dataset_type'] == 'rocknet_seismic'
        assert info['total_samples'] == 3
        assert info['sac_files'] == 3
        assert info['catalog_entries'] == 3
    
    def test_brazilian_dataset_complete_loading(self, realistic_brazilian_dataset):
        """Test complete loading of Brazilian dataset."""
        loader = BrazilianRockfallDatasetLoader(realistic_brazilian_dataset)
        
        # Validate dataset structure
        assert loader.validate_dataset()
        
        # Load data
        data_points = loader.load_data()
        assert len(data_points) == 144  # 3 sites × 48 hours
        
        # Verify temporal patterns
        timestamps = [dp.timestamp for dp in data_points if dp.timestamp]
        assert len(set(timestamps)) > 1  # Multiple unique timestamps
        
        # Check environmental data patterns
        high_risk_points = [dp for dp in data_points if dp.ground_truth == RiskLevel.HIGH]
        medium_risk_points = [dp for dp in data_points if dp.ground_truth == RiskLevel.MEDIUM]
        low_risk_points = [dp for dp in data_points if dp.ground_truth == RiskLevel.LOW]
        
        assert len(high_risk_points) > 0
        assert len(medium_risk_points) > 0
        assert len(low_risk_points) > 0
        
        # Verify sensor data completeness
        for data_point in data_points:
            assert data_point.environmental is not None
            assert data_point.sensor_readings is not None
            assert data_point.location is not None
            
            # Check coordinate ranges (Brazilian coordinates)
            assert -25.0 < data_point.location.latitude < -20.0
            assert -50.0 < data_point.location.longitude < -45.0
        
        # Verify dataset info
        info = loader.get_dataset_info()
        assert info['dataset_type'] == 'brazilian_rockfall_slope'
        assert info['total_samples'] == 144
        assert info['data_files'] == 2  # Main data + summary
    
    def test_cross_dataset_comparison(self, realistic_open_pit_dataset, 
                                     realistic_brazilian_dataset):
        """Test loading and comparing multiple dataset types."""
        registry = DataLoaderRegistry()
        
        # Load Open Pit Mine data
        mine_loader = registry.get_loader('open_pit_mine', realistic_open_pit_dataset)
        mine_data = mine_loader.load_data()
        
        # Load Brazilian data
        brazilian_loader = registry.get_loader('brazilian_rockfall', realistic_brazilian_dataset)
        brazilian_data = brazilian_loader.load_data()
        
        # Compare dataset characteristics
        assert len(mine_data) < len(brazilian_data)  # Brazilian has time series
        
        # Check data type consistency
        for data_point in mine_data:
            assert data_point.imagery is not None
            assert data_point.seismic is None
            
        for data_point in brazilian_data[:10]:  # Check first 10
            assert data_point.imagery is None
            assert data_point.environmental is not None
            assert data_point.sensor_readings is not None
    
    def test_dataset_auto_detection_realistic(self, realistic_open_pit_dataset,
                                            realistic_seismic_dataset,
                                            realistic_brazilian_dataset):
        """Test automatic dataset type detection with realistic datasets."""
        registry = DataLoaderRegistry()
        
        # Test auto-detection
        assert registry.auto_detect_dataset_type(realistic_open_pit_dataset) == 'open_pit_mine'
        assert registry.auto_detect_dataset_type(realistic_seismic_dataset) == 'rocknet_seismic'
        assert registry.auto_detect_dataset_type(realistic_brazilian_dataset) == 'brazilian_rockfall'
    
    def test_performance_with_large_datasets(self, realistic_brazilian_dataset):
        """Test performance characteristics with larger datasets."""
        import time
        
        loader = BrazilianRockfallDatasetLoader(realistic_brazilian_dataset)
        
        # Measure loading time
        start_time = time.time()
        data_points = loader.load_data()
        load_time = time.time() - start_time
        
        # Performance assertions
        assert load_time < 5.0  # Should load within 5 seconds
        assert len(data_points) > 0
        
        # Memory efficiency check
        import sys
        memory_per_point = sys.getsizeof(data_points) / len(data_points)
        assert memory_per_point < 10000  # Reasonable memory usage per data point
    
    def test_error_recovery_and_partial_loading(self, realistic_brazilian_dataset):
        """Test error recovery and partial dataset loading."""
        # Create a dataset with some corrupted files
        corrupted_path = realistic_brazilian_dataset / "corrupted_data.csv"
        with open(corrupted_path, 'w') as f:
            f.write("invalid,csv,format\nwith,incomplete")
        
        loader = BrazilianRockfallDatasetLoader(realistic_brazilian_dataset)
        
        # Should still load valid data despite corrupted file
        data_points = loader.load_data()
        assert len(data_points) > 0  # Should have loaded the valid data
        
        # Dataset info should reflect the situation
        info = loader.get_dataset_info()
        assert info['data_files'] >= 2  # Should count valid files


if __name__ == "__main__":
    pytest.main([__file__])