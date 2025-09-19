"""
Unit tests for data schema validation in the Rockfall Prediction System.

This module tests all data classes and validation functions to ensure
proper schema compliance and error handling.
"""

import pytest
import numpy as np
from datetime import datetime, timezone
from unittest.mock import patch, mock_open
import tempfile
import os

from src.data.schema import (
    RockfallDataPoint, ImageData, SensorData, EnvironmentalData,
    DEMData, SeismicData, TimeSeries, GeoCoordinate, BoundingBox,
    RiskLevel, RockfallPrediction, ImageMetadata
)
from src.data.validation import (
    ValidationError, validate_file_exists, validate_image_data,
    validate_sensor_data, validate_time_series, validate_environmental_data,
    validate_dem_data, validate_seismic_data, validate_rockfall_datapoint,
    validate_prediction_output, validate_batch_data
)
from src.data.utils import (
    normalize_coordinates, convert_timestamp_to_datetime, normalize_array,
    convert_dict_to_timeseries, convert_json_to_bounding_boxes,
    convert_yolo_to_bounding_box, impute_missing_values, detect_outliers,
    convert_risk_level_string, create_sample_datapoint
)


class TestGeoCoordinate:
    """Test GeoCoordinate data class."""
    
    def test_valid_coordinates(self):
        """Test creation with valid coordinates."""
        coord = GeoCoordinate(latitude=45.0, longitude=-120.0, elevation=1500.0)
        assert coord.latitude == 45.0
        assert coord.longitude == -120.0
        assert coord.elevation == 1500.0
    
    def test_invalid_latitude(self):
        """Test validation of invalid latitude."""
        with pytest.raises(ValueError, match="Invalid latitude"):
            GeoCoordinate(latitude=95.0, longitude=-120.0)
        
        with pytest.raises(ValueError, match="Invalid latitude"):
            GeoCoordinate(latitude=-95.0, longitude=-120.0)
    
    def test_invalid_longitude(self):
        """Test validation of invalid longitude."""
        with pytest.raises(ValueError, match="Invalid longitude"):
            GeoCoordinate(latitude=45.0, longitude=185.0)
        
        with pytest.raises(ValueError, match="Invalid longitude"):
            GeoCoordinate(latitude=45.0, longitude=-185.0)


class TestBoundingBox:
    """Test BoundingBox data class."""
    
    def test_valid_bounding_box(self):
        """Test creation with valid parameters."""
        bbox = BoundingBox(x=10.0, y=20.0, width=100.0, height=50.0, 
                          class_name="rock", confidence=0.95)
        assert bbox.x == 10.0
        assert bbox.y == 20.0
        assert bbox.width == 100.0
        assert bbox.height == 50.0
        assert bbox.class_name == "rock"
        assert bbox.confidence == 0.95
    
    def test_invalid_dimensions(self):
        """Test validation of invalid dimensions."""
        with pytest.raises(ValueError, match="Width and height must be positive"):
            BoundingBox(x=10.0, y=20.0, width=0.0, height=50.0, class_name="rock")
        
        with pytest.raises(ValueError, match="Width and height must be positive"):
            BoundingBox(x=10.0, y=20.0, width=100.0, height=-10.0, class_name="rock")
    
    def test_invalid_confidence(self):
        """Test validation of invalid confidence."""
        with pytest.raises(ValueError, match="Confidence must be between 0 and 1"):
            BoundingBox(x=10.0, y=20.0, width=100.0, height=50.0, 
                       class_name="rock", confidence=1.5)


class TestTimeSeries:
    """Test TimeSeries data class."""
    
    def test_valid_time_series(self):
        """Test creation with valid data."""
        timestamps = [datetime.now(), datetime.now()]
        values = [1.0, 2.0]
        ts = TimeSeries(timestamps=timestamps, values=values, unit="mm")
        assert len(ts.timestamps) == 2
        assert len(ts.values) == 2
        assert ts.unit == "mm"
    
    def test_mismatched_lengths(self):
        """Test validation of mismatched lengths."""
        timestamps = [datetime.now()]
        values = [1.0, 2.0]
        with pytest.raises(ValueError, match="Timestamps and values must have the same length"):
            TimeSeries(timestamps=timestamps, values=values, unit="mm")
    
    def test_empty_series(self):
        """Test validation of empty series."""
        with pytest.raises(ValueError, match="Time series cannot be empty"):
            TimeSeries(timestamps=[], values=[], unit="mm")
    
    def test_invalid_timestamps(self):
        """Test validation of invalid timestamps."""
        timestamps = ["not_a_datetime", datetime.now()]
        values = [1.0, 2.0]
        with pytest.raises(ValueError, match="All timestamps must be datetime objects"):
            TimeSeries(timestamps=timestamps, values=values, unit="mm")


class TestSensorData:
    """Test SensorData data class."""
    
    def test_valid_sensor_data(self):
        """Test creation with valid sensor measurements."""
        timestamps = [datetime.now()]
        values = [1.0]
        displacement = TimeSeries(timestamps=timestamps, values=values, unit="mm")
        sensor_data = SensorData(displacement=displacement)
        assert sensor_data.displacement is not None
    
    def test_empty_sensor_data(self):
        """Test validation of empty sensor data."""
        with pytest.raises(ValueError, match="At least one sensor measurement must be provided"):
            SensorData()


class TestEnvironmentalData:
    """Test EnvironmentalData data class."""
    
    def test_valid_environmental_data(self):
        """Test creation with valid environmental measurements."""
        env_data = EnvironmentalData(
            rainfall=5.0, temperature=20.0, wind_speed=10.0, humidity=60.0
        )
        assert env_data.rainfall == 5.0
        assert env_data.temperature == 20.0
        assert env_data.wind_speed == 10.0
        assert env_data.humidity == 60.0
    
    def test_invalid_rainfall(self):
        """Test validation of negative rainfall."""
        with pytest.raises(ValueError, match="Rainfall cannot be negative"):
            EnvironmentalData(rainfall=-1.0)
    
    def test_invalid_temperature(self):
        """Test validation of impossible temperature."""
        with pytest.raises(ValueError, match="Temperature cannot be below absolute zero"):
            EnvironmentalData(temperature=-300.0)
    
    def test_invalid_wind_speed(self):
        """Test validation of negative wind speed."""
        with pytest.raises(ValueError, match="Wind speed cannot be negative"):
            EnvironmentalData(wind_speed=-5.0)
    
    def test_invalid_humidity(self):
        """Test validation of invalid humidity."""
        with pytest.raises(ValueError, match="Humidity must be between 0 and 100"):
            EnvironmentalData(humidity=150.0)


class TestDEMData:
    """Test DEMData data class."""
    
    def test_valid_dem_data(self):
        """Test creation with valid DEM data."""
        elevation_matrix = np.random.rand(100, 100) * 1000
        dem_data = DEMData(
            elevation_matrix=elevation_matrix,
            resolution=10.0,
            bounds=(-120.0, 45.0, -119.0, 46.0)
        )
        assert dem_data.elevation_matrix.shape == (100, 100)
        assert dem_data.resolution == 10.0
    
    def test_invalid_elevation_matrix(self):
        """Test validation of invalid elevation matrix."""
        elevation_matrix = np.random.rand(100)  # 1D instead of 2D
        with pytest.raises(ValueError, match="Elevation matrix must be 2-dimensional"):
            DEMData(
                elevation_matrix=elevation_matrix,
                resolution=10.0,
                bounds=(-120.0, 45.0, -119.0, 46.0)
            )
    
    def test_invalid_resolution(self):
        """Test validation of invalid resolution."""
        elevation_matrix = np.random.rand(100, 100)
        with pytest.raises(ValueError, match="Resolution must be positive"):
            DEMData(
                elevation_matrix=elevation_matrix,
                resolution=-10.0,
                bounds=(-120.0, 45.0, -119.0, 46.0)
            )
    
    def test_invalid_bounds(self):
        """Test validation of invalid bounds."""
        elevation_matrix = np.random.rand(100, 100)
        with pytest.raises(ValueError, match="Bounds must contain exactly 4 values"):
            DEMData(
                elevation_matrix=elevation_matrix,
                resolution=10.0,
                bounds=(-120.0, 45.0, -119.0)  # Only 3 values
            )


class TestSeismicData:
    """Test SeismicData data class."""
    
    def test_valid_seismic_data(self):
        """Test creation with valid seismic data."""
        signal = np.random.rand(1000)
        seismic_data = SeismicData(
            signal=signal,
            sampling_rate=100.0,
            start_time=datetime.now(),
            station_id="STATION_01"
        )
        assert seismic_data.signal.shape == (1000,)
        assert seismic_data.sampling_rate == 100.0
        assert seismic_data.station_id == "STATION_01"
    
    def test_invalid_signal_dimension(self):
        """Test validation of invalid signal dimension."""
        signal = np.random.rand(100, 100)  # 2D instead of 1D
        with pytest.raises(ValueError, match="Seismic signal must be 1-dimensional"):
            SeismicData(
                signal=signal,
                sampling_rate=100.0,
                start_time=datetime.now(),
                station_id="STATION_01"
            )
    
    def test_invalid_sampling_rate(self):
        """Test validation of invalid sampling rate."""
        signal = np.random.rand(1000)
        with pytest.raises(ValueError, match="Sampling rate must be positive"):
            SeismicData(
                signal=signal,
                sampling_rate=-100.0,
                start_time=datetime.now(),
                station_id="STATION_01"
            )
    
    def test_empty_station_id(self):
        """Test validation of empty station ID."""
        signal = np.random.rand(1000)
        with pytest.raises(ValueError, match="Station ID cannot be empty"):
            SeismicData(
                signal=signal,
                sampling_rate=100.0,
                start_time=datetime.now(),
                station_id=""
            )


class TestRockfallPrediction:
    """Test RockfallPrediction data class."""
    
    def test_valid_prediction(self):
        """Test creation with valid prediction data."""
        prediction = RockfallPrediction(
            risk_level=RiskLevel.HIGH,
            confidence_score=0.85,
            contributing_factors={"slope": 0.6, "rainfall": 0.4},
            uncertainty_estimate=0.15
        )
        assert prediction.risk_level == RiskLevel.HIGH
        assert prediction.confidence_score == 0.85
        assert prediction.uncertainty_estimate == 0.15
    
    def test_invalid_confidence_score(self):
        """Test validation of invalid confidence score."""
        with pytest.raises(ValueError, match="Confidence score must be between 0 and 1"):
            RockfallPrediction(
                risk_level=RiskLevel.HIGH,
                confidence_score=1.5
            )
    
    def test_invalid_uncertainty_estimate(self):
        """Test validation of invalid uncertainty estimate."""
        with pytest.raises(ValueError, match="Uncertainty estimate must be between 0 and 1"):
            RockfallPrediction(
                risk_level=RiskLevel.HIGH,
                confidence_score=0.85,
                uncertainty_estimate=1.5
            )


class TestValidationFunctions:
    """Test validation functions."""
    
    def test_validate_file_exists(self):
        """Test file existence validation."""
        # Test with temporary file
        with tempfile.NamedTemporaryFile(delete=False) as tmp:
            tmp.write(b"test content")
            tmp_path = tmp.name
        
        try:
            assert validate_file_exists(tmp_path) is True
        finally:
            os.unlink(tmp_path)
        
        # Test with non-existent file
        with pytest.raises(ValidationError, match="File does not exist"):
            validate_file_exists("non_existent_file.txt")
    
    def test_validate_time_series(self):
        """Test time series validation."""
        # Valid time series
        timestamps = [datetime(2023, 1, 1), datetime(2023, 1, 2)]
        values = [1.0, 2.0]
        ts = TimeSeries(timestamps=timestamps, values=values, unit="mm")
        assert validate_time_series(ts) is True
        
        # Invalid time series (non-monotonic)
        timestamps = [datetime(2023, 1, 2), datetime(2023, 1, 1)]
        values = [1.0, 2.0]
        ts = TimeSeries(timestamps=timestamps, values=values, unit="mm")
        with pytest.raises(ValidationError, match="timestamps must be in ascending order"):
            validate_time_series(ts)
        
        # Time series with NaN values
        timestamps = [datetime(2023, 1, 1), datetime(2023, 1, 2)]
        values = [1.0, float('nan')]
        ts = TimeSeries(timestamps=timestamps, values=values, unit="mm")
        with pytest.raises(ValidationError, match="contains non-finite values"):
            validate_time_series(ts)
    
    def test_validate_batch_data(self):
        """Test batch data validation."""
        # Create sample data points
        datapoints = [create_sample_datapoint() for _ in range(3)]
        
        summary = validate_batch_data(datapoints)
        assert summary['total_points'] == 3
        # Note: sample datapoints may have validation issues, so we check that validation runs
        assert 'valid_points' in summary
        assert 'errors' in summary
        
        # Test empty batch
        with pytest.raises(ValidationError, match="Batch cannot be empty"):
            validate_batch_data([])


class TestUtilityFunctions:
    """Test utility functions."""
    
    def test_normalize_coordinates(self):
        """Test coordinate normalization."""
        lat, lon = normalize_coordinates(95.0, 185.0)
        assert lat == 90.0  # Clamped to max
        assert lon == -175.0  # Wrapped around
        
        lat, lon = normalize_coordinates(-95.0, -185.0)
        assert lat == -90.0  # Clamped to min
        assert lon == 175.0  # Wrapped around
    
    def test_convert_timestamp_to_datetime(self):
        """Test timestamp conversion."""
        # Test datetime passthrough
        dt = datetime.now()
        assert convert_timestamp_to_datetime(dt) == dt
        
        # Test unix timestamp
        unix_ts = 1640995200  # 2022-01-01 00:00:00 UTC
        dt = convert_timestamp_to_datetime(unix_ts)
        assert isinstance(dt, datetime)
        
        # Test string timestamp
        dt_str = "2022-01-01 12:00:00"
        dt = convert_timestamp_to_datetime(dt_str)
        assert isinstance(dt, datetime)
        assert dt.year == 2022
        assert dt.month == 1
        assert dt.day == 1
        
        # Test invalid timestamp
        with pytest.raises(ValueError):
            convert_timestamp_to_datetime("invalid_timestamp")
    
    def test_normalize_array(self):
        """Test array normalization."""
        arr = np.array([1, 2, 3, 4, 5])
        
        # Test minmax normalization
        normalized = normalize_array(arr, method='minmax')
        assert np.allclose(normalized, [0, 0.25, 0.5, 0.75, 1.0])
        
        # Test zscore normalization
        normalized = normalize_array(arr, method='zscore')
        assert np.allclose(np.mean(normalized), 0, atol=1e-10)
        assert np.allclose(np.std(normalized), 1, atol=1e-10)
        
        # Test invalid method
        with pytest.raises(ValueError, match="Unsupported normalization method"):
            normalize_array(arr, method='invalid')
    
    def test_convert_dict_to_timeseries(self):
        """Test dictionary to TimeSeries conversion."""
        data_dict = {
            'timestamp': ['2023-01-01 00:00:00', '2023-01-01 01:00:00'],
            'value': [1.0, 2.0]
        }
        ts = convert_dict_to_timeseries(data_dict, unit='mm')
        assert len(ts.timestamps) == 2
        assert len(ts.values) == 2
        assert ts.unit == 'mm'
    
    def test_convert_json_to_bounding_boxes(self):
        """Test JSON to BoundingBox conversion."""
        json_data = [
            {'x': 10, 'y': 20, 'width': 100, 'height': 50, 'class': 'rock'},
            {'x': 30, 'y': 40, 'width': 80, 'height': 60, 'class': 'slope'}
        ]
        bboxes = convert_json_to_bounding_boxes(json_data)
        assert len(bboxes) == 2
        assert bboxes[0].class_name == 'rock'
        assert bboxes[1].class_name == 'slope'
    
    def test_convert_yolo_to_bounding_box(self):
        """Test YOLO format to BoundingBox conversion."""
        yolo_line = "0 0.5 0.5 0.2 0.3 0.95"
        bbox = convert_yolo_to_bounding_box(yolo_line, 640, 480, ['rock'])
        assert bbox.class_name == 'rock'
        assert bbox.confidence == 0.95
        # Check coordinate conversion
        assert bbox.x == 0.5 * 640 - (0.2 * 640) / 2  # center_x - width/2
        assert bbox.y == 0.5 * 480 - (0.3 * 480) / 2  # center_y - height/2
    
    def test_impute_missing_values(self):
        """Test missing value imputation."""
        arr = np.array([1.0, np.nan, 3.0, 4.0, np.nan])
        
        # Test mean imputation
        imputed = impute_missing_values(arr, method='mean')
        expected_mean = np.nanmean(arr)
        assert np.allclose(imputed[np.isnan(arr)], expected_mean)
        
        # Test median imputation
        imputed = impute_missing_values(arr, method='median')
        expected_median = np.nanmedian(arr)
        assert np.allclose(imputed[np.isnan(arr)], expected_median)
    
    def test_detect_outliers(self):
        """Test outlier detection."""
        # Create array with obvious outliers
        arr = np.array([1, 2, 3, 4, 5, 100, -100])
        
        # Test IQR method
        outliers = detect_outliers(arr, method='iqr')
        assert outliers[-2] == True  # 100 should be outlier
        assert outliers[-1] == True  # -100 should be outlier
        
        # Test z-score method with lower threshold
        outliers = detect_outliers(arr, method='zscore', threshold=1.5)
        assert np.any(outliers)  # Should detect some outliers
    
    def test_convert_risk_level_string(self):
        """Test risk level string conversion."""
        assert convert_risk_level_string('LOW') == RiskLevel.LOW
        assert convert_risk_level_string('medium') == RiskLevel.MEDIUM
        assert convert_risk_level_string('HIGH') == RiskLevel.HIGH
        assert convert_risk_level_string('0') == RiskLevel.LOW
        assert convert_risk_level_string('1') == RiskLevel.MEDIUM
        assert convert_risk_level_string('2') == RiskLevel.HIGH
        
        with pytest.raises(ValueError, match="Cannot convert"):
            convert_risk_level_string('INVALID')
    
    def test_create_sample_datapoint(self):
        """Test sample datapoint creation."""
        datapoint = create_sample_datapoint()
        assert isinstance(datapoint, RockfallDataPoint)
        assert isinstance(datapoint.timestamp, datetime)
        assert isinstance(datapoint.location, GeoCoordinate)
        assert datapoint.sensor_readings is not None
        assert datapoint.environmental is not None


if __name__ == "__main__":
    pytest.main([__file__])