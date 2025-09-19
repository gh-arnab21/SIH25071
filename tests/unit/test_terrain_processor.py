"""Unit tests for TerrainProcessor class."""

import unittest
import numpy as np
import tempfile
import os
from unittest.mock import patch, MagicMock

from src.data.processors.terrain_processor import TerrainProcessor
from src.data.schemas import DEMData, GeoCoordinate


class TestTerrainProcessor(unittest.TestCase):
    """Test cases for TerrainProcessor class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.processor = TerrainProcessor()
        
        # Create sample DEM data
        self.sample_elevation = np.array([
            [100, 105, 110, 115, 120],
            [102, 107, 112, 117, 122],
            [104, 109, 114, 119, 124],
            [106, 111, 116, 121, 126],
            [108, 113, 118, 123, 128]
        ], dtype=np.float64)
        
        self.sample_dem = DEMData(
            elevation_matrix=self.sample_elevation,
            resolution=10.0,
            bounds={
                'west': -120.0,
                'east': -119.9,
                'south': 35.0,
                'north': 35.1
            }
        )
        
        # Create more complex terrain for advanced testing
        self.complex_elevation = self._create_complex_terrain()
        self.complex_dem = DEMData(
            elevation_matrix=self.complex_elevation,
            resolution=5.0,
            bounds={
                'west': -121.0,
                'east': -120.8,
                'south': 36.0,
                'north': 36.2
            }
        )
    
    def _create_complex_terrain(self):
        """Create a more complex terrain for testing."""
        x = np.linspace(0, 4*np.pi, 20)
        y = np.linspace(0, 4*np.pi, 20)
        X, Y = np.meshgrid(x, y)
        
        # Create terrain with hills, valleys, and ridges
        elevation = (
            100 + 
            20 * np.sin(X) * np.cos(Y) +  # Rolling hills
            10 * np.sin(2*X) +             # Ridge pattern
            5 * np.random.normal(0, 1, X.shape)  # Noise
        )
        
        return elevation.astype(np.float64)
    
    def test_initialization_default_config(self):
        """Test TerrainProcessor initialization with default configuration."""
        processor = TerrainProcessor()
        
        self.assertIsInstance(processor.config, dict)
        self.assertEqual(processor.config['window_size'], 3)
        self.assertEqual(processor.config['slope_method'], 'horn')
        self.assertEqual(processor.config['curvature_method'], 'zevenbergen')
        self.assertEqual(processor.config['roughness_method'], 'std')
        self.assertEqual(processor.config['stability_threshold'], 30.0)
        self.assertTrue(processor.config['normalize_features'])
        self.assertFalse(processor.is_fitted)
    
    def test_initialization_custom_config(self):
        """Test TerrainProcessor initialization with custom configuration."""
        custom_config = {
            'window_size': 5,
            'slope_method': 'gradient',
            'normalize_features': False
        }
        
        processor = TerrainProcessor(custom_config)
        
        self.assertEqual(processor.config['window_size'], 5)
        self.assertEqual(processor.config['slope_method'], 'gradient')
        self.assertFalse(processor.config['normalize_features'])
        # Check that defaults are still applied for unspecified parameters
        self.assertEqual(processor.config['curvature_method'], 'zevenbergen')
    
    def test_calculate_slope_horn_method(self):
        """Test slope calculation using Horn's method."""
        processor = TerrainProcessor({'slope_method': 'horn'})
        slope = processor.calculate_slope(self.sample_elevation, 10.0)
        
        self.assertEqual(slope.shape, self.sample_elevation.shape)
        self.assertTrue(np.all(slope >= 0))  # Slopes should be non-negative
        self.assertTrue(np.all(slope <= 90))  # Slopes should be <= 90 degrees
        
        # For our sample data with consistent gradient, slopes should be similar
        # The gradient is 5m elevation change over 10m horizontal = ~26.57 degrees
        expected_slope = np.degrees(np.arctan(5/10))
        self.assertAlmostEqual(np.mean(slope[1:-1, 1:-1]), expected_slope, delta=5)
    
    def test_calculate_slope_gradient_method(self):
        """Test slope calculation using gradient method."""
        processor = TerrainProcessor({'slope_method': 'gradient'})
        slope = processor.calculate_slope(self.sample_elevation, 10.0)
        
        self.assertEqual(slope.shape, self.sample_elevation.shape)
        self.assertTrue(np.all(slope >= 0))
        self.assertTrue(np.all(slope <= 90))
    
    def test_calculate_aspect(self):
        """Test aspect calculation."""
        aspect = self.processor.calculate_aspect(self.sample_elevation, 10.0)
        
        self.assertEqual(aspect.shape, self.sample_elevation.shape)
        self.assertTrue(np.all(aspect >= 0))
        self.assertTrue(np.all(aspect < 360))
        
        # For our sample data with consistent gradient, aspect should be consistent
        # Just check that we get reasonable values
        mean_aspect = np.mean(aspect[1:-1, 1:-1])
        self.assertTrue(0 <= mean_aspect < 360)  # Valid aspect range
    
    def test_calculate_curvature_zevenbergen(self):
        """Test curvature calculation using Zevenbergen method."""
        processor = TerrainProcessor({'curvature_method': 'zevenbergen'})
        curvature = processor.calculate_curvature(self.complex_elevation, 5.0)
        
        self.assertIn('profile', curvature)
        self.assertIn('planform', curvature)
        self.assertIn('mean', curvature)
        
        for curv_type in curvature.values():
            self.assertEqual(curv_type.shape, self.complex_elevation.shape)
            self.assertTrue(np.all(np.isfinite(curv_type)))
    
    def test_calculate_curvature_wood(self):
        """Test curvature calculation using Wood method."""
        processor = TerrainProcessor({'curvature_method': 'wood'})
        curvature = processor.calculate_curvature(self.complex_elevation, 5.0)
        
        self.assertIn('profile', curvature)
        self.assertIn('planform', curvature)
        self.assertIn('mean', curvature)
        
        for curv_type in curvature.values():
            self.assertEqual(curv_type.shape, self.complex_elevation.shape)
    
    def test_calculate_roughness_std(self):
        """Test roughness calculation using standard deviation method."""
        processor = TerrainProcessor({'roughness_method': 'std'})
        roughness = processor.calculate_roughness(self.complex_elevation, 5.0)
        
        self.assertEqual(roughness.shape, self.complex_elevation.shape)
        self.assertTrue(np.all(roughness >= 0))
        
        # Complex terrain should have higher roughness than simple terrain
        simple_roughness = processor.calculate_roughness(self.sample_elevation, 10.0)
        self.assertGreater(np.mean(roughness), np.mean(simple_roughness))
    
    def test_calculate_roughness_tri(self):
        """Test roughness calculation using TRI method."""
        processor = TerrainProcessor({'roughness_method': 'tri'})
        roughness = processor.calculate_roughness(self.complex_elevation, 5.0)
        
        self.assertEqual(roughness.shape, self.complex_elevation.shape)
        self.assertTrue(np.all(roughness >= 0))
    
    def test_calculate_stability_indicators(self):
        """Test stability indicators calculation."""
        slope = self.processor.calculate_slope(self.complex_elevation, 5.0)
        aspect = self.processor.calculate_aspect(self.complex_elevation, 5.0)
        curvature = self.processor.calculate_curvature(self.complex_elevation, 5.0)
        
        indicators = self.processor.calculate_stability_indicators(slope, aspect, curvature)
        
        self.assertEqual(len(indicators), 8)  # Should return 8 indicators
        self.assertTrue(np.all(np.isfinite(indicators)))
        
        # Check that ratios are between 0 and 1
        ratio_indices = [0, 4, 6, 7]  # Indices of ratio-based indicators
        for idx in ratio_indices:
            self.assertTrue(0 <= indicators[idx] <= 1)
    
    def test_extract_terrain_features(self):
        """Test comprehensive terrain feature extraction."""
        features = self.processor._extract_terrain_features(self.complex_dem)
        
        self.assertIsInstance(features, np.ndarray)
        self.assertTrue(len(features) > 0)
        self.assertTrue(np.all(np.isfinite(features)))
        
        # Features should include statistics from all terrain attributes
        # Expected: 7 elevation + 6 slope + 6 aspect + 9 curvature + 5 roughness + 8 stability = 41 features
        expected_feature_count = 7 + 6 + 6 + 9 + 5 + 8
        self.assertEqual(len(features), expected_feature_count)
    
    def test_fit_and_transform_single(self):
        """Test fitting and transforming a single DEM."""
        processor = TerrainProcessor({'normalize_features': True})
        
        # Fit on training data
        processor.fit([self.sample_dem, self.complex_dem])
        self.assertTrue(processor.is_fitted)
        
        # Transform single DEM
        features = processor.transform(self.sample_dem)
        
        self.assertIsInstance(features, np.ndarray)
        self.assertTrue(len(features) > 0)
        self.assertTrue(np.all(np.isfinite(features)))
    
    def test_fit_and_transform_multiple(self):
        """Test fitting and transforming multiple DEMs."""
        processor = TerrainProcessor({'normalize_features': True})
        
        dem_list = [self.sample_dem, self.complex_dem]
        
        # Fit and transform
        features = processor.fit_transform(dem_list)
        
        self.assertEqual(features.shape[0], len(dem_list))
        self.assertTrue(features.shape[1] > 0)
        self.assertTrue(np.all(np.isfinite(features)))
    
    def test_transform_without_normalization(self):
        """Test transformation without feature normalization."""
        processor = TerrainProcessor({'normalize_features': False})
        processor.fit([self.sample_dem])
        
        features = processor.transform(self.sample_dem)
        
        self.assertIsInstance(features, np.ndarray)
        self.assertTrue(len(features) > 0)
    
    @patch('os.path.exists')
    @patch('rasterio.open')
    def test_read_geotiff_success(self, mock_rasterio_open, mock_exists):
        """Test successful GeoTIFF file reading."""
        # Mock file existence
        mock_exists.return_value = True
        
        # Mock rasterio dataset
        mock_dataset = MagicMock()
        mock_dataset.read.return_value = self.sample_elevation
        mock_dataset.nodata = None
        mock_dataset.transform = [10.0, 0, -120.0, 0, -10.0, 35.1]
        mock_dataset.bounds = MagicMock()
        mock_dataset.bounds.left = -120.0
        mock_dataset.bounds.right = -119.9
        mock_dataset.bounds.bottom = 35.0
        mock_dataset.bounds.top = 35.1
        
        mock_rasterio_open.return_value.__enter__.return_value = mock_dataset
        
        # Test reading
        dem_data = self.processor.read_geotiff('test.tif')
        
        self.assertIsInstance(dem_data, DEMData)
        np.testing.assert_array_equal(dem_data.elevation_matrix, self.sample_elevation)
        self.assertEqual(dem_data.resolution, 10.0)
        self.assertEqual(dem_data.bounds['west'], -120.0)
    
    @patch('os.path.exists')
    @patch('rasterio.open')
    def test_read_geotiff_with_nodata(self, mock_rasterio_open, mock_exists):
        """Test GeoTIFF reading with nodata values."""
        # Mock file existence
        mock_exists.return_value = True
        
        # Create elevation with nodata values
        elevation_with_nodata = self.sample_elevation.copy()
        elevation_with_nodata[0, 0] = -9999
        
        mock_dataset = MagicMock()
        mock_dataset.read.return_value = elevation_with_nodata
        mock_dataset.nodata = -9999
        mock_dataset.transform = [10.0, 0, -120.0, 0, -10.0, 35.1]
        mock_dataset.bounds = MagicMock()
        mock_dataset.bounds.left = -120.0
        mock_dataset.bounds.right = -119.9
        mock_dataset.bounds.bottom = 35.0
        mock_dataset.bounds.top = 35.1
        
        mock_rasterio_open.return_value.__enter__.return_value = mock_dataset
        
        dem_data = self.processor.read_geotiff('test.tif')
        
        # Check that nodata value was converted to NaN
        self.assertTrue(np.isnan(dem_data.elevation_matrix[0, 0]))
    
    def test_read_geotiff_file_not_found(self):
        """Test GeoTIFF reading with non-existent file."""
        with self.assertRaises(FileNotFoundError):
            self.processor.read_geotiff('nonexistent.tif')
    
    def test_preprocess_dem_no_nan(self):
        """Test DEM preprocessing without NaN values."""
        processed_dem = self.processor.preprocess_dem(self.sample_dem)
        
        self.assertIsInstance(processed_dem, DEMData)
        np.testing.assert_array_equal(processed_dem.elevation_matrix, self.sample_elevation)
        self.assertEqual(processed_dem.resolution, self.sample_dem.resolution)
    
    def test_preprocess_dem_with_nan(self):
        """Test DEM preprocessing with NaN values."""
        # Create DEM with NaN values
        elevation_with_nan = self.sample_elevation.copy()
        elevation_with_nan[2, 2] = np.nan
        
        dem_with_nan = DEMData(
            elevation_matrix=elevation_with_nan,
            resolution=10.0,
            bounds=self.sample_dem.bounds
        )
        
        processed_dem = self.processor.preprocess_dem(dem_with_nan)
        
        # Check that NaN values were filled
        self.assertFalse(np.any(np.isnan(processed_dem.elevation_matrix)))
    
    def test_fill_nan_values(self):
        """Test NaN value filling."""
        # Create elevation with NaN in the center
        elevation_with_nan = self.sample_elevation.copy()
        elevation_with_nan[2, 2] = np.nan
        
        filled_elevation = self.processor._fill_nan_values(elevation_with_nan)
        
        self.assertFalse(np.any(np.isnan(filled_elevation)))
        # The filled value should be close to neighboring values
        self.assertTrue(110 <= filled_elevation[2, 2] <= 120)
    
    def test_fill_nan_values_all_nan(self):
        """Test NaN filling when all values are NaN."""
        all_nan_elevation = np.full_like(self.sample_elevation, np.nan)
        
        with self.assertRaises(ValueError):
            self.processor._fill_nan_values(all_nan_elevation)
    
    def test_extract_terrain_profile(self):
        """Test terrain profile extraction."""
        start_coord = GeoCoordinate(latitude=36.0, longitude=-121.0)
        end_coord = GeoCoordinate(latitude=36.2, longitude=-120.8)
        
        profile = self.processor.extract_terrain_profile(
            self.complex_dem, start_coord, end_coord, num_points=50
        )
        
        self.assertIn('distances', profile)
        self.assertIn('elevations', profile)
        self.assertIn('slopes', profile)
        self.assertIn('coordinates', profile)
        
        self.assertEqual(len(profile['distances']), 50)
        self.assertEqual(len(profile['elevations']), 50)
        self.assertEqual(len(profile['slopes']), 50)
        self.assertEqual(len(profile['coordinates']), 50)
        
        # Distances should be monotonically increasing
        self.assertTrue(np.all(np.diff(profile['distances']) >= 0))
    
    def test_elevation_statistics(self):
        """Test elevation statistics calculation."""
        stats = self.processor._calculate_elevation_statistics(self.sample_elevation)
        
        self.assertEqual(len(stats), 7)  # mean, std, min, max, q25, q75, range
        
        # Check specific values for our known sample data
        self.assertAlmostEqual(stats[0], np.mean(self.sample_elevation))
        self.assertAlmostEqual(stats[2], np.min(self.sample_elevation))
        self.assertAlmostEqual(stats[3], np.max(self.sample_elevation))
    
    def test_slope_statistics(self):
        """Test slope statistics calculation."""
        slope = self.processor.calculate_slope(self.sample_elevation, 10.0)
        stats = self.processor._calculate_slope_statistics(slope)
        
        self.assertEqual(len(stats), 6)  # mean, std, min, max, p90, p95
        self.assertTrue(np.all(stats >= 0))  # All slope stats should be non-negative
    
    def test_aspect_statistics(self):
        """Test aspect statistics calculation."""
        aspect = self.processor.calculate_aspect(self.sample_elevation, 10.0)
        stats = self.processor._calculate_aspect_statistics(aspect)
        
        self.assertEqual(len(stats), 6)  # circular_mean, circular_var, N, E, S, W ratios
        
        # Check that ratios sum to approximately 1
        ratio_sum = stats[2] + stats[3] + stats[4] + stats[5]
        self.assertAlmostEqual(ratio_sum, 1.0, delta=0.01)
    
    def test_curvature_statistics(self):
        """Test curvature statistics calculation."""
        curvature = self.processor.calculate_curvature(self.complex_elevation, 5.0)
        stats = self.processor._calculate_curvature_statistics(curvature)
        
        self.assertEqual(len(stats), 9)  # 3 for each curvature type
        self.assertTrue(np.all(np.isfinite(stats)))
    
    def test_roughness_statistics(self):
        """Test roughness statistics calculation."""
        roughness = self.processor.calculate_roughness(self.complex_elevation, 5.0)
        stats = self.processor._calculate_roughness_statistics(roughness)
        
        self.assertEqual(len(stats), 5)  # mean, std, max, p90, p95
        self.assertTrue(np.all(stats >= 0))  # All roughness stats should be non-negative


if __name__ == '__main__':
    unittest.main()