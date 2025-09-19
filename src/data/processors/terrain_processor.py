"""Terrain data processing pipeline for Digital Elevation Model (DEM) data."""

import os
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import rasterio
from rasterio.transform import from_bounds
from scipy import ndimage
from scipy.spatial.distance import pdist, squareform
from sklearn.preprocessing import StandardScaler

from ..base import BaseDataProcessor
from ..schemas import DEMData, GeoCoordinate


class TerrainProcessor(BaseDataProcessor):
    """Processor for Digital Elevation Model (DEM) data with terrain feature extraction."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize the terrain processor.
        
        Args:
            config: Configuration dictionary with processing parameters
                - window_size: Size of the sliding window for calculations (default: 3)
                - slope_method: Method for slope calculation ('gradient' or 'horn', default: 'horn')
                - curvature_method: Method for curvature calculation ('zevenbergen' or 'wood', default: 'zevenbergen')
                - roughness_method: Method for roughness calculation ('std' or 'tri', default: 'std')
                - stability_threshold: Threshold angle for stability analysis (default: 30 degrees)
                - normalize_features: Whether to normalize extracted features (default: True)
        """
        super().__init__(config)
        
        # Default configuration
        default_config = {
            'window_size': 3,
            'slope_method': 'horn',
            'curvature_method': 'zevenbergen',
            'roughness_method': 'std',
            'stability_threshold': 30.0,  # degrees
            'normalize_features': True,
            'feature_resolution': 10.0,  # meters per pixel for feature extraction
        }
        
        self.config = {**default_config, **self.config}
        self.scaler = StandardScaler() if self.config['normalize_features'] else None
        
    def fit(self, data: List[DEMData]) -> 'TerrainProcessor':
        """Fit the processor to training data.
        
        Args:
            data: List of DEMData objects
            
        Returns:
            Self for method chaining
        """
        if self.config['normalize_features']:
            # Extract features from all training data for normalization
            all_features = []
            for dem_data in data:
                features = self._extract_terrain_features(dem_data)
                all_features.append(features)
            
            if all_features:
                combined_features = np.vstack(all_features)
                self.scaler.fit(combined_features)
        
        self._is_fitted = True
        return self
    
    def transform(self, data: Union[DEMData, List[DEMData]]) -> np.ndarray:
        """Transform DEM data to processed terrain features.
        
        Args:
            data: DEMData object or list of DEMData objects
            
        Returns:
            Processed terrain features as numpy array
        """
        if isinstance(data, list):
            features_list = []
            for dem_data in data:
                features = self._extract_terrain_features(dem_data)
                if self.config['normalize_features'] and self.scaler is not None:
                    features = self.scaler.transform(features.reshape(1, -1)).flatten()
                features_list.append(features)
            return np.array(features_list)
        else:
            features = self._extract_terrain_features(data)
            if self.config['normalize_features'] and self.scaler is not None:
                features = self.scaler.transform(features.reshape(1, -1)).flatten()
            return features
    
    def _extract_terrain_features(self, dem_data: DEMData) -> np.ndarray:
        """Extract comprehensive terrain features from DEM data.
        
        Args:
            dem_data: DEMData object containing elevation matrix
            
        Returns:
            Feature vector containing terrain characteristics
        """
        elevation = dem_data.elevation_matrix
        resolution = dem_data.resolution
        
        # Calculate primary terrain derivatives
        slope = self.calculate_slope(elevation, resolution)
        aspect = self.calculate_aspect(elevation, resolution)
        curvature = self.calculate_curvature(elevation, resolution)
        
        # Calculate terrain roughness and stability indicators
        roughness = self.calculate_roughness(elevation, resolution)
        stability_indicators = self.calculate_stability_indicators(slope, aspect, curvature)
        
        # Extract statistical features
        elevation_stats = self._calculate_elevation_statistics(elevation)
        slope_stats = self._calculate_slope_statistics(slope)
        aspect_stats = self._calculate_aspect_statistics(aspect)
        curvature_stats = self._calculate_curvature_statistics(curvature)
        roughness_stats = self._calculate_roughness_statistics(roughness)
        
        # Combine all features
        features = np.concatenate([
            elevation_stats,
            slope_stats,
            aspect_stats,
            curvature_stats,
            roughness_stats,
            stability_indicators
        ])
        
        return features
    
    def calculate_slope(self, elevation: np.ndarray, resolution: float) -> np.ndarray:
        """Calculate slope from elevation data.
        
        Args:
            elevation: 2D elevation matrix
            resolution: Spatial resolution in meters per pixel
            
        Returns:
            Slope matrix in degrees
        """
        if self.config['slope_method'] == 'horn':
            return self._calculate_slope_horn(elevation, resolution)
        else:
            return self._calculate_slope_gradient(elevation, resolution)
    
    def _calculate_slope_horn(self, elevation: np.ndarray, resolution: float) -> np.ndarray:
        """Calculate slope using Horn's method (3x3 neighborhood).
        
        Args:
            elevation: 2D elevation matrix
            resolution: Spatial resolution in meters per pixel
            
        Returns:
            Slope matrix in degrees
        """
        # Horn's method kernels
        kernel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]) / (8 * resolution)
        kernel_y = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]]) / (8 * resolution)
        
        # Calculate gradients
        grad_x = ndimage.convolve(elevation, kernel_x, mode='nearest')
        grad_y = ndimage.convolve(elevation, kernel_y, mode='nearest')
        
        # Calculate slope in radians then convert to degrees
        slope_rad = np.arctan(np.sqrt(grad_x**2 + grad_y**2))
        slope_deg = np.degrees(slope_rad)
        
        return slope_deg
    
    def _calculate_slope_gradient(self, elevation: np.ndarray, resolution: float) -> np.ndarray:
        """Calculate slope using numpy gradient.
        
        Args:
            elevation: 2D elevation matrix
            resolution: Spatial resolution in meters per pixel
            
        Returns:
            Slope matrix in degrees
        """
        grad_y, grad_x = np.gradient(elevation, resolution)
        slope_rad = np.arctan(np.sqrt(grad_x**2 + grad_y**2))
        slope_deg = np.degrees(slope_rad)
        
        return slope_deg
    
    def calculate_aspect(self, elevation: np.ndarray, resolution: float) -> np.ndarray:
        """Calculate aspect (slope direction) from elevation data.
        
        Args:
            elevation: 2D elevation matrix
            resolution: Spatial resolution in meters per pixel
            
        Returns:
            Aspect matrix in degrees (0-360, where 0 is North)
        """
        # Calculate gradients
        grad_y, grad_x = np.gradient(elevation, resolution)
        
        # Calculate aspect in radians
        aspect_rad = np.arctan2(-grad_y, grad_x)
        
        # Convert to degrees and adjust to geographic convention (0 = North, clockwise)
        aspect_deg = np.degrees(aspect_rad)
        aspect_deg = 90 - aspect_deg
        aspect_deg = np.where(aspect_deg < 0, aspect_deg + 360, aspect_deg)
        aspect_deg = np.where(aspect_deg >= 360, aspect_deg - 360, aspect_deg)
        
        return aspect_deg
    
    def calculate_curvature(self, elevation: np.ndarray, resolution: float) -> Dict[str, np.ndarray]:
        """Calculate various curvature measures from elevation data.
        
        Args:
            elevation: 2D elevation matrix
            resolution: Spatial resolution in meters per pixel
            
        Returns:
            Dictionary containing different curvature measures
        """
        if self.config['curvature_method'] == 'zevenbergen':
            return self._calculate_curvature_zevenbergen(elevation, resolution)
        else:
            return self._calculate_curvature_wood(elevation, resolution)
    
    def _calculate_curvature_zevenbergen(self, elevation: np.ndarray, resolution: float) -> Dict[str, np.ndarray]:
        """Calculate curvature using Zevenbergen & Thorne method.
        
        Args:
            elevation: 2D elevation matrix
            resolution: Spatial resolution in meters per pixel
            
        Returns:
            Dictionary with profile, planform, and mean curvature
        """
        # Calculate second derivatives using finite differences
        grad_y, grad_x = np.gradient(elevation, resolution)
        grad_yy, grad_xy = np.gradient(grad_y, resolution)
        grad_yx, grad_xx = np.gradient(grad_x, resolution)
        
        # Calculate curvature components
        p = grad_x
        q = grad_y
        r = grad_xx
        s = grad_xy
        t = grad_yy
        
        # Profile curvature (curvature in the direction of maximum slope)
        denominator = (p**2 + q**2)**(3/2)
        denominator = np.where(denominator == 0, 1e-10, denominator)
        profile_curvature = -(r*p**2 + 2*s*p*q + t*q**2) / denominator
        
        # Planform curvature (curvature perpendicular to the direction of maximum slope)
        denominator = (p**2 + q**2)**(1/2)
        denominator = np.where(denominator == 0, 1e-10, denominator)
        planform_curvature = (r*q**2 - 2*s*p*q + t*p**2) / denominator
        
        # Mean curvature
        mean_curvature = (profile_curvature + planform_curvature) / 2
        
        return {
            'profile': profile_curvature,
            'planform': planform_curvature,
            'mean': mean_curvature
        }
    
    def _calculate_curvature_wood(self, elevation: np.ndarray, resolution: float) -> Dict[str, np.ndarray]:
        """Calculate curvature using Wood's method.
        
        Args:
            elevation: 2D elevation matrix
            resolution: Spatial resolution in meters per pixel
            
        Returns:
            Dictionary with profile, planform, and mean curvature
        """
        # Use 3x3 neighborhood for second derivatives
        kernel_xx = np.array([[0, 0, 0], [1, -2, 1], [0, 0, 0]]) / resolution**2
        kernel_yy = np.array([[0, 1, 0], [0, -2, 0], [0, 1, 0]]) / resolution**2
        kernel_xy = np.array([[-1, 0, 1], [0, 0, 0], [1, 0, -1]]) / (4 * resolution**2)
        
        # Calculate second derivatives
        d2z_dx2 = ndimage.convolve(elevation, kernel_xx, mode='nearest')
        d2z_dy2 = ndimage.convolve(elevation, kernel_yy, mode='nearest')
        d2z_dxdy = ndimage.convolve(elevation, kernel_xy, mode='nearest')
        
        # Calculate first derivatives
        grad_y, grad_x = np.gradient(elevation, resolution)
        
        # Calculate curvatures
        p = grad_x
        q = grad_y
        
        # Profile curvature
        denominator = (p**2 + q**2)**(3/2)
        denominator = np.where(denominator == 0, 1e-10, denominator)
        profile_curvature = -(d2z_dx2*p**2 + 2*d2z_dxdy*p*q + d2z_dy2*q**2) / denominator
        
        # Planform curvature
        denominator = (p**2 + q**2)**(1/2)
        denominator = np.where(denominator == 0, 1e-10, denominator)
        planform_curvature = (d2z_dx2*q**2 - 2*d2z_dxdy*p*q + d2z_dy2*p**2) / denominator
        
        # Mean curvature
        mean_curvature = (profile_curvature + planform_curvature) / 2
        
        return {
            'profile': profile_curvature,
            'planform': planform_curvature,
            'mean': mean_curvature
        }    

    def calculate_roughness(self, elevation: np.ndarray, resolution: float) -> np.ndarray:
        """Calculate terrain roughness.
        
        Args:
            elevation: 2D elevation matrix
            resolution: Spatial resolution in meters per pixel
            
        Returns:
            Roughness matrix
        """
        if self.config['roughness_method'] == 'std':
            return self._calculate_roughness_std(elevation)
        else:
            return self._calculate_roughness_tri(elevation, resolution)
    
    def _calculate_roughness_std(self, elevation: np.ndarray) -> np.ndarray:
        """Calculate roughness using standard deviation in a moving window.
        
        Args:
            elevation: 2D elevation matrix
            
        Returns:
            Roughness matrix based on local standard deviation
        """
        window_size = self.config['window_size']
        
        # Create a kernel for the moving window
        kernel = np.ones((window_size, window_size)) / (window_size**2)
        
        # Calculate local mean
        local_mean = ndimage.convolve(elevation, kernel, mode='nearest')
        
        # Calculate local variance
        local_variance = ndimage.convolve((elevation - local_mean)**2, kernel, mode='nearest')
        
        # Roughness is the standard deviation
        roughness = np.sqrt(local_variance)
        
        return roughness
    
    def _calculate_roughness_tri(self, elevation: np.ndarray, resolution: float) -> np.ndarray:
        """Calculate Terrain Ruggedness Index (TRI).
        
        Args:
            elevation: 2D elevation matrix
            resolution: Spatial resolution in meters per pixel
            
        Returns:
            TRI roughness matrix
        """
        # Calculate mean elevation in 3x3 neighborhood
        kernel = np.ones((3, 3)) / 9
        local_mean = ndimage.convolve(elevation, kernel, mode='nearest')
        
        # TRI is the mean absolute difference from the local mean
        tri = np.abs(elevation - local_mean)
        
        return tri
    
    def calculate_stability_indicators(self, slope: np.ndarray, aspect: np.ndarray, 
                                     curvature: Dict[str, np.ndarray]) -> np.ndarray:
        """Calculate terrain stability indicators.
        
        Args:
            slope: Slope matrix in degrees
            aspect: Aspect matrix in degrees
            curvature: Dictionary of curvature matrices
            
        Returns:
            Array of stability indicator features
        """
        stability_threshold = self.config['stability_threshold']
        
        # Calculate stability indicators
        indicators = []
        
        # 1. Percentage of area with slope > threshold
        unstable_slope_ratio = np.mean(slope > stability_threshold)
        indicators.append(unstable_slope_ratio)
        
        # 2. Maximum slope value
        max_slope = np.max(slope)
        indicators.append(max_slope)
        
        # 3. Mean slope in unstable areas
        unstable_areas = slope > stability_threshold
        if np.any(unstable_areas):
            mean_unstable_slope = np.mean(slope[unstable_areas])
        else:
            mean_unstable_slope = 0.0
        indicators.append(mean_unstable_slope)
        
        # 4. Aspect variability (indicates terrain complexity)
        # Convert aspect to radians for circular statistics
        aspect_rad = np.radians(aspect)
        aspect_variability = 1 - np.abs(np.mean(np.exp(1j * aspect_rad)))
        indicators.append(aspect_variability)
        
        # 5. Profile curvature statistics (concave areas are more unstable)
        profile_curvature = curvature['profile']
        concave_ratio = np.mean(profile_curvature < 0)  # Negative curvature = concave
        indicators.append(concave_ratio)
        
        # 6. Mean absolute profile curvature
        mean_abs_profile_curvature = np.mean(np.abs(profile_curvature))
        indicators.append(mean_abs_profile_curvature)
        
        # 7. Planform curvature statistics
        planform_curvature = curvature['planform']
        convergent_ratio = np.mean(planform_curvature < 0)  # Convergent areas
        indicators.append(convergent_ratio)
        
        # 8. Combined instability index
        # Areas with high slope AND concave profile curvature
        combined_instability = np.mean((slope > stability_threshold) & (profile_curvature < 0))
        indicators.append(combined_instability)
        
        return np.array(indicators)
    
    def _calculate_elevation_statistics(self, elevation: np.ndarray) -> np.ndarray:
        """Calculate statistical features from elevation data.
        
        Args:
            elevation: 2D elevation matrix
            
        Returns:
            Array of elevation statistics
        """
        stats = [
            np.mean(elevation),
            np.std(elevation),
            np.min(elevation),
            np.max(elevation),
            np.percentile(elevation, 25),
            np.percentile(elevation, 75),
            np.max(elevation) - np.min(elevation),  # Range
        ]
        return np.array(stats)
    
    def _calculate_slope_statistics(self, slope: np.ndarray) -> np.ndarray:
        """Calculate statistical features from slope data.
        
        Args:
            slope: Slope matrix in degrees
            
        Returns:
            Array of slope statistics
        """
        stats = [
            np.mean(slope),
            np.std(slope),
            np.min(slope),
            np.max(slope),
            np.percentile(slope, 90),  # 90th percentile for extreme slopes
            np.percentile(slope, 95),  # 95th percentile
        ]
        return np.array(stats)
    
    def _calculate_aspect_statistics(self, aspect: np.ndarray) -> np.ndarray:
        """Calculate statistical features from aspect data.
        
        Args:
            aspect: Aspect matrix in degrees
            
        Returns:
            Array of aspect statistics
        """
        # Convert to radians for circular statistics
        aspect_rad = np.radians(aspect)
        
        # Calculate circular mean and variance
        circular_mean = np.angle(np.mean(np.exp(1j * aspect_rad)))
        circular_variance = 1 - np.abs(np.mean(np.exp(1j * aspect_rad)))
        
        # Convert back to degrees
        circular_mean_deg = np.degrees(circular_mean)
        if circular_mean_deg < 0:
            circular_mean_deg += 360
        
        # Calculate aspect distribution (percentage in each quadrant)
        north_ratio = np.mean((aspect >= 315) | (aspect < 45))
        east_ratio = np.mean((aspect >= 45) & (aspect < 135))
        south_ratio = np.mean((aspect >= 135) & (aspect < 225))
        west_ratio = np.mean((aspect >= 225) & (aspect < 315))
        
        stats = [
            circular_mean_deg,
            circular_variance,
            north_ratio,
            east_ratio,
            south_ratio,
            west_ratio,
        ]
        return np.array(stats)
    
    def _calculate_curvature_statistics(self, curvature: Dict[str, np.ndarray]) -> np.ndarray:
        """Calculate statistical features from curvature data.
        
        Args:
            curvature: Dictionary of curvature matrices
            
        Returns:
            Array of curvature statistics
        """
        profile = curvature['profile']
        planform = curvature['planform']
        mean_curv = curvature['mean']
        
        stats = [
            # Profile curvature stats
            np.mean(profile),
            np.std(profile),
            np.mean(profile < 0),  # Concave ratio
            
            # Planform curvature stats
            np.mean(planform),
            np.std(planform),
            np.mean(planform < 0),  # Convergent ratio
            
            # Mean curvature stats
            np.mean(mean_curv),
            np.std(mean_curv),
            np.mean(np.abs(mean_curv)),  # Mean absolute curvature
        ]
        return np.array(stats)
    
    def _calculate_roughness_statistics(self, roughness: np.ndarray) -> np.ndarray:
        """Calculate statistical features from roughness data.
        
        Args:
            roughness: Roughness matrix
            
        Returns:
            Array of roughness statistics
        """
        stats = [
            np.mean(roughness),
            np.std(roughness),
            np.max(roughness),
            np.percentile(roughness, 90),
            np.percentile(roughness, 95),
        ]
        return np.array(stats)
    
    def read_geotiff(self, file_path: str) -> DEMData:
        """Read GeoTIFF file and create DEMData object.
        
        Args:
            file_path: Path to GeoTIFF file
            
        Returns:
            DEMData object with elevation matrix and metadata
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"GeoTIFF file not found: {file_path}")
        
        try:
            with rasterio.open(file_path) as src:
                # Read elevation data
                elevation_matrix = src.read(1).astype(np.float64)
                
                # Handle nodata values
                if src.nodata is not None:
                    elevation_matrix = np.where(
                        elevation_matrix == src.nodata, 
                        np.nan, 
                        elevation_matrix
                    )
                
                # Get spatial information
                transform = src.transform
                resolution = abs(transform[0])  # Assuming square pixels
                
                # Get bounds
                bounds = {
                    'west': src.bounds.left,
                    'east': src.bounds.right,
                    'south': src.bounds.bottom,
                    'north': src.bounds.top
                }
                
                return DEMData(
                    elevation_matrix=elevation_matrix,
                    resolution=resolution,
                    bounds=bounds
                )
                
        except Exception as e:
            raise ValueError(f"Error reading GeoTIFF file {file_path}: {str(e)}")
    
    def preprocess_dem(self, dem_data: DEMData) -> DEMData:
        """Preprocess DEM data (fill gaps, smooth, etc.).
        
        Args:
            dem_data: Input DEMData object
            
        Returns:
            Preprocessed DEMData object
        """
        elevation = dem_data.elevation_matrix.copy()
        
        # Fill NaN values using interpolation
        if np.any(np.isnan(elevation)):
            elevation = self._fill_nan_values(elevation)
        
        # Optional smoothing to reduce noise
        if self.config.get('smooth_dem', False):
            sigma = self.config.get('smooth_sigma', 1.0)
            elevation = ndimage.gaussian_filter(elevation, sigma=sigma)
        
        return DEMData(
            elevation_matrix=elevation,
            resolution=dem_data.resolution,
            bounds=dem_data.bounds
        )
    
    def _fill_nan_values(self, elevation: np.ndarray) -> np.ndarray:
        """Fill NaN values in elevation matrix using interpolation.
        
        Args:
            elevation: Elevation matrix with potential NaN values
            
        Returns:
            Elevation matrix with filled values
        """
        # Create mask for valid values
        valid_mask = ~np.isnan(elevation)
        
        if not np.any(valid_mask):
            raise ValueError("All elevation values are NaN")
        
        # Get coordinates of valid and invalid points
        rows, cols = np.indices(elevation.shape)
        valid_points = np.column_stack((rows[valid_mask], cols[valid_mask]))
        valid_values = elevation[valid_mask]
        
        # Find NaN locations
        nan_mask = np.isnan(elevation)
        if not np.any(nan_mask):
            return elevation  # No NaN values to fill
        
        nan_points = np.column_stack((rows[nan_mask], cols[nan_mask]))
        
        # Use nearest neighbor interpolation for simplicity
        from scipy.spatial.distance import cdist
        
        # Calculate distances from each NaN point to all valid points
        distances = cdist(nan_points, valid_points)
        
        # Find nearest valid point for each NaN point
        nearest_indices = np.argmin(distances, axis=1)
        
        # Fill NaN values with nearest valid values
        filled_elevation = elevation.copy()
        filled_elevation[nan_mask] = valid_values[nearest_indices]
        
        return filled_elevation
    
    def extract_terrain_profile(self, dem_data: DEMData, start_coord: GeoCoordinate, 
                               end_coord: GeoCoordinate, num_points: int = 100) -> Dict[str, np.ndarray]:
        """Extract elevation profile along a line.
        
        Args:
            dem_data: DEMData object
            start_coord: Starting coordinate
            end_coord: Ending coordinate
            num_points: Number of points along the profile
            
        Returns:
            Dictionary with profile data (distances, elevations, slopes)
        """
        # Convert geographic coordinates to pixel coordinates
        bounds = dem_data.bounds
        resolution = dem_data.resolution
        
        # Calculate pixel coordinates
        start_x = int((start_coord.longitude - bounds['west']) / resolution)
        start_y = int((bounds['north'] - start_coord.latitude) / resolution)
        end_x = int((end_coord.longitude - bounds['west']) / resolution)
        end_y = int((bounds['north'] - end_coord.latitude) / resolution)
        
        # Generate line coordinates
        x_coords = np.linspace(start_x, end_x, num_points).astype(int)
        y_coords = np.linspace(start_y, end_y, num_points).astype(int)
        
        # Ensure coordinates are within bounds
        height, width = dem_data.elevation_matrix.shape
        x_coords = np.clip(x_coords, 0, width - 1)
        y_coords = np.clip(y_coords, 0, height - 1)
        
        # Extract elevations along the profile
        elevations = dem_data.elevation_matrix[y_coords, x_coords]
        
        # Calculate distances along the profile
        distances = np.linspace(0, np.sqrt(
            ((end_coord.longitude - start_coord.longitude) * 111320)**2 +
            ((end_coord.latitude - start_coord.latitude) * 110540)**2
        ), num_points)
        
        # Calculate slopes along the profile
        slopes = np.zeros_like(elevations)
        if len(elevations) > 1:
            elevation_diff = np.diff(elevations)
            distance_diff = np.diff(distances)
            distance_diff = np.where(distance_diff == 0, 1e-10, distance_diff)
            profile_slopes = np.degrees(np.arctan(elevation_diff / distance_diff))
            slopes[1:] = profile_slopes
        
        return {
            'distances': distances,
            'elevations': elevations,
            'slopes': slopes,
            'coordinates': list(zip(x_coords, y_coords))
        }