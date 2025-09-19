"""Example usage of TerrainProcessor for DEM data processing."""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import matplotlib.pyplot as plt
from src.data.processors.terrain_processor import TerrainProcessor
from src.data.schemas import DEMData, GeoCoordinate


def create_sample_dem():
    """Create a sample DEM for demonstration."""
    # Create a synthetic terrain with hills and valleys
    x = np.linspace(0, 10, 50)
    y = np.linspace(0, 10, 50)
    X, Y = np.meshgrid(x, y)
    
    # Create terrain with multiple features
    elevation = (
        1000 +  # Base elevation
        200 * np.sin(X * 0.5) * np.cos(Y * 0.5) +  # Large hills
        50 * np.sin(X * 2) * np.sin(Y * 2) +       # Smaller features
        20 * np.random.normal(0, 1, X.shape)        # Noise
    )
    
    return DEMData(
        elevation_matrix=elevation,
        resolution=20.0,  # 20 meters per pixel
        bounds={
            'west': -120.0,
            'east': -119.8,
            'south': 35.0,
            'north': 35.2
        }
    )


def demonstrate_terrain_processing():
    """Demonstrate terrain processing capabilities."""
    print("Terrain Processing Example")
    print("=" * 50)
    
    # Create sample DEM data
    print("1. Creating sample DEM data...")
    dem_data = create_sample_dem()
    print(f"   DEM shape: {dem_data.elevation_matrix.shape}")
    print(f"   Resolution: {dem_data.resolution} meters/pixel")
    print(f"   Elevation range: {np.min(dem_data.elevation_matrix):.1f} - {np.max(dem_data.elevation_matrix):.1f} m")
    
    # Initialize terrain processor
    print("\n2. Initializing TerrainProcessor...")
    config = {
        'slope_method': 'horn',
        'curvature_method': 'zevenbergen',
        'roughness_method': 'std',
        'stability_threshold': 25.0,  # 25 degrees
        'normalize_features': True
    }
    processor = TerrainProcessor(config)
    
    # Calculate individual terrain attributes
    print("\n3. Calculating terrain attributes...")
    
    # Slope calculation
    slope = processor.calculate_slope(dem_data.elevation_matrix, dem_data.resolution)
    print(f"   Slope - Mean: {np.mean(slope):.2f}°, Max: {np.max(slope):.2f}°")
    
    # Aspect calculation
    aspect = processor.calculate_aspect(dem_data.elevation_matrix, dem_data.resolution)
    print(f"   Aspect - Range: {np.min(aspect):.1f}° - {np.max(aspect):.1f}°")
    
    # Curvature calculation
    curvature = processor.calculate_curvature(dem_data.elevation_matrix, dem_data.resolution)
    print(f"   Profile curvature - Mean: {np.mean(curvature['profile']):.6f}")
    print(f"   Planform curvature - Mean: {np.mean(curvature['planform']):.6f}")
    
    # Roughness calculation
    roughness = processor.calculate_roughness(dem_data.elevation_matrix, dem_data.resolution)
    print(f"   Roughness - Mean: {np.mean(roughness):.2f}, Max: {np.max(roughness):.2f}")
    
    # Stability indicators
    stability = processor.calculate_stability_indicators(slope, aspect, curvature)
    print(f"   Unstable slope ratio: {stability[0]:.3f}")
    print(f"   Maximum slope: {stability[1]:.2f}°")
    print(f"   Combined instability index: {stability[7]:.3f}")
    
    # Extract comprehensive features
    print("\n4. Extracting comprehensive terrain features...")
    processor.fit([dem_data])
    features = processor.transform(dem_data)
    print(f"   Total features extracted: {len(features)}")
    print(f"   Feature vector shape: {features.shape}")
    
    # Demonstrate terrain profile extraction
    print("\n5. Extracting terrain profile...")
    start_coord = GeoCoordinate(latitude=35.05, longitude=-119.95)
    end_coord = GeoCoordinate(latitude=35.15, longitude=-119.85)
    
    profile = processor.extract_terrain_profile(dem_data, start_coord, end_coord, num_points=100)
    print(f"   Profile length: {profile['distances'][-1]:.0f} meters")
    print(f"   Elevation change: {profile['elevations'][-1] - profile['elevations'][0]:.1f} meters")
    print(f"   Max slope along profile: {np.max(np.abs(profile['slopes'])):.2f}°")
    
    # Create visualizations
    print("\n6. Creating visualizations...")
    create_terrain_visualizations(dem_data, slope, aspect, curvature, roughness, profile)
    
    print("\nTerrain processing example completed!")
    print("Check the generated plots for visual results.")


def create_terrain_visualizations(dem_data, slope, aspect, curvature, roughness, profile):
    """Create visualization plots for terrain analysis."""
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle('Terrain Analysis Results', fontsize=16)
    
    # Elevation
    im1 = axes[0, 0].imshow(dem_data.elevation_matrix, cmap='terrain', aspect='equal')
    axes[0, 0].set_title('Elevation (m)')
    plt.colorbar(im1, ax=axes[0, 0])
    
    # Slope
    im2 = axes[0, 1].imshow(slope, cmap='Reds', aspect='equal')
    axes[0, 1].set_title('Slope (degrees)')
    plt.colorbar(im2, ax=axes[0, 1])
    
    # Aspect
    im3 = axes[0, 2].imshow(aspect, cmap='hsv', aspect='equal')
    axes[0, 2].set_title('Aspect (degrees)')
    plt.colorbar(im3, ax=axes[0, 2])
    
    # Profile curvature
    im4 = axes[1, 0].imshow(curvature['profile'], cmap='RdBu_r', aspect='equal')
    axes[1, 0].set_title('Profile Curvature')
    plt.colorbar(im4, ax=axes[1, 0])
    
    # Roughness
    im5 = axes[1, 1].imshow(roughness, cmap='viridis', aspect='equal')
    axes[1, 1].set_title('Terrain Roughness')
    plt.colorbar(im5, ax=axes[1, 1])
    
    # Terrain profile
    axes[1, 2].plot(profile['distances'] / 1000, profile['elevations'], 'b-', linewidth=2)
    axes[1, 2].set_xlabel('Distance (km)')
    axes[1, 2].set_ylabel('Elevation (m)')
    axes[1, 2].set_title('Terrain Profile')
    axes[1, 2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('terrain_analysis_results.png', dpi=300, bbox_inches='tight')
    plt.show()


def demonstrate_multiple_dem_processing():
    """Demonstrate processing multiple DEM datasets."""
    print("\nMultiple DEM Processing Example")
    print("=" * 50)
    
    # Create multiple sample DEMs
    dem_list = []
    for i in range(3):
        # Create different terrain types
        x = np.linspace(0, 8, 40)
        y = np.linspace(0, 8, 40)
        X, Y = np.meshgrid(x, y)
        
        if i == 0:  # Mountainous terrain
            elevation = 2000 + 500 * np.sin(X * 0.3) * np.cos(Y * 0.3) + 100 * np.random.normal(0, 1, X.shape)
        elif i == 1:  # Rolling hills
            elevation = 500 + 100 * np.sin(X * 0.8) * np.sin(Y * 0.8) + 20 * np.random.normal(0, 1, X.shape)
        else:  # Flat with small features
            elevation = 100 + 10 * np.sin(X * 2) + 5 * np.random.normal(0, 1, X.shape)
        
        dem = DEMData(
            elevation_matrix=elevation,
            resolution=25.0,
            bounds={'west': -120.0 - i*0.1, 'east': -119.8 - i*0.1, 'south': 35.0, 'north': 35.2}
        )
        dem_list.append(dem)
    
    print(f"Created {len(dem_list)} sample DEMs")
    
    # Process all DEMs
    processor = TerrainProcessor({'normalize_features': True})
    
    # Fit on all DEMs and transform
    all_features = processor.fit_transform(dem_list)
    
    print(f"Extracted features shape: {all_features.shape}")
    print(f"Features per DEM: {all_features.shape[1]}")
    
    # Show feature statistics
    print("\nFeature statistics across DEMs:")
    print(f"Mean feature values: {np.mean(all_features, axis=0)[:10]}...")  # Show first 10
    print(f"Feature std deviation: {np.std(all_features, axis=0)[:10]}...")  # Show first 10
    
    # Compare terrain characteristics
    terrain_types = ['Mountainous', 'Rolling Hills', 'Flat']
    for i, (features, terrain_type) in enumerate(zip(all_features, terrain_types)):
        print(f"\n{terrain_type} terrain:")
        print(f"  Mean elevation: {features[0]:.2f}")
        print(f"  Elevation std: {features[1]:.2f}")
        print(f"  Mean slope: {features[7]:.2f}")
        print(f"  Max slope: {features[10]:.2f}")


if __name__ == "__main__":
    # Run the main demonstration
    demonstrate_terrain_processing()
    
    # Run multiple DEM processing example
    demonstrate_multiple_dem_processing()