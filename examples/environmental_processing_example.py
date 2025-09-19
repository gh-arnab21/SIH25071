"""Example usage of EnvironmentalProcessor for processing weather and environmental data."""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from datetime import datetime, timedelta
from src.data.processors.environmental_processor import EnvironmentalProcessor
from src.data.schemas import EnvironmentalData


def create_sample_environmental_data():
    """Create sample environmental data for demonstration."""
    # Generate 100 samples of environmental data with various conditions
    environmental_data = []
    
    for i in range(100):
        # Simulate different weather patterns
        if i < 30:  # Normal conditions
            rainfall = np.random.exponential(2.0)
            temperature = np.random.normal(20.0, 5.0)
            vibrations = np.random.exponential(0.5)
            wind_speed = np.random.exponential(8.0)
        elif i < 60:  # Rainy season
            rainfall = np.random.exponential(8.0)
            temperature = np.random.normal(15.0, 3.0)
            vibrations = np.random.exponential(1.0)
            wind_speed = np.random.exponential(12.0)
        elif i < 80:  # Winter conditions
            rainfall = np.random.exponential(1.0)
            temperature = np.random.normal(-5.0, 8.0)
            vibrations = np.random.exponential(0.3)
            wind_speed = np.random.exponential(15.0)
        else:  # Extreme weather
            rainfall = np.random.exponential(15.0)
            temperature = np.random.normal(35.0, 10.0)
            vibrations = np.random.exponential(3.0)
            wind_speed = np.random.exponential(25.0)
        
        env_data = EnvironmentalData(
            rainfall=max(0, rainfall),
            temperature=temperature,
            vibrations=max(0, vibrations),
            wind_speed=max(0, wind_speed)
        )
        environmental_data.append(env_data)
    
    return environmental_data


def demonstrate_environmental_processing():
    """Demonstrate the EnvironmentalProcessor functionality."""
    print("Environmental Data Processing Example")
    print("=" * 50)
    
    # Create sample data
    print("1. Creating sample environmental data...")
    environmental_data = create_sample_environmental_data()
    print(f"   Generated {len(environmental_data)} environmental data points")
    
    # Initialize and fit the processor
    print("\n2. Initializing and fitting EnvironmentalProcessor...")
    processor = EnvironmentalProcessor(config={
        'rainfall_trigger_threshold': 30.0,
        'vibration_threshold': 3.0,
        'normalization': 'standard'
    })
    
    # Fit the processor
    processor.fit(environmental_data)
    print("   Processor fitted successfully!")
    
    # Display statistics
    print("\n3. Environmental parameter statistics:")
    for param in ['rainfall', 'temperature', 'vibrations', 'wind_speed']:
        stats = processor.get_parameter_statistics(param)
        if stats:
            print(f"   {param.capitalize()}:")
            print(f"     Mean: {stats['mean']:.2f}, Std: {stats['std']:.2f}")
            print(f"     Range: [{stats['min']:.2f}, {stats['max']:.2f}]")
    
    # Display trigger thresholds
    print("\n4. Trigger thresholds:")
    thresholds = processor.get_trigger_thresholds()
    for param, threshold in thresholds.items():
        print(f"   {param}: {threshold:.2f}")
    
    # Process some test data
    print("\n5. Processing test environmental data...")
    
    # Test case 1: Normal conditions
    normal_data = EnvironmentalData(
        rainfall=5.0, temperature=22.0, vibrations=1.0, wind_speed=8.0
    )
    
    # Test case 2: Extreme conditions
    extreme_data = EnvironmentalData(
        rainfall=45.0, temperature=42.0, vibrations=8.0, wind_speed=30.0
    )
    
    # Test case 3: Freeze-thaw conditions
    freeze_thaw_data = EnvironmentalData(
        rainfall=10.0, temperature=-1.0, vibrations=2.0, wind_speed=12.0
    )
    
    test_cases = [
        ("Normal conditions", normal_data),
        ("Extreme conditions", extreme_data),
        ("Freeze-thaw conditions", freeze_thaw_data)
    ]
    
    for case_name, test_data in test_cases:
        print(f"\n   {case_name}:")
        print(f"     Input: Rainfall={test_data.rainfall}, Temp={test_data.temperature}, "
              f"Vibrations={test_data.vibrations}, Wind={test_data.wind_speed}")
        
        # Extract features
        features = processor.transform(test_data)
        print(f"     Extracted {len(features)} features")
        
        # Validate data
        validation = processor.validate_environmental_data(test_data)
        print(f"     Validation: {'Valid' if validation['is_valid'] else 'Invalid'}")
        if validation['warnings']:
            print(f"     Warnings: {len(validation['warnings'])}")
        if validation['errors']:
            print(f"     Errors: {len(validation['errors'])}")
        
        # Detect triggers
        triggers = processor.detect_trigger_conditions(test_data)
        print(f"     Trigger status: {triggers['overall_trigger_status']}")
        print(f"     Risk level: {triggers['risk_level']}")
        if triggers['triggered_conditions']:
            print(f"     Triggered conditions: {', '.join(triggers['triggered_conditions'])}")
    
    # Demonstrate cumulative effects
    print("\n6. Demonstrating cumulative effects analysis...")
    
    # Create a time series of environmental data
    time_series_data = []
    base_time = datetime.now()
    
    for hour in range(48):  # 48 hours of data
        timestamp = (base_time + timedelta(hours=hour)).timestamp()
        
        # Simulate a storm event in the middle
        if 20 <= hour <= 30:
            rainfall = np.random.exponential(12.0)
            temperature = 5.0 + np.random.normal(0, 2.0)
            vibrations = np.random.exponential(4.0)
            wind_speed = 20.0 + np.random.exponential(10.0)
        else:
            rainfall = np.random.exponential(2.0)
            temperature = 15.0 + np.random.normal(0, 5.0)
            vibrations = np.random.exponential(1.0)
            wind_speed = np.random.exponential(8.0)
        
        env_data = EnvironmentalData(
            rainfall=max(0, rainfall),
            temperature=temperature,
            vibrations=max(0, vibrations),
            wind_speed=max(0, wind_speed)
        )
        time_series_data.append((timestamp, env_data))
    
    # Calculate cumulative effects
    cumulative_effects = processor.calculate_cumulative_effects(time_series_data)
    
    print("   Cumulative effects analysis:")
    if 'cumulative_rainfall' in cumulative_effects:
        rainfall_effects = cumulative_effects['cumulative_rainfall']
        print(f"     Cumulative rainfall: {rainfall_effects['value']:.2f} mm")
        print(f"     Intensity: {rainfall_effects['intensity']}")
        print(f"     Trigger exceeded: {rainfall_effects['trigger_exceeded']}")
    
    if 'temperature_effects' in cumulative_effects:
        temp_effects = cumulative_effects['temperature_effects']
        print(f"     Freeze-thaw cycles: {temp_effects['freeze_thaw_cycles']}")
        print(f"     Temperature trend: {temp_effects['trend']}")
        print(f"     Temperature range: {temp_effects['temperature_range']:.2f}°C")
    
    if 'overall_risk_assessment' in cumulative_effects:
        risk_assessment = cumulative_effects['overall_risk_assessment']
        print(f"     Overall risk level: {risk_assessment['risk_level']}")
        print(f"     Risk factors: {', '.join(risk_assessment['risk_factors'])}")
    
    # Display feature names
    print("\n7. Available features:")
    feature_names = processor.get_feature_names()
    print(f"   Total features: {len(feature_names)}")
    print("   Feature categories:")
    
    categories = {
        'Rainfall': [name for name in feature_names if name.startswith('rainfall')],
        'Temperature': [name for name in feature_names if name.startswith('temperature')],
        'Vibrations': [name for name in feature_names if name.startswith('vibrations')],
        'Wind Speed': [name for name in feature_names if name.startswith('wind')],
        'Combined Risk': [name for name in feature_names if name in ['freeze_thaw_risk', 'rain_vibration_risk', 'extreme_weather_risk', 'environmental_stress_index']]
    }
    
    for category, features in categories.items():
        print(f"     {category}: {len(features)} features")
        for feature in features[:3]:  # Show first 3 features
            print(f"       - {feature}")
        if len(features) > 3:
            print(f"       ... and {len(features) - 3} more")
    
    print("\nEnvironmental data processing demonstration completed!")


if __name__ == "__main__":
    demonstrate_environmental_processing()