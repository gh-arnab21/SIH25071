"""Example usage of SensorDataProcessor for geotechnical sensor data analysis."""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

from src.data.processors.sensor_processor import SensorDataProcessor
from src.data.schemas import SensorData, TimeSeries


def create_sample_sensor_data():
    """Create sample sensor data for demonstration."""
    # Generate 24 hours of data at 1-minute intervals
    duration_hours = 24
    sampling_interval = 60  # seconds
    num_points = duration_hours * 60  # 1440 points
    
    # Create timestamps
    start_time = 0
    timestamps = np.arange(start_time, start_time + num_points * sampling_interval, sampling_interval)
    
    # Generate displacement data with trend and noise
    displacement_trend = 0.001 * timestamps / 3600  # 0.001 mm/hour trend
    displacement_noise = np.random.normal(0, 0.1, len(timestamps))
    displacement_values = 5.0 + displacement_trend + displacement_noise
    
    # Add some anomalies
    displacement_values[500:510] += 2.0  # Sudden increase
    displacement_values[1000] = 15.0     # Single spike
    
    displacement_ts = TimeSeries(
        timestamps=timestamps,
        values=displacement_values,
        unit='mm',
        sampling_rate=1/60  # 1 sample per minute
    )
    
    # Generate strain data
    strain_base = 500  # microstrain
    strain_variation = 50 * np.sin(2 * np.pi * timestamps / (12 * 3600))  # 12-hour cycle
    strain_noise = np.random.normal(0, 10, len(timestamps))
    strain_values = strain_base + strain_variation + strain_noise
    
    # Add strain anomalies
    strain_values[800:820] += 200  # Sustained increase
    
    strain_ts = TimeSeries(
        timestamps=timestamps,
        values=strain_values,
        unit='microstrain',
        sampling_rate=1/60
    )
    
    # Generate pore pressure data
    pp_base = 200  # kPa
    pp_trend = 0.5 * timestamps / 3600  # 0.5 kPa/hour increase
    pp_noise = np.random.normal(0, 5, len(timestamps))
    pp_values = pp_base + pp_trend + pp_noise
    
    # Add pore pressure anomalies
    pp_values[1200:1250] += 50  # Rapid increase
    
    pore_pressure_ts = TimeSeries(
        timestamps=timestamps,
        values=pp_values,
        unit='kPa',
        sampling_rate=1/60
    )
    
    return SensorData(
        displacement=displacement_ts,
        strain=strain_ts,
        pore_pressure=pore_pressure_ts
    )


def demonstrate_sensor_processing():
    """Demonstrate sensor data processing capabilities."""
    print("Rockfall Prediction System - Sensor Data Processing Example")
    print("=" * 60)
    
    # Create sample data
    print("1. Creating sample sensor data...")
    sensor_data = create_sample_sensor_data()
    
    # Initialize processor
    print("2. Initializing SensorDataProcessor...")
    config = {
        'sampling_rate': 1/60,  # 1 sample per minute
        'filter_type': 'lowpass',
        'filter_params': {'cutoff': 1/3600, 'order': 4},  # 1-hour cutoff
        'normalization': 'standard',
        'anomaly_method': 'isolation_forest',
        'trend_window': 60,  # 1-hour window
    }
    
    processor = SensorDataProcessor(config)
    
    # Fit the processor
    print("3. Fitting processor to training data...")
    processor.fit([sensor_data])
    
    # Extract features
    print("4. Extracting features from sensor data...")
    features = processor.transform(sensor_data)
    print(f"   Extracted {len(features)} features")
    
    # Analyze displacement data
    print("\n5. Analyzing displacement data...")
    displacement_results = processor.process_displacement_data(sensor_data.displacement)
    print(f"   Sensor type: {displacement_results['sensor_type']}")
    print(f"   Number of features: {len(displacement_results['features'])}")
    print(f"   Trend: {displacement_results['trends']['trend']}")
    print(f"   Trend significance: {displacement_results['trends']['significance']:.3f}")
    print(f"   Number of anomalies: {len(displacement_results['anomalies']['anomalies'])}")
    
    if displacement_results['alerts']:
        print("   Alerts:")
        for alert in displacement_results['alerts']:
            print(f"     - {alert}")
    
    # Analyze strain data
    print("\n6. Analyzing strain data...")
    strain_results = processor.process_strain_data(sensor_data.strain)
    print(f"   Sensor type: {strain_results['sensor_type']}")
    print(f"   Trend: {strain_results['trends']['trend']}")
    print(f"   Number of anomalies: {len(strain_results['anomalies']['anomalies'])}")
    
    if strain_results['alerts']:
        print("   Alerts:")
        for alert in strain_results['alerts']:
            print(f"     - {alert}")
    
    # Analyze pore pressure data
    print("\n7. Analyzing pore pressure data...")
    pp_results = processor.process_pore_pressure_data(sensor_data.pore_pressure)
    print(f"   Sensor type: {pp_results['sensor_type']}")
    print(f"   Trend: {pp_results['trends']['trend']}")
    print(f"   Number of anomalies: {len(pp_results['anomalies']['anomalies'])}")
    
    if pp_results['alerts']:
        print("   Alerts:")
        for alert in pp_results['alerts']:
            print(f"     - {alert}")
    
    # Get sensor statistics
    print("\n8. Sensor statistics:")
    for sensor_type in ['displacement', 'strain', 'pore_pressure']:
        stats = processor.get_sensor_statistics(sensor_type)
        if stats:
            print(f"   {sensor_type.capitalize()}:")
            print(f"     Mean: {stats['mean']:.3f}")
            print(f"     Std: {stats['std']:.3f}")
            print(f"     Range: [{stats['min']:.3f}, {stats['max']:.3f}]")
    
    # Demonstrate anomaly detection
    print("\n9. Anomaly detection details:")
    anomalies = processor.detect_anomalies(sensor_data.displacement, 'displacement')
    print(f"   Method: {anomalies['method']}")
    print(f"   Total anomalies detected: {len(anomalies['anomalies'])}")
    
    if anomalies['anomalies']:
        print("   First few anomalies:")
        for i, anomaly in enumerate(anomalies['anomalies'][:3]):
            timestamp_hours = anomaly['timestamp'] / 3600
            print(f"     {i+1}. Time: {timestamp_hours:.2f}h, Value: {anomaly['value']:.3f}, Score: {anomaly['score']:.3f}")
    
    # Demonstrate trend analysis
    print("\n10. Trend analysis details:")
    trends = processor.analyze_trends(sensor_data.displacement)
    print(f"    Trend direction: {trends['trend']}")
    print(f"    Slope: {trends['slope']:.6f} mm/sample")
    print(f"    R-squared: {trends['r_squared']:.3f}")
    print(f"    P-value: {trends['p_value']:.6f}")
    print(f"    Trend strength: {trends['strength']:.3f}")
    
    print("\n" + "=" * 60)
    print("Sensor data processing demonstration completed!")
    
    return {
        'processor': processor,
        'sensor_data': sensor_data,
        'displacement_results': displacement_results,
        'strain_results': strain_results,
        'pp_results': pp_results
    }


def plot_sensor_analysis(results):
    """Plot sensor data analysis results."""
    try:
        import matplotlib.pyplot as plt
        
        sensor_data = results['sensor_data']
        displacement_results = results['displacement_results']
        
        fig, axes = plt.subplots(3, 1, figsize=(12, 10))
        
        # Plot displacement data with anomalies
        timestamps_hours = sensor_data.displacement.timestamps / 3600
        axes[0].plot(timestamps_hours, sensor_data.displacement.values, 'b-', alpha=0.7, label='Displacement')
        
        # Mark anomalies
        anomalies = displacement_results['anomalies']['anomalies']
        if anomalies:
            anomaly_times = [a['timestamp'] / 3600 for a in anomalies]
            anomaly_values = [a['value'] for a in anomalies]
            axes[0].scatter(anomaly_times, anomaly_values, color='red', s=50, label='Anomalies', zorder=5)
        
        # Plot trend line
        trends = displacement_results['trends']
        if 'fitted_line' in trends:
            axes[0].plot(timestamps_hours, trends['fitted_line'], 'r--', alpha=0.8, label='Trend')
        
        axes[0].set_ylabel('Displacement (mm)')
        axes[0].set_title('Displacement Analysis')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # Plot strain data
        axes[1].plot(timestamps_hours, sensor_data.strain.values, 'g-', alpha=0.7, label='Strain')
        axes[1].set_ylabel('Strain (microstrain)')
        axes[1].set_title('Strain Analysis')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        # Plot pore pressure data
        axes[2].plot(timestamps_hours, sensor_data.pore_pressure.values, 'm-', alpha=0.7, label='Pore Pressure')
        axes[2].set_ylabel('Pore Pressure (kPa)')
        axes[2].set_xlabel('Time (hours)')
        axes[2].set_title('Pore Pressure Analysis')
        axes[2].legend()
        axes[2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('sensor_analysis_results.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        print("Sensor analysis plot saved as 'sensor_analysis_results.png'")
        
    except ImportError:
        print("Matplotlib not available for plotting. Install with: pip install matplotlib")


if __name__ == "__main__":
    # Run the demonstration
    results = demonstrate_sensor_processing()
    
    # Create plots if matplotlib is available
    plot_sensor_analysis(results)