"""Example usage of SeismicProcessor for seismic data analysis and rockfall detection."""

import numpy as np
from datetime import datetime, timedelta
import matplotlib.pyplot as plt

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data.processors.seismic_processor import SeismicProcessor
from src.data.schemas import SeismicData


def create_synthetic_seismic_data(duration=30.0, sampling_rate=100.0, add_event=True):
    """Create synthetic seismic data for demonstration."""
    num_samples = int(duration * sampling_rate)
    t = np.linspace(0, duration, num_samples)
    
    # Base noise
    signal = np.random.normal(0, 0.1, num_samples)
    
    # Add some background seismic activity
    signal += 0.2 * np.sin(2 * np.pi * 3 * t)  # 3 Hz component
    signal += 0.1 * np.sin(2 * np.pi * 8 * t)  # 8 Hz component
    
    # Add a simulated rockfall event
    if add_event:
        event_start = int(0.4 * num_samples)
        event_duration = int(2.0 * sampling_rate)  # 2 second event
        event_end = event_start + event_duration
        
        # Create rockfall signature: high amplitude, broadband, exponential decay
        event_t = np.linspace(0, 2.0, event_duration)
        rockfall_signal = 3.0 * np.exp(-2 * event_t) * np.random.normal(1, 0.3, event_duration)
        
        # Add high frequency content typical of rockfall
        rockfall_signal += 1.0 * np.exp(-3 * event_t) * np.sin(2 * np.pi * 25 * event_t)
        
        signal[event_start:event_end] += rockfall_signal
    
    return SeismicData(
        signal=signal,
        sampling_rate=sampling_rate,
        start_time=datetime.now(),
        station_id="DEMO.STA.00.HHZ"
    )


def main():
    """Demonstrate SeismicProcessor functionality."""
    print("Seismic Data Processing Example")
    print("=" * 40)
    
    # Create synthetic training data
    print("1. Creating synthetic training data...")
    training_data = []
    for i in range(5):
        # Create varied synthetic data for training
        duration = np.random.uniform(20, 40)
        has_event = np.random.choice([True, False])
        
        seismic_data = create_synthetic_seismic_data(
            duration=duration,
            sampling_rate=100.0,
            add_event=has_event
        )
        seismic_data.station_id = f"TRAIN.STA.{i:02d}.HHZ"
        seismic_data.start_time = datetime.now() + timedelta(hours=i)
        
        training_data.append(seismic_data)
    
    print(f"Created {len(training_data)} training samples")
    
    # Initialize and fit processor
    print("\n2. Initializing and fitting SeismicProcessor...")
    config = {
        'bandpass_filter': {'freqmin': 1.0, 'freqmax': 45.0, 'corners': 4},
        'sta_length': 0.5,
        'lta_length': 10.0,
        'trigger_on': 3.0,
        'trigger_off': 1.5,
        'normalization': 'standard'
    }
    
    processor = SeismicProcessor(config)
    processor.fit(training_data)
    
    print("Processor fitted successfully!")
    print(f"Feature count: {processor._get_feature_count()}")
    
    # Get processing statistics
    stats = processor.get_processing_statistics()
    print(f"Signal statistics: {stats['signal_statistics']}")
    
    # Create test data with rockfall event
    print("\n3. Processing test data with rockfall event...")
    test_data_with_event = create_synthetic_seismic_data(
        duration=30.0,
        sampling_rate=100.0,
        add_event=True
    )
    test_data_with_event.station_id = "TEST.ROCKFALL.00.HHZ"
    
    # Extract features
    features_with_event = processor.transform(test_data_with_event)
    print(f"Extracted {len(features_with_event)} features from signal with rockfall")
    
    # Detect rockfall signatures
    detection_results = processor.detect_rockfall_signatures(test_data_with_event)
    
    print(f"Detection results:")
    print(f"  - Station: {detection_results['station_id']}")
    print(f"  - Number of triggers: {len(detection_results['triggers'])}")
    print(f"  - Anomaly score: {detection_results['anomaly_score']:.3f}")
    print(f"  - Is anomaly: {detection_results.get('is_anomaly', 'N/A')}")
    
    # Analyze signal characteristics
    print("\n4. Analyzing signal characteristics...")
    analysis = processor.analyze_signal_characteristics(test_data_with_event)
    
    print(f"Signal analysis:")
    print(f"  - Duration: {analysis['duration']:.1f} seconds")
    print(f"  - Peak amplitude: {analysis['temporal_analysis']['peak_amplitude']:.3f}")
    print(f"  - Dominant frequency: {analysis['spectral_analysis']['dominant_frequency']:.1f} Hz")
    print(f"  - SNR estimate: {analysis['signal_quality']['snr_estimate']:.1f} dB")
    
    # Create test data without rockfall event
    print("\n5. Processing test data without rockfall event...")
    test_data_no_event = create_synthetic_seismic_data(
        duration=30.0,
        sampling_rate=100.0,
        add_event=False
    )
    test_data_no_event.station_id = "TEST.NORMAL.00.HHZ"
    
    features_no_event = processor.transform(test_data_no_event)
    detection_results_no_event = processor.detect_rockfall_signatures(test_data_no_event)
    
    print(f"Normal signal detection results:")
    print(f"  - Number of triggers: {len(detection_results_no_event['triggers'])}")
    print(f"  - Anomaly score: {detection_results_no_event['anomaly_score']:.3f}")
    
    # Compare features
    print("\n6. Comparing features between signals...")
    print(f"Feature differences (with event - without event):")
    feature_diff = features_with_event - features_no_event
    significant_diffs = np.where(np.abs(feature_diff) > 1.0)[0]
    
    if len(significant_diffs) > 0:
        print(f"  - {len(significant_diffs)} features show significant differences")
        print(f"  - Largest difference: {np.max(np.abs(feature_diff)):.3f}")
    else:
        print("  - No significant feature differences detected")
    
    # Add a rockfall template
    print("\n7. Adding rockfall template for pattern recognition...")
    # Extract a portion of the signal with rockfall as template
    processed_signal = detection_results['processed_signal']
    if processed_signal is not None and len(processed_signal) > 500:
        # Use middle portion as template (where rockfall event likely occurred)
        template_start = len(processed_signal) // 3
        template_end = template_start + 500
        template = processed_signal[template_start:template_end]
        
        template_id = processor.add_rockfall_template(template)
        print(f"Added rockfall template with ID: {template_id}")
        
        # Test template matching on new data
        new_test_data = create_synthetic_seismic_data(duration=25.0, add_event=True)
        new_detection_results = processor.detect_rockfall_signatures(new_test_data)
        
        print(f"Template matching results:")
        print(f"  - Number of detections: {len(new_detection_results['detections'])}")
        
        for i, detection in enumerate(new_detection_results['detections']):
            print(f"  - Detection {i+1}: correlation={detection['correlation']:.3f}, "
                  f"confidence={detection['confidence']:.3f}")
    
    print("\n8. Processing statistics summary...")
    final_stats = processor.get_processing_statistics()
    print(f"Final processor statistics:")
    print(f"  - Status: {final_stats['status']}")
    print(f"  - Number of templates: {final_stats['num_templates']}")
    print(f"  - Feature scaling method: {final_stats['feature_scaling']['method']}")
    print(f"  - Total feature count: {final_stats['feature_scaling']['feature_count']}")
    
    print("\nSeismic processing example completed successfully!")


if __name__ == "__main__":
    main()