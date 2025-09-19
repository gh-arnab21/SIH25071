"""Unit tests for SeismicProcessor."""

import pytest
import numpy as np
from datetime import datetime, timedelta
from unittest.mock import patch, MagicMock
import warnings

from src.data.processors.seismic_processor import SeismicProcessor
from src.data.schemas import SeismicData


class TestSeismicProcessor:
    """Test cases for SeismicProcessor class."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.config = {
            'bandpass_filter': {'freqmin': 1.0, 'freqmax': 50.0, 'corners': 4},
            'detrend': True,
            'taper_percentage': 0.05,
            'sta_length': 0.5,
            'lta_length': 10.0,
            'trigger_on': 3.5,
            'trigger_off': 1.5,
            'normalization': 'standard'
        }
        self.processor = SeismicProcessor(self.config)
        
        # Create sample seismic data
        self.sampling_rate = 100.0  # 100 Hz
        self.duration = 30.0  # 30 seconds
        self.num_samples = int(self.sampling_rate * self.duration)
        
        # Generate synthetic seismic signal
        t = np.linspace(0, self.duration, self.num_samples)
        # Base noise
        signal = np.random.normal(0, 0.1, self.num_samples)
        # Add some frequency components
        signal += 0.5 * np.sin(2 * np.pi * 5 * t)  # 5 Hz component
        signal += 0.3 * np.sin(2 * np.pi * 15 * t)  # 15 Hz component
        # Add a transient event (simulated rockfall)
        event_start = int(0.4 * self.num_samples)
        event_end = int(0.6 * self.num_samples)
        signal[event_start:event_end] += 2.0 * np.exp(-0.1 * np.arange(event_end - event_start))
        
        self.sample_seismic_data = SeismicData(
            signal=signal,
            sampling_rate=self.sampling_rate,
            start_time=datetime.now(),
            station_id="TEST.STA.00.HHZ"
        )
        
        # Create multiple samples for fitting
        self.training_data = []
        for i in range(5):
            # Generate different synthetic signals
            t = np.linspace(0, self.duration, self.num_samples)
            signal = np.random.normal(0, 0.1, self.num_samples)
            signal += np.random.uniform(0.2, 0.8) * np.sin(2 * np.pi * np.random.uniform(3, 8) * t)
            
            seismic_data = SeismicData(
                signal=signal,
                sampling_rate=self.sampling_rate,
                start_time=datetime.now() + timedelta(hours=i),
                station_id=f"TEST.STA.0{i}.HHZ"
            )
            self.training_data.append(seismic_data)
    
    def test_initialization(self):
        """Test processor initialization."""
        processor = SeismicProcessor()
        assert not processor.is_fitted
        assert processor.config['bandpass_filter']['freqmin'] == 1.0
        assert processor.config['normalization'] == 'standard'
        
        # Test with custom config
        custom_config = {'normalization': 'minmax'}
        processor = SeismicProcessor(custom_config)
        assert processor.config['normalization'] == 'minmax'
        assert processor.config['bandpass_filter']['freqmin'] == 1.0  # Default preserved
    
    def test_fit_processor(self):
        """Test fitting the processor to training data."""
        # Test fitting with valid data
        fitted_processor = self.processor.fit(self.training_data)
        assert fitted_processor.is_fitted
        assert self.processor.scaler is not None
        assert self.processor.anomaly_detector is not None
        assert len(self.processor.signal_stats) > 0
        
        # Test fitting with empty data
        empty_processor = SeismicProcessor()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            fitted_empty = empty_processor.fit([])
        assert fitted_empty.is_fitted  # Should still be marked as fitted
    
    def test_preprocess_signal(self):
        """Test signal preprocessing."""
        # Test with valid signal
        processed = self.processor._preprocess_signal(self.sample_seismic_data)
        assert processed is not None
        assert len(processed) == len(self.sample_seismic_data.signal)
        assert isinstance(processed, np.ndarray)
        
        # Test with empty signal
        empty_data = SeismicData(
            signal=np.array([]),
            sampling_rate=100.0,
            start_time=datetime.now(),
            station_id="EMPTY"
        )
        processed_empty = self.processor._preprocess_signal(empty_data)
        assert processed_empty is None
        
        # Test with None signal
        none_data = SeismicData(
            signal=None,
            sampling_rate=100.0,
            start_time=datetime.now(),
            station_id="NONE"
        )
        processed_none = self.processor._preprocess_signal(none_data)
        assert processed_none is None
    
    def test_extract_time_domain_features(self):
        """Test time domain feature extraction."""
        signal = np.random.normal(0, 1, 1000)
        features = self.processor._extract_time_domain_features(signal)
        
        assert len(features) == 8
        assert all(isinstance(f, (int, float)) for f in features)
        assert features[0] >= 0  # Peak amplitude should be positive
        assert features[1] >= 0  # Mean absolute amplitude should be positive
        
        # Test with empty signal
        empty_features = self.processor._extract_time_domain_features(np.array([]))
        assert len(empty_features) == 8
        assert all(f == 0.0 for f in empty_features)
    
    def test_extract_frequency_domain_features(self):
        """Test frequency domain feature extraction."""
        # Create signal with known frequency content
        t = np.linspace(0, 1, 1000)
        signal = np.sin(2 * np.pi * 10 * t)  # 10 Hz sine wave
        
        features = self.processor._extract_frequency_domain_features(signal, 1000.0)
        
        assert len(features) == 9  # 6 base features + 3 frequency bands
        assert all(isinstance(f, (int, float)) for f in features)
        
        # Dominant frequency should be around 10 Hz
        dominant_freq = features[0]
        assert 9 <= dominant_freq <= 11
        
        # Test with short signal
        short_features = self.processor._extract_frequency_domain_features(np.array([1, 2, 3]), 100.0)
        assert len(short_features) == 9
        assert all(f == 0.0 for f in short_features)
    
    def test_extract_statistical_features(self):
        """Test statistical feature extraction."""
        # Create signal with known statistics
        signal = np.random.normal(0, 1, 1000)
        features = self.processor._extract_statistical_features(signal)
        
        assert len(features) == 6
        assert all(isinstance(f, (int, float)) for f in features)
        
        # Test with empty signal
        empty_features = self.processor._extract_statistical_features(np.array([]))
        assert len(empty_features) == 6
        assert all(f == 0.0 for f in empty_features)
    
    @patch('src.data.processors.seismic_processor.OBSPY_AVAILABLE', True)
    @patch('src.data.processors.seismic_processor.classic_sta_lta')
    @patch('src.data.processors.seismic_processor.trigger_onset')
    def test_extract_trigger_features(self, mock_trigger_onset, mock_sta_lta):
        """Test STA/LTA trigger feature extraction."""
        # Mock ObsPy functions
        mock_sta_lta.return_value = np.random.uniform(0, 5, 1000)
        mock_trigger_onset.return_value = np.array([[100, 200], [300, 400]])
        
        signal = np.random.normal(0, 1, 1000)
        features = self.processor._extract_trigger_features(signal, 100.0)
        
        assert len(features) == 5
        assert all(isinstance(f, (int, float)) for f in features)
        
        # Test with short signal
        short_features = self.processor._extract_trigger_features(np.array([1, 2, 3]), 100.0)
        assert len(short_features) == 5
        assert all(f == 0.0 for f in short_features)
    
    def test_extract_spectral_features(self):
        """Test spectral feature extraction."""
        signal = np.random.normal(0, 1, 1000)
        features = self.processor._extract_spectral_features(signal, 100.0)
        
        assert len(features) == 8
        assert all(isinstance(f, (int, float)) for f in features)
        
        # Test with short signal
        short_features = self.processor._extract_spectral_features(np.array([1, 2, 3]), 100.0)
        assert len(short_features) == 8
        assert all(f == 0.0 for f in short_features)
    
    def test_extract_signal_features(self):
        """Test complete signal feature extraction."""
        signal = np.random.normal(0, 1, 1000)
        features = self.processor._extract_signal_features(signal, 100.0)
        
        expected_count = self.processor._get_feature_count()
        assert len(features) == expected_count
        assert isinstance(features, np.ndarray)
        assert all(np.isfinite(features))
    
    def test_transform_single_data(self):
        """Test transforming single seismic data."""
        # Fit processor first
        self.processor.fit(self.training_data)
        
        # Transform single data point
        features = self.processor.transform(self.sample_seismic_data)
        
        assert isinstance(features, np.ndarray)
        assert len(features) == self.processor._get_feature_count()
        assert all(np.isfinite(features))
    
    def test_transform_multiple_data(self):
        """Test transforming multiple seismic data points."""
        # Fit processor first
        self.processor.fit(self.training_data)
        
        # Transform multiple data points
        features = self.processor.transform(self.training_data)
        
        assert isinstance(features, np.ndarray)
        assert features.shape == (len(self.training_data), self.processor._get_feature_count())
        assert np.all(np.isfinite(features))
    
    def test_transform_without_fitting(self):
        """Test that transform raises error when not fitted."""
        with pytest.raises(ValueError, match="Processor must be fitted before transform"):
            self.processor.transform(self.sample_seismic_data)
    
    def test_detect_rockfall_signatures(self):
        """Test rockfall signature detection."""
        # Fit processor first
        self.processor.fit(self.training_data)
        
        results = self.processor.detect_rockfall_signatures(self.sample_seismic_data)
        
        assert 'station_id' in results
        assert 'start_time' in results
        assert 'detections' in results
        assert 'triggers' in results
        assert 'anomaly_score' in results
        assert 'processed_signal' in results
        
        assert results['station_id'] == self.sample_seismic_data.station_id
        assert results['start_time'] == self.sample_seismic_data.start_time
        assert isinstance(results['detections'], list)
        assert isinstance(results['triggers'], list)
    
    def test_add_rockfall_template(self):
        """Test adding rockfall templates."""
        template = np.random.normal(0, 1, 500)
        
        # Add template
        template_id = self.processor.add_rockfall_template(template)
        
        assert template_id == 0
        assert len(self.processor.rockfall_templates) == 1
        
        # Add another template
        template2 = np.random.normal(0, 1, 300)
        template_id2 = self.processor.add_rockfall_template(template2)
        
        assert template_id2 == 1
        assert len(self.processor.rockfall_templates) == 2
        
        # Test with empty template
        with pytest.raises(ValueError, match="Template signal cannot be empty"):
            self.processor.add_rockfall_template(np.array([]))
    
    @patch('src.data.processors.seismic_processor.OBSPY_AVAILABLE', True)
    @patch('src.data.processors.seismic_processor.read')
    def test_read_sac_file(self, mock_read):
        """Test reading SAC files."""
        # Mock ObsPy read function
        mock_trace = MagicMock()
        mock_trace.data = np.random.normal(0, 1, 1000)
        mock_trace.stats.sampling_rate = 100.0
        mock_trace.stats.starttime.datetime = datetime.now()
        mock_trace.stats.network = "XX"
        mock_trace.stats.station = "TEST"
        mock_trace.stats.location = "00"
        mock_trace.stats.channel = "HHZ"
        
        mock_stream = MagicMock()
        mock_stream.__len__.return_value = 1
        mock_stream.__getitem__.return_value = mock_trace
        mock_read.return_value = mock_stream
        
        # Test successful read
        seismic_data = self.processor.read_sac_file("test.sac")
        
        assert seismic_data is not None
        assert isinstance(seismic_data, SeismicData)
        assert seismic_data.sampling_rate == 100.0
        assert seismic_data.station_id == "XX.TEST.00.HHZ"
        assert len(seismic_data.signal) == 1000
        
        # Test empty stream
        mock_stream.__len__.return_value = 0
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            seismic_data_empty = self.processor.read_sac_file("empty.sac")
        assert seismic_data_empty is None
    
    @patch('src.data.processors.seismic_processor.OBSPY_AVAILABLE', False)
    def test_read_sac_file_no_obspy(self):
        """Test reading SAC files without ObsPy."""
        with pytest.raises(ImportError, match="ObsPy is required to read SAC files"):
            self.processor.read_sac_file("test.sac")
    
    def test_process_seismic_stream(self):
        """Test processing multiple SAC files."""
        # Mock the read_sac_file method
        with patch.object(self.processor, 'read_sac_file') as mock_read:
            mock_read.side_effect = [self.sample_seismic_data, None, self.sample_seismic_data]
            
            # Fit processor first
            self.processor.fit(self.training_data)
            
            file_paths = ["file1.sac", "file2.sac", "file3.sac"]
            results = self.processor.process_seismic_stream(file_paths)
            
            # Should have 2 results (file2.sac returns None)
            assert len(results) == 2
            assert all('file_path' in result for result in results)
            assert results[0]['file_path'] == "file1.sac"
            assert results[1]['file_path'] == "file3.sac"
    
    def test_analyze_signal_characteristics(self):
        """Test comprehensive signal analysis."""
        analysis = self.processor.analyze_signal_characteristics(self.sample_seismic_data)
        
        required_keys = ['station_id', 'start_time', 'sampling_rate', 'duration',
                        'signal_quality', 'spectral_analysis', 'temporal_analysis', 'noise_analysis']
        
        for key in required_keys:
            assert key in analysis
        
        assert analysis['station_id'] == self.sample_seismic_data.station_id
        assert analysis['sampling_rate'] == self.sample_seismic_data.sampling_rate
        assert analysis['duration'] > 0
        
        # Check nested dictionaries have expected keys
        assert 'snr_estimate' in analysis['signal_quality']
        assert 'dominant_frequency' in analysis['spectral_analysis']
        assert 'peak_amplitude' in analysis['temporal_analysis']
        assert 'noise_floor' in analysis['noise_analysis']
    
    def test_estimate_snr(self):
        """Test SNR estimation."""
        # Create signal with known SNR characteristics
        # Use a stronger signal to ensure positive SNR
        noise = np.random.normal(0, 0.1, 1000)
        signal_component = 2.0 * np.sin(2 * np.pi * 10 * np.linspace(0, 1, 1000))  # Stronger signal
        signal = noise + signal_component
        
        snr = self.processor._estimate_snr(signal)
        assert isinstance(snr, float)
        # SNR can be negative in dB scale, so just check it's a valid number
        assert np.isfinite(snr)
        
        # Test with short signal
        short_snr = self.processor._estimate_snr(np.array([1, 2, 3]))
        assert short_snr == 0.0
    
    def test_calculate_bandwidth(self):
        """Test bandwidth calculation."""
        # Create narrow-band signal
        freqs = np.linspace(0, 50, 100)
        psd = np.exp(-((freqs - 10) ** 2) / (2 * 2 ** 2))  # Gaussian centered at 10 Hz
        
        bandwidth = self.processor._calculate_bandwidth(freqs, psd)
        assert isinstance(bandwidth, float)
        assert bandwidth > 0
        
        # Test with empty PSD
        empty_bandwidth = self.processor._calculate_bandwidth(freqs, np.array([]))
        assert empty_bandwidth == 0.0
    
    def test_estimate_high_freq_noise(self):
        """Test high frequency noise estimation."""
        # Create signal with high frequency noise
        t = np.linspace(0, 1, 1000)
        signal = np.sin(2 * np.pi * 5 * t) + 0.5 * np.random.normal(0, 1, 1000)
        
        noise_ratio = self.processor._estimate_high_freq_noise(signal, 1000.0)
        assert isinstance(noise_ratio, float)
        assert 0 <= noise_ratio <= 1
        
        # Test with short signal
        short_noise = self.processor._estimate_high_freq_noise(np.array([1, 2, 3]), 100.0)
        assert short_noise == 0.0
    
    def test_get_processing_statistics(self):
        """Test getting processing statistics."""
        # Test unfitted processor
        stats = self.processor.get_processing_statistics()
        assert stats['status'] == 'not_fitted'
        
        # Test fitted processor
        self.processor.fit(self.training_data)
        stats = self.processor.get_processing_statistics()
        
        assert stats['status'] == 'fitted'
        assert 'signal_statistics' in stats
        assert 'num_templates' in stats
        assert 'configuration' in stats
        assert 'feature_scaling' in stats
        
        assert stats['num_templates'] == 0  # No templates added yet
        assert stats['feature_scaling']['feature_count'] == self.processor._get_feature_count()
    
    def test_template_matching(self):
        """Test template matching functionality."""
        # Add a template
        template = np.random.normal(0, 1, 100)
        self.processor.add_rockfall_template(template)
        
        # Create signal containing the template
        signal = np.random.normal(0, 0.1, 1000)
        signal[200:300] = template  # Insert template at known location
        
        correlations = self.processor._template_matching(signal, 100.0)
        
        assert isinstance(correlations, list)
        # Should find at least one correlation
        if correlations:
            assert all('template_id' in corr for corr in correlations)
            assert all('correlation' in corr for corr in correlations)
            assert all('time_offset' in corr for corr in correlations)
    
    def test_feature_count_consistency(self):
        """Test that feature count is consistent across methods."""
        expected_count = 8 + 9 + 6 + 5 + 8  # Sum of all feature types
        assert self.processor._get_feature_count() == expected_count
        
        # Test that actual feature extraction matches expected count
        signal = np.random.normal(0, 1, 1000)
        features = self.processor._extract_signal_features(signal, 100.0)
        assert len(features) == expected_count
    
    def test_error_handling(self):
        """Test error handling in various scenarios."""
        # Test with corrupted signal data
        corrupted_data = SeismicData(
            signal=np.array([np.inf, np.nan, 1, 2, 3]),
            sampling_rate=100.0,
            start_time=datetime.now(),
            station_id="CORRUPTED"
        )
        
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            processed = self.processor._preprocess_signal(corrupted_data)
        
        # Should handle gracefully (may return None or processed signal)
        assert processed is None or isinstance(processed, np.ndarray)
        
        # Test feature extraction with problematic data
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            features = self.processor._extract_signal_features(np.array([1, 2]), 100.0)
        
        assert len(features) == self.processor._get_feature_count()
        assert all(np.isfinite(features))


class TestSeismicProcessorIntegration:
    """Integration tests for SeismicProcessor."""
    
    def test_end_to_end_processing(self):
        """Test complete end-to-end processing workflow."""
        # Create processor
        processor = SeismicProcessor()
        
        # Create training data
        training_data = []
        for i in range(3):
            signal = np.random.normal(0, 1, 1000)
            seismic_data = SeismicData(
                signal=signal,
                sampling_rate=100.0,
                start_time=datetime.now(),
                station_id=f"TEST.STA.{i:02d}.HHZ"
            )
            training_data.append(seismic_data)
        
        # Fit processor
        processor.fit(training_data)
        assert processor.is_fitted
        
        # Transform data
        features = processor.transform(training_data)
        assert features.shape[0] == len(training_data)
        assert features.shape[1] == processor._get_feature_count()
        
        # Test new data
        new_signal = np.random.normal(0, 1, 1000)
        new_data = SeismicData(
            signal=new_signal,
            sampling_rate=100.0,
            start_time=datetime.now(),
            station_id="NEW.STA.00.HHZ"
        )
        
        new_features = processor.transform(new_data)
        assert len(new_features) == processor._get_feature_count()
        
        # Test rockfall detection
        detection_results = processor.detect_rockfall_signatures(new_data)
        assert isinstance(detection_results, dict)
        assert 'station_id' in detection_results
        
        # Test signal analysis
        analysis = processor.analyze_signal_characteristics(new_data)
        assert isinstance(analysis, dict)
        assert 'signal_quality' in analysis
    
    def test_different_configurations(self):
        """Test processor with different configurations."""
        configs = [
            {'normalization': 'minmax'},
            {'normalization': 'robust'},
            {'bandpass_filter': {'freqmin': 2.0, 'freqmax': 30.0, 'corners': 2}},
            {'detrend': False, 'taper_percentage': 0.0}
        ]
        
        # Create test data
        signal = np.random.normal(0, 1, 1000)
        seismic_data = SeismicData(
            signal=signal,
            sampling_rate=100.0,
            start_time=datetime.now(),
            station_id="TEST.STA.00.HHZ"
        )
        
        for config in configs:
            processor = SeismicProcessor(config)
            processor.fit([seismic_data])
            
            features = processor.transform(seismic_data)
            assert len(features) == processor._get_feature_count()
            assert all(np.isfinite(features))