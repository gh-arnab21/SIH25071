"""Seismic data processing pipeline for rockfall detection."""

import warnings
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
from datetime import datetime, timedelta
from scipy import signal, stats
from scipy.fft import fft, fftfreq
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.ensemble import IsolationForest
from sklearn.cluster import DBSCAN

try:
    from obspy import read, Stream, Trace
    from obspy.signal import filter as obspy_filter
    from obspy.signal.trigger import classic_sta_lta, trigger_onset
    from obspy.signal.cross_correlation import correlate, xcorr_max
    OBSPY_AVAILABLE = True
except ImportError:
    warnings.warn("ObsPy not available. Some seismic processing features will be limited.")
    OBSPY_AVAILABLE = False

from ..base import BaseDataProcessor
from ..schemas import SeismicData


class SeismicProcessor(BaseDataProcessor):
    """Processor for seismic signal analysis and rockfall pattern recognition."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize the seismic data processor.
        
        Args:
            config: Configuration dictionary with processing parameters
        """
        super().__init__(config)
        
        # Default configuration
        default_config = {
            # Signal preprocessing
            'bandpass_filter': {'freqmin': 1.0, 'freqmax': 50.0, 'corners': 4},
            'detrend': True,
            'taper_percentage': 0.05,
            
            # STA/LTA trigger detection
            'sta_length': 0.5,  # seconds
            'lta_length': 10.0,  # seconds
            'trigger_on': 3.5,
            'trigger_off': 1.5,
            
            # Spectral analysis
            'window_length': 2.0,  # seconds for spectral windows
            'overlap': 0.5,  # 50% overlap
            'nfft': 1024,
            
            # Feature extraction
            'frequency_bands': {
                'low': (1, 5),      # Low frequency band (Hz)
                'mid': (5, 20),     # Mid frequency band (Hz)
                'high': (20, 50)    # High frequency band (Hz)
            },
            
            # Pattern recognition
            'template_length': 5.0,  # seconds
            'correlation_threshold': 0.7,
            'anomaly_contamination': 0.1,
            
            # Normalization
            'normalization': 'standard',  # 'standard', 'minmax', or 'robust'
        }
        
        self.config = {**default_config, **self.config}
        
        # Initialize scalers and detectors
        self.scaler = None
        self.anomaly_detector = None
        
        # Rockfall signature templates
        self.rockfall_templates = []
        
        # Statistics for fitted data
        self.signal_stats = {}
        
    def fit(self, data: List[SeismicData]) -> 'SeismicProcessor':
        """Fit the processor to training seismic data."""
        all_features = []
        all_signals = []
        
        for seismic_data in data:
            try:
                # Preprocess signal
                processed_signal = self._preprocess_signal(seismic_data)
                if processed_signal is not None:
                    all_signals.append(processed_signal)
                    
                    # Extract features for fitting
                    features = self._extract_signal_features(processed_signal, seismic_data.sampling_rate)
                    all_features.append(features)
                    
            except Exception as e:
                warnings.warn(f"Failed to process seismic data during fitting: {str(e)}")
                continue
        
        if not all_features:
            warnings.warn("No valid seismic data found for fitting")
            self._is_fitted = True
            return self
        
        # Fit scaler
        features_array = np.array(all_features)
        self._fit_scaler(features_array)
        
        # Fit anomaly detector
        self._fit_anomaly_detector(features_array)
        
        # Calculate signal statistics
        self._calculate_signal_statistics(all_signals)
        
        self._is_fitted = True
        return self
    
    def _fit_scaler(self, features: np.ndarray):
        """Fit the feature scaler."""
        if self.config['normalization'] == 'standard':
            self.scaler = StandardScaler()
        elif self.config['normalization'] == 'minmax':
            self.scaler = MinMaxScaler()
        else:  # robust
            from sklearn.preprocessing import RobustScaler
            self.scaler = RobustScaler()
        
        self.scaler.fit(features)
    
    def _fit_anomaly_detector(self, features: np.ndarray):
        """Fit the anomaly detector."""
        self.anomaly_detector = IsolationForest(
            contamination=self.config['anomaly_contamination'],
            random_state=42
        )
        self.anomaly_detector.fit(features)
    
    def _calculate_signal_statistics(self, signals: List[np.ndarray]):
        """Calculate statistics for the fitted signals."""
        if not signals:
            return
        
        all_amplitudes = np.concatenate([np.abs(sig) for sig in signals])
        
        self.signal_stats = {
            'mean_amplitude': np.mean(all_amplitudes),
            'std_amplitude': np.std(all_amplitudes),
            'max_amplitude': np.max(all_amplitudes),
            'min_amplitude': np.min(all_amplitudes),
            'median_amplitude': np.median(all_amplitudes),
            'num_signals': len(signals),
            'total_samples': len(all_amplitudes)
        }
    
    def transform(self, data: Union[SeismicData, List[SeismicData]]) -> np.ndarray:
        """Transform seismic data to processed features."""
        if not self._is_fitted:
            raise ValueError("Processor must be fitted before transform")
        
        if isinstance(data, list):
            return np.array([self._process_single_seismic_data(seismic_data) for seismic_data in data])
        else:
            return self._process_single_seismic_data(data)
    
    def _process_single_seismic_data(self, seismic_data: SeismicData) -> np.ndarray:
        """Process a single SeismicData object."""
        try:
            # Preprocess signal
            processed_signal = self._preprocess_signal(seismic_data)
            if processed_signal is None:
                return np.zeros(self._get_feature_count())
            
            # Extract features
            features = self._extract_signal_features(processed_signal, seismic_data.sampling_rate)
            
            # Normalize features
            if self.scaler is not None:
                features = self.scaler.transform(features.reshape(1, -1)).flatten()
            
            return features
            
        except Exception as e:
            warnings.warn(f"Failed to process seismic data: {str(e)}")
            return np.zeros(self._get_feature_count())
    
    def _preprocess_signal(self, seismic_data: SeismicData) -> Optional[np.ndarray]:
        """Preprocess seismic signal with filtering and detrending."""
        if seismic_data.signal is None or len(seismic_data.signal) == 0:
            return None
        
        try:
            signal_data = seismic_data.signal.copy()
            
            # Remove DC component and detrend
            if self.config['detrend']:
                signal_data = signal.detrend(signal_data)
            
            # Apply taper to reduce edge effects
            if self.config['taper_percentage'] > 0:
                taper_samples = int(len(signal_data) * self.config['taper_percentage'])
                taper = signal.windows.tukey(len(signal_data), alpha=self.config['taper_percentage'])
                signal_data = signal_data * taper
            
            # Apply bandpass filter
            if OBSPY_AVAILABLE:
                # Use ObsPy filtering if available (more robust for seismic data)
                try:
                    # Create ObsPy trace for filtering
                    trace = Trace(data=signal_data)
                    trace.stats.sampling_rate = seismic_data.sampling_rate
                    
                    # Apply bandpass filter
                    trace.filter('bandpass', 
                               freqmin=self.config['bandpass_filter']['freqmin'],
                               freqmax=self.config['bandpass_filter']['freqmax'],
                               corners=self.config['bandpass_filter']['corners'])
                    
                    signal_data = trace.data
                    
                except Exception as e:
                    warnings.warn(f"ObsPy filtering failed, using scipy: {str(e)}")
                    signal_data = self._apply_scipy_filter(signal_data, seismic_data.sampling_rate)
            else:
                signal_data = self._apply_scipy_filter(signal_data, seismic_data.sampling_rate)
            
            return signal_data
            
        except Exception as e:
            warnings.warn(f"Failed to preprocess seismic signal: {str(e)}")
            return None
    
    def _apply_scipy_filter(self, signal_data: np.ndarray, sampling_rate: float) -> np.ndarray:
        """Apply bandpass filter using scipy."""
        nyquist = sampling_rate / 2
        low = self.config['bandpass_filter']['freqmin'] / nyquist
        high = self.config['bandpass_filter']['freqmax'] / nyquist
        
        # Ensure filter frequencies are valid
        low = max(low, 0.01)  # Avoid very low frequencies
        high = min(high, 0.99)  # Avoid Nyquist frequency
        
        if low >= high:
            warnings.warn("Invalid filter frequencies, skipping filtering")
            return signal_data
        
        try:
            sos = signal.butter(self.config['bandpass_filter']['corners'], 
                              [low, high], btype='band', output='sos')
            filtered_signal = signal.sosfilt(sos, signal_data)
            return filtered_signal
        except Exception as e:
            warnings.warn(f"Scipy filtering failed: {str(e)}")
            return signal_data
    
    def _extract_signal_features(self, signal_data: np.ndarray, sampling_rate: float) -> np.ndarray:
        """Extract comprehensive features from seismic signal."""
        features = []
        
        # Time domain features
        features.extend(self._extract_time_domain_features(signal_data))
        
        # Frequency domain features
        features.extend(self._extract_frequency_domain_features(signal_data, sampling_rate))
        
        # Statistical features
        features.extend(self._extract_statistical_features(signal_data))
        
        # STA/LTA trigger features
        features.extend(self._extract_trigger_features(signal_data, sampling_rate))
        
        # Spectral features
        features.extend(self._extract_spectral_features(signal_data, sampling_rate))
        
        return np.array(features)
    
    def _extract_time_domain_features(self, signal_data: np.ndarray) -> List[float]:
        """Extract time domain features."""
        if len(signal_data) == 0:
            return [0.0] * 8
        
        features = [
            np.max(np.abs(signal_data)),  # Peak amplitude
            np.mean(np.abs(signal_data)),  # Mean absolute amplitude
            np.std(signal_data),  # Standard deviation
            np.sqrt(np.mean(signal_data**2)),  # RMS amplitude
            len(signal_data),  # Signal length
            np.sum(np.abs(np.diff(signal_data))),  # Total variation
            np.max(np.abs(np.diff(signal_data))),  # Maximum gradient
            np.mean(np.abs(np.diff(signal_data)))  # Mean gradient
        ]
        
        return features    

    def _extract_frequency_domain_features(self, signal_data: np.ndarray, sampling_rate: float) -> List[float]:
        """Extract frequency domain features."""
        if len(signal_data) < 8:
            return [0.0] * 9
        
        try:
            # Compute FFT
            fft_data = fft(signal_data)
            freqs = fftfreq(len(signal_data), 1/sampling_rate)
            
            # Take positive frequencies only
            positive_freqs = freqs[:len(freqs)//2]
            magnitude = np.abs(fft_data[:len(fft_data)//2])
            
            if len(magnitude) == 0:
                return [0.0] * 9
            
            # Power spectral density
            psd = magnitude ** 2
            total_power = np.sum(psd)
            
            if total_power == 0:
                return [0.0] * 9
            
            # Dominant frequency
            dominant_freq = positive_freqs[np.argmax(magnitude)]
            
            # Spectral centroid
            spectral_centroid = np.sum(positive_freqs * psd) / total_power
            
            # Spectral rolloff (95% of energy)
            cumulative_power = np.cumsum(psd)
            rolloff_idx = np.where(cumulative_power >= 0.95 * total_power)[0]
            spectral_rolloff = positive_freqs[rolloff_idx[0]] if len(rolloff_idx) > 0 else positive_freqs[-1]
            
            # Spectral bandwidth
            spectral_bandwidth = np.sqrt(np.sum(((positive_freqs - spectral_centroid) ** 2) * psd) / total_power)
            
            # Spectral flatness (measure of noise-like vs tonal characteristics)
            geometric_mean = stats.gmean(magnitude[magnitude > 0]) if np.any(magnitude > 0) else 0
            arithmetic_mean = np.mean(magnitude)
            spectral_flatness = geometric_mean / arithmetic_mean if arithmetic_mean > 0 else 0
            
            # Energy in frequency bands
            band_energies = []
            for band_name, (low_freq, high_freq) in self.config['frequency_bands'].items():
                band_mask = (positive_freqs >= low_freq) & (positive_freqs <= high_freq)
                band_energy = np.sum(psd[band_mask]) / total_power if total_power > 0 else 0
                band_energies.append(band_energy)
            
            features = [
                dominant_freq,
                spectral_centroid,
                spectral_rolloff,
                spectral_bandwidth,
                spectral_flatness,
                total_power
            ] + band_energies
            
            return features
            
        except Exception as e:
            warnings.warn(f"Failed to extract frequency domain features: {str(e)}")
            return [0.0] * 9
    
    def _extract_statistical_features(self, signal_data: np.ndarray) -> List[float]:
        """Extract statistical features from signal."""
        if len(signal_data) == 0:
            return [0.0] * 6
        
        features = [
            stats.skew(signal_data),  # Skewness
            stats.kurtosis(signal_data),  # Kurtosis
            np.percentile(signal_data, 25),  # 25th percentile
            np.percentile(signal_data, 75),  # 75th percentile
            np.median(signal_data),  # Median
            stats.iqr(signal_data)  # Interquartile range
        ]
        
        return features
    
    def _extract_trigger_features(self, signal_data: np.ndarray, sampling_rate: float) -> List[float]:
        """Extract STA/LTA trigger-based features."""
        if len(signal_data) < 100 or not OBSPY_AVAILABLE:
            return [0.0] * 5
        
        try:
            # Calculate STA/LTA
            sta_samples = int(self.config['sta_length'] * sampling_rate)
            lta_samples = int(self.config['lta_length'] * sampling_rate)
            
            # Ensure we have enough samples
            if len(signal_data) < lta_samples + sta_samples:
                return [0.0] * 5
            
            cft = classic_sta_lta(signal_data, sta_samples, lta_samples)
            
            # Find triggers
            triggers = trigger_onset(cft, self.config['trigger_on'], self.config['trigger_off'])
            
            # Calculate trigger features
            max_cft = np.max(cft) if len(cft) > 0 else 0
            mean_cft = np.mean(cft) if len(cft) > 0 else 0
            num_triggers = len(triggers)
            
            # Calculate trigger duration statistics
            if num_triggers > 0:
                trigger_durations = [(end - start) / sampling_rate for start, end in triggers]
                mean_trigger_duration = np.mean(trigger_durations)
                max_trigger_duration = np.max(trigger_durations)
            else:
                mean_trigger_duration = 0
                max_trigger_duration = 0
            
            features = [
                max_cft,
                mean_cft,
                num_triggers,
                mean_trigger_duration,
                max_trigger_duration
            ]
            
            return features
            
        except Exception as e:
            warnings.warn(f"Failed to extract trigger features: {str(e)}")
            return [0.0] * 5
    
    def _extract_spectral_features(self, signal_data: np.ndarray, sampling_rate: float) -> List[float]:
        """Extract spectral features using windowed analysis."""
        if len(signal_data) < 64:
            return [0.0] * 8
        
        try:
            # Calculate window parameters
            window_samples = int(self.config['window_length'] * sampling_rate)
            overlap_samples = int(window_samples * self.config['overlap'])
            
            # Ensure we have enough samples for at least one window
            if len(signal_data) < window_samples:
                window_samples = len(signal_data)
                overlap_samples = 0
            
            # Compute spectrogram
            freqs, times, Sxx = signal.spectrogram(
                signal_data, 
                fs=sampling_rate,
                window='hann',
                nperseg=window_samples,
                noverlap=overlap_samples,
                nfft=self.config['nfft']
            )
            
            if Sxx.size == 0:
                return [0.0] * 8
            
            # Spectral features across time
            mean_spectrum = np.mean(Sxx, axis=1)
            max_spectrum = np.max(Sxx, axis=1)
            
            # Spectral peak frequency
            peak_freq_idx = np.argmax(mean_spectrum)
            peak_frequency = freqs[peak_freq_idx] if len(freqs) > peak_freq_idx else 0
            
            # Spectral spread (variance in frequency domain)
            total_power = np.sum(mean_spectrum)
            if total_power > 0:
                spectral_mean = np.sum(freqs * mean_spectrum) / total_power
                spectral_spread = np.sqrt(np.sum(((freqs - spectral_mean) ** 2) * mean_spectrum) / total_power)
            else:
                spectral_spread = 0
            
            # Temporal features of spectrum
            spectrum_variation = np.std(Sxx, axis=1)
            mean_spectrum_variation = np.mean(spectrum_variation)
            
            # High frequency content
            high_freq_mask = freqs > sampling_rate / 4  # Above quarter Nyquist
            high_freq_power = np.sum(mean_spectrum[high_freq_mask]) / total_power if total_power > 0 else 0
            
            # Spectral entropy
            normalized_spectrum = mean_spectrum / total_power if total_power > 0 else mean_spectrum
            spectral_entropy = -np.sum(normalized_spectrum * np.log(normalized_spectrum + 1e-12))
            
            # Number of spectral peaks
            peaks, _ = signal.find_peaks(mean_spectrum, height=np.max(mean_spectrum) * 0.1)
            num_peaks = len(peaks)
            
            features = [
                peak_frequency,
                spectral_spread,
                mean_spectrum_variation,
                high_freq_power,
                spectral_entropy,
                num_peaks,
                np.max(Sxx),  # Maximum spectral power
                np.mean(Sxx)  # Mean spectral power
            ]
            
            return features
            
        except Exception as e:
            warnings.warn(f"Failed to extract spectral features: {str(e)}")
            return [0.0] * 8
    
    def _get_feature_count(self) -> int:
        """Get the total number of features extracted."""
        # Time domain: 8, Frequency domain: 9, Statistical: 6, Trigger: 5, Spectral: 8
        return 8 + 9 + 6 + 5 + 8
    
    def detect_rockfall_signatures(self, seismic_data: SeismicData) -> Dict[str, Any]:
        """Detect potential rockfall signatures in seismic data."""
        results = {
            'station_id': seismic_data.station_id,
            'start_time': seismic_data.start_time,
            'detections': [],
            'triggers': [],
            'anomaly_score': 0.0,
            'processed_signal': None
        }
        
        # Preprocess signal
        processed_signal = self._preprocess_signal(seismic_data)
        if processed_signal is None:
            return results
        
        results['processed_signal'] = processed_signal
        
        # Extract features and check for anomalies
        if self._is_fitted:
            features = self._extract_signal_features(processed_signal, seismic_data.sampling_rate)
            
            if self.anomaly_detector is not None:
                try:
                    anomaly_score = self.anomaly_detector.decision_function(features.reshape(1, -1))[0]
                    is_anomaly = self.anomaly_detector.predict(features.reshape(1, -1))[0] == -1
                    
                    results['anomaly_score'] = anomaly_score
                    results['is_anomaly'] = is_anomaly
                except Exception as e:
                    warnings.warn(f"Anomaly detection failed: {str(e)}")
        
        # STA/LTA trigger detection
        if OBSPY_AVAILABLE and len(processed_signal) > 100:
            try:
                sta_samples = int(self.config['sta_length'] * seismic_data.sampling_rate)
                lta_samples = int(self.config['lta_length'] * seismic_data.sampling_rate)
                
                if len(processed_signal) >= lta_samples + sta_samples:
                    cft = classic_sta_lta(processed_signal, sta_samples, lta_samples)
                    triggers = trigger_onset(cft, self.config['trigger_on'], self.config['trigger_off'])
                    
                    for start_idx, end_idx in triggers:
                        trigger_start_time = seismic_data.start_time + timedelta(seconds=start_idx/seismic_data.sampling_rate)
                        trigger_end_time = seismic_data.start_time + timedelta(seconds=end_idx/seismic_data.sampling_rate)
                        duration = (end_idx - start_idx) / seismic_data.sampling_rate
                        
                        results['triggers'].append({
                            'start_time': trigger_start_time,
                            'end_time': trigger_end_time,
                            'duration': duration,
                            'start_index': start_idx,
                            'end_index': end_idx,
                            'max_cft': np.max(cft[start_idx:end_idx]) if end_idx > start_idx else 0
                        })
            except Exception as e:
                warnings.warn(f"STA/LTA trigger detection failed: {str(e)}")
        
        # Template matching for known rockfall signatures
        if self.rockfall_templates:
            correlations = self._template_matching(processed_signal, seismic_data.sampling_rate)
            
            for correlation in correlations:
                if correlation['correlation'] >= self.config['correlation_threshold']:
                    detection_time = seismic_data.start_time + timedelta(seconds=correlation['time_offset'])
                    
                    results['detections'].append({
                        'detection_time': detection_time,
                        'correlation': correlation['correlation'],
                        'template_id': correlation['template_id'],
                        'confidence': correlation['correlation']
                    })
        
        return results
    
    def _template_matching(self, signal_data: np.ndarray, sampling_rate: float) -> List[Dict[str, Any]]:
        """Perform template matching for rockfall signature detection."""
        correlations = []
        
        for i, template in enumerate(self.rockfall_templates):
            try:
                # Cross-correlate signal with template
                correlation = np.correlate(signal_data, template, mode='valid')
                
                # Normalize correlation
                template_energy = np.sum(template ** 2)
                signal_energy = np.array([np.sum(signal_data[j:j+len(template)] ** 2) 
                                        for j in range(len(correlation))])
                
                # Avoid division by zero
                valid_indices = signal_energy > 0
                normalized_correlation = np.zeros_like(correlation)
                normalized_correlation[valid_indices] = (correlation[valid_indices] / 
                                                       np.sqrt(template_energy * signal_energy[valid_indices]))
                
                # Find peaks in correlation
                peaks, properties = signal.find_peaks(normalized_correlation, 
                                                    height=self.config['correlation_threshold'])
                
                for peak_idx in peaks:
                    time_offset = peak_idx / sampling_rate
                    correlation_value = normalized_correlation[peak_idx]
                    
                    correlations.append({
                        'template_id': i,
                        'time_offset': time_offset,
                        'correlation': correlation_value,
                        'peak_index': peak_idx
                    })
                    
            except Exception as e:
                warnings.warn(f"Template matching failed for template {i}: {str(e)}")
                continue
        
        # Sort by correlation strength
        correlations.sort(key=lambda x: x['correlation'], reverse=True)
        
        return correlations
    
    def add_rockfall_template(self, template_signal: np.ndarray, template_id: Optional[str] = None) -> int:
        """Add a rockfall signature template for pattern recognition."""
        if len(template_signal) == 0:
            raise ValueError("Template signal cannot be empty")
        
        # Normalize template
        normalized_template = template_signal / np.linalg.norm(template_signal)
        
        self.rockfall_templates.append(normalized_template)
        template_index = len(self.rockfall_templates) - 1
        
        return template_index
    
    def read_sac_file(self, file_path: str) -> Optional[SeismicData]:
        """Read SAC seismic file and convert to SeismicData format."""
        if not OBSPY_AVAILABLE:
            raise ImportError("ObsPy is required to read SAC files")
        
        try:
            # Read SAC file using ObsPy
            stream = read(file_path)
            
            if len(stream) == 0:
                warnings.warn(f"No data found in SAC file: {file_path}")
                return None
            
            # Use first trace (SAC files typically contain one trace)
            trace = stream[0]
            
            # Extract metadata
            stats = trace.stats
            sampling_rate = stats.sampling_rate
            start_time = stats.starttime.datetime
            station_id = f"{stats.network}.{stats.station}.{stats.location}.{stats.channel}"
            
            # Get signal data
            signal_data = trace.data
            
            return SeismicData(
                signal=signal_data,
                sampling_rate=sampling_rate,
                start_time=start_time,
                station_id=station_id
            )
            
        except Exception as e:
            warnings.warn(f"Failed to read SAC file {file_path}: {str(e)}")
            return None
    
    def process_seismic_stream(self, file_paths: List[str]) -> List[Dict[str, Any]]:
        """Process multiple SAC files and detect rockfall signatures."""
        results = []
        
        for file_path in file_paths:
            try:
                # Read SAC file
                seismic_data = self.read_sac_file(file_path)
                if seismic_data is None:
                    continue
                
                # Detect rockfall signatures
                detection_results = self.detect_rockfall_signatures(seismic_data)
                detection_results['file_path'] = file_path
                
                results.append(detection_results)
                
            except Exception as e:
                warnings.warn(f"Failed to process SAC file {file_path}: {str(e)}")
                continue
        
        return results
    
    def analyze_signal_characteristics(self, seismic_data: SeismicData) -> Dict[str, Any]:
        """Analyze comprehensive characteristics of seismic signal."""
        analysis = {
            'station_id': seismic_data.station_id,
            'start_time': seismic_data.start_time,
            'sampling_rate': seismic_data.sampling_rate,
            'duration': len(seismic_data.signal) / seismic_data.sampling_rate,
            'signal_quality': {},
            'spectral_analysis': {},
            'temporal_analysis': {},
            'noise_analysis': {}
        }
        
        # Preprocess signal
        processed_signal = self._preprocess_signal(seismic_data)
        if processed_signal is None:
            return analysis
        
        # Signal quality metrics
        analysis['signal_quality'] = {
            'snr_estimate': self._estimate_snr(processed_signal),
            'clipping_ratio': np.sum(np.abs(seismic_data.signal) >= 0.99 * np.max(np.abs(seismic_data.signal))) / len(seismic_data.signal),
            'zero_crossings': len(np.where(np.diff(np.signbit(processed_signal)))[0]),
            'dynamic_range': 20 * np.log10(np.max(np.abs(processed_signal)) / (np.mean(np.abs(processed_signal)) + 1e-12))
        }
        
        # Spectral analysis
        freqs, psd = signal.welch(processed_signal, fs=seismic_data.sampling_rate, nperseg=min(len(processed_signal)//4, 1024))
        
        analysis['spectral_analysis'] = {
            'dominant_frequency': freqs[np.argmax(psd)],
            'bandwidth': self._calculate_bandwidth(freqs, psd),
            'spectral_centroid': np.sum(freqs * psd) / np.sum(psd),
            'total_power': np.sum(psd)
        }
        
        # Temporal analysis
        envelope = np.abs(signal.hilbert(processed_signal))
        
        analysis['temporal_analysis'] = {
            'peak_amplitude': np.max(np.abs(processed_signal)),
            'rms_amplitude': np.sqrt(np.mean(processed_signal**2)),
            'envelope_max': np.max(envelope),
            'envelope_mean': np.mean(envelope),
            'crest_factor': np.max(np.abs(processed_signal)) / np.sqrt(np.mean(processed_signal**2))
        }
        
        # Noise analysis
        analysis['noise_analysis'] = {
            'noise_floor': np.percentile(np.abs(processed_signal), 10),
            'signal_variability': np.std(processed_signal) / np.mean(np.abs(processed_signal)),
            'high_freq_noise': self._estimate_high_freq_noise(processed_signal, seismic_data.sampling_rate)
        }
        
        return analysis
    
    def _estimate_snr(self, signal_data: np.ndarray) -> float:
        """Estimate signal-to-noise ratio."""
        if len(signal_data) < 100:
            return 0.0
        
        # Simple SNR estimation: ratio of signal power to noise power
        # Assume first and last 10% of signal represent noise
        noise_samples = int(0.1 * len(signal_data))
        noise_power = np.mean(signal_data[:noise_samples]**2) + np.mean(signal_data[-noise_samples:]**2)
        noise_power /= 2
        
        signal_power = np.mean(signal_data**2)
        
        if noise_power > 0:
            snr = 10 * np.log10(signal_power / noise_power)
        else:
            snr = float('inf')
        
        return snr
    
    def _calculate_bandwidth(self, freqs: np.ndarray, psd: np.ndarray) -> float:
        """Calculate -3dB bandwidth of the signal."""
        if len(psd) == 0:
            return 0.0
        
        max_power = np.max(psd)
        half_power = max_power / 2
        
        # Find frequencies where power is above half maximum
        above_half_power = psd >= half_power
        
        if not np.any(above_half_power):
            return 0.0
        
        freq_indices = np.where(above_half_power)[0]
        bandwidth = freqs[freq_indices[-1]] - freqs[freq_indices[0]]
        
        return bandwidth
    
    def _estimate_high_freq_noise(self, signal_data: np.ndarray, sampling_rate: float) -> float:
        """Estimate high frequency noise content."""
        if len(signal_data) < 64:
            return 0.0
        
        try:
            # Apply high-pass filter to isolate high frequency content
            nyquist = sampling_rate / 2
            high_cutoff = 0.8  # 80% of Nyquist frequency
            
            sos = signal.butter(4, high_cutoff, btype='high', output='sos')
            high_freq_signal = signal.sosfilt(sos, signal_data)
            
            # Calculate power ratio
            total_power = np.mean(signal_data**2)
            high_freq_power = np.mean(high_freq_signal**2)
            
            noise_ratio = high_freq_power / total_power if total_power > 0 else 0
            
            return noise_ratio
            
        except Exception:
            return 0.0
    
    def get_processing_statistics(self) -> Dict[str, Any]:
        """Get statistics about the fitted processor."""
        if not self._is_fitted:
            return {'status': 'not_fitted'}
        
        stats = {
            'status': 'fitted',
            'signal_statistics': self.signal_stats,
            'num_templates': len(self.rockfall_templates),
            'configuration': self.config.copy()
        }
        
        if self.scaler is not None:
            stats['feature_scaling'] = {
                'method': self.config['normalization'],
                'feature_count': self._get_feature_count()
            }
        
        return stats