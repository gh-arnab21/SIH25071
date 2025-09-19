"""Sensor data processing pipeline for time-series geotechnical data."""

import warnings
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
from scipy import signal, stats
from scipy.interpolate import interp1d
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.ensemble import IsolationForest
from sklearn.cluster import DBSCAN

from ..base import BaseDataProcessor
from ..schemas import SensorData, TimeSeries


class SensorDataProcessor(BaseDataProcessor):
    """Processor for time-series geotechnical sensor data including displacement, strain, and pore pressure."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize the sensor data processor.
        
        Args:
            config: Configuration dictionary with processing parameters
        """
        super().__init__(config)
        
        # Default configuration
        default_config = {
            'sampling_rate': 1.0,  # 1 Hz default
            'filter_type': 'lowpass',
            'filter_params': {'cutoff': 0.1, 'order': 4},
            'normalization': 'standard',
            'anomaly_method': 'isolation_forest',
            'anomaly_params': {'contamination': 0.1, 'random_state': 42},
            'trend_window': 24,  # 24 hour window for trend analysis
            'interpolation_method': 'linear',
            'max_gap_size': 10,  # Maximum gap size to interpolate
            'outlier_threshold': 3.0,  # Z-score threshold for outlier detection
        }
        
        self.config = {**default_config, **self.config}
        
        # Initialize scalers for each sensor type
        self.scalers = {
            'displacement': None,
            'strain': None,
            'pore_pressure': None
        }
        
        # Initialize anomaly detectors
        self.anomaly_detectors = {
            'displacement': None,
            'strain': None,
            'pore_pressure': None
        }
        
        # Statistics for fitted data
        self.data_stats = {}
    
    def fit(self, data: List[SensorData]) -> 'SensorDataProcessor':
        """Fit the processor to training data."""
        # Collect all sensor readings for fitting scalers and anomaly detectors
        all_displacement = []
        all_strain = []
        all_pore_pressure = []
        
        for sensor_data in data:
            if sensor_data.displacement is not None:
                processed_disp = self._preprocess_timeseries(sensor_data.displacement)
                if processed_disp is not None:
                    all_displacement.extend(processed_disp.values)
            
            if sensor_data.strain is not None:
                processed_strain = self._preprocess_timeseries(sensor_data.strain)
                if processed_strain is not None:
                    all_strain.extend(processed_strain.values)
            
            if sensor_data.pore_pressure is not None:
                processed_pp = self._preprocess_timeseries(sensor_data.pore_pressure)
                if processed_pp is not None:
                    all_pore_pressure.extend(processed_pp.values)
        
        # Fit scalers and anomaly detectors for each sensor type
        self._fit_sensor_processors('displacement', all_displacement)
        self._fit_sensor_processors('strain', all_strain)
        self._fit_sensor_processors('pore_pressure', all_pore_pressure)
        
        self._is_fitted = True
        return self
    
    def _fit_sensor_processors(self, sensor_type: str, values: List[float]):
        """Fit scalers and anomaly detectors for a specific sensor type."""
        if not values:
            return
        
        values_array = np.array(values).reshape(-1, 1)
        
        # Fit scaler
        if self.config['normalization'] == 'standard':
            scaler = StandardScaler()
        elif self.config['normalization'] == 'minmax':
            scaler = MinMaxScaler()
        else:  # robust
            from sklearn.preprocessing import RobustScaler
            scaler = RobustScaler()
        
        scaler.fit(values_array)
        self.scalers[sensor_type] = scaler
        
        # Fit anomaly detector
        if self.config['anomaly_method'] == 'isolation_forest':
            detector = IsolationForest(**self.config['anomaly_params'])
            detector.fit(values_array)
            self.anomaly_detectors[sensor_type] = detector
        
        # Store statistics
        self.data_stats[sensor_type] = {
            'mean': np.mean(values),
            'std': np.std(values),
            'min': np.min(values),
            'max': np.max(values),
            'median': np.median(values)
        }
    
    def transform(self, data: Union[SensorData, List[SensorData]]) -> np.ndarray:
        """Transform sensor data to processed features."""
        if not self._is_fitted:
            raise ValueError("Processor must be fitted before transform")
        
        if isinstance(data, list):
            return np.array([self._process_single_sensor_data(sensor_data) for sensor_data in data])
        else:
            return self._process_single_sensor_data(data)
    
    def _process_single_sensor_data(self, sensor_data: SensorData) -> np.ndarray:
        """Process a single SensorData object."""
        features = []
        
        # Process displacement data
        if sensor_data.displacement is not None:
            disp_features = self._extract_sensor_features(sensor_data.displacement, 'displacement')
            features.extend(disp_features)
        else:
            # Add zero features if displacement data is missing
            features.extend([0.0] * self._get_feature_count())
        
        # Process strain data
        if sensor_data.strain is not None:
            strain_features = self._extract_sensor_features(sensor_data.strain, 'strain')
            features.extend(strain_features)
        else:
            # Add zero features if strain data is missing
            features.extend([0.0] * self._get_feature_count())
        
        # Process pore pressure data
        if sensor_data.pore_pressure is not None:
            pp_features = self._extract_sensor_features(sensor_data.pore_pressure, 'pore_pressure')
            features.extend(pp_features)
        else:
            # Add zero features if pore pressure data is missing
            features.extend([0.0] * self._get_feature_count())
        
        return np.array(features)
    
    def _extract_sensor_features(self, timeseries: TimeSeries, sensor_type: str) -> List[float]:
        """Extract features from a time series."""
        # Preprocess the time series
        processed_ts = self._preprocess_timeseries(timeseries)
        if processed_ts is None or len(processed_ts.values) == 0:
            return [0.0] * self._get_feature_count()
        
        features = []
        values = processed_ts.values
        
        # Statistical features
        features.extend(self._extract_statistical_features(values))
        
        # Temporal features
        features.extend(self._extract_temporal_features(processed_ts))
        
        # Trend features
        features.extend(self._extract_trend_features(values))
        
        # Anomaly features
        features.extend(self._extract_anomaly_features(values, sensor_type))
        
        # Frequency domain features
        features.extend(self._extract_frequency_features(values, processed_ts.sampling_rate))
        
        return features
    
    def _preprocess_timeseries(self, timeseries: TimeSeries) -> Optional[TimeSeries]:
        """Preprocess a time series with filtering and interpolation."""
        if timeseries is None or len(timeseries.values) == 0:
            return None
        
        try:
            # Handle missing values
            clean_ts = self._handle_missing_values(timeseries)
            if clean_ts is None:
                return None
            
            # Apply filtering
            filtered_ts = self._apply_filter(clean_ts)
            
            # Resample if needed
            resampled_ts = self._resample_timeseries(filtered_ts)
            
            return resampled_ts
            
        except Exception as e:
            warnings.warn(f"Failed to preprocess time series: {str(e)}")
            return None
    
    def _handle_missing_values(self, timeseries: TimeSeries) -> Optional[TimeSeries]:
        """Handle missing values in time series data."""
        timestamps = timeseries.timestamps
        values = timeseries.values
        
        # Find valid (non-NaN) values
        valid_mask = ~np.isnan(values)
        
        if not np.any(valid_mask):
            return None
        
        valid_timestamps = timestamps[valid_mask]
        valid_values = values[valid_mask]
        
        # If too few valid points, return None
        if len(valid_values) < 3:
            return None
        
        # Interpolate missing values for small gaps
        if len(valid_values) < len(values):
            # Create interpolation function
            if self.config['interpolation_method'] == 'linear':
                interp_func = interp1d(valid_timestamps, valid_values, kind='linear', 
                                     bounds_error=False, fill_value='extrapolate')
            elif self.config['interpolation_method'] == 'cubic':
                interp_func = interp1d(valid_timestamps, valid_values, kind='cubic',
                                     bounds_error=False, fill_value='extrapolate')
            else:  # nearest
                interp_func = interp1d(valid_timestamps, valid_values, kind='nearest',
                                     bounds_error=False, fill_value='extrapolate')
            
            # Interpolate only small gaps
            interpolated_values = values.copy()
            gap_starts = []
            gap_ends = []
            
            # Find gaps
            in_gap = False
            for i, is_valid in enumerate(valid_mask):
                if not is_valid and not in_gap:
                    gap_starts.append(i)
                    in_gap = True
                elif is_valid and in_gap:
                    gap_ends.append(i - 1)
                    in_gap = False
            
            if in_gap:
                gap_ends.append(len(valid_mask) - 1)
            
            # Interpolate small gaps
            for start, end in zip(gap_starts, gap_ends):
                gap_size = end - start + 1
                if gap_size <= self.config['max_gap_size']:
                    gap_timestamps = timestamps[start:end+1]
                    interpolated_values[start:end+1] = interp_func(gap_timestamps)
            
            # Remove remaining NaN values
            final_mask = ~np.isnan(interpolated_values)
            if not np.any(final_mask):
                return None
            
            return TimeSeries(
                timestamps=timestamps[final_mask],
                values=interpolated_values[final_mask],
                unit=timeseries.unit,
                sampling_rate=timeseries.sampling_rate
            )
        
        return timeseries
    
    def _apply_filter(self, timeseries: TimeSeries) -> TimeSeries:
        """Apply digital filter to time series data."""
        if len(timeseries.values) < 10:  # Need minimum points for filtering
            return timeseries
        
        try:
            sampling_rate = timeseries.sampling_rate or self._estimate_sampling_rate(timeseries.timestamps)
            nyquist = sampling_rate / 2
            
            filter_type = self.config['filter_type']
            filter_params = self.config['filter_params']
            
            if filter_type == 'lowpass':
                cutoff = filter_params['cutoff']
                if cutoff >= nyquist:
                    return timeseries  # Skip filtering if cutoff too high
                
                sos = signal.butter(filter_params['order'], cutoff / nyquist, 
                                  btype='low', output='sos')
                
            elif filter_type == 'highpass':
                cutoff = filter_params['cutoff']
                if cutoff >= nyquist:
                    return timeseries
                
                sos = signal.butter(filter_params['order'], cutoff / nyquist,
                                  btype='high', output='sos')
                
            elif filter_type == 'bandpass':
                low_cutoff = filter_params['low_cutoff']
                high_cutoff = filter_params['high_cutoff']
                
                if high_cutoff >= nyquist or low_cutoff >= high_cutoff:
                    return timeseries
                
                sos = signal.butter(filter_params['order'], 
                                  [low_cutoff / nyquist, high_cutoff / nyquist],
                                  btype='band', output='sos')
            else:
                return timeseries  # Unknown filter type
            
            # Apply filter
            filtered_values = signal.sosfilt(sos, timeseries.values)
            
            return TimeSeries(
                timestamps=timeseries.timestamps,
                values=filtered_values,
                unit=timeseries.unit,
                sampling_rate=sampling_rate
            )
            
        except Exception as e:
            warnings.warn(f"Failed to apply filter: {str(e)}")
            return timeseries
    
    def _resample_timeseries(self, timeseries: TimeSeries) -> TimeSeries:
        """Resample time series to target sampling rate."""
        current_rate = timeseries.sampling_rate or self._estimate_sampling_rate(timeseries.timestamps)
        target_rate = self.config['sampling_rate']
        
        if abs(current_rate - target_rate) < 1e-6:  # Already at target rate
            return timeseries
        
        try:
            # Create uniform time grid
            start_time = timeseries.timestamps[0]
            end_time = timeseries.timestamps[-1]
            duration = end_time - start_time
            
            num_samples = int(duration * target_rate) + 1
            new_timestamps = np.linspace(start_time, end_time, num_samples)
            
            # Interpolate values to new time grid
            interp_func = interp1d(timeseries.timestamps, timeseries.values,
                                 kind='linear', bounds_error=False, fill_value='extrapolate')
            new_values = interp_func(new_timestamps)
            
            return TimeSeries(
                timestamps=new_timestamps,
                values=new_values,
                unit=timeseries.unit,
                sampling_rate=target_rate
            )
            
        except Exception as e:
            warnings.warn(f"Failed to resample time series: {str(e)}")
            return timeseries
    
    def _estimate_sampling_rate(self, timestamps: np.ndarray) -> float:
        """Estimate sampling rate from timestamps."""
        if len(timestamps) < 2:
            return 1.0
        
        # Calculate time differences
        time_diffs = np.diff(timestamps)
        median_diff = np.median(time_diffs)
        
        if median_diff <= 0:
            return 1.0
        
        return 1.0 / median_diff
    
    def _extract_statistical_features(self, values: np.ndarray) -> List[float]:
        """Extract statistical features from time series values."""
        if len(values) == 0:
            return [0.0] * 8
        
        features = [
            np.mean(values),
            np.std(values),
            np.min(values),
            np.max(values),
            np.median(values),
            np.percentile(values, 25),  # Q1
            np.percentile(values, 75),  # Q3
            stats.skew(values) if len(values) > 2 else 0.0
        ]
        
        return features
    
    def _extract_temporal_features(self, timeseries: TimeSeries) -> List[float]:
        """Extract temporal features from time series."""
        values = timeseries.values
        
        if len(values) < 2:
            return [0.0] * 6
        
        # Rate of change features
        diff_values = np.diff(values)
        
        features = [
            np.mean(diff_values),  # Average rate of change
            np.std(diff_values),   # Variability in rate of change
            np.max(diff_values),   # Maximum increase
            np.min(diff_values),   # Maximum decrease
            len(values),           # Number of data points
            timeseries.timestamps[-1] - timeseries.timestamps[0]  # Duration
        ]
        
        return features
    
    def _extract_trend_features(self, values: np.ndarray) -> List[float]:
        """Extract trend analysis features."""
        if len(values) < 3:
            return [0.0] * 4
        
        # Linear trend
        x = np.arange(len(values))
        slope, intercept, r_value, p_value, std_err = stats.linregress(x, values)
        
        # Moving average trend
        window_size = min(self.config['trend_window'], len(values) // 2)
        if window_size > 1:
            moving_avg = pd.Series(values).rolling(window=window_size, center=True).mean()
            trend_strength = np.nanstd(moving_avg) / np.std(values) if np.std(values) > 0 else 0
        else:
            trend_strength = 0
        
        features = [
            slope,           # Linear trend slope
            r_value ** 2,    # R-squared of linear fit
            trend_strength,  # Strength of trend relative to noise
            p_value          # Statistical significance of trend
        ]
        
        return features
    
    def _extract_anomaly_features(self, values: np.ndarray, sensor_type: str) -> List[float]:
        """Extract anomaly detection features."""
        if len(values) == 0:
            return [0.0] * 4
        
        features = []
        
        # Statistical outliers using Z-score
        z_scores = np.abs(stats.zscore(values))
        outlier_count = np.sum(z_scores > self.config['outlier_threshold'])
        outlier_ratio = outlier_count / len(values)
        
        features.extend([outlier_count, outlier_ratio])
        
        # Anomaly detection using fitted detector
        if (self.anomaly_detectors[sensor_type] is not None and 
            self.config['anomaly_method'] == 'isolation_forest'):
            
            try:
                anomaly_scores = self.anomaly_detectors[sensor_type].decision_function(values.reshape(-1, 1))
                anomaly_predictions = self.anomaly_detectors[sensor_type].predict(values.reshape(-1, 1))
                
                anomaly_count = np.sum(anomaly_predictions == -1)
                anomaly_ratio = anomaly_count / len(values)
                
                features.extend([anomaly_count, anomaly_ratio])
            except Exception:
                features.extend([0.0, 0.0])
        else:
            features.extend([0.0, 0.0])
        
        return features
    
    def _extract_frequency_features(self, values: np.ndarray, sampling_rate: Optional[float]) -> List[float]:
        """Extract frequency domain features."""
        if len(values) < 8 or sampling_rate is None:
            return [0.0] * 6
        
        try:
            # Compute power spectral density
            freqs, psd = signal.welch(values, fs=sampling_rate, nperseg=min(len(values)//2, 256))
            
            # Remove DC component
            if len(freqs) > 1:
                freqs = freqs[1:]
                psd = psd[1:]
            
            if len(psd) == 0:
                return [0.0] * 6
            
            # Frequency domain features
            total_power = np.sum(psd)
            dominant_freq = freqs[np.argmax(psd)] if len(psd) > 0 else 0
            
            # Spectral centroid (weighted average frequency)
            spectral_centroid = np.sum(freqs * psd) / total_power if total_power > 0 else 0
            
            # Spectral rolloff (frequency below which 85% of energy is contained)
            cumulative_power = np.cumsum(psd)
            rolloff_idx = np.where(cumulative_power >= 0.85 * total_power)[0]
            spectral_rolloff = freqs[rolloff_idx[0]] if len(rolloff_idx) > 0 else freqs[-1]
            
            # Spectral bandwidth
            spectral_bandwidth = np.sqrt(np.sum(((freqs - spectral_centroid) ** 2) * psd) / total_power) if total_power > 0 else 0
            
            # High frequency energy ratio
            nyquist = sampling_rate / 2
            high_freq_mask = freqs > nyquist * 0.5
            high_freq_energy = np.sum(psd[high_freq_mask]) / total_power if total_power > 0 else 0
            
            features = [
                dominant_freq,
                spectral_centroid,
                spectral_rolloff,
                spectral_bandwidth,
                high_freq_energy,
                total_power
            ]
            
            return features
            
        except Exception as e:
            warnings.warn(f"Failed to extract frequency features: {str(e)}")
            return [0.0] * 6
    
    def _get_feature_count(self) -> int:
        """Get the number of features extracted per sensor type."""
        # Statistical: 8, Temporal: 6, Trend: 4, Anomaly: 4, Frequency: 6
        return 8 + 6 + 4 + 4 + 6
    
    def detect_anomalies(self, timeseries: TimeSeries, sensor_type: str) -> Dict[str, Any]:
        """Detect anomalies in time series data."""
        if not self._is_fitted:
            raise ValueError("Processor must be fitted before anomaly detection")
        
        processed_ts = self._preprocess_timeseries(timeseries)
        if processed_ts is None:
            return {'anomalies': [], 'anomaly_scores': [], 'method': 'none'}
        
        values = processed_ts.values
        timestamps = processed_ts.timestamps
        
        results = {
            'timestamps': timestamps,
            'values': values,
            'anomalies': [],
            'anomaly_scores': [],
            'method': self.config['anomaly_method']
        }
        
        if self.config['anomaly_method'] == 'isolation_forest':
            detector = self.anomaly_detectors.get(sensor_type)
            if detector is not None:
                try:
                    anomaly_scores = detector.decision_function(values.reshape(-1, 1))
                    anomaly_predictions = detector.predict(values.reshape(-1, 1))
                    
                    anomaly_indices = np.where(anomaly_predictions == -1)[0]
                    
                    results['anomaly_scores'] = anomaly_scores
                    results['anomalies'] = [
                        {
                            'timestamp': timestamps[idx],
                            'value': values[idx],
                            'score': anomaly_scores[idx],
                            'index': idx
                        }
                        for idx in anomaly_indices
                    ]
                except Exception as e:
                    warnings.warn(f"Isolation forest anomaly detection failed: {str(e)}")
        
        elif self.config['anomaly_method'] == 'statistical':
            # Z-score based anomaly detection
            z_scores = np.abs(stats.zscore(values))
            threshold = self.config['outlier_threshold']
            
            anomaly_indices = np.where(z_scores > threshold)[0]
            
            results['anomaly_scores'] = z_scores
            results['anomalies'] = [
                {
                    'timestamp': timestamps[idx],
                    'value': values[idx],
                    'score': z_scores[idx],
                    'index': idx
                }
                for idx in anomaly_indices
            ]
        
        return results
    
    def analyze_trends(self, timeseries: TimeSeries) -> Dict[str, Any]:
        """Analyze trends in time series data."""
        processed_ts = self._preprocess_timeseries(timeseries)
        if processed_ts is None:
            return {'trend': 'unknown', 'strength': 0, 'significance': 0}
        
        values = processed_ts.values
        timestamps = processed_ts.timestamps
        
        if len(values) < 3:
            return {'trend': 'insufficient_data', 'strength': 0, 'significance': 0}
        
        # Linear trend analysis
        x = np.arange(len(values))
        slope, intercept, r_value, p_value, std_err = stats.linregress(x, values)
        
        # Determine trend direction
        if p_value < 0.05:  # Statistically significant
            if slope > 0:
                trend_direction = 'increasing'
            else:
                trend_direction = 'decreasing'
        else:
            trend_direction = 'stable'
        
        # Moving average trend
        window_size = min(self.config['trend_window'], len(values) // 2)
        if window_size > 1:
            moving_avg = pd.Series(values).rolling(window=window_size, center=True).mean()
            trend_strength = np.nanstd(moving_avg) / np.std(values) if np.std(values) > 0 else 0
        else:
            trend_strength = abs(r_value)
        
        return {
            'trend': trend_direction,
            'slope': slope,
            'r_squared': r_value ** 2,
            'p_value': p_value,
            'strength': trend_strength,
            'significance': 1 - p_value,
            'timestamps': timestamps,
            'values': values,
            'fitted_line': intercept + slope * x
        }    

    def normalize_sensor_data(self, timeseries: TimeSeries, sensor_type: str) -> TimeSeries:
        """Normalize sensor data using fitted scaler."""
        if not self._is_fitted:
            raise ValueError("Processor must be fitted before normalization")
        
        scaler = self.scalers.get(sensor_type)
        if scaler is None:
            warnings.warn(f"No scaler fitted for sensor type: {sensor_type}")
            return timeseries
        
        processed_ts = self._preprocess_timeseries(timeseries)
        if processed_ts is None:
            return timeseries
        
        try:
            normalized_values = scaler.transform(processed_ts.values.reshape(-1, 1)).flatten()
            
            return TimeSeries(
                timestamps=processed_ts.timestamps,
                values=normalized_values,
                unit=f"normalized_{timeseries.unit}",
                sampling_rate=processed_ts.sampling_rate
            )
        except Exception as e:
            warnings.warn(f"Failed to normalize sensor data: {str(e)}")
            return timeseries
    
    def get_sensor_statistics(self, sensor_type: str) -> Dict[str, float]:
        """Get statistics for a specific sensor type."""
        if not self._is_fitted:
            raise ValueError("Processor must be fitted before getting statistics")
        
        return self.data_stats.get(sensor_type, {})
    
    def process_displacement_data(self, displacement_ts: TimeSeries) -> Dict[str, Any]:
        """Process displacement sensor data with specific analysis."""
        results = {
            'sensor_type': 'displacement',
            'processed_timeseries': None,
            'features': None,
            'anomalies': None,
            'trends': None,
            'alerts': []
        }
        
        # Preprocess data
        processed_ts = self._preprocess_timeseries(displacement_ts)
        if processed_ts is None:
            results['alerts'].append('Failed to preprocess displacement data')
            return results
        
        results['processed_timeseries'] = processed_ts
        
        # Extract features
        features = self._extract_sensor_features(displacement_ts, 'displacement')
        results['features'] = features
        
        # Detect anomalies
        anomalies = self.detect_anomalies(displacement_ts, 'displacement')
        results['anomalies'] = anomalies
        
        # Analyze trends
        trends = self.analyze_trends(displacement_ts)
        results['trends'] = trends
        
        # Generate alerts based on displacement-specific thresholds
        if len(processed_ts.values) > 0:
            max_displacement = np.max(np.abs(processed_ts.values))
            rate_of_change = np.max(np.abs(np.diff(processed_ts.values))) if len(processed_ts.values) > 1 else 0
            
            # Alert thresholds (these would be configurable in practice)
            if max_displacement > 10.0:  # mm
                results['alerts'].append(f'High displacement detected: {max_displacement:.2f} mm')
            
            if rate_of_change > 5.0:  # mm per time unit
                results['alerts'].append(f'Rapid displacement change: {rate_of_change:.2f} mm/unit')
            
            if trends['trend'] == 'increasing' and trends['significance'] > 0.95:
                results['alerts'].append('Significant increasing displacement trend detected')
        
        return results
    
    def process_strain_data(self, strain_ts: TimeSeries) -> Dict[str, Any]:
        """Process strain sensor data with specific analysis."""
        results = {
            'sensor_type': 'strain',
            'processed_timeseries': None,
            'features': None,
            'anomalies': None,
            'trends': None,
            'alerts': []
        }
        
        # Preprocess data
        processed_ts = self._preprocess_timeseries(strain_ts)
        if processed_ts is None:
            results['alerts'].append('Failed to preprocess strain data')
            return results
        
        results['processed_timeseries'] = processed_ts
        
        # Extract features
        features = self._extract_sensor_features(strain_ts, 'strain')
        results['features'] = features
        
        # Detect anomalies
        anomalies = self.detect_anomalies(strain_ts, 'strain')
        results['anomalies'] = anomalies
        
        # Analyze trends
        trends = self.analyze_trends(strain_ts)
        results['trends'] = trends
        
        # Generate alerts based on strain-specific thresholds
        if len(processed_ts.values) > 0:
            max_strain = np.max(np.abs(processed_ts.values))
            strain_variance = np.var(processed_ts.values)
            
            # Alert thresholds (these would be configurable in practice)
            if max_strain > 1000.0:  # microstrain
                results['alerts'].append(f'High strain detected: {max_strain:.2f} microstrain')
            
            if strain_variance > 10000.0:  # high variability
                results['alerts'].append(f'High strain variability: {strain_variance:.2f}')
            
            if len(anomalies['anomalies']) > len(processed_ts.values) * 0.1:
                results['alerts'].append(f'High anomaly rate in strain data: {len(anomalies["anomalies"])} anomalies')
        
        return results
    
    def process_pore_pressure_data(self, pore_pressure_ts: TimeSeries) -> Dict[str, Any]:
        """Process pore pressure sensor data with specific analysis."""
        results = {
            'sensor_type': 'pore_pressure',
            'processed_timeseries': None,
            'features': None,
            'anomalies': None,
            'trends': None,
            'alerts': []
        }
        
        # Preprocess data
        processed_ts = self._preprocess_timeseries(pore_pressure_ts)
        if processed_ts is None:
            results['alerts'].append('Failed to preprocess pore pressure data')
            return results
        
        results['processed_timeseries'] = processed_ts
        
        # Extract features
        features = self._extract_sensor_features(pore_pressure_ts, 'pore_pressure')
        results['features'] = features
        
        # Detect anomalies
        anomalies = self.detect_anomalies(pore_pressure_ts, 'pore_pressure')
        results['anomalies'] = anomalies
        
        # Analyze trends
        trends = self.analyze_trends(pore_pressure_ts)
        results['trends'] = trends
        
        # Generate alerts based on pore pressure-specific thresholds
        if len(processed_ts.values) > 0:
            max_pressure = np.max(processed_ts.values)
            pressure_increase_rate = 0
            
            if len(processed_ts.values) > 1:
                pressure_changes = np.diff(processed_ts.values)
                pressure_increase_rate = np.max(pressure_changes[pressure_changes > 0]) if np.any(pressure_changes > 0) else 0
            
            # Alert thresholds (these would be configurable in practice)
            if max_pressure > 500.0:  # kPa
                results['alerts'].append(f'High pore pressure detected: {max_pressure:.2f} kPa')
            
            if pressure_increase_rate > 50.0:  # kPa per time unit
                results['alerts'].append(f'Rapid pore pressure increase: {pressure_increase_rate:.2f} kPa/unit')
            
            if trends['trend'] == 'increasing' and trends['significance'] > 0.9:
                results['alerts'].append('Significant increasing pore pressure trend detected')
        
        return results