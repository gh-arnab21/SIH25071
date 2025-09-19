"""Environmental data processing pipeline for weather and environmental measurements."""

import warnings
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler

from ..base import BaseDataProcessor, DataValidator
from ..schemas import EnvironmentalData


class EnvironmentalProcessor(BaseDataProcessor):
    """Processor for environmental data including rainfall, temperature, and vibration measurements."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize the environmental data processor.
        
        Args:
            config: Configuration dictionary with processing parameters
        """
        super().__init__(config)
        
        # Default configuration
        default_config = {
            'normalization': 'standard',
            'rainfall_cumulative_window': 24,  # hours for cumulative rainfall
            'temperature_smoothing_window': 6,  # hours for temperature smoothing
            'vibration_threshold': 5.0,  # threshold for significant vibrations
            'rainfall_trigger_threshold': 50.0,  # mm cumulative rainfall trigger
            'temperature_freeze_threshold': 0.0,  # Celsius
            'temperature_extreme_high': 40.0,  # Celsius
            'temperature_extreme_low': -20.0,  # Celsius
            'vibration_anomaly_threshold': 3.0,  # Z-score threshold
            'missing_value_strategy': 'interpolate',  # 'interpolate', 'mean', 'median', 'zero'
            'outlier_detection_method': 'iqr',  # 'iqr', 'zscore', 'none'
            'outlier_threshold': 3.0,  # threshold for outlier detection
        }
        
        self.config = {**default_config, **self.config}
        
        # Initialize scalers for each environmental parameter
        self.scalers = {
            'rainfall': None,
            'temperature': None,
            'vibrations': None,
            'wind_speed': None
        }
        
        # Statistics for fitted data
        self.data_stats = {}
        
        # Trigger thresholds learned from data
        self.trigger_thresholds = {}
    
    def fit(self, data: List[EnvironmentalData]) -> 'EnvironmentalProcessor':
        """Fit the processor to training data."""
        # Collect all environmental measurements for fitting scalers
        all_rainfall = []
        all_temperature = []
        all_vibrations = []
        all_wind_speed = []
        
        for env_data in data:
            if env_data.rainfall is not None and not np.isnan(env_data.rainfall):
                all_rainfall.append(env_data.rainfall)
            
            if env_data.temperature is not None and not np.isnan(env_data.temperature):
                all_temperature.append(env_data.temperature)
            
            if env_data.vibrations is not None and not np.isnan(env_data.vibrations):
                all_vibrations.append(env_data.vibrations)
            
            if env_data.wind_speed is not None and not np.isnan(env_data.wind_speed):
                all_wind_speed.append(env_data.wind_speed)
        
        # Fit scalers and calculate statistics for each parameter
        self._fit_parameter_processors('rainfall', all_rainfall)
        self._fit_parameter_processors('temperature', all_temperature)
        self._fit_parameter_processors('vibrations', all_vibrations)
        self._fit_parameter_processors('wind_speed', all_wind_speed)
        
        # Calculate trigger thresholds based on data distribution
        self._calculate_trigger_thresholds()
        
        self._is_fitted = True
        return self
    
    def _fit_parameter_processors(self, param_name: str, values: List[float]):
        """Fit scalers and calculate statistics for a specific parameter."""
        if not values:
            return
        
        values_array = np.array(values).reshape(-1, 1)
        
        # Fit scaler
        if self.config['normalization'] == 'standard':
            scaler = StandardScaler()
        elif self.config['normalization'] == 'minmax':
            scaler = MinMaxScaler()
        else:  # robust
            scaler = RobustScaler()
        
        scaler.fit(values_array)
        self.scalers[param_name] = scaler
        
        # Store statistics
        self.data_stats[param_name] = {
            'mean': np.mean(values),
            'std': np.std(values),
            'min': np.min(values),
            'max': np.max(values),
            'median': np.median(values),
            'q25': np.percentile(values, 25),
            'q75': np.percentile(values, 75),
            'count': len(values)
        }
    
    def _calculate_trigger_thresholds(self):
        """Calculate trigger thresholds based on data statistics."""
        # Rainfall trigger threshold (95th percentile or configured value)
        if 'rainfall' in self.data_stats:
            rainfall_95th = np.percentile([self.data_stats['rainfall']['max']], 95) if self.data_stats['rainfall']['count'] > 0 else 0
            self.trigger_thresholds['rainfall'] = max(self.config['rainfall_trigger_threshold'], rainfall_95th)
        else:
            self.trigger_thresholds['rainfall'] = self.config['rainfall_trigger_threshold']
        
        # Temperature extreme thresholds
        self.trigger_thresholds['temperature_freeze'] = self.config['temperature_freeze_threshold']
        self.trigger_thresholds['temperature_extreme_high'] = self.config['temperature_extreme_high']
        self.trigger_thresholds['temperature_extreme_low'] = self.config['temperature_extreme_low']
        
        # Vibration threshold (mean + 3*std or configured value)
        if 'vibrations' in self.data_stats and self.data_stats['vibrations']['count'] > 0:
            vibration_threshold = (self.data_stats['vibrations']['mean'] + 
                                 3 * self.data_stats['vibrations']['std'])
            self.trigger_thresholds['vibrations'] = max(self.config['vibration_threshold'], vibration_threshold)
        else:
            self.trigger_thresholds['vibrations'] = self.config['vibration_threshold']
        
        # Wind speed threshold (90th percentile)
        if 'wind_speed' in self.data_stats and self.data_stats['wind_speed']['count'] > 0:
            wind_90th = self.data_stats['wind_speed']['q75'] + 1.5 * (
                self.data_stats['wind_speed']['q75'] - self.data_stats['wind_speed']['q25'])
            self.trigger_thresholds['wind_speed'] = wind_90th
        else:
            self.trigger_thresholds['wind_speed'] = 20.0  # Default 20 m/s
    
    def transform(self, data: Union[EnvironmentalData, List[EnvironmentalData]]) -> np.ndarray:
        """Transform environmental data to processed features."""
        if not self._is_fitted:
            raise ValueError("Processor must be fitted before transform")
        
        if isinstance(data, list):
            return np.array([self._process_single_environmental_data(env_data) for env_data in data])
        else:
            return self._process_single_environmental_data(data)
    
    def _process_single_environmental_data(self, env_data: EnvironmentalData) -> np.ndarray:
        """Process a single EnvironmentalData object."""
        features = []
        
        # Process rainfall data
        rainfall_features = self._extract_rainfall_features(env_data.rainfall)
        features.extend(rainfall_features)
        
        # Process temperature data
        temperature_features = self._extract_temperature_features(env_data.temperature)
        features.extend(temperature_features)
        
        # Process vibration data
        vibration_features = self._extract_vibration_features(env_data.vibrations)
        features.extend(vibration_features)
        
        # Process wind speed data
        wind_features = self._extract_wind_features(env_data.wind_speed)
        features.extend(wind_features)
        
        # Calculate combined environmental risk indicators
        combined_features = self._calculate_combined_risk_indicators(env_data)
        features.extend(combined_features)
        
        return np.array(features)
    
    def _extract_rainfall_features(self, rainfall: Optional[float]) -> List[float]:
        """Extract features from rainfall data."""
        if rainfall is None or np.isnan(rainfall):
            return [0.0] * 6
        
        # Handle negative rainfall (data error)
        rainfall = max(0.0, rainfall)
        
        features = []
        
        # Raw and normalized rainfall
        features.append(rainfall)
        
        # Normalized rainfall
        if self.scalers['rainfall'] is not None:
            try:
                normalized = self.scalers['rainfall'].transform([[rainfall]])[0][0]
                features.append(normalized)
            except Exception:
                features.append(0.0)
        else:
            features.append(0.0)
        
        # Rainfall intensity categories
        features.append(1.0 if rainfall > 0.1 else 0.0)  # Light rain
        features.append(1.0 if rainfall > 2.5 else 0.0)  # Moderate rain
        features.append(1.0 if rainfall > 10.0 else 0.0)  # Heavy rain
        
        # Trigger threshold indicator
        trigger_exceeded = 1.0 if rainfall > self.trigger_thresholds.get('rainfall', 50.0) else 0.0
        features.append(trigger_exceeded)
        
        return features
    
    def _extract_temperature_features(self, temperature: Optional[float]) -> List[float]:
        """Extract features from temperature data."""
        if temperature is None or np.isnan(temperature):
            return [0.0] * 8
        
        features = []
        
        # Raw and normalized temperature
        features.append(temperature)
        
        # Normalized temperature
        if self.scalers['temperature'] is not None:
            try:
                normalized = self.scalers['temperature'].transform([[temperature]])[0][0]
                features.append(normalized)
            except Exception:
                features.append(0.0)
        else:
            features.append(0.0)
        
        # Temperature condition indicators
        features.append(1.0 if temperature <= self.trigger_thresholds['temperature_freeze'] else 0.0)  # Freezing
        features.append(1.0 if temperature >= self.trigger_thresholds['temperature_extreme_high'] else 0.0)  # Extreme hot
        features.append(1.0 if temperature <= self.trigger_thresholds['temperature_extreme_low'] else 0.0)  # Extreme cold
        
        # Temperature ranges
        features.append(1.0 if -5 <= temperature <= 5 else 0.0)  # Freeze-thaw range
        features.append(1.0 if 20 <= temperature <= 30 else 0.0)  # Optimal range
        
        # Distance from freezing point
        features.append(abs(temperature - 0.0))
        
        return features
    
    def _extract_vibration_features(self, vibrations: Optional[float]) -> List[float]:
        """Extract features from vibration data."""
        if vibrations is None or np.isnan(vibrations):
            return [0.0] * 6
        
        # Handle negative vibrations (use absolute value)
        vibrations = abs(vibrations)
        
        features = []
        
        # Raw and normalized vibrations
        features.append(vibrations)
        
        # Normalized vibrations
        if self.scalers['vibrations'] is not None:
            try:
                normalized = self.scalers['vibrations'].transform([[vibrations]])[0][0]
                features.append(normalized)
            except Exception:
                features.append(0.0)
        else:
            features.append(0.0)
        
        # Vibration intensity levels
        features.append(1.0 if vibrations > 1.0 else 0.0)  # Low vibrations
        features.append(1.0 if vibrations > self.config['vibration_threshold'] else 0.0)  # Significant vibrations
        features.append(1.0 if vibrations > self.trigger_thresholds.get('vibrations', 5.0) else 0.0)  # High vibrations
        
        # Logarithmic vibration (for wide range of values)
        log_vibration = np.log1p(vibrations)  # log(1 + x) to handle zero values
        features.append(log_vibration)
        
        return features
    
    def _extract_wind_features(self, wind_speed: Optional[float]) -> List[float]:
        """Extract features from wind speed data."""
        if wind_speed is None or np.isnan(wind_speed):
            return [0.0] * 6
        
        # Handle negative wind speed (use absolute value)
        wind_speed = abs(wind_speed)
        
        features = []
        
        # Raw and normalized wind speed
        features.append(wind_speed)
        
        # Normalized wind speed
        if self.scalers['wind_speed'] is not None:
            try:
                normalized = self.scalers['wind_speed'].transform([[wind_speed]])[0][0]
                features.append(normalized)
            except Exception:
                features.append(0.0)
        else:
            features.append(0.0)
        
        # Wind speed categories (Beaufort scale inspired)
        features.append(1.0 if wind_speed > 5.0 else 0.0)  # Fresh breeze
        features.append(1.0 if wind_speed > 10.0 else 0.0)  # Strong breeze
        features.append(1.0 if wind_speed > 15.0 else 0.0)  # Near gale
        
        # High wind threshold
        trigger_exceeded = 1.0 if wind_speed > self.trigger_thresholds.get('wind_speed', 20.0) else 0.0
        features.append(trigger_exceeded)
        
        return features
    
    def _calculate_combined_risk_indicators(self, env_data: EnvironmentalData) -> List[float]:
        """Calculate combined environmental risk indicators."""
        features = []
        
        # Multi-factor risk combinations
        rainfall = env_data.rainfall or 0.0
        temperature = env_data.temperature or 0.0
        vibrations = env_data.vibrations or 0.0
        wind_speed = env_data.wind_speed or 0.0
        
        # Rain + freeze-thaw risk
        freeze_thaw_risk = 0.0
        if rainfall > 0.1 and -2 <= temperature <= 2:
            freeze_thaw_risk = 1.0
        features.append(freeze_thaw_risk)
        
        # Heavy rain + vibration risk
        rain_vibration_risk = 0.0
        if rainfall > 5.0 and vibrations > self.config['vibration_threshold']:
            rain_vibration_risk = 1.0
        features.append(rain_vibration_risk)
        
        # Extreme weather combination
        extreme_weather_risk = 0.0
        extreme_conditions = [
            rainfall > self.trigger_thresholds.get('rainfall', 50.0),
            temperature <= self.trigger_thresholds['temperature_extreme_low'],
            temperature >= self.trigger_thresholds['temperature_extreme_high'],
            wind_speed > self.trigger_thresholds.get('wind_speed', 20.0)
        ]
        if sum(extreme_conditions) >= 2:
            extreme_weather_risk = 1.0
        features.append(extreme_weather_risk)
        
        # Environmental stress index (normalized combination)
        stress_components = []
        if rainfall > 0:
            stress_components.append(min(rainfall / self.trigger_thresholds.get('rainfall', 50.0), 1.0))
        if abs(temperature) > 0:
            temp_stress = max(
                abs(temperature - self.trigger_thresholds['temperature_extreme_low']) / 20.0,
                abs(temperature - self.trigger_thresholds['temperature_extreme_high']) / 20.0
            )
            stress_components.append(min(temp_stress, 1.0))
        if vibrations > 0:
            vib_stress = vibrations / self.trigger_thresholds.get('vibrations', 5.0)
            stress_components.append(min(vib_stress, 1.0))
        if wind_speed > 0:
            wind_stress = wind_speed / self.trigger_thresholds.get('wind_speed', 20.0)
            stress_components.append(min(wind_stress, 1.0))
        
        environmental_stress_index = np.mean(stress_components) if stress_components else 0.0
        features.append(environmental_stress_index)
        
        return features
    
    def validate_environmental_data(self, env_data: EnvironmentalData) -> Dict[str, Any]:
        """Validate environmental data measurements."""
        validation_results = {
            'is_valid': True,
            'errors': [],
            'warnings': [],
            'corrected_data': None
        }
        
        corrected_data = EnvironmentalData(
            rainfall=env_data.rainfall,
            temperature=env_data.temperature,
            vibrations=env_data.vibrations,
            wind_speed=env_data.wind_speed
        )
        
        # Validate rainfall
        if env_data.rainfall is not None:
            if env_data.rainfall < 0:
                validation_results['errors'].append("Rainfall cannot be negative")
                corrected_data.rainfall = 0.0
                validation_results['is_valid'] = False
            elif env_data.rainfall > 500:  # Extreme rainfall threshold
                validation_results['warnings'].append(f"Extremely high rainfall: {env_data.rainfall} mm")
        
        # Validate temperature
        if env_data.temperature is not None:
            if env_data.temperature < -100 or env_data.temperature > 100:
                validation_results['errors'].append(f"Temperature out of reasonable range: {env_data.temperature}°C")
                validation_results['is_valid'] = False
            elif env_data.temperature < -50 or env_data.temperature > 60:
                validation_results['warnings'].append(f"Extreme temperature: {env_data.temperature}°C")
        
        # Validate vibrations
        if env_data.vibrations is not None:
            if env_data.vibrations < 0:
                validation_results['warnings'].append("Negative vibration value, using absolute value")
                corrected_data.vibrations = abs(env_data.vibrations)
            elif env_data.vibrations > 100:  # Very high vibration threshold
                validation_results['warnings'].append(f"Extremely high vibrations: {env_data.vibrations}")
        
        # Validate wind speed
        if env_data.wind_speed is not None:
            if env_data.wind_speed < 0:
                validation_results['warnings'].append("Negative wind speed, using absolute value")
                corrected_data.wind_speed = abs(env_data.wind_speed)
            elif env_data.wind_speed > 200:  # Hurricane-force winds
                validation_results['warnings'].append(f"Extremely high wind speed: {env_data.wind_speed} m/s")
        
        validation_results['corrected_data'] = corrected_data
        return validation_results
    
    def calculate_cumulative_effects(self, environmental_data_series: List[Tuple[float, EnvironmentalData]]) -> Dict[str, Any]:
        """Calculate cumulative effects over time for environmental parameters.
        
        Args:
            environmental_data_series: List of (timestamp, EnvironmentalData) tuples sorted by time
        """
        if not environmental_data_series:
            return {}
        
        # Extract time series data
        timestamps = [item[0] for item in environmental_data_series]
        rainfall_values = [item[1].rainfall or 0.0 for item in environmental_data_series]
        temperature_values = [item[1].temperature for item in environmental_data_series if item[1].temperature is not None]
        vibration_values = [item[1].vibrations for item in environmental_data_series if item[1].vibrations is not None]
        
        cumulative_effects = {}
        
        # Cumulative rainfall over different windows
        if rainfall_values:
            rainfall_array = np.array(rainfall_values)
            window_hours = self.config['rainfall_cumulative_window']
            
            # Calculate cumulative rainfall for the window
            cumulative_rainfall = np.sum(rainfall_array[-window_hours:]) if len(rainfall_array) >= window_hours else np.sum(rainfall_array)
            
            cumulative_effects['cumulative_rainfall'] = {
                'value': cumulative_rainfall,
                'window_hours': window_hours,
                'trigger_exceeded': cumulative_rainfall > self.trigger_thresholds.get('rainfall', 50.0),
                'intensity': 'low' if cumulative_rainfall < 10 else 'moderate' if cumulative_rainfall < 50 else 'high'
            }
        
        # Temperature trends and freeze-thaw cycles
        if temperature_values:
            temp_array = np.array(temperature_values)
            
            # Count freeze-thaw cycles
            freeze_thaw_cycles = 0
            if len(temp_array) > 1:
                crossings = np.where(np.diff(np.sign(temp_array)))[0]
                freeze_thaw_cycles = len(crossings)
            
            # Temperature trend
            if len(temp_array) > 2:
                x = np.arange(len(temp_array))
                slope, _, r_value, p_value, _ = stats.linregress(x, temp_array)
                temp_trend = 'increasing' if slope > 0 and p_value < 0.05 else 'decreasing' if slope < 0 and p_value < 0.05 else 'stable'
            else:
                temp_trend = 'stable'
                slope = 0
                r_value = 0
            
            cumulative_effects['temperature_effects'] = {
                'freeze_thaw_cycles': freeze_thaw_cycles,
                'trend': temp_trend,
                'trend_slope': slope,
                'trend_strength': abs(r_value),
                'current_temperature': temp_array[-1],
                'min_temperature': np.min(temp_array),
                'max_temperature': np.max(temp_array),
                'temperature_range': np.max(temp_array) - np.min(temp_array)
            }
        
        # Vibration accumulation and patterns
        if vibration_values:
            vib_array = np.array(vibration_values)
            
            # Cumulative vibration energy (sum of squares)
            cumulative_vibration_energy = np.sum(vib_array ** 2)
            
            # High vibration events
            high_vib_threshold = self.trigger_thresholds.get('vibrations', 5.0)
            high_vibration_events = np.sum(vib_array > high_vib_threshold)
            
            cumulative_effects['vibration_effects'] = {
                'cumulative_energy': cumulative_vibration_energy,
                'high_vibration_events': high_vibration_events,
                'max_vibration': np.max(vib_array),
                'mean_vibration': np.mean(vib_array),
                'vibration_variability': np.std(vib_array)
            }
        
        # Combined risk assessment
        risk_factors = []
        
        if 'cumulative_rainfall' in cumulative_effects:
            if cumulative_effects['cumulative_rainfall']['trigger_exceeded']:
                risk_factors.append('high_rainfall')
        
        if 'temperature_effects' in cumulative_effects:
            if cumulative_effects['temperature_effects']['freeze_thaw_cycles'] > 5:
                risk_factors.append('frequent_freeze_thaw')
        
        if 'vibration_effects' in cumulative_effects:
            if cumulative_effects['vibration_effects']['high_vibration_events'] > 3:
                risk_factors.append('frequent_vibrations')
        
        cumulative_effects['overall_risk_assessment'] = {
            'risk_factors': risk_factors,
            'risk_level': 'high' if len(risk_factors) >= 2 else 'moderate' if len(risk_factors) == 1 else 'low',
            'assessment_window_hours': max(self.config['rainfall_cumulative_window'], len(environmental_data_series))
        }
        
        return cumulative_effects
    
    def detect_trigger_conditions(self, env_data: EnvironmentalData) -> Dict[str, Any]:
        """Detect if environmental conditions exceed trigger thresholds."""
        triggers = {
            'triggered_conditions': [],
            'trigger_scores': {},
            'overall_trigger_status': False,
            'risk_level': 'low'
        }
        
        # Check rainfall trigger
        if env_data.rainfall is not None:
            rainfall_trigger = env_data.rainfall > self.trigger_thresholds.get('rainfall', 50.0)
            if rainfall_trigger:
                triggers['triggered_conditions'].append('rainfall')
            triggers['trigger_scores']['rainfall'] = min(env_data.rainfall / self.trigger_thresholds.get('rainfall', 50.0), 2.0)
        
        # Check temperature triggers
        if env_data.temperature is not None:
            temp_freeze_trigger = env_data.temperature <= self.trigger_thresholds['temperature_freeze']
            temp_extreme_high_trigger = env_data.temperature >= self.trigger_thresholds['temperature_extreme_high']
            temp_extreme_low_trigger = env_data.temperature <= self.trigger_thresholds['temperature_extreme_low']
            
            if temp_freeze_trigger:
                triggers['triggered_conditions'].append('temperature_freeze')
            if temp_extreme_high_trigger:
                triggers['triggered_conditions'].append('temperature_extreme_high')
            if temp_extreme_low_trigger:
                triggers['triggered_conditions'].append('temperature_extreme_low')
            
            # Temperature trigger score based on distance from normal range
            temp_score = 0.0
            if env_data.temperature <= 0:
                temp_score = abs(env_data.temperature) / 20.0
            elif env_data.temperature >= 35:
                temp_score = (env_data.temperature - 35) / 20.0
            triggers['trigger_scores']['temperature'] = min(temp_score, 2.0)
        
        # Check vibration trigger
        if env_data.vibrations is not None:
            vibration_trigger = env_data.vibrations > self.trigger_thresholds.get('vibrations', 5.0)
            if vibration_trigger:
                triggers['triggered_conditions'].append('vibrations')
            triggers['trigger_scores']['vibrations'] = min(env_data.vibrations / self.trigger_thresholds.get('vibrations', 5.0), 2.0)
        
        # Check wind speed trigger
        if env_data.wind_speed is not None:
            wind_trigger = env_data.wind_speed > self.trigger_thresholds.get('wind_speed', 20.0)
            if wind_trigger:
                triggers['triggered_conditions'].append('wind_speed')
            triggers['trigger_scores']['wind_speed'] = min(env_data.wind_speed / self.trigger_thresholds.get('wind_speed', 20.0), 2.0)
        
        # Overall trigger status
        triggers['overall_trigger_status'] = len(triggers['triggered_conditions']) > 0
        
        # Risk level assessment
        num_triggers = len(triggers['triggered_conditions'])
        max_score = max(triggers['trigger_scores'].values()) if triggers['trigger_scores'] else 0.0
        
        if num_triggers >= 3 or max_score >= 1.5:
            triggers['risk_level'] = 'high'
        elif num_triggers >= 2 or max_score >= 1.0:
            triggers['risk_level'] = 'moderate'
        elif num_triggers >= 1 or max_score >= 0.5:
            triggers['risk_level'] = 'low'
        else:
            triggers['risk_level'] = 'minimal'
        
        return triggers
    
    def get_feature_names(self) -> List[str]:
        """Get names of all extracted features."""
        feature_names = []
        
        # Rainfall features (6)
        feature_names.extend([
            'rainfall_raw', 'rainfall_normalized', 'rainfall_light', 'rainfall_moderate', 
            'rainfall_heavy', 'rainfall_trigger_exceeded'
        ])
        
        # Temperature features (8)
        feature_names.extend([
            'temperature_raw', 'temperature_normalized', 'temperature_freezing', 
            'temperature_extreme_hot', 'temperature_extreme_cold', 'temperature_freeze_thaw_range',
            'temperature_optimal_range', 'temperature_distance_from_freezing'
        ])
        
        # Vibration features (6)
        feature_names.extend([
            'vibrations_raw', 'vibrations_normalized', 'vibrations_low', 
            'vibrations_significant', 'vibrations_high', 'vibrations_log'
        ])
        
        # Wind features (6)
        feature_names.extend([
            'wind_speed_raw', 'wind_speed_normalized', 'wind_fresh_breeze', 
            'wind_strong_breeze', 'wind_near_gale', 'wind_trigger_exceeded'
        ])
        
        # Combined risk indicators (4)
        feature_names.extend([
            'freeze_thaw_risk', 'rain_vibration_risk', 'extreme_weather_risk', 
            'environmental_stress_index'
        ])
        
        return feature_names
    
    def get_parameter_statistics(self, parameter: str) -> Dict[str, float]:
        """Get statistics for a specific environmental parameter."""
        if not self._is_fitted:
            raise ValueError("Processor must be fitted before getting statistics")
        
        return self.data_stats.get(parameter, {})
    
    def get_trigger_thresholds(self) -> Dict[str, float]:
        """Get all trigger thresholds."""
        return self.trigger_thresholds.copy()