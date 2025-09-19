"""LSTM Temporal Feature Extractor for sequential sensor data processing."""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, Any, Optional, Union, List, Tuple
import os
import joblib
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
import warnings

from ...data.base import BaseFeatureExtractor
from ...data.schemas import SensorData, TimeSeries


class LSTMTemporalExtractor(BaseFeatureExtractor):
    """LSTM-based temporal feature extractor for geotechnical sensor data."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize LSTM temporal feature extractor.
        
        Args:
            config: Configuration dictionary with parameters:
                - sequence_length: Length of input sequences (default: 50)
                - hidden_size: LSTM hidden dimension (default: 128)
                - num_layers: Number of LSTM layers (default: 2)
                - feature_dim: Output feature dimension (default: 64)
                - dropout: Dropout rate (default: 0.2)
                - bidirectional: Use bidirectional LSTM (default: True)
                - network_type: 'lstm' or 'gru' (default: 'lstm')
                - device: 'cpu' or 'cuda' (default: auto-detect)
                - scaler_type: 'standard' or 'minmax' (default: 'standard')
        """
        super().__init__(config)
        
        # Default configuration
        default_config = {
            'sequence_length': 50,
            'hidden_size': 128,
            'num_layers': 2,
            'feature_dim': 64,
            'dropout': 0.2,
            'bidirectional': True,
            'network_type': 'lstm',
            'device': 'cuda' if torch.cuda.is_available() else 'cpu',
            'scaler_type': 'standard'
        }
        self.config = {**default_config, **self.config}
        
        # Initialize model components
        self.device = torch.device(self.config['device'])
        self.model = None
        self._feature_dim = self.config['feature_dim']
        
        # Data preprocessing
        self.scaler = None
        self._init_scaler()
        
        # Build the model
        self._build_model()
    
    def _init_scaler(self):
        """Initialize the data scaler."""
        if self.config['scaler_type'] == 'standard':
            self.scaler = StandardScaler()
        elif self.config['scaler_type'] == 'minmax':
            self.scaler = MinMaxScaler()
        else:
            raise ValueError(f"Unsupported scaler type: {self.config['scaler_type']}")
    
    def _build_model(self):
        """Build the LSTM/GRU model."""
        self.model = TemporalFeatureNetwork(
            input_size=3,  # displacement, strain, pore_pressure
            hidden_size=self.config['hidden_size'],
            num_layers=self.config['num_layers'],
            feature_dim=self.config['feature_dim'],
            dropout=self.config['dropout'],
            bidirectional=self.config['bidirectional'],
            network_type=self.config['network_type']
        )
        self.model.to(self.device)
        self.model.eval()
    
    def _prepare_sensor_data(self, sensor_data: SensorData) -> np.ndarray:
        """Prepare sensor data for processing.
        
        Args:
            sensor_data: SensorData object containing time series
            
        Returns:
            Processed sensor array with shape (n_timesteps, 3)
        """
        # Extract time series data
        displacement = sensor_data.displacement.values if sensor_data.displacement else None
        strain = sensor_data.strain.values if sensor_data.strain else None
        pore_pressure = sensor_data.pore_pressure.values if sensor_data.pore_pressure else None
        
        # Determine the common length
        lengths = []
        if displacement is not None:
            lengths.append(len(displacement))
        if strain is not None:
            lengths.append(len(strain))
        if pore_pressure is not None:
            lengths.append(len(pore_pressure))
        
        if not lengths:
            raise ValueError("No sensor data available")
        
        # Use the minimum length to ensure all series have the same length
        min_length = min(lengths)
        
        # Create the combined array
        combined_data = np.zeros((min_length, 3))
        
        if displacement is not None:
            combined_data[:, 0] = displacement[:min_length]
        if strain is not None:
            combined_data[:, 1] = strain[:min_length]
        if pore_pressure is not None:
            combined_data[:, 2] = pore_pressure[:min_length]
        
        return combined_data
    
    def _create_sequences(self, data: np.ndarray) -> np.ndarray:
        """Create sequences from time series data.
        
        Args:
            data: Input time series data with shape (n_timesteps, n_features)
            
        Returns:
            Sequences with shape (n_sequences, sequence_length, n_features)
        """
        sequence_length = self.config['sequence_length']
        
        if len(data) < sequence_length:
            # Pad with zeros if data is too short
            padding = np.zeros((sequence_length - len(data), data.shape[1]))
            data = np.vstack([padding, data])
        
        sequences = []
        for i in range(len(data) - sequence_length + 1):
            sequences.append(data[i:i + sequence_length])
        
        return np.array(sequences)
    
    def fit_scaler(self, sensor_data_list: List[SensorData]):
        """Fit the scaler on training data.
        
        Args:
            sensor_data_list: List of SensorData objects for fitting
        """
        all_data = []
        
        for sensor_data in sensor_data_list:
            try:
                processed_data = self._prepare_sensor_data(sensor_data)
                all_data.append(processed_data)
            except Exception as e:
                warnings.warn(f"Failed to process sensor data: {e}")
                continue
        
        if not all_data:
            raise ValueError("No valid sensor data found for fitting scaler")
        
        # Concatenate all data and fit scaler
        combined_data = np.vstack(all_data)
        self.scaler.fit(combined_data)
    
    def extract_features(self, data: Union[SensorData, np.ndarray]) -> np.ndarray:
        """Extract temporal features from sensor data.
        
        Args:
            data: SensorData object or preprocessed numpy array
            
        Returns:
            Feature vector as numpy array
        """
        if isinstance(data, SensorData):
            # Prepare sensor data
            processed_data = self._prepare_sensor_data(data)
        elif isinstance(data, np.ndarray):
            processed_data = data
        else:
            raise ValueError(f"Unsupported input type: {type(data)}")
        
        # Scale the data
        if self.scaler is None:
            raise ValueError("Scaler not fitted. Call fit_scaler() first.")
        
        try:
            scaled_data = self.scaler.transform(processed_data)
        except Exception as e:
            if "not fitted" in str(e).lower():
                raise ValueError("Scaler not fitted. Call fit_scaler() first.")
            else:
                raise e
        
        # Create sequences
        sequences = self._create_sequences(scaled_data)
        
        # Extract features from all sequences and aggregate
        all_features = []
        
        for sequence in sequences:
            # Convert to tensor
            sequence_tensor = torch.FloatTensor(sequence).unsqueeze(0).to(self.device)
            
            # Extract features
            with torch.no_grad():
                features = self.model(sequence_tensor)
                features = features.cpu().numpy().flatten()
                all_features.append(features)
        
        # Aggregate features (mean pooling)
        if all_features:
            aggregated_features = np.mean(all_features, axis=0)
        else:
            # Return zero features if no sequences available
            aggregated_features = np.zeros(self.config['feature_dim'])
        
        return aggregated_features
    
    def extract_batch_features(self, sensor_data_list: List[SensorData]) -> np.ndarray:
        """Extract features from a batch of sensor data.
        
        Args:
            sensor_data_list: List of SensorData objects
            
        Returns:
            Feature matrix with shape (n_samples, feature_dim)
        """
        batch_features = []
        
        for sensor_data in sensor_data_list:
            try:
                features = self.extract_features(sensor_data)
                batch_features.append(features)
            except Exception as e:
                warnings.warn(f"Failed to process sensor data: {e}")
                # Add zero features for failed samples
                batch_features.append(np.zeros(self.config['feature_dim']))
        
        return np.array(batch_features)
    
    def identify_precursor_patterns(self, sensor_data: SensorData, 
                                  threshold: float = 0.8) -> Dict[str, Any]:
        """Identify precursor patterns in sensor readings.
        
        Args:
            sensor_data: SensorData object to analyze
            threshold: Threshold for pattern detection
            
        Returns:
            Dictionary containing pattern analysis results
        """
        processed_data = self._prepare_sensor_data(sensor_data)
        scaled_data = self.scaler.transform(processed_data)
        sequences = self._create_sequences(scaled_data)
        
        pattern_scores = []
        anomaly_indices = []
        
        for i, sequence in enumerate(sequences):
            sequence_tensor = torch.FloatTensor(sequence).unsqueeze(0).to(self.device)
            
            with torch.no_grad():
                # Get features and attention weights
                features = self.model(sequence_tensor)
                
                # Calculate pattern strength (using feature magnitude)
                pattern_strength = torch.norm(features).item()
                pattern_scores.append(pattern_strength)
                
                # Identify potential anomalies
                if pattern_strength > threshold:
                    anomaly_indices.append(i)
        
        # Calculate trend analysis
        displacement_trend = self._calculate_trend(processed_data[:, 0])
        strain_trend = self._calculate_trend(processed_data[:, 1])
        pressure_trend = self._calculate_trend(processed_data[:, 2])
        
        return {
            'pattern_scores': np.array(pattern_scores),
            'anomaly_indices': anomaly_indices,
            'displacement_trend': displacement_trend,
            'strain_trend': strain_trend,
            'pressure_trend': pressure_trend,
            'max_pattern_score': max(pattern_scores) if pattern_scores else 0.0,
            'mean_pattern_score': np.mean(pattern_scores) if pattern_scores else 0.0
        }
    
    def _calculate_trend(self, data: np.ndarray) -> Dict[str, float]:
        """Calculate trend statistics for time series data.
        
        Args:
            data: Time series data
            
        Returns:
            Dictionary with trend statistics
        """
        if len(data) < 2:
            return {'slope': 0.0, 'acceleration': 0.0, 'volatility': 0.0}
        
        # Calculate first derivative (velocity)
        velocity = np.diff(data)
        
        # Calculate second derivative (acceleration)
        acceleration = np.diff(velocity) if len(velocity) > 1 else np.array([0.0])
        
        # Calculate trend slope using linear regression
        x = np.arange(len(data))
        slope = np.polyfit(x, data, 1)[0] if len(data) > 1 else 0.0
        
        # Calculate volatility (standard deviation of changes)
        volatility = np.std(velocity) if len(velocity) > 0 else 0.0
        
        return {
            'slope': float(slope),
            'acceleration': float(np.mean(acceleration)),
            'volatility': float(volatility)
        }
    
    def save_model(self, model_path: str):
        """Save the trained model and scaler to disk.
        
        Args:
            model_path: Path to save the model
        """
        model_dir = os.path.dirname(model_path)
        if model_dir:
            os.makedirs(model_dir, exist_ok=True)
        
        # Save model state dict, config, and scaler
        save_dict = {
            'model_state_dict': self.model.state_dict(),
            'config': self.config,
            'scaler': self.scaler
        }
        torch.save(save_dict, model_path)
    
    def load_model(self, model_path: str):
        """Load a trained model from disk.
        
        Args:
            model_path: Path to load the model from
        """
        checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)
        
        # Update config and rebuild model if necessary
        self.config.update(checkpoint['config'])
        self._feature_dim = self.config['feature_dim']
        self._build_model()
        
        # Load model weights and scaler
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.scaler = checkpoint['scaler']
        self.model.eval()
    
    def save_features(self, features: np.ndarray, output_path: str):
        """Save extracted features to disk.
        
        Args:
            features: Feature array to save
            output_path: Path to save the features
        """
        output_dir = os.path.dirname(output_path)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        joblib.dump(features, output_path)
    
    def load_features(self, input_path: str) -> np.ndarray:
        """Load features from disk.
        
        Args:
            input_path: Path to load features from
            
        Returns:
            Loaded feature array
        """
        return joblib.load(input_path)


class TemporalFeatureNetwork(nn.Module):
    """Neural network for temporal feature extraction."""
    
    def __init__(self, input_size: int, hidden_size: int, num_layers: int,
                 feature_dim: int, dropout: float = 0.2, 
                 bidirectional: bool = True, network_type: str = 'lstm'):
        """Initialize the temporal feature network.
        
        Args:
            input_size: Number of input features
            hidden_size: Hidden dimension size
            num_layers: Number of recurrent layers
            feature_dim: Output feature dimension
            dropout: Dropout rate
            bidirectional: Whether to use bidirectional RNN
            network_type: 'lstm' or 'gru'
        """
        super().__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.network_type = network_type.lower()
        
        # Recurrent layer
        if self.network_type == 'lstm':
            self.rnn = nn.LSTM(
                input_size=input_size,
                hidden_size=hidden_size,
                num_layers=num_layers,
                dropout=dropout if num_layers > 1 else 0,
                bidirectional=bidirectional,
                batch_first=True
            )
        elif self.network_type == 'gru':
            self.rnn = nn.GRU(
                input_size=input_size,
                hidden_size=hidden_size,
                num_layers=num_layers,
                dropout=dropout if num_layers > 1 else 0,
                bidirectional=bidirectional,
                batch_first=True
            )
        else:
            raise ValueError(f"Unsupported network type: {network_type}")
        
        # Calculate RNN output size
        rnn_output_size = hidden_size * (2 if bidirectional else 1)
        
        # Attention mechanism
        self.attention = nn.Sequential(
            nn.Linear(rnn_output_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1),
            nn.Softmax(dim=1)
        )
        
        # Feature extraction head
        self.feature_head = nn.Sequential(
            nn.Linear(rnn_output_size, hidden_size),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, feature_dim),
            nn.ReLU(inplace=True),
            # Additional layer for precursor pattern detection
            nn.Linear(feature_dim, feature_dim),
            nn.Tanh()  # Normalize features to [-1, 1] range
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the network.
        
        Args:
            x: Input tensor with shape (batch_size, sequence_length, input_size)
            
        Returns:
            Feature tensor with shape (batch_size, feature_dim)
        """
        # RNN forward pass
        rnn_output, _ = self.rnn(x)
        
        # Apply attention mechanism
        attention_weights = self.attention(rnn_output)
        attended_output = torch.sum(rnn_output * attention_weights, dim=1)
        
        # Extract features
        features = self.feature_head(attended_output)
        
        return features