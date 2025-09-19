"""Multi-modal feature fusion module for combining features from different data modalities."""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, Any, Optional, Union, List, Tuple
import warnings
import os
import joblib
from sklearn.preprocessing import StandardScaler

from ...data.base import BaseFeatureExtractor
from ...data.schemas import RockfallDataPoint


class AttentionMechanism(nn.Module):
    """Attention mechanism to weight feature importance across modalities."""
    
    def __init__(self, feature_dims: Dict[str, int], hidden_dim: int = 128):
        """Initialize attention mechanism.
        
        Args:
            feature_dims: Dictionary mapping modality names to feature dimensions
            hidden_dim: Hidden dimension for attention computation
        """
        super().__init__()
        self.feature_dims = feature_dims
        self.hidden_dim = hidden_dim
        
        # Create attention networks for each modality
        self.attention_networks = nn.ModuleDict()
        for modality, dim in feature_dims.items():
            self.attention_networks[modality] = nn.Sequential(
                nn.Linear(dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, 1),
                nn.Sigmoid()
            )
        
        # Global attention to weight modalities
        self.global_attention = nn.Sequential(
            nn.Linear(len(feature_dims), hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, len(feature_dims)),
            nn.Softmax(dim=-1)
        )
    
    def forward(self, features: Dict[str, torch.Tensor]) -> Tuple[Dict[str, torch.Tensor], torch.Tensor]:
        """Apply attention mechanism to features.
        
        Args:
            features: Dictionary of feature tensors for each modality
            
        Returns:
            Tuple of (weighted_features, attention_weights)
        """
        weighted_features = {}
        modality_scores = []
        
        # Compute attention weights for each modality
        for modality, feature_tensor in features.items():
            if modality in self.attention_networks:
                attention_weight = self.attention_networks[modality](feature_tensor)
                weighted_features[modality] = feature_tensor * attention_weight
                modality_scores.append(attention_weight.mean())
            else:
                weighted_features[modality] = feature_tensor
                modality_scores.append(torch.tensor(1.0))
        
        # Compute global attention weights across modalities
        if modality_scores and len(modality_scores) == len(self.feature_dims):
            modality_scores_tensor = torch.stack(modality_scores)
            global_weights = self.global_attention(modality_scores_tensor.unsqueeze(0))
        else:
            # Handle case where not all modalities are present
            num_present = len(modality_scores) if modality_scores else len(features)
            global_weights = torch.ones(1, num_present) / num_present if num_present > 0 else torch.ones(1, 1)
        
        return weighted_features, global_weights


class MultiModalFusion(BaseFeatureExtractor):
    """Multi-modal feature fusion module to combine features from all data modalities."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize multi-modal fusion module.
        
        Args:
            config: Configuration dictionary with parameters:
                - fusion_method: 'concatenation', 'attention', 'weighted_sum' (default: 'attention')
                - output_dim: Output feature dimension (default: 256)
                - hidden_dim: Hidden dimension for fusion networks (default: 128)
                - dropout_rate: Dropout rate (default: 0.3)
                - normalize_features: Whether to normalize input features (default: True)
                - handle_missing: How to handle missing modalities ('zero', 'mean', 'drop') (default: 'zero')
                - device: 'cpu' or 'cuda' (default: auto-detect)
        """
        super().__init__(config)
        
        # Default configuration
        default_config = {
            'fusion_method': 'attention',
            'output_dim': 256,
            'hidden_dim': 128,
            'dropout_rate': 0.3,
            'normalize_features': True,
            'handle_missing': 'zero',
            'device': 'cuda' if torch.cuda.is_available() else 'cpu'
        }
        self.config = {**default_config, **self.config}
        
        # Initialize components
        self.device = torch.device(self.config['device'])
        self.feature_dims = {}
        self.scalers = {}
        self.fusion_network = None
        self.attention_mechanism = None
        self._is_fitted = False
        self._feature_dim = self.config['output_dim']
        
        # Expected modalities
        self.modalities = ['image', 'temporal', 'tabular', 'terrain', 'seismic']
    
    def _initialize_scalers(self):
        """Initialize feature scalers for each modality."""
        if self.config['normalize_features']:
            for modality in self.modalities:
                self.scalers[modality] = StandardScaler()
    
    def _build_fusion_network(self):
        """Build the fusion network based on configuration."""
        if not self.feature_dims:
            raise ValueError("Feature dimensions not set. Call fit() first.")
        
        total_input_dim = sum(self.feature_dims.values())
        
        if self.config['fusion_method'] == 'attention':
            # Build attention mechanism
            self.attention_mechanism = AttentionMechanism(
                self.feature_dims, 
                self.config['hidden_dim']
            )
            
            # Fusion network after attention
            self.fusion_network = nn.Sequential(
                nn.Linear(total_input_dim, self.config['hidden_dim']),
                nn.ReLU(),
                nn.Dropout(self.config['dropout_rate']),
                nn.Linear(self.config['hidden_dim'], self.config['hidden_dim'] // 2),
                nn.ReLU(),
                nn.Dropout(self.config['dropout_rate']),
                nn.Linear(self.config['hidden_dim'] // 2, self.config['output_dim']),
                nn.Tanh()  # Normalize output to [-1, 1]
            )
            
        elif self.config['fusion_method'] == 'concatenation':
            # Simple concatenation with MLP
            self.fusion_network = nn.Sequential(
                nn.Linear(total_input_dim, self.config['hidden_dim'] * 2),
                nn.ReLU(),
                nn.Dropout(self.config['dropout_rate']),
                nn.Linear(self.config['hidden_dim'] * 2, self.config['hidden_dim']),
                nn.ReLU(),
                nn.Dropout(self.config['dropout_rate']),
                nn.Linear(self.config['hidden_dim'], self.config['output_dim']),
                nn.Tanh()
            )
            
        elif self.config['fusion_method'] == 'weighted_sum':
            # Learnable weighted sum
            self.modality_weights = nn.Parameter(torch.ones(len(self.feature_dims)))
            self.fusion_network = nn.Sequential(
                nn.Linear(max(self.feature_dims.values()), self.config['hidden_dim']),
                nn.ReLU(),
                nn.Dropout(self.config['dropout_rate']),
                nn.Linear(self.config['hidden_dim'], self.config['output_dim']),
                nn.Tanh()
            )
        
        # Move to device
        if self.attention_mechanism:
            self.attention_mechanism.to(self.device)
        self.fusion_network.to(self.device)
    
    def _prepare_features(self, features: Dict[str, np.ndarray]) -> Dict[str, torch.Tensor]:
        """Prepare and normalize features for fusion.
        
        Args:
            features: Dictionary of feature arrays for each modality
            
        Returns:
            Dictionary of normalized feature tensors
        """
        prepared_features = {}
        
        for modality, feature_array in features.items():
            if feature_array is None:
                continue
                
            # Handle different input shapes
            if feature_array.ndim == 1:
                feature_array = feature_array.reshape(1, -1)
            elif feature_array.ndim > 2:
                feature_array = feature_array.reshape(feature_array.shape[0], -1)
            
            # Normalize features if scaler is available and fitted
            if (self.config['normalize_features'] and 
                modality in self.scalers and 
                hasattr(self.scalers[modality], 'mean_')):
                try:
                    feature_array = self.scalers[modality].transform(feature_array)
                except Exception as e:
                    warnings.warn(f"Failed to normalize {modality} features: {e}")
            
            # Convert to tensor
            feature_tensor = torch.FloatTensor(feature_array).to(self.device)
            prepared_features[modality] = feature_tensor
        
        return prepared_features
    
    def _handle_missing_modalities(self, features: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Handle missing modalities based on configuration.
        
        Args:
            features: Dictionary of available feature tensors
            
        Returns:
            Dictionary with missing modalities handled
        """
        complete_features = {}
        
        # First, add all available features
        for modality, feature_tensor in features.items():
            if feature_tensor is not None:
                complete_features[modality] = feature_tensor
        
        # If no features are available at all, return empty dict
        if not complete_features:
            return complete_features
        
        # Handle missing modalities only if we have some features
        for modality in self.modalities:
            if modality not in complete_features:
                # Handle missing modality
                if self.config['handle_missing'] == 'zero':
                    # Use zero features
                    if modality in self.feature_dims:
                        batch_size = next(iter(complete_features.values())).shape[0]
                        zero_features = torch.zeros(
                            batch_size, self.feature_dims[modality], 
                            device=self.device
                        )
                        complete_features[modality] = zero_features
                elif self.config['handle_missing'] == 'mean':
                    # Use mean features (if available from training)
                    if hasattr(self, f'_mean_features_{modality}'):
                        mean_features = getattr(self, f'_mean_features_{modality}')
                        batch_size = next(iter(complete_features.values())).shape[0]
                        mean_tensor = torch.FloatTensor(mean_features).to(self.device)
                        complete_features[modality] = mean_tensor.repeat(batch_size, 1)
                # For 'drop' method, we simply don't add the missing modality
        
        return complete_features
    
    def fit(self, feature_data: List[Dict[str, np.ndarray]]) -> 'MultiModalFusion':
        """Fit the fusion module to training data.
        
        Args:
            feature_data: List of dictionaries containing features for each modality
            
        Returns:
            Self for method chaining
        """
        # Initialize scalers
        self._initialize_scalers()
        
        # Collect features by modality for fitting scalers and determining dimensions
        modality_features = {modality: [] for modality in self.modalities}
        
        for sample_features in feature_data:
            for modality, features in sample_features.items():
                if features is not None and modality in self.modalities:
                    # Ensure 2D array
                    if features.ndim == 1:
                        features = features.reshape(1, -1)
                    elif features.ndim > 2:
                        features = features.reshape(features.shape[0], -1)
                    
                    modality_features[modality].append(features)
        
        # Determine feature dimensions and fit scalers
        for modality in self.modalities:
            if modality_features[modality]:
                # Concatenate all features for this modality
                all_features = np.vstack(modality_features[modality])
                self.feature_dims[modality] = all_features.shape[1]
                
                # Fit scaler
                if self.config['normalize_features'] and modality in self.scalers:
                    self.scalers[modality].fit(all_features)
                
                # Store mean features for missing modality handling
                if self.config['handle_missing'] == 'mean':
                    setattr(self, f'_mean_features_{modality}', np.mean(all_features, axis=0))
        
        # Build fusion network
        self._build_fusion_network()
        self._is_fitted = True
        
        return self
    
    def extract_features(self, features: Dict[str, np.ndarray]) -> np.ndarray:
        """Extract unified features from multi-modal input.
        
        Args:
            features: Dictionary containing features for available modalities
            
        Returns:
            Unified feature representation as numpy array
        """
        if not self._is_fitted:
            raise ValueError("MultiModalFusion must be fitted before extracting features")
        
        # Prepare features
        prepared_features = self._prepare_features(features)
        
        # Handle missing modalities
        complete_features = self._handle_missing_modalities(prepared_features)
        
        if not complete_features:
            # Return zero features if no modalities available
            return np.zeros(self.config['output_dim'])
        
        # Apply fusion method
        with torch.no_grad():
            if self.config['fusion_method'] == 'attention':
                if complete_features:
                    # Apply attention mechanism
                    weighted_features, attention_weights = self.attention_mechanism(complete_features)
                    
                    # Concatenate weighted features
                    feature_list = []
                    for modality in self.modalities:
                        if modality in weighted_features:
                            feature_list.append(weighted_features[modality])
                    
                    if feature_list:
                        concatenated = torch.cat(feature_list, dim=-1)
                        fused_features = self.fusion_network(concatenated)
                    else:
                        fused_features = torch.zeros(1, self.config['output_dim'], device=self.device)
                else:
                    fused_features = torch.zeros(1, self.config['output_dim'], device=self.device)
                    
            elif self.config['fusion_method'] == 'concatenation':
                if complete_features:
                    # Simple concatenation
                    feature_list = []
                    for modality in self.modalities:
                        if modality in complete_features:
                            feature_list.append(complete_features[modality])
                    
                    if feature_list:
                        concatenated = torch.cat(feature_list, dim=-1)
                        fused_features = self.fusion_network(concatenated)
                    else:
                        fused_features = torch.zeros(1, self.config['output_dim'], device=self.device)
                else:
                    fused_features = torch.zeros(1, self.config['output_dim'], device=self.device)
                    
            elif self.config['fusion_method'] == 'weighted_sum':
                if complete_features and self.feature_dims:
                    # Weighted sum of features
                    weighted_sum = torch.zeros(1, max(self.feature_dims.values()), device=self.device)
                    total_weight = 0
                    
                    for i, modality in enumerate(self.modalities):
                        if modality in complete_features:
                            # Pad or truncate to max dimension
                            features = complete_features[modality]
                            max_dim = max(self.feature_dims.values())
                            
                            if features.shape[1] < max_dim:
                                # Pad with zeros
                                padding = torch.zeros(features.shape[0], max_dim - features.shape[1], device=self.device)
                                features = torch.cat([features, padding], dim=1)
                            elif features.shape[1] > max_dim:
                                # Truncate
                                features = features[:, :max_dim]
                            
                            weight = torch.softmax(self.modality_weights, dim=0)[i]
                            weighted_sum += weight * features
                            total_weight += weight
                    
                    if total_weight > 0:
                        weighted_sum = weighted_sum / total_weight
                    
                    fused_features = self.fusion_network(weighted_sum)
                else:
                    fused_features = torch.zeros(1, self.config['output_dim'], device=self.device)
        
        return fused_features.cpu().numpy().flatten()
    
    def extract_batch_features(self, feature_batch: List[Dict[str, np.ndarray]]) -> np.ndarray:
        """Extract features from a batch of multi-modal inputs.
        
        Args:
            feature_batch: List of feature dictionaries
            
        Returns:
            Batch of unified feature representations
        """
        batch_features = []
        
        for features in feature_batch:
            try:
                # Check if features contain None values
                valid_features = {k: v for k, v in features.items() if v is not None}
                if not valid_features:
                    # All features are None, warn and use zero features
                    warnings.warn("All features are None for sample")
                    batch_features.append(np.zeros(self.config['output_dim']))
                else:
                    unified_features = self.extract_features(valid_features)
                    batch_features.append(unified_features)
            except Exception as e:
                warnings.warn(f"Failed to process sample: {e}")
                # Add zero features for failed samples
                batch_features.append(np.zeros(self.config['output_dim']))
        
        return np.array(batch_features)
    
    def get_attention_weights(self, features: Dict[str, np.ndarray]) -> Optional[Dict[str, float]]:
        """Get attention weights for each modality.
        
        Args:
            features: Dictionary containing features for available modalities
            
        Returns:
            Dictionary of attention weights or None if attention not used
        """
        if (not self._is_fitted or 
            self.config['fusion_method'] != 'attention' or 
            self.attention_mechanism is None):
            return None
        
        # Prepare features
        prepared_features = self._prepare_features(features)
        complete_features = self._handle_missing_modalities(prepared_features)
        
        if not complete_features:
            return None
        
        with torch.no_grad():
            _, attention_weights = self.attention_mechanism(complete_features)
            
            # Convert to dictionary
            weights_dict = {}
            for i, modality in enumerate(self.modalities):
                if modality in complete_features and i < attention_weights.shape[1]:
                    weights_dict[modality] = float(attention_weights[0, i])
        
        return weights_dict
    
    def get_modality_contributions(self, features: Dict[str, np.ndarray]) -> Dict[str, float]:
        """Get contribution scores for each modality.
        
        Args:
            features: Dictionary containing features for available modalities
            
        Returns:
            Dictionary of contribution scores
        """
        if not self._is_fitted:
            raise ValueError("MultiModalFusion must be fitted before getting contributions")
        
        contributions = {}
        
        # Get baseline prediction with all modalities
        baseline_features = self.extract_features(features)
        baseline_norm = np.linalg.norm(baseline_features)
        
        # Test contribution of each modality by removing it
        for modality in features.keys():
            if features[modality] is not None:
                # Create features without this modality
                reduced_features = {k: v for k, v in features.items() if k != modality}
                
                if reduced_features:
                    reduced_output = self.extract_features(reduced_features)
                    reduced_norm = np.linalg.norm(reduced_output)
                    
                    # Contribution is the difference in output magnitude
                    contribution = abs(baseline_norm - reduced_norm) / (baseline_norm + 1e-8)
                    contributions[modality] = float(contribution)
                else:
                    contributions[modality] = 1.0  # Only modality available
        
        # Normalize contributions to sum to 1
        total_contribution = sum(contributions.values())
        if total_contribution > 0:
            contributions = {k: v / total_contribution for k, v in contributions.items()}
        
        return contributions
    
    def save_model(self, model_path: str):
        """Save the fusion model to disk.
        
        Args:
            model_path: Path to save the model
        """
        if not self._is_fitted:
            raise ValueError("Cannot save unfitted model")
        
        model_dir = os.path.dirname(model_path)
        if model_dir:
            os.makedirs(model_dir, exist_ok=True)
        
        # Prepare save dictionary
        save_dict = {
            'config': self.config,
            'feature_dims': self.feature_dims,
            'scalers': self.scalers,
            'is_fitted': self._is_fitted
        }
        
        # Save model state dicts
        if self.fusion_network:
            save_dict['fusion_network_state'] = self.fusion_network.state_dict()
        if self.attention_mechanism:
            save_dict['attention_mechanism_state'] = self.attention_mechanism.state_dict()
        
        # Save mean features for missing modality handling
        for modality in self.modalities:
            if hasattr(self, f'_mean_features_{modality}'):
                save_dict[f'mean_features_{modality}'] = getattr(self, f'_mean_features_{modality}')
        
        torch.save(save_dict, model_path)
    
    def load_model(self, model_path: str):
        """Load a fusion model from disk.
        
        Args:
            model_path: Path to load the model from
        """
        checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)
        
        # Update configuration
        self.config.update(checkpoint['config'])
        self.feature_dims = checkpoint['feature_dims']
        self.scalers = checkpoint['scalers']
        self._is_fitted = checkpoint['is_fitted']
        self._feature_dim = self.config['output_dim']
        
        # Rebuild networks
        self._build_fusion_network()
        
        # Load model states
        if 'fusion_network_state' in checkpoint and self.fusion_network:
            self.fusion_network.load_state_dict(checkpoint['fusion_network_state'])
        if 'attention_mechanism_state' in checkpoint and self.attention_mechanism:
            self.attention_mechanism.load_state_dict(checkpoint['attention_mechanism_state'])
        
        # Load mean features
        for modality in self.modalities:
            if f'mean_features_{modality}' in checkpoint:
                setattr(self, f'_mean_features_{modality}', checkpoint[f'mean_features_{modality}'])
        
        # Set to evaluation mode
        if self.fusion_network:
            self.fusion_network.eval()
        if self.attention_mechanism:
            self.attention_mechanism.eval()
        
        # Set device
        self.device = torch.device(self.config['device'])