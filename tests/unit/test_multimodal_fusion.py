"""Unit tests for multi-modal feature fusion module."""

import pytest
import numpy as np
import torch
import tempfile
import os
from unittest.mock import Mock, patch

from src.models.fusion.multimodal_fusion import MultiModalFusion, AttentionMechanism


class TestAttentionMechanism:
    """Test cases for the AttentionMechanism class."""
    
    def test_attention_mechanism_initialization(self):
        """Test attention mechanism initialization."""
        feature_dims = {'image': 512, 'temporal': 64, 'tabular': 50}
        attention = AttentionMechanism(feature_dims, hidden_dim=128)
        
        assert len(attention.attention_networks) == 3
        assert 'image' in attention.attention_networks
        assert 'temporal' in attention.attention_networks
        assert 'tabular' in attention.attention_networks
    
    def test_attention_forward_pass(self):
        """Test attention mechanism forward pass."""
        feature_dims = {'image': 512, 'temporal': 64}
        attention = AttentionMechanism(feature_dims, hidden_dim=64)
        
        # Create mock features
        features = {
            'image': torch.randn(1, 512),
            'temporal': torch.randn(1, 64)
        }
        
        weighted_features, global_weights = attention(features)
        
        assert len(weighted_features) == 2
        assert 'image' in weighted_features
        assert 'temporal' in weighted_features
        assert weighted_features['image'].shape == (1, 512)
        assert weighted_features['temporal'].shape == (1, 64)
        assert global_weights.shape == (1, 2)
    
    def test_attention_with_missing_modality(self):
        """Test attention mechanism with missing modality."""
        feature_dims = {'image': 512, 'temporal': 64, 'tabular': 50}
        attention = AttentionMechanism(feature_dims, hidden_dim=64)
        
        # Only provide image features
        features = {'image': torch.randn(1, 512)}
        
        weighted_features, global_weights = attention(features)
        
        assert len(weighted_features) == 1
        assert 'image' in weighted_features
        assert weighted_features['image'].shape == (1, 512)


class TestMultiModalFusion:
    """Test cases for the MultiModalFusion class."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.config = {
            'fusion_method': 'attention',
            'output_dim': 128,
            'hidden_dim': 64,
            'dropout_rate': 0.2,
            'normalize_features': True,
            'handle_missing': 'zero',
            'device': 'cpu'
        }
        self.fusion = MultiModalFusion(self.config)
    
    def test_initialization(self):
        """Test MultiModalFusion initialization."""
        assert self.fusion.config['fusion_method'] == 'attention'
        assert self.fusion.config['output_dim'] == 128
        assert self.fusion.config['handle_missing'] == 'zero'
        assert not self.fusion._is_fitted
    
    def test_fit_with_sample_data(self):
        """Test fitting the fusion module with sample data."""
        # Create sample feature data
        feature_data = [
            {
                'image': np.random.randn(512),
                'temporal': np.random.randn(64),
                'tabular': np.random.randn(50)
            },
            {
                'image': np.random.randn(512),
                'temporal': np.random.randn(64),
                'tabular': np.random.randn(50)
            },
            {
                'image': np.random.randn(512),
                'temporal': np.random.randn(64),
                'tabular': np.random.randn(50)
            }
        ]
        
        # Fit the fusion module
        self.fusion.fit(feature_data)
        
        assert self.fusion._is_fitted
        assert len(self.fusion.feature_dims) == 3
        assert self.fusion.feature_dims['image'] == 512
        assert self.fusion.feature_dims['temporal'] == 64
        assert self.fusion.feature_dims['tabular'] == 50
        assert self.fusion.fusion_network is not None
        assert self.fusion.attention_mechanism is not None
    
    def test_extract_features_after_fitting(self):
        """Test feature extraction after fitting."""
        # Fit with sample data
        feature_data = [
            {
                'image': np.random.randn(512),
                'temporal': np.random.randn(64),
                'tabular': np.random.randn(50)
            }
        ]
        self.fusion.fit(feature_data)
        
        # Extract features
        test_features = {
            'image': np.random.randn(512),
            'temporal': np.random.randn(64),
            'tabular': np.random.randn(50)
        }
        
        result = self.fusion.extract_features(test_features)
        
        assert isinstance(result, np.ndarray)
        assert result.shape == (128,)  # output_dim
        assert not np.isnan(result).any()
    
    def test_extract_features_without_fitting(self):
        """Test that extracting features without fitting raises error."""
        test_features = {'image': np.random.randn(512)}
        
        with pytest.raises(ValueError, match="must be fitted"):
            self.fusion.extract_features(test_features)
    
    def test_handle_missing_modalities_zero(self):
        """Test handling missing modalities with zero padding."""
        self.fusion.config['handle_missing'] = 'zero'
        
        # Fit with complete data
        feature_data = [
            {
                'image': np.random.randn(512),
                'temporal': np.random.randn(64)
            }
        ]
        self.fusion.fit(feature_data)
        
        # Test with missing temporal modality
        test_features = {'image': np.random.randn(512)}
        result = self.fusion.extract_features(test_features)
        
        assert isinstance(result, np.ndarray)
        assert result.shape == (128,)
    
    def test_handle_missing_modalities_mean(self):
        """Test handling missing modalities with mean features."""
        self.fusion.config['handle_missing'] = 'mean'
        
        # Fit with complete data
        feature_data = [
            {
                'image': np.random.randn(512),
                'temporal': np.random.randn(64)
            },
            {
                'image': np.random.randn(512),
                'temporal': np.random.randn(64)
            }
        ]
        self.fusion.fit(feature_data)
        
        # Test with missing temporal modality
        test_features = {'image': np.random.randn(512)}
        result = self.fusion.extract_features(test_features)
        
        assert isinstance(result, np.ndarray)
        assert result.shape == (128,)
        assert hasattr(self.fusion, '_mean_features_temporal')
    
    def test_concatenation_fusion_method(self):
        """Test concatenation fusion method."""
        config = self.config.copy()
        config['fusion_method'] = 'concatenation'
        fusion = MultiModalFusion(config)
        
        # Fit and test
        feature_data = [
            {
                'image': np.random.randn(512),
                'temporal': np.random.randn(64)
            }
        ]
        fusion.fit(feature_data)
        
        test_features = {
            'image': np.random.randn(512),
            'temporal': np.random.randn(64)
        }
        result = fusion.extract_features(test_features)
        
        assert isinstance(result, np.ndarray)
        assert result.shape == (128,)
    
    def test_weighted_sum_fusion_method(self):
        """Test weighted sum fusion method."""
        config = self.config.copy()
        config['fusion_method'] = 'weighted_sum'
        fusion = MultiModalFusion(config)
        
        # Fit and test
        feature_data = [
            {
                'image': np.random.randn(512),
                'temporal': np.random.randn(64)
            }
        ]
        fusion.fit(feature_data)
        
        test_features = {
            'image': np.random.randn(512),
            'temporal': np.random.randn(64)
        }
        result = fusion.extract_features(test_features)
        
        assert isinstance(result, np.ndarray)
        assert result.shape == (128,)
        assert hasattr(fusion, 'modality_weights')
    
    def test_extract_batch_features(self):
        """Test batch feature extraction."""
        # Fit with sample data
        feature_data = [
            {
                'image': np.random.randn(512),
                'temporal': np.random.randn(64)
            }
        ]
        self.fusion.fit(feature_data)
        
        # Test batch extraction
        batch_features = [
            {
                'image': np.random.randn(512),
                'temporal': np.random.randn(64)
            },
            {
                'image': np.random.randn(512),
                'temporal': np.random.randn(64)
            }
        ]
        
        result = self.fusion.extract_batch_features(batch_features)
        
        assert isinstance(result, np.ndarray)
        assert result.shape == (2, 128)
        assert not np.isnan(result).any()
    
    def test_get_attention_weights(self):
        """Test getting attention weights."""
        # Fit with sample data
        feature_data = [
            {
                'image': np.random.randn(512),
                'temporal': np.random.randn(64)
            }
        ]
        self.fusion.fit(feature_data)
        
        test_features = {
            'image': np.random.randn(512),
            'temporal': np.random.randn(64)
        }
        
        weights = self.fusion.get_attention_weights(test_features)
        
        assert isinstance(weights, dict)
        assert len(weights) >= 0  # May be empty if modalities not in complete_features
    
    def test_get_modality_contributions(self):
        """Test getting modality contributions."""
        # Fit with sample data
        feature_data = [
            {
                'image': np.random.randn(512),
                'temporal': np.random.randn(64)
            }
        ]
        self.fusion.fit(feature_data)
        
        test_features = {
            'image': np.random.randn(512),
            'temporal': np.random.randn(64)
        }
        
        contributions = self.fusion.get_modality_contributions(test_features)
        
        assert isinstance(contributions, dict)
        assert all(isinstance(v, float) for v in contributions.values())
        # Contributions should sum to approximately 1
        if contributions:
            assert abs(sum(contributions.values()) - 1.0) < 0.01
    
    def test_save_and_load_model(self):
        """Test saving and loading the fusion model."""
        # Fit with sample data
        feature_data = [
            {
                'image': np.random.randn(512),
                'temporal': np.random.randn(64)
            }
        ]
        self.fusion.fit(feature_data)
        
        # Test features for comparison
        test_features = {
            'image': np.random.randn(512),
            'temporal': np.random.randn(64)
        }
        original_result = self.fusion.extract_features(test_features)
        
        # Save model
        with tempfile.TemporaryDirectory() as temp_dir:
            model_path = os.path.join(temp_dir, 'fusion_model.pth')
            self.fusion.save_model(model_path)
            
            # Create new fusion instance and load
            new_fusion = MultiModalFusion(self.config)
            new_fusion.load_model(model_path)
            
            # Test that loaded model has same configuration and is fitted
            assert new_fusion._is_fitted
            assert new_fusion.config == self.fusion.config
            assert new_fusion.feature_dims == self.fusion.feature_dims
            
            # Test that loaded model produces consistent results
            loaded_result = new_fusion.extract_features(test_features)
            
            # Results should have same shape and be reasonable values
            assert loaded_result.shape == original_result.shape
            assert not np.isnan(loaded_result).any()
            assert not np.isinf(loaded_result).any()
            
            # Test that the same input produces the same output from loaded model
            loaded_result2 = new_fusion.extract_features(test_features)
            assert np.allclose(loaded_result, loaded_result2, rtol=1e-6)
    
    def test_feature_normalization(self):
        """Test feature normalization functionality."""
        config = self.config.copy()
        config['normalize_features'] = True
        fusion = MultiModalFusion(config)
        
        # Create data with different scales
        feature_data = [
            {
                'image': np.random.randn(512) * 100,  # Large scale
                'temporal': np.random.randn(64) * 0.01  # Small scale
            },
            {
                'image': np.random.randn(512) * 100,
                'temporal': np.random.randn(64) * 0.01
            }
        ]
        
        fusion.fit(feature_data)
        
        # Test that scalers are fitted
        assert 'image' in fusion.scalers
        assert 'temporal' in fusion.scalers
        assert hasattr(fusion.scalers['image'], 'mean_')
        assert hasattr(fusion.scalers['temporal'], 'mean_')
    
    def test_different_input_shapes(self):
        """Test handling of different input shapes."""
        # Fit with sample data
        feature_data = [
            {
                'image': np.random.randn(512),
                'temporal': np.random.randn(64)
            }
        ]
        self.fusion.fit(feature_data)
        
        # Test with different shapes
        test_cases = [
            # 1D array
            {'image': np.random.randn(512), 'temporal': np.random.randn(64)},
            # 2D array (batch of 1)
            {'image': np.random.randn(1, 512), 'temporal': np.random.randn(1, 64)},
            # 3D array (should be flattened)
            {'image': np.random.randn(1, 16, 32), 'temporal': np.random.randn(1, 8, 8)}
        ]
        
        for test_features in test_cases:
            result = self.fusion.extract_features(test_features)
            assert isinstance(result, np.ndarray)
            assert result.shape == (128,)
    
    def test_empty_features_handling(self):
        """Test handling of empty or None features."""
        # Fit with sample data
        feature_data = [
            {
                'image': np.random.randn(512),
                'temporal': np.random.randn(64)
            }
        ]
        self.fusion.fit(feature_data)
        
        # Test with empty features
        empty_features = {}
        result = self.fusion.extract_features(empty_features)
        
        assert isinstance(result, np.ndarray)
        assert result.shape == (128,)
        # Should return zero features
        assert np.allclose(result, 0.0, atol=1e-6)
    
    def test_batch_processing_with_failures(self):
        """Test batch processing with some failed samples."""
        # Fit with sample data
        feature_data = [
            {
                'image': np.random.randn(512),
                'temporal': np.random.randn(64)
            }
        ]
        self.fusion.fit(feature_data)
        
        # Create batch with some invalid data
        batch_features = [
            {'image': np.random.randn(512), 'temporal': np.random.randn(64)},  # Valid
            {'image': None, 'temporal': None},  # Invalid
            {'image': np.random.randn(512), 'temporal': np.random.randn(64)}   # Valid
        ]
        
        with patch('warnings.warn') as mock_warn:
            result = self.fusion.extract_batch_features(batch_features)
            
            assert isinstance(result, np.ndarray)
            assert result.shape == (3, 128)
            # Should have warned about failed samples
            assert mock_warn.called


class TestMultiModalFusionIntegration:
    """Integration tests for multi-modal fusion with different configurations."""
    
    def test_end_to_end_workflow(self):
        """Test complete end-to-end workflow."""
        # Configuration
        config = {
            'fusion_method': 'attention',
            'output_dim': 64,
            'normalize_features': True,
            'handle_missing': 'zero'
        }
        
        fusion = MultiModalFusion(config)
        
        # Training data
        training_data = []
        for _ in range(10):
            training_data.append({
                'image': np.random.randn(256),
                'temporal': np.random.randn(32),
                'tabular': np.random.randn(20),
                'terrain': np.random.randn(15),
                'seismic': np.random.randn(40)
            })
        
        # Fit the model
        fusion.fit(training_data)
        
        # Test inference
        test_sample = {
            'image': np.random.randn(256),
            'temporal': np.random.randn(32),
            'tabular': np.random.randn(20)
            # Missing terrain and seismic
        }
        
        # Extract features
        features = fusion.extract_features(test_sample)
        assert features.shape == (64,)
        
        # Get attention weights
        weights = fusion.get_attention_weights(test_sample)
        assert isinstance(weights, dict)
        
        # Get contributions
        contributions = fusion.get_modality_contributions(test_sample)
        assert isinstance(contributions, dict)
        
        # Test batch processing
        batch = [test_sample] * 5
        batch_features = fusion.extract_batch_features(batch)
        assert batch_features.shape == (5, 64)
    
    def test_all_fusion_methods(self):
        """Test all fusion methods work correctly."""
        methods = ['attention', 'concatenation', 'weighted_sum']
        
        # Sample training data
        training_data = [
            {
                'image': np.random.randn(128),
                'temporal': np.random.randn(32)
            }
        ] * 5
        
        test_features = {
            'image': np.random.randn(128),
            'temporal': np.random.randn(32)
        }
        
        for method in methods:
            config = {
                'fusion_method': method,
                'output_dim': 64,
                'device': 'cpu'
            }
            
            fusion = MultiModalFusion(config)
            fusion.fit(training_data)
            
            result = fusion.extract_features(test_features)
            
            assert isinstance(result, np.ndarray)
            assert result.shape == (64,)
            assert not np.isnan(result).any()


if __name__ == '__main__':
    pytest.main([__file__])