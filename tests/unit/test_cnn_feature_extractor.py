"""Unit tests for CNN Feature Extractor."""

import pytest
import torch
import numpy as np
from PIL import Image
import tempfile
import os
from unittest.mock import patch, MagicMock

from src.models.extractors.cnn_feature_extractor import CNNFeatureExtractor
from src.data.schemas import ImageData, ImageMetadata, BoundingBox


class TestCNNFeatureExtractor:
    """Test cases for CNNFeatureExtractor class."""
    
    @pytest.fixture
    def sample_config(self):
        """Sample configuration for testing."""
        return {
            'backbone': 'resnet50',
            'feature_dim': 256,
            'pretrained': False,  # Use False for faster testing
            'freeze_backbone': True,
            'input_size': 224,
            'device': 'cpu'  # Force CPU for testing
        }
    
    @pytest.fixture
    def sample_image(self):
        """Create a sample RGB image for testing."""
        # Create a 224x224 RGB image with random values
        image_array = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
        return Image.fromarray(image_array)
    
    @pytest.fixture
    def sample_image_path(self, sample_image):
        """Create a temporary image file for testing."""
        with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as tmp_file:
            sample_image.save(tmp_file.name)
            yield tmp_file.name
        os.unlink(tmp_file.name)
    
    @pytest.fixture
    def sample_image_data(self, sample_image_path):
        """Create sample ImageData object."""
        metadata = ImageMetadata(width=224, height=224, channels=3)
        annotations = [
            BoundingBox(x=10, y=10, width=50, height=50, class_id=1, confidence=0.9)
        ]
        return ImageData(
            image_path=sample_image_path,
            annotations=annotations,
            metadata=metadata
        )
    
    def test_initialization_default_config(self):
        """Test CNN feature extractor initialization with default config."""
        extractor = CNNFeatureExtractor()
        
        assert extractor.config['backbone'] == 'resnet50'
        assert extractor.config['feature_dim'] == 512
        assert extractor.config['pretrained'] == True
        assert extractor.config['freeze_backbone'] == True
        assert extractor.config['input_size'] == 224
        assert extractor.feature_dim == 512
        assert extractor.model is not None
        assert extractor.backbone is not None
        assert extractor.custom_head is not None
    
    def test_initialization_custom_config(self, sample_config):
        """Test CNN feature extractor initialization with custom config."""
        extractor = CNNFeatureExtractor(sample_config)
        
        assert extractor.config['backbone'] == 'resnet50'
        assert extractor.config['feature_dim'] == 256
        assert extractor.config['pretrained'] == False
        assert extractor.feature_dim == 256
        assert extractor.device.type == 'cpu'
    
    def test_initialization_efficientnet(self):
        """Test initialization with EfficientNet backbone."""
        config = {
            'backbone': 'efficientnet_b0',
            'pretrained': False,
            'device': 'cpu'
        }
        extractor = CNNFeatureExtractor(config)
        
        assert extractor.config['backbone'] == 'efficientnet_b0'
        assert extractor.model is not None
    
    def test_initialization_invalid_backbone(self):
        """Test initialization with invalid backbone raises error."""
        config = {'backbone': 'invalid_backbone'}
        
        with pytest.raises(ValueError, match="Unsupported backbone"):
            CNNFeatureExtractor(config)
    
    def test_extract_features_from_pil_image(self, sample_config, sample_image):
        """Test feature extraction from PIL Image."""
        extractor = CNNFeatureExtractor(sample_config)
        features = extractor.extract_features(sample_image)
        
        assert isinstance(features, np.ndarray)
        assert features.shape == (256,)  # feature_dim from config
        assert not np.isnan(features).any()
        assert not np.isinf(features).any()
    
    def test_extract_features_from_file_path(self, sample_config, sample_image_path):
        """Test feature extraction from image file path."""
        extractor = CNNFeatureExtractor(sample_config)
        features = extractor.extract_features(sample_image_path)
        
        assert isinstance(features, np.ndarray)
        assert features.shape == (256,)
        assert not np.isnan(features).any()
    
    def test_extract_features_from_numpy_array(self, sample_config):
        """Test feature extraction from numpy array."""
        extractor = CNNFeatureExtractor(sample_config)
        image_array = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
        features = extractor.extract_features(image_array)
        
        assert isinstance(features, np.ndarray)
        assert features.shape == (256,)
        assert not np.isnan(features).any()
    
    def test_extract_features_from_image_data(self, sample_config, sample_image_data):
        """Test feature extraction from ImageData object."""
        extractor = CNNFeatureExtractor(sample_config)
        features = extractor.extract_features(sample_image_data)
        
        assert isinstance(features, np.ndarray)
        assert features.shape == (256,)
        assert not np.isnan(features).any()
    
    def test_extract_features_invalid_input(self, sample_config):
        """Test feature extraction with invalid input type."""
        extractor = CNNFeatureExtractor(sample_config)
        
        with pytest.raises(ValueError, match="Unsupported input type"):
            extractor.extract_features(123)  # Invalid input type
    
    def test_extract_batch_features(self, sample_config, sample_image_path):
        """Test batch feature extraction."""
        extractor = CNNFeatureExtractor(sample_config)
        
        # Create multiple copies of the same image path for testing
        image_paths = [sample_image_path] * 3
        features = extractor.extract_batch_features(image_paths)
        
        assert isinstance(features, np.ndarray)
        assert features.shape == (3, 256)  # 3 images, 256 features each
        assert not np.isnan(features).any()
    
    def test_extract_batch_features_with_invalid_path(self, sample_config, sample_image_path):
        """Test batch feature extraction with some invalid paths."""
        extractor = CNNFeatureExtractor(sample_config)
        
        image_paths = [sample_image_path, 'invalid_path.jpg', sample_image_path]
        
        # Capture print output to verify warning
        with patch('builtins.print') as mock_print:
            features = extractor.extract_batch_features(image_paths)
        
        assert isinstance(features, np.ndarray)
        assert features.shape == (3, 256)
        # Check that warning was printed for invalid path
        mock_print.assert_called()
        
        # Second row should be zeros due to failed processing
        assert np.allclose(features[1], np.zeros(256))
    
    def test_save_and_load_features(self, sample_config):
        """Test saving and loading features."""
        extractor = CNNFeatureExtractor(sample_config)
        features = np.random.rand(10, 256)
        
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = os.path.join(tmp_dir, 'features.pkl')
            
            # Test saving
            extractor.save_features(features, tmp_path)
            assert os.path.exists(tmp_path)
            
            # Test loading
            loaded_features = extractor.load_features(tmp_path)
            np.testing.assert_array_equal(features, loaded_features)
    
    def test_save_and_load_model(self, sample_config):
        """Test saving and loading model."""
        extractor = CNNFeatureExtractor(sample_config)
        
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = os.path.join(tmp_dir, 'model.pth')
            
            # Test saving
            extractor.save_model(tmp_path)
            assert os.path.exists(tmp_path)
            
            # Create new extractor and load model
            new_extractor = CNNFeatureExtractor()
            new_extractor.load_model(tmp_path)
            
            # Verify config was loaded correctly
            assert new_extractor.config['feature_dim'] == 256
            assert new_extractor.feature_dim == 256
    
    def test_model_architecture_resnet(self, sample_config):
        """Test that ResNet architecture is built correctly."""
        extractor = CNNFeatureExtractor(sample_config)
        
        # Test that model can process input
        dummy_input = torch.randn(1, 3, 224, 224)
        with torch.no_grad():
            output = extractor.model(dummy_input)
        
        assert output.shape == (1, 256)
        assert not torch.isnan(output).any()
    
    def test_model_architecture_efficientnet(self):
        """Test that EfficientNet architecture is built correctly."""
        config = {
            'backbone': 'efficientnet_b0',
            'feature_dim': 128,
            'pretrained': False,
            'device': 'cpu'
        }
        extractor = CNNFeatureExtractor(config)
        
        # Test that model can process input
        dummy_input = torch.randn(1, 3, 224, 224)
        with torch.no_grad():
            output = extractor.model(dummy_input)
        
        assert output.shape == (1, 128)
        assert not torch.isnan(output).any()
    
    def test_freeze_backbone_functionality(self, sample_config):
        """Test that backbone freezing works correctly."""
        extractor = CNNFeatureExtractor(sample_config)
        
        # Check that backbone parameters are frozen
        backbone_params = list(extractor.backbone.parameters())
        for param in backbone_params:
            assert not param.requires_grad
        
        # Check that custom head parameters are not frozen
        head_params = list(extractor.custom_head.parameters())
        for param in head_params:
            assert param.requires_grad
    
    def test_feature_importance(self, sample_config, sample_image):
        """Test feature importance calculation."""
        extractor = CNNFeatureExtractor(sample_config)
        importance_scores = extractor.get_feature_importance(sample_image)
        
        assert isinstance(importance_scores, np.ndarray)
        assert importance_scores.shape == (3,)  # RGB channels
        assert not np.isnan(importance_scores).any()
        assert (importance_scores >= 0).all()  # Gradients should be non-negative
    
    def test_fine_tune_method_exists(self, sample_config):
        """Test that fine_tune method exists and can be called."""
        extractor = CNNFeatureExtractor(sample_config)
        
        # Create mock data loaders
        mock_train_loader = MagicMock()
        mock_val_loader = MagicMock()
        mock_train_loader.__len__ = MagicMock(return_value=1)
        mock_train_loader.__iter__ = MagicMock(return_value=iter([
            (torch.randn(2, 3, 224, 224), torch.randn(2, 256))
        ]))
        
        # Test that method can be called without errors
        try:
            extractor.fine_tune(mock_train_loader, mock_val_loader, num_epochs=1)
        except Exception as e:
            # Method should exist and be callable
            assert False, f"fine_tune method failed: {e}"
    
    def test_get_config(self, sample_config):
        """Test get_config method."""
        extractor = CNNFeatureExtractor(sample_config)
        config = extractor.get_config()
        
        assert isinstance(config, dict)
        assert config['feature_dim'] == 256
        assert config['backbone'] == 'resnet50'
        
        # Ensure it's a copy, not reference
        config['feature_dim'] = 999
        assert extractor.config['feature_dim'] == 256
    
    def test_feature_consistency(self, sample_config, sample_image):
        """Test that same image produces consistent features."""
        extractor = CNNFeatureExtractor(sample_config)
        
        features1 = extractor.extract_features(sample_image)
        features2 = extractor.extract_features(sample_image)
        
        # Features should be identical for same input
        np.testing.assert_array_equal(features1, features2)
    
    def test_different_input_sizes(self, sample_config):
        """Test that different input image sizes are handled correctly."""
        extractor = CNNFeatureExtractor(sample_config)
        
        # Test with different sized images
        small_image = Image.fromarray(np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8))
        large_image = Image.fromarray(np.random.randint(0, 255, (500, 500, 3), dtype=np.uint8))
        
        features_small = extractor.extract_features(small_image)
        features_large = extractor.extract_features(large_image)
        
        # Both should produce same feature dimension
        assert features_small.shape == features_large.shape == (256,)
        assert not np.isnan(features_small).any()
        assert not np.isnan(features_large).any()


if __name__ == '__main__':
    pytest.main([__file__])