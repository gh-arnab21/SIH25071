"""CNN Feature Extractor for mining imagery analysis."""

import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
from typing import Dict, Any, Optional, Union, List
import os
import joblib
from pathlib import Path

from ...data.base import BaseFeatureExtractor
from ...data.schemas import ImageData


class CNNFeatureExtractor(BaseFeatureExtractor):
    """CNN-based feature extractor for mining imagery using transfer learning."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize CNN feature extractor.
        
        Args:
            config: Configuration dictionary with parameters:
                - backbone: 'resnet50', 'resnet101', 'efficientnet_b0', etc.
                - feature_dim: Output feature dimension (default: 512)
                - pretrained: Whether to use pretrained weights (default: True)
                - freeze_backbone: Whether to freeze backbone weights (default: True)
                - input_size: Input image size (default: 224)
                - device: 'cpu' or 'cuda' (default: auto-detect)
        """
        super().__init__(config)
        
        # Default configuration
        default_config = {
            'backbone': 'resnet50',
            'feature_dim': 512,
            'pretrained': True,
            'freeze_backbone': True,
            'input_size': 224,
            'device': 'cuda' if torch.cuda.is_available() else 'cpu'
        }
        self.config = {**default_config, **self.config}
        
        # Initialize model components
        self.device = torch.device(self.config['device'])
        self.backbone = None
        self.custom_head = None
        self.model = None
        self._feature_dim = self.config['feature_dim']
        
        # Image preprocessing pipeline
        self.transform = transforms.Compose([
            transforms.Resize((self.config['input_size'], self.config['input_size'])),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
        
        self._build_model()
    
    def _build_model(self):
        """Build the CNN model with backbone and custom head."""
        # Load pre-trained backbone
        backbone_name = self.config['backbone'].lower()
        
        if 'resnet' in backbone_name:
            if backbone_name == 'resnet50':
                weights = models.ResNet50_Weights.IMAGENET1K_V1 if self.config['pretrained'] else None
                self.backbone = models.resnet50(weights=weights)
                backbone_features = 2048
            elif backbone_name == 'resnet101':
                weights = models.ResNet101_Weights.IMAGENET1K_V1 if self.config['pretrained'] else None
                self.backbone = models.resnet101(weights=weights)
                backbone_features = 2048
            else:
                raise ValueError(f"Unsupported ResNet variant: {backbone_name}")
            
            # Remove the final classification layer
            self.backbone = nn.Sequential(*list(self.backbone.children())[:-1])
            
        elif 'efficientnet' in backbone_name:
            if backbone_name == 'efficientnet_b0':
                weights = models.EfficientNet_B0_Weights.IMAGENET1K_V1 if self.config['pretrained'] else None
                self.backbone = models.efficientnet_b0(weights=weights)
                backbone_features = 1280
            elif backbone_name == 'efficientnet_b3':
                weights = models.EfficientNet_B3_Weights.IMAGENET1K_V1 if self.config['pretrained'] else None
                self.backbone = models.efficientnet_b3(weights=weights)
                backbone_features = 1536
            else:
                raise ValueError(f"Unsupported EfficientNet variant: {backbone_name}")
            
            # Remove the final classification layer
            self.backbone.classifier = nn.Identity()
            
        else:
            raise ValueError(f"Unsupported backbone: {backbone_name}")
        
        # Freeze backbone if specified
        if self.config['freeze_backbone']:
            for param in self.backbone.parameters():
                param.requires_grad = False
        
        # Custom head for mining-specific features
        self.custom_head = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)) if 'resnet' in backbone_name else nn.Identity(),
            nn.Flatten(),
            nn.Dropout(0.5),
            nn.Linear(backbone_features, 1024),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(1024),
            nn.Dropout(0.3),
            nn.Linear(1024, self.config['feature_dim']),
            nn.ReLU(inplace=True),
            # Additional layer for slope instability pattern detection
            nn.Linear(self.config['feature_dim'], self.config['feature_dim']),
            nn.Sigmoid()  # Normalize features to [0, 1] range
        )
        
        # Combine backbone and custom head
        self.model = nn.Sequential(self.backbone, self.custom_head)
        self.model.to(self.device)
        self.model.eval()  # Set to evaluation mode by default
    
    def extract_features(self, data: Union[str, Image.Image, np.ndarray, ImageData]) -> np.ndarray:
        """Extract features from image data.
        
        Args:
            data: Input image data (file path, PIL Image, numpy array, or ImageData)
            
        Returns:
            Feature vector as numpy array
        """
        # Convert input to PIL Image
        if isinstance(data, str):
            image = Image.open(data).convert('RGB')
        elif isinstance(data, ImageData):
            image = Image.open(data.image_path).convert('RGB')
        elif isinstance(data, np.ndarray):
            image = Image.fromarray(data).convert('RGB')
        elif isinstance(data, Image.Image):
            image = data.convert('RGB')
        else:
            raise ValueError(f"Unsupported input type: {type(data)}")
        
        # Preprocess image
        input_tensor = self.transform(image).unsqueeze(0).to(self.device)
        
        # Extract features
        with torch.no_grad():
            features = self.model(input_tensor)
            features = features.cpu().numpy().flatten()
        
        return features
    
    def extract_batch_features(self, image_paths: List[str]) -> np.ndarray:
        """Extract features from a batch of images.
        
        Args:
            image_paths: List of image file paths
            
        Returns:
            Feature matrix with shape (n_images, feature_dim)
        """
        batch_features = []
        
        for image_path in image_paths:
            try:
                features = self.extract_features(image_path)
                batch_features.append(features)
            except Exception as e:
                print(f"Warning: Failed to process {image_path}: {e}")
                # Add zero features for failed images
                batch_features.append(np.zeros(self.config['feature_dim']))
        
        return np.array(batch_features)
    
    def save_features(self, features: np.ndarray, output_path: str):
        """Save extracted features to disk.
        
        Args:
            features: Feature array to save
            output_path: Path to save the features
        """
        output_dir = os.path.dirname(output_path)
        if output_dir:  # Only create directory if path has a directory component
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
    
    def save_model(self, model_path: str):
        """Save the trained model to disk.
        
        Args:
            model_path: Path to save the model
        """
        model_dir = os.path.dirname(model_path)
        if model_dir:  # Only create directory if path has a directory component
            os.makedirs(model_dir, exist_ok=True)
        
        # Save model state dict and config
        save_dict = {
            'model_state_dict': self.model.state_dict(),
            'config': self.config
        }
        torch.save(save_dict, model_path)
    
    def load_model(self, model_path: str):
        """Load a trained model from disk.
        
        Args:
            model_path: Path to load the model from
        """
        checkpoint = torch.load(model_path, map_location=self.device)
        
        # Update config and rebuild model if necessary
        self.config.update(checkpoint['config'])
        self._feature_dim = self.config['feature_dim']
        self._build_model()
        
        # Load model weights
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()
    
    def fine_tune(self, train_loader, val_loader, num_epochs: int = 10, 
                  learning_rate: float = 1e-4):
        """Fine-tune the model on mining-specific data.
        
        Args:
            train_loader: Training data loader
            val_loader: Validation data loader
            num_epochs: Number of training epochs
            learning_rate: Learning rate for fine-tuning
        """
        # Unfreeze some layers for fine-tuning
        if self.config['freeze_backbone']:
            # Unfreeze the last few layers of backbone
            for param in list(self.backbone.parameters())[-10:]:
                param.requires_grad = True
        
        # Set model to training mode
        self.model.train()
        
        # Define optimizer and loss function
        optimizer = torch.optim.Adam(
            filter(lambda p: p.requires_grad, self.model.parameters()),
            lr=learning_rate
        )
        criterion = nn.MSELoss()  # For feature learning
        
        for epoch in range(num_epochs):
            train_loss = 0.0
            
            for batch_idx, (images, targets) in enumerate(train_loader):
                images = images.to(self.device)
                targets = targets.to(self.device)
                
                optimizer.zero_grad()
                features = self.model(images)
                
                # Self-supervised learning objective (reconstruction)
                loss = criterion(features, targets)
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
            
            print(f"Epoch {epoch+1}/{num_epochs}, Loss: {train_loss/len(train_loader):.4f}")
        
        # Set back to evaluation mode
        self.model.eval()
    
    def get_feature_importance(self, image: Union[str, Image.Image]) -> np.ndarray:
        """Get feature importance using gradient-based attribution.
        
        Args:
            image: Input image
            
        Returns:
            Feature importance scores
        """
        # Convert to tensor
        if isinstance(image, str):
            image = Image.open(image).convert('RGB')
        elif isinstance(image, Image.Image):
            image = image.convert('RGB')
        
        input_tensor = self.transform(image).unsqueeze(0).to(self.device)
        input_tensor.requires_grad_(True)
        
        # Forward pass
        features = self.model(input_tensor)
        
        # Compute gradients
        feature_sum = features.sum()
        feature_sum.backward()
        
        # Get gradient magnitudes as importance scores
        gradients = input_tensor.grad.abs().mean(dim=(2, 3)).cpu().numpy().flatten()
        
        return gradients