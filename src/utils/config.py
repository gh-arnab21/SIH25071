"""Configuration management utilities."""

import os
import yaml
from typing import Any, Dict, Optional
from pathlib import Path


class ConfigManager:
    """Manages configuration loading and access."""
    
    def __init__(self, config_path: Optional[str] = None):
        """Initialize configuration manager.
        
        Args:
            config_path: Path to configuration file. If None, uses default config.
        """
        self.config_path = config_path or self._get_default_config_path()
        self._config = self._load_config()
    
    def _get_default_config_path(self) -> str:
        """Get path to default configuration file."""
        project_root = Path(__file__).parent.parent.parent
        return str(project_root / "config" / "default_config.yaml")
    
    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from YAML file."""
        try:
            with open(self.config_path, 'r') as file:
                config = yaml.safe_load(file)
            return config or {}
        except FileNotFoundError:
            print(f"Configuration file not found: {self.config_path}")
            return {}
        except yaml.YAMLError as e:
            print(f"Error parsing configuration file: {e}")
            return {}
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value by key.
        
        Args:
            key: Configuration key (supports dot notation, e.g., 'model.cnn.backbone')
            default: Default value if key not found
            
        Returns:
            Configuration value or default
        """
        keys = key.split('.')
        value = self._config
        
        try:
            for k in keys:
                value = value[k]
            return value
        except (KeyError, TypeError):
            return default
    
    def set(self, key: str, value: Any) -> None:
        """Set configuration value by key.
        
        Args:
            key: Configuration key (supports dot notation)
            value: Value to set
        """
        keys = key.split('.')
        config = self._config
        
        # Navigate to the parent of the target key
        for k in keys[:-1]:
            if k not in config:
                config[k] = {}
            config = config[k]
        
        # Set the value
        config[keys[-1]] = value
    
    def update(self, updates: Dict[str, Any]) -> None:
        """Update configuration with dictionary of values.
        
        Args:
            updates: Dictionary of key-value pairs to update
        """
        for key, value in updates.items():
            self.set(key, value)
    
    def save(self, path: Optional[str] = None) -> None:
        """Save configuration to file.
        
        Args:
            path: Path to save configuration. If None, uses current config_path.
        """
        save_path = path or self.config_path
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        with open(save_path, 'w') as file:
            yaml.dump(self._config, file, default_flow_style=False, indent=2)
    
    def get_data_paths(self) -> Dict[str, str]:
        """Get all data-related paths from configuration."""
        return {
            'raw_data_dir': self.get('data.raw_data_dir', 'data/raw'),
            'processed_data_dir': self.get('data.processed_data_dir', 'data/processed'),
            'models_dir': self.get('data.models_dir', 'models/saved')
        }
    
    def get_model_config(self, model_type: str) -> Dict[str, Any]:
        """Get configuration for specific model type.
        
        Args:
            model_type: Type of model ('cnn', 'lstm', 'ensemble', etc.)
            
        Returns:
            Model configuration dictionary
        """
        return self.get(f'model.{model_type}', {})
    
    def get_training_config(self) -> Dict[str, Any]:
        """Get training configuration."""
        return self.get('training', {})
    
    def get_preprocessing_config(self) -> Dict[str, Any]:
        """Get preprocessing configuration."""
        return self.get('preprocessing', {})
    
    def to_dict(self) -> Dict[str, Any]:
        """Get full configuration as dictionary."""
        return self._config.copy()


# Global configuration instance
config = ConfigManager()