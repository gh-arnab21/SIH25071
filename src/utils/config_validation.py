"""
Configuration validation utilities for the Rockfall Prediction Model.
Provides schema validation, type checking, and range validation for configuration files.
"""

import yaml
import logging as std_logging
from typing import Dict, Any, List, Optional, Union, Tuple
from pathlib import Path
import numpy as np

logger = std_logging.getLogger(__name__)


class ConfigurationError(Exception):
    """Custom exception for configuration validation errors."""
    pass


class ConfigValidator:
    """
    Validates configuration files against predefined schemas.
    Supports type checking, range validation, and dependency validation.
    """
    
    def __init__(self):
        self.schema = self._define_schema()
        
    def _define_schema(self) -> Dict[str, Any]:
        """Define the configuration schema with validation rules."""
        return {
            'data': {
                'required': True,
                'type': dict,
                'schema': {
                    'raw_data_dir': {'type': str, 'required': True},
                    'processed_data_dir': {'type': str, 'required': True},
                    'models_dir': {'type': str, 'required': True}
                }
            },
            'model': {
                'required': True,
                'type': dict,
                'schema': {
                    'image': {
                        'type': dict,
                        'required': True,
                        'schema': {
                            'input_size': {'type': list, 'required': True},
                            'batch_size': {'type': int, 'range': (1, 512), 'required': True},
                            'augmentation': {'type': bool, 'required': True},
                            'normalization': {
                                'type': dict,
                                'required': True,
                                'schema': {
                                    'mean': {'type': list, 'required': True},
                                    'std': {'type': list, 'required': True}
                                }
                            }
                        }
                    },
                    'cnn': {
                        'type': dict,
                        'required': True,
                        'schema': {
                            'backbone': {'type': str, 'choices': ['resnet50', 'resnet101', 'vgg16', 'densenet121'], 'required': True},
                            'pretrained': {'type': bool, 'required': True},
                            'freeze_backbone': {'type': bool, 'required': True},
                            'feature_dim': {'type': int, 'range': (1, 10000), 'required': True}
                        }
                    },
                    'lstm': {
                        'type': dict,
                        'required': True,
                        'schema': {
                            'hidden_size': {'type': int, 'range': (1, 1000), 'required': True},
                            'num_layers': {'type': int, 'range': (1, 10), 'required': True},
                            'dropout': {'type': float, 'range': (0.0, 1.0), 'required': True},
                            'sequence_length': {'type': int, 'range': (1, 1000), 'required': True}
                        }
                    },
                    'ensemble': {
                        'type': dict,
                        'required': True,
                        'schema': {
                            'algorithms': {'type': list, 'required': True},
                            'voting_method': {'type': str, 'choices': ['hard', 'soft'], 'required': True},
                            'use_stacking': {'type': bool, 'required': True}
                        }
                    },
                    'random_forest': {
                        'type': dict,
                        'required': False,
                        'schema': {
                            'n_estimators': {'type': int, 'range': (1, 1000), 'required': True},
                            'max_depth': {'type': int, 'range': (1, 100), 'required': True},
                            'min_samples_split': {'type': int, 'range': (2, 100), 'required': True},
                            'random_state': {'type': int, 'required': True}
                        }
                    },
                    'xgboost': {
                        'type': dict,
                        'required': False,
                        'schema': {
                            'n_estimators': {'type': int, 'range': (1, 1000), 'required': True},
                            'max_depth': {'type': int, 'range': (1, 20), 'required': True},
                            'learning_rate': {'type': float, 'range': (0.001, 1.0), 'required': True},
                            'random_state': {'type': int, 'required': True}
                        }
                    }
                }
            },
            'training': {
                'required': False,
                'type': dict,
                'schema': {
                    'validation_split': {'type': float, 'range': (0.0, 1.0), 'required': True},
                    'test_split': {'type': float, 'range': (0.0, 1.0), 'required': True},
                    'cross_validation_folds': {'type': int, 'range': (2, 20), 'required': False},
                    'early_stopping_patience': {'type': int, 'range': (1, 100), 'required': False},
                    'max_epochs': {'type': int, 'range': (1, 1000), 'required': False},
                    'batch_size': {'type': int, 'range': (1, 512), 'required': False}
                }
            },
            'preprocessing': {
                'required': False,
                'type': dict,
                'schema': {
                    'imputation': {
                        'type': dict,
                        'required': False,
                        'schema': {
                            'numerical_strategy': {'type': str, 'choices': ['mean', 'median', 'mode'], 'required': True},
                            'categorical_strategy': {'type': str, 'choices': ['mode', 'constant'], 'required': True}
                        }
                    },
                    'outlier_detection': {
                        'type': dict,
                        'required': False,
                        'schema': {
                            'method': {'type': str, 'choices': ['iqr', 'zscore', 'isolation_forest'], 'required': True},
                            'threshold': {'type': float, 'range': (0.1, 5.0), 'required': True}
                        }
                    },
                    'class_balance': {
                        'type': dict,
                        'required': False,
                        'schema': {
                            'method': {'type': str, 'choices': ['smote', 'adasyn', 'random_oversample'], 'required': True},
                            'sampling_strategy': {'type': str, 'required': True}
                        }
                    },
                    'scaling': {
                        'type': dict,
                        'required': False,
                        'schema': {
                            'method': {'type': str, 'choices': ['standard', 'minmax', 'robust'], 'required': True}
                        }
                    }
                }
            },
            'evaluation': {
                'required': False,
                'type': dict,
                'schema': {
                    'metrics': {'type': list, 'required': True},
                    'average': {'type': str, 'choices': ['micro', 'macro', 'weighted'], 'required': False}
                }
            },
            'logging': {
                'required': False,
                'type': dict,
                'schema': {
                    'level': {'type': str, 'choices': ['DEBUG', 'INFO', 'WARNING', 'ERROR'], 'required': True},
                    'format': {'type': str, 'required': False},
                    'file': {'type': str, 'required': False}
                }
            }
        }
    
    def validate_config(self, config: Dict[str, Any], config_path: str = None) -> Tuple[bool, List[str]]:
        """
        Validate a configuration dictionary against the schema.
        
        Args:
            config: Configuration dictionary to validate
            config_path: Optional path for error reporting
            
        Returns:
            Tuple of (is_valid, list_of_errors)
        """
        errors = []
        
        try:
            self._validate_dict(config, self.schema, "root", errors)
            
            # Perform cross-field validation
            self._validate_cross_dependencies(config, errors)
            
            # Validate data splits
            self._validate_data_splits(config, errors)
            
            return len(errors) == 0, errors
            
        except Exception as e:
            errors.append(f"Validation failed with exception: {str(e)}")
            return False, errors
    
    def _validate_dict(self, data: Dict[str, Any], schema: Dict[str, Any], path: str, errors: List[str]):
        """Recursively validate a dictionary against its schema."""
        # Check required fields
        for key, spec in schema.items():
            if spec.get('required', False) and key not in data:
                errors.append(f"Missing required field: {path}.{key}")
                continue
                
            if key not in data:
                continue
                
            value = data[key]
            field_path = f"{path}.{key}"
            
            # Type validation
            expected_type = spec.get('type')
            if expected_type and not self._validate_type(value, expected_type):
                errors.append(f"Invalid type at {field_path}: expected {expected_type.__name__}, got {type(value).__name__}")
                continue
            
            # Range validation
            if 'range' in spec:
                if not self._validate_range(value, spec['range']):
                    errors.append(f"Value out of range at {field_path}: {value} not in {spec['range']}")
            
            # Choice validation
            if 'choices' in spec:
                if value not in spec['choices']:
                    errors.append(f"Invalid choice at {field_path}: {value} not in {spec['choices']}")
            
            # Recursive validation for nested dictionaries
            if 'schema' in spec and isinstance(value, dict):
                self._validate_dict(value, spec['schema'], field_path, errors)
    
    def _validate_type(self, value: Any, expected_type: type) -> bool:
        """Validate the type of a value."""
        if expected_type == list:
            return isinstance(value, list)
        elif expected_type == dict:
            return isinstance(value, dict)
        elif expected_type == str:
            return isinstance(value, str)
        elif expected_type == int:
            return isinstance(value, (int, np.integer)) and not isinstance(value, bool)
        elif expected_type == float:
            return isinstance(value, (int, float, np.number)) and not isinstance(value, bool)
        elif expected_type == bool:
            return isinstance(value, bool)
        else:
            return isinstance(value, expected_type)
    
    def _validate_range(self, value: Union[int, float], range_spec: Tuple[Union[int, float], Union[int, float]]) -> bool:
        """Validate that a numeric value is within the specified range."""
        if not isinstance(value, (int, float, np.number)):
            return False
        return range_spec[0] <= value <= range_spec[1]
    
    def _validate_cross_dependencies(self, config: Dict[str, Any], errors: List[str]):
        """Validate cross-field dependencies and consistency."""
        # Validate ensemble algorithms are supported
        model_config = config.get('model', {})
        ensemble_config = model_config.get('ensemble', {})
        if 'algorithms' in ensemble_config:
            algorithms = ensemble_config['algorithms']
            supported_algorithms = ['random_forest', 'xgboost', 'neural_network', 'svm', 'gradient_boosting']
            for algo in algorithms:
                if algo not in supported_algorithms:
                    errors.append(f"Unsupported ensemble algorithm: {algo}. Supported: {supported_algorithms}")
        
        # Validate image input size
        image_config = model_config.get('image', {})
        if 'input_size' in image_config:
            input_size = image_config['input_size']
            if not (isinstance(input_size, list) and len(input_size) == 2):
                errors.append("Image input_size must be a list of 2 integers [height, width]")
            elif not all(isinstance(dim, int) and dim > 0 for dim in input_size):
                errors.append("Image input_size dimensions must be positive integers")
        
        # Validate normalization values
        if 'normalization' in image_config:
            norm_config = image_config['normalization']
            if 'mean' in norm_config and 'std' in norm_config:
                mean = norm_config['mean']
                std = norm_config['std']
                if len(mean) != len(std):
                    errors.append(f"Normalization mean length ({len(mean)}) must match std length ({len(std)})")
                if len(mean) != 3:
                    errors.append("Normalization mean and std must have 3 values (RGB channels)")
    
    def _validate_data_splits(self, config: Dict[str, Any], errors: List[str]):
        """Validate data split ratios sum to reasonable values."""
        training_config = config.get('training', {})
        if 'validation_split' in training_config and 'test_split' in training_config:
            val_split = training_config['validation_split']
            test_split = training_config['test_split']
            
            if val_split + test_split >= 1.0:
                errors.append(f"Validation split ({val_split}) + test split ({test_split}) must be < 1.0 to leave data for training")
            
            if val_split >= 0.5:
                errors.append(f"Validation split ({val_split}) should be less than 0.5 to leave sufficient data for training")
    
    def validate_file(self, config_path: str) -> Tuple[bool, List[str], Optional[Dict[str, Any]]]:
        """
        Validate a configuration file.
        
        Args:
            config_path: Path to the configuration file
            
        Returns:
            Tuple of (is_valid, list_of_errors, config_dict)
        """
        try:
            config_path = Path(config_path)
            if not config_path.exists():
                return False, [f"Configuration file not found: {config_path}"], None
            
            with open(config_path, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
            
            if config is None:
                return False, ["Configuration file is empty or invalid YAML"], None
            
            is_valid, errors = self.validate_config(config, str(config_path))
            return is_valid, errors, config
            
        except yaml.YAMLError as e:
            return False, [f"YAML parsing error: {str(e)}"], None
        except Exception as e:
            return False, [f"Error reading configuration file: {str(e)}"], None
    
    def suggest_fixes(self, errors: List[str]) -> List[str]:
        """
        Suggest fixes for common configuration errors.
        
        Args:
            errors: List of validation errors
            
        Returns:
            List of suggested fixes
        """
        suggestions = []
        
        for error in errors:
            if "Missing required field" in error:
                field = error.split(": ")[-1]
                suggestions.append(f"Add the required field {field} to your configuration")
            
            elif "Invalid type" in error:
                suggestions.append("Check the data type of the specified field matches the expected type")
            
            elif "Value out of range" in error:
                suggestions.append("Adjust the value to be within the valid range")
            
            elif "Invalid choice" in error:
                suggestions.append("Use one of the valid choices for this field")
            
            elif "sum to reasonable values" in error:
                suggestions.append("Ensure validation_split + test_split < 1.0 to leave data for training")
            
            else:
                suggestions.append("Review the configuration against the schema requirements")
        
        return list(set(suggestions))  # Remove duplicates


def validate_config_file(config_path: str, verbose: bool = True) -> bool:
    """
    Convenience function to validate a configuration file and print results.
    
    Args:
        config_path: Path to the configuration file
        verbose: Whether to print detailed validation results
        
    Returns:
        True if validation passes, False otherwise
    """
    validator = ConfigValidator()
    is_valid, errors, config = validator.validate_file(config_path)
    
    if verbose:
        # Configure basic logging for this function
        std_logging.basicConfig(level=std_logging.INFO, format='%(message)s')
        
        if is_valid:
            print(f"✓ Configuration file {config_path} is valid")
        else:
            print(f"✗ Configuration file {config_path} has {len(errors)} errors:")
            for error in errors:
                print(f"  - {error}")
            
            suggestions = validator.suggest_fixes(errors)
            if suggestions:
                print("Suggested fixes:")
                for suggestion in suggestions:
                    print(f"  - {suggestion}")
    
    return is_valid


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) != 2:
        print("Usage: python config_validation.py <config_file>")
        sys.exit(1)
    
    config_file = sys.argv[1]
    is_valid = validate_config_file(config_file, verbose=True)
    sys.exit(0 if is_valid else 1)