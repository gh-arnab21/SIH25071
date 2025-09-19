#!/usr/bin/env python3
"""
Enhanced Training Script for the Rockfall Prediction System.

This script provides a comprehensive command-line interface for training
the ensemble model with extensive configuration options, logging, and
model comparison capabilities.
"""

import argparse
import logging
import sys
import os
import json
import time
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional, List
import yaml
import traceback

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

from src.utils.config_validation import ConfigValidator
from src.utils.training_logger import TrainingLogger, training_session, setup_training_logger


class TrainingScriptError(Exception):
    """Custom exception for training script errors."""
    pass


def parse_arguments():
    """Parse command line arguments with comprehensive options."""
    parser = argparse.ArgumentParser(
        description="Enhanced Training Script for Rockfall Prediction Model",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Configuration and data arguments
    config_group = parser.add_argument_group('Configuration')
    config_group.add_argument(
        "--config", "-c",
        type=str,
        default="config/default_config.yaml",
        help="Path to configuration file"
    )
    
    config_group.add_argument(
        "--data-dir", "-d",
        type=str,
        help="Override data directory from config"
    )
    
    config_group.add_argument(
        "--output-dir", "-o",
        type=str,
        help="Override output directory from config"
    )
    
    config_group.add_argument(
        "--dataset-type",
        type=str,
        choices=['open_pit_mine', 'rocknet_seismic', 'brazilian_rockfall', 'mining_object_detection', 'mining_segmentation', 'mining_combined', 'auto'],
        default='auto',
        help="Dataset type to load (auto-detect by default)"
    )
    
    # Experiment management
    experiment_group = parser.add_argument_group('Experiment Management')
    experiment_group.add_argument(
        "--experiment-name", "-n",
        type=str,
        default=None,
        help="Name for this training experiment"
    )
    
    experiment_group.add_argument(
        "--tags",
        type=str,
        nargs='+',
        default=[],
        help="Tags to associate with this experiment"
    )
    
    experiment_group.add_argument(
        "--description",
        type=str,
        help="Description of this experiment"
    )
    
    # Training parameters
    training_group = parser.add_argument_group('Training Parameters')
    training_group.add_argument(
        "--epochs", "-e",
        type=int,
        help="Number of training epochs (overrides config)"
    )
    
    training_group.add_argument(
        "--batch-size", "-b",
        type=int,
        help="Batch size for training (overrides config)"
    )
    
    training_group.add_argument(
        "--learning-rate", "-lr",
        type=float,
        help="Learning rate (overrides config)"
    )
    
    # Cross-validation and optimization
    cv_group = parser.add_argument_group('Cross-Validation and Optimization')
    cv_group.add_argument(
        "--cv-folds", "-k",
        type=int,
        default=None,
        help="Number of cross-validation folds"
    )
    
    cv_group.add_argument(
        "--optimize-hyperparams",
        action="store_true",
        help="Enable hyperparameter optimization"
    )
    
    cv_group.add_argument(
        "--optimization-trials",
        type=int,
        default=100,
        help="Number of hyperparameter optimization trials"
    )
    
    # Model and checkpoint management
    model_group = parser.add_argument_group('Model Management')
    model_group.add_argument(
        "--resume-from",
        type=str,
        help="Path to checkpoint to resume training from"
    )
    
    model_group.add_argument(
        "--save-best-only",
        action="store_true",
        default=True,
        help="Save only the best model during training"
    )
    
    # Evaluation and comparison
    eval_group = parser.add_argument_group('Evaluation and Comparison')
    eval_group.add_argument(
        "--evaluate-only",
        action="store_true",
        help="Only run evaluation on existing model"
    )
    
    eval_group.add_argument(
        "--compare-models",
        type=str,
        nargs='+',
        help="Paths to models to compare"
    )
    
    # Logging and monitoring
    log_group = parser.add_argument_group('Logging and Monitoring')
    log_group.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level"
    )
    
    log_group.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug mode with additional logging"
    )
    
    # Miscellaneous
    misc_group = parser.add_argument_group('Miscellaneous')
    misc_group.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility"
    )
    
    misc_group.add_argument(
        "--dry-run",
        action="store_true",
        help="Perform a dry run without actual training"
    )
    
    misc_group.add_argument(
        "--force",
        action="store_true",
        help="Force overwrite existing experiment directory"
    )
    
    return parser.parse_args()


def setup_experiment_directory(args: argparse.Namespace) -> Path:
    """Set up experiment directory structure."""
    # Generate experiment name if not provided
    if args.experiment_name is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.experiment_name = f"experiment_{timestamp}"
    
    # Create experiment directory
    output_dir = Path(args.output_dir) if args.output_dir else Path("experiments")
    experiment_dir = output_dir / args.experiment_name
    
    if experiment_dir.exists() and not args.force:
        if not args.resume_from:
            raise TrainingScriptError(
                f"Experiment directory {experiment_dir} already exists. "
                "Use --force to overwrite or --resume-from to continue."
            )
    
    # Create directory structure
    experiment_dir.mkdir(parents=True, exist_ok=True)
    (experiment_dir / "checkpoints").mkdir(exist_ok=True)
    (experiment_dir / "logs").mkdir(exist_ok=True)
    (experiment_dir / "reports").mkdir(exist_ok=True)
    (experiment_dir / "config").mkdir(exist_ok=True)
    
    return experiment_dir


def load_and_merge_config(args: argparse.Namespace) -> Dict[str, Any]:
    """Load and merge configuration from file and command line arguments."""
    # Load base config
    config_path = Path(args.config)
    if not config_path.exists():
        raise TrainingScriptError(f"Configuration file not found: {config_path}")
    
    # Validate configuration
    validator = ConfigValidator()
    is_valid, errors, config = validator.validate_file(str(config_path))
    if not is_valid:
        error_msg = f"Configuration validation failed with {len(errors)} errors:\n"
        for error in errors:
            error_msg += f"  - {error}\n"
        
        suggestions = validator.suggest_fixes(errors)
        if suggestions:
            error_msg += "Suggested fixes:\n"
            for suggestion in suggestions:
                error_msg += f"  - {suggestion}\n"
        
        raise TrainingScriptError(error_msg)
    
    # Override with command line arguments
    if args.data_dir:
        config.setdefault('data', {})['raw_data_dir'] = args.data_dir
    
    if args.output_dir:
        config.setdefault('training', {})['output_dir'] = args.output_dir
    
    if args.epochs:
        config.setdefault('training', {})['max_epochs'] = args.epochs
    
    if args.batch_size:
        config.setdefault('training', {})['batch_size'] = args.batch_size
    
    if args.learning_rate:
        config.setdefault('training', {})['learning_rate'] = args.learning_rate
    
    # Set optimization parameters
    config.setdefault('optimization', {})['enabled'] = args.optimize_hyperparams
    config['optimization']['n_trials'] = args.optimization_trials
    
    return config


def save_experiment_metadata(args: argparse.Namespace, experiment_dir: Path, config: Dict[str, Any]):
    """Save experiment metadata and configuration."""
    metadata = {
        "experiment_name": args.experiment_name,
        "timestamp": datetime.now().isoformat(),
        "tags": args.tags,
        "description": args.description,
        "command_line_args": vars(args),
        "python_version": sys.version,
        "config": config
    }
    
    metadata_file = experiment_dir / "metadata.json"
    with open(metadata_file, 'w') as f:
        json.dump(metadata, f, indent=2, default=str)
    
    # Save config separately for easy access
    config_file = experiment_dir / "config" / "training_config.yaml"
    with open(config_file, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)


def setup_logging_system(args: argparse.Namespace, experiment_dir: Path) -> logging.Logger:
    """Set up comprehensive logging system."""
    log_file = experiment_dir / "logs" / f"training_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    log_file.parent.mkdir(parents=True, exist_ok=True)
    
    # Setup logging
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(sys.stdout)
        ]
    )
    
    logger = logging.getLogger(__name__)
    
    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)
        logger.debug("Debug mode enabled")
    
    return logger


def main():
    """Main function with enhanced logging and training session management."""
    try:
        # Parse arguments
        args = parse_arguments()
        
        # Handle dry run
        if args.dry_run:
            print("=== DRY RUN MODE ===")
            print(f"Configuration: {args.config}")
            print(f"Experiment name: {args.experiment_name or 'auto-generated'}")
            print(f"Data directory: {args.data_dir or 'from config'}")
            print(f"Dataset type: {args.dataset_type}")
            print(f"Epochs: {args.epochs or 'from config'}")
            print(f"Batch size: {args.batch_size or 'from config'}")
            print(f"CV folds: {args.cv_folds or 'none'}")
            print(f"Optimize hyperparams: {args.optimize_hyperparams}")
            print("=" * 50)
            return 0
        
        # Set random seed for reproducibility
        import random
        import numpy as np
        random.seed(args.seed)
        np.random.seed(args.seed)
        
        # Setup experiment directory
        experiment_dir = setup_experiment_directory(args)
        
        # Load and merge configuration
        config = load_and_merge_config(args)
        
        # Setup enhanced training logger
        training_logger = setup_training_logger(experiment_dir)
        
        # Setup standard logging as well
        standard_logger = setup_logging_system(args, experiment_dir)
        
        # Save experiment metadata
        save_experiment_metadata(args, experiment_dir, config)
        
        # Create args dict for training session
        args_dict = vars(args)
        
        # Use training session context manager
        with training_session(training_logger, config, args_dict):
            # Demonstrate functionality
            training_logger.info("Training script setup completed successfully")
            training_logger.info("Ready to integrate with existing training components")
            
            # Show what would be done
            if args.evaluate_only:
                training_logger.info("Mode: Evaluation only")
            elif args.compare_models:
                training_logger.info(f"Mode: Model comparison ({len(args.compare_models)} models)")
            else:
                training_logger.info("Mode: Training")
                if args.cv_folds:
                    training_logger.info(f"Cross-validation: {args.cv_folds} folds")
                if args.optimize_hyperparams:
                    training_logger.info(f"Hyperparameter optimization: {args.optimization_trials} trials")
            
            # Simulate some training progress for demonstration
            if not args.evaluate_only and not args.compare_models:
                training_logger.info("Demonstrating training progress tracking...")
                
                progress = training_logger.create_progress_tracker(3, "Demo Training")
                import time
                
                for i in range(3):
                    # Simulate some work
                    time.sleep(0.5)
                    progress.update(1, f"Phase {i+1}")
                
                training_logger.info("Training progress demonstration completed")
        
        training_logger.info("TRAINING SCRIPT DEMONSTRATION COMPLETED")
        
        return 0
        
    except KeyboardInterrupt:
        print("\nTraining interrupted by user")
        return 130
    except TrainingScriptError as e:
        print(f"Training script error: {e}")
        return 1
    except Exception as e:
        print(f"Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        return 1
        
        return 0
        
    except TrainingScriptError as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1
    except Exception as e:
        print(f"Unexpected error: {e}", file=sys.stderr)
        print(traceback.format_exc(), file=sys.stderr)
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code if exit_code is not None else 0)