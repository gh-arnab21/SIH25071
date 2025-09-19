"""
Enhanced logging and progress tracking for the Rockfall Prediction System.
Provides comprehensive logging with training metrics, progress bars, and
checkpoint information.
"""

import logging
import sys
import time
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional, List, Union
import json
from dataclasses import dataclass, field
from contextlib import contextmanager
import threading
import queue


@dataclass
class TrainingMetrics:
    """Data class for tracking training metrics."""
    epoch: int = 0
    train_loss: float = 0.0
    val_loss: float = 0.0
    train_accuracy: float = 0.0
    val_accuracy: float = 0.0
    train_f1: float = 0.0
    val_f1: float = 0.0
    learning_rate: float = 0.0
    epoch_time: float = 0.0
    timestamp: datetime = field(default_factory=datetime.now)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert metrics to dictionary for logging."""
        return {
            'epoch': self.epoch,
            'train_loss': self.train_loss,
            'val_loss': self.val_loss,
            'train_accuracy': self.train_accuracy,
            'val_accuracy': self.val_accuracy,
            'train_f1': self.train_f1,
            'val_f1': self.val_f1,
            'learning_rate': self.learning_rate,
            'epoch_time': self.epoch_time,
            'timestamp': self.timestamp.isoformat()
        }


class ProgressTracker:
    """Simple progress tracker with text-based progress bar."""
    
    def __init__(self, total: int, description: str = "Progress", width: int = 50):
        self.total = total
        self.current = 0
        self.description = description
        self.width = width
        self.start_time = time.time()
        
    def update(self, increment: int = 1, message: str = ""):
        """Update progress and display bar."""
        self.current += increment
        if self.current > self.total:
            self.current = self.total
            
        # Calculate progress
        progress = self.current / self.total if self.total > 0 else 0
        filled_width = int(self.width * progress)
        
        # Create progress bar
        bar = '█' * filled_width + '░' * (self.width - filled_width)
        
        # Calculate time estimates
        elapsed = time.time() - self.start_time
        if self.current > 0:
            eta = (elapsed / self.current) * (self.total - self.current)
            eta_str = f"{eta:.1f}s"
        else:
            eta_str = "N/A"
        
        # Format display
        percent = progress * 100
        display = f"\r{self.description}: |{bar}| {self.current}/{self.total} [{percent:6.2f}%] ETA: {eta_str}"
        
        if message:
            display += f" - {message}"
        
        # Print with carriage return for updating same line
        print(display, end='', flush=True)
        
        # Print newline when complete
        if self.current >= self.total:
            print()
    
    def reset(self):
        """Reset progress tracker."""
        self.current = 0
        self.start_time = time.time()


class TrainingLogger:
    """
    Enhanced logger for training with metrics tracking and progress monitoring.
    """
    
    def __init__(self, name: str, log_dir: Optional[Path] = None, 
                 console_level: str = "INFO", file_level: str = "DEBUG"):
        self.name = name
        self.log_dir = Path(log_dir) if log_dir else None
        self.logger = logging.getLogger(name)
        self.logger.setLevel(logging.DEBUG)
        
        # Clear existing handlers
        self.logger.handlers = []
        
        # Setup console handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(getattr(logging, console_level.upper()))
        console_formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        console_handler.setFormatter(console_formatter)
        self.logger.addHandler(console_handler)
        
        # Setup file handler if log directory is provided
        if self.log_dir:
            self.log_dir.mkdir(parents=True, exist_ok=True)
            
            # Main log file
            log_file = self.log_dir / f"{name}.log"
            file_handler = logging.FileHandler(log_file)
            file_handler.setLevel(getattr(logging, file_level.upper()))
            file_formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s'
            )
            file_handler.setFormatter(file_formatter)
            self.logger.addHandler(file_handler)
            
            # Metrics file
            self.metrics_file = self.log_dir / f"{name}_metrics.jsonl"
            self.metrics = []
        else:
            self.metrics_file = None
            self.metrics = []
    
    def info(self, message: str, **kwargs):
        """Log info message."""
        self.logger.info(message, **kwargs)
    
    def debug(self, message: str, **kwargs):
        """Log debug message."""
        self.logger.debug(message, **kwargs)
    
    def warning(self, message: str, **kwargs):
        """Log warning message."""
        self.logger.warning(message, **kwargs)
    
    def error(self, message: str, **kwargs):
        """Log error message."""
        self.logger.error(message, **kwargs)
    
    def log_metrics(self, metrics: TrainingMetrics):
        """Log training metrics."""
        self.metrics.append(metrics)
        
        # Log to console with nice formatting
        self.info(
            f"Epoch {metrics.epoch:3d} | "
            f"Train Loss: {metrics.train_loss:.4f} | "
            f"Val Loss: {metrics.val_loss:.4f} | "
            f"Train Acc: {metrics.train_accuracy:.3f} | "
            f"Val Acc: {metrics.val_accuracy:.3f} | "
            f"Time: {metrics.epoch_time:.1f}s"
        )
        
        # Save to metrics file
        if self.metrics_file:
            with open(self.metrics_file, 'a') as f:
                json.dump(metrics.to_dict(), f)
                f.write('\n')
    
    def log_checkpoint(self, epoch: int, model_path: Path, metrics: TrainingMetrics):
        """Log checkpoint creation."""
        self.info(f"Checkpoint saved at epoch {epoch}: {model_path}")
        
        if self.log_dir:
            checkpoint_info = {
                'epoch': epoch,
                'model_path': str(model_path),
                'metrics': metrics.to_dict(),
                'timestamp': datetime.now().isoformat()
            }
            
            checkpoint_file = self.log_dir / "checkpoints.jsonl"
            with open(checkpoint_file, 'a') as f:
                json.dump(checkpoint_info, f)
                f.write('\n')
    
    def log_experiment_start(self, config: Dict[str, Any], args: Dict[str, Any]):
        """Log experiment start with configuration."""
        self.info("="*60)
        self.info("TRAINING EXPERIMENT STARTED")
        self.info("="*60)
        
        self.info(f"Experiment: {args.get('experiment_name', 'unnamed')}")
        self.info(f"Description: {args.get('description', 'No description')}")
        self.info(f"Tags: {', '.join(args.get('tags', []))}")
        
        # Log key configuration
        self.info("\nKey Configuration:")
        if 'training' in config:
            training = config['training']
            self.info(f"  Max Epochs: {training.get('max_epochs', 'N/A')}")
            self.info(f"  Batch Size: {training.get('batch_size', 'N/A')}")
            self.info(f"  Validation Split: {training.get('validation_split', 'N/A')}")
        
        if 'model' in config and 'ensemble' in config['model']:
            ensemble = config['model']['ensemble']
            self.info(f"  Ensemble Algorithms: {ensemble.get('algorithms', 'N/A')}")
            self.info(f"  Voting Method: {ensemble.get('voting_method', 'N/A')}")
        
        self.info("")
    
    def log_experiment_end(self, success: bool, duration: float, best_metrics: Optional[TrainingMetrics] = None):
        """Log experiment completion."""
        status = "COMPLETED SUCCESSFULLY" if success else "FAILED"
        self.info("="*60)
        self.info(f"TRAINING EXPERIMENT {status}")
        self.info("="*60)
        
        self.info(f"Total Duration: {duration:.2f} seconds ({duration/60:.2f} minutes)")
        
        if best_metrics:
            self.info("\nBest Results:")
            self.info(f"  Best Validation Loss: {best_metrics.val_loss:.4f}")
            self.info(f"  Best Validation Accuracy: {best_metrics.val_accuracy:.3f}")
            self.info(f"  Best Validation F1: {best_metrics.val_f1:.3f}")
            self.info(f"  Achieved at Epoch: {best_metrics.epoch}")
        
        self.info("")
    
    def create_progress_tracker(self, total: int, description: str = "Progress") -> ProgressTracker:
        """Create a progress tracker for training phases."""
        return ProgressTracker(total, description)
    
    def log_model_comparison(self, comparison_results: Dict[str, Any]):
        """Log model comparison results."""
        self.info("="*60)
        self.info("MODEL COMPARISON RESULTS")
        self.info("="*60)
        
        for model_name, results in comparison_results.items():
            self.info(f"\n{model_name}:")
            if isinstance(results, dict):
                for metric, value in results.items():
                    if isinstance(value, float):
                        self.info(f"  {metric}: {value:.4f}")
                    else:
                        self.info(f"  {metric}: {value}")
    
    def get_metrics_summary(self) -> Dict[str, Any]:
        """Get summary of all training metrics."""
        if not self.metrics:
            return {}
        
        # Calculate best metrics
        best_val_loss = min(self.metrics, key=lambda x: x.val_loss)
        best_val_acc = max(self.metrics, key=lambda x: x.val_accuracy)
        best_val_f1 = max(self.metrics, key=lambda x: x.val_f1)
        
        return {
            'total_epochs': len(self.metrics),
            'best_val_loss': {
                'value': best_val_loss.val_loss,
                'epoch': best_val_loss.epoch
            },
            'best_val_accuracy': {
                'value': best_val_acc.val_accuracy,
                'epoch': best_val_acc.epoch
            },
            'best_val_f1': {
                'value': best_val_f1.val_f1,
                'epoch': best_val_f1.epoch
            },
            'final_metrics': self.metrics[-1].to_dict() if self.metrics else None,
            'average_epoch_time': sum(m.epoch_time for m in self.metrics) / len(self.metrics)
        }


@contextmanager
def training_session(logger: TrainingLogger, config: Dict[str, Any], args: Dict[str, Any]):
    """Context manager for training session with automatic start/end logging."""
    start_time = time.time()
    success = False
    best_metrics = None
    
    try:
        logger.log_experiment_start(config, args)
        yield logger
        success = True
    except Exception as e:
        logger.error(f"Training failed with error: {str(e)}")
        raise
    finally:
        duration = time.time() - start_time
        
        # Get best metrics if available
        if logger.metrics:
            best_metrics = min(logger.metrics, key=lambda x: x.val_loss)
        
        logger.log_experiment_end(success, duration, best_metrics)


def setup_training_logger(experiment_dir: Path, name: str = "training") -> TrainingLogger:
    """Setup training logger with appropriate configuration."""
    log_dir = experiment_dir / "logs"
    return TrainingLogger(name, log_dir)


# Example usage for integration testing
if __name__ == "__main__":
    import tempfile
    import shutil
    
    # Create temporary directory for testing
    temp_dir = Path(tempfile.mkdtemp())
    
    try:
        # Setup logger
        logger = TrainingLogger("test_training", temp_dir)
        
        # Test configuration
        config = {
            'training': {
                'max_epochs': 10,
                'batch_size': 32,
                'validation_split': 0.2
            },
            'model': {
                'ensemble': {
                    'algorithms': ['random_forest', 'xgboost'],
                    'voting_method': 'soft'
                }
            }
        }
        
        args = {
            'experiment_name': 'test_experiment',
            'description': 'Test logging functionality',
            'tags': ['test', 'logging']
        }
        
        # Test training session
        with training_session(logger, config, args):
            # Simulate training epochs
            progress = logger.create_progress_tracker(5, "Training")
            
            for epoch in range(5):
                # Simulate epoch metrics
                metrics = TrainingMetrics(
                    epoch=epoch + 1,
                    train_loss=1.0 - (epoch * 0.1),
                    val_loss=1.2 - (epoch * 0.08),
                    train_accuracy=0.5 + (epoch * 0.08),
                    val_accuracy=0.45 + (epoch * 0.07),
                    train_f1=0.48 + (epoch * 0.075),
                    val_f1=0.43 + (epoch * 0.065),
                    learning_rate=0.001,
                    epoch_time=2.5
                )
                
                logger.log_metrics(metrics)
                progress.update(1, f"Epoch {epoch + 1}")
                time.sleep(0.1)  # Simulate training time
        
        print(f"\nTest completed. Logs saved to: {temp_dir}")
        print("Metrics summary:", logger.get_metrics_summary())
        
    finally:
        # Cleanup
        shutil.rmtree(temp_dir, ignore_errors=True)