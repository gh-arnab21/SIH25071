"""
Model Checkpointing and Early Stopping for Rockfall Prediction System.

This module provides functionality for saving model checkpoints during training
and implementing early stopping to prevent overfitting.
"""

import logging
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Union, Any
from dataclasses import dataclass
import json
import pickle
from datetime import datetime
import shutil

# Local imports
from ..models.classifiers import EnsembleClassifier

logger = logging.getLogger(__name__)


@dataclass
class CheckpointMetadata:
    """Metadata for model checkpoints."""
    checkpoint_id: str
    timestamp: str
    epoch: int
    validation_score: float
    training_score: Optional[float] = None
    model_config: Optional[Dict[str, Any]] = None
    feature_names: Optional[List[str]] = None
    additional_metrics: Optional[Dict[str, float]] = None


class ModelCheckpoint:
    """
    Manages model checkpointing during training.
    
    Saves model states at regular intervals and keeps track of the best
    performing models based on validation metrics.
    """
    
    def __init__(self, checkpoint_dir: Union[str, Path], save_frequency: int = 5,
                 max_checkpoints: int = 10, save_best_only: bool = False):
        """
        Initialize model checkpoint manager.
        
        Args:
            checkpoint_dir: Directory to save checkpoints
            save_frequency: Save checkpoint every N epochs
            max_checkpoints: Maximum number of checkpoints to keep
            save_best_only: Only save checkpoints that improve validation score
        """
        self.checkpoint_dir = Path(checkpoint_dir)
        self.save_frequency = save_frequency
        self.max_checkpoints = max_checkpoints
        self.save_best_only = save_best_only
        
        # Create checkpoint directory
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize tracking
        self.checkpoints = []
        self.best_score = -np.inf
        self.best_checkpoint = None
        
        # Load existing checkpoints if any
        self._load_checkpoint_registry()
    
    def save_checkpoint(self, model: EnsembleClassifier, epoch: int, 
                       validation_score: float, training_score: Optional[float] = None,
                       additional_metrics: Optional[Dict[str, float]] = None,
                       feature_names: Optional[List[str]] = None) -> Optional[str]:
        """
        Save a model checkpoint.
        
        Args:
            model: Model to checkpoint
            epoch: Current epoch number
            validation_score: Validation score for this epoch
            training_score: Optional training score
            additional_metrics: Optional additional metrics
            feature_names: Optional feature names
            
        Returns:
            Checkpoint ID if saved, None if not saved
        """
        # Check if we should save this checkpoint
        should_save = self._should_save_checkpoint(epoch, validation_score)
        
        if not should_save:
            return None
        
        # Generate checkpoint ID
        checkpoint_id = f"checkpoint_epoch_{epoch}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # Create checkpoint metadata
        metadata = CheckpointMetadata(
            checkpoint_id=checkpoint_id,
            timestamp=datetime.now().isoformat(),
            epoch=epoch,
            validation_score=validation_score,
            training_score=training_score,
            model_config=model.config.__dict__ if hasattr(model, 'config') else None,
            feature_names=feature_names,
            additional_metrics=additional_metrics
        )
        
        # Save model
        model_path = self.checkpoint_dir / f"{checkpoint_id}.pkl"
        model.save(model_path)
        
        # Save metadata
        metadata_path = self.checkpoint_dir / f"{checkpoint_id}_metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(metadata.__dict__, f, indent=2, default=str)
        
        # Update tracking
        self.checkpoints.append(metadata)
        
        # Update best checkpoint if this is better
        if validation_score > self.best_score:
            self.best_score = validation_score
            self.best_checkpoint = checkpoint_id
            logger.info(f"New best checkpoint: {checkpoint_id} (score: {validation_score:.4f})")
        
        # Clean up old checkpoints if necessary
        self._cleanup_checkpoints()
        
        # Save checkpoint registry
        self._save_checkpoint_registry()
        
        logger.info(f"Saved checkpoint: {checkpoint_id}")
        
        return checkpoint_id
    
    def load_checkpoint(self, checkpoint_id: Optional[str] = None) -> Optional[EnsembleClassifier]:
        """
        Load a model checkpoint.
        
        Args:
            checkpoint_id: Specific checkpoint to load, or None for best checkpoint
            
        Returns:
            Loaded model or None if not found
        """
        if checkpoint_id is None:
            checkpoint_id = self.best_checkpoint
        
        if checkpoint_id is None:
            logger.warning("No checkpoint available to load")
            return None
        
        model_path = self.checkpoint_dir / f"{checkpoint_id}.pkl"
        
        if not model_path.exists():
            logger.error(f"Checkpoint file not found: {model_path}")
            return None
        
        try:
            model = EnsembleClassifier.load(model_path)
            logger.info(f"Loaded checkpoint: {checkpoint_id}")
            return model
        except Exception as e:
            logger.error(f"Failed to load checkpoint {checkpoint_id}: {e}")
            return None
    
    def get_checkpoint_metadata(self, checkpoint_id: str) -> Optional[CheckpointMetadata]:
        """
        Get metadata for a specific checkpoint.
        
        Args:
            checkpoint_id: Checkpoint identifier
            
        Returns:
            Checkpoint metadata or None if not found
        """
        for checkpoint in self.checkpoints:
            if checkpoint.checkpoint_id == checkpoint_id:
                return checkpoint
        return None
    
    def list_checkpoints(self) -> List[CheckpointMetadata]:
        """
        List all available checkpoints.
        
        Returns:
            List of checkpoint metadata
        """
        return self.checkpoints.copy()
    
    def get_best_checkpoint(self) -> Optional[CheckpointMetadata]:
        """
        Get the best checkpoint metadata.
        
        Returns:
            Best checkpoint metadata or None
        """
        if self.best_checkpoint:
            return self.get_checkpoint_metadata(self.best_checkpoint)
        return None
    
    def delete_checkpoint(self, checkpoint_id: str) -> bool:
        """
        Delete a specific checkpoint.
        
        Args:
            checkpoint_id: Checkpoint to delete
            
        Returns:
            True if deleted successfully, False otherwise
        """
        try:
            # Remove files
            model_path = self.checkpoint_dir / f"{checkpoint_id}.pkl"
            metadata_path = self.checkpoint_dir / f"{checkpoint_id}_metadata.json"
            
            if model_path.exists():
                model_path.unlink()
            if metadata_path.exists():
                metadata_path.unlink()
            
            # Remove from tracking
            self.checkpoints = [cp for cp in self.checkpoints if cp.checkpoint_id != checkpoint_id]
            
            # Update best checkpoint if necessary
            if self.best_checkpoint == checkpoint_id:
                self._update_best_checkpoint()
            
            # Save updated registry
            self._save_checkpoint_registry()
            
            logger.info(f"Deleted checkpoint: {checkpoint_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to delete checkpoint {checkpoint_id}: {e}")
            return False
    
    def _should_save_checkpoint(self, epoch: int, validation_score: float) -> bool:
        """Check if checkpoint should be saved."""
        # Always save if save_frequency is met
        if epoch % self.save_frequency == 0:
            return True
        
        # Save if this is the best score and save_best_only is True
        if self.save_best_only and validation_score > self.best_score:
            return True
        
        return False
    
    def _cleanup_checkpoints(self):
        """Remove old checkpoints if exceeding max_checkpoints."""
        if len(self.checkpoints) <= self.max_checkpoints:
            return
        
        # Sort by validation score (keep best ones)
        sorted_checkpoints = sorted(self.checkpoints, key=lambda x: x.validation_score, reverse=True)
        
        # Keep only the best max_checkpoints
        checkpoints_to_keep = sorted_checkpoints[:self.max_checkpoints]
        checkpoints_to_delete = sorted_checkpoints[self.max_checkpoints:]
        
        # Delete old checkpoints
        for checkpoint in checkpoints_to_delete:
            self.delete_checkpoint(checkpoint.checkpoint_id)
    
    def _update_best_checkpoint(self):
        """Update the best checkpoint after deletion."""
        if not self.checkpoints:
            self.best_checkpoint = None
            self.best_score = -np.inf
            return
        
        best_checkpoint = max(self.checkpoints, key=lambda x: x.validation_score)
        self.best_checkpoint = best_checkpoint.checkpoint_id
        self.best_score = best_checkpoint.validation_score
    
    def _save_checkpoint_registry(self):
        """Save checkpoint registry to disk."""
        registry_path = self.checkpoint_dir / "checkpoint_registry.json"
        registry_data = {
            'checkpoints': [cp.__dict__ for cp in self.checkpoints],
            'best_checkpoint': self.best_checkpoint,
            'best_score': self.best_score
        }
        
        with open(registry_path, 'w') as f:
            json.dump(registry_data, f, indent=2, default=str)
    
    def _load_checkpoint_registry(self):
        """Load checkpoint registry from disk."""
        registry_path = self.checkpoint_dir / "checkpoint_registry.json"
        
        if not registry_path.exists():
            return
        
        try:
            with open(registry_path, 'r') as f:
                registry_data = json.load(f)
            
            # Load checkpoints
            self.checkpoints = []
            for cp_data in registry_data.get('checkpoints', []):
                checkpoint = CheckpointMetadata(**cp_data)
                self.checkpoints.append(checkpoint)
            
            self.best_checkpoint = registry_data.get('best_checkpoint')
            self.best_score = registry_data.get('best_score', -np.inf)
            
            logger.info(f"Loaded {len(self.checkpoints)} checkpoints from registry")
            
        except Exception as e:
            logger.warning(f"Failed to load checkpoint registry: {e}")


class EarlyStopping:
    """
    Implements early stopping to prevent overfitting.
    
    Monitors validation metrics and stops training when no improvement
    is observed for a specified number of epochs.
    """
    
    def __init__(self, patience: int = 10, min_delta: float = 0.0001, 
                 mode: str = 'max', restore_best_weights: bool = True):
        """
        Initialize early stopping.
        
        Args:
            patience: Number of epochs to wait for improvement
            min_delta: Minimum change to qualify as improvement
            mode: 'max' for maximizing metric, 'min' for minimizing
            restore_best_weights: Whether to restore best weights when stopping
        """
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.restore_best_weights = restore_best_weights
        
        # Initialize tracking
        self.best_score = None
        self.best_epoch = 0
        self.wait = 0
        self.stopped_epoch = 0
        self.should_stop = False
        
        # Set comparison function based on mode
        if mode == 'max':
            self.monitor_op = np.greater
            self.min_delta *= 1
        else:
            self.monitor_op = np.less
            self.min_delta *= -1
    
    def __call__(self, current_score: float, epoch: int) -> bool:
        """
        Check if training should stop.
        
        Args:
            current_score: Current validation score
            epoch: Current epoch number
            
        Returns:
            True if training should stop, False otherwise
        """
        if self.best_score is None:
            self.best_score = current_score
            self.best_epoch = epoch
            return False
        
        # Check if current score is better than best score
        if self.monitor_op(current_score, self.best_score + self.min_delta):
            self.best_score = current_score
            self.best_epoch = epoch
            self.wait = 0
            logger.info(f"Validation score improved to {current_score:.4f}")
        else:
            self.wait += 1
            logger.info(f"No improvement for {self.wait} epochs (best: {self.best_score:.4f})")
            
            if self.wait >= self.patience:
                self.stopped_epoch = epoch
                self.should_stop = True
                logger.info(f"Early stopping triggered at epoch {epoch}")
                logger.info(f"Best score: {self.best_score:.4f} at epoch {self.best_epoch}")
                return True
        
        return False
    
    def reset(self):
        """Reset early stopping state."""
        self.best_score = None
        self.best_epoch = 0
        self.wait = 0
        self.stopped_epoch = 0
        self.should_stop = False
    
    def get_best_score(self) -> Optional[float]:
        """Get the best score observed."""
        return self.best_score
    
    def get_best_epoch(self) -> int:
        """Get the epoch with the best score."""
        return self.best_epoch
    
    def get_stopped_epoch(self) -> int:
        """Get the epoch where training was stopped."""
        return self.stopped_epoch


class TrainingMonitor:
    """
    Monitors training progress and provides utilities for tracking metrics.
    
    Combines checkpointing and early stopping with additional monitoring
    capabilities like learning curves and metric tracking.
    """
    
    def __init__(self, checkpoint_manager: ModelCheckpoint, early_stopping: EarlyStopping):
        """
        Initialize training monitor.
        
        Args:
            checkpoint_manager: Model checkpoint manager
            early_stopping: Early stopping instance
        """
        self.checkpoint_manager = checkpoint_manager
        self.early_stopping = early_stopping
        
        # Training history
        self.training_history = []
        self.validation_history = []
        self.metric_history = {}
        
    def update(self, model: EnsembleClassifier, epoch: int, 
              training_score: float, validation_score: float,
              additional_metrics: Optional[Dict[str, float]] = None,
              feature_names: Optional[List[str]] = None) -> bool:
        """
        Update monitoring with current epoch results.
        
        Args:
            model: Current model
            epoch: Current epoch
            training_score: Training score
            validation_score: Validation score
            additional_metrics: Additional metrics to track
            feature_names: Feature names
            
        Returns:
            True if training should continue, False if should stop
        """
        # Update history
        self.training_history.append(training_score)
        self.validation_history.append(validation_score)
        
        if additional_metrics:
            for metric_name, metric_value in additional_metrics.items():
                if metric_name not in self.metric_history:
                    self.metric_history[metric_name] = []
                self.metric_history[metric_name].append(metric_value)
        
        # Save checkpoint
        self.checkpoint_manager.save_checkpoint(
            model, epoch, validation_score, training_score, 
            additional_metrics, feature_names
        )
        
        # Check early stopping
        should_stop = self.early_stopping(validation_score, epoch)
        
        return not should_stop
    
    def get_training_summary(self) -> Dict[str, Any]:
        """Get summary of training progress."""
        summary = {
            'epochs_completed': len(self.training_history),
            'best_validation_score': self.early_stopping.get_best_score(),
            'best_epoch': self.early_stopping.get_best_epoch(),
            'early_stopped': self.early_stopping.should_stop,
            'stopped_epoch': self.early_stopping.get_stopped_epoch(),
            'training_history': self.training_history,
            'validation_history': self.validation_history,
            'metric_history': self.metric_history,
            'total_checkpoints': len(self.checkpoint_manager.checkpoints)
        }
        
        return summary
    
    def save_training_curves(self, output_path: Union[str, Path]):
        """Save training curves to file."""
        import matplotlib.pyplot as plt
        
        output_path = Path(output_path)
        
        # Create training curves plot
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # Training vs Validation Score
        epochs = range(1, len(self.training_history) + 1)
        axes[0, 0].plot(epochs, self.training_history, label='Training')
        axes[0, 0].plot(epochs, self.validation_history, label='Validation')
        axes[0, 0].set_title('Training vs Validation Score')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Score')
        axes[0, 0].legend()
        axes[0, 0].grid(True)
        
        # Mark best epoch
        if self.early_stopping.best_epoch > 0:
            axes[0, 0].axvline(x=self.early_stopping.best_epoch, color='red', 
                              linestyle='--', label='Best Epoch')
        
        # Additional metrics
        if self.metric_history:
            metric_names = list(self.metric_history.keys())
            for i, metric_name in enumerate(metric_names[:3]):  # Plot up to 3 additional metrics
                row = (i + 1) // 2
                col = (i + 1) % 2
                if row < 2:
                    axes[row, col].plot(epochs, self.metric_history[metric_name])
                    axes[row, col].set_title(f'{metric_name.title()} Over Time')
                    axes[row, col].set_xlabel('Epoch')
                    axes[row, col].set_ylabel(metric_name.title())
                    axes[row, col].grid(True)
        
        plt.tight_layout()
        plt.savefig(output_path)
        plt.close()
        
        logger.info(f"Training curves saved to {output_path}")
    
    def reset(self):
        """Reset monitoring state."""
        self.early_stopping.reset()
        self.training_history = []
        self.validation_history = []
        self.metric_history = {}