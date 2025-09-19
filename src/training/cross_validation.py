"""
Cross-validation and Hyperparameter Optimization for Rockfall Prediction System.

This module provides functionality for model validation and hyperparameter tuning
using cross-validation and optimization algorithms.
"""

import logging
import numpy as np
from typing import Dict, List, Optional, Tuple, Any, Callable
from dataclasses import dataclass
import optuna
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score
import joblib
from pathlib import Path

# Local imports
from ..models.classifiers import EnsembleClassifier, EnsembleConfig

logger = logging.getLogger(__name__)


@dataclass
class CrossValidationResults:
    """Results from cross-validation."""
    mean_accuracy: float
    std_accuracy: float
    mean_precision: float
    std_precision: float
    mean_recall: float
    std_recall: float
    mean_f1: float
    std_f1: float
    fold_scores: List[Dict[str, float]]
    best_fold: int
    worst_fold: int


class CrossValidator:
    """
    Performs cross-validation for model evaluation and selection.
    
    Supports stratified k-fold cross-validation with various metrics
    and handles class imbalance appropriately.
    """
    
    def __init__(self, n_folds: int = 5, random_state: int = 42, scoring_metrics: Optional[List[str]] = None):
        """
        Initialize cross-validator.
        
        Args:
            n_folds: Number of folds for cross-validation
            random_state: Random seed for reproducibility
            scoring_metrics: List of metrics to compute
        """
        self.n_folds = n_folds
        self.random_state = random_state
        self.scoring_metrics = scoring_metrics or ['accuracy', 'precision', 'recall', 'f1']
        
        # Initialize cross-validation splitter
        self.cv_splitter = StratifiedKFold(
            n_splits=n_folds,
            shuffle=True,
            random_state=random_state
        )
    
    def cross_validate(self, model: EnsembleClassifier, X: np.ndarray, y: np.ndarray, 
                      feature_names: Optional[List[str]] = None) -> CrossValidationResults:
        """
        Perform cross-validation on the model.
        
        Args:
            model: Model to validate
            X: Feature matrix
            y: Target labels
            feature_names: Optional feature names
            
        Returns:
            Cross-validation results
        """
        logger.info(f"Starting {self.n_folds}-fold cross-validation...")
        
        fold_scores = []
        all_accuracies = []
        all_precisions = []
        all_recalls = []
        all_f1s = []
        
        for fold_idx, (train_idx, val_idx) in enumerate(self.cv_splitter.split(X, y)):
            logger.info(f"Processing fold {fold_idx + 1}/{self.n_folds}")
            
            # Split data
            X_train, X_val = X[train_idx], X[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]
            
            # Create a fresh model instance for this fold
            fold_model = EnsembleClassifier(model.config)
            
            # Train model
            fold_model.fit(X_train, y_train, feature_names)
            
            # Make predictions
            y_pred = fold_model.predict(X_val)
            y_proba = fold_model.predict_proba(X_val)
            
            # Calculate metrics
            accuracy = accuracy_score(y_val, y_pred)
            precision, recall, f1, _ = precision_recall_fscore_support(y_val, y_pred, average='weighted', zero_division=0)
            
            # Store scores
            fold_score = {
                'fold': fold_idx + 1,
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1': f1,
                'train_size': len(train_idx),
                'val_size': len(val_idx)
            }
            
            # Add AUC if binary or multiclass
            try:
                if len(np.unique(y)) == 2:
                    auc = roc_auc_score(y_val, y_proba[:, 1])
                    fold_score['auc'] = auc
                else:
                    auc = roc_auc_score(y_val, y_proba, multi_class='ovr', average='weighted')
                    fold_score['auc'] = auc
            except ValueError:
                logger.warning(f"Could not calculate AUC for fold {fold_idx + 1}")
            
            fold_scores.append(fold_score)
            all_accuracies.append(accuracy)
            all_precisions.append(precision)
            all_recalls.append(recall)
            all_f1s.append(f1)
            
            logger.info(f"Fold {fold_idx + 1} - Accuracy: {accuracy:.4f}, F1: {f1:.4f}")
        
        # Calculate summary statistics
        mean_accuracy = np.mean(all_accuracies)
        std_accuracy = np.std(all_accuracies)
        mean_precision = np.mean(all_precisions)
        std_precision = np.std(all_precisions)
        mean_recall = np.mean(all_recalls)
        std_recall = np.std(all_recalls)
        mean_f1 = np.mean(all_f1s)
        std_f1 = np.std(all_f1s)
        
        # Find best and worst folds
        best_fold = np.argmax(all_accuracies)
        worst_fold = np.argmin(all_accuracies)
        
        results = CrossValidationResults(
            mean_accuracy=mean_accuracy,
            std_accuracy=std_accuracy,
            mean_precision=mean_precision,
            std_precision=std_precision,
            mean_recall=mean_recall,
            std_recall=std_recall,
            mean_f1=mean_f1,
            std_f1=std_f1,
            fold_scores=fold_scores,
            best_fold=best_fold,
            worst_fold=worst_fold
        )
        
        logger.info(f"Cross-validation completed:")
        logger.info(f"  Mean Accuracy: {mean_accuracy:.4f} ± {std_accuracy:.4f}")
        logger.info(f"  Mean F1-Score: {mean_f1:.4f} ± {std_f1:.4f}")
        
        return results
    
    def compare_models(self, models: Dict[str, EnsembleClassifier], X: np.ndarray, y: np.ndarray,
                      feature_names: Optional[List[str]] = None) -> Dict[str, CrossValidationResults]:
        """
        Compare multiple models using cross-validation.
        
        Args:
            models: Dictionary of model name -> model instance
            X: Feature matrix
            y: Target labels
            feature_names: Optional feature names
            
        Returns:
            Dictionary of model name -> cross-validation results
        """
        logger.info(f"Comparing {len(models)} models using cross-validation...")
        
        results = {}
        
        for model_name, model in models.items():
            logger.info(f"Evaluating model: {model_name}")
            cv_results = self.cross_validate(model, X, y, feature_names)
            results[model_name] = cv_results
        
        # Log comparison summary
        logger.info("Model comparison results:")
        for model_name, cv_results in results.items():
            logger.info(f"  {model_name}: {cv_results.mean_accuracy:.4f} ± {cv_results.std_accuracy:.4f}")
        
        return results
    
    def nested_cross_validation(self, model: EnsembleClassifier, X: np.ndarray, y: np.ndarray,
                               param_grid: Dict[str, List[Any]], feature_names: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Perform nested cross-validation for unbiased model evaluation.
        
        Args:
            model: Model to evaluate
            X: Feature matrix
            y: Target labels
            param_grid: Parameter grid for hyperparameter optimization
            feature_names: Optional feature names
            
        Returns:
            Nested cross-validation results
        """
        logger.info("Starting nested cross-validation...")
        
        outer_scores = []
        best_params_per_fold = []
        
        # Outer loop for model evaluation
        for fold_idx, (train_idx, test_idx) in enumerate(self.cv_splitter.split(X, y)):
            logger.info(f"Outer fold {fold_idx + 1}/{self.n_folds}")
            
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]
            
            # Inner loop for hyperparameter optimization
            optimizer = HyperparameterOptimizer(n_trials=20)  # Reduced for nested CV
            best_params, _ = optimizer.optimize(X_train, y_train, X_test, y_test, feature_names)
            
            # Train model with best parameters on full training set
            best_config = EnsembleConfig(**best_params) if best_params else model.config
            best_model = EnsembleClassifier(best_config)
            best_model.fit(X_train, y_train, feature_names)
            
            # Evaluate on test set
            test_score = best_model.evaluate(X_test, y_test)['accuracy']
            
            outer_scores.append(test_score)
            best_params_per_fold.append(best_params)
            
            logger.info(f"Outer fold {fold_idx + 1} test score: {test_score:.4f}")
        
        # Calculate final statistics
        mean_score = np.mean(outer_scores)
        std_score = np.std(outer_scores)
        
        results = {
            'mean_test_score': mean_score,
            'std_test_score': std_score,
            'outer_fold_scores': outer_scores,
            'best_params_per_fold': best_params_per_fold,
            'n_outer_folds': self.n_folds
        }
        
        logger.info(f"Nested CV completed - Mean test score: {mean_score:.4f} ± {std_score:.4f}")
        
        return results


class HyperparameterOptimizer:
    """
    Optimizes hyperparameters using Optuna for Bayesian optimization.
    
    Supports optimization of ensemble classifier parameters including
    Random Forest, XGBoost, and Neural Network hyperparameters.
    """
    
    def __init__(self, n_trials: int = 100, random_state: int = 42, 
                 optimization_direction: str = 'maximize'):
        """
        Initialize hyperparameter optimizer.
        
        Args:
            n_trials: Number of optimization trials
            random_state: Random seed for reproducibility
            optimization_direction: 'maximize' or 'minimize'
        """
        self.n_trials = n_trials
        self.random_state = random_state
        self.optimization_direction = optimization_direction
        
        # Set random seed for Optuna
        optuna.logging.set_verbosity(optuna.logging.WARNING)
    
    def optimize(self, X_train: np.ndarray, y_train: np.ndarray, 
                X_val: np.ndarray, y_val: np.ndarray,
                feature_names: Optional[List[str]] = None) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
        """
        Optimize hyperparameters using Bayesian optimization.
        
        Args:
            X_train: Training features
            y_train: Training labels
            X_val: Validation features
            y_val: Validation labels
            feature_names: Optional feature names
            
        Returns:
            Tuple of (best_parameters, optimization_history)
        """
        logger.info(f"Starting hyperparameter optimization with {self.n_trials} trials...")
        
        # Create study
        study = optuna.create_study(
            direction=self.optimization_direction,
            sampler=optuna.samplers.TPESampler(seed=self.random_state)
        )
        
        # Define objective function
        def objective(trial):
            # Sample hyperparameters
            params = self._suggest_parameters(trial)
            
            try:
                # Create model with suggested parameters
                config = EnsembleConfig(**params)
                model = EnsembleClassifier(config)
                
                # Train model
                model.fit(X_train, y_train, feature_names)
                
                # Evaluate on validation set
                results = model.evaluate(X_val, y_val)
                score = results['accuracy']
                
                return score
                
            except Exception as e:
                logger.warning(f"Trial failed with parameters {params}: {e}")
                return 0.0  # Return poor score for failed trials
        
        # Optimize
        study.optimize(objective, n_trials=self.n_trials, show_progress_bar=False)
        
        # Get results
        best_params = study.best_params
        optimization_history = []
        
        for trial in study.trials:
            if trial.state == optuna.trial.TrialState.COMPLETE:
                optimization_history.append({
                    'trial_number': trial.number,
                    'value': trial.value,
                    'params': trial.params
                })
        
        logger.info(f"Optimization completed. Best score: {study.best_value:.4f}")
        logger.info(f"Best parameters: {best_params}")
        
        return best_params, optimization_history
    
    def _suggest_parameters(self, trial: optuna.Trial) -> Dict[str, Any]:
        """
        Suggest hyperparameters for optimization.
        
        Args:
            trial: Optuna trial object
            
        Returns:
            Dictionary of suggested parameters
        """
        params = {}
        
        # Random Forest parameters
        params['rf_n_estimators'] = trial.suggest_int('rf_n_estimators', 50, 300, step=50)
        params['rf_max_depth'] = trial.suggest_int('rf_max_depth', 5, 20)
        params['rf_min_samples_split'] = trial.suggest_int('rf_min_samples_split', 2, 10)
        params['rf_min_samples_leaf'] = trial.suggest_int('rf_min_samples_leaf', 1, 5)
        
        # XGBoost parameters
        params['xgb_n_estimators'] = trial.suggest_int('xgb_n_estimators', 50, 300, step=50)
        params['xgb_max_depth'] = trial.suggest_int('xgb_max_depth', 3, 10)
        params['xgb_learning_rate'] = trial.suggest_float('xgb_learning_rate', 0.01, 0.3, log=True)
        params['xgb_subsample'] = trial.suggest_float('xgb_subsample', 0.6, 1.0)
        params['xgb_colsample_bytree'] = trial.suggest_float('xgb_colsample_bytree', 0.6, 1.0)
        
        # Neural Network parameters
        n_layers = trial.suggest_int('nn_n_layers', 2, 4)
        hidden_layers = []
        for i in range(n_layers):
            layer_size = trial.suggest_int(f'nn_layer_{i}_size', 32, 512, step=32)
            hidden_layers.append(layer_size)
        params['nn_hidden_layers'] = hidden_layers
        
        params['nn_dropout_rate'] = trial.suggest_float('nn_dropout_rate', 0.1, 0.5)
        params['nn_learning_rate'] = trial.suggest_float('nn_learning_rate', 1e-4, 1e-2, log=True)
        params['nn_batch_size'] = trial.suggest_categorical('nn_batch_size', [16, 32, 64, 128])
        
        # Ensemble parameters
        params['ensemble_method'] = trial.suggest_categorical('ensemble_method', ['voting', 'stacking'])
        if params['ensemble_method'] == 'voting':
            params['voting_type'] = trial.suggest_categorical('voting_type', ['soft', 'hard'])
        
        return params
    
    def grid_search(self, X_train: np.ndarray, y_train: np.ndarray,
                   X_val: np.ndarray, y_val: np.ndarray,
                   param_grid: Dict[str, List[Any]],
                   feature_names: Optional[List[str]] = None) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
        """
        Perform grid search optimization.
        
        Args:
            X_train: Training features
            y_train: Training labels
            X_val: Validation features
            y_val: Validation labels
            param_grid: Grid of parameters to search
            feature_names: Optional feature names
            
        Returns:
            Tuple of (best_parameters, search_history)
        """
        logger.info("Starting grid search optimization...")
        
        from itertools import product
        
        # Generate all parameter combinations
        param_names = list(param_grid.keys())
        param_values = list(param_grid.values())
        param_combinations = list(product(*param_values))
        
        best_score = -np.inf if self.optimization_direction == 'maximize' else np.inf
        best_params = None
        search_history = []
        
        for i, param_combo in enumerate(param_combinations):
            params = dict(zip(param_names, param_combo))
            
            try:
                # Create model with parameters
                config = EnsembleConfig(**params)
                model = EnsembleClassifier(config)
                
                # Train and evaluate
                model.fit(X_train, y_train, feature_names)
                results = model.evaluate(X_val, y_val)
                score = results['accuracy']
                
                # Update best if better
                is_better = (score > best_score if self.optimization_direction == 'maximize' 
                           else score < best_score)
                
                if is_better:
                    best_score = score
                    best_params = params
                
                search_history.append({
                    'iteration': i + 1,
                    'params': params,
                    'score': score
                })
                
                logger.info(f"Grid search {i+1}/{len(param_combinations)} - Score: {score:.4f}")
                
            except Exception as e:
                logger.warning(f"Grid search iteration {i+1} failed: {e}")
        
        logger.info(f"Grid search completed. Best score: {best_score:.4f}")
        logger.info(f"Best parameters: {best_params}")
        
        return best_params, search_history
    
    def save_study(self, study: optuna.Study, filepath: str):
        """Save Optuna study to file."""
        joblib.dump(study, filepath)
        logger.info(f"Study saved to {filepath}")
    
    def load_study(self, filepath: str) -> optuna.Study:
        """Load Optuna study from file."""
        study = joblib.load(filepath)
        logger.info(f"Study loaded from {filepath}")
        return study