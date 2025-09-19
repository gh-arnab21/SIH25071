"""
Ensemble Classifier for Rockfall Prediction System.

This module implements an ensemble classifier that combines Random Forest, XGBoost,
and Neural Network models to predict rockfall risk levels with confidence scores
and uncertainty estimates.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union, Any
from dataclasses import dataclass
import joblib
import logging
from pathlib import Path

# ML Libraries
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from sklearn.preprocessing import StandardScaler
from sklearn.base import BaseEstimator, ClassifierMixin, is_classifier
from sklearn.utils.validation import check_X_y, check_array
import xgboost as xgb
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# Local imports
from ...data.schema import RiskLevel, RockfallPrediction


logger = logging.getLogger(__name__)


@dataclass
class EnsembleConfig:
    """Configuration for ensemble classifier."""
    # Random Forest parameters
    rf_n_estimators: int = 100
    rf_max_depth: Optional[int] = None
    rf_min_samples_split: int = 2
    rf_min_samples_leaf: int = 1
    rf_random_state: int = 42
    
    # XGBoost parameters
    xgb_n_estimators: int = 100
    xgb_max_depth: int = 6
    xgb_learning_rate: float = 0.1
    xgb_subsample: float = 0.8
    xgb_colsample_bytree: float = 0.8
    xgb_random_state: int = 42
    
    # Neural Network parameters
    nn_hidden_layers: List[int] = None
    nn_dropout_rate: float = 0.3
    nn_learning_rate: float = 0.001
    nn_batch_size: int = 32
    nn_epochs: int = 100
    nn_early_stopping_patience: int = 10
    
    # Ensemble parameters
    ensemble_method: str = "voting"  # "voting" or "stacking"
    voting_type: str = "soft"  # "hard" or "soft"
    cv_folds: int = 5
    random_state: int = 42
    
    def __post_init__(self):
        """Set default values for mutable fields."""
        if self.nn_hidden_layers is None:
            self.nn_hidden_layers = [128, 64, 32]


class NeuralNetworkClassifier(BaseEstimator, ClassifierMixin):
    """Neural Network classifier compatible with scikit-learn interface."""
    
    def __init__(self, hidden_layers: List[int] = None, dropout_rate: float = 0.3,
                 learning_rate: float = 0.001, batch_size: int = 32,
                 epochs: int = 100, early_stopping_patience: int = 10,
                 random_state: int = 42):
        """Initialize Neural Network classifier."""
        self.hidden_layers = hidden_layers or [128, 64, 32]
        self.dropout_rate = dropout_rate
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.epochs = epochs
        self.early_stopping_patience = early_stopping_patience
        self.random_state = random_state
        self.model = None
        self.scaler = StandardScaler()
        self.classes_ = None
        self.n_classes_ = None
        
    def _more_tags(self):
        """Return tags for scikit-learn compatibility."""
        return {
            'requires_fit': True,
            'requires_y': True,
            'X_types': ['2darray'],
            'y_types': ['1dlabels'],
            'multiclass': True,
            'binary_only': False,
            'multilabel': False,
            'multioutput': False,
            'multioutput_only': False,
            'no_validation': False,
            'non_deterministic': False,
            'pairwise': False,
            'poor_score': False,
            'requires_positive_X': False,
            'stateless': False,
            '_xfail_checks': {}
        }
        
    def _build_model(self, input_dim: int, n_classes: int) -> keras.Model:
        """Build neural network model."""
        tf.random.set_seed(self.random_state)
        
        model = keras.Sequential()
        model.add(layers.Input(shape=(input_dim,)))
        
        # Hidden layers
        for i, units in enumerate(self.hidden_layers):
            model.add(layers.Dense(units, activation='relu', name=f'hidden_{i+1}'))
            model.add(layers.Dropout(self.dropout_rate, name=f'dropout_{i+1}'))
        
        # Output layer
        if n_classes == 2:
            model.add(layers.Dense(1, activation='sigmoid', name='output'))
            loss = 'binary_crossentropy'
        else:
            model.add(layers.Dense(n_classes, activation='softmax', name='output'))
            loss = 'sparse_categorical_crossentropy'
        
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=self.learning_rate),
            loss=loss,
            metrics=['accuracy']
        )
        
        return model
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> 'NeuralNetworkClassifier':
        """Fit the neural network classifier."""
        # Validate input
        X, y = check_X_y(X, y)
        
        # Store classes
        self.classes_ = np.unique(y)
        self.n_classes_ = len(self.classes_)
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        # Build model
        self.model = self._build_model(X.shape[1], self.n_classes_)
        
        # Prepare callbacks
        callbacks = [
            keras.callbacks.EarlyStopping(
                patience=self.early_stopping_patience,
                restore_best_weights=True,
                verbose=0
            )
        ]
        
        # Train model
        self.model.fit(
            X_scaled, y,
            batch_size=self.batch_size,
            epochs=self.epochs,
            validation_split=0.2,
            callbacks=callbacks,
            verbose=0
        )
        
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions."""
        if self.model is None:
            raise ValueError("Model not fitted yet.")
        
        # Validate input
        X = check_array(X)
        
        X_scaled = self.scaler.transform(X)
        predictions = self.model.predict(X_scaled, verbose=0)
        
        if self.n_classes_ == 2:
            return (predictions > 0.5).astype(int).flatten()
        else:
            return np.argmax(predictions, axis=1)
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict class probabilities."""
        if self.model is None:
            raise ValueError("Model not fitted yet.")
        
        # Validate input
        X = check_array(X)
        
        X_scaled = self.scaler.transform(X)
        predictions = self.model.predict(X_scaled, verbose=0)
        
        if self.n_classes_ == 2:
            proba = np.column_stack([1 - predictions.flatten(), predictions.flatten()])
        else:
            proba = predictions
        
        return proba
    
    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        """Return the mean accuracy on the given test data and labels."""
        predictions = self.predict(X)
        return np.mean(predictions == y)


class EnsembleClassifier:
    """
    Ensemble classifier combining Random Forest, XGBoost, and Neural Network
    for rockfall risk prediction with confidence scores and uncertainty estimates.
    """
    
    def __init__(self, config: Optional[EnsembleConfig] = None):
        """Initialize ensemble classifier."""
        self.config = config or EnsembleConfig()
        self.rf_classifier = None
        self.xgb_classifier = None
        self.nn_classifier = None
        self.ensemble_model = None
        self.scaler = StandardScaler()
        self.is_fitted = False
        self.feature_names = None
        self.classes_ = None
        
        # Initialize individual classifiers
        self._initialize_classifiers()
        
    def _initialize_classifiers(self):
        """Initialize individual classifiers with configuration."""
        # Random Forest
        self.rf_classifier = RandomForestClassifier(
            n_estimators=self.config.rf_n_estimators,
            max_depth=self.config.rf_max_depth,
            min_samples_split=self.config.rf_min_samples_split,
            min_samples_leaf=self.config.rf_min_samples_leaf,
            random_state=self.config.rf_random_state,
            n_jobs=-1
        )
        
        # XGBoost
        self.xgb_classifier = xgb.XGBClassifier(
            n_estimators=self.config.xgb_n_estimators,
            max_depth=self.config.xgb_max_depth,
            learning_rate=self.config.xgb_learning_rate,
            subsample=self.config.xgb_subsample,
            colsample_bytree=self.config.xgb_colsample_bytree,
            random_state=self.config.xgb_random_state,
            eval_metric='mlogloss',
            verbosity=0
        )
        
        # Neural Network
        self.nn_classifier = NeuralNetworkClassifier(
            hidden_layers=self.config.nn_hidden_layers,
            dropout_rate=self.config.nn_dropout_rate,
            learning_rate=self.config.nn_learning_rate,
            batch_size=self.config.nn_batch_size,
            epochs=self.config.nn_epochs,
            early_stopping_patience=self.config.nn_early_stopping_patience,
            random_state=self.config.random_state
        )
        
    def fit(self, X: Union[np.ndarray, pd.DataFrame], y: Union[np.ndarray, pd.Series],
            feature_names: Optional[List[str]] = None) -> 'EnsembleClassifier':
        """
        Fit the ensemble classifier.
        
        Args:
            X: Feature matrix
            y: Target labels (RiskLevel enum values or integers)
            feature_names: Optional feature names for interpretability
            
        Returns:
            Fitted ensemble classifier
        """
        logger.info("Starting ensemble classifier training...")
        
        # Convert inputs to numpy arrays
        X = np.array(X)
        y = np.array(y)
        
        # Convert RiskLevel enums to integers if necessary
        if len(y) > 0 and isinstance(y[0], RiskLevel):
            y = np.array([risk.value for risk in y])
        
        # Store feature names and classes
        self.feature_names = feature_names or [f"feature_{i}" for i in range(X.shape[1])]
        self.classes_ = np.unique(y)
        
        # Scale features for neural network
        X_scaled = self.scaler.fit_transform(X)
        
        # Train individual classifiers
        logger.info("Training Random Forest...")
        self.rf_classifier.fit(X, y)
        
        logger.info("Training XGBoost...")
        self.xgb_classifier.fit(X, y)
        
        logger.info("Training Neural Network...")
        self.nn_classifier.fit(X_scaled, y)
        
        # Create ensemble model (only use VotingClassifier for RF and XGB)
        if self.config.ensemble_method == "voting":
            self.ensemble_model = VotingClassifier(
                estimators=[
                    ('rf', self.rf_classifier),
                    ('xgb', self.xgb_classifier)
                ],
                voting=self.config.voting_type
            )
            self.ensemble_model.fit(X, y)
        
        self.is_fitted = True
        logger.info("Ensemble classifier training completed.")
        
        return self
    
    def predict(self, X: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
        """
        Make predictions using the ensemble classifier.
        
        Args:
            X: Feature matrix
            
        Returns:
            Predicted risk levels as integers
        """
        if not self.is_fitted:
            raise ValueError("Ensemble classifier not fitted yet.")
        
        X = np.array(X)
        
        if self.config.ensemble_method == "voting":
            # Get predictions from RF and XGB ensemble
            ensemble_pred = self.ensemble_model.predict(X)
            # Get NN predictions separately
            nn_pred = self.nn_classifier.predict(self.scaler.transform(X))
            
            # Combine with simple majority voting
            predictions = np.column_stack([ensemble_pred, nn_pred])
            return np.array([np.bincount(row).argmax() for row in predictions])
        else:
            # Stacking method - simple averaging for now
            rf_pred = self.rf_classifier.predict(X)
            xgb_pred = self.xgb_classifier.predict(X)
            nn_pred = self.nn_classifier.predict(self.scaler.transform(X))
            
            # Simple majority voting
            predictions = np.column_stack([rf_pred, xgb_pred, nn_pred])
            return np.array([np.bincount(row).argmax() for row in predictions])
    
    def predict_proba(self, X: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
        """
        Predict class probabilities using the ensemble classifier.
        
        Args:
            X: Feature matrix
            
        Returns:
            Class probabilities for each risk level
        """
        if not self.is_fitted:
            raise ValueError("Ensemble classifier not fitted yet.")
        
        X = np.array(X)
        
        if self.config.ensemble_method == "voting":
            # Get probabilities from RF and XGB ensemble
            ensemble_proba = self.ensemble_model.predict_proba(X)
            # Get NN probabilities separately
            nn_proba = self.nn_classifier.predict_proba(self.scaler.transform(X))
            
            # Average the probabilities
            return (ensemble_proba + nn_proba) / 2
        else:
            # Stacking method - average probabilities
            rf_proba = self.rf_classifier.predict_proba(X)
            xgb_proba = self.xgb_classifier.predict_proba(X)
            nn_proba = self.nn_classifier.predict_proba(self.scaler.transform(X))
            
            return (rf_proba + xgb_proba + nn_proba) / 3
    
    def predict_with_confidence(self, X: Union[np.ndarray, pd.DataFrame]) -> List[RockfallPrediction]:
        """
        Make predictions with confidence scores and uncertainty estimates.
        
        Args:
            X: Feature matrix
            
        Returns:
            List of RockfallPrediction objects with confidence and uncertainty
        """
        if not self.is_fitted:
            raise ValueError("Ensemble classifier not fitted yet.")
        
        X = np.array(X)
        
        # Get predictions and probabilities
        predictions = self.predict(X)
        probabilities = self.predict_proba(X)
        
        # Calculate individual model predictions for uncertainty estimation
        rf_proba = self.rf_classifier.predict_proba(X)
        xgb_proba = self.xgb_classifier.predict_proba(X)
        nn_proba = self.nn_classifier.predict_proba(self.scaler.transform(X))
        
        results = []
        
        for i in range(len(X)):
            # Get prediction and confidence
            pred_class = predictions[i]
            confidence = probabilities[i, pred_class]
            
            # Calculate uncertainty as variance across models
            model_probas = np.array([rf_proba[i], xgb_proba[i], nn_proba[i]])
            uncertainty = np.var(model_probas[:, pred_class])
            
            # Get contributing factors (feature importance from Random Forest)
            contributing_factors = {}
            if hasattr(self.rf_classifier, 'feature_importances_'):
                importances = self.rf_classifier.feature_importances_
                for j, importance in enumerate(importances):
                    feature_name = self.feature_names[j] if j < len(self.feature_names) else f"feature_{j}"
                    contributing_factors[feature_name] = float(importance)
            
            # Create prediction object
            prediction = RockfallPrediction(
                risk_level=RiskLevel(pred_class),
                confidence_score=float(confidence),
                contributing_factors=contributing_factors,
                uncertainty_estimate=float(uncertainty),
                model_version="ensemble_v1.0"
            )
            
            results.append(prediction)
        
        return results
    
    def get_feature_importance(self) -> Dict[str, float]:
        """
        Get feature importance scores from the ensemble.
        
        Returns:
            Dictionary mapping feature names to importance scores
        """
        if not self.is_fitted:
            raise ValueError("Ensemble classifier not fitted yet.")
        
        # Combine feature importances from Random Forest and XGBoost
        rf_importance = self.rf_classifier.feature_importances_
        xgb_importance = self.xgb_classifier.feature_importances_
        
        # Average the importances
        avg_importance = (rf_importance + xgb_importance) / 2
        
        importance_dict = {}
        for i, importance in enumerate(avg_importance):
            feature_name = self.feature_names[i] if i < len(self.feature_names) else f"feature_{i}"
            importance_dict[feature_name] = float(importance)
        
        return importance_dict
    
    def evaluate(self, X: Union[np.ndarray, pd.DataFrame], y: Union[np.ndarray, pd.Series]) -> Dict[str, Any]:
        """
        Evaluate the ensemble classifier performance.
        
        Args:
            X: Feature matrix
            y: True labels
            
        Returns:
            Dictionary containing evaluation metrics
        """
        if not self.is_fitted:
            raise ValueError("Ensemble classifier not fitted yet.")
        
        X = np.array(X)
        y = np.array(y)
        
        # Convert RiskLevel enums to integers if necessary
        if len(y) > 0 and isinstance(y[0], RiskLevel):
            y = np.array([risk.value for risk in y])
        
        # Make predictions
        y_pred = self.predict(X)
        y_proba = self.predict_proba(X)
        
        # Calculate metrics
        results = {
            'accuracy': float(np.mean(y_pred == y)),
            'classification_report': classification_report(y, y_pred, output_dict=True),
            'confusion_matrix': confusion_matrix(y, y_pred).tolist()
        }
        
        # Add AUC scores for multiclass
        if len(self.classes_) > 2:
            try:
                auc_scores = {}
                for i, class_label in enumerate(self.classes_):
                    y_binary = (y == class_label).astype(int)
                    auc_scores[f'auc_class_{class_label}'] = roc_auc_score(y_binary, y_proba[:, i])
                results['auc_scores'] = auc_scores
            except ValueError:
                logger.warning("Could not calculate AUC scores for multiclass classification")
        
        return results
    
    def cross_validate(self, X: Union[np.ndarray, pd.DataFrame], y: Union[np.ndarray, pd.Series]) -> Dict[str, float]:
        """
        Perform cross-validation on the ensemble classifier.
        
        Args:
            X: Feature matrix
            y: Target labels
            
        Returns:
            Dictionary containing cross-validation scores
        """
        X = np.array(X)
        y = np.array(y)
        
        # Convert RiskLevel enums to integers if necessary
        if len(y) > 0 and isinstance(y[0], RiskLevel):
            y = np.array([risk.value for risk in y])
        
        cv = StratifiedKFold(n_splits=self.config.cv_folds, shuffle=True, random_state=self.config.random_state)
        
        # Cross-validate individual models
        rf_scores = cross_val_score(self.rf_classifier, X, y, cv=cv, scoring='accuracy')
        xgb_scores = cross_val_score(self.xgb_classifier, X, y, cv=cv, scoring='accuracy')
        
        # For neural network, we need to handle scaling
        nn_scores = []
        for train_idx, val_idx in cv.split(X, y):
            X_train, X_val = X[train_idx], X[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]
            
            # Create and fit scaler
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_val_scaled = scaler.transform(X_val)
            
            # Create and fit neural network
            nn = NeuralNetworkClassifier(
                hidden_layers=self.config.nn_hidden_layers,
                dropout_rate=self.config.nn_dropout_rate,
                learning_rate=self.config.nn_learning_rate,
                batch_size=self.config.nn_batch_size,
                epochs=self.config.nn_epochs,
                early_stopping_patience=self.config.nn_early_stopping_patience,
                random_state=self.config.random_state
            )
            nn.fit(X_train_scaled, y_train)
            score = nn.score(X_val_scaled, y_val)
            nn_scores.append(score)
        
        nn_scores = np.array(nn_scores)
        
        return {
            'rf_mean_accuracy': float(rf_scores.mean()),
            'rf_std_accuracy': float(rf_scores.std()),
            'xgb_mean_accuracy': float(xgb_scores.mean()),
            'xgb_std_accuracy': float(xgb_scores.std()),
            'nn_mean_accuracy': float(nn_scores.mean()),
            'nn_std_accuracy': float(nn_scores.std())
        }
    
    def save(self, filepath: Union[str, Path]) -> None:
        """
        Save the ensemble classifier to disk.
        
        Args:
            filepath: Path to save the model
        """
        if not self.is_fitted:
            raise ValueError("Cannot save unfitted ensemble classifier.")
        
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        # Save the entire ensemble classifier
        joblib.dump(self, filepath)
        logger.info(f"Ensemble classifier saved to {filepath}")
    
    @classmethod
    def load(cls, filepath: Union[str, Path]) -> 'EnsembleClassifier':
        """
        Load an ensemble classifier from disk.
        
        Args:
            filepath: Path to the saved model
            
        Returns:
            Loaded ensemble classifier
        """
        filepath = Path(filepath)
        if not filepath.exists():
            raise FileNotFoundError(f"Model file not found: {filepath}")
        
        ensemble = joblib.load(filepath)
        logger.info(f"Ensemble classifier loaded from {filepath}")
        return ensemble