"""
Ensemble model combining multiple predictors for IPL match prediction.
Implements voting and stacking strategies for improved accuracy.
"""
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, roc_auc_score
from typing import Dict, List, Optional, Tuple
import logging
import joblib

from .xgboost_model import XGBoostPredictor

logger = logging.getLogger(__name__)

class EnsemblePredictor:
    """Ensemble model combining multiple prediction approaches."""
    
    def __init__(self, models: Optional[List] = None, weights: Optional[List[float]] = None):
        self.models = models or []
        self.weights = weights or []
        self.is_trained = False
        self.model_names = []
        
    def add_model(self, model, name: str, weight: float = 1.0):
        """Add a model to the ensemble."""
        self.models.append(model)
        self.model_names.append(name)
        self.weights.append(weight)
        logger.info(f"Added {name} to ensemble with weight {weight}")
    
    def train(self, X: pd.DataFrame, y: pd.Series, 
              validation_split: float = 0.2) -> Dict[str, Dict[str, float]]:
        """Train all models in the ensemble."""
        logger.info("Training ensemble models")
        
        if not self.models:
            raise ValueError("No models added to ensemble")
        
        # Normalize weights
        total_weight = sum(self.weights)
        self.weights = [w / total_weight for w in self.weights]
        
        model_metrics = {}
        
        for i, (model, name) in enumerate(zip(self.models, self.model_names)):
            logger.info(f"Training {name}")
            try:
                metrics = model.train(X, y, validation_split)
                model_metrics[name] = metrics
                logger.info(f"{name} validation accuracy: {metrics['val_accuracy']:.4f}")
            except Exception as e:
                logger.error(f"Error training {name}: {e}")
                model_metrics[name] = {'error': str(e)}
        
        self.is_trained = True
        return model_metrics
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Make ensemble predictions using weighted voting."""
        if not self.is_trained:
            raise ValueError("Ensemble must be trained before prediction")
        
        all_predictions = []
        
        for model, weight in zip(self.models, self.weights):
            try:
                pred = model.predict(X)
                all_predictions.append(pred * weight)
            except Exception as e:
                logger.warning(f"Model prediction failed: {e}")
                # Use neutral prediction for failed model
                all_predictions.append(np.full(len(X), 0.5) * weight)
        
        # Weighted average
        ensemble_pred = np.sum(all_predictions, axis=0)
        return (ensemble_pred > 0.5).astype(int)
    
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Predict probabilities using weighted ensemble."""
        if not self.is_trained:
            raise ValueError("Ensemble must be trained before prediction")
        
        all_probabilities = []
        
        for model, weight in zip(self.models, self.weights):
            try:
                proba = model.predict_proba(X)
                # Take probability of positive class
                positive_proba = proba[:, 1] if proba.shape[1] == 2 else proba.flatten()
                all_probabilities.append(positive_proba * weight)
            except Exception as e:
                logger.warning(f"Model probability prediction failed: {e}")
                # Use neutral probability for failed model
                all_probabilities.append(np.full(len(X), 0.5) * weight)
        
        # Weighted average probability
        ensemble_proba = np.sum(all_probabilities, axis=0)
        
        # Return as [prob_class_0, prob_class_1]
        prob_class_0 = 1 - ensemble_proba
        prob_class_1 = ensemble_proba
        return np.column_stack([prob_class_0, prob_class_1])
    
    def get_model_contributions(self, X: pd.DataFrame) -> Dict[str, np.ndarray]:
        """Get individual model contributions to predictions."""
        contributions = {}
        
        for model, name, weight in zip(self.models, self.model_names, self.weights):
            try:
                proba = model.predict_proba(X)
                positive_proba = proba[:, 1] if proba.shape[1] == 2 else proba.flatten()
                contributions[name] = positive_proba * weight
            except Exception as e:
                logger.warning(f"Could not get contribution from {name}: {e}")
                contributions[name] = np.full(len(X), 0.5) * weight
        
        return contributions
    
    def evaluate_ensemble(self, X: pd.DataFrame, y: pd.Series) -> Dict[str, float]:
        """Evaluate ensemble performance."""
        if not self.is_trained:
            raise ValueError("Ensemble must be trained before evaluation")
        
        predictions = self.predict(X)
        probabilities = self.predict_proba(X)[:, 1]
        
        metrics = {
            'accuracy': float(accuracy_score(y, predictions)),
            'auc': float(roc_auc_score(y, probabilities))
        }
        
        return metrics
    
    def save_ensemble(self, filepath: str):
        """Save entire ensemble."""
        if not self.is_trained:
            raise ValueError("Cannot save untrained ensemble")
        
        ensemble_data = {
            'model_names': list(self.model_names),
            'weights': [float(w) for w in self.weights],  # Convert to native Python float
            'models': []
        }
        
        # Save each model
        for i, (model, name) in enumerate(zip(self.models, self.model_names)):
            model_file = f"{filepath}_{name}_model"
            try:
                model.save_model(model_file)
                ensemble_data['models'].append(model_file)
            except Exception as e:
                logger.error(f"Failed to save {name}: {e}")
                ensemble_data['models'].append(None)
        
        # Save ensemble metadata
        joblib.dump(ensemble_data, f"{filepath}_ensemble.pkl")
        logger.info(f"Ensemble saved to {filepath}")
    
    def load_ensemble(self, filepath: str):
        """Load entire ensemble."""
        # Load ensemble metadata
        ensemble_data = joblib.load(f"{filepath}_ensemble.pkl")
        
        self.model_names = ensemble_data['model_names']
        self.weights = ensemble_data['weights']
        self.models = []
        
        # Load each model
        for model_file, name in zip(ensemble_data['models'], self.model_names):
            if model_file:
                try:
                    if 'xgboost' in name.lower():
                        model = XGBoostPredictor()
                    else:
                        # Add other model types as needed
                        continue
                    
                    model.load_model(model_file)
                    self.models.append(model)
                except Exception as e:
                    logger.error(f"Failed to load {name}: {e}")
                    continue
        
        if self.models:
            self.is_trained = True
            logger.info(f"Ensemble loaded from {filepath}")
        else:
            raise ValueError("No models could be loaded")
    
    def get_feature_importance(self, top_n: int = 20) -> Dict[str, pd.Series]:
        """Get feature importance from models that support it."""
        importance_dict = {}
        
        for model, name in zip(self.models, self.model_names):
            try:
                if hasattr(model, 'get_feature_importance'):
                    importance = model.get_feature_importance(top_n)
                    importance_dict[name] = importance
            except Exception as e:
                logger.warning(f"Could not get feature importance from {name}: {e}")
        
        return importance_dict