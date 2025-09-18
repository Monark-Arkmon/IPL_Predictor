"""
XGBoost model for IPL match prediction.
Optimized for sports analytics with interpretable features.
"""
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import train_test_split
import joblib
from typing import Dict, Optional
import logging

logger = logging.getLogger(__name__)

class XGBoostPredictor:
    """XGBoost model optimized for IPL match prediction."""
    
    def __init__(self, model_params: Optional[Dict] = None):
        self.model_params = model_params or {
            'objective': 'binary:logistic',
            'eval_metric': 'logloss',
            'max_depth': 6,
            'learning_rate': 0.1,
            'n_estimators': 200,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'random_state': 42,
            'n_jobs': -1
        }
        self.model = None
        self.feature_importance = None
        self.is_trained = False
        
    def train(self, X: pd.DataFrame, y: pd.Series, 
              validation_split: float = 0.2) -> Dict[str, float]:
        """Train XGBoost model with validation."""
        logger.info("Training XGBoost model")
        
        # Split data
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=validation_split, random_state=42, stratify=y
        )
        
        # Create XGBoost model
        self.model = xgb.XGBClassifier(**self.model_params)
        
        # Train with early stopping
        self.model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            verbose=False
        )
        
        # Calculate metrics
        train_pred = self.model.predict(X_train)
        val_pred = self.model.predict(X_val)
        train_proba = self.model.predict_proba(X_train)[:, 1]
        val_proba = self.model.predict_proba(X_val)[:, 1]
        
        metrics = {
            'train_accuracy': float(accuracy_score(y_train, train_pred)),
            'val_accuracy': float(accuracy_score(y_val, val_pred)),
            'train_auc': float(roc_auc_score(y_train, train_proba)),
            'val_auc': float(roc_auc_score(y_val, val_proba))
        }
        
        # Store feature importance
        self.feature_importance = pd.Series(
            self.model.feature_importances_, 
            index=X.columns
        ).sort_values(ascending=False)
        
        self.is_trained = True
        logger.info(f"Training completed - Val Accuracy: {metrics['val_accuracy']:.4f}")
        
        return metrics
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Make predictions."""
        if not self.is_trained or self.model is None:
            raise ValueError("Model must be trained before prediction")
        
        return self.model.predict(X)
    
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Predict probabilities."""
        if not self.is_trained or self.model is None:
            raise ValueError("Model must be trained before prediction")
        
        return self.model.predict_proba(X)
    
    def get_feature_importance(self, top_n: int = 20) -> pd.Series:
        """Get top feature importances."""
        if self.feature_importance is None:
            raise ValueError("Model must be trained to get feature importance")
        
        return self.feature_importance.head(top_n)
    
    def save_model(self, filepath: str):
        """Save trained model."""
        if not self.is_trained:
            raise ValueError("Cannot save untrained model")
        
        model_data = {
            'model': self.model,
            'feature_importance': self.feature_importance,
            'model_params': self.model_params
        }
        joblib.dump(model_data, filepath)
        logger.info(f"Model saved to {filepath}")
    
    def load_model(self, filepath: str):
        """Load trained model."""
        model_data = joblib.load(filepath)
        self.model = model_data['model']
        self.feature_importance = model_data['feature_importance']
        self.model_params = model_data['model_params']
        self.is_trained = True
        logger.info(f"Model loaded from {filepath}")
    
    def explain_prediction(self, X: pd.DataFrame, instance_idx: int = 0) -> Dict[str, float]:
        """Explain individual prediction using feature contributions."""
        if not self.is_trained or self.feature_importance is None:
            raise ValueError("Model must be trained for explanations")
        
        # Get SHAP-like feature contributions (simplified)
        feature_values = X.iloc[instance_idx]
        importance = self.feature_importance
        
        # Calculate weighted contributions
        contributions = {}
        for feature in X.columns:
            if feature in importance.index:
                contrib = float(feature_values[feature] * importance[feature])
                contributions[feature] = contrib
        
        # Sort by absolute contribution
        contributions = dict(sorted(
            contributions.items(), 
            key=lambda x: abs(x[1]), 
            reverse=True
        ))
        
        return contributions