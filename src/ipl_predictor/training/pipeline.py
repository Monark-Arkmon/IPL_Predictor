"""
Training pipeline for IPL prediction models.
Handles data preparation, model training, and evaluation.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import cross_val_score, StratifiedKFold
import logging
from typing import Dict, Tuple, Optional
import json

from ..features import FeaturePipeline
from ..models.xgboost_model import XGBoostPredictor
from ..models.neural_network import NeuralNetworkPredictor
from ..models.ensemble import EnsemblePredictor

logger = logging.getLogger(__name__)


class TrainingPipeline:
    """Complete training pipeline for IPL prediction models."""

    def __init__(self, data_path: str = "data", output_path: str = "models"):
        self.data_path = data_path
        self.output_path = Path(output_path)
        self.output_path.mkdir(exist_ok=True)

        self.feature_pipeline = FeaturePipeline(data_path)
        self.training_results = {}

    def prepare_training_data(
        self, min_date: str = "2015-01-01", save_features: bool = True
    ) -> Tuple[pd.DataFrame, pd.Series]:
        """Prepare training dataset with comprehensive features."""
        logger.info("Preparing training data with advanced features")

        # Extract features for all matches
        X, y = self.feature_pipeline.extract_training_dataset(min_date)

        logger.info(f"Training data shape: {X.shape}")
        logger.info(f"Target distribution: {y.value_counts().to_dict()}")

        # Save processed features
        if save_features:
            feature_path = self.output_path / "training_features.csv"
            self.feature_pipeline.save_feature_dataset(X, y, str(feature_path))

        return X, y

    def train_xgboost_model(
        self, X: pd.DataFrame, y: pd.Series, model_params: Optional[Dict] = None
    ) -> Dict:
        """Train XGBoost model with cross-validation."""
        logger.info("Training XGBoost model")

        # Initialize model
        xgb_model = XGBoostPredictor(model_params)

        # Train model
        train_metrics = xgb_model.train(X, y)

        # Cross-validation
        if xgb_model.model is not None:
            cv_scores = cross_val_score(
                xgb_model.model,
                X,
                y,
                cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42),
                scoring="accuracy",
            )
        else:
            cv_scores = np.array([0.0])

        # Save model
        model_path = self.output_path / "xgboost_model.pkl"
        xgb_model.save_model(str(model_path))

        # Store results
        results = {
            "model_type": "XGBoost",
            "train_metrics": train_metrics,
            "cv_mean": float(cv_scores.mean()),
            "cv_std": float(cv_scores.std()),
            "feature_importance": xgb_model.get_feature_importance().to_dict(),
            "model_path": str(model_path),
        }

        self.training_results["xgboost"] = results
        logger.info(
            f"XGBoost CV Accuracy: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})"
        )

        return results

    def train_neural_network_model(
        self, X: pd.DataFrame, y: pd.Series, model_params: Optional[Dict] = None
    ) -> Dict:
        """Train Neural Network model."""
        logger.info("Training Neural Network model")

        # Initialize model with proper parameters
        if model_params:
            hidden_layers = model_params.get("hidden_layers", [128, 64, 32])
            dropout_rate = model_params.get("dropout_rate", 0.3)
        else:
            hidden_layers = [128, 64, 32]
            dropout_rate = 0.3

        nn_model = NeuralNetworkPredictor(
            hidden_layers=hidden_layers, dropout_rate=dropout_rate
        )

        # Train model
        train_metrics = nn_model.train(X, y)

        # Save model
        model_path = self.output_path / "neural_network_model"
        nn_model.save_model(str(model_path))

        # Store results
        results = {
            "model_type": "NeuralNetwork",
            "train_metrics": train_metrics,
            "model_path": str(model_path),
        }

        self.training_results["neural_network"] = results
        logger.info(f"Neural Network Val Accuracy: {train_metrics['val_accuracy']:.4f}")

        return results

    def train_ensemble_model(self, X: pd.DataFrame, y: pd.Series) -> Dict:
        """Train ensemble model with multiple algorithms."""
        logger.info("Training Ensemble model")

        # Initialize ensemble
        ensemble = EnsemblePredictor()

        # Add XGBoost model
        xgb_model = XGBoostPredictor()
        ensemble.add_model(xgb_model, "XGBoost", weight=0.5)

        # Add Neural Network model
        nn_model = NeuralNetworkPredictor(hidden_layers=[128, 64, 32], dropout_rate=0.3)
        ensemble.add_model(nn_model, "NeuralNetwork", weight=0.3)

        # Add simplified baseline model (using XGBoost with different params)
        baseline_params = {
            "objective": "binary:logistic",
            "max_depth": 3,
            "learning_rate": 0.05,
            "n_estimators": 100,
            "random_state": 42,
        }
        baseline_model = XGBoostPredictor(baseline_params)
        ensemble.add_model(baseline_model, "Baseline", weight=0.2)

        # Train ensemble
        ensemble_metrics = ensemble.train(X, y)

        # Evaluate ensemble
        evaluation_metrics = ensemble.evaluate_ensemble(X, y)

        # Save ensemble
        ensemble_path = self.output_path / "ensemble_model"
        ensemble.save_ensemble(str(ensemble_path))

        # Store results
        results = {
            "model_type": "Ensemble",
            "ensemble_metrics": ensemble_metrics,
            "evaluation_metrics": evaluation_metrics,
            "model_path": str(ensemble_path),
        }

        self.training_results["ensemble"] = results
        logger.info(f"Ensemble Accuracy: {evaluation_metrics['accuracy']:.4f}")

        return results

    def run_complete_training(self, min_date: str = "2015-01-01") -> Dict:
        """Run complete training pipeline."""
        logger.info("Starting complete training pipeline")

        try:
            # Prepare data
            X, y = self.prepare_training_data(min_date)

            # Train individual models
            xgb_results = self.train_xgboost_model(X, y)
            nn_results = self.train_neural_network_model(X, y)

            # Train ensemble
            ensemble_results = self.train_ensemble_model(X, y)

            # Generate training report
            report = self.generate_training_report(X, y)

            # Save training summary
            summary = {
                "data_info": {
                    "samples": int(len(X)),
                    "features": int(len(X.columns)),
                    "min_date": min_date,
                    "target_distribution": {
                        str(k): int(v) for k, v in y.value_counts().to_dict().items()
                    },
                },
                "models": self.training_results,
                "report": report,
            }

            summary_path = self.output_path / "training_summary.json"
            with open(summary_path, "w") as f:
                json.dump(summary, f, indent=2, default=str)

            logger.info("Training pipeline completed successfully")
            return summary

        except Exception as e:
            logger.error(f"Training pipeline failed: {e}")
            raise

    def generate_training_report(self, X: pd.DataFrame, y: pd.Series) -> Dict:
        """Generate comprehensive training report."""
        report = {
            "data_summary": {
                "total_samples": int(len(X)),
                "total_features": int(len(X.columns)),
                "class_balance": {
                    str(k): float(v)
                    for k, v in y.value_counts(normalize=True).to_dict().items()
                },
                "feature_types": {
                    str(k): int(v) for k, v in X.dtypes.value_counts().to_dict().items()
                },
            },
            "top_features": {},
            "model_comparison": {},
        }

        # Add feature importance if available
        if "xgboost" in self.training_results:
            xgb_importance = self.training_results["xgboost"]["feature_importance"]
            top_features = dict(list(xgb_importance.items())[:10])
            report["top_features"]["xgboost"] = top_features

        # Model comparison
        for model_name, results in self.training_results.items():
            if "cv_mean" in results:
                report["model_comparison"][model_name] = {
                    "cv_accuracy": results["cv_mean"],
                    "cv_std": results["cv_std"],
                }
            elif "evaluation_metrics" in results:
                report["model_comparison"][model_name] = results["evaluation_metrics"]

        return report

    def load_and_evaluate_model(
        self, model_path: str, X_test: pd.DataFrame, y_test: pd.Series
    ) -> Dict:
        """Load and evaluate a trained model."""
        logger.info(f"Evaluating model from {model_path}")

        try:
            # Load model based on type
            if "ensemble" in model_path:
                model = EnsemblePredictor()
                model.load_ensemble(model_path)
            else:
                model = XGBoostPredictor()
                model.load_model(model_path)

            # Make predictions
            predictions = model.predict(X_test)
            probabilities = model.predict_proba(X_test)[:, 1]

            # Calculate metrics
            from sklearn.metrics import (
                accuracy_score,
                roc_auc_score,
                precision_recall_fscore_support,
            )

            accuracy = accuracy_score(y_test, predictions)
            auc = roc_auc_score(y_test, probabilities)
            precision, recall, f1, _ = precision_recall_fscore_support(
                y_test, predictions, average="binary"
            )

            results = {
                "accuracy": accuracy,
                "auc": auc,
                "precision": precision,
                "recall": recall,
                "f1_score": f1,
                "predictions": predictions.tolist(),
                "probabilities": probabilities.tolist(),
            }

            logger.info(f"Model evaluation - Accuracy: {accuracy:.4f}, AUC: {auc:.4f}")
            return results

        except Exception as e:
            logger.error(f"Model evaluation failed: {e}")
            raise
