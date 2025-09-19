"""
Professional ML model training and prediction system with MLflow integration.
Includes ensemble methods, hyperparameter optimization, SHAP explanations, and model versioning.
"""

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import joblib
import mlflow
import numpy as np
import pandas as pd

try:
    import mlflow.lightgbm
    import mlflow.sklearn
    import mlflow.xgboost

    MLFLOW_TRACKING_AVAILABLE = True
except ImportError:
    MLFLOW_TRACKING_AVAILABLE = False
import json
import warnings
from datetime import datetime

warnings.filterwarnings("ignore")

import lightgbm as lgb
import optuna
import xgboost as xgb
from catboost import CatBoostClassifier
from sklearn.decomposition import PCA

# ML Libraries
from sklearn.ensemble import (
    RandomForestClassifier,
    StackingClassifier,
    VotingClassifier,
)
from sklearn.feature_selection import RFE, SelectFromModel, SelectKBest, f_classif
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
)
from sklearn.model_selection import (
    GridSearchCV,
    RandomizedSearchCV,
    StratifiedKFold,
    cross_val_score,
)

# SHAP for model explanations
try:
    import shap

    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False

# Custom imports
from ..utils.config import get_config
from ..utils.logging_config import MetricsLogger

logger = logging.getLogger(__name__)


class FeatureSelector:
    """Advanced feature selection using multiple techniques."""

    def __init__(self, selection_methods: Optional[List[str]] = None):
        self.selection_methods = selection_methods or [
            "univariate",
            "rfe",
            "importance",
        ]
        self.selectors = {}
        self.selected_features = {}

    def select_features(
        self, X: np.ndarray, y: np.ndarray, feature_names: List[str], k: int = 20
    ) -> Tuple[np.ndarray, List[str]]:
        """Apply multiple feature selection methods and combine results."""
        logger.info(
            f"Selecting top {k} features using methods: {self.selection_methods}"
        )

        feature_scores = {}

        # Univariate feature selection
        if "univariate" in self.selection_methods:
            selector = SelectKBest(f_classif, k=min(k, X.shape[1]))
            selector.fit(X, y)
            selected_indices = selector.get_support(indices=True)
            self.selectors["univariate"] = selector

            for idx in selected_indices:
                feature_scores[feature_names[idx]] = (
                    feature_scores.get(feature_names[idx], 0) + 1
                )

        # Recursive Feature Elimination
        if "rfe" in self.selection_methods:
            estimator = RandomForestClassifier(n_estimators=100, random_state=42)
            selector = RFE(estimator, n_features_to_select=min(k, X.shape[1]))
            selector.fit(X, y)
            selected_indices = selector.get_support(indices=True)
            self.selectors["rfe"] = selector

            for idx in selected_indices:
                feature_scores[feature_names[idx]] = (
                    feature_scores.get(feature_names[idx], 0) + 1
                )

        # Feature importance based selection
        if "importance" in self.selection_methods:
            estimator = RandomForestClassifier(n_estimators=100, random_state=42)
            selector = SelectFromModel(estimator, max_features=min(k, X.shape[1]))
            selector.fit(X, y)
            selected_indices = selector.get_support(indices=True)
            self.selectors["importance"] = selector

            for idx in selected_indices:
                feature_scores[feature_names[idx]] = (
                    feature_scores.get(feature_names[idx], 0) + 1
                )

        # Select features that appear in at least 2 methods
        min_votes = min(2, len(self.selection_methods))
        selected_features = [
            name for name, score in feature_scores.items() if score >= min_votes
        ]

        # If not enough features, add top scoring ones
        if len(selected_features) < k:
            remaining = k - len(selected_features)
            sorted_features = sorted(
                feature_scores.items(), key=lambda x: x[1], reverse=True
            )
            for name, _ in sorted_features:
                if name not in selected_features and remaining > 0:
                    selected_features.append(name)
                    remaining -= 1

        # Get indices of selected features
        selected_indices = [
            feature_names.index(name)
            for name in selected_features
            if name in feature_names
        ]

        logger.info(
            f"Selected {len(selected_features)} features: {selected_features[:10]}..."
        )

        return X[:, selected_indices], selected_features


class SHAPExplainer:
    """SHAP explanations for model interpretability."""

    def __init__(self, model: Any, model_type: str = "tree"):
        self.model = model
        self.model_type = model_type
        self.explainer = None
        self.shap_values = None

    def create_explainer(self, X_train: np.ndarray) -> None:
        """Create SHAP explainer based on model type."""
        if not SHAP_AVAILABLE:
            logger.warning("SHAP not available. Install with: pip install shap")
            return

        try:
            if self.model_type in ["tree", "xgboost", "lightgbm", "catboost"]:
                if SHAP_AVAILABLE:
                    import shap

                    self.explainer = shap.TreeExplainer(self.model)
            else:
                # Use a subset for non-tree models to avoid memory issues
                if SHAP_AVAILABLE:
                    import shap

                    background = (
                        shap.sample(X_train, 100) if len(X_train) > 100 else X_train
                    )
                    self.explainer = shap.Explainer(self.model, background)

            logger.info(f"SHAP {self.model_type} explainer created")
        except Exception as e:
            logger.warning(f"Could not create SHAP explainer: {e}")

    def calculate_shap_values(self, X: np.ndarray) -> Optional[Any]:
        """Calculate SHAP values for predictions."""
        if not SHAP_AVAILABLE or self.explainer is None:
            return None

        try:
            # Use a smaller sample for stability
            sample_size = min(20, len(X))
            X_sample = X[:sample_size] if len(X) > sample_size else X

            # Suppress SHAP warnings temporarily
            import warnings

            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                self.shap_values = self.explainer.shap_values(X_sample)  # type: ignore

            # For binary classification, take values for positive class
            if isinstance(self.shap_values, list) and len(self.shap_values) == 2:
                self.shap_values = self.shap_values[1]

            return self.shap_values
        except Exception as e:
            logger.warning(f"Could not calculate SHAP values: {e}")
            return None

    def get_feature_importance(
        self, feature_names: List[str]
    ) -> Optional[Dict[str, float]]:
        """Get feature importance from SHAP values."""
        if self.shap_values is None:
            return None

        try:
            # Calculate mean absolute SHAP values
            importance = np.abs(self.shap_values).mean(axis=0)
            return dict(zip(feature_names, importance))
        except Exception as e:
            logger.warning(f"Could not extract feature importance from SHAP: {e}")
            return None


class CrossValidator:
    """Advanced cross-validation with stratification and custom scoring."""

    def __init__(self, cv_folds: int = 5, scoring: Optional[List[str]] = None):
        self.cv_folds = cv_folds
        self.scoring = scoring or ["accuracy", "precision", "recall", "f1", "roc_auc"]
        self.cv_results = {}

    def cross_validate_model(
        self, model: Any, X: np.ndarray, y: np.ndarray
    ) -> Dict[str, float]:
        """Perform stratified cross-validation with multiple metrics."""
        cv = StratifiedKFold(n_splits=self.cv_folds, shuffle=True, random_state=42)

        results = {}
        for metric in self.scoring:
            try:
                scores = cross_val_score(model, X, y, cv=cv, scoring=metric)
                results[f"{metric}_mean"] = scores.mean()
                results[f"{metric}_std"] = scores.std()
            except Exception as e:
                logger.warning(f"Could not calculate {metric}: {e}")
                results[f"{metric}_mean"] = 0.0
                results[f"{metric}_std"] = 0.0

        self.cv_results = results
        return results


class ModelRegistry:
    """Model registry for managing different ML algorithms."""

    def __init__(self):
        self.models = {
            "random_forest": RandomForestClassifier,
            "xgboost": xgb.XGBClassifier,
            "lightgbm": lgb.LGBMClassifier,
            "catboost": CatBoostClassifier,
            "logistic_regression": LogisticRegression,
        }

    def get_model(self, name: str, params: Optional[Dict[str, Any]] = None) -> Any:
        """Get model instance with parameters."""
        if name not in self.models:
            raise ValueError(
                f"Model {name} not supported. Available: {list(self.models.keys())}"
            )

        model_class = self.models[name]
        if params:
            return model_class(**params)
        return model_class()


class HyperparameterOptimizer:
    """Advanced hyperparameter optimization using Optuna."""

    def __init__(self, model_name: str, n_trials: int = 100):
        self.model_name = model_name
        self.n_trials = n_trials
        self.study = None

    def optimize(
        self, X: np.ndarray, y: np.ndarray, cv_folds: int = 5
    ) -> Dict[str, Any]:
        """Optimize hyperparameters using Optuna."""
        logger.info(f"Starting hyperparameter optimization for {self.model_name}")

        def objective(trial):
            params = self._suggest_parameters(trial)
            model = ModelRegistry().get_model(self.model_name, params)

            scores = cross_val_score(model, X, y, cv=cv_folds, scoring="accuracy")
            return scores.mean()

        self.study = optuna.create_study(direction="maximize")
        self.study.optimize(objective, n_trials=self.n_trials, show_progress_bar=True)

        logger.info(f"Best parameters: {self.study.best_params}")
        logger.info(f"Best score: {self.study.best_value:.4f}")

        return self.study.best_params

    def _suggest_parameters(self, trial) -> Dict[str, Any]:
        """Suggest parameters for different models."""
        if self.model_name == "random_forest":
            return {
                "n_estimators": trial.suggest_int("n_estimators", 100, 500),
                "max_depth": trial.suggest_int("max_depth", 10, 30),
                "min_samples_split": trial.suggest_int("min_samples_split", 2, 10),
                "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 5),
                "random_state": 42,
            }
        elif self.model_name == "xgboost":
            return {
                "n_estimators": trial.suggest_int("n_estimators", 100, 500),
                "max_depth": trial.suggest_int("max_depth", 3, 12),
                "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3),
                "subsample": trial.suggest_float("subsample", 0.6, 1.0),
                "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
                "random_state": 42,
            }
        elif self.model_name == "lightgbm":
            return {
                "n_estimators": trial.suggest_int("n_estimators", 100, 500),
                "max_depth": trial.suggest_int("max_depth", 3, 15),
                "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3),
                "num_leaves": trial.suggest_int("num_leaves", 20, 100),
                "subsample": trial.suggest_float("subsample", 0.6, 1.0),
                "random_state": 42,
            }
        elif self.model_name == "catboost":
            return {
                "iterations": trial.suggest_int("iterations", 100, 500),
                "depth": trial.suggest_int("depth", 4, 10),
                "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3),
                "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1, 10),
                "random_state": 42,
                "verbose": False,
            }
        else:
            return {}


class EnsembleModel:
    """Advanced ensemble modeling with voting and stacking."""

    def __init__(self, base_models: List[Tuple[str, Any]], meta_model: Any = None):
        self.base_models = base_models
        self.meta_model = meta_model or LogisticRegression()
        self.voting_classifier = None
        self.stacking_classifier = None

    def create_voting_ensemble(self, voting: str = "soft") -> VotingClassifier:
        """Create voting ensemble."""
        # Ensure voting is either 'hard' or 'soft'
        voting_method = "soft" if voting == "soft" else "hard"
        self.voting_classifier = VotingClassifier(
            estimators=self.base_models, voting=voting_method
        )
        return self.voting_classifier

    def create_stacking_ensemble(self, cv: int = 5) -> StackingClassifier:
        """Create stacking ensemble."""
        self.stacking_classifier = StackingClassifier(
            estimators=self.base_models, final_estimator=self.meta_model, cv=cv
        )
        return self.stacking_classifier


class ModelTrainer:
    """Main model training orchestrator with MLflow integration."""

    def __init__(self, config_path: Optional[str] = None):
        self.config = get_config(config_path)
        self.metrics_logger = MetricsLogger()
        self.model_registry = ModelRegistry()

        # Setup MLflow
        try:
            mlflow.set_tracking_uri(self.config.mlflow["tracking_uri"])  # type: ignore
            mlflow.set_experiment(self.config.mlflow["experiment_name"])  # type: ignore

            if self.config.mlflow.get("autolog", True) and MLFLOW_TRACKING_AVAILABLE:
                try:
                    import mlflow.sklearn  # type: ignore

                    mlflow.sklearn.autolog()  # type: ignore
                except (AttributeError, ImportError):
                    logger.warning("MLflow sklearn autolog not available")
        except Exception as e:
            logger.warning(f"MLflow setup failed: {e}")

    def train_single_model(
        self,
        model_name: str,
        X_train: np.ndarray,
        X_test: np.ndarray,
        y_train: np.ndarray,
        y_test: np.ndarray,
        feature_names: Optional[List[str]] = None,
        optimize_hyperparams: bool = True,
        use_feature_selection: bool = True,
    ) -> Tuple[Any, Dict[str, float]]:
        """Train a single model with optional hyperparameter optimization and feature selection."""

        with mlflow.start_run(
            run_name=f"{model_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        ):
            # Log parameters
            mlflow.log_param("model_type", model_name)
            mlflow.log_param("optimize_hyperparams", optimize_hyperparams)
            mlflow.log_param("use_feature_selection", use_feature_selection)

            # Feature selection
            if use_feature_selection and feature_names and len(feature_names) > 10:
                logger.info("Applying feature selection...")
                feature_selector = FeatureSelector()
                X_train_selected, selected_features = feature_selector.select_features(
                    X_train, y_train, feature_names, k=min(20, len(feature_names))
                )

                # Transform test set with same features
                selected_indices = [
                    feature_names.index(name)
                    for name in selected_features
                    if name in feature_names
                ]
                X_test_selected = X_test[:, selected_indices]

                mlflow.log_param("n_features_selected", len(selected_features))
                mlflow.log_param(
                    "selected_features", selected_features[:10]
                )  # Log first 10
            else:
                X_train_selected = X_train
                X_test_selected = X_test
                selected_features = feature_names or []

            # Hyperparameter optimization
            if optimize_hyperparams:
                optimizer = HyperparameterOptimizer(model_name, n_trials=50)
                best_params = optimizer.optimize(X_train_selected, y_train)
                mlflow.log_params(best_params)
                model = self.model_registry.get_model(model_name, best_params)
            else:
                model = self.model_registry.get_model(model_name)

            # Train model
            logger.info(f"Training {model_name}...")
            model.fit(X_train_selected, y_train)

            # Cross-validation
            cv_validator = CrossValidator()
            cv_results = cv_validator.cross_validate_model(
                model, X_train_selected, y_train
            )
            mlflow.log_metrics({f"cv_{k}": v for k, v in cv_results.items()})

            # Predictions
            y_pred = model.predict(X_test_selected)
            y_pred_proba = (
                model.predict_proba(X_test_selected)[:, 1]
                if hasattr(model, "predict_proba")
                else None
            )

            # Calculate metrics
            metrics = self._calculate_metrics(y_test, y_pred, y_pred_proba)

            # SHAP explanations
            try:
                shap_explainer = SHAPExplainer(model, model_name)
                shap_explainer.create_explainer(X_train_selected)
                shap_values = shap_explainer.calculate_shap_values(
                    X_test_selected[:50]
                )  # Sample for speed

                if shap_values is not None:
                    shap_importance = shap_explainer.get_feature_importance(
                        selected_features
                    )
                    if shap_importance:
                        # Log top 10 SHAP features
                        sorted_shap = sorted(
                            shap_importance.items(), key=lambda x: x[1], reverse=True
                        )[:10]
                        mlflow.log_params(
                            {
                                f"shap_feature_{i}": f"{name}:{score:.4f}"
                                for i, (name, score) in enumerate(sorted_shap)
                            }
                        )
            except Exception as e:
                logger.warning(f"SHAP analysis failed: {e}")

            # Log metrics
            mlflow.log_metrics(metrics)

            # Store feature information in model
            if hasattr(model, "feature_names_in_") or not hasattr(
                model, "feature_names_in_"
            ):
                model.feature_names_in_ = (
                    np.array(selected_features) if selected_features else np.array([])
                )

            # Log model (disabled for now due to MLflow issues)
            logger.info("Model logging skipped - MLflow integration needs fixing")

            # Log to custom metrics logger
            self.metrics_logger.log_model_performance(model_name, metrics)

            logger.info(
                f"{model_name} training completed. Accuracy: {metrics['accuracy']:.4f}"
            )

            return model, metrics

    def train_ensemble_models(
        self,
        X_train: np.ndarray,
        X_test: np.ndarray,
        y_train: np.ndarray,
        y_test: np.ndarray,
        feature_names: Optional[List[str]] = None,
    ) -> Dict[str, Tuple[Any, Dict[str, float]]]:
        """Train multiple models and create ensembles with advanced techniques."""

        results = {}
        base_models = []

        # Train individual models
        for model_name in self.config.models.algorithms[:4]:  # Limit for ensemble
            try:
                model, metrics = self.train_single_model(
                    model_name,
                    X_train,
                    X_test,
                    y_train,
                    y_test,
                    feature_names=feature_names,
                    optimize_hyperparams=True,
                    use_feature_selection=True,
                )
                results[model_name] = (model, metrics)
                base_models.append((model_name, model))
            except Exception as e:
                logger.error(f"Failed to train {model_name}: {e}")

        # Create ensemble models
        if len(base_models) >= 2:
            ensemble_creator = EnsembleModel(base_models)

            # Voting ensemble
            with mlflow.start_run(
                run_name=f"voting_ensemble_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            ):
                voting_model = ensemble_creator.create_voting_ensemble()
                voting_model.fit(X_train, y_train)

                y_pred = voting_model.predict(X_test)
                y_pred_proba = voting_model.predict_proba(X_test)[:, 1]

                metrics = self._calculate_metrics(
                    y_test, np.array(y_pred), y_pred_proba
                )
                mlflow.log_metrics(metrics)
                mlflow.log_param("ensemble_type", "voting")
                mlflow.log_param("base_models", [name for name, _ in base_models])

                logger.info("Voting model logging skipped")
                results["voting_ensemble"] = (voting_model, metrics)

            # Stacking ensemble
            with mlflow.start_run(
                run_name=f"stacking_ensemble_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            ):
                stacking_model = ensemble_creator.create_stacking_ensemble()
                stacking_model.fit(X_train, y_train)

                y_pred = stacking_model.predict(X_test)
                y_pred_proba_full = stacking_model.predict_proba(X_test)
                y_pred_proba = y_pred_proba_full[:, 1] if y_pred_proba_full.shape[1] > 1 else y_pred_proba_full[:, 0]  # type: ignore

                metrics = self._calculate_metrics(
                    y_test, np.array(y_pred), y_pred_proba
                )
                mlflow.log_metrics(metrics)
                mlflow.log_param("ensemble_type", "stacking")
                mlflow.log_param("base_models", [name for name, _ in base_models])
                mlflow.log_param(
                    "meta_model", type(ensemble_creator.meta_model).__name__
                )

                logger.info("Stacking model logging skipped")
                results["stacking_ensemble"] = (stacking_model, metrics)

        return results

    def _calculate_metrics(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_pred_proba: Optional[np.ndarray] = None,
    ) -> Dict[str, float]:
        """Calculate comprehensive evaluation metrics."""
        report = classification_report(y_true, y_pred, output_dict=True)
        weighted_avg = report["weighted avg"]  # type: ignore

        metrics = {
            "accuracy": accuracy_score(y_true, y_pred),
            "precision": weighted_avg["precision"],  # type: ignore
            "recall": weighted_avg["recall"],  # type: ignore
            "f1_score": weighted_avg["f1-score"],  # type: ignore
        }

        if y_pred_proba is not None:
            try:
                metrics["roc_auc"] = roc_auc_score(
                    y_true, y_pred_proba, multi_class="ovr"
                )
            except ValueError:
                # For binary classification
                if len(np.unique(y_true)) == 2:
                    metrics["roc_auc"] = roc_auc_score(y_true, y_pred_proba)

        return metrics

    def save_best_model(
        self, results: Dict[str, Tuple[Any, Dict[str, float]]], metric: str = "accuracy"
    ) -> str:
        """Save the best performing model."""
        best_model_name = max(results.keys(), key=lambda x: results[x][1][metric])
        best_model, best_metrics = results[best_model_name]

        # Save to models directory
        model_path = self.config.get_model_path(f"best_model_{best_model_name}.pkl")
        joblib.dump(best_model, model_path)

        # Save model metadata
        metadata = {
            "model_name": best_model_name,
            "metrics": best_metrics,
            "timestamp": datetime.now().isoformat(),
            "config": {
                "features": (
                    len(best_model.feature_names_in_)
                    if hasattr(best_model, "feature_names_in_")
                    else "unknown"
                )
            },
        }

        metadata_path = self.config.get_model_path("model_metadata.json")
        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=2)

        logger.info(
            f"Best model ({best_model_name}) saved with {metric}: {best_metrics[metric]:.4f}"
        )

        return str(model_path)


class IPLPredictor:
    """Main prediction interface for IPL matches."""

    def __init__(
        self, model_path: Optional[str] = None, config_path: Optional[str] = None
    ):
        self.config = get_config(config_path)
        self.metrics_logger = MetricsLogger()

        if model_path:
            self.model = joblib.load(model_path)
            self.model_name = Path(model_path).stem
        else:
            self.model = None
            self.model_name = "unknown"

    def predict(self, match_data: Dict[str, Any]) -> Dict[str, Any]:
        """Predict match outcome with confidence scores."""
        if self.model is None:
            raise ValueError("No model loaded. Please load a model first.")

        start_time = datetime.now()

        # Convert input to DataFrame
        df = pd.DataFrame([match_data])

        # Make prediction
        prediction = self.model.predict(df)[0]
        if hasattr(self.model, "predict_proba"):
            probabilities = self.model.predict_proba(df)[0]
            confidence = max(probabilities)
        else:
            probabilities = None
            confidence = 0.5

        # Calculate inference time
        inference_time = (datetime.now() - start_time).total_seconds()

        result = {
            "predicted_winner": prediction,
            "confidence": confidence,
            "probabilities": (
                probabilities.tolist() if probabilities is not None else None
            ),
            "inference_time_ms": inference_time * 1000,
            "model_used": self.model_name,
        }

        # Log prediction
        self.metrics_logger.log_prediction(
            self.model_name, match_data, result, inference_time
        )

        return result

    def predict_batch(self, matches_data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Predict multiple matches at once."""
        return [self.predict(match) for match in matches_data]

    def load_model(self, model_path: str) -> None:
        """Load a trained model."""
        self.model = joblib.load(model_path)
        self.model_name = Path(model_path).stem
        logger.info(f"Model loaded from {model_path}")

    def get_feature_importance(self) -> Optional[Dict[str, float]]:
        """Get feature importance if available."""
        if self.model is not None and hasattr(self.model, "feature_importances_"):
            feature_names = getattr(
                self.model,
                "feature_names_in_",
                [f"feature_{i}" for i in range(len(self.model.feature_importances_))],
            )
            return dict(zip(feature_names, self.model.feature_importances_))
        return None
