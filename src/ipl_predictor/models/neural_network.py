"""
Neural network model for IPL match prediction.
Deep learning approach for pattern recognition in sports data.
"""

import numpy as np
import pandas as pd
from typing import Dict, List
import pickle
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import train_test_split
import logging
from pathlib import Path
import tensorflow as tf
from tensorflow import keras  # type: ignore
from tensorflow.keras import layers  # type: ignore
from tensorflow.keras.callbacks import EarlyStopping  # type: ignore

logger = logging.getLogger(__name__)


class NeuralNetworkPredictor:
    """
    Neural Network-based predictor for IPL match outcomes.
    Uses deep learning to capture complex patterns in match data.
    """

    def __init__(
        self, hidden_layers: List[int] = [128, 64, 32], dropout_rate: float = 0.3
    ):
        """
        Initialize Neural Network predictor.

        Args:
            hidden_layers: List of hidden layer sizes
            dropout_rate: Dropout rate for regularization
        """
        self.hidden_layers = hidden_layers
        self.dropout_rate = dropout_rate
        self.model = None
        self.scaler = StandardScaler()
        self.history = None
        self.is_trained = False

    def _build_model(self, input_dim: int) -> keras.Sequential:
        """Build the neural network architecture."""
        model = keras.Sequential()

        # Input layer
        model.add(keras.Input(shape=(input_dim,)))

        # Hidden layers with dropout
        for i, units in enumerate(self.hidden_layers):
            model.add(layers.Dense(units, activation="relu", name=f"dense_{i+1}"))
            model.add(layers.Dropout(self.dropout_rate, name=f"dropout_{i+1}"))

        # Output layer (binary classification)
        model.add(layers.Dense(1, activation="sigmoid", name="output"))

        # Compile model
        model.compile(
            optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"]
        )

        return model

    def train(
        self, X: pd.DataFrame, y: pd.Series, validation_split: float = 0.2
    ) -> Dict:
        """
        Train the neural network model.

        Args:
            X: Feature matrix
            y: Target vector
            validation_split: Fraction of data to use for validation

        Returns:
            Dictionary with training metrics
        """
        logger.info("Training Neural Network model")

        # Convert to numpy arrays
        X_array = X.values
        y_array = y.values

        # Split data
        X_train, X_val, y_train, y_val = train_test_split(
            X_array, y_array, test_size=validation_split, random_state=42, stratify=y_array  # type: ignore
        )

        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_val_scaled = self.scaler.transform(X_val)

        # Build model
        self.model = self._build_model(X_train_scaled.shape[1])

        logger.info(f"Model architecture:\n{self.model.summary()}")

        # Early stopping callback
        early_stopping = EarlyStopping(
            monitor="val_loss", patience=10, restore_best_weights=True, verbose=1
        )

        # Train model
        self.history = self.model.fit(
            X_train_scaled,
            y_train,
            validation_data=(X_val_scaled, y_val),
            epochs=100,
            batch_size=32,
            callbacks=[early_stopping],
            verbose=1,
        )

        # Evaluate on validation set
        val_predictions = self.model.predict(X_val_scaled)
        val_predictions_binary = (val_predictions > 0.5).astype(int).flatten()

        val_accuracy = accuracy_score(y_val, val_predictions_binary)
        val_auc = roc_auc_score(y_val, val_predictions.flatten())

        # Training set evaluation
        train_predictions = self.model.predict(X_train_scaled)
        train_predictions_binary = (train_predictions > 0.5).astype(int).flatten()

        train_accuracy = accuracy_score(y_train, train_predictions_binary)
        train_auc = roc_auc_score(y_train, train_predictions.flatten())

        self.is_trained = True

        metrics = {
            "train_accuracy": float(train_accuracy),
            "train_auc": float(train_auc),
            "val_accuracy": float(val_accuracy),
            "val_auc": float(val_auc),
            "epochs_trained": len(self.history.history["loss"]),
            "final_train_loss": float(self.history.history["loss"][-1]),
            "final_val_loss": float(self.history.history["val_loss"][-1]),
        }

        logger.info(f"Neural Network training completed:")
        logger.info(
            f"  Train Accuracy: {train_accuracy:.4f}, Train AUC: {train_auc:.4f}"
        )
        logger.info(f"  Val Accuracy: {val_accuracy:.4f}, Val AUC: {val_auc:.4f}")
        logger.info(f"  Epochs trained: {metrics['epochs_trained']}")

        return metrics

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Make predictions on new data."""
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")

        X_scaled = self.scaler.transform(X.values)
        predictions = self.model.predict(X_scaled)  # type: ignore
        return (predictions > 0.5).astype(int).flatten()

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Get prediction probabilities."""
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")

        X_scaled = self.scaler.transform(X.values)
        probabilities = self.model.predict(X_scaled)  # type: ignore

        # Return probabilities for both classes
        return np.column_stack([1 - probabilities.flatten(), probabilities.flatten()])

    def save_model(self, filepath):
        """Save the trained model and scaler."""
        if not self.is_trained:
            raise ValueError("Cannot save untrained model")

        # Convert to Path object if string
        filepath = Path(filepath)

        # Save Keras model
        model_path = filepath.parent / f"{filepath.stem}_keras_model.h5"
        self.model.save(model_path)  # type: ignore

        # Save scaler and metadata
        metadata = {
            "scaler": self.scaler,
            "hidden_layers": self.hidden_layers,
            "dropout_rate": self.dropout_rate,
            "is_trained": self.is_trained,
            "model_path": str(model_path),
        }

        with open(filepath, "wb") as f:
            pickle.dump(metadata, f)

        logger.info(f"Neural Network model saved to {filepath}")

    def load_model(self, filepath):
        """Load a trained model and scaler."""
        # Convert to Path object if string
        filepath = Path(filepath)

        # Load metadata
        with open(filepath, "rb") as f:
            metadata = pickle.load(f)

        self.scaler = metadata["scaler"]
        self.hidden_layers = metadata["hidden_layers"]
        self.dropout_rate = metadata["dropout_rate"]
        self.is_trained = metadata["is_trained"]

        # Load Keras model
        model_path = Path(metadata["model_path"])
        self.model = keras.models.load_model(model_path)

        logger.info(f"Neural Network model loaded from {filepath}")

    def get_feature_importance(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Get feature importance using permutation importance.
        Note: This is computationally expensive for neural networks.
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before getting feature importance")

        # For neural networks, we'll use a simple gradient-based approach
        X_scaled = self.scaler.transform(X.values)

        # Get gradients with respect to input features
        with tf.GradientTape() as tape:
            X_tensor = tf.Variable(X_scaled.astype(np.float32))
            predictions = self.model(X_tensor)  # type: ignore

        gradients = tape.gradient(predictions, X_tensor)

        # Calculate mean absolute gradients as feature importance
        importance_scores = np.mean(np.abs(gradients.numpy()), axis=0)  # type: ignore

        feature_importance = pd.DataFrame(
            {"feature": X.columns, "importance": importance_scores}
        ).sort_values("importance", ascending=False)

        return feature_importance
