"""
Professional logging configuration for IPL Predictor.
Provides structured logging with different handlers and formatters.
"""

import json
import logging
import logging.config
import os
from datetime import datetime
from pathlib import Path
from typing import Optional


def setup_logging(log_level: str = "INFO", log_file: Optional[str] = None) -> None:
    """Setup comprehensive logging configuration."""

    # Ensure logs directory exists
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)

    if log_file is None:
        log_file = str(
            log_dir / f"ipl_predictor_{datetime.now().strftime('%Y%m%d')}.log"
        )

    logging_config = {
        "version": 1,
        "disable_existing_loggers": False,
        "formatters": {
            "standard": {
                "format": "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
                "datefmt": "%Y-%m-%d %H:%M:%S",
            },
            "detailed": {
                "format": "%(asctime)s [%(levelname)s] %(name)s:%(lineno)d: %(message)s",
                "datefmt": "%Y-%m-%d %H:%M:%S",
            },
            "json": {"format": "%(message)s"},
        },
        "handlers": {
            "console": {
                "level": log_level,
                "class": "logging.StreamHandler",
                "formatter": "standard",
                "stream": "ext://sys.stdout",
            },
            "file": {
                "level": "DEBUG",
                "class": "logging.handlers.RotatingFileHandler",
                "filename": str(log_file),
                "formatter": "detailed",
                "maxBytes": 10485760,  # 10MB
                "backupCount": 5,
            },
            "metrics": {
                "level": "INFO",
                "class": "logging.handlers.RotatingFileHandler",
                "filename": str(log_dir / "metrics.log"),
                "formatter": "json",
                "maxBytes": 10485760,  # 10MB
                "backupCount": 3,
            },
        },
        "loggers": {
            "ipl_predictor": {
                "level": log_level,
                "handlers": ["console", "file"],
                "propagate": False,
            },
            "metrics": {"level": "INFO", "handlers": ["metrics"], "propagate": False},
        },
        "root": {"level": log_level, "handlers": ["console", "file"]},
    }

    logging.config.dictConfig(logging_config)


class MetricsLogger:
    """Specialized logger for metrics and monitoring."""

    def __init__(self):
        self.logger = logging.getLogger("metrics")

    def log_prediction(
        self, model_name: str, input_data: dict, prediction: dict, inference_time: float
    ) -> None:
        """Log prediction metrics."""
        metrics = {
            "timestamp": datetime.now().isoformat(),
            "event_type": "prediction",
            "model_name": model_name,
            "inference_time_ms": inference_time * 1000,
            "input_features": len(input_data),
            "prediction": prediction,
        }
        self.logger.info(json.dumps(metrics))

    def log_model_performance(self, model_name: str, metrics: dict) -> None:
        """Log model performance metrics."""
        log_data = {
            "timestamp": datetime.now().isoformat(),
            "event_type": "model_performance",
            "model_name": model_name,
            "metrics": metrics,
        }
        self.logger.info(json.dumps(log_data))

    def log_data_drift(
        self, feature_name: str, drift_score: float, threshold: float
    ) -> None:
        """Log data drift detection."""
        log_data = {
            "timestamp": datetime.now().isoformat(),
            "event_type": "data_drift",
            "feature_name": feature_name,
            "drift_score": drift_score,
            "threshold": threshold,
            "alert": drift_score > threshold,
        }
        self.logger.info(json.dumps(log_data))
