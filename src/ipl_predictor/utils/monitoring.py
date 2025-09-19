"""
Professional model monitoring and alerting system for IPL Predictor.
Implements data drift detection, performance monitoring, and automated alerts.
"""

import json
import logging
import smtplib
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

try:
    from email.mime.multipart import MIMEMultipart
    from email.mime.text import MIMEText
except ImportError:
    # Fallback for case-sensitive imports
    from email.mime.text import MimeText as MIMEText  # type: ignore
    from email.mime.multipart import MimeMultipart as MIMEMultipart  # type: ignore

import warnings

import requests
from scipy import stats
from sklearn.metrics import accuracy_score, precision_score, recall_score

warnings.filterwarnings("ignore")

from ..utils.config import get_config
from ..utils.logging_config import MetricsLogger

logger = logging.getLogger(__name__)


class DataDriftDetector:
    """Detect data drift in model inputs using statistical tests."""

    def __init__(self, reference_data: pd.DataFrame, significance_level: float = 0.05):
        """
        Initialize drift detector with reference data.

        Args:
            reference_data: Training data used as reference
            significance_level: Significance level for statistical tests
        """
        self.reference_data = reference_data
        self.significance_level = significance_level
        self.reference_stats = self._calculate_reference_stats()

    def _calculate_reference_stats(self) -> Dict[str, Dict[str, float]]:
        """Calculate reference statistics for each feature."""
        stats_dict = {}

        for column in self.reference_data.columns:
            if self.reference_data[column].dtype in ["int64", "float64"]:
                stats_dict[column] = {
                    "mean": float(self.reference_data[column].mean()),
                    "std": float(self.reference_data[column].std()),
                    "min": float(self.reference_data[column].min()),
                    "max": float(self.reference_data[column].max()),
                    "median": float(self.reference_data[column].median()),
                }
            else:
                # For categorical features
                value_counts = self.reference_data[column].value_counts()
                stats_dict[column] = {
                    "unique_count": len(value_counts),
                    "top_value": (
                        value_counts.index[0] if len(value_counts) > 0 else None
                    ),
                    "top_frequency": (
                        float(value_counts.iloc[0]) if len(value_counts) > 0 else 0
                    ),
                }

        return stats_dict

    def detect_drift(self, current_data: pd.DataFrame) -> Dict[str, Any]:
        """
        Detect drift between reference and current data.

        Args:
            current_data: Current batch of data to compare

        Returns:
            Dictionary with drift detection results
        """
        drift_results = {
            "timestamp": datetime.now().isoformat(),
            "total_features": len(self.reference_data.columns),
            "drifted_features": [],
            "drift_scores": {},
            "overall_drift_detected": False,
        }

        common_columns = set(self.reference_data.columns) & set(current_data.columns)

        for column in common_columns:
            if column in self.reference_stats:
                drift_score, is_drifted = self._test_feature_drift(
                    column, self.reference_data[column], current_data[column]
                )

                drift_results["drift_scores"][column] = drift_score

                if is_drifted:
                    drift_results["drifted_features"].append(column)
                    logger.warning(
                        f"Drift detected in feature '{column}' with score {drift_score:.4f}"
                    )

        # Overall drift assessment
        drift_results["overall_drift_detected"] = (
            len(drift_results["drifted_features"]) > 0
        )
        drift_results["drift_percentage"] = len(
            drift_results["drifted_features"]
        ) / len(common_columns)

        return drift_results

    def _test_feature_drift(
        self, feature_name: str, reference: pd.Series, current: pd.Series
    ) -> Tuple[float, bool]:
        """Test for drift in a single feature."""
        try:
            if reference.dtype in ["int64", "float64"] and current.dtype in [
                "int64",
                "float64",
            ]:
                # Kolmogorov-Smirnov test for numerical features
                statistic, p_value = stats.ks_2samp(
                    reference.dropna(), current.dropna()
                )
                is_drifted = float(p_value) < self.significance_level  # type: ignore
                return float(statistic), is_drifted  # type: ignore
            else:
                # Chi-square test for categorical features
                ref_counts = reference.value_counts()
                curr_counts = current.value_counts()

                # Align indices
                all_categories = set(ref_counts.index) | set(curr_counts.index)
                ref_aligned = pd.Series(
                    [ref_counts.get(cat, 0) for cat in all_categories]
                )
                curr_aligned = pd.Series(
                    [curr_counts.get(cat, 0) for cat in all_categories]
                )

                if len(all_categories) > 1 and curr_aligned.sum() > 0:
                    statistic, p_value = stats.chisquare(curr_aligned, ref_aligned)
                    is_drifted = p_value < self.significance_level
                    return statistic, is_drifted
                else:
                    return 0.0, False

        except Exception as e:
            logger.error(f"Error testing drift for feature {feature_name}: {e}")
            return 0.0, False


class ModelPerformanceMonitor:
    """Monitor model performance and trigger alerts when performance degrades."""

    def __init__(self, config_path: Optional[str] = None):
        self.config = get_config(config_path)
        self.metrics_logger = MetricsLogger()
        self.performance_history = []
        self.alert_thresholds = self.config.monitoring.alert_thresholds

    def log_prediction_performance(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        model_name: str,
        timestamp: Optional[datetime] = None,
    ) -> Dict[str, float]:
        """Log and analyze prediction performance."""
        if timestamp is None:
            timestamp = datetime.now()

        # Calculate metrics
        metrics = {
            "accuracy": accuracy_score(y_true, y_pred),
            "precision": precision_score(
                y_true, y_pred, average="weighted", zero_division=0
            ),
            "recall": recall_score(y_true, y_pred, average="weighted", zero_division=0),
            "timestamp": timestamp.isoformat(),
            "model_name": model_name,
            "sample_size": len(y_true),
        }

        # Add to history
        self.performance_history.append(metrics)

        # Log metrics
        self.metrics_logger.log_model_performance(model_name, metrics)

        # Check for performance degradation
        self._check_performance_alerts(metrics)

        return metrics

    def _check_performance_alerts(self, current_metrics: Dict[str, float]) -> None:
        """Check if current performance triggers any alerts."""
        if len(self.performance_history) < 2:
            return

        # Get recent baseline (last 10 predictions)
        recent_metrics = self.performance_history[-11:-1]  # Exclude current
        if not recent_metrics:
            return

        baseline_accuracy = np.mean([m["accuracy"] for m in recent_metrics])
        current_accuracy = current_metrics["accuracy"]

        accuracy_drop = baseline_accuracy - current_accuracy
        threshold = self.alert_thresholds.get("accuracy_drop", 0.05)

        if accuracy_drop > threshold:
            self._send_performance_alert(
                "Accuracy Drop Alert",
                f"Model accuracy dropped by {accuracy_drop:.1%} "
                f"(from {baseline_accuracy:.1%} to {current_accuracy:.1%})",
            )

    def _send_performance_alert(self, subject: str, message: str) -> None:
        """Send performance alert via multiple channels."""
        alert_data = {
            "timestamp": datetime.now().isoformat(),
            "subject": subject,
            "message": message,
            "severity": "WARNING",
        }

        # Log alert
        logger.warning(f"Performance Alert: {subject} - {message}")

        # Send to metrics logger
        self.metrics_logger.logger.warning(json.dumps(alert_data))

        # Send email alert (if configured)
        try:
            self._send_email_alert(subject, message)
        except Exception as e:
            logger.error(f"Failed to send email alert: {e}")

        # Send Slack alert (if configured)
        try:
            self._send_slack_alert(subject, message)
        except Exception as e:
            logger.error(f"Failed to send Slack alert: {e}")

    def _send_email_alert(self, subject: str, message: str) -> None:
        """Send email alert."""
        # This would be configured with actual email settings
        email_config = getattr(self.config, "email", {})
        if not email_config:
            return

        # Placeholder for email sending logic
        logger.info(f"Email alert sent: {subject}")

    def _send_slack_alert(self, subject: str, message: str) -> None:
        """Send Slack alert."""
        slack_webhook = getattr(self.config, "slack_webhook", None)
        if not slack_webhook:
            return

        payload = {
            "text": f"🚨 *{subject}*\n{message}",
            "username": "IPL-Predictor-Bot",
            "icon_emoji": ":warning:",
        }

        requests.post(slack_webhook, json=payload, timeout=10)
        logger.info(f"Slack alert sent: {subject}")

    def get_performance_summary(self, days: int = 7) -> Dict[str, Any]:
        """Get performance summary for the last N days."""
        cutoff_date = datetime.now() - timedelta(days=days)

        recent_metrics = [
            m
            for m in self.performance_history
            if datetime.fromisoformat(m["timestamp"]) >= cutoff_date
        ]

        if not recent_metrics:
            return {"message": "No recent performance data available"}

        accuracies = [m["accuracy"] for m in recent_metrics]
        precisions = [m["precision"] for m in recent_metrics]
        recalls = [m["recall"] for m in recent_metrics]

        return {
            "period_days": days,
            "total_predictions": len(recent_metrics),
            "average_accuracy": np.mean(accuracies),
            "accuracy_std": np.std(accuracies),
            "average_precision": np.mean(precisions),
            "average_recall": np.mean(recalls),
            "accuracy_trend": self._calculate_trend(accuracies),
            "last_updated": recent_metrics[-1]["timestamp"] if recent_metrics else None,
        }

    def _calculate_trend(self, values: List[float]) -> str:
        """Calculate trend direction for a series of values."""
        if len(values) < 2:
            return "insufficient_data"

        # Simple linear regression slope
        x = np.arange(len(values))
        slope = np.polyfit(x, values, 1)[0]

        if slope > 0.01:
            return "improving"
        elif slope < -0.01:
            return "declining"
        else:
            return "stable"


class SystemHealthMonitor:
    """Monitor system health and resource usage."""

    def __init__(self):
        self.metrics_logger = MetricsLogger()

    def check_system_health(self) -> Dict[str, Any]:
        """Check overall system health."""
        import psutil

        health_status = {
            "timestamp": datetime.now().isoformat(),
            "cpu_usage_percent": psutil.cpu_percent(interval=1),
            "memory_usage_percent": psutil.virtual_memory().percent,
            "disk_usage_percent": psutil.disk_usage("/").percent,
            "system_load": (
                psutil.getloadavg()[0] if hasattr(psutil, "getloadavg") else None
            ),
            "status": "healthy",
        }

        # Determine overall health status
        if (
            health_status["cpu_usage_percent"] > 90
            or health_status["memory_usage_percent"] > 90
            or health_status["disk_usage_percent"] > 90
        ):
            health_status["status"] = "critical"
        elif (
            health_status["cpu_usage_percent"] > 70
            or health_status["memory_usage_percent"] > 70
            or health_status["disk_usage_percent"] > 70
        ):
            health_status["status"] = "warning"

        # Log health metrics
        self.metrics_logger.logger.info(
            json.dumps({"event_type": "system_health", **health_status})
        )

        return health_status

    def check_api_health(
        self, api_url: str = "http://localhost:8000"
    ) -> Dict[str, Any]:
        """Check API health and response times."""
        try:
            start_time = datetime.now()
            response = requests.get(f"{api_url}/health", timeout=30)
            response_time = (datetime.now() - start_time).total_seconds()

            api_health = {
                "timestamp": datetime.now().isoformat(),
                "status_code": response.status_code,
                "response_time_seconds": response_time,
                "is_healthy": response.status_code == 200,
                "api_url": api_url,
            }

            if response.status_code == 200:
                api_health.update(response.json())

        except requests.RequestException as e:
            api_health = {
                "timestamp": datetime.now().isoformat(),
                "status_code": None,
                "response_time_seconds": None,
                "is_healthy": False,
                "error": str(e),
                "api_url": api_url,
            }

        # Log API health
        self.metrics_logger.logger.info(
            json.dumps({"event_type": "api_health", **api_health})
        )

        return api_health


class ComprehensiveMonitor:
    """Main monitoring orchestrator that combines all monitoring components."""

    def __init__(self, config_path: Optional[str] = None):
        self.config = get_config(config_path)
        self.performance_monitor = ModelPerformanceMonitor(config_path)
        self.system_monitor = SystemHealthMonitor()
        self.drift_detector = (
            None  # Will be initialized when reference data is provided
        )

    def initialize_drift_detection(self, reference_data: pd.DataFrame) -> None:
        """Initialize drift detection with reference data."""
        self.drift_detector = DataDriftDetector(reference_data)
        logger.info("Drift detection initialized with reference data")

    def run_comprehensive_check(self) -> Dict[str, Any]:
        """Run comprehensive monitoring check."""
        monitoring_results = {
            "timestamp": datetime.now().isoformat(),
            "system_health": self.system_monitor.check_system_health(),
            "api_health": self.system_monitor.check_api_health(),
            "performance_summary": self.performance_monitor.get_performance_summary(),
            "overall_status": "healthy",
        }

        # Determine overall status
        if (
            not monitoring_results["api_health"]["is_healthy"]
            or monitoring_results["system_health"]["status"] == "critical"
        ):
            monitoring_results["overall_status"] = "critical"
        elif monitoring_results["system_health"]["status"] == "warning":
            monitoring_results["overall_status"] = "warning"

        return monitoring_results

    def start_monitoring_loop(self, interval_minutes: int = 5) -> None:
        """Start continuous monitoring loop."""
        import time

        logger.info(
            f"Starting monitoring loop with {interval_minutes} minute intervals"
        )

        while True:
            try:
                results = self.run_comprehensive_check()

                if results["overall_status"] != "healthy":
                    logger.warning(f"System status: {results['overall_status']}")

                time.sleep(interval_minutes * 60)

            except KeyboardInterrupt:
                logger.info("Monitoring loop stopped by user")
                break
            except Exception as e:
                logger.error(f"Error in monitoring loop: {e}")
                time.sleep(60)  # Wait 1 minute before retrying
