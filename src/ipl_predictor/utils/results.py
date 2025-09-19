"""
Results and Logging System
==========================

Logging and results reporting for ML pipeline.
"""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

logger = logging.getLogger(__name__)


class ResultsLogger:
    """Comprehensive results logging and reporting."""

    def __init__(self, output_dir: str = "logs"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)

        self.run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.results = {
            "run_id": self.run_id,
            "timestamp": datetime.now().isoformat(),
            "verification": {},
            "testing": {},
            "training": {},
            "models": {},
        }

    def log_verification_results(self, verification_results: Dict[str, bool]) -> None:
        """Log data verification results."""
        logger.info("Logging verification results...")

        self.results["verification"] = {
            "passed": all(verification_results.values()),
            "details": verification_results,
            "timestamp": datetime.now().isoformat(),
        }

    def log_testing_results(self, test_results: Dict[str, bool]) -> None:
        """Log system testing results."""
        logger.info("Logging testing results...")

        self.results["testing"] = {
            "passed": all(test_results.values()),
            "details": test_results,
            "timestamp": datetime.now().isoformat(),
        }

    def log_training_results(self, training_results: Dict[str, Any]) -> None:
        """Log model training results."""
        logger.info("Logging training results...")

        self.results["training"] = {
            "completed": True,
            "details": training_results,
            "timestamp": datetime.now().isoformat(),
        }

    def log_model_performance(self, model_name: str, metrics: Dict[str, float]) -> None:
        """Log individual model performance."""
        logger.info(f"Logging {model_name} performance...")

        self.results["models"][model_name] = {
            "metrics": metrics,
            "timestamp": datetime.now().isoformat(),
        }

    def save_results(self) -> str:
        """Save all results to files."""
        logger.info("Saving pipeline results...")

        # Save JSON results
        json_path = self.output_dir / f"pipeline_results_{self.run_id}.json"
        with open(json_path, "w") as f:
            json.dump(self.results, f, indent=2)

        # Save human-readable summary
        summary_path = self.output_dir / f"pipeline_summary_{self.run_id}.txt"
        with open(summary_path, "w") as f:
            f.write(self.generate_summary())

        logger.info(f"Results saved to: {json_path}")
        return str(json_path)

    def generate_summary(self) -> str:
        """Generate human-readable summary."""
        summary = []
        summary.append("=" * 70)
        summary.append("IPL ML PIPELINE EXECUTION SUMMARY")
        summary.append("=" * 70)
        summary.append(f"Run ID: {self.run_id}")
        summary.append(f"Timestamp: {self.results['timestamp']}")
        summary.append("")

        # Verification Summary
        summary.append("DATA VERIFICATION")
        summary.append("-" * 30)
        if self.results["verification"]:
            verification = self.results["verification"]
            status = "PASSED" if verification["passed"] else "FAILED"
            summary.append(f"Overall: {status}")
            for check, result in verification["details"].items():
                check_status = "PASSED" if result else "FAILED"
                summary.append(f"  {check}: {check_status}")
        else:
            summary.append("No verification results")
        summary.append("")

        # Testing Summary
        summary.append("SYSTEM TESTING")
        summary.append("-" * 30)
        if self.results["testing"]:
            testing = self.results["testing"]
            status = "PASSED" if testing["passed"] else "FAILED"
            summary.append(f"Overall: {status}")
            for test, result in testing["details"].items():
                test_status = "PASSED" if result else "FAILED"
                summary.append(f"  {test}: {test_status}")
        else:
            summary.append("No testing results")
        summary.append("")

        # Training Summary
        summary.append("MODEL TRAINING")
        summary.append("-" * 30)
        if self.results["training"]:
            summary.append("Training completed")
            training = self.results["training"]["details"]
            if "duration" in training:
                summary.append(f"Duration: {training['duration']:.2f} seconds")
        else:
            summary.append("No training results")
        summary.append("")

        # Model Performance
        summary.append("MODEL PERFORMANCE")
        summary.append("-" * 30)
        if self.results["models"]:
            for model_name, model_data in self.results["models"].items():
                summary.append(f"{model_name}:")
                metrics = model_data["metrics"]
                for metric, value in metrics.items():
                    if isinstance(value, float):
                        summary.append(f"  {metric}: {value:.4f}")
                    else:
                        summary.append(f"  {metric}: {value}")
                summary.append("")
        else:
            summary.append("No model performance data")

        summary.append("=" * 70)
        summary.append("PIPELINE EXECUTION COMPLETE")
        summary.append("=" * 70)

        return "\n".join(summary)

    def print_summary(self) -> None:
        """Print summary to console."""
        print(self.generate_summary())
