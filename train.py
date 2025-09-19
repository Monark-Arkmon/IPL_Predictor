#!/usr/bin/env python3
"""
IPL ML Pipeline Orchestrator
============================

Simple orchestrator that runs the complete ML pipeline:
1. Data Verification
2. System Testing
3. Model Training
4. Results Logging

Usage:
    python train.py
"""

import logging
import subprocess
import sys
import time
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

# Import pipeline components
from ipl_predictor.data.verification import DataVerificationPipeline
from ipl_predictor.training.pipeline import TrainingPipeline
from ipl_predictor.utils.logging_config import setup_logging
from ipl_predictor.utils.results import ResultsLogger


def run_pipeline():
    """Run the complete ML pipeline."""

    # Setup logging
    setup_logging()
    logger = logging.getLogger("pipeline")

    # Initialize results logger
    results_logger = ResultsLogger()

    print("Starting IPL ML Pipeline")
    print("=" * 50)

    start_time = time.time()

    try:
        # STEP 1: Data Setup and Validation
        print("\nSTEP 1: Data Setup and Validation")
        print("-" * 30)

        # Run data validation and integration automatically
        try:
            print("Running data validation...")
            validation_path = Path(
                "src/ipl_predictor/data/validation/data_validation.py"
            )
            if validation_path.exists():
                result = subprocess.run(
                    [sys.executable, str(validation_path)],
                    capture_output=True,
                    text=True,
                    cwd=Path.cwd(),
                )
                if result.returncode == 0:
                    print("Data validation completed successfully")
                else:
                    print(f"Data validation warning: {result.stderr}")

            print("Running data integration...")
            integration_path = Path(
                "src/ipl_predictor/data/validation/data_integration.py"
            )
            if integration_path.exists():
                result = subprocess.run(
                    [sys.executable, str(integration_path)],
                    capture_output=True,
                    text=True,
                    cwd=Path.cwd(),
                )
                if result.returncode == 0:
                    print("Data integration completed successfully")
                else:
                    print(f"Data integration warning: {result.stderr}")

        except Exception as e:
            print(f"Data setup failed: {e}")
            print("Continuing with existing data...")

        # Standard data verification
        verifier = DataVerificationPipeline()
        verification_results = verifier.verify_all_data()
        results_logger.log_verification_results(verification_results)

        print(verifier.get_verification_summary())

        if not all(verification_results.values()):
            print("Data verification failed. Please fix data issues before training.")
            return False

        # STEP 2: System Testing
        print("\nSTEP 2: System Testing")
        print("-" * 30)

        # Run comprehensive system tests sequentially
        test_files = [
            "tests/test_system_components.py",
            "tests/test_data_loading.py",
            "tests/test_feature_engineering.py",
            "tests/test_pre_training_validation.py",
        ]

        test_results = {}
        all_tests_passed = True

        for test_file in test_files:
            test_name = (
                Path(test_file).stem.replace("test_", "").replace("_", " ").title()
            )
            print(f"Running {test_name}...")

            try:
                result = subprocess.run(
                    [sys.executable, "-m", "pytest", test_file, "-v", "--tb=short"],
                    capture_output=True,
                    text=True,
                    cwd=Path.cwd(),
                    timeout=120,
                )

                if result.returncode == 0:
                    print(f"{test_name} - PASSED")
                    test_results[test_file] = True
                else:
                    print(f"{test_name} - FAILED")
                    print(f"Error: {result.stderr}")
                    test_results[test_file] = False
                    all_tests_passed = False

            except subprocess.TimeoutExpired:
                print(f"{test_name} - TIMEOUT")
                test_results[test_file] = False
                all_tests_passed = False
            except Exception as e:
                print(f"{test_name} - ERROR: {e}")
                test_results[test_file] = False
                all_tests_passed = False

        # Print test summary
        print("\nTest Summary:")
        print("-" * 20)
        for test_file, passed in test_results.items():
            test_name = (
                Path(test_file).stem.replace("test_", "").replace("_", " ").title()
            )
            status = "PASS" if passed else "FAIL"
            print(f"{test_name:25} | {status}")

        if not all_tests_passed:
            print("\nSYSTEM TESTS FAILED - Fix issues before training!")
            return False

        print("\nAll system tests passed - Ready for training!")

        # STEP 3: Model Training
        print("\nSTEP 3: Model Training")
        print("-" * 30)

        trainer = TrainingPipeline()

        # Prepare training data
        print("Preparing training data...")
        X, y = trainer.prepare_training_data()
        print(f"Training data prepared: {X.shape}")

        # Train XGBoost
        print("Training XGBoost model...")
        xgb_results = trainer.train_xgboost_model(X, y)
        results_logger.log_model_performance("XGBoost", xgb_results)
        print(f"XGBoost trained - Accuracy: {xgb_results.get('accuracy', 0):.4f}")

        # Train Neural Network
        print("Training Neural Network...")
        nn_results = trainer.train_neural_network_model(X, y)
        results_logger.log_model_performance("NeuralNetwork", nn_results)
        print(f"Neural Network trained - Accuracy: {nn_results.get('accuracy', 0):.4f}")

        # Train Ensemble
        print("Training Ensemble...")
        ensemble_results = trainer.train_ensemble_model(X, y)
        results_logger.log_model_performance("Ensemble", ensemble_results)
        print(f"Ensemble trained - Accuracy: {ensemble_results.get('accuracy', 0):.4f}")

        # Log training completion
        duration = time.time() - start_time
        training_summary = {
            "duration": duration,
            "models_trained": 3,
            "best_accuracy": max(
                xgb_results.get("accuracy", 0),
                nn_results.get("accuracy", 0),
                ensemble_results.get("accuracy", 0),
            ),
        }
        results_logger.log_training_results(training_summary)

        # STEP 4: Results Logging
        print("\nSTEP 4: Results Logging")
        print("-" * 30)

        results_file = results_logger.save_results()
        print(f"Results saved to: {results_file}")

        # Print final summary
        print("\nPIPELINE SUMMARY")
        print("=" * 50)
        results_logger.print_summary()

        print(f"\nPipeline completed successfully in {duration:.2f} seconds!")
        return True

    except Exception as e:
        logger.error(f"Pipeline failed: {e}")
        print(f"Pipeline failed: {e}")
        return False


if __name__ == "__main__":
    success = run_pipeline()
    sys.exit(0 if success else 1)
