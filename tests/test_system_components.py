"""
System Component Tests
Tests that all code components work correctly BEFORE training
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from ipl_predictor.data.loader import IPLDataLoader
from ipl_predictor.data.verification import DataVerificationPipeline
from ipl_predictor.models.ensemble import EnsemblePredictor
from ipl_predictor.models.neural_network import NeuralNetworkPredictor
from ipl_predictor.models.xgboost_model import XGBoostPredictor


class TestDataVerificationPipeline:
    """Test that data verification pipeline works correctly."""

    def test_data_verification_pipeline_runs(self):
        """Test that data verification pipeline executes without errors."""
        verifier = DataVerificationPipeline()

        # Should run without throwing exceptions
        try:
            results = verifier.verify_all_data()
            assert isinstance(results, dict), "Verification should return a dictionary"
            print("✅ Data verification pipeline runs successfully")
        except Exception as e:
            pytest.fail(f"Data verification pipeline failed: {e}")

    def test_verification_returns_expected_keys(self):
        """Test that verification returns expected result keys."""
        verifier = DataVerificationPipeline()
        results = verifier.verify_all_data()

        # Should have boolean results for different verification checks
        assert isinstance(results, dict), "Results should be a dictionary"
        assert len(results) > 0, "Should have at least one verification result"

        # All values should be booleans
        for key, value in results.items():
            assert isinstance(
                value, bool
            ), f"Verification result {key} should be boolean, got {type(value)}"

        print(f"✅ Verification results: {len(results)} checks completed")

    def test_verification_summary_generation(self):
        """Test that verification summary can be generated."""
        verifier = DataVerificationPipeline()
        verifier.verify_all_data()

        try:
            summary = verifier.get_verification_summary()
            assert isinstance(summary, str), "Summary should be a string"
            assert len(summary) > 0, "Summary should not be empty"
            print("✅ Verification summary generated successfully")
        except Exception as e:
            pytest.fail(f"Verification summary generation failed: {e}")


class TestModelInstantiation:
    """Test that all model classes can be instantiated correctly."""

    def test_xgboost_model_instantiation(self):
        """Test that XGBoost model can be instantiated."""
        try:
            model = XGBoostPredictor()
            assert model is not None, "XGBoost model should be instantiated"
            assert hasattr(model, "train"), "XGBoost model should have train method"
            assert hasattr(model, "predict"), "XGBoost model should have predict method"
            assert hasattr(
                model, "predict_proba"
            ), "XGBoost model should have predict_proba method"
            print("✅ XGBoost model instantiation successful")
        except Exception as e:
            pytest.fail(f"XGBoost model instantiation failed: {e}")

    def test_neural_network_model_instantiation(self):
        """Test that Neural Network model can be instantiated."""
        try:
            model = NeuralNetworkPredictor()
            assert model is not None, "Neural Network model should be instantiated"
            assert hasattr(
                model, "train"
            ), "Neural Network model should have train method"
            assert hasattr(
                model, "predict"
            ), "Neural Network model should have predict method"
            assert hasattr(
                model, "predict_proba"
            ), "Neural Network model should have predict_proba method"
            print("✅ Neural Network model instantiation successful")
        except Exception as e:
            pytest.fail(f"Neural Network model instantiation failed: {e}")

    def test_ensemble_model_instantiation(self):
        """Test that Ensemble model can be instantiated."""
        try:
            model = EnsemblePredictor()
            assert model is not None, "Ensemble model should be instantiated"
            assert hasattr(
                model, "add_model"
            ), "Ensemble model should have add_model method"
            assert hasattr(model, "train"), "Ensemble model should have train method"
            assert hasattr(
                model, "predict"
            ), "Ensemble model should have predict method"
            print("✅ Ensemble model instantiation successful")
        except Exception as e:
            pytest.fail(f"Ensemble model instantiation failed: {e}")

    def test_model_initial_state(self):
        """Test that models start in correct initial state."""
        xgb_model = XGBoostPredictor()
        nn_model = NeuralNetworkPredictor()
        ensemble_model = EnsemblePredictor()

        # Models should not be trained initially
        assert not xgb_model.is_trained, "XGBoost should not be trained initially"
        assert not nn_model.is_trained, "Neural Network should not be trained initially"
        assert not ensemble_model.is_trained, "Ensemble should not be trained initially"

        # Model objects should be None initially
        assert xgb_model.model is None, "XGBoost model object should be None initially"
        assert (
            nn_model.model is None
        ), "Neural Network model object should be None initially"

        print("✅ All models start in correct initial state")


class TestTrainingPipelineComponents:
    """Test that training pipeline components work correctly."""

    def test_training_pipeline_instantiation(self):
        """Test that training pipeline can be instantiated."""
        try:
            sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
            from ipl_predictor.training.pipeline import TrainingPipeline

            trainer = TrainingPipeline()
            assert trainer is not None, "Training pipeline should be instantiated"
            assert hasattr(
                trainer, "prepare_training_data"
            ), "Should have prepare_training_data method"
            assert hasattr(
                trainer, "train_xgboost_model"
            ), "Should have train_xgboost_model method"
            assert hasattr(
                trainer, "train_neural_network_model"
            ), "Should have train_neural_network_model method"
            assert hasattr(
                trainer, "train_ensemble_model"
            ), "Should have train_ensemble_model method"
            print("✅ Training pipeline instantiation successful")
        except Exception as e:
            pytest.fail(f"Training pipeline instantiation failed: {e}")

    def test_feature_pipeline_instantiation(self):
        """Test that feature pipeline can be instantiated."""
        try:
            from ipl_predictor.features import FeaturePipeline

            pipeline = FeaturePipeline("data")
            assert pipeline is not None, "Feature pipeline should be instantiated"
            assert hasattr(
                pipeline, "extract_match_features"
            ), "Should have extract_match_features method"
            assert hasattr(
                pipeline, "extract_training_dataset"
            ), "Should have extract_training_dataset method"
            print("✅ Feature pipeline instantiation successful")
        except Exception as e:
            pytest.fail(f"Feature pipeline instantiation failed: {e}")


class TestResultsLogging:
    """Test that results logging system works correctly."""

    def test_results_logger_instantiation(self):
        """Test that results logger can be instantiated."""
        try:
            from ipl_predictor.utils.results import ResultsLogger

            logger = ResultsLogger()
            assert logger is not None, "Results logger should be instantiated"
            assert hasattr(
                logger, "log_verification_results"
            ), "Should have log_verification_results method"
            assert hasattr(
                logger, "log_testing_results"
            ), "Should have log_testing_results method"
            assert hasattr(
                logger, "log_model_performance"
            ), "Should have log_model_performance method"
            assert hasattr(logger, "save_results"), "Should have save_results method"
            print("✅ Results logger instantiation successful")
        except Exception as e:
            pytest.fail(f"Results logger instantiation failed: {e}")

    def test_results_logger_methods_work(self):
        """Test that results logger methods execute without errors."""
        from ipl_predictor.utils.results import ResultsLogger

        logger = ResultsLogger()

        # Test verification results logging
        try:
            logger.log_verification_results({"test_check": True})
            print("✅ Verification results logging works")
        except Exception as e:
            pytest.fail(f"Verification results logging failed: {e}")

        # Test testing results logging
        try:
            logger.log_testing_results({"test_passed": True})
            print("✅ Testing results logging works")
        except Exception as e:
            pytest.fail(f"Testing results logging failed: {e}")

        # Test model performance logging
        try:
            logger.log_model_performance("TestModel", {"accuracy": 0.75})
            print("✅ Model performance logging works")
        except Exception as e:
            pytest.fail(f"Model performance logging failed: {e}")


class TestLoggingConfiguration:
    """Test that logging configuration works correctly."""

    def test_logging_setup(self):
        """Test that logging setup executes without errors."""
        try:
            from ipl_predictor.utils.logging_config import setup_logging

            setup_logging()
            print("✅ Logging setup successful")
        except Exception as e:
            pytest.fail(f"Logging setup failed: {e}")

    def test_logger_creation(self):
        """Test that loggers can be created after setup."""
        import logging

        from ipl_predictor.utils.logging_config import setup_logging

        setup_logging()

        # Test creating different loggers
        pipeline_logger = logging.getLogger("pipeline")
        data_logger = logging.getLogger("data")
        model_logger = logging.getLogger("model")

        assert pipeline_logger is not None, "Pipeline logger should be created"
        assert data_logger is not None, "Data logger should be created"
        assert model_logger is not None, "Model logger should be created"

        print("✅ Logger creation successful")


class TestDirectoryStructure:
    """Test that required directories and files exist."""

    def test_data_directories_exist(self):
        """Test that required data directories exist."""
        required_dirs = [
            Path("data"),
            Path("data/raw"),
            Path("data/raw/matches"),
            Path("data/raw/ball_by_ball"),
            Path("data/raw/players"),
        ]

        for dir_path in required_dirs:
            assert dir_path.exists(), f"Required directory missing: {dir_path}"

        print("✅ All required data directories exist")

    def test_required_data_files_exist(self):
        """Test that critical data files exist."""
        required_files = [
            Path("data/raw/matches/ipl_match_info_2008_2024.csv"),
            Path("data/raw/ball_by_ball/ipl_ball_by_ball_2008_2024.csv"),
            Path("data/raw/players/ipl_players_2024.csv"),
        ]

        for file_path in required_files:
            assert file_path.exists(), f"Required data file missing: {file_path}"

        print("✅ All required data files exist")

    def test_source_code_structure(self):
        """Test that source code structure is correct."""
        required_paths = [
            Path("src/ipl_predictor"),
            Path("src/ipl_predictor/data"),
            Path("src/ipl_predictor/models"),
            Path("src/ipl_predictor/features"),
            Path("src/ipl_predictor/training"),
            Path("src/ipl_predictor/utils"),
        ]

        for path in required_paths:
            assert path.exists(), f"Required source path missing: {path}"

        print("✅ Source code structure is correct")


class TestImportPaths:
    """Test that all critical imports work correctly."""

    def test_data_imports(self):
        """Test that data-related imports work."""
        try:
            from ipl_predictor.data.loader import IPLDataLoader
            from ipl_predictor.data.verification import DataVerificationPipeline

            print("✅ Data imports successful")
        except ImportError as e:
            pytest.fail(f"Data imports failed: {e}")

    def test_model_imports(self):
        """Test that model-related imports work."""
        try:
            from ipl_predictor.models.ensemble import EnsemblePredictor
            from ipl_predictor.models.neural_network import NeuralNetworkPredictor
            from ipl_predictor.models.xgboost_model import XGBoostPredictor

            print("✅ Model imports successful")
        except ImportError as e:
            pytest.fail(f"Model imports failed: {e}")

    def test_feature_imports(self):
        """Test that feature-related imports work."""
        try:
            from ipl_predictor.features import FeaturePipeline

            print("✅ Feature imports successful")
        except ImportError as e:
            pytest.fail(f"Feature imports failed: {e}")

    def test_training_imports(self):
        """Test that training-related imports work."""
        try:
            from ipl_predictor.training.pipeline import TrainingPipeline

            print("✅ Training imports successful")
        except ImportError as e:
            pytest.fail(f"Training imports failed: {e}")

    def test_utils_imports(self):
        """Test that utility imports work."""
        try:
            from ipl_predictor.utils.logging_config import setup_logging
            from ipl_predictor.utils.results import ResultsLogger

            print("✅ Utils imports successful")
        except ImportError as e:
            pytest.fail(f"Utils imports failed: {e}")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
