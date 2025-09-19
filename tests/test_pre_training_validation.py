"""
Pre-Training Validation Tests
Tests that ensure the system is ready for training BEFORE attempting to train models
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from ipl_predictor.features import FeaturePipeline


class TestPreTrainingDataValidation:
    """Test that data is ready for training."""

    def test_feature_generation_produces_valid_output(self):
        """Test that feature generation works without errors and produces valid output."""
        pipeline = FeaturePipeline("data")

        # Test basic feature generation
        try:
            features = pipeline.extract_match_features(
                team1="Mumbai Indians",
                team2="Chennai Super Kings",
                venue="Wankhede Stadium",
                match_date="2024-04-01",
            )

            # Should return a dictionary
            assert isinstance(
                features, dict
            ), "Features should be returned as dictionary"
            assert len(features) > 0, "Should generate at least some features"

            # All values should be numeric and finite
            for key, value in features.items():
                assert isinstance(
                    value, (int, float, np.integer, np.floating)
                ), f"Feature {key} should be numeric, got {type(value)}"
                assert np.isfinite(
                    value
                ), f"Feature {key} should be finite, got {value}"

            print(f"✅ Feature generation: {len(features)} valid features generated")

        except Exception as e:
            pytest.fail(f"Feature generation failed: {e}")

    def test_training_dataset_can_be_created(self):
        """Test that training dataset can be created without errors."""
        pipeline = FeaturePipeline("data")

        try:
            # Attempt to create training dataset
            X, y = pipeline.extract_training_dataset(min_date="2023-01-01")

            # Should return pandas DataFrame and Series
            assert isinstance(X, pd.DataFrame), "Features should be DataFrame"
            assert isinstance(y, pd.Series), "Targets should be Series"

            # Should have matching lengths
            assert len(X) == len(
                y
            ), f"Features ({len(X)}) and targets ({len(y)}) should have same length"

            # Should have reasonable number of samples
            assert len(X) > 10, f"Expected reasonable sample size, got {len(X)}"

            # Should have reasonable number of features
            assert X.shape[1] > 50, f"Expected substantial features, got {X.shape[1]}"

            # Should not have any missing values
            assert (
                not X.isnull().any().any()
            ), "Training features should not have missing values"
            assert (
                not y.isnull().any()
            ), "Training targets should not have missing values"

            print(
                f"✅ Training dataset creation: {X.shape[0]} samples, {X.shape[1]} features"
            )

        except Exception as e:
            pytest.fail(f"Training dataset creation failed: {e}")

    def test_feature_pipeline_handles_different_teams(self):
        """Test that feature pipeline can handle different team combinations."""
        pipeline = FeaturePipeline("data")

        test_cases = [
            ("Mumbai Indians", "Chennai Super Kings"),
            ("Royal Challengers Bangalore", "Kolkata Knight Riders"),
            ("Delhi Capitals", "Punjab Kings"),
            ("Gujarat Titans", "Lucknow Super Giants"),
        ]

        for team1, team2 in test_cases:
            try:
                features = pipeline.extract_match_features(
                    team1=team1,
                    team2=team2,
                    venue="Test Stadium",
                    match_date="2024-04-01",
                )

                assert (
                    len(features) > 50
                ), f"Failed to generate adequate features for {team1} vs {team2}"

            except Exception as e:
                pytest.fail(f"Feature generation failed for {team1} vs {team2}: {e}")

        print(f"✅ Team handling: {len(test_cases)} team combinations work correctly")


class TestDataIntegrity:
    """Test data integrity before training."""

    def test_no_data_corruption(self):
        """Test that loaded data is not corrupted."""
        from ipl_predictor.data.loader import IPLDataLoader

        loader = IPLDataLoader("data")

        # Test matches data
        matches = loader.load_matches()
        assert not matches.empty, "Matches data should not be empty"

        # Check for basic data sanity
        assert (
            matches["team1"].notna().sum() > len(matches) * 0.9
        ), "Most matches should have team1"
        assert (
            matches["team2"].notna().sum() > len(matches) * 0.9
        ), "Most matches should have team2"
        assert (
            matches["winner"].notna().sum() > len(matches) * 0.8
        ), "Most matches should have winner"

        # Test ball-by-ball data
        balls = loader.load_ball_by_ball()
        assert not balls.empty, "Ball-by-ball data should not be empty"

        # Check for basic data sanity
        assert (
            balls["Batter"].notna().sum() > len(balls) * 0.9
        ), "Most balls should have batter"
        assert (
            balls["Bowler"].notna().sum() > len(balls) * 0.9
        ), "Most balls should have bowler"

        print("✅ Data integrity: No corruption detected")

    def test_data_schema_consistency(self):
        """Test that data schemas are as expected."""
        from ipl_predictor.data.loader import IPLDataLoader

        loader = IPLDataLoader("data")

        # Test matches schema
        matches = loader.load_matches()
        expected_match_columns = ["team1", "team2", "winner", "match_date", "venue"]
        for col in expected_match_columns:
            assert col in matches.columns, f"Missing expected column in matches: {col}"

        # Test ball-by-ball schema
        balls = loader.load_ball_by_ball()
        expected_ball_columns = ["Batter", "Bowler", "BatsmanRun", "ID"]
        for col in expected_ball_columns:
            assert col in balls.columns, f"Missing expected column in balls: {col}"

        print("✅ Data schema: All expected columns present")


class TestModelReadiness:
    """Test that models are ready for training."""

    def test_models_can_handle_training_data_format(self):
        """Test that models can accept the training data format."""
        from ipl_predictor.features import FeaturePipeline
        from ipl_predictor.models.xgboost_model import XGBoostPredictor

        # Generate sample training data
        pipeline = FeaturePipeline("data")
        X, y = pipeline.extract_training_dataset(min_date="2023-01-01")

        # Take small subset for testing
        X_sample = X.head(20)
        y_sample = y.head(20)

        # Test that model can accept this data format
        model = XGBoostPredictor(
            model_params={"n_estimators": 2}
        )  # Minimal for testing

        try:
            # Should not throw errors on data format
            result = model.train(X_sample, y_sample, validation_split=0.2)
            assert isinstance(result, dict), "Training should return results dictionary"

            print("✅ Model readiness: XGBoost accepts training data format")

        except Exception as e:
            pytest.fail(f"Model cannot handle training data format: {e}")

    def test_feature_count_consistency(self):
        """Test that feature count is consistent across different predictions."""
        pipeline = FeaturePipeline("data")

        # Generate features for different matches
        features1 = pipeline.extract_match_features(
            team1="Mumbai Indians",
            team2="Chennai Super Kings",
            venue="Wankhede Stadium",
            match_date="2024-04-01",
        )

        features2 = pipeline.extract_match_features(
            team1="Royal Challengers Bangalore",
            team2="Kolkata Knight Riders",
            venue="M Chinnaswamy Stadium",
            match_date="2024-04-02",
        )

        # Feature count should be consistent
        assert len(features1) == len(
            features2
        ), f"Inconsistent feature count: {len(features1)} vs {len(features2)}"

        print(
            f"✅ Feature consistency: {len(features1)} features generated consistently"
        )


class TestConfigurationValidation:
    """Test that configuration and dependencies are correct."""

    def test_required_dependencies_available(self):
        """Test that all required Python packages are available."""
        required_packages = ["pandas", "numpy", "sklearn", "xgboost", "tensorflow"]

        for package in required_packages:
            try:
                __import__(package)
            except ImportError:
                pytest.fail(f"Required package not available: {package}")

        print("✅ Dependencies: All required packages available")

    def test_model_parameters_valid(self):
        """Test that default model parameters are valid."""
        from ipl_predictor.models.neural_network import NeuralNetworkPredictor
        from ipl_predictor.models.xgboost_model import XGBoostPredictor

        # Test XGBoost parameters
        try:
            xgb_model = XGBoostPredictor()
            assert hasattr(
                xgb_model, "model_params"
            ), "XGBoost should have model_params"
            assert isinstance(
                xgb_model.model_params, dict
            ), "model_params should be dict"

        except Exception as e:
            pytest.fail(f"XGBoost parameter validation failed: {e}")

        # Test Neural Network parameters
        try:
            nn_model = NeuralNetworkPredictor()
            assert hasattr(
                nn_model, "hidden_layers"
            ), "Neural Network should have hidden_layers"
            assert isinstance(
                nn_model.hidden_layers, list
            ), "hidden_layers should be list"

        except Exception as e:
            pytest.fail(f"Neural Network parameter validation failed: {e}")

        print("✅ Model parameters: Default parameters are valid")


class TestOutputDirectories:
    """Test that output directories are ready."""

    def test_output_directories_exist_or_can_be_created(self):
        """Test that required output directories exist or can be created."""
        required_output_dirs = [Path("models"), Path("logs")]

        for dir_path in required_output_dirs:
            if not dir_path.exists():
                try:
                    dir_path.mkdir(parents=True, exist_ok=True)
                except Exception as e:
                    pytest.fail(f"Cannot create required directory {dir_path}: {e}")

            assert dir_path.exists(), f"Required output directory missing: {dir_path}"
            assert dir_path.is_dir(), f"Path exists but is not a directory: {dir_path}"

        print("✅ Output directories: All required directories ready")

    def test_write_permissions(self):
        """Test that we have write permissions to output directories."""
        test_dirs = [Path("models"), Path("logs")]

        for dir_path in test_dirs:
            test_file = dir_path / "test_write_permission.tmp"
            try:
                # Try to write a test file
                test_file.write_text("test")
                test_file.unlink()  # Clean up

            except Exception as e:
                pytest.fail(f"No write permission to {dir_path}: {e}")

        print("✅ Write permissions: Can write to all output directories")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
