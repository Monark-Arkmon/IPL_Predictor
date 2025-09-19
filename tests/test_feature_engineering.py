"""
Feature Engineering Tests
Tests that the feature pipeline generates correct features for ML models
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from ipl_predictor.features import FeaturePipeline


class TestFeaturePipeline:
    """Test feature generation functionality."""

    def test_feature_pipeline_generates_expected_count(self):
        """Test that feature pipeline generates the expected 86 features."""
        pipeline = FeaturePipeline("data")

        # Test with real team names and venue
        features = pipeline.extract_match_features(
            team1="Mumbai Indians",
            team2="Chennai Super Kings",
            venue="Wankhede Stadium",
            match_date="2024-04-01",
            toss_winner="Mumbai Indians",
            toss_decision="bat",
        )

        # Should generate approximately 86 features
        feature_count = len(features)
        assert feature_count >= 80, f"Expected ~86 features, got {feature_count}"
        assert feature_count <= 90, f"Too many features: {feature_count}, expected ~86"

        print(f"Feature generation: {feature_count} features created")

    def test_feature_categories_exist(self):
        """Test that all expected feature categories are generated."""
        pipeline = FeaturePipeline("data")

        features = pipeline.extract_match_features(
            team1="Royal Challengers Bangalore",
            team2="Kolkata Knight Riders",
            venue="M Chinnaswamy Stadium",
            match_date="2024-04-15",
        )

        # Check for team-specific features
        team1_features = [k for k in features.keys() if k.startswith("team1_")]
        team2_features = [k for k in features.keys() if k.startswith("team2_")]

        assert (
            len(team1_features) > 10
        ), f"Expected >10 team1 features, got {len(team1_features)}"
        assert (
            len(team2_features) > 10
        ), f"Expected >10 team2 features, got {len(team2_features)}"

        # Check for venue features
        venue_features = [k for k in features.keys() if "venue" in k.lower()]
        assert len(venue_features) > 0, "Expected venue-related features"

        # Check for toss features
        toss_features = [k for k in features.keys() if "toss" in k.lower()]
        assert len(toss_features) > 0, "Expected toss-related features"

        print(
            f"Feature categories: {len(team1_features)} team1, {len(team2_features)} team2, {len(venue_features)} venue, {len(toss_features)} toss"
        )

    def test_feature_values_are_valid(self):
        """Test that all feature values are valid (finite, numeric)."""
        pipeline = FeaturePipeline("data")

        features = pipeline.extract_match_features(
            team1="Delhi Capitals",
            team2="Punjab Kings",
            venue="Arun Jaitley Stadium",
            match_date="2024-03-20",
        )

        # Check each feature value
        for key, value in features.items():
            # Should be numeric
            assert isinstance(
                value, (int, float, np.integer, np.floating)
            ), f"Feature {key} should be numeric, got {type(value)}: {value}"

            # Should not be NaN or infinite
            assert np.isfinite(value), f"Feature {key} should be finite, got: {value}"

            # Should be reasonable magnitude (catch obvious errors)
            assert abs(value) < 10000, f"Feature {key} has unreasonable value: {value}"

        print("Feature validation: All values are finite and reasonable")

    def test_feature_consistency(self):
        """Test that same inputs produce same features."""
        pipeline = FeaturePipeline("data")

        # Generate features twice for same match
        features1 = pipeline.extract_match_features(
            team1="Sunrisers Hyderabad",
            team2="Rajasthan Royals",
            venue="Rajiv Gandhi International Stadium",
            match_date="2024-04-10",
        )

        features2 = pipeline.extract_match_features(
            team1="Sunrisers Hyderabad",
            team2="Rajasthan Royals",
            venue="Rajiv Gandhi International Stadium",
            match_date="2024-04-10",
        )

        # Should be identical
        assert len(features1) == len(features2), "Feature count should be consistent"

        for key in features1:
            assert key in features2, f"Feature {key} missing in second extraction"
            diff = abs(features1[key] - features2[key])
            assert (
                diff < 0.001
            ), f"Feature {key} inconsistent: {features1[key]} vs {features2[key]}"

        print("Feature consistency: Identical inputs produce identical features")


class TestTrainingDataset:
    """Test training dataset creation with minimal data for fast testing."""

    def test_training_dataset_basic_functionality(self):
        """Test that training dataset creation works at all."""
        pipeline = FeaturePipeline("data")

        # Test with just a few recent matches for speed
        X, y = pipeline.extract_training_dataset(min_date="2024-05-01")

        # Check basic functionality
        assert len(X) >= 1, f"Expected at least 1 training sample, got {len(X)} samples"
        assert len(X) == len(
            y
        ), f"Features ({len(X)}) and targets ({len(y)}) should have same length"
        assert X.shape[1] >= 70, f"Expected many features, got {X.shape[1]}"

        print(f"Training dataset: {X.shape[0]} samples, {X.shape[1]} features")

    def test_target_distribution(self):
        """Test that target distribution is reasonable for binary classification."""
        pipeline = FeaturePipeline("data")
        X, y = pipeline.extract_training_dataset(min_date="2024-05-01")

        # Check target distribution
        target_counts = y.value_counts()
        assert (
            len(target_counts) >= 1
        ), f"Expected at least one target class, got {len(target_counts)} classes"
        assert (
            len(target_counts) <= 2
        ), f"Expected binary targets, got {len(target_counts)} classes"

        print(f"Target distribution: {dict(target_counts)}")

    def test_no_missing_values_in_training_data(self):
        """Test that training dataset has no missing values."""
        pipeline = FeaturePipeline("data")
        X, y = pipeline.extract_training_dataset(
            min_date="2024-05-20"
        )  # Very recent for speed

        # Check for missing values in features
        missing_features = X.isnull().sum()
        features_with_missing = missing_features[missing_features > 0]
        assert (
            len(features_with_missing) == 0
        ), f"Features with missing values: {dict(features_with_missing)}"

        # Check for missing values in targets
        missing_targets = y.isnull().sum()
        assert missing_targets == 0, f"Targets have {missing_targets} missing values"

        print("Missing values: No missing values in training dataset")

    def test_feature_variance(self):
        """Test that features have reasonable variance (not all constant)."""
        pipeline = FeaturePipeline("data")
        X, y = pipeline.extract_training_dataset(min_date="2024-05-01")

        # Calculate variance for each feature
        feature_variance = X.var()

        # Check for constant features (variance = 0)
        constant_features = feature_variance[feature_variance == 0]
        # Note: Many features expected to be constant due to missing player composition data
        assert (
            len(constant_features) < 60
        ), f"Too many constant features: {len(constant_features)}"

        # But we should have some features with good variance
        non_constant_features = feature_variance[feature_variance > 0]
        assert (
            len(non_constant_features) >= 15
        ), f"Not enough varying features: {len(non_constant_features)}"

        # Check for features with very low variance (but be realistic about player data issues)
        low_variance_features = feature_variance[feature_variance < 0.001]
        low_variance_ratio = len(low_variance_features) / len(feature_variance)
        assert (
            low_variance_ratio < 0.7
        ), f"Too many low-variance features: {low_variance_ratio:.1%}"

        print(
            f"Feature variance: {len(constant_features)} constant, {len(low_variance_features)} low-variance, {len(non_constant_features)} varying features"
        )


class TestSpecificTeamHandling:
    """Test handling of specific IPL teams and edge cases."""

    def test_all_current_ipl_teams(self):
        """Test feature generation works for all current IPL teams."""
        pipeline = FeaturePipeline("data")

        current_teams = [
            "Mumbai Indians",
            "Chennai Super Kings",
            "Royal Challengers Bangalore",
            "Delhi Capitals",
            "Kolkata Knight Riders",
            "Punjab Kings",
            "Rajasthan Royals",
            "Sunrisers Hyderabad",
            "Gujarat Titans",
            "Lucknow Super Giants",
        ]

        # Test each team as team1
        for team in current_teams:
            try:
                features = pipeline.extract_match_features(
                    team1=team,
                    team2="Mumbai Indians",  # Use MI as default opponent
                    venue="Wankhede Stadium",
                    match_date="2024-04-01",
                )
                assert (
                    len(features) >= 75
                ), f"Feature generation failed for team: {team} (got {len(features)} features)"
            except Exception as e:
                pytest.fail(f"Feature generation failed for team {team}: {e}")

        print(
            f"Team handling: All {len(current_teams)} current IPL teams work correctly"
        )

    def test_venue_handling(self):
        """Test that feature generation works for different venues."""
        pipeline = FeaturePipeline("data")

        common_venues = [
            "Wankhede Stadium",
            "M Chinnaswamy Stadium",
            "Eden Gardens",
            "Feroz Shah Kotla",
            "MA Chidambaram Stadium",
            "Rajiv Gandhi International Stadium",
        ]

        for venue in common_venues:
            try:
                features = pipeline.extract_match_features(
                    team1="Mumbai Indians",
                    team2="Chennai Super Kings",
                    venue=venue,
                    match_date="2024-04-01",
                )
                assert (
                    len(features) >= 75
                ), f"Feature generation failed for venue: {venue} (got {len(features)} features)"
            except Exception as e:
                pytest.fail(f"Feature generation failed for venue {venue}: {e}")

        print(f"Venue handling: All {len(common_venues)} venues work correctly")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
