"""
Data Loading and Quality Tests
Tests that the IPL data files load correctly and meet quality standards
"""

import sys
from pathlib import Path

import pandas as pd
import pytest

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from ipl_predictor.data.loader import IPLDataLoader


class TestDataLoading:
    """Test actual data loading functionality."""

    def test_matches_data_loads_correctly(self):
        """Test that matches data loads with expected structure and volume."""
        loader = IPLDataLoader("data")
        matches = loader.load_matches()

        # Volume validation
        assert len(matches) > 1000, f"Expected >1000 matches, got {len(matches)}"

        # Structure validation
        required_columns = ["team1", "team2", "winner", "match_date", "venue"]
        for col in required_columns:
            assert col in matches.columns, f"Missing required column: {col}"

        # Data type validation
        assert pd.api.types.is_datetime64_any_dtype(
            matches["match_date"]
        ), "match_date should be datetime"

        print(f"✅ Matches data: {len(matches)} records loaded successfully")

    def test_ball_by_ball_data_loads_correctly(self):
        """Test that ball-by-ball data loads with expected structure and volume."""
        loader = IPLDataLoader("data")
        balls = loader.load_ball_by_ball()

        # Volume validation
        assert len(balls) > 200000, f"Expected >200k ball records, got {len(balls)}"

        # Structure validation
        required_columns = ["Batter", "Bowler", "BatsmanRun", "ID", "BattingTeam"]
        for col in required_columns:
            assert col in balls.columns, f"Missing required column: {col}"

        print(f"✅ Ball-by-ball data: {len(balls)} records loaded successfully")

    def test_players_data_loads_correctly(self):
        """Test that players data loads correctly."""
        loader = IPLDataLoader("data")
        players = loader.load_players()

        # Volume validation
        assert len(players) > 200, f"Expected >200 players, got {len(players)}"

        # Structure validation
        required_columns = ["Name", "battingStyles", "bowlingStyles"]
        for col in required_columns:
            assert col in players.columns, f"Missing required column: {col}"

        print(f"✅ Players data: {len(players)} records loaded successfully")

    def test_data_consistency_between_sources(self):
        """Test that match IDs are consistent between matches and ball-by-ball data."""
        loader = IPLDataLoader("data")
        matches = loader.load_matches()
        balls = loader.load_ball_by_ball()

        # Get match IDs from both sources
        match_ids_in_matches = set(matches["match_number"].unique())
        match_ids_in_balls = set(balls["ID"].unique())

        # Find common matches
        common_matches = match_ids_in_matches & match_ids_in_balls

        # Should have substantial overlap
        assert (
            len(common_matches) > 500
        ), f"Only {len(common_matches)} common match IDs found"

        # Check coverage
        coverage = len(common_matches) / len(match_ids_in_matches)
        assert (
            coverage > 0.8
        ), f"Low coverage: {coverage:.2%} of matches have ball-by-ball data"

        print(f"✅ Data consistency: {len(common_matches)} matches with complete data")


class TestDataQuality:
    """Test data quality and integrity."""

    def test_team_win_distribution_is_balanced(self):
        """Test that no single team dominates win distribution."""
        loader = IPLDataLoader("data")
        matches = loader.load_matches()

        # Check win distribution
        team_wins = matches["winner"].value_counts()
        total_matches = len(matches)

        # No team should have more than 20% of total wins
        max_win_percentage = team_wins.max() / total_matches
        assert (
            max_win_percentage < 0.20
        ), f"One team has {max_win_percentage:.1%} of wins - data may be unbalanced"

        # Should have multiple teams with wins
        teams_with_wins = len(team_wins)
        assert teams_with_wins >= 8, f"Only {teams_with_wins} teams have wins"

        print(
            f"✅ Win distribution: {teams_with_wins} teams, max {max_win_percentage:.1%} per team"
        )

    def test_date_range_coverage(self):
        """Test that data covers multiple IPL seasons."""
        loader = IPLDataLoader("data")
        matches = loader.load_matches()

        # Ensure dates are parsed
        matches["match_date"] = pd.to_datetime(matches["match_date"])

        # Check date range
        min_date = matches["match_date"].min()
        max_date = matches["match_date"].max()
        date_range = (max_date - min_date).days

        # Should span multiple years
        assert (
            date_range > 5000
        ), f"Date range only {date_range} days - expected multiple IPL seasons"

        # Check we have recent data
        assert (
            max_date.year >= 2020
        ), f"Latest data from {max_date.year} - may be outdated"

        print(f"✅ Date coverage: {min_date.year}-{max_date.year} ({date_range} days)")

    def test_no_excessive_missing_values(self):
        """Test that critical columns don't have excessive missing values."""
        loader = IPLDataLoader("data")
        matches = loader.load_matches()

        # Critical columns that shouldn't have many missing values
        critical_columns = ["team1", "team2", "winner", "venue"]

        for col in critical_columns:
            if col in matches.columns:
                missing_pct = matches[col].isnull().mean()
                assert (
                    missing_pct < 0.05
                ), f"Column {col} has {missing_pct:.1%} missing values"

        print("✅ Missing values check passed for critical columns")

    def test_venue_and_team_name_consistency(self):
        """Test that venue and team names are reasonably consistent."""
        loader = IPLDataLoader("data")
        matches = loader.load_matches()

        # Check number of unique venues (IPL has many venues over the years)
        unique_venues = matches["venue"].nunique()
        assert (
            30 <= unique_venues <= 80
        ), f"Unexpected number of venues: {unique_venues}"

        # Check number of teams (IPL has had various teams over the years)
        all_teams = pd.concat([matches["team1"], matches["team2"]]).unique()
        num_teams = len(all_teams)
        assert 8 <= num_teams <= 20, f"Unexpected number of teams: {num_teams}"

        print(f"✅ Consistency: {num_teams} teams, {unique_venues} venues")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
