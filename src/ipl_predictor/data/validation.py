"""
Data validation module for IPL Match Predictor.
Implements comprehensive validation for real IPL data constraints.
"""

import logging
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class IPLDataValidator:
    """Comprehensive validator for IPL data with real constraints."""

    # IPL teams (all teams that have played in IPL)
    VALID_IPL_TEAMS = {
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
        "Delhi Daredevils",
        "Kings XI Punjab",
        "Rising Pune Supergiant",
        "Gujarat Lions",
        "Kochi Tuskers Kerala",
        "Pune Warriors India",
        "Deccan Chargers",
    }

    # Valid IPL cities/venues
    VALID_IPL_CITIES = {
        "Mumbai",
        "Chennai",
        "Bangalore",
        "Delhi",
        "Kolkata",
        "Jaipur",
        "Hyderabad",
        "Pune",
        "Ahmedabad",
        "Lucknow",
        "Mohali",
        "Indore",
        "Rajkot",
        "Visakhapatnam",
        "Guwahati",
        "Ranchi",
        "Dharamsala",
        "Kochi",
        "Nagpur",
        "Cuttack",
        "Raipur",
        "Kanpur",
    }

    # Valid seasons
    VALID_SEASONS = list(range(2008, 2025))  # IPL started in 2008

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize validator with configuration."""
        self.config = config or {}
        self.validation_errors = []
        self.validation_warnings = []

    def validate_matches_data(
        self, matches_df: pd.DataFrame
    ) -> Tuple[bool, List[str], List[str]]:
        """Validate matches dataset against IPL constraints."""
        logger.info("Validating matches data against IPL constraints...")

        self.validation_errors = []
        self.validation_warnings = []

        # Required columns check
        required_columns = ["id", "season", "date", "team1", "team2", "winner"]
        self._check_required_columns(matches_df, required_columns, "matches")

        # Data types validation
        self._validate_matches_data_types(matches_df)

        # IPL-specific validations
        self._validate_teams(matches_df)
        self._validate_seasons(matches_df)
        self._validate_cities_venues(matches_df)
        self._validate_match_logic(matches_df)
        self._validate_date_consistency(matches_df)

        # Data quality checks
        self._check_data_quality(matches_df, "matches")

        is_valid = len(self.validation_errors) == 0
        return is_valid, self.validation_errors, self.validation_warnings

    def validate_deliveries_data(
        self, deliveries_df: pd.DataFrame
    ) -> Tuple[bool, List[str], List[str]]:
        """Validate deliveries dataset against IPL constraints."""
        logger.info("Validating deliveries data against IPL constraints...")

        self.validation_errors = []
        self.validation_warnings = []

        # Required columns check
        required_columns = [
            "match_id",
            "inning",
            "batting_team",
            "bowling_team",
            "over",
            "ball",
            "batsman_runs",
            "total_runs",
        ]
        self._check_required_columns(deliveries_df, required_columns, "deliveries")

        # Data types validation
        self._validate_deliveries_data_types(deliveries_df)

        # Cricket-specific validations
        self._validate_cricket_logic(deliveries_df)
        self._validate_teams_in_deliveries(deliveries_df)

        # Data quality checks
        self._check_data_quality(deliveries_df, "deliveries")

        is_valid = len(self.validation_errors) == 0
        return is_valid, self.validation_errors, self.validation_warnings

    def validate_combined_data(
        self, matches_df: pd.DataFrame, deliveries_df: pd.DataFrame
    ) -> Tuple[bool, List[str], List[str]]:
        """Validate consistency between matches and deliveries data."""
        logger.info("Validating consistency between matches and deliveries...")

        self.validation_errors = []
        self.validation_warnings = []

        # Check match ID consistency
        match_ids_matches = set(matches_df["id"].unique())
        match_ids_deliveries = set(deliveries_df["match_id"].unique())

        missing_in_deliveries = match_ids_matches - match_ids_deliveries
        extra_in_deliveries = match_ids_deliveries - match_ids_matches

        if missing_in_deliveries:
            self.validation_warnings.append(
                f"Matches without deliveries data: {len(missing_in_deliveries)} matches"
            )

        if extra_in_deliveries:
            self.validation_warnings.append(
                f"Deliveries data for non-existent matches: {len(extra_in_deliveries)} matches"
            )

        # Validate team consistency
        self._validate_team_consistency(matches_df, deliveries_df)

        is_valid = len(self.validation_errors) == 0
        return is_valid, self.validation_errors, self.validation_warnings

    def _check_required_columns(
        self, df: pd.DataFrame, required_cols: List[str], dataset_name: str
    ):
        """Check if all required columns are present."""
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            self.validation_errors.append(
                f"{dataset_name}: Missing required columns: {missing_cols}"
            )

    def _validate_matches_data_types(self, df: pd.DataFrame):
        """Validate data types for matches dataset."""
        try:
            # Check if date can be converted to datetime
            pd.to_datetime(df["date"])
        except:
            self.validation_errors.append(
                "matches: 'date' column cannot be converted to datetime"
            )

        # Check numeric columns
        numeric_cols = ["id", "season"]
        for col in numeric_cols:
            if col in df.columns and not pd.api.types.is_numeric_dtype(df[col]):
                self.validation_errors.append(f"matches: '{col}' should be numeric")

    def _validate_deliveries_data_types(self, df: pd.DataFrame):
        """Validate data types for deliveries dataset."""
        numeric_cols = [
            "match_id",
            "inning",
            "over",
            "ball",
            "batsman_runs",
            "extra_runs",
            "total_runs",
        ]

        for col in numeric_cols:
            if col in df.columns and not pd.api.types.is_numeric_dtype(df[col]):
                self.validation_errors.append(f"deliveries: '{col}' should be numeric")

    def _validate_teams(self, df: pd.DataFrame):
        """Validate team names against known IPL teams."""
        team_columns = ["team1", "team2", "winner", "toss_winner"]

        for col in team_columns:
            if col in df.columns:
                invalid_teams = set(df[col].dropna().unique()) - self.VALID_IPL_TEAMS
                if invalid_teams:
                    self.validation_warnings.append(
                        f"matches: Unknown team names in '{col}': {invalid_teams}"
                    )

    def _validate_seasons(self, df: pd.DataFrame):
        """Validate season numbers."""
        if "season" in df.columns:
            invalid_seasons = set(df["season"].dropna().unique()) - set(
                self.VALID_SEASONS
            )
            if invalid_seasons:
                self.validation_warnings.append(
                    f"matches: Invalid seasons: {invalid_seasons}"
                )

    def _validate_cities_venues(self, df: pd.DataFrame):
        """Validate cities and venues."""
        if "city" in df.columns:
            invalid_cities = set(df["city"].dropna().unique()) - self.VALID_IPL_CITIES
            if invalid_cities:
                self.validation_warnings.append(
                    f"matches: Unknown cities: {invalid_cities}"
                )

    def _validate_match_logic(self, df: pd.DataFrame):
        """Validate match logic constraints."""
        # Team1 and Team2 should be different
        if "team1" in df.columns and "team2" in df.columns:
            same_teams = df[df["team1"] == df["team2"]]
            if len(same_teams) > 0:
                self.validation_errors.append(
                    f"matches: {len(same_teams)} matches where team1 == team2"
                )

        # Winner should be either team1 or team2
        if all(col in df.columns for col in ["team1", "team2", "winner"]):
            invalid_winners = df[
                ~df["winner"].isin(df["team1"]) & ~df["winner"].isin(df["team2"])
            ]
            if len(invalid_winners) > 0:
                self.validation_errors.append(
                    f"matches: {len(invalid_winners)} matches with invalid winners"
                )

    def _validate_date_consistency(self, df: pd.DataFrame):
        """Validate date consistency with seasons."""
        if "date" in df.columns and "season" in df.columns:
            df["date"] = pd.to_datetime(df["date"])
            df["year"] = df["date"].dt.year

            # IPL seasons typically align with calendar years
            inconsistent_dates = df[df["year"] != df["season"]]
            if len(inconsistent_dates) > 0:
                self.validation_warnings.append(
                    f"matches: {len(inconsistent_dates)} matches with season-date mismatch"
                )

    def _validate_cricket_logic(self, df: pd.DataFrame):
        """Validate cricket-specific logic in deliveries."""
        # Over should be between 1-20 for T20
        if "over" in df.columns:
            invalid_overs = df[(df["over"] < 1) | (df["over"] > 20)]
            if len(invalid_overs) > 0:
                self.validation_errors.append(
                    f"deliveries: {len(invalid_overs)} invalid over numbers"
                )

        # Ball should be between 1-6
        if "ball" in df.columns:
            invalid_balls = df[(df["ball"] < 1) | (df["ball"] > 6)]
            if len(invalid_balls) > 0:
                self.validation_errors.append(
                    f"deliveries: {len(invalid_balls)} invalid ball numbers"
                )

        # Runs validation
        if "batsman_runs" in df.columns:
            invalid_runs = df[(df["batsman_runs"] < 0) | (df["batsman_runs"] > 6)]
            if len(invalid_runs) > 0:
                self.validation_errors.append(
                    f"deliveries: {len(invalid_runs)} invalid batsman runs"
                )

        # Inning should be 1 or 2
        if "inning" in df.columns:
            invalid_innings = df[~df["inning"].isin([1, 2])]
            if len(invalid_innings) > 0:
                self.validation_errors.append(
                    f"deliveries: {len(invalid_innings)} invalid inning numbers"
                )

    def _validate_teams_in_deliveries(self, df: pd.DataFrame):
        """Validate team names in deliveries data."""
        team_cols = ["batting_team", "bowling_team"]

        for col in team_cols:
            if col in df.columns:
                invalid_teams = set(df[col].dropna().unique()) - self.VALID_IPL_TEAMS
                if invalid_teams:
                    self.validation_warnings.append(
                        f"deliveries: Unknown teams in '{col}': {invalid_teams}"
                    )

    def _validate_team_consistency(
        self, matches_df: pd.DataFrame, deliveries_df: pd.DataFrame
    ):
        """Validate team consistency between matches and deliveries."""
        # Get teams from both datasets
        match_teams = set(
            pd.concat([matches_df["team1"], matches_df["team2"]]).unique()
        )
        delivery_teams = set(
            pd.concat(
                [deliveries_df["batting_team"], deliveries_df["bowling_team"]]
            ).unique()
        )

        teams_only_in_matches = match_teams - delivery_teams
        teams_only_in_deliveries = delivery_teams - match_teams

        if teams_only_in_matches:
            self.validation_warnings.append(
                f"Teams in matches but not in deliveries: {teams_only_in_matches}"
            )

        if teams_only_in_deliveries:
            self.validation_warnings.append(
                f"Teams in deliveries but not in matches: {teams_only_in_deliveries}"
            )

    def _check_data_quality(self, df: pd.DataFrame, dataset_name: str):
        """Check general data quality metrics."""
        # Check for empty dataset
        if df.empty:
            self.validation_errors.append(f"{dataset_name}: Dataset is empty")
            return

        # Check missing values
        missing_pct = (df.isnull().sum() / len(df)) * 100
        high_missing = missing_pct[missing_pct > 50]

        if len(high_missing) > 0:
            self.validation_warnings.append(
                f"{dataset_name}: Columns with >50% missing values: {high_missing.to_dict()}"
            )

        # Check for duplicates
        if "id" in df.columns:
            duplicates = df.duplicated(subset=["id"]).sum()
            if duplicates > 0:
                self.validation_errors.append(
                    f"{dataset_name}: {duplicates} duplicate records"
                )

        logger.info(
            f"{dataset_name} quality check: {len(df)} records, "
            f"{df.isnull().sum().sum()} total missing values"
        )


def validate_ipl_data(
    matches_df: pd.DataFrame,
    deliveries_df: pd.DataFrame,
    config: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Comprehensive validation function for IPL data.

    Returns:
        Dict with validation results and recommendations
    """
    validator = IPLDataValidator(config)

    # Validate individual datasets
    matches_valid, matches_errors, matches_warnings = validator.validate_matches_data(
        matches_df
    )
    deliveries_valid, deliveries_errors, deliveries_warnings = (
        validator.validate_deliveries_data(deliveries_df)
    )

    # Validate combined data
    combined_valid, combined_errors, combined_warnings = (
        validator.validate_combined_data(matches_df, deliveries_df)
    )

    overall_valid = matches_valid and deliveries_valid and combined_valid

    return {
        "overall_valid": overall_valid,
        "matches": {
            "valid": matches_valid,
            "errors": matches_errors,
            "warnings": matches_warnings,
        },
        "deliveries": {
            "valid": deliveries_valid,
            "errors": deliveries_errors,
            "warnings": deliveries_warnings,
        },
        "combined": {
            "valid": combined_valid,
            "errors": combined_errors,
            "warnings": combined_warnings,
        },
        "summary": {
            "total_matches": len(matches_df),
            "total_deliveries": len(deliveries_df),
            "unique_teams": len(
                set(pd.concat([matches_df["team1"], matches_df["team2"]]).unique())
            ),
            "seasons_covered": (
                sorted(matches_df["season"].unique())
                if "season" in matches_df.columns
                else []
            ),
            "date_range": (
                (matches_df["date"].min(), matches_df["date"].max())
                if "date" in matches_df.columns
                else None
            ),
        },
    }
