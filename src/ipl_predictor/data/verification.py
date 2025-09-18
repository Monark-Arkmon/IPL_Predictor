"""
Data verification for IPL ML pipeline.
Ensures data integrity before training.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import logging
from datetime import datetime

from .pipeline import DataValidator
from ..utils.logging_config import setup_logging

logger = logging.getLogger(__name__)


class DataVerificationPipeline:
    """Complete data verification pipeline."""

    def __init__(self, data_path: str = "data"):
        self.data_path = Path(data_path)
        self.results = {}

    def verify_all_data(self) -> Dict[str, bool]:
        """Run complete data verification pipeline."""
        logger.info("Starting comprehensive data verification...")

        verification_results = {}

        # 1. Verify data files exist
        verification_results["files_exist"] = self._verify_files_exist()

        # 2. Verify data quality
        verification_results["data_quality"] = self._verify_data_quality()

        # 3. Verify data completeness
        verification_results["data_completeness"] = self._verify_data_completeness()

        # 4. Verify data consistency
        verification_results["data_consistency"] = self._verify_data_consistency()

        # Summary
        all_passed = all(verification_results.values())

        if all_passed:
            logger.info("All data verification checks passed!")
        else:
            failed_checks = [k for k, v in verification_results.items() if not v]
            logger.error(f"Data verification failed: {failed_checks}")

        self.results = verification_results
        return verification_results

    def _verify_files_exist(self) -> bool:
        """Verify required data files exist."""
        logger.info("Verifying data files exist...")

        required_files = [
            "raw/matches/ipl_match_info_2008_2024.csv",
            "raw/ball_by_ball",
        ]

        for file_path in required_files:
            full_path = self.data_path / file_path
            if not full_path.exists():
                logger.error(f"Missing required file: {file_path}")
                return False

        logger.info("All required files exist")
        return True

    def _verify_data_quality(self) -> bool:
        """Verify data quality standards."""
        logger.info("Verifying data quality...")

        try:
            # Load matches data
            matches_path = self.data_path / "raw/matches/ipl_match_info_2008_2024.csv"
            matches_df = pd.read_csv(matches_path)

            # Check data quality
            validator = DataValidator({})

            if not validator.validate_dataframe(matches_df, "matches"):
                logger.error("Matches data quality check failed")
                return False

            if not validator.validate_team_data(matches_df):
                logger.error("Team data validation failed")
                return False

            logger.info("Data quality checks passed")
            return True

        except Exception as e:
            logger.error(f"Data quality verification failed: {e}")
            return False

    def _verify_data_completeness(self) -> bool:
        """Verify data completeness."""
        logger.info("Verifying data completeness...")

        try:
            matches_path = self.data_path / "raw/matches/ipl_match_info_2008_2024.csv"
            matches_df = pd.read_csv(matches_path)

            # Check minimum data requirements
            min_matches = 500
            min_teams = 8
            min_seasons = 10

            if len(matches_df) < min_matches:
                logger.error(f"Insufficient matches: {len(matches_df)} < {min_matches}")
                return False

            unique_teams = len(
                pd.concat([matches_df["team1"], matches_df["team2"]]).unique()
            )
            if unique_teams < min_teams:
                logger.error(f"Insufficient teams: {unique_teams} < {min_teams}")
                return False

            # Check date range
            matches_df["match_date"] = pd.to_datetime(matches_df["match_date"])
            date_range = (
                matches_df["match_date"].max() - matches_df["match_date"].min()
            ).days
            min_days = min_seasons * 365

            if date_range < min_days:
                logger.error(
                    f"Insufficient date range: {date_range} days < {min_days} days"
                )
                return False

            logger.info(
                f"Data completeness verified: {len(matches_df)} matches, {unique_teams} teams, {date_range} days"
            )
            return True

        except Exception as e:
            logger.error(f"Data completeness verification failed: {e}")
            return False

    def _verify_data_consistency(self) -> bool:
        """Verify data consistency."""
        logger.info("Verifying data consistency...")

        try:
            matches_path = self.data_path / "raw/matches/ipl_match_info_2008_2024.csv"
            matches_df = pd.read_csv(matches_path)

            # Check for required columns
            required_columns = ["team1", "team2", "winner", "match_date", "venue"]
            missing_columns = [
                col for col in required_columns if col not in matches_df.columns
            ]

            if missing_columns:
                logger.error(f"Missing required columns: {missing_columns}")
                return False

            # Check data consistency
            # Winners should be either team1 or team2
            valid_winners = (
                matches_df["winner"]
                .isin(pd.concat([matches_df["team1"], matches_df["team2"]]))
                .sum()
            )

            total_with_winners = len(matches_df.dropna(subset=["winner"]))
            consistency_rate = (
                valid_winners / total_with_winners if total_with_winners > 0 else 0
            )

            if consistency_rate < 0.95:  # 95% consistency threshold
                logger.error(f"Data consistency too low: {consistency_rate:.2%}")
                return False

            logger.info(f"Data consistency verified: {consistency_rate:.2%} valid")
            return True

        except Exception as e:
            logger.error(f"Data consistency verification failed: {e}")
            return False

    def get_verification_summary(self) -> str:
        """Get human-readable verification summary."""
        if not self.results:
            return "No verification results available"

        summary = "\n" + "=" * 50 + "\n"
        summary += "DATA VERIFICATION SUMMARY\n"
        summary += "=" * 50 + "\n"

        for check, passed in self.results.items():
            status = "PASS" if passed else "FAIL"
            check_name = check.replace("_", " ").title()
            summary += f"{check_name:20} | {status}\n"

        summary += "=" * 50 + "\n"

        overall = (
            "READY FOR TRAINING"
            if all(self.results.values())
            else "FIX ISSUES BEFORE TRAINING"
        )
        summary += f"Overall Status: {overall}\n"
        summary += "=" * 50

        return summary
