"""
Data Validation and Integration Script
Validates main dataset against alternative sources and prepares data integration
"""

import pandas as pd
import numpy as np
import json
import os
from pathlib import Path
import warnings

warnings.filterwarnings("ignore")


class IPLDataValidator:
    def __init__(self):
        self.base_path = Path(".")
        self.main_ball_data: pd.DataFrame = pd.DataFrame()
        self.alt_ball_data: pd.DataFrame = pd.DataFrame()
        self.main_player_data: pd.DataFrame = pd.DataFrame()
        self.orig_player_data: pd.DataFrame = pd.DataFrame()
        self.validation_results = {}

    def load_datasets(self):
        """Load all datasets for comparison"""
        print("Loading datasets...")

        # Main datasets
        self.main_ball_data = pd.read_csv(
            "data/raw/ball_by_ball/ipl_ball_by_ball_2008_2024.csv"
        )
        self.main_player_data = pd.read_csv("data/raw/players/ipl_players_2024.csv")

        # Alternative datasets
        self.alt_ball_data = pd.read_csv(
            "data/external/ipl_ball_by_ball_alternative.csv", low_memory=False
        )
        self.orig_player_data = pd.read_csv("data/external/ipl_dataset_original.csv")

        print(f"Main ball-by-ball: {self.main_ball_data.shape}")
        print(f"Alternative ball-by-ball: {self.alt_ball_data.shape}")
        print(f"Main players: {self.main_player_data.shape}")
        print(f"Original players: {self.orig_player_data.shape}")

    def validate_ball_data_coverage(self):
        """Compare coverage between main and alternative ball-by-ball data"""
        print("\n=== BALL-BY-BALL DATA VALIDATION ===")

        # Schema comparison
        main_cols = set(self.main_ball_data.columns)
        alt_cols = set(self.alt_ball_data.columns)

        print(f"Main dataset columns: {len(main_cols)}")
        print(f"Alternative dataset columns: {len(alt_cols)}")
        print(f"Common columns: {len(main_cols & alt_cols)}")
        print(f"Only in main: {main_cols - alt_cols}")
        print(f"Only in alternative: {alt_cols - main_cols}")

        # Season coverage comparison
        print("\n--- Season Coverage ---")
        try:
            main_seasons = (
                sorted(self.main_ball_data["season"].unique())
                if "season" in self.main_ball_data.columns
                else []
            )
            alt_seasons = (
                self.alt_ball_data["Season"].unique()
                if "Season" in self.alt_ball_data.columns
                else []
            )
            # Handle mixed types in Season column
            alt_seasons_clean = [str(s) for s in alt_seasons if pd.notna(s)]
            alt_seasons_sorted = sorted(set(alt_seasons_clean))

            print(f"Main seasons: {main_seasons}")
            print(f"Alternative seasons: {alt_seasons_sorted}")
        except Exception as e:
            print(f"Season comparison error: {e}")

        # Match coverage
        main_matches = (
            len(self.main_ball_data["match_id"].unique())
            if "match_id" in self.main_ball_data.columns
            else 0
        )
        alt_matches = (
            len(self.alt_ball_data["Match id"].unique())
            if "Match id" in self.alt_ball_data.columns
            else 0
        )

        print(f"\nMain dataset matches: {main_matches}")
        print(f"Alternative dataset matches: {alt_matches}")

        self.validation_results["ball_data"] = {
            "main_shape": self.main_ball_data.shape,
            "alt_shape": self.alt_ball_data.shape,
            "main_matches": main_matches,
            "alt_matches": alt_matches,
            "common_columns": len(main_cols & alt_cols),
            "schema_diff": {
                "only_main": list(main_cols - alt_cols),
                "only_alt": list(alt_cols - main_cols),
            },
        }

    def validate_player_data(self):
        """Compare player datasets and identify enhancement opportunities"""
        print("\n=== PLAYER DATA VALIDATION ===")

        # Schema comparison
        main_cols = set(self.main_player_data.columns)
        orig_cols = set(self.orig_player_data.columns)

        print(f"Main player dataset columns: {len(main_cols)}")
        print(f"Original player dataset columns: {len(orig_cols)}")
        print(f"Common columns: {len(main_cols & orig_cols)}")

        print(f"\nMain columns: {list(main_cols)}")
        print(f"\nOriginal columns: {list(orig_cols)}")

        # Player overlap
        main_players = (
            set(self.main_player_data["player_name"].str.lower())
            if "player_name" in self.main_player_data.columns
            else set()
        )
        orig_players = (
            set(self.orig_player_data["Player"].str.lower())
            if "Player" in self.orig_player_data.columns
            else set()
        )

        overlap = len(main_players & orig_players)
        print(f"\nPlayer name overlap: {overlap} players")
        print(f"Only in main: {len(main_players - orig_players)}")
        print(f"Only in original: {len(orig_players - main_players)}")

        # Sample career statistics available
        if "Bat Avg" in self.orig_player_data.columns:
            bat_avg_available = self.orig_player_data["Bat Avg"].notna().sum()
            print(f"Players with batting averages: {bat_avg_available}")

        if "Bowl Avg" in self.orig_player_data.columns:
            bowl_avg_available = self.orig_player_data["Bowl Avg"].notna().sum()
            print(f"Players with bowling averages: {bowl_avg_available}")

        self.validation_results["player_data"] = {
            "main_players": len(main_players),
            "orig_players": len(orig_players),
            "overlap": overlap,
            "enhancement_columns": list(orig_cols - main_cols),
        }

    def analyze_json_structure(self):
        """Analyze JSON match data structure"""
        print("\n=== JSON DATA ANALYSIS ===")

        json_dir = Path("data/raw/json_match_details/ipl_match")
        if not json_dir.exists():
            print("JSON directory not found")
            return

        json_files = list(json_dir.glob("*.json"))
        print(f"Total JSON files: {len(json_files)}")

        if json_files:
            # Sample a few files to understand structure
            sample_file = json_files[0]
            try:
                with open(sample_file, "r") as f:
                    sample_data = json.load(f)

                print(f"Sample file: {sample_file.name}")
                print(f"Top-level keys: {list(sample_data.keys())}")

                if "info" in sample_data:
                    info_keys = list(sample_data["info"].keys())
                    print(f"Info keys: {info_keys}")

                    # Check for useful context data
                    useful_keys = [
                        "city",
                        "venue",
                        "toss",
                        "player_of_match",
                        "outcome",
                    ]
                    available_useful = [key for key in useful_keys if key in info_keys]
                    print(f"Useful context keys available: {available_useful}")

            except Exception as e:
                print(f"Error reading JSON: {e}")

        self.validation_results["json_data"] = {
            "total_files": len(json_files),
            "sample_structure": "analyzed" if json_files else "unavailable",
        }

    def generate_integration_plan(self):
        """Generate specific integration recommendations"""
        print("\n=== INTEGRATION RECOMMENDATIONS ===")

        # Ball-by-ball data recommendations
        if self.validation_results.get("ball_data"):
            ball_results = self.validation_results["ball_data"]
            print(f"1. Ball-by-ball data:")
            print(f"   - Main has {ball_results['main_shape'][0]:,} records")
            print(f"   - Alternative has {ball_results['alt_shape'][0]:,} records")

            if ball_results["alt_shape"][0] < ball_results["main_shape"][0]:
                print(
                    "    Recommendation: Use main dataset as primary, alternative for validation"
                )
            else:
                print(
                    "    Recommendation: Compare data quality, potentially merge datasets"
                )

        # Player data recommendations
        if self.validation_results.get("player_data"):
            player_results = self.validation_results["player_data"]
            enhancement_cols = player_results.get("enhancement_columns", [])

            print(f"\n2. Player data enhancement:")
            print(f"   - {player_results['overlap']} players can be enhanced")
            print(f"   - Available enhancement columns: {len(enhancement_cols)}")

            if enhancement_cols:
                career_stats = [
                    col
                    for col in enhancement_cols
                    if any(
                        stat in col.lower() for stat in ["avg", "sr", "runs", "wickets"]
                    )
                ]
                print(f"   - Career statistics columns: {len(career_stats)}")
                print(
                    "    Recommendation: Merge career stats for enhanced player features"
                )

        print(f"\n3. Next steps:")
        print("   - Create data merge scripts")
        print("   - Implement feature engineering pipeline")
        print("   - Test enhanced model performance")

    def run_validation(self):
        """Run complete validation process"""
        print("Starting IPL Data Validation...")

        self.load_datasets()
        self.validate_ball_data_coverage()
        self.validate_player_data()
        self.analyze_json_structure()
        self.generate_integration_plan()

        print("\n=== VALIDATION COMPLETE ===")
        return self.validation_results


if __name__ == "__main__":
    validator = IPLDataValidator()
    results = validator.run_validation()
