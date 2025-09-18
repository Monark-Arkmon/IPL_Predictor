"""
Data Integration and Enhancement Script
Merges alternative datasets to enhance IPL prediction features
"""

import pandas as pd
import numpy as np
import json
import os
from pathlib import Path
from fuzzywuzzy import fuzz, process
import warnings

warnings.filterwarnings("ignore")


class IPLDataIntegrator:
    def __init__(self):
        self.base_path = Path(".")
        self.main_ball_data: pd.DataFrame = pd.DataFrame()
        self.main_player_data: pd.DataFrame = pd.DataFrame()
        self.orig_player_data: pd.DataFrame = pd.DataFrame()
        self.enhanced_players: pd.DataFrame = pd.DataFrame()
        self.match_context: pd.DataFrame = pd.DataFrame()

    def load_data(self):
        """Load main datasets"""
        print("Loading main datasets...")

        self.main_ball_data = pd.read_csv(
            "data/raw/ball_by_ball/ipl_ball_by_ball_2008_2024.csv"
        )
        self.main_player_data = pd.read_csv("data/raw/players/ipl_players_2024.csv")
        self.orig_player_data = pd.read_csv("data/external/ipl_dataset_original.csv")

        print(f"Loaded {len(self.main_player_data)} current players")
        print(f"Loaded {len(self.orig_player_data)} historical players with stats")

    def fuzzy_match_players(self, threshold=80):
        """Match players between datasets using fuzzy string matching"""
        print(f"\nMatching players with fuzzy matching (threshold={threshold})...")

        # Prepare player names
        current_players = self.main_player_data["Name"].str.strip().str.lower()
        historical_players = self.orig_player_data["Player"].str.strip().str.lower()

        matches = []

        for idx, current_name in enumerate(current_players):
            # Find best match in historical data
            best_match = process.extractOne(
                current_name, historical_players, scorer=fuzz.ratio
            )

            if best_match and best_match[1] >= threshold:
                historical_idx = historical_players[
                    historical_players == best_match[0]
                ].index[0]
                matches.append(
                    {
                        "current_idx": idx,
                        "historical_idx": historical_idx,
                        "current_name": self.main_player_data.iloc[idx]["Name"],
                        "historical_name": self.orig_player_data.iloc[historical_idx][
                            "Player"
                        ],
                        "match_score": best_match[1],
                    }
                )

        print(f"Successfully matched {len(matches)} players")
        return matches

    def enhance_player_data(self):
        """Enhance current player data with historical statistics"""
        print("\\nEnhancing player data with career statistics...")

        # Get player matches
        matches = self.fuzzy_match_players()

        # Start with current player data
        self.enhanced_players = self.main_player_data.copy()

        # Add enhancement columns
        enhancement_columns = [
            "Runs",
            "Avg",
            "SR",
            "HS",
            "100s",
            "50s",
            "4s",
            "6s",  # Batting
            "B_Wkts",
            "B_Avg",
            "B_SR",
            "B_Econ",
            "B_5w",
            "B_4w",  # Bowling
            "Mat",
            "SOLD_PRICE",
            "AGE",
            "CAPTAINCY EXP",  # General
        ]

        # Initialize new columns
        for col in enhancement_columns:
            self.enhanced_players[col] = np.nan

        # Fill matched player data
        for match in matches:
            current_idx = match["current_idx"]
            historical_idx = match["historical_idx"]

            for col in enhancement_columns:
                if col in self.orig_player_data.columns:
                    value = self.orig_player_data.iloc[historical_idx][col]
                    self.enhanced_players.at[current_idx, col] = value

        # Fill missing numerical values with 0
        for col in enhancement_columns:
            if col in ["SOLD_PRICE", "AGE", "CAPTAINCY EXP"]:
                continue  # Keep NaN for these
            self.enhanced_players[col] = pd.to_numeric(
                self.enhanced_players[col], errors="coerce"
            ).fillna(0)

        print(f"Enhanced {len(matches)} players with career statistics")

        # Display sample enhancement
        enhanced_sample = self.enhanced_players[self.enhanced_players["Runs"] > 0].head(
            3
        )
        print("\\nSample enhanced players:")
        for _, player in enhanced_sample.iterrows():
            print(
                f"  {player['Name']}: {player['Runs']} runs, {player['Avg']:.1f} avg, {player['B_Wkts']} wickets"
            )

    def extract_match_context_from_json(self, limit=100):
        """Extract match context from JSON files"""
        print(
            f"\\nExtracting match context from JSON files (processing {limit} files)..."
        )

        json_dir = Path("data/raw/json_match_details/ipl_match")
        json_files = list(json_dir.glob("*.json"))[:limit]  # Limit for performance

        match_contexts = []

        for json_file in json_files:
            try:
                with open(json_file, "r") as f:
                    data = json.load(f)

                if "info" in data:
                    info = data["info"]
                    match_id = json_file.stem

                    context = {
                        "match_id": match_id,
                        "venue": info.get("venue", ""),
                        "city": info.get("city", ""),
                        "toss_winner": info.get("toss", {}).get("winner", ""),
                        "toss_decision": info.get("toss", {}).get("decision", ""),
                        "player_of_match": (
                            info.get("player_of_match", [""])[0]
                            if info.get("player_of_match")
                            else ""
                        ),
                        "season": info.get("season", ""),
                        "match_type": info.get("match_type", ""),
                        "outcome_winner": info.get("outcome", {}).get("winner", ""),
                        "outcome_margin": info.get("outcome", {}).get("by", {}),
                    }

                    match_contexts.append(context)

            except Exception as e:
                print(f"Error processing {json_file}: {e}")
                continue

        self.match_context = pd.DataFrame(match_contexts)
        print(f"Extracted context for {len(self.match_context)} matches")

        # Display sample context
        if not self.match_context.empty:
            print("\\nSample match contexts:")
            for _, match in self.match_context.head(3).iterrows():
                print(
                    f"  Match {match['match_id']}: {match['venue']}, won by {match['outcome_winner']}"
                )

    def create_enhanced_features(self):
        """Create new feature columns from enhanced data"""
        print("\\nCreating enhanced features...")

        if not self.enhanced_players.empty:
            # Batting performance categories
            self.enhanced_players["batting_tier"] = pd.cut(
                self.enhanced_players["Avg"],
                bins=[0, 20, 30, 40, 100],
                labels=["Poor", "Average", "Good", "Excellent"],
            )

            # Bowling performance categories
            self.enhanced_players["bowling_tier"] = pd.cut(
                self.enhanced_players["B_Avg"],
                bins=[0, 20, 25, 30, 100],
                labels=["Excellent", "Good", "Average", "Poor"],
            )

            # All-rounder score (combined batting + bowling)
            batting_score = (self.enhanced_players["Avg"] / 50) * 50  # Normalize to 50
            bowling_score = np.where(
                self.enhanced_players["B_Avg"] > 0,
                (35 / self.enhanced_players["B_Avg"]) * 50,  # Lower avg is better
                0,
            )
            self.enhanced_players["allrounder_score"] = batting_score + bowling_score

            # Experience level
            self.enhanced_players["experience_level"] = pd.cut(
                self.enhanced_players["Mat"],
                bins=[0, 20, 50, 100, 500],
                labels=["Rookie", "Developing", "Experienced", "Veteran"],
            )

            print("Created enhanced feature columns:")
            print("  - batting_tier, bowling_tier")
            print("  - allrounder_score")
            print("  - experience_level")

    def save_enhanced_data(self):
        """Save enhanced datasets"""
        print("\\nSaving enhanced datasets...")

        # Create enhanced data directory
        enhanced_dir = Path("data/enhanced")
        enhanced_dir.mkdir(exist_ok=True)

        # Save enhanced players
        if not self.enhanced_players.empty:
            output_path = enhanced_dir / "enhanced_players.csv"
            self.enhanced_players.to_csv(output_path, index=False)
            print(f"Saved enhanced players to {output_path}")

        # Save match context
        if not self.match_context.empty:
            context_path = enhanced_dir / "match_context.csv"
            self.match_context.to_csv(context_path, index=False)
            print(f"Saved match context to {context_path}")

        print("\\nEnhancement summary:")
        print(f"  - Enhanced {len(self.enhanced_players)} players with career stats")
        print(f"  - Extracted context for {len(self.match_context)} matches")
        print(f"  - Added 4 new feature categories")

    def run_integration(self):
        """Run complete data integration process"""
        print("Starting IPL Data Integration...")

        self.load_data()
        self.enhance_player_data()
        self.extract_match_context_from_json()
        self.create_enhanced_features()
        self.save_enhanced_data()

        print("\\n=== INTEGRATION COMPLETE ===")
        return {
            "enhanced_players": len(self.enhanced_players),
            "match_contexts": len(self.match_context),
        }


if __name__ == "__main__":
    integrator = IPLDataIntegrator()
    results = integrator.run_integration()
