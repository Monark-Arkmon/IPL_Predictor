"""
Team dynamics and composition feature engineering.
Analyzes team strength, balance, and performance patterns.
Enhanced with career statistics and player profiles.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
import logging
from collections import Counter
from pathlib import Path

logger = logging.getLogger(__name__)


class TeamFeatureEngine:
    """Generate team-level features and dynamics with enhanced player data."""

    def __init__(self):
        self.team_cache = {}
        self.enhanced_players = self._load_enhanced_players()

    def _load_enhanced_players(self) -> pd.DataFrame:
        """Load enhanced player data if available."""
        enhanced_path = Path("data/enhanced/enhanced_players.csv")
        if enhanced_path.exists():
            logger.info("Loading enhanced player data for team features")
            return pd.read_csv(enhanced_path)
        else:
            logger.warning("Enhanced player data not found for team features")
            return pd.DataFrame()

    def extract_team_composition_features(
        self, team_players: List[str]
    ) -> Dict[str, float]:
        """Extract team composition features using enhanced player profiles."""
        if self.enhanced_players.empty or not team_players:
            return self._get_default_composition_features()

        # Find players in enhanced dataset
        team_player_data = []
        for player_name in team_players:
            player_row = self.enhanced_players[
                self.enhanced_players["Name"].str.lower() == player_name.lower()
            ]
            if not player_row.empty:
                team_player_data.append(player_row.iloc[0])

        if not team_player_data:
            return self._get_default_composition_features()

        # Convert to DataFrame for easier analysis
        team_df = pd.DataFrame(team_player_data)

        # Batting strength features
        batting_features = {
            "avg_career_runs": team_df["Runs"].fillna(0).mean(),
            "avg_batting_avg": team_df["Avg"].fillna(0).mean(),
            "avg_strike_rate": team_df["SR"].fillna(0).mean(),
            "total_centuries": team_df["100s"].fillna(0).sum(),
            "total_fifties": team_df["50s"].fillna(0).sum(),
            "power_hitters": len(
                team_df[team_df["6s"].fillna(0) > 50]
            ),  # Players with 50+ sixes
            "consistent_batters": len(team_df[team_df["Avg"].fillna(0) > 30]),
        }

        # Bowling strength features
        bowling_features = {
            "avg_wickets": team_df["B_Wkts"].fillna(0).mean(),
            "avg_bowling_avg": team_df["B_Avg"]
            .fillna(50)
            .mean(),  # Default 50 for non-bowlers
            "avg_economy": team_df["B_Econ"]
            .fillna(8)
            .mean(),  # Default 8 for non-bowlers
            "total_5w_hauls": team_df["B_5w"].fillna(0).sum(),
            "quality_bowlers": len(
                team_df[team_df["B_Avg"].fillna(50) < 25]
            ),  # Good bowling average
            "economical_bowlers": len(team_df[team_df["B_Econ"].fillna(8) < 7]),
        }

        # Team balance and experience
        balance_features = {
            "avg_experience": team_df["Mat"].fillna(0).mean(),
            "veteran_players": len(team_df[team_df["Mat"].fillna(0) > 100]),
            "young_players": len(team_df[team_df["AGE"].fillna(30) < 25]),
            "allrounders": len(
                team_df[
                    (team_df["Runs"].fillna(0) > 500)
                    & (team_df["B_Wkts"].fillna(0) > 10)
                ]
            ),
            "premium_players": len(
                team_df[team_df["SOLD_PRICE"].str.contains("cr", na=False)]
            ),
            "captaincy_options": team_df["CAPTAINCY EXP"].fillna(0).sum(),
        }

        # Team financial investment
        financial_features = {
            "team_value_estimate": self._calculate_team_value(team_df),
            "value_distribution": team_df["SOLD_PRICE"]
            .apply(self._convert_price_to_float)
            .std(),
        }

        # Combine all features
        all_features = {
            **batting_features,
            **bowling_features,
            **balance_features,
            **financial_features,
        }
        return all_features

    def _convert_price_to_float(self, price_str):
        """Convert auction price to float for calculations."""
        if pd.isna(price_str) or price_str == "Unsold":
            return 0.0
        price_str = str(price_str).lower()
        if "cr" in price_str:
            return float(price_str.replace("cr", "").strip())
        elif "l" in price_str:
            return float(price_str.replace("l", "").strip()) / 100
        return 0.0

    def _calculate_team_value(self, team_df: pd.DataFrame) -> float:
        """Calculate total team value from auction prices."""
        total_value = 0.0
        for price in team_df["SOLD_PRICE"].fillna("0"):
            total_value += self._convert_price_to_float(price)
        return total_value

    def _get_default_composition_features(self) -> Dict[str, float]:
        """Default team composition features when no data available."""
        return {
            "avg_career_runs": 1000.0,
            "avg_batting_avg": 25.0,
            "avg_strike_rate": 120.0,
            "total_centuries": 2.0,
            "total_fifties": 10.0,
            "power_hitters": 3.0,
            "consistent_batters": 4.0,
            "avg_wickets": 50.0,
            "avg_bowling_avg": 30.0,
            "avg_economy": 7.5,
            "total_5w_hauls": 1.0,
            "quality_bowlers": 4.0,
            "economical_bowlers": 3.0,
            "avg_experience": 75.0,
            "veteran_players": 3.0,
            "young_players": 4.0,
            "allrounders": 2.0,
            "premium_players": 6.0,
            "captaincy_options": 2.0,
            "team_value_estimate": 85.0,
            "value_distribution": 5.0,
        }

    def extract_team_form_features(
        self,
        matches_data: pd.DataFrame,
        team: str,
        current_date: str,
        lookback: int = 5,
    ) -> Dict[str, float]:
        """Extract recent team form and momentum."""
        team_matches = (
            matches_data[
                ((matches_data["team1"] == team) | (matches_data["team2"] == team))
                & (matches_data["match_date"] < current_date)
            ]
            .sort_values("match_date", ascending=False)
            .head(lookback)
        )

        if len(team_matches) == 0:
            return self._get_default_team_features()

        # Calculate wins and performance metrics
        wins = len(team_matches[team_matches["winner"] == team])
        win_rate = wins / len(team_matches)

        # Analyze margins of victory/defeat
        margins = []
        for _, match in team_matches.iterrows():
            if match["result"] == "Win" and match["winner"] == team:
                if "runs" in str(match["result"]).lower():
                    margins.append(1.0)  # Won by runs (strong batting)
                else:
                    margins.append(0.8)  # Won by wickets (good chase)
            elif match["winner"] != team:
                margins.append(0.0)  # Loss

        features = {
            "recent_win_rate": win_rate,
            "recent_matches": len(team_matches),
            "form_momentum": np.mean(margins) if margins else 0.5,
            "consistency": 1.0 - np.std(margins) if len(margins) > 1 else 0.5,
        }

        return features

    def extract_playing_xi_features(
        self, match_data: pd.Series, team: str
    ) -> Dict[str, float]:
        """Analyze playing XI composition and balance."""
        if team == match_data["team1"]:
            players_str = match_data["team1_players"]
        else:
            players_str = match_data["team2_players"]

        if pd.isna(players_str):
            return self._get_default_xi_features()

        players = [p.strip() for p in players_str.split(",")]

        # Analyze team balance
        features = {
            "team_size": len(players),
            "team_balance_score": self._calculate_team_balance(players),
            "experience_factor": self._calculate_experience_factor(players),
            "star_player_count": self._count_star_players(players),
        }

        return features

    def extract_head_to_head_team_features(
        self, matches_data: pd.DataFrame, team1: str, team2: str
    ) -> Dict[str, float]:
        """Extract head-to-head team performance."""
        h2h_matches = matches_data[
            ((matches_data["team1"] == team1) & (matches_data["team2"] == team2))
            | ((matches_data["team1"] == team2) & (matches_data["team2"] == team1))
        ]

        if len(h2h_matches) == 0:
            return {"h2h_matches": 0, "h2h_win_rate": 0.5}

        team1_wins = len(h2h_matches[h2h_matches["winner"] == team1])

        return {
            "h2h_matches": len(h2h_matches),
            "h2h_win_rate": team1_wins / len(h2h_matches),
        }

    def extract_venue_advantage_features(
        self, matches_data: pd.DataFrame, team: str, venue: str
    ) -> Dict[str, float]:
        """Calculate team's performance at specific venue."""
        venue_matches = matches_data[
            (matches_data["venue"] == venue)
            & ((matches_data["team1"] == team) | (matches_data["team2"] == team))
        ]

        if len(venue_matches) == 0:
            return {"venue_experience": 0, "venue_win_rate": 0.5}

        wins = len(venue_matches[venue_matches["winner"] == team])

        return {
            "venue_experience": len(venue_matches),
            "venue_win_rate": wins / len(venue_matches),
        }

    def extract_toss_strategy_features(
        self, matches_data: pd.DataFrame, team: str, venue: str
    ) -> Dict[str, float]:
        """Analyze team's toss decisions and success."""
        team_toss_matches = matches_data[
            (matches_data["toss_winner"] == team) & (matches_data["venue"] == venue)
        ]

        if len(team_toss_matches) == 0:
            return {
                "toss_wins": 0,
                "bat_first_preference": 0.5,
                "toss_win_success": 0.5,
            }

        bat_first = len(team_toss_matches[team_toss_matches["toss_decision"] == "bat"])
        wins_after_toss = len(team_toss_matches[team_toss_matches["winner"] == team])

        return {
            "toss_wins": len(team_toss_matches),
            "bat_first_preference": bat_first / len(team_toss_matches),
            "toss_win_success": wins_after_toss / len(team_toss_matches),
        }

    def _calculate_team_balance(self, players: List[str]) -> float:
        """Calculate team balance score (simplified)."""
        return 0.7  # Default balanced team score

    def _calculate_experience_factor(self, players: List[str]) -> float:
        """Calculate team experience factor (simplified)."""
        return 0.6  # Default experience score

    def _count_star_players(self, players: List[str]) -> int:
        """Count star players in team (simplified)."""
        return 3  # Default star player count

    def _get_default_team_features(self) -> Dict[str, float]:
        """Default team features."""
        return {
            "recent_win_rate": 0.5,
            "recent_matches": 0,
            "form_momentum": 0.5,
            "consistency": 0.5,
        }

    def _get_default_xi_features(self) -> Dict[str, float]:
        """Default playing XI features."""
        return {
            "team_size": 11,
            "team_balance_score": 0.7,
            "experience_factor": 0.6,
            "star_player_count": 3,
        }
