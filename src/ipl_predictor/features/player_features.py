"""
Player performance feature engineering for IPL prediction.
Generates player-specific metrics and performance indicators.
Enhanced with career statistics and performance tiers.
"""

import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class PlayerFeatureEngine:
    """Generate player performance features with enhanced career data."""

    def __init__(self):
        self.feature_cache = {}
        self.enhanced_players = self._load_enhanced_players()

    def _load_enhanced_players(self) -> pd.DataFrame:
        """Load enhanced player data if available."""
        enhanced_path = Path("data/enhanced/enhanced_players.csv")
        if enhanced_path.exists():
            logger.info("Loading enhanced player data")
            return pd.read_csv(enhanced_path)
        else:
            logger.warning("Enhanced player data not found, using basic features only")
            return pd.DataFrame()

    def get_enhanced_player_features(self, player_name: str) -> Dict[str, float]:
        """Get enhanced career features for a player."""
        if self.enhanced_players.empty:
            return {}

        # Find player in enhanced dataset
        player_row = self.enhanced_players[
            self.enhanced_players["Name"].str.lower() == player_name.lower()
        ]

        if player_row.empty:
            return {}

        player = player_row.iloc[0]

        # Extract enhanced features
        features = {
            # Career batting statistics
            "career_runs": float(player.get("Runs", 0)),
            "career_avg": float(player.get("Avg", 0)),
            "career_strike_rate": float(player.get("SR", 0)),
            "career_highest_score": float(player.get("HS", 0)),
            "career_centuries": float(player.get("100s", 0)),
            "career_fifties": float(player.get("50s", 0)),
            "career_fours": float(player.get("4s", 0)),
            "career_sixes": float(player.get("6s", 0)),
            # Career bowling statistics
            "career_wickets": float(player.get("B_Wkts", 0)),
            "career_bowl_avg": float(player.get("B_Avg", 0)),
            "career_bowl_sr": float(player.get("B_SR", 0)),
            "career_economy": float(player.get("B_Econ", 0)),
            "career_5w": float(player.get("B_5w", 0)),
            "career_4w": float(player.get("B_4w", 0)),
            # General career data
            "total_matches": float(player.get("Mat", 0)),
            "player_age": float(player.get("AGE", 25)),
            "auction_price": self._convert_auction_price(player.get("SOLD_PRICE", "0")),
            "captaincy_experience": float(player.get("CAPTAINCY EXP", 0)),
            # Enhanced derived features
            "allrounder_score": float(player.get("allrounder_score", 0)),
            "batting_tier": self._encode_tier(player.get("batting_tier", "Average")),
            "bowling_tier": self._encode_tier(player.get("bowling_tier", "Average")),
            "experience_level": self._encode_experience(
                player.get("experience_level", "Developing")
            ),
            # Performance indicators
            "is_premium_player": (
                1.0
                if self._convert_auction_price(player.get("SOLD_PRICE", "0")) > 5.0
                else 0.0
            ),
            "is_experienced": 1.0 if float(player.get("Mat", 0)) > 50 else 0.0,
            "is_allrounder": (
                1.0
                if (
                    float(player.get("Runs", 0)) > 500
                    and float(player.get("B_Wkts", 0)) > 10
                )
                else 0.0
            ),
        }

        return features

    def _encode_tier(self, tier):
        """Encode tier categories to numerical values."""
        tier_map = {"Poor": 1, "Average": 2, "Good": 3, "Excellent": 4}
        return float(tier_map.get(str(tier), 2))  # Default to Average

    def _encode_experience(self, exp_level):
        """Encode experience level to numerical values."""
        exp_map = {"Rookie": 1, "Developing": 2, "Experienced": 3, "Veteran": 4}
        return float(exp_map.get(str(exp_level), 2))  # Default to Developing

    def _convert_auction_price(self, price_str):
        """Convert auction price string to float (in crores)."""
        if pd.isna(price_str) or price_str == "Unsold" or price_str == "":
            return 0.0

        price_str = str(price_str).lower().strip()

        if "cr" in price_str:
            # Extract number from "12cr" format
            return float(price_str.replace("cr", "").strip())
        elif "l" in price_str or "lakh" in price_str:
            # Convert lakhs to crores
            return float(price_str.replace("l", "").replace("lakh", "").strip()) / 100
        else:
            try:
                return float(price_str)
            except:
                return 0.0

    def extract_batting_features(
        self, ball_data: pd.DataFrame, player_name: str, lookback_matches: int = 10
    ) -> Dict[str, float]:
        """Extract batting performance features for a player with enhanced career data."""
        player_balls = ball_data[ball_data["Batter"] == player_name].copy()

        # Start with enhanced career features
        features = self.get_enhanced_player_features(player_name)

        if len(player_balls) == 0:
            features.update(self._get_default_batting_features())
            return features

        # Recent form calculation
        recent_balls = player_balls.tail(lookback_matches * 120)  # ~120 balls per match

        # Current season/recent features
        current_features = {
            "current_batting_avg": player_balls["BatsmanRun"].mean(),
            "current_strike_rate": (
                player_balls["BatsmanRun"].sum() / len(player_balls)
            )
            * 100,
            "recent_avg": recent_balls["BatsmanRun"].mean(),
            "recent_strike_rate": (recent_balls["BatsmanRun"].sum() / len(recent_balls))
            * 100,
            "boundary_rate": len(player_balls[player_balls["BatsmanRun"].isin([4, 6])])
            / len(player_balls),
            "dot_ball_rate": len(player_balls[player_balls["BatsmanRun"] == 0])
            / len(player_balls),
            "consistency": player_balls["BatsmanRun"].std(),
            "total_balls_faced": len(player_balls),
        }

        features.update(current_features)

        # Enhanced form analysis using career vs current performance
        if features.get("career_avg", 0) > 0:
            features["form_vs_career"] = (
                current_features["current_batting_avg"] / features["career_avg"]
            )
        else:
            features["form_vs_career"] = 1.0

        return features

    def extract_bowling_features(
        self, ball_data: pd.DataFrame, player_name: str, lookback_matches: int = 10
    ) -> Dict[str, float]:
        """Extract bowling performance features for a player with enhanced career data."""
        player_balls = ball_data[ball_data["Bowler"] == player_name].copy()

        # Start with enhanced career features
        features = self.get_enhanced_player_features(player_name)

        if len(player_balls) == 0:
            features.update(self._get_default_bowling_features())
            return features

        recent_balls = player_balls.tail(
            lookback_matches * 24
        )  # ~24 balls per match for bowler

        # Calculate economy and wicket rate
        total_runs = player_balls["BatsmanRun"].sum() + player_balls["ExtrasRun"].sum()
        total_overs = len(player_balls) / 6
        wickets = len(player_balls[player_balls["IsWicketDelivery"] == 1])

        recent_runs = recent_balls["BatsmanRun"].sum() + recent_balls["ExtrasRun"].sum()
        recent_overs = len(recent_balls) / 6
        recent_wickets = len(recent_balls[recent_balls["IsWicketDelivery"] == 1])

        current_features = {
            "current_economy_rate": (
                total_runs / total_overs if total_overs > 0 else 8.0
            ),
            "current_wicket_rate": wickets / total_overs if total_overs > 0 else 0.0,
            "recent_economy": recent_runs / recent_overs if recent_overs > 0 else 8.0,
            "recent_wicket_rate": (
                recent_wickets / recent_overs if recent_overs > 0 else 0.0
            ),
            "dot_ball_percentage": len(player_balls[player_balls["TotalRun"] == 0])
            / len(player_balls),
            "boundary_conceded_rate": len(
                player_balls[player_balls["BatsmanRun"].isin([4, 6])]
            )
            / len(player_balls),
            "total_balls_bowled": len(player_balls),
        }

        features.update(current_features)

        # Enhanced form analysis using career vs current performance
        if features.get("career_economy", 0) > 0:
            features["economy_vs_career"] = (
                current_features["current_economy_rate"] / features["career_economy"]
            )
        else:
            features["economy_vs_career"] = 1.0

        return features

    def extract_head_to_head_features(
        self, ball_data: pd.DataFrame, batter: str, bowler: str
    ) -> Dict[str, float]:
        """Extract head-to-head performance between batter and bowler."""
        h2h_balls = ball_data[
            (ball_data["Batter"] == batter) & (ball_data["Bowler"] == bowler)
        ]

        if len(h2h_balls) == 0:
            return {
                "h2h_balls": 0,
                "h2h_avg": 0.0,
                "h2h_strike_rate": 0.0,
                "h2h_dismissals": 0,
            }

        dismissals = len(h2h_balls[h2h_balls["IsWicketDelivery"] == 1])
        total_runs = h2h_balls["BatsmanRun"].sum()

        return {
            "h2h_balls": len(h2h_balls),
            "h2h_avg": total_runs / len(h2h_balls),
            "h2h_strike_rate": (total_runs / len(h2h_balls)) * 100,
            "h2h_dismissals": dismissals,
        }

    def extract_venue_features(
        self,
        ball_data: pd.DataFrame,
        matches_data: pd.DataFrame,
        player_name: str,
        venue: str,
    ) -> Dict[str, float]:
        """Extract player performance at specific venue."""
        venue_matches = matches_data[matches_data["venue"] == venue][
            "match_number"
        ].values
        venue_balls = ball_data[
            (ball_data["ID"].isin(venue_matches))
            & (
                (ball_data["Batter"] == player_name)
                | (ball_data["Bowler"] == player_name)
            )
        ]

        if len(venue_balls) == 0:
            return {"venue_experience": 0, "venue_performance": 0.0}

        # Calculate venue-specific performance
        if player_name in venue_balls["Batter"].values:
            batting_balls = venue_balls[venue_balls["Batter"] == player_name]
            venue_perf = batting_balls["BatsmanRun"].mean()
        else:
            bowling_balls = venue_balls[venue_balls["Bowler"] == player_name]
            runs_conceded = (
                bowling_balls["BatsmanRun"].sum() + bowling_balls["ExtrasRun"].sum()
            )
            venue_perf = (
                runs_conceded / (len(bowling_balls) / 6)
                if len(bowling_balls) > 0
                else 8.0
            )

        return {"venue_experience": len(venue_balls), "venue_performance": venue_perf}

    def _get_default_batting_features(self) -> Dict[str, float]:
        """Default features for players with no batting data."""
        return {
            "batting_avg": 0.0,
            "strike_rate": 100.0,
            "recent_avg": 0.0,
            "recent_strike_rate": 100.0,
            "boundary_rate": 0.1,
            "dot_ball_rate": 0.5,
            "consistency": 2.0,
            "total_balls_faced": 0,
        }

    def _get_default_bowling_features(self) -> Dict[str, float]:
        """Default features for players with no bowling data."""
        return {
            "economy_rate": 8.0,
            "wicket_rate": 0.0,
            "recent_economy": 8.0,
            "recent_wicket_rate": 0.0,
            "dot_ball_percentage": 0.3,
            "boundary_conceded_rate": 0.2,
            "total_balls_bowled": 0,
        }
