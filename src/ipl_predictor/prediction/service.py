"""
Prediction service for IPL match outcomes.
Provides match prediction with confidence scoring and explanations.
"""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import pandas as pd

from ..features import FeaturePipeline
from ..models.ensemble import EnsemblePredictor
from ..models.xgboost_model import XGBoostPredictor

logger = logging.getLogger(__name__)


class IPLPredictor:
    """Complete IPL match prediction service."""

    def __init__(self, model_path: str = "models", data_path: str = "data"):
        self.model_path = Path(model_path)
        self.data_path = data_path
        self.feature_pipeline = FeaturePipeline(data_path)

        self.models = {}
        self.is_loaded = False

    def load_models(self):
        """Load all available trained models."""
        logger.info("Loading trained models")

        # Load XGBoost model
        xgb_path = self.model_path / "xgboost_model.pkl"
        if xgb_path.exists():
            self.models["xgboost"] = XGBoostPredictor()
            self.models["xgboost"].load_model(str(xgb_path))
            logger.info("XGBoost model loaded")

        # Load Ensemble model
        ensemble_path = self.model_path / "ensemble_model"
        if Path(f"{ensemble_path}_ensemble.pkl").exists():
            self.models["ensemble"] = EnsemblePredictor()
            self.models["ensemble"].load_ensemble(str(ensemble_path))
            logger.info("Ensemble model loaded")

        if not self.models:
            raise ValueError("No trained models found. Run training first.")

        self.is_loaded = True
        logger.info(f"Loaded {len(self.models)} models")

    def predict_match(
        self,
        team1: str,
        team2: str,
        venue: str,
        toss_winner: Optional[str] = None,
        toss_decision: Optional[str] = None,
        model_name: str = "ensemble",
    ) -> Dict:
        """Predict match outcome with detailed analysis."""
        if not self.is_loaded:
            self.load_models()

        if model_name not in self.models:
            available_models = list(self.models.keys())
            raise ValueError(
                f"Model {model_name} not available. Available: {available_models}"
            )

        logger.info(f"Predicting {team1} vs {team2} at {venue}")

        # Use current date for prediction
        match_date = datetime.now().strftime("%Y-%m-%d")

        # Extract features
        features = self.feature_pipeline.extract_match_features(
            team1, team2, venue, match_date, toss_winner, toss_decision
        )

        # Convert to DataFrame
        feature_df = pd.DataFrame([features])

        # Make prediction
        model = self.models[model_name]
        prediction = model.predict(feature_df)[0]
        probabilities = model.predict_proba(feature_df)[0]

        # Calculate confidence
        confidence = max(probabilities) * 100

        # Get prediction explanation
        explanation = self._get_prediction_explanation(
            model, feature_df, features, model_name
        )

        # Prepare result
        result = {
            "match_info": {
                "team1": team1,
                "team2": team2,
                "venue": venue,
                "prediction_date": match_date,
            },
            "prediction": {
                "winner": team1 if prediction == 1 else team2,
                "probability_team1": float(probabilities[1]),
                "probability_team2": float(probabilities[0]),
                "confidence": float(confidence),
                "model_used": model_name,
            },
            "analysis": explanation,
            "toss_info": (
                {"toss_winner": toss_winner, "toss_decision": toss_decision}
                if toss_winner
                else None
            ),
        }

        logger.info(
            f"Prediction: {result['prediction']['winner']} (confidence: {confidence:.1f}%)"
        )
        return result

    def predict_multiple_scenarios(self, team1: str, team2: str, venue: str) -> Dict:
        """Predict match under different toss scenarios."""
        scenarios = {}

        # Scenario 1: Team1 wins toss and bats
        scenarios["team1_bat"] = self.predict_match(team1, team2, venue, team1, "bat")

        # Scenario 2: Team1 wins toss and fields
        scenarios["team1_field"] = self.predict_match(
            team1, team2, venue, team1, "field"
        )

        # Scenario 3: Team2 wins toss and bats
        scenarios["team2_bat"] = self.predict_match(team1, team2, venue, team2, "bat")

        # Scenario 4: Team2 wins toss and fields
        scenarios["team2_field"] = self.predict_match(
            team1, team2, venue, team2, "field"
        )

        # Calculate overall prediction (average across scenarios)
        overall_prob_team1 = np.mean(
            [
                scenario["prediction"]["probability_team1"]
                for scenario in scenarios.values()
            ]
        )

        overall_result = {
            "match_info": {"team1": team1, "team2": team2, "venue": venue},
            "overall_prediction": {
                "probability_team1": float(overall_prob_team1),
                "probability_team2": float(1 - overall_prob_team1),
                "likely_winner": team1 if overall_prob_team1 > 0.5 else team2,
                "confidence": float(abs(overall_prob_team1 - 0.5) * 200),
            },
            "toss_scenarios": scenarios,
            "toss_impact": self._analyze_toss_impact(scenarios),
        }

        return overall_result

    def get_team_strength_analysis(
        self, team: str, venue: Optional[str] = None
    ) -> Dict:
        """Analyze team strength and recent performance."""
        matches_data = self.feature_pipeline.data_loader.load_matches()

        # Get recent matches
        recent_matches = self.feature_pipeline.data_loader.get_recent_matches(team, 10)

        if len(recent_matches) == 0:
            return {"error": f"No recent match data found for {team}"}

        # Calculate team statistics
        wins = len(recent_matches[recent_matches["winner"] == team])
        win_rate = wins / len(recent_matches)

        # Venue-specific analysis
        venue_stats = {}
        if venue:
            venue_matches = matches_data[
                (matches_data["venue"] == venue)
                & ((matches_data["team1"] == team) | (matches_data["team2"] == team))
            ]
            venue_wins = len(venue_matches[venue_matches["winner"] == team])
            venue_stats = {
                "matches_at_venue": len(venue_matches),
                "wins_at_venue": venue_wins,
                "venue_win_rate": (
                    venue_wins / len(venue_matches) if len(venue_matches) > 0 else 0
                ),
            }

        analysis = {
            "team": team,
            "recent_form": {
                "matches_played": len(recent_matches),
                "wins": wins,
                "losses": len(recent_matches) - wins,
                "win_rate": win_rate,
                "form_rating": (
                    "Excellent"
                    if win_rate >= 0.7
                    else "Good" if win_rate >= 0.5 else "Poor"
                ),
            },
            "venue_performance": venue_stats,
            "recent_matches": recent_matches[
                ["match_date", "team1", "team2", "winner", "venue"]
            ].to_dict("records"),
        }

        return analysis

    def _get_prediction_explanation(
        self, model, feature_df: pd.DataFrame, features: Dict, model_name: str
    ) -> Dict:
        """Generate explanation for the prediction."""
        explanation = {
            "key_factors": [],
            "feature_contributions": {},
            "model_insights": {},
        }

        # Get feature importance if available
        if hasattr(model, "get_feature_importance"):
            try:
                importance = model.get_feature_importance(10)
                explanation["feature_contributions"] = importance.to_dict()

                # Identify key factors
                top_features = importance.head(5)
                for feature, score in top_features.items():
                    if feature in features:
                        explanation["key_factors"].append(
                            {
                                "factor": self._humanize_feature_name(feature),
                                "importance": float(score),
                                "value": features[feature],
                            }
                        )
            except Exception as e:
                logger.warning(f"Could not get feature importance: {e}")

        # Add model-specific insights
        explanation["model_insights"] = {
            "model_type": model_name,
            "total_features_used": len(feature_df.columns),
            "prediction_basis": "Advanced machine learning analysis of team performance, player statistics, and match context",
        }

        return explanation

    def _analyze_toss_impact(self, scenarios: Dict) -> Dict:
        """Analyze the impact of toss on match outcome."""
        team1_advantages = []

        for scenario_name, scenario in scenarios.items():
            team1_prob = scenario["prediction"]["probability_team1"]
            team1_advantages.append(team1_prob)

        max_advantage = max(team1_advantages)
        min_advantage = min(team1_advantages)
        toss_impact = max_advantage - min_advantage

        return {
            "toss_advantage_range": float(toss_impact),
            "impact_level": (
                "High"
                if toss_impact > 0.2
                else "Medium" if toss_impact > 0.1 else "Low"
            ),
            "recommendation": self._get_toss_recommendation(scenarios),
        }

    def _get_toss_recommendation(self, scenarios: Dict) -> str:
        """Get toss decision recommendation."""
        team1_bat_prob = scenarios["team1_bat"]["prediction"]["probability_team1"]
        team1_field_prob = scenarios["team1_field"]["prediction"]["probability_team1"]

        if team1_bat_prob > team1_field_prob:
            return "Team1 should choose to bat if they win the toss"
        else:
            return "Team1 should choose to field if they win the toss"

    def _humanize_feature_name(self, feature: str) -> str:
        """Convert feature names to human-readable format."""
        feature_map = {
            "team1_recent_win_rate": "Team 1 Recent Form",
            "team2_recent_win_rate": "Team 2 Recent Form",
            "h2h_win_rate": "Head-to-Head Record",
            "venue_experience": "Venue Experience",
            "toss_importance": "Toss Impact at Venue",
            "form_momentum": "Current Momentum",
            "team1_league_position": "Team 1 League Position",
            "team2_league_position": "Team 2 League Position",
        }
        return feature_map.get(feature, feature.replace("_", " ").title())

    def save_prediction(self, prediction_result: Dict, filepath: str):
        """Save prediction result to file."""
        with open(filepath, "w") as f:
            json.dump(prediction_result, f, indent=2, default=str)
        logger.info(f"Prediction saved to {filepath}")
