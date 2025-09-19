#!/usr/bin/env python3
"""
IPL Match Prediction Tool - Uses trained models for match outcome prediction
"""

import argparse
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import tensorflow as tf

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

try:
    from ipl_predictor.features import FeaturePipeline
    from ipl_predictor.models.ensemble import EnsemblePredictor
    from ipl_predictor.models.neural_network import NeuralNetworkPredictor
    from ipl_predictor.models.xgboost_model import XGBoostPredictor
except ImportError as e:
    print(f"Error importing modules: {e}")
    sys.exit(1)


class IPLMatchPredictor:
    def __init__(self):
        self.models_dir = Path("models")
        self.models = {}

        # Team mappings
        self.teams = {
            "MI": "Mumbai Indians",
            "CSK": "Chennai Super Kings",
            "RCB": "Royal Challengers Bangalore",
            "DC": "Delhi Capitals",
            "KKR": "Kolkata Knight Riders",
            "PBKS": "Punjab Kings",
            "RR": "Rajasthan Royals",
            "SRH": "Sunrisers Hyderabad",
            "GT": "Gujarat Titans",
            "LSG": "Lucknow Super Giants",
        }

        self.feature_pipeline = FeaturePipeline("data")
        self.load_models()

    def normalize_team(self, team):
        return self.teams.get(team, team)

    def load_models(self):
        print("Loading trained models...")

        # Load XGBoost
        xgb_path = self.models_dir / "xgboost_model.pkl"
        if xgb_path.exists():
            try:
                self.models["xgboost"] = XGBoostPredictor()
                self.models["xgboost"].load_model(str(xgb_path))
                print("XGBoost model loaded successfully")
            except Exception as e:
                print(f"Failed to load XGBoost: {e}")

        # Load Neural Network
        nn_path = self.models_dir / "neural_network_model"
        if nn_path.exists():
            try:
                self.models["neural_network"] = NeuralNetworkPredictor()
                self.models["neural_network"].load_model(str(nn_path))
                print("Neural Network model loaded successfully")
            except Exception as e:
                print(f"Failed to load Neural Network: {e}")

        # Load Ensemble
        ensemble_path = self.models_dir / "ensemble_model"
        if ensemble_path.exists():
            try:
                self.models["ensemble"] = EnsemblePredictor()
                self.models["ensemble"].load_ensemble(str(ensemble_path))
                print("Ensemble model loaded successfully")
            except Exception as e:
                print(f"Failed to load Ensemble: {e}")

        if not self.models:
            print("No models found! Please run train.py first.")
            sys.exit(1)

        print(f"Successfully loaded {len(self.models)} models")

    def predict(self, team1, team2, venue, toss_winner=None, toss_decision=None):
        """Predict match outcome using trained models."""
        team1 = self.normalize_team(team1)
        team2 = self.normalize_team(team2)

        print(f"\nPredicting: {team1} vs {team2} at {venue}")
        print("=" * 60)

        try:
            # Create match features using the feature pipeline
            from datetime import datetime

            match_date = datetime.now().strftime("%Y-%m-%d")

            # Ensure team names are not None
            if not team1 or not team2:
                print("Error: Team names cannot be empty")
                return

            features_dict = self.feature_pipeline.extract_match_features(
                team1=str(team1),
                team2=str(team2),
                venue=str(venue),
                match_date=match_date,
                toss_winner=toss_winner,
                toss_decision=toss_decision,
            )

            # Add missing toss features (4 features)
            if toss_winner and toss_decision:
                features_dict["toss_won_by_team1"] = (
                    1.0 if toss_winner == team1 else 0.0
                )
                features_dict["toss_decision_bat"] = (
                    1.0 if toss_decision == "bat" else 0.0
                )
                features_dict["team1_won_toss_and_bat"] = (
                    1.0 if (toss_winner == team1 and toss_decision == "bat") else 0.0
                )
                features_dict["team1_won_toss_and_field"] = (
                    1.0 if (toss_winner == team1 and toss_decision == "field") else 0.0
                )
            else:
                # Default values when toss info not provided
                features_dict["toss_won_by_team1"] = 0.5
                features_dict["toss_decision_bat"] = 0.5
                features_dict["team1_won_toss_and_bat"] = 0.0
                features_dict["team1_won_toss_and_field"] = 0.0

            # Load expected feature order from training data
            expected_features = [
                "team1_recent_win_rate",
                "team1_recent_matches",
                "team1_form_momentum",
                "team1_consistency",
                "team2_recent_win_rate",
                "team2_recent_matches",
                "team2_form_momentum",
                "team2_consistency",
                "h2h_matches",
                "h2h_win_rate",
                "total_matches_played",
                "bat_first_advantage",
                "field_first_advantage",
                "toss_importance",
                "bat_first_preference",
                "team1_venue_experience",
                "team1_venue_win_rate",
                "team2_venue_experience",
                "team2_venue_win_rate",
                "month",
                "day_of_year",
                "is_weekend",
                "season_phase",
                "days_since_season_start",
                "season_progress",
                "team1_league_position",
                "team2_league_position",
                "points_gap",
                "playoff_pressure",
                "elimination_context",
                "team1_days_rest",
                "team2_days_rest",
                "rest_advantage",
                "both_teams_rested",
                "team1_comp_avg_career_runs",
                "team1_comp_avg_batting_avg",
                "team1_comp_avg_strike_rate",
                "team1_comp_total_centuries",
                "team1_comp_total_fifties",
                "team1_comp_power_hitters",
                "team1_comp_consistent_batters",
                "team1_comp_avg_wickets",
                "team1_comp_avg_bowling_avg",
                "team1_comp_avg_economy",
                "team1_comp_total_5w_hauls",
                "team1_comp_quality_bowlers",
                "team1_comp_economical_bowlers",
                "team1_comp_avg_experience",
                "team1_comp_veteran_players",
                "team1_comp_young_players",
                "team1_comp_allrounders",
                "team1_comp_premium_players",
                "team1_comp_captaincy_options",
                "team1_comp_team_value_estimate",
                "team1_comp_value_distribution",
                "team2_comp_avg_career_runs",
                "team2_comp_avg_batting_avg",
                "team2_comp_avg_strike_rate",
                "team2_comp_total_centuries",
                "team2_comp_total_fifties",
                "team2_comp_power_hitters",
                "team2_comp_consistent_batters",
                "team2_comp_avg_wickets",
                "team2_comp_avg_bowling_avg",
                "team2_comp_avg_economy",
                "team2_comp_total_5w_hauls",
                "team2_comp_quality_bowlers",
                "team2_comp_economical_bowlers",
                "team2_comp_avg_experience",
                "team2_comp_veteran_players",
                "team2_comp_young_players",
                "team2_comp_allrounders",
                "team2_comp_premium_players",
                "team2_comp_captaincy_options",
                "team2_comp_team_value_estimate",
                "team2_comp_value_distribution",
                "toss_won_by_team1",
                "toss_decision_bat",
                "team1_won_toss_and_bat",
                "team1_won_toss_and_field",
                "team1_encoded",
                "team2_encoded",
                "venue_encoded",
                "enhanced_bat_first_rate",
                "enhanced_toss_advantage",
                "venue_json_matches",
            ]

            # Reorder features_dict to match training order and fill missing values
            ordered_features = []
            for feature in expected_features:
                ordered_features.append(features_dict.get(feature, 0.0))

            # Convert to DataFrame with correct column order
            features_df = pd.DataFrame([ordered_features], columns=expected_features)

            print(f"Generated {len(expected_features)} features for prediction")

            predictions = {}

            # XGBoost prediction
            if "xgboost" in self.models:
                try:
                    prob = self.models["xgboost"].predict_proba(features_df)
                    predictions["XGBoost"] = prob
                except Exception as e:
                    print(f"XGBoost prediction failed: {e}")

            # Neural Network prediction
            if "neural_network" in self.models:
                try:
                    prob_array = self.models["neural_network"].predict_proba(
                        features_df
                    )
                    predictions["Neural Network"] = prob_array
                except Exception as e:
                    print(f"Neural Network prediction failed: {e}")

            # Ensemble prediction
            if "ensemble" in self.models:
                try:
                    prob_array = self.models["ensemble"].predict_proba(features_df)
                    predictions["Ensemble"] = prob_array
                except Exception as e:
                    print(f"Ensemble prediction failed: {e}")

            # Display results
            if predictions:
                print("PREDICTION RESULTS:")
                print("-" * 40)

                for model_name, prob in predictions.items():
                    # Handle different probability formats
                    if isinstance(prob, np.ndarray):
                        if prob.shape == (
                            1,
                            2,
                        ):  # XGBoost format: [[prob_class0, prob_class1]]
                            team1_prob = float(
                                prob[0][1]
                            )  # Take probability for class 1 (team1 wins)
                        elif (
                            len(prob.shape) == 2 and prob.shape[1] == 1
                        ):  # Neural network: [[prob]]
                            team1_prob = float(prob[0][0])
                        elif len(prob.shape) == 1:  # 1D array
                            team1_prob = float(prob[0])
                        else:
                            team1_prob = float(prob.flat[0])  # Flatten and take first
                    else:
                        team1_prob = float(prob)

                    winner = team1 if team1_prob > 0.5 else team2
                    confidence = team1_prob if team1_prob > 0.5 else (1 - team1_prob)

                    print(f"{model_name:15} | {winner:20} | {confidence*100:.1f}%")

                # Overall prediction
                if len(predictions) > 1:
                    # Properly extract probabilities
                    all_probs = []
                    for prob in predictions.values():
                        if isinstance(prob, np.ndarray):
                            if prob.shape == (1, 2):  # XGBoost format
                                all_probs.append(float(prob[0][1]))
                            elif (
                                len(prob.shape) == 2 and prob.shape[1] == 1
                            ):  # Neural network
                                all_probs.append(float(prob[0][0]))
                            elif len(prob.shape) == 1:
                                all_probs.append(float(prob[0]))
                            else:
                                all_probs.append(float(prob.flat[0]))
                        else:
                            all_probs.append(float(prob))

                    avg_prob = float(np.mean(all_probs))
                    overall_winner = team1 if avg_prob > 0.5 else team2
                    overall_confidence = avg_prob if avg_prob > 0.5 else (1 - avg_prob)

                    print("-" * 40)
                    print(
                        f"{'CONSENSUS':15} | {overall_winner:20} | {overall_confidence*100:.1f}%"
                    )
            else:
                print("No predictions could be generated.")

        except Exception as e:
            print(f"Prediction failed: {e}")
            import traceback

            traceback.print_exc()


def main():
    parser = argparse.ArgumentParser(description="Predict IPL match outcomes")
    parser.add_argument("team1", help='First team (e.g., "Mumbai Indians" or "MI")')
    parser.add_argument(
        "team2", help='Second team (e.g., "Chennai Super Kings" or "CSK")'
    )
    parser.add_argument("venue", help='Venue name (e.g., "Wankhede Stadium")')
    parser.add_argument("--toss-winner", help="Team that won the toss (optional)")
    parser.add_argument(
        "--toss-decision", choices=["bat", "field"], help="Toss decision (optional)"
    )

    args = parser.parse_args()

    predictor = IPLMatchPredictor()
    predictor.predict(
        args.team1, args.team2, args.venue, args.toss_winner, args.toss_decision
    )


if __name__ == "__main__":
    main()
