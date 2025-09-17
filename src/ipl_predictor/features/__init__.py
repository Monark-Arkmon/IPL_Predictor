"""
Feature engineering pipeline coordinator.
Orchestrates all feature extraction components.
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
import logging
from pathlib import Path

from ..data.loader import IPLDataLoader
from .player_features import PlayerFeatureEngine
from .team_features import TeamFeatureEngine
from .context_features import ContextFeatureEngine

logger = logging.getLogger(__name__)

class FeaturePipeline:
    """Feature engineering pipeline for IPL prediction."""
    
    def __init__(self, data_path: str = "data"):
        self.data_loader = IPLDataLoader(data_path)
        self.player_engine = PlayerFeatureEngine()
        self.team_engine = TeamFeatureEngine()
        self.context_engine = ContextFeatureEngine()
        
    def extract_match_features(self, team1: str, team2: str, venue: str,
                              match_date: str, toss_winner: Optional[str] = None,
                              toss_decision: Optional[str] = None) -> Dict[str, float]:
        """Extract features for a match prediction."""
        logger.info(f"Extracting features for {team1} vs {team2} at {venue}")
        
        # Load required data
        matches_data = self.data_loader.load_matches()
        ball_data = self.data_loader.load_ball_by_ball()
        
        features = {}
        
        # Team-level features
        team1_features = self.team_engine.extract_team_form_features(
            matches_data, team1, match_date
        )
        team2_features = self.team_engine.extract_team_form_features(
            matches_data, team2, match_date
        )
        
        # Add team prefixes
        for key, value in team1_features.items():
            features[f'team1_{key}'] = value
        for key, value in team2_features.items():
            features[f'team2_{key}'] = value
        
        # Head-to-head features
        h2h_features = self.team_engine.extract_head_to_head_team_features(
            matches_data, team1, team2
        )
        features.update(h2h_features)
        
        # Venue features
        venue_features = self.context_engine.extract_venue_features(matches_data, venue)
        features.update(venue_features)
        
        # Team venue advantage
        team1_venue = self.team_engine.extract_venue_advantage_features(
            matches_data, team1, venue
        )
        team2_venue = self.team_engine.extract_venue_advantage_features(
            matches_data, team2, venue
        )
        
        for key, value in team1_venue.items():
            features[f'team1_{key}'] = value
        for key, value in team2_venue.items():
            features[f'team2_{key}'] = value
        
        # Match context features
        timing_features = self.context_engine.extract_match_timing_features(match_date)
        features.update(timing_features)
        
        tournament_features = self.context_engine.extract_tournament_context_features(
            matches_data, match_date, team1, team2
        )
        features.update(tournament_features)
        
        # Rest and fatigue features
        rest_features = self.context_engine.extract_rest_and_fatigue_features(
            matches_data, match_date, team1, team2
        )
        features.update(rest_features)
        
        # Enhanced team composition features (using enhanced player data)
        try:
            # Get sample player lists for teams (in real scenario, would have actual rosters)
            team1_composition = self.team_engine.extract_team_composition_features([team1])  # Simplified
            team2_composition = self.team_engine.extract_team_composition_features([team2])  # Simplified
            
            # Add team prefixes to composition features
            for key, value in team1_composition.items():
                features[f'team1_comp_{key}'] = value
            for key, value in team2_composition.items():
                features[f'team2_comp_{key}'] = value
                
        except Exception as e:
            logger.warning(f"Could not extract team composition features: {e}")
        
        # Enhanced player features (get top players for each team if available)
        try:
            # This is a simplified approach - in production, you'd have actual team rosters
            from ..data.loader import IPLDataLoader
            data_loader = IPLDataLoader(str(self.data_loader.data_path))
            ball_data = data_loader.load_ball_by_ball()
            
            # Get top performers for each team (as proxy for key players)
            team1_players = self._get_key_players_for_team(ball_data, team1)
            team2_players = self._get_key_players_for_team(ball_data, team2)
            
            # Extract enhanced player features for top performers
            team1_player_features = self._extract_enhanced_player_features(ball_data, team1_players[:3])  # Top 3
            team2_player_features = self._extract_enhanced_player_features(ball_data, team2_players[:3])  # Top 3
            
            features.update(team1_player_features)
            features.update(team2_player_features)
            
        except Exception as e:
            logger.warning(f"Could not extract enhanced player features: {e}")
        
        # Toss features (if available)
        if toss_winner and toss_decision:
            features.update(self._extract_toss_features(toss_winner, toss_decision, team1, team2))
        
        logger.info(f"Extracted {len(features)} features")
        return features
    
    def extract_training_dataset(self, min_date: str = "2015-01-01") -> Tuple[pd.DataFrame, pd.Series]:
        """Extract features for all matches to create training dataset."""
        logger.info("Creating training dataset from historical matches")
        
        matches_data = self.data_loader.load_matches()
        matches_data = matches_data[matches_data['match_date'] >= min_date].copy()
        
        feature_rows = []
        targets = []
        
        for idx, match in matches_data.iterrows():
            try:
                # Extract features
                features = self.extract_match_features(
                    match['team1'], match['team2'], match['venue'],
                    match['match_date'].strftime('%Y-%m-%d'),
                    match['toss_winner'], match['toss_decision']
                )
                
                # Add basic match info
                features['team1_encoded'] = self._encode_team(match['team1'])
                features['team2_encoded'] = self._encode_team(match['team2'])
                features['venue_encoded'] = self._encode_venue(match['venue'])
                
                feature_rows.append(features)
                
                # Target: 1 if team1 wins, 0 if team2 wins
                target = 1 if match['winner'] == match['team1'] else 0
                targets.append(target)
                
            except Exception as e:
                logger.warning(f"Error processing match {match['match_number']}: {e}")
                continue
        
        # Convert to DataFrame
        feature_df = pd.DataFrame(feature_rows)
        target_series = pd.Series(targets)
        
        # Fill missing values
        feature_df = feature_df.fillna(feature_df.median())
        
        logger.info(f"Created training dataset with {len(feature_df)} samples and {len(feature_df.columns)} features")
        return feature_df, target_series
    
    def save_feature_dataset(self, feature_df: pd.DataFrame, target_series: pd.Series,
                           filepath: str = "data/processed/training_features.csv"):
        """Save processed features to file."""
        output_path = Path(filepath)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Combine features and target
        dataset = feature_df.copy()
        dataset['target'] = target_series
        
        dataset.to_csv(output_path, index=False)
        logger.info(f"Saved feature dataset to {output_path}")
    
    def _extract_toss_features(self, toss_winner: str, toss_decision: str,
                              team1: str, team2: str) -> Dict[str, float]:
        """Extract toss-related features."""
        return {
            'toss_won_by_team1': 1.0 if toss_winner == team1 else 0.0,
            'toss_decision_bat': 1.0 if toss_decision == 'bat' else 0.0,
            'team1_won_toss_and_bat': 1.0 if (toss_winner == team1 and toss_decision == 'bat') else 0.0,
            'team1_won_toss_and_field': 1.0 if (toss_winner == team1 and toss_decision == 'field') else 0.0
        }
    
    def _encode_team(self, team: str) -> int:
        """Simple team encoding (would use proper label encoding in production)."""
        teams = ['Chennai Super Kings', 'Mumbai Indians', 'Royal Challengers Bangalore',
                'Kolkata Knight Riders', 'Delhi Capitals', 'Punjab Kings',
                'Rajasthan Royals', 'Sunrisers Hyderabad']
        try:
            return teams.index(team)
        except ValueError:
            return len(teams)  # Unknown team
    
    def _encode_venue(self, venue: str) -> int:
        """Simple venue encoding."""
        # This would be more sophisticated in production
        return hash(venue) % 100
        
    def _get_key_players_for_team(self, ball_data: pd.DataFrame, team: str) -> List[str]:
        """Get key players for a team based on performance data."""
        try:
            # Get players who have played for this team (simplified approach)
            team_players = ball_data[ball_data['BattingTeam'] == team]['Batter'].value_counts()
            team_bowlers = ball_data[ball_data['BowlingTeam'] != team]['Bowler'].value_counts()
            
            # Combine and get top performers
            all_players = list(team_players.index[:5]) + list(team_bowlers.index[:3])
            return list(set(all_players))  # Remove duplicates
        except:
            return [team]  # Fallback to team name
            
    def _extract_enhanced_player_features(self, ball_data: pd.DataFrame, players: List[str]) -> Dict[str, float]:
        """Extract enhanced features for key players."""
        if not players:
            return {}
            
        try:
            # Get aggregated enhanced features for these players
            aggregated_features = {}
            player_count = 0
            
            for player in players:
                player_features = self.player_engine.get_enhanced_player_features(player)
                if player_features:
                    player_count += 1
                    for key, value in player_features.items():
                        if key not in aggregated_features:
                            aggregated_features[key] = []
                        aggregated_features[key].append(value)
                        
            # Calculate averages for aggregated features
            final_features = {}
            for key, values in aggregated_features.items():
                if values:
                    final_features[f'avg_{key}'] = np.mean(values)
                    
            return final_features
        except Exception as e:
            logger.warning(f"Error extracting enhanced player features: {e}")
            return {}