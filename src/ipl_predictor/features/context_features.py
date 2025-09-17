"""
Match context and situational feature engineering.
Handles venue, conditions, and match situation features.
Enhanced with JSON match context data.
"""
import pandas as pd
import numpy as np
from typing import Dict
import logging
from datetime import datetime
from pathlib import Path

logger = logging.getLogger(__name__)

class ContextFeatureEngine:
    """Generate match context and situational features with enhanced data."""
    
    def __init__(self):
        self.venue_stats = {}
        self.match_context = self._load_match_context()
        
    def _load_match_context(self) -> pd.DataFrame:
        """Load enhanced match context data if available."""
        context_path = Path("data/enhanced/match_context.csv")
        if context_path.exists():
            logger.info("Loading enhanced match context data")
            return pd.read_csv(context_path)
        else:
            logger.warning("Enhanced match context not found, using basic features only")
            return pd.DataFrame()
    
    def extract_venue_features(self, matches_data: pd.DataFrame, venue: str) -> Dict[str, float]:
        """Extract venue-specific characteristics with enhanced context data."""
        if venue in self.venue_stats:
            return self.venue_stats[venue]
        
        venue_matches = matches_data[matches_data['venue'] == venue]
        
        if len(venue_matches) == 0:
            features = self._get_default_venue_features()
        else:
            # Analyze venue characteristics from main data
            total_matches = len(venue_matches)
            toss_win_bat_first = len(venue_matches[
                (venue_matches['toss_decision'] == 'bat') & 
                (venue_matches['toss_winner'] == venue_matches['winner'])
            ])
            toss_win_field_first = len(venue_matches[
                (venue_matches['toss_decision'] == 'field') & 
                (venue_matches['toss_winner'] == venue_matches['winner'])
            ])
            
            bat_first_decisions = len(venue_matches[venue_matches['toss_decision'] == 'bat'])
            
            features = {
                'total_matches_played': total_matches,
                'bat_first_advantage': (toss_win_bat_first / bat_first_decisions) if bat_first_decisions > 0 else 0.5,
                'field_first_advantage': (toss_win_field_first / (total_matches - bat_first_decisions)) if (total_matches - bat_first_decisions) > 0 else 0.5,
                'toss_importance': (toss_win_bat_first + toss_win_field_first) / total_matches,
                'bat_first_preference': bat_first_decisions / total_matches
            }
        
        # Enhance with JSON context data if available
        if not self.match_context.empty:
            enhanced_features = self._extract_enhanced_venue_features(venue)
            features.update(enhanced_features)
        
        self.venue_stats[venue] = features
        return features
        
    def _extract_enhanced_venue_features(self, venue: str) -> Dict[str, float]:
        """Extract additional venue features from JSON context data."""
        # Find matches at this venue in enhanced context
        venue_context = self.match_context[
            self.match_context['venue'].str.contains(venue, case=False, na=False)
        ]
        
        if venue_context.empty:
            return {}
            
        enhanced_features = {}
        
        # Toss decision patterns
        toss_bat_decisions = len(venue_context[venue_context['toss_decision'] == 'bat'])
        total_decisions = len(venue_context[venue_context['toss_decision'].notna()])
        
        if total_decisions > 0:
            enhanced_features['enhanced_bat_first_rate'] = toss_bat_decisions / total_decisions
            
        # Toss winner advantage
        toss_wins_match = len(venue_context[
            venue_context['toss_winner'] == venue_context['outcome_winner']
        ])
        total_toss_data = len(venue_context[venue_context['toss_winner'].notna()])
        
        if total_toss_data > 0:
            enhanced_features['enhanced_toss_advantage'] = toss_wins_match / total_toss_data
            
        # Venue experience (total JSON matches available)
        enhanced_features['venue_json_matches'] = len(venue_context)
        
        return enhanced_features
        return features
    
    def extract_match_timing_features(self, match_date: str) -> Dict[str, float]:
        """Extract features based on match timing."""
        match_dt = pd.to_datetime(match_date)
        
        # Extract temporal features
        features = {
            'month': match_dt.month,
            'day_of_year': match_dt.dayofyear,
            'is_weekend': 1.0 if match_dt.weekday() >= 5 else 0.0,
            'season_phase': self._get_season_phase(match_dt),
            'days_since_season_start': self._days_since_season_start(match_dt)
        }
        
        return features
    
    def extract_tournament_context_features(self, matches_data: pd.DataFrame,
                                          match_date: str, team1: str, team2: str) -> Dict[str, float]:
        """Extract tournament stage and pressure context."""
        current_season_matches = matches_data[
            matches_data['match_date'].dt.year == pd.to_datetime(match_date).year
        ]
        
        current_date = pd.to_datetime(match_date)
        played_matches = current_season_matches[current_season_matches['match_date'] < current_date]
        
        # Calculate league position (simplified)
        team1_points = self._calculate_team_points(played_matches, team1)
        team2_points = self._calculate_team_points(played_matches, team2)
        
        total_teams = len(current_season_matches['team1'].unique())
        
        # Handle division by zero for season progress
        if len(current_season_matches) > 0:
            season_progress = len(played_matches) / len(current_season_matches)
        else:
            season_progress = 0.5  # Default mid-season value
        
        features = {
            'season_progress': season_progress,
            'team1_league_position': team1_points,
            'team2_league_position': team2_points,
            'points_gap': abs(team1_points - team2_points),
            'playoff_pressure': 1.0 if season_progress > 0.8 else season_progress * 0.5,
            'elimination_context': self._get_elimination_context(match_date, matches_data)
        }
        
        return features
    
    def extract_rest_and_fatigue_features(self, matches_data: pd.DataFrame,
                                        match_date: str, team1: str, team2: str) -> Dict[str, float]:
        """Calculate team rest and fatigue factors."""
        current_date = pd.to_datetime(match_date)
        
        # Get last match for each team
        team1_last = self._get_last_match_date(matches_data, team1, current_date)
        team2_last = self._get_last_match_date(matches_data, team2, current_date)
        
        team1_rest = (current_date - team1_last).days if team1_last else 7
        team2_rest = (current_date - team2_last).days if team2_last else 7
        
        features = {
            'team1_days_rest': min(team1_rest, 10),  # Cap at 10 days
            'team2_days_rest': min(team2_rest, 10),
            'rest_advantage': team1_rest - team2_rest,
            'both_teams_rested': 1.0 if min(team1_rest, team2_rest) >= 3 else 0.0
        }
        
        return features
    
    def _get_season_phase(self, match_date: datetime) -> float:
        """Determine season phase (0=start, 1=end)."""
        if match_date.month <= 4:
            return (match_date.month - 3) / 2  # March-April
        else:
            return (match_date.month - 3) / 3  # May onwards
    
    def _days_since_season_start(self, match_date: datetime) -> int:
        """Calculate days since season start (approximate)."""
        season_start = datetime(match_date.year, 3, 15)  # Typical IPL start
        return max((match_date - season_start).days, 0)
    
    def _calculate_team_points(self, matches_data: pd.DataFrame, team: str) -> float:
        """Calculate team points in league (simplified)."""
        team_matches = matches_data[
            (matches_data['team1'] == team) | (matches_data['team2'] == team)
        ]
        wins = len(team_matches[team_matches['winner'] == team])
        total = len(team_matches)
        return (wins * 2) / max(total, 1)  # 2 points per win
    
    def _get_elimination_context(self, match_date: str, matches_data: pd.DataFrame) -> float:
        """Determine if match is in elimination context."""
        # Simplified - would need tournament format details
        season_progress = pd.to_datetime(match_date).month
        return 1.0 if season_progress >= 5 else 0.0  # May onwards = playoffs
    
    def _get_last_match_date(self, matches_data: pd.DataFrame, team: str, current_date: datetime):
        """Get last match date for team."""
        team_matches = matches_data[
            ((matches_data['team1'] == team) | (matches_data['team2'] == team)) &
            (matches_data['match_date'] < current_date)
        ].sort_values('match_date', ascending=False)
        
        return team_matches.iloc[0]['match_date'] if len(team_matches) > 0 else None
    
    def _get_default_venue_features(self) -> Dict[str, float]:
        """Default venue features."""
        return {
            'total_matches_played': 0, 'bat_first_advantage': 0.5,
            'field_first_advantage': 0.5, 'toss_importance': 0.5,
            'bat_first_preference': 0.5
        }