"""
Data loader for IPL prediction system.
Handles loading and basic preprocessing of raw IPL data.
"""
import pandas as pd
from pathlib import Path
from typing import Dict, Tuple, Optional
import logging

logger = logging.getLogger(__name__)

class IPLDataLoader:
    """Efficient data loader for IPL datasets."""
    
    def __init__(self, data_path: str = "data"):
        self.data_path = Path(data_path)
        self._cached_data = {}
    
    def load_matches(self) -> pd.DataFrame:
        """Load match information data."""
        if 'matches' not in self._cached_data:
            file_path = self.data_path / "raw" / "matches" / "ipl_match_info_2008_2024.csv"
            df = pd.read_csv(file_path)
            df['match_date'] = pd.to_datetime(df['match_date'])
            self._cached_data['matches'] = df
            logger.info(f"Loaded {len(df)} matches")
        return self._cached_data['matches']
    
    def load_ball_by_ball(self) -> pd.DataFrame:
        """Load ball-by-ball data."""
        if 'ball_by_ball' not in self._cached_data:
            file_path = self.data_path / "raw" / "ball_by_ball" / "ipl_ball_by_ball_2008_2024.csv"
            df = pd.read_csv(file_path)
            self._cached_data['ball_by_ball'] = df
            logger.info(f"Loaded {len(df)} ball-by-ball records")
        return self._cached_data['ball_by_ball']
    
    def load_players(self) -> pd.DataFrame:
        """Load player information."""
        if 'players' not in self._cached_data:
            file_path = self.data_path / "raw" / "players" / "ipl_players_2024.csv"
            df = pd.read_csv(file_path)
            df['dob'] = pd.to_datetime(df['dob'], format='%d/%m/%Y', errors='coerce')
            self._cached_data['players'] = df
            logger.info(f"Loaded {len(df)} players")
        return self._cached_data['players']
    
    def load_teams(self) -> pd.DataFrame:
        """Load team information."""
        if 'teams' not in self._cached_data:
            file_path = self.data_path / "raw" / "teams" / "ipl_teams_info.csv"
            df = pd.read_csv(file_path)
            self._cached_data['teams'] = df
            logger.info(f"Loaded {len(df)} teams")
        return self._cached_data['teams']
    
    def get_match_data(self, match_id: int) -> Tuple[pd.Series, pd.DataFrame]:
        """Get match info and ball-by-ball data for specific match."""
        matches = self.load_matches()
        ball_by_ball = self.load_ball_by_ball()
        
        match_info = matches[matches['match_number'] == match_id].iloc[0]
        match_balls = ball_by_ball[ball_by_ball['ID'] == match_id]
        
        return match_info, match_balls
    
    def get_recent_matches(self, team: str, n: int = 5) -> pd.DataFrame:
        """Get recent matches for a team."""
        matches = self.load_matches()
        team_matches = matches[
            (matches['team1'] == team) | (matches['team2'] == team)
        ].sort_values('match_date', ascending=False)
        
        return team_matches.head(n)
    
    def clear_cache(self):
        """Clear cached data."""
        self._cached_data.clear()
        logger.info("Data cache cleared")