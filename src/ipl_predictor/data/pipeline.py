"""
Professional data pipeline for IPL match prediction.
Implements ETL processes, data validation, and feature engineering.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
import logging
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
import joblib
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

from ..utils.config import get_config
from ..utils.logging_config import setup_logging

logger = logging.getLogger(__name__)

class DataValidator:
    """Data validation and quality checks."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.validation_rules = config.get('validation', {})
    
    def validate_dataframe(self, df: pd.DataFrame, name: str) -> bool:
        """Validate DataFrame quality and structure."""
        logger.info(f"Validating {name} dataset...")
        
        # Check for empty DataFrame
        if df.empty:
            logger.error(f"{name} dataset is empty")
            return False
        
        # Check missing values ratio
        missing_ratio = df.isnull().sum().sum() / (df.shape[0] * df.shape[1])
        max_missing = self.validation_rules.get('max_missing_values_ratio', 0.1)
        
        if missing_ratio > max_missing:
            logger.warning(f"{name} has {missing_ratio:.2%} missing values (threshold: {max_missing:.2%})")
        
        # Check data types
        object_cols = df.select_dtypes(include=['object']).columns
        logger.info(f"{name} - Shape: {df.shape}, Missing: {missing_ratio:.2%}, "
                   f"Categorical columns: {len(object_cols)}")
        
        return True
    
    def validate_team_data(self, df: pd.DataFrame) -> bool:
        """Validate team-specific data requirements."""
        required_cols = ['team1', 'team2', 'winner']
        missing_cols = [col for col in required_cols if col not in df.columns]
        
        if missing_cols:
            logger.error(f"Missing required columns: {missing_cols}")
            return False
        
        # Check minimum matches per team
        min_matches = self.validation_rules.get('min_matches_per_team', 10)
        teams = pd.concat([df['team1'], df['team2']]).value_counts()
        low_data_teams = teams[teams < min_matches].index.tolist()
        
        if low_data_teams:
            logger.warning(f"Teams with insufficient data (<{min_matches} matches): {low_data_teams}")
        
        return True

class FeatureEngineer:
    """Advanced feature engineering for IPL match prediction."""
    
    def __init__(self):
        self.team_encoders = {}
        self.scalers = {}
        self.feature_names = []
    
    def create_team_features(self, matches_df: pd.DataFrame, deliveries_df: pd.DataFrame) -> pd.DataFrame:
        """Create comprehensive team-level features."""
        logger.info("Creating team features...")
        
        # Team win rates (rolling window)
        team_stats = self._calculate_team_statistics(matches_df)
        
        # Batting and bowling features from ball-by-ball data
        batting_stats = self._calculate_batting_features(deliveries_df)
        bowling_stats = self._calculate_bowling_features(deliveries_df)
        
        # Recent form features
        form_features = self._calculate_recent_form(matches_df)
        
        # Venue-specific features
        venue_features = self._calculate_venue_features(matches_df)
        
        # Player impact features
        player_features = self._calculate_player_impact_features(deliveries_df)
        
        # Head-to-head features
        h2h_features = self._calculate_head_to_head_features(matches_df)
        
        # Merge all features
        features = team_stats.merge(batting_stats, on='team', how='left')
        features = features.merge(bowling_stats, on='team', how='left')
        features = features.merge(form_features, on='team', how='left')
        features = features.merge(venue_features, on='team', how='left')
        features = features.merge(player_features, on='team', how='left')
        features = features.merge(h2h_features, on='team', how='left')
        
        return features.fillna(0)
    
    def _calculate_team_statistics(self, matches_df: pd.DataFrame) -> pd.DataFrame:
        """Calculate basic team statistics."""
        # Team performance metrics
        team_matches = pd.concat([
            matches_df[['team1', 'date']].rename(columns={'team1': 'team'}),
            matches_df[['team2', 'date']].rename(columns={'team2': 'team'})
        ])
        
        team_wins = matches_df.groupby('winner').size().reset_index(name='wins')
        total_matches = team_matches.groupby('team').size().reset_index(name='total_matches')
        
        team_stats = total_matches.merge(team_wins, left_on='team', right_on='winner', how='left')
        team_stats['wins'] = team_stats['wins'].fillna(0)
        team_stats['win_rate'] = team_stats['wins'] / team_stats['total_matches']
        
        return team_stats[['team', 'win_rate', 'total_matches']]
    
    def _calculate_batting_features(self, deliveries_df: pd.DataFrame) -> pd.DataFrame:
        """Calculate advanced batting strength features from ball-by-ball data."""
        # Aggregate basic batting stats
        batting_stats = deliveries_df.groupby('batting_team').agg({
            'total_runs': ['sum', 'mean'],
            'batsman_runs': ['sum', 'mean'],
            'extra_runs': 'mean',
            'match_id': 'count'
        }).round(3)
        
        batting_stats.columns = ['total_runs_sum', 'total_runs_avg', 'batsman_runs_sum', 
                               'batsman_runs_avg', 'extra_runs_avg', 'balls_faced']
        
        # Calculate advanced metrics
        batting_stats['avg_strike_rate'] = (batting_stats['batsman_runs_sum'] / 
                                          batting_stats['balls_faced']) * 100
        
        # Calculate boundary percentage (4s and 6s)
        boundaries = deliveries_df.groupby('batting_team')['batsman_runs'].apply(
            lambda x: (x >= 4).mean()
        )
        batting_stats['boundaries_rate'] = boundaries
        
        # Calculate dot ball percentage
        dot_balls = deliveries_df.groupby('batting_team')['batsman_runs'].apply(
            lambda x: (x == 0).mean()
        )
        batting_stats['dot_ball_rate'] = dot_balls
        
        # Calculate average powerplay score (first 6 overs)
        powerplay_runs = deliveries_df[deliveries_df['over'] <= 6].groupby('batting_team')['total_runs'].mean()
        batting_stats['powerplay_avg'] = powerplay_runs
        
        # Calculate death overs performance (overs 16-20)
        death_runs = deliveries_df[deliveries_df['over'] >= 16].groupby('batting_team')['total_runs'].mean()
        batting_stats['death_overs_avg'] = death_runs
        
        # Player diversity - number of unique batsmen who contributed runs
        player_diversity = deliveries_df[deliveries_df['batsman_runs'] > 0].groupby('batting_team')['batter'].nunique()
        batting_stats['batting_depth'] = player_diversity
        
        return batting_stats.reset_index().rename(columns={'batting_team': 'team'})
    
    def _calculate_bowling_features(self, deliveries_df: pd.DataFrame) -> pd.DataFrame:
        """Calculate advanced bowling strength features from ball-by-ball data."""
        # Basic bowling aggregations
        bowling_stats = deliveries_df.groupby('bowling_team').agg({
            'total_runs': ['sum', 'mean'],
            'is_wicket': 'sum',
            'match_id': 'count'
        }).round(3)
        
        bowling_stats.columns = ['runs_conceded_sum', 'runs_conceded_avg', 'wickets_taken', 'balls_bowled']
        
        # Calculate advanced bowling metrics
        bowling_stats['economy_rate'] = (bowling_stats['runs_conceded_sum'] / 
                                       bowling_stats['balls_bowled']) * 6
        bowling_stats['wicket_rate'] = bowling_stats['wickets_taken'] / bowling_stats['balls_bowled']
        
        # Calculate strike rate (balls per wicket)
        bowling_stats['bowling_strike_rate'] = bowling_stats['balls_bowled'] / bowling_stats['wickets_taken']
        bowling_stats['bowling_strike_rate'] = bowling_stats['bowling_strike_rate'].replace([np.inf, -np.inf], 999)
        
        # Calculate powerplay economy (first 6 overs)
        powerplay_econ = deliveries_df[deliveries_df['over'] <= 6].groupby('bowling_team')['total_runs'].mean() * 6
        bowling_stats['powerplay_economy'] = powerplay_econ
        
        # Calculate death overs economy (overs 16-20)
        death_econ = deliveries_df[deliveries_df['over'] >= 16].groupby('bowling_team')['total_runs'].mean() * 6
        bowling_stats['death_economy'] = death_econ
        
        # Bowling depth - number of unique bowlers
        bowling_depth = deliveries_df.groupby('bowling_team')['bowler'].nunique()
        bowling_stats['bowling_depth'] = bowling_depth
        
        # Wicket distribution - how evenly wickets are distributed among bowlers
        wicket_distribution = deliveries_df[deliveries_df['is_wicket'] == 1].groupby('bowling_team')['bowler'].nunique()
        bowling_stats['wicket_taking_bowlers'] = wicket_distribution
        
        # Extra runs conceded rate
        extras_rate = deliveries_df.groupby('bowling_team')['extra_runs'].mean()
        bowling_stats['extras_rate'] = extras_rate
        
        return bowling_stats.reset_index().rename(columns={'bowling_team': 'team'})
    
    def _calculate_recent_form(self, matches_df: pd.DataFrame, window: int = 5) -> pd.DataFrame:
        """Calculate recent form features."""
        matches_df['date'] = pd.to_datetime(matches_df['date'])
        matches_sorted = matches_df.sort_values('date')
        
        form_data = []
        for team in pd.concat([matches_df['team1'], matches_df['team2']]).unique():
            team_matches = matches_sorted[(matches_sorted['team1'] == team) | 
                                        (matches_sorted['team2'] == team)].copy()
            
            team_matches['is_winner'] = (team_matches['winner'] == team).astype(int)
            recent_form = team_matches['is_winner'].rolling(window=window, min_periods=1).mean().iloc[-1]
            
            form_data.append({
                'team': team,
                'recent_form': recent_form,
                'last_5_wins': team_matches['is_winner'].tail(5).sum()
            })
        
        return pd.DataFrame(form_data)
    
    def _calculate_venue_features(self, matches_df: pd.DataFrame) -> pd.DataFrame:
        """Calculate comprehensive venue-specific performance features."""
        venue_stats = []
        
        for team in pd.concat([matches_df['team1'], matches_df['team2']]).unique():
            team_matches = matches_df[
                ((matches_df['team1'] == team) | (matches_df['team2'] == team))
            ]
            
            # Venue-wise performance analysis
            venue_performance = {}
            city_performance = {}
            
            for _, match in team_matches.iterrows():
                venue = match['venue']
                city = match['city']
                
                # Track venue performance
                if venue not in venue_performance:
                    venue_performance[venue] = {'wins': 0, 'total': 0}
                venue_performance[venue]['total'] += 1
                if match['winner'] == team:
                    venue_performance[venue]['wins'] += 1
                
                # Track city performance
                if city not in city_performance:
                    city_performance[city] = {'wins': 0, 'total': 0}
                city_performance[city]['total'] += 1
                if match['winner'] == team:
                    city_performance[city]['wins'] += 1
            
            # Calculate venue statistics
            venue_win_rates = [
                wins['wins'] / wins['total'] if wins['total'] > 0 else 0 
                for wins in venue_performance.values()
            ]
            
            city_win_rates = [
                wins['wins'] / wins['total'] if wins['total'] > 0 else 0 
                for wins in city_performance.values()
            ]
            
            # Find best and worst performing venues
            best_venue_rate = max(venue_win_rates) if venue_win_rates else 0
            worst_venue_rate = min(venue_win_rates) if venue_win_rates else 0
            venue_consistency = np.std(venue_win_rates) if len(venue_win_rates) > 1 else 0
            
            # Home venue identification (most frequently played venue)
            home_venue = max(venue_performance.items(), key=lambda x: x[1]['total'])[0] if venue_performance else None
            home_venue_rate = venue_performance[home_venue]['wins'] / venue_performance[home_venue]['total'] if home_venue else 0
            
            # Calculate venue diversity
            venue_diversity = len(venue_performance)
            
            # Batting/Bowling friendly venue analysis
            # (This would require more detailed analysis with runs scored)
            avg_venue_advantage = np.mean(venue_win_rates) if venue_win_rates else 0
            
            venue_stats.append({
                'team': team,
                'venue_advantage': avg_venue_advantage,
                'venues_played': venue_diversity,
                'best_venue_rate': best_venue_rate,
                'worst_venue_rate': worst_venue_rate,
                'venue_consistency': venue_consistency,
                'home_venue_rate': home_venue_rate,
                'cities_played': len(city_performance),
                'avg_city_performance': np.mean(city_win_rates) if city_win_rates else 0
            })
        
        return pd.DataFrame(venue_stats)
    
    def _calculate_player_impact_features(self, deliveries_df: pd.DataFrame) -> pd.DataFrame:
        """Calculate player impact and performance features."""
        # Top performer impact for each team
        player_stats = []
        
        for team in deliveries_df['batting_team'].unique():
            team_deliveries = deliveries_df[deliveries_df['batting_team'] == team]
            
            # Top batsman performance
            batsman_runs = team_deliveries.groupby('batter')['batsman_runs'].agg(['sum', 'mean', 'count'])
            if not batsman_runs.empty:
                top_batsman = batsman_runs.loc[batsman_runs['sum'].idxmax()]
                avg_top_batsman_contribution = top_batsman['sum'] / batsman_runs['sum'].sum()
                top_batsman_avg = top_batsman['mean']
                top_batsman_consistency = top_batsman['count']
            else:
                avg_top_batsman_contribution = 0
                top_batsman_avg = 0
                top_batsman_consistency = 0
            
            # Calculate team batting depth (how many players score regularly)
            regular_scorers = len(batsman_runs[batsman_runs['mean'] > 10])
            
            player_stats.append({
                'team': team,
                'top_batsman_contribution': avg_top_batsman_contribution,
                'top_batsman_avg': top_batsman_avg,
                'top_batsman_consistency': top_batsman_consistency,
                'regular_scorers': regular_scorers
            })
        
        # Add bowling impact features
        bowling_stats = []
        for team in deliveries_df['bowling_team'].unique():
            team_bowling = deliveries_df[deliveries_df['bowling_team'] == team]
            
            # Top bowler performance
            bowler_wickets = team_bowling.groupby('bowler')['is_wicket'].agg(['sum', 'count'])
            bowler_economy = team_bowling.groupby('bowler').apply(
                lambda x: (x['total_runs'].sum() / x['match_id'].count()) * 6 if x['match_id'].count() > 0 else 999
            )
            
            if not bowler_wickets.empty and not bowler_economy.empty:
                # Combine wickets and economy for bowler rating
                bowler_rating = (bowler_wickets['sum'] * 10) - bowler_economy
                if not bowler_rating.empty:
                    top_bowler_idx = bowler_rating.idxmax()
                    top_bowler_wickets = bowler_wickets.loc[top_bowler_idx, 'sum']
                    top_bowler_economy = bowler_economy.loc[top_bowler_idx]
                else:
                    top_bowler_wickets = 0
                    top_bowler_economy = 8.0
            else:
                top_bowler_wickets = 0
                top_bowler_economy = 8.0
            
            bowling_stats.append({
                'team': team,
                'top_bowler_wickets': top_bowler_wickets,
                'top_bowler_economy': top_bowler_economy
            })
        
        # Merge batting and bowling player features
        player_features = pd.DataFrame(player_stats)
        bowling_features = pd.DataFrame(bowling_stats)
        
        return player_features.merge(bowling_features, on='team', how='outer').fillna(0)
    
    def _calculate_head_to_head_features(self, matches_df: pd.DataFrame) -> pd.DataFrame:
        """Calculate head-to-head performance features between teams."""
        h2h_stats = []
        
        teams = pd.concat([matches_df['team1'], matches_df['team2']]).unique()
        
        for team in teams:
            team_matches = matches_df[
                (matches_df['team1'] == team) | (matches_df['team2'] == team)
            ]
            
            # Overall head-to-head record
            opponents = []
            for _, match in team_matches.iterrows():
                opponent = match['team2'] if match['team1'] == team else match['team1']
                opponents.append(opponent)
            
            # Calculate dominance against frequent opponents
            opponent_counts = pd.Series(opponents).value_counts()
            frequent_opponents = opponent_counts[opponent_counts >= 3].index
            
            avg_h2h_performance = 0
            if len(frequent_opponents) > 0:
                h2h_wins = 0
                h2h_total = 0
                
                for opponent in frequent_opponents:
                    opponent_matches = matches_df[
                        ((matches_df['team1'] == team) & (matches_df['team2'] == opponent)) |
                        ((matches_df['team1'] == opponent) & (matches_df['team2'] == team))
                    ]
                    h2h_total += len(opponent_matches)
                    h2h_wins += len(opponent_matches[opponent_matches['winner'] == team])
                
                avg_h2h_performance = h2h_wins / h2h_total if h2h_total > 0 else 0
            
            # Performance against top teams (teams with >60% win rate)
            team_win_rates = matches_df.groupby('winner').size() / matches_df.groupby(['team1', 'team2']).size().groupby(['team1', 'team2']).first().reset_index().groupby(['team1']).size()
            
            h2h_stats.append({
                'team': team,
                'h2h_performance': avg_h2h_performance,
                'unique_opponents': len(opponent_counts),
                'most_played_opponent_rate': opponent_counts.iloc[0] / len(team_matches) if len(opponent_counts) > 0 else 0
            })
        
        return pd.DataFrame(h2h_stats)
    
    def create_match_features(self, matches_df: pd.DataFrame, team_features: pd.DataFrame) -> pd.DataFrame:
        """Create match-level features by combining team features and temporal patterns."""
        logger.info("Creating match features...")
        
        match_features = matches_df.copy()
        
        # Add temporal features
        match_features = self._add_temporal_features(match_features)
        
        # Merge team features for both teams
        match_features = match_features.merge(
            team_features.add_suffix('_team1'), 
            left_on='team1', right_on='team_team1', how='left'
        )
        match_features = match_features.merge(
            team_features.add_suffix('_team2'), 
            left_on='team2', right_on='team_team2', how='left'
        )
        
        # Create relative features
        match_features['win_rate_diff'] = (match_features['win_rate_team1'] - 
                                         match_features['win_rate_team2'])
        match_features['form_diff'] = (match_features['recent_form_team1'] - 
                                     match_features['recent_form_team2'])
        match_features['batting_strength_diff'] = (match_features['avg_strike_rate_team1'] - 
                                                 match_features['avg_strike_rate_team2'])
        match_features['bowling_strength_diff'] = (match_features['economy_rate_team2'] - 
                                                 match_features['economy_rate_team1'])  # Lower is better
        
        # Toss features
        match_features['toss_advantage'] = np.where(
            match_features['toss_winner'] == match_features['team1'], 1, 0
        )
        
        # Create match context features
        match_features['is_playoff'] = np.where(
            match_features['season'].notna(), 0, 0  # Placeholder - would need playoff identification logic
        )
        
        # Remove intermediate columns
        cols_to_drop = [col for col in match_features.columns if col.endswith(('_team1', '_team2'))]
        team_cols_to_drop = ['team_team1', 'team_team2']
        match_features = match_features.drop(columns=[col for col in cols_to_drop + team_cols_to_drop 
                                                    if col in match_features.columns])
        
        return match_features
    
    def _add_temporal_features(self, match_features: pd.DataFrame) -> pd.DataFrame:
        """Add time-based and seasonal features."""
        match_features['date'] = pd.to_datetime(match_features['date'])
        
        # Season phase features
        match_features['month'] = match_features['date'].dt.month
        match_features['day_of_year'] = match_features['date'].dt.dayofyear
        
        # IPL typically runs March-May, so create season phase
        match_features['season_phase'] = np.where(
            match_features['month'] == 3, 'early',
            np.where(match_features['month'] == 4, 'mid', 'late')
        )
        
        # Weekend vs weekday
        match_features['is_weekend'] = (match_features['date'].dt.weekday >= 5).astype(int)
        
        # Days since season start (if we have season info)
        if 'season' in match_features.columns:
            season_starts = match_features.groupby('season')['date'].min()
            match_features['days_since_season_start'] = match_features.apply(
                lambda row: (row['date'] - season_starts[row['season']]).days 
                if row['season'] in season_starts.index else 0, axis=1
            )
        
        # Tournament momentum (match number in season)
        match_features = match_features.sort_values('date')
        match_features['match_number_in_season'] = match_features.groupby('season').cumcount() + 1
        
        return match_features

class DataPipeline:
    """Main data pipeline orchestrator."""
    
    def __init__(self, config_path: Optional[str] = None):
        self.config = get_config(config_path)
        setup_logging(self.config.monitoring.log_level)
        
        self.validator = DataValidator(self.config.data.validation)
        self.feature_engineer = FeatureEngineer()
        self.label_encoders = {}
        self.scaler = StandardScaler()
        
    def load_raw_data(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Load raw data from CSV files."""
        logger.info("Loading raw data...")
        
        matches_path = self.config.get_data_path(self.config.data.matches_file)
        deliveries_path = self.config.get_data_path(self.config.data.deliveries_file)
        
        matches_df = pd.read_csv(matches_path)
        deliveries_df = pd.read_csv(deliveries_path)
        
        # Validate data
        self.validator.validate_dataframe(matches_df, "matches")
        self.validator.validate_dataframe(deliveries_df, "deliveries")
        self.validator.validate_team_data(matches_df)
        
        return matches_df, deliveries_df
    
    def preprocess_data(self, matches_df: pd.DataFrame, deliveries_df: pd.DataFrame) -> pd.DataFrame:
        """Preprocess and engineer features."""
        logger.info("Starting data preprocessing...")
        
        # Clean data
        matches_df = self._clean_matches_data(matches_df)
        deliveries_df = self._clean_deliveries_data(deliveries_df)
        
        # Engineer features
        team_features = self.feature_engineer.create_team_features(matches_df, deliveries_df)
        match_features = self.feature_engineer.create_match_features(matches_df, team_features)
        
        # Encode categorical variables
        match_features = self._encode_categorical_features(match_features)
        
        # Save processed data
        processed_path = self.config.get_data_path("processed_features.csv", "processed")
        match_features.to_csv(processed_path, index=False)
        logger.info(f"Processed data saved to {processed_path}")
        
        return match_features
    
    def _clean_matches_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean matches dataset."""
        # Remove matches with no result
        df = df.dropna(subset=['winner'])
        
        # Convert date column
        df['date'] = pd.to_datetime(df['date'])
        
        # Fill missing values
        df['city'] = df['city'].fillna('Unknown')
        df['venue'] = df['venue'].fillna('Unknown')
        df['toss_winner'] = df['toss_winner'].fillna(df['team1'])
        df['toss_decision'] = df['toss_decision'].fillna('bat')
        
        return df
    
    def _clean_deliveries_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean deliveries dataset."""
        # Fill missing values
        df['batsman_runs'] = df['batsman_runs'].fillna(0)
        df['extra_runs'] = df['extra_runs'].fillna(0)
        df['total_runs'] = df['total_runs'].fillna(0)
        
        return df
    
    def _encode_categorical_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Encode categorical features."""
        categorical_cols = ['team1', 'team2', 'toss_winner', 'toss_decision', 'city', 'venue']
        
        for col in categorical_cols:
            if col in df.columns:
                le = LabelEncoder()
                df[col] = le.fit_transform(df[col].astype(str))
                self.label_encoders[col] = le
        
        return df
    
    def prepare_training_data(self, df: pd.DataFrame, target_col: str = 'winner') -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """Prepare data for model training."""
        logger.info("Preparing training data...")
        
        # Encode target variable
        le_target = LabelEncoder()
        y = np.array(le_target.fit_transform(df[target_col]))
        self.label_encoders['target'] = le_target
        
        # Select features
        feature_cols = [col for col in df.columns if col not in ['winner', 'date', 'id']]
        X = df[feature_cols].select_dtypes(include=[np.number])
        
        # Handle missing values
        X = X.fillna(X.median())
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        self.feature_names = X.columns.tolist()
        
        logger.info(f"Training data prepared: {X_scaled.shape[0]} samples, {X_scaled.shape[1]} features")
        
        return X_scaled, y, self.feature_names
    
    def save_preprocessors(self) -> None:
        """Save preprocessing objects."""
        preprocessors = {
            'label_encoders': self.label_encoders,
            'scaler': self.scaler,
            'feature_names': self.feature_names
        }
        
        save_path = self.config.get_model_path('preprocessors.pkl')
        joblib.dump(preprocessors, save_path)
        logger.info(f"Preprocessors saved to {save_path}")
    
    def load_preprocessors(self) -> None:
        """Load preprocessing objects."""
        load_path = self.config.get_model_path('preprocessors.pkl')
        preprocessors = joblib.load(load_path)
        
        self.label_encoders = preprocessors['label_encoders']
        self.scaler = preprocessors['scaler']
        self.feature_names = preprocessors['feature_names']
        
        logger.info(f"Preprocessors loaded from {load_path}")