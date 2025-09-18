"""
Configuration management module for IPL Predictor.
"""

import yaml
from pathlib import Path
from typing import Dict, Any, Optional
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)

@dataclass
class AppConfig:
    """Application configuration."""
    name: str
    version: str
    debug: bool
    timezone: str

@dataclass
class DataConfig:
    """Data pipeline configuration."""
    raw_data_path: str
    processed_data_path: str
    matches_file: str
    deliveries_file: str
    feature_store_path: str
    validation: Dict[str, Any]

@dataclass
class ModelConfig:
    """Model configuration."""
    model_path: str
    experiment_name: str
    algorithms: list
    hyperparameters: Dict[str, Dict[str, list]]

@dataclass
class APIConfig:
    """API configuration."""
    host: str
    port: int
    workers: int
    reload: bool
    cors_origins: list

@dataclass
class MonitoringConfig:
    """Monitoring configuration."""
    log_level: str
    log_file: str
    metrics_file: str
    alert_thresholds: Dict[str, float]

class Config:
    """Main configuration class that loads and validates all configurations."""
    
    def __init__(self, config_path: Optional[str] = None):
        """Initialize configuration from YAML file."""
        if config_path is None:
            config_path = str(Path(__file__).parent.parent.parent / "config" / "config.yaml")
        
        self.config_path = Path(config_path)
        self._raw_config = self._load_config()
        self._validate_config()
        self._parse_config()
    
    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from YAML file."""
        try:
            with open(self.config_path, 'r') as f:
                config = yaml.safe_load(f)
            logger.info(f"Configuration loaded from {self.config_path}")
            return config
        except FileNotFoundError:
            logger.error(f"Configuration file not found: {self.config_path}")
            raise
        except yaml.YAMLError as e:
            logger.error(f"Error parsing YAML configuration: {e}")
            raise
    
    def _validate_config(self) -> None:
        """Validate required configuration sections."""
        required_sections = ['app', 'data', 'models', 'api', 'monitoring']
        for section in required_sections:
            if section not in self._raw_config:
                raise ValueError(f"Missing required configuration section: {section}")
    
    def _parse_config(self) -> None:
        """Parse configuration into structured dataclasses."""
        self.app = AppConfig(**self._raw_config['app'])
        self.data = DataConfig(**self._raw_config['data'])
        self.models = ModelConfig(**self._raw_config['models'])
        self.api = APIConfig(**self._raw_config['api'])
        self.monitoring = MonitoringConfig(**self._raw_config['monitoring'])
        
        # Additional configurations as dictionaries
        self.features = self._raw_config.get('features', {})
        self.mlflow = self._raw_config.get('mlflow', {})
        self.database = self._raw_config.get('database', {})
    
    def get_data_path(self, filename: str, data_type: str = 'raw') -> Path:
        """Get full path for data files."""
        base_path = Path(self.data.raw_data_path if data_type == 'raw' else self.data.processed_data_path)
        return base_path / filename
    
    def get_model_path(self, filename: str) -> Path:
        """Get full path for model files."""
        return Path(self.models.model_path) / filename
    
    def update_config(self, section: str, key: str, value: Any) -> None:
        """Update configuration value dynamically."""
        if hasattr(self, section):
            config_obj = getattr(self, section)
            if hasattr(config_obj, key):
                setattr(config_obj, key, value)
                logger.info(f"Updated {section}.{key} = {value}")
            else:
                logger.warning(f"Key {key} not found in section {section}")
        else:
            logger.warning(f"Section {section} not found in configuration")

# Global configuration instance
config = None

def get_config(config_path: Optional[str] = None) -> Config:
    """Get or create global configuration instance."""
    global config
    if config is None:
        config = Config(config_path)
    return config

def reload_config(config_path: Optional[str] = None) -> Config:
    """Reload configuration from file."""
    global config
    config = Config(config_path)
    return config