__version__ = "1.0.1"
__author__ = "Arkapratim Mondal"
__email__ = "arkapratimmondal@gmail.com"

from .data.pipeline import DataPipeline
from .models.predictor import IPLPredictor
from .utils.config import Config

__all__ = ["IPLPredictor", "DataPipeline", "Config"]
