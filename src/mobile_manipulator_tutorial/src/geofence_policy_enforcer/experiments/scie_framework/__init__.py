"""
SCIE-level Experiment Framework for Geofence Policy Enforcer

This framework provides:
1. Factorial experiment design with full combinatorial coverage
2. Structured data logging in JSON/CSV formats
3. Statistical analysis (ANOVA, t-test, effect size)
4. Publication-ready visualization

Author: Geofence Research Team
"""

from .config import ExperimentConfig, ScenarioConfig, MethodConfig
from .data_logger import ExperimentLogger, TrialResult
from .metrics_collector import MetricsCollector
from .experiment_runner import FactorialExperimentRunner
from .statistical_analysis import StatisticalAnalyzer
from .visualization import PaperFigureGenerator

__all__ = [
    'ExperimentConfig',
    'ScenarioConfig',
    'MethodConfig',
    'ExperimentLogger',
    'TrialResult',
    'MetricsCollector',
    'FactorialExperimentRunner',
    'StatisticalAnalyzer',
    'PaperFigureGenerator',
]

__version__ = '1.0.0'
