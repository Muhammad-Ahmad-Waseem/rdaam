"""
r-daam - Reverse Deep Attention Attribution Maps

A Python package for analyzing Stable Diffusion attention maps to understand
word-to-region and region-to-word attributions in generated images.

Public API:
    - run_reverse_daam: Analyze which words contributed to a bbox region
    - run_forward_daam: Analyze which regions a word contributed to  
    - ReverseDAAMAnalyzer: Stateful analyzer for interactive applications
"""

__version__ = "0.1.0"
__author__ = "DAAM Analysis Team"

# Public API
from .analysis.reverse import run_reverse_daam
from .analysis.forward import run_forward_daam
from .ui.analyzer import ReverseDAAMAnalyzer

# Core components (for advanced users)
from .core.hooks import AttentionHookManager
from .core.processing import process_attention_maps, extract_bbox_distribution

__all__ = [
    # Main analysis functions
    "run_reverse_daam",
    "run_forward_daam",
    "ReverseDAAMAnalyzer",
    
    # Core components
    "AttentionHookManager",
    "process_attention_maps",
    "extract_bbox_distribution",
]
