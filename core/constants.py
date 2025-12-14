"""
Constants Module for DAAM Analysis

Centralized constants used across the DAAM analysis codebase.
"""

# Stable Diffusion Model Constants
DEFAULT_MODEL_ID = "CompVis/stable-diffusion-v1-4"
SD_CONTEXT_LENGTH = 77  # Maximum token sequence length for SD models

# Generation Parameters
DEFAULT_SEED = 42
DEFAULT_STEPS = 50
DEFAULT_GUIDANCE_SCALE = 7.5
DEFAULT_QUICK_STEPS = 30  # For faster testing

# DAAM Parameters
DEFAULT_TAU = 0.5  # Threshold factor for binary mask in DAAM
MIN_TOKEN_INDEX = 1  # Skip start token
MAX_TOKEN_OFFSET = -1  # Skip end token

# Image Parameters
DEFAULT_IMAGE_SIZE = 512

# Visualization
DEFAULT_FIGURE_SIZE = (12, 6)
DEFAULT_BAR_COLOR = 'skyblue'
DEFAULT_BBOX_COLOR = 'r'
DEFAULT_BBOX_LINEWIDTH = 2

# File Paths
DEFAULT_OUTPUT_FILE = "reverse_daam_result.png"
