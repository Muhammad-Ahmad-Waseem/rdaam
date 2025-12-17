"""
Reverse DAAM Batch Analyzer

Optimized class for batch processing multiple bboxes per caption.
Separates global map generation from bbox extraction for maximum efficiency.

Usage:
    analyzer = ReverseDAAMBatch(model_id="CompVis/stable-diffusion-v1-4")
    
    # Generate once per caption
    global_map, decoded_tokens = analyzer.generate_map(prompt, seed, steps)
    
    # Extract multiple bboxes from same map
    scores1 = analyzer.extract_bbox_scores(global_map, decoded_tokens, bbox1)
    scores2 = analyzer.extract_bbox_scores(global_map, decoded_tokens, bbox2)
"""

# Standard library imports
import logging
from typing import Tuple, List, Optional

# Third-party imports
import torch
from diffusers import DiffusionPipeline
from diffusers.utils import logging as diffusers_logging
from PIL import Image
import numpy as np
import numpy.typing as npt

# Local imports
from ..core.hooks import AttentionHookManager, remove_hooks
from ..core.processing import process_attention_maps, extract_bbox_distribution
from ..core.constants import DEFAULT_MODEL_ID, DEFAULT_SEED, DEFAULT_QUICK_STEPS
from ..utils.logging import get_daam_logger

# Module logger
logger = get_daam_logger(__name__)


class ReverseDAAMBatch:
    """
    Batch-optimized Reverse DAAM analyzer.
    
    Separates map generation from bbox extraction for efficient batch processing.
    Use this when analyzing multiple bboxes per caption.
    """
    
    def __init__(self, model_id: str = DEFAULT_MODEL_ID, device: Optional[str] = None):
        """
        Initialize the batch analyzer with a pre-loaded pipeline.
        
        Args:
            model_id: Hugging Face model ID
            device: Computing device (defaults to cuda if available)
        """
        self.model_id = model_id
        self.device = device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        
        logger.info(f"Initializing ReverseDAAMBatch on {self.device}")
        logger.info(f"Loading model: {model_id}")
        
        # Disable diffusers progress bar for cleaner output
        diffusers_logging.disable_progress_bar()
        
        self.pipe = DiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16)
        self.pipe.to(self.device)
        
        # Disable progress bar for this pipeline instance
        self.pipe.set_progress_bar_config(disable=True)
        
        logger.info("Pipeline loaded successfully")
    
    def generate_map(
        self,
        prompt: str,
        seed: int = DEFAULT_SEED,
        steps: int = DEFAULT_QUICK_STEPS
    ) -> Tuple[torch.Tensor, List[str], Image.Image]:
        """
        Generate image and global attention map for a prompt.
        
        This should be called ONCE per caption, then use extract_bbox_scores
        multiple times for different bboxes.
        
        Args:
            prompt: Text prompt for generation
            seed: Random seed
            steps: Number of inference steps
            
        Returns:
            Tuple of (global_map, decoded_tokens, generated_image)
        """
        logger.debug(f"Generating map for prompt: '{prompt}'")
        
        # Tokenize
        tokenizer = self.pipe.tokenizer
        tokens = tokenizer.encode(prompt)
        decoded_tokens = [tokenizer.decode([t]) for t in tokens]
        
        # Setup hooks
        hook_manager = AttentionHookManager()
        hook_handles = hook_manager.register_hooks(self.pipe.unet)
        
        try:
            # Generate image
            logger.debug("Generating image...")
            generator = torch.Generator(self.device).manual_seed(seed)
            with torch.no_grad():
                image = self.pipe(
                    prompt, 
                    generator=generator, 
                    num_inference_steps=steps
                ).images[0]
            logger.debug("Image generation complete")
        finally:
            remove_hooks(hook_handles)
        
        # Process attention maps
        logger.debug("Processing attention maps...")
        global_map = process_attention_maps(hook_manager.attention_maps, image.size)
        
        logger.debug(f"Map generated for {len(decoded_tokens)} tokens")
        return global_map, decoded_tokens, image
    
    def extract_bbox_scores(
        self,
        global_map: torch.Tensor,
        decoded_tokens: List[str],
        bbox: List[int]
    ) -> Tuple[npt.NDArray, List[str]]:
        """
        Extract token scores for a specific bbox from pre-generated map.
        
        Args:
            global_map: Pre-generated global attention map
            decoded_tokens: Token list from generate_map
            bbox: Bounding box [x_min, y_min, x_max, y_max]
            
        Returns:
            Tuple of (token_scores, tokens_plot)
        """
        logger.debug(f"Extracting scores for bbox: {bbox}")
        return extract_bbox_distribution(global_map, bbox, decoded_tokens)
