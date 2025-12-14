"""
DAAM Analyzer for Interactive Applications

Provides a stateful analyzer class for interactive Reverse DAAM analysis.
Designed for use in web interfaces like Gradio where state needs to be maintained
between image generation and region analysis.

Usage:
    analyzer = ReverseDAAMAnalyzer()
    image = analyzer.generate("A cat with a hat", seed=42)
    fig = analyzer.analyze_mask(mask_array)
"""

# Standard library imports
import logging
from typing import Tuple, Optional, List

# Third-party imports
import torch
import numpy as np
import matplotlib.pyplot as plt
from diffusers import DiffusionPipeline
from PIL import Image

# Local imports
from ..core.hooks import AttentionHookManager, remove_hooks
from ..core.processing import process_attention_maps
from ..core.constants import DEFAULT_MODEL_ID, DEFAULT_QUICK_STEPS, DEFAULT_BAR_COLOR
from ..utils.logging import get_daam_logger

# Module logger
logger = get_daam_logger(__name__)


class ReverseDAAMAnalyzer:
    """
    Analyzer class for Reverse DAAM with state management.
    
    This class maintains state between image generation and analysis,
    making it suitable for interactive applications.
    
    Attributes:
        device: Computing device (cuda/cpu)
        pipe: Stable Diffusion pipeline
        hook_manager: Manages attention hooks
        global_map: Processed attention map
        decoded_tokens: Tokenized prompt
        last_image: Most recently generated image
    """
    
    def __init__(
        self, 
        model_id: str = DEFAULT_MODEL_ID, 
        device: Optional[str] = None
    ):
        """
        Initialize the analyzer with a diffusion model.
        
        Args:
            model_id: Hugging Face model identifier
            device: Computing device (defaults to cuda if available)
        """
        self.device = device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Initializing ReverseDAAMAnalyzer on {self.device}")
        logger.info(f"Loading model: {model_id}")
        
        self.pipe = DiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16)
        self.pipe.to(self.device)
        
        # State management
        self.hook_manager = AttentionHookManager()
        self.global_map: Optional[torch.Tensor] = None
        self.decoded_tokens: List[str] = []
        self.last_image: Optional[Image.Image] = None
        
        logger.info("ReverseDAAMAnalyzer initialized successfully")

    def generate(
        self, 
        prompt: str, 
        seed: int = 42, 
        steps: int = DEFAULT_QUICK_STEPS
    ) -> Image.Image:
        """
        Generate an image and collect attention maps.
        
        Args:
            prompt: Text prompt for generation
            seed: Random seed for reproducibility
            steps: Number of inference steps
            
        Returns:
            Generated PIL Image
        """
        logger.info(f"Generating image for prompt: '{prompt}'")
        logger.debug(f"Parameters: seed={seed}, steps={steps}")
        
        # Reset state
        self.hook_manager.reset()
        self.global_map = None
        
        # Tokenize
        tokenizer = self.pipe.tokenizer
        tokens = tokenizer.encode(prompt)
        self.decoded_tokens = [tokenizer.decode([t]) for t in tokens]
        logger.debug(f"Tokenized into {len(self.decoded_tokens)} tokens")
        
        # Register hooks
        hook_handles = self.hook_manager.register_hooks(self.pipe.unet)
                
        try:
            # Inference
            generator = torch.Generator(self.device).manual_seed(int(seed))
            with torch.no_grad():
                image = self.pipe(
                    prompt, 
                    generator=generator, 
                    num_inference_steps=int(steps)
                ).images[0]
                self.last_image = image
            logger.info("Image generation complete")
        finally:
            remove_hooks(hook_handles)
        
        # Process maps immediately
        self._process_maps(image.size)
        
        return image

    def _process_maps(self, size: Tuple[int, int]):
        """
        Process collected attention maps into global attention map.
        
        Args:
            size: (W, H) tuple of image size
        """
        logger.info("Processing attention maps...")
        self.global_map = process_attention_maps(self.hook_manager.attention_maps, size)
        logger.info("Attention maps processed")

    def analyze_mask(self, mask: np.ndarray) -> Optional[plt.Figure]:
        """
        Analyze a mask region and return token attribution plot.
        
        Args:
            mask: numpy array (H, W) or (H, W, C) where values > 0 indicate selected region
            
        Returns:
            matplotlib figure or None if no valid mask/map
        """
        if self.global_map is None:
            logger.warning("No global map available. Generate image first.")
            return None
            
        # Process mask
        logger.debug(f"Raw mask shape: {mask.shape}, dtype: {mask.dtype}")
        logger.debug(f"Raw mask unique values: {np.unique(mask)}")

        if mask.ndim == 3:
            # Take first channel
            mask = mask[:, :, 0]
            logger.debug("Converted 3D mask to 2D using first channel")
        
        # Normalize mask to 0-1
        mask = (mask > 0).astype(np.float32)
        
        logger.debug(f"Processed mask shape: {mask.shape}, sum: {mask.sum()}")
        
        # Check if mask is empty
        if mask.sum() == 0:
            logger.warning("Mask is empty, cannot analyze")
            return None
            
        # Convert to torch
        mask_tensor = torch.tensor(mask)
        
        # Log selected region
        y_idxs, x_idxs = np.where(mask > 0)
        if len(y_idxs) > 0:
            logger.debug(
                f"Selected region: "
                f"Y:[{y_idxs.min()}, {y_idxs.max()}], "
                f"X:[{x_idxs.min()}, {x_idxs.max()}]"
            )
        
        # Aggregate attention over mask
        # global_map: (H, W, 77)
        masked_attn = self.global_map * mask_tensor.unsqueeze(-1)
        
        # Mean over spatial dims
        token_scores = masked_attn.sum(dim=(0, 1)) / mask_tensor.sum()
        token_scores = token_scores.numpy()
        
        # Filter tokens
        valid_scores = token_scores[:len(self.decoded_tokens)]
        tokens_plot = self.decoded_tokens
        
        # Remove start/end tokens
        if len(tokens_plot) > 2:
            valid_scores = valid_scores[1:-1]
            tokens_plot = tokens_plot[1:-1]
            logger.debug(f"Using {len(tokens_plot)} tokens (removed start/end)")
            
        # Create plot
        logger.info("Creating token attribution plot...")
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.bar(range(len(tokens_plot)), valid_scores, color=DEFAULT_BAR_COLOR)
        ax.set_xticks(range(len(tokens_plot)))
        ax.set_xticklabels(tokens_plot, rotation=90)
        ax.set_title("Word Distribution in Selected Region")
        ax.set_xlabel("Tokens")
        ax.set_ylabel("Attention Score")
        plt.tight_layout()
        
        logger.info("Plot created successfully")
        return fig
