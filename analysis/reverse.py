"""
Reverse DAAM Analysis

Analyzes generated images using Reverse Deep Attention Attribution Maps (DAAM).
Given a bounding box, extracts the distribution of attention scores across
all tokens in the prompt to understand which words contributed to that region.
"""

# Standard library imports
import logging
from typing import Tuple, List

# Third-party imports
import torch
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from diffusers import DiffusionPipeline
from PIL import Image
import numpy.typing as npt

# Local imports
from ..core.hooks import AttentionHookManager, remove_hooks
from ..core.processing import process_attention_maps, extract_bbox_distribution
from ..core.constants import (
    DEFAULT_MODEL_ID,
    DEFAULT_SEED,
    DEFAULT_QUICK_STEPS,
    DEFAULT_FIGURE_SIZE,
    DEFAULT_BAR_COLOR,
    DEFAULT_BBOX_COLOR,
    DEFAULT_BBOX_LINEWIDTH,
    DEFAULT_OUTPUT_FILE
)
from ..utils.logging import get_daam_logger

# Module logger
logger = get_daam_logger(__name__)


def run_reverse_daam(
    prompt: str,
    seed: int,
    steps: int,
    model_id: str,
    bbox: List[int],
    pipe=None,  # Optional: pre-loaded pipeline for reuse
    return_global_map: bool = False
) -> Tuple[Image.Image, List[str], npt.NDArray]:
    """
    Run Reverse DAAM analysis on a generated image.
    
    Args:
        prompt: Text prompt for image generation
        seed: Random seed for reproducibility
        steps: Number of inference steps
        model_id: Hugging Face model ID
        bbox: Bounding box coordinates [x_min, y_min, x_max, y_max]
        pipe: Optional pre-loaded DiffusionPipeline for reuse (recommended for batch processing)
        
    Returns:
        Tuple containing:
            - generated_image: PIL Image object
            - decoded_tokens_plot: List of token strings (excluding start/end)
            - token_scores: numpy array of attention scores for plotting
    """
    logger.info(f"Starting Reverse DAAM analysis with prompt: '{prompt}'")
    logger.debug(f"Parameters: seed={seed}, steps={steps}, bbox={bbox}")
    
    # Setup device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Load pipeline only if not provided
    if pipe is None:
        logger.info(f"Using device: {device}")
        logger.info(f"Loading model: {model_id}")
        pipe = DiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16)
        pipe.to(device)
    else:
        logger.debug("Using pre-loaded pipeline")
        device = pipe.device
    
    # Tokenize
    tokenizer = pipe.tokenizer
    tokens = tokenizer.encode(prompt)
    decoded_tokens = [tokenizer.decode([t]) for t in tokens]
    logger.debug(f"Tokenized prompt into {len(decoded_tokens)} tokens")

    # Setup hooks
    hook_manager = AttentionHookManager()
    hook_handles = hook_manager.register_hooks(pipe.unet)

    try:
        # Generate image
        logger.info("Generating image...")
        generator = torch.Generator(device).manual_seed(seed)
        with torch.no_grad():
            image = pipe(prompt, generator=generator, num_inference_steps=steps).images[0]
        logger.info("Image generation complete")
    finally:
        remove_hooks(hook_handles)

    # Process attention maps
    logger.info("Processing attention maps...")
    global_map = process_attention_maps(hook_manager.attention_maps, image.size)

    if return_global_map:
        return image, global_map, decoded_tokens

    # Extract bbox distribution
    logger.info("Extracting token distribution from bounding box...")
    token_scores, decoded_tokens_plot = extract_bbox_distribution(
        global_map, bbox, decoded_tokens
    )
    
    logger.info(f"Analysis complete. Processed {len(decoded_tokens_plot)} tokens")

    
    return image, decoded_tokens_plot, token_scores


def main():
    """Main execution function for standalone usage."""
    prompt = "A cat with blue eyes and wearing a red hat"
    seed = DEFAULT_SEED
    steps = DEFAULT_QUICK_STEPS
    model_id = DEFAULT_MODEL_ID
    bbox = [180, 100, 380, 250]
    
    logger.info("="*60)
    logger.info("Reverse DAAM Analysis - Standalone Execution")
    logger.info("="*60)
    
    # Run analysis
    generated_image, decoded_tokens_plot, token_scores = run_reverse_daam(
        prompt, seed, steps, model_id, bbox
    )
    
    # Visualization
    logger.info("Creating visualization...")
    x_min, y_min, x_max, y_max = bbox
    plt.figure(figsize=DEFAULT_FIGURE_SIZE)

    ax1 = plt.subplot(1, 2, 1)
    ax1.imshow(generated_image)
    rect = patches.Rectangle(
        (x_min, y_min), x_max - x_min, y_max - y_min, 
        linewidth=DEFAULT_BBOX_LINEWIDTH, edgecolor=DEFAULT_BBOX_COLOR, facecolor='none'
    )
    ax1.add_patch(rect)
    ax1.set_title("Generated Image + BBox")
    ax1.axis("off")

    ax2 = plt.subplot(1, 2, 2)
    ax2.bar(range(len(decoded_tokens_plot)), token_scores, color=DEFAULT_BAR_COLOR)
    ax2.set_xticks(range(len(decoded_tokens_plot)))
    ax2.set_xticklabels(decoded_tokens_plot, rotation=90)
    ax2.set_title("Word Attention Distribution in BBox")
    ax2.set_xlabel("Tokens")
    ax2.set_ylabel("Attention Score")

    plt.tight_layout()
    plt.savefig(DEFAULT_OUTPUT_FILE)
    logger.info(f"Saved result to {DEFAULT_OUTPUT_FILE}")


if __name__ == "__main__":
    main()
