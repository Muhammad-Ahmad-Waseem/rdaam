"""
Forward DAAM Analysis

Implements forward DAAM (word-to-region mapping) as opposed to reverse DAAM.
Given a specific target word, generates a heatmap showing which regions
of the image that word contributed to.
"""

# Standard library imports
import logging
from typing import Optional, Tuple

# Third-party imports
import torch
from PIL import Image

# Local imports
from ..core.hooks import AttentionHookManager, remove_hooks
from ..core.processing import process_attention_maps
from ..core.constants import DEFAULT_SEED, DEFAULT_QUICK_STEPS, DEFAULT_TAU, DEFAULT_MODEL_ID
from ..utils.logging import get_daam_logger

# Module logger
logger = get_daam_logger(__name__)


def run_forward_daam(
    pipe,
    prompt: str,
    word: str,
    seed: int = DEFAULT_SEED,
    steps: int = DEFAULT_QUICK_STEPS,
    tau: float = DEFAULT_TAU
) -> Tuple[Optional[Image.Image], Optional[torch.Tensor], Optional[torch.Tensor]]:
    """
    Run forward DAAM extraction for a specific target word.
    
    Args:
        pipe: Diffusers pipeline (pre-loaded)
        prompt: Text prompt for generation
        word: Target word to extract attention for
        seed: Random seed for reproducibility
        steps: Number of inference steps
        tau: Threshold factor for binary mask (0.0 to 1.0)
        
    Returns:
        Tuple containing:
            - image: Generated PIL Image (or None if word not found)
            - aggregated_heatmap: Attention heatmap for the word (H, W)
            - binary_mask: Binary mask after thresholding
    """
    device = pipe.device
    logger.info(f"Running forward DAAM for word: '{word}' in prompt: '{prompt}'")
    logger.debug(f"Parameters: seed={seed}, steps={steps}, tau={tau}")
    
    # Tokenize to find word index
    tokenizer = pipe.tokenizer
    tokens = tokenizer.encode(prompt)
    decoded = [tokenizer.decode([t]) for t in tokens]
    logger.debug(f"Tokenized prompt: {decoded}")
    
    # Find word indices
    word_indices = []
    for idx, token_str in enumerate(decoded):
        if word in token_str:
            word_indices.append(idx)
            logger.debug(f"Found '{word}' at token index {idx}: '{token_str}'")
    
    if not word_indices:
        logger.warning(f"Word '{word}' not found in tokenized prompt")
        return None, None, None

    logger.info(f"Target word found at {len(word_indices)} position(s): {word_indices}")
    
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

    # Process maps for specific word tokens
    logger.info(f"Processing attention maps for word '{word}'...")
    aggregated_heatmap = process_attention_maps(
        hook_manager.attention_maps, 
        image.size,
        token_indices=word_indices
    )

    # Thresholding
    max_val = aggregated_heatmap.max()
    threshold = tau * max_val
    binary_mask = (aggregated_heatmap > threshold).float()
    
    logger.info(
        f"Heatmap created: max={max_val:.4f}, "
        f"threshold={threshold:.4f}, "
        f"mask pixels={binary_mask.sum().item():.0f}"
    )
    
    return image, aggregated_heatmap, binary_mask


def main():
    """Main execution function for testing."""
    from diffusers import DiffusionPipeline
    
    logger.info("="*60)
    logger.info("Forward DAAM - Test Execution")
    logger.info("="*60)
    
    # Setup
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Using device: {device}")
    logger.info(f"Loading model: {DEFAULT_MODEL_ID}")
    
    pipe = DiffusionPipeline.from_pretrained(DEFAULT_MODEL_ID, torch_dtype=torch.float16)
    pipe.to(device)
    
    # Test
    prompt = "A cat"
    word = "cat"
    logger.info(f"Testing with prompt='{prompt}', word='{word}'")
    
    img, heat, mask = run_forward_daam(pipe, prompt, word)
    
    if img:
        logger.info("✓ Test run successful")
    else:
        logger.error("✗ Test run failed")


if __name__ == "__main__":
    main()
