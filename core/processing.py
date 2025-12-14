"""
Core Processing Module

Functions for processing and aggregating attention maps.
"""

# Standard library imports
import logging
from typing import Dict, List, Tuple, Optional

# Third-party imports
import torch
import numpy as np
import torch.nn.functional as F

# Local imports
from .constants import SD_CONTEXT_LENGTH
from ..utils.logging import get_daam_logger

# Module logger
logger = get_daam_logger(__name__)


def process_attention_maps(
    attention_maps: Dict[str, List[torch.Tensor]], 
    image_size: Tuple[int, int],
    token_indices: Optional[List[int]] = None
) -> torch.Tensor:
    """
    Process raw attention maps into a global attention map.
    
    This function aggregates attention maps from all layers and timesteps,
    resizes them to match the image size, and optionally filters for specific tokens.
    
    Args:
        attention_maps: Dictionary mapping layer names to lists of attention tensors
        image_size: (W, H) tuple of target image size
        token_indices: Optional list of token indices to extract. If None, returns all tokens.
        
    Returns:
        torch.Tensor: Aggregated attention map
            - If token_indices is None: shape (H, W, 77) with all token scores
            - If token_indices provided: shape (H, W) with averaged scores for specified tokens
    """
    W, H = image_size
    
    if token_indices is None:
        global_map = torch.zeros((H, W, SD_CONTEXT_LENGTH), dtype=torch.float32)
        logger.debug(f"Processing attention maps for all {SD_CONTEXT_LENGTH} tokens")
    else:
        global_map = torch.zeros((H, W), dtype=torch.float32)
        logger.debug(f"Processing attention maps for {len(token_indices)} specific tokens")
    
    count = 0
    
    for layer_name, maps_list in attention_maps.items():
        for attn_map in maps_list:
            bs_heads = attn_map.shape[0]
            num_heads = bs_heads // 2
            
            # Take conditional part
            cond_map = attn_map[num_heads:].mean(dim=0)
            
            hw = cond_map.shape[0]
            h_layer = int(np.sqrt(hw))
            
            if token_indices is not None:
                # Extract specific tokens and average
                cond_map = cond_map[:, token_indices].mean(dim=-1)
                cond_map = cond_map.view(1, 1, h_layer, h_layer)
                resized_map = F.interpolate(cond_map, size=(H, W), mode='bicubic', align_corners=False)
                global_map += resized_map.squeeze()
            else:
                # Keep all tokens
                context_len = cond_map.shape[1]
                cond_map = cond_map.permute(1, 0).view(1, context_len, h_layer, h_layer)
                resized_map = F.interpolate(cond_map, size=(H, W), mode='bicubic', align_corners=False)
                global_map += resized_map.squeeze().permute(1, 2, 0)
            
            count += 1
    
    if count > 0:
        global_map /= count
        logger.debug(f"Processed {count} attention maps")
    else:
        logger.warning("No attention maps to process")
    
    return global_map


def extract_bbox_distribution(
    global_map: torch.Tensor,
    bbox: List[int],
    decoded_tokens: List[str]
) -> Tuple[np.ndarray, List[str]]:
    """
    Extract token distribution within a bounding box.
    
    Args:
        global_map: Attention map of shape (H, W, 77)
        bbox: Bounding box as [x_min, y_min, x_max, y_max]
        decoded_tokens: List of decoded token strings
        
    Returns:
        Tuple of (token_scores, tokens_plot):
            - token_scores: numpy array of attention scores (excluding start/end tokens)
            - tokens_plot: List of token strings (excluding start/end tokens)
    """
    H, W = global_map.shape[0], global_map.shape[1]
    x_min, y_min, x_max, y_max = bbox
    
    # Clamp coordinates
    x_min = max(0, x_min)
    y_min = max(0, y_min)
    x_max = min(W, x_max)
    y_max = min(H, y_max)
    
    logger.debug(f"Extracting distribution from bbox: [{x_min}, {y_min}, {x_max}, {y_max}]")
    
    # Extract region and average
    crop = global_map[y_min:y_max, x_min:x_max, :]
    token_scores = crop.mean(dim=(0, 1)).numpy()
    
    # Slice to actual tokens
    token_scores = token_scores[:len(decoded_tokens)]
    
    # Remove start/end tokens
    if len(decoded_tokens) > 2:
        token_scores = token_scores[1:-1]
        tokens_plot = decoded_tokens[1:-1]
        logger.debug(f"Removed start/end tokens, {len(tokens_plot)} remaining")
    else:
        tokens_plot = decoded_tokens
    
    return token_scores, tokens_plot
