"""
Core Hooks Module

Provides AttentionHookManager for extracting attention maps from Stable Diffusion models.
"""

# Standard library imports
import logging
from typing import Dict, List

# Third-party imports
import torch

# Local imports  
from .constants import SD_CONTEXT_LENGTH
from ..utils.logging import get_daam_logger

# Module logger
logger = get_daam_logger(__name__)


class AttentionHookManager:
    """
    Manages attention map extraction from Stable Diffusion models.
    
    This class provides a centralized way to register hooks on cross-attention
    layers and collect attention maps during image generation.
    """
    
    def __init__(self):
        """Initialize the hook manager with empty attention map storage."""
        self.attention_maps: Dict[str, List[torch.Tensor]] = {}
        logger.debug("AttentionHookManager initialized")
    
    def get_hook(self, layer_name: str):
        """
        Creates an attention hook function for a specific layer.
        
        Args:
            layer_name: Unique identifier for the layer
            
        Returns:
            Hook function that can be registered with register_forward_hook
        """
        def hook(module, args, kwargs, output):
            try:
                # Extract hidden states and encoder hidden states
                if len(args) > 0:
                    hidden_states = args[0]
                else:
                    hidden_states = kwargs.get('hidden_states')

                if len(args) > 1:
                    encoder_hidden_states = args[1]
                else:
                    encoder_hidden_states = kwargs.get('encoder_hidden_states')
                
                if hidden_states is None or encoder_hidden_states is None:
                    return

                # Compute attention scores
                head_dim = module.to_q.out_features // module.heads
                scale = head_dim ** -0.5
                
                query = module.head_to_batch_dim(module.to_q(hidden_states))
                key = module.head_to_batch_dim(module.to_k(encoder_hidden_states))
                
                attention_scores = torch.baddbmm(
                    torch.empty(query.shape[0], query.shape[1], key.shape[1], 
                               dtype=query.dtype, device=query.device).zero_(),
                    query,
                    key.transpose(-1, -2),
                    beta=0,
                    alpha=scale,
                )
                
                attention_probs = attention_scores.softmax(dim=-1)
                
                # Store attention maps
                if layer_name not in self.attention_maps:
                    self.attention_maps[layer_name] = []
                self.attention_maps[layer_name].append(attention_probs.detach().cpu())
                
            except Exception as e:
                logger.error(f"Error in hook for {layer_name}: {e}")
        
        return hook
    
    def register_hooks(self, unet) -> List:
        """
        Register hooks on all cross-attention layers in the UNet.
        
        Args:
            unet: The UNet model from Stable Diffusion pipeline
            
        Returns:
            List of hook handles that can be used to remove hooks later
        """
        hook_handles = []
        for name, module in unet.named_modules():
            if "attn2" in name and module.__class__.__name__ == "Attention":
                handle = module.register_forward_hook(
                    self.get_hook(name), 
                    with_kwargs=True
                )
                hook_handles.append(handle)
        
        logger.debug(f"Registered {len(hook_handles)} attention hooks")
        return hook_handles
    
    def reset(self):
        """Clear all stored attention maps."""
        self.attention_maps = {}
        logger.debug("Attention maps reset")


def remove_hooks(hook_handles: List):
    """
    Safely remove all registered hooks.
    
    Args:
        hook_handles: List of hook handles returned by register_hooks
    """
    for handle in hook_handles:
        handle.remove()
    logger.debug(f"Removed {len(hook_handles)} hooks")
