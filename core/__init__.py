"""Core module for attention hook management and processing."""

from .hooks import AttentionHookManager, remove_hooks
from .processing import process_attention_maps, extract_bbox_distribution
from .constants import *

__all__ = [
    "AttentionHookManager",
    "remove_hooks",
    "process_attention_maps",
    "extract_bbox_distribution",
]
