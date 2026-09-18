"""
SARGCL — State-Aware Hypergraph Representation Learning
with Difficulty-Regulated Contrastive Alignment for Multimodal Understanding.

Authors: Shahid Shafi Dar, Arnav Jain, Nagendra Kumar
         IIT Indore
"""

__version__ = "1.0.0"

from .model import SARGCL
from .dataset import DisasterDataset, collate_fn
from .evaluate import evaluate_loop

__all__ = [
    "SARGCL",
    "DisasterDataset",
    "collate_fn",
    "evaluate_loop",
]
