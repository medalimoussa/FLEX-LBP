"""FLEX-LBP: learnable local binary patterns with exact binary inference (PyTorch)."""
from .layer import BRANCHES, PAIRS, PRESETS, TEMPLATES, ColorTransform, FLEXLBP, FLEXLBPBranch
from .losses import ProxyCosineLoss, flex_loss, geometry_regularizer, threshold_regularizer
from .postprocess import postprocess, power_normalize, standardized_1nn, uniform_u2, uniform_u2_mapping

__version__ = "0.1.0"
__all__ = ["FLEXLBP", "FLEXLBPBranch", "ColorTransform", "PRESETS", "TEMPLATES", "PAIRS", "BRANCHES",
           "ProxyCosineLoss", "flex_loss", "geometry_regularizer", "threshold_regularizer",
           "postprocess", "power_normalize", "uniform_u2", "uniform_u2_mapping", "standardized_1nn"]
