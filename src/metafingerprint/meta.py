"""Backward-compatible aliases for structural-subspace adaptation."""
from .adaptation import adapt_structural_branch, first_order_meta_step, freeze_except_structural

__all__ = ["adapt_structural_branch", "first_order_meta_step", "freeze_except_structural"]
