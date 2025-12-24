# processor_bc.py
from typing import Dict, Any
import torch


def make_my_custom_policy_pre_post_processors(
    config,
) -> tuple[
    PolicyProcessorPipeline[dict[str, Any], dict[str, Any]],
    PolicyProcessorPipeline[PolicyAction, PolicyAction],
]:
    """Create preprocessing and postprocessing functions for your policy."""
    pass  # Define your preprocessing and postprocessing logic here
