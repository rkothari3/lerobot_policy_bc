"""
LeRobot BC (Behavior Cloning) Policy

A CNN-based behavior cloning policy implementation for LeRobot.
"""

from .configuration_bc import BCConfig
from .modeling_bc import BCPolicy
from .processor_bc import BCProcessor

__all__ = ["BCConfig", "BCPolicy", "BCProcessor"]

__version__ = "0.1.0"
