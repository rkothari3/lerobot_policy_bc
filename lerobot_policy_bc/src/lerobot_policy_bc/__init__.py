# __init__.py
"""BC package for LeRobot."""

# Checks if lerobot is installed
try:
    import lerobot  # noqa: F401
except ImportError:
    raise ImportError(
        "lerobot is not installed. Please install lerobot to use this policy package."
    )

# Imports policy classes so others can use
from .configuration_bc import BCConfig
from .modeling_bc import BC
from .processor_bc import make_bc_pre_post_processors

# Declares what's public via __all__ list
__all__ = ["BC", "BCConfig", "make_bc_pre_post_processors"] 