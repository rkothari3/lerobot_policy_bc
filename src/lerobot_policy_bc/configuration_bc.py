# configuration_bc.py
from dataclasses import dataclass, field
from lerobot.configs.policies import PreTrainedConfig
from lerobot.configs.types import NormalizationMode

@PreTrainedConfig.register_subclass("bc")
@dataclass
class BCPolicyConfig(PreTrainedConfig):
    """
    Configuration class for BC

    Args:
        n_obs_steps: How many past observations to look at.
        horizon: How many future actions to predict at once.
        n_action_steps: How many of the predicted actions to execute.
        hidden_dim: Size of internal network layers.
        input_shapes: Image dimensions (CNN-specific)
        output_shapes: Action dimensions (CNN-specific)
    """
    
