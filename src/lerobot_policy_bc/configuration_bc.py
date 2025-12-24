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
        input_normalization_modes: How to normalize inputs
        output_normalization_modes: How to normalize outputs
    """
    n_obs_steps: int = 2
    horizon: int = 16
    n_action_steps: int = 8

    hidden_dim: int = 512
    input_shapes: dict = field(default_factory=lambda: {"observation.image": [3, 96, 96]})
    output_shapes: dict = field(default_factory=lambda: {"action": [2]})

    input_normalization_modes: dict = field(default_factory=lambda: {"observation.image": NormalizationMode.MEAN_STD})
    output_normalization_modes: dict = field(default_factory=lambda: {"action": NormalizationMode.MEAN_STD})

    """
    Runs after dataclass initializes. Used for validation and for computing derived values.
    """
    def __post_init__(self):
        super().__post_init__()
        # Check that n_action_steps doesn't exceed horizon
        if self.n_action_steps > self.horizon:
            raise ValueError(f"n_action_steps ({self.n_action_steps}) cannot exceed horizon ({self.horizon})")
        
    def validate_features(self) -> None:
        """Validate input/output feature compatibility."""
        # Check we have image input
        if "observation.image" not in self.input_shapes:
            raise ValueError("BC policy requires 'observation.image' input")
        
        # Check we have action output
        if "action" not in self.output_shapes:
            raise ValueError("BC policy requires 'action' output")
