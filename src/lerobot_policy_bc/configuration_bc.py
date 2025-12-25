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
    
    @property
    def observation_delta_indices(self) -> list:
        """
        Delta indices for observations relative to current timestep.
        For n_obs_steps=2, we want frames at t-1 and t (current).
        Returns: list like [-1, 0] for n_obs_steps=2
        """
        return list(range(1 - self.n_obs_steps, 1))
    
    @property
    def action_delta_indices(self) -> list:
        """
        Delta indices for actions relative to current timestep.
        For horizon=16, we predict actions from t to t+15.
        Returns: list like [0, 1, 2, ..., 15] for horizon=16
        """
        return list(range(self.horizon))
    
    @property
    def reward_delta_indices(self) -> list:
        """
        Delta indices for rewards. BC doesn't use rewards, so return empty list.
        """
        return []
    
    def get_optimizer_preset(self) -> OptimizerPreset:
        """
        Return optimizer configuration.
        Using AdamW with standard learning rate for vision-based policies.
        """
        return OptimizerPreset(
            type="AdamW",
            lr=1e-4,
            betas=(0.9, 0.999),
            eps=1e-8,
            weight_decay=1e-6,
        )
    
    def get_scheduler_preset(self) -> SchedulerPreset:
        """
        Return learning rate scheduler configuration.
        Using cosine annealing with warmup.
        """
        return SchedulerPreset(
            type="cosine",
            warmup_steps=500,
        )







