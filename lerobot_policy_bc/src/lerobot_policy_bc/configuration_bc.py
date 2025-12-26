# configuration_bc.py
from dataclasses import dataclass, field
from lerobot.configs.policies import PreTrainedConfig
from lerobot.configs.types import NormalizationMode
from lerobot.optim.optimizers import AdamConfig
from lerobot.optim.schedulers import DiffuserSchedulerConfig

@PreTrainedConfig.register_subclass("bc")
@dataclass
class BCConfig(PreTrainedConfig):
    """
    Configuration class for BC

    Args:
        n_obs_steps: How many past observations to look at.
        horizon: How many future actions to predict at once.
        n_action_steps: How many of the predicted actions to execute.
        hidden_dim: Size of internal network layers.
        normalization_mapping: Dict mapping feature types to normalization modes.
        optimizer_lr: Learning rate for the optimizer.
        optimizer_betas: Betas for the Adam optimizer.
        optimizer_eps: Epsilon for the Adam optimizer.
        optimizer_weight_decay: Weight decay for the Adam optimizer.
        scheduler_name: Name of the learning rate scheduler.
        scheduler_warmup_steps: Number of warmup steps for the scheduler.
    """
    n_obs_steps: int = 2
    horizon: int = 16
    n_action_steps: int = 8

    hidden_dim: int = 512

    # Removed input/output feature names; lerobot handles this in factory.py
    # Replaced normalization modes
    normalization_mapping: dict[str, NormalizationMode] = field(
    default_factory=lambda: {
            "VISUAL": NormalizationMode.MEAN_STD,   # All images
            "ACTION": NormalizationMode.MIN_MAX,     # All actions
        }
    )
    
    # Configs for the training process
    # Training loop won't know how to optimize the model without these.
    optimizer_lr: float = 1e-4
    optimizer_betas: tuple = (0.95, 0.999)
    optimizer_eps: float = 1e-8
    optimizer_weight_decay: float = 1e-6
    scheduler_name: str = "cosine"
    scheduler_warmup_steps: int = 500

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
        if len(self.image_features) == 0:
            raise ValueError("BC policy requires at least one image input")
        
        # Check we have action output
        if "action" not in self.output_shapes:
            raise ValueError("BC policy requires 'action' output")
    
    # Implement abstract methods defined in PreTrainedConfig
    # Else, we get typerror
    def get_optimizer_preset(self) -> AdamConfig:
        """
        Returns optimizer config for the training loop
        """
        return AdamConfig(
            lr=self.optimizer_lr,
            betas=self.optimizer_betas,
            eps=self.optimizer_eps,
            weight_decay=self.optimizer_weight_decay,
        )

    def get_scheduler_preset(self) -> DiffuserSchedulerConfig:
        """
        Returns the lr scheduler config
        """
        return DiffuserSchedulerConfig(
            name=self.scheduler_name,
            num_warmup_steps=self.scheduler_warmup_steps,
        )
    
    # Temporal Index properties
    # Without, training loop wouldn't know which temporal frames to load.
    @property
    def observation_delta_indices(self) -> list:
        """Which observation frames to use relative to current timestep."""
        return list(range(1 - self.n_obs_steps, 1))

    @property
    def action_delta_indices(self) -> list:
        """Which action frames to predict relative to current timestep."""
        return list(range(1 - self.n_obs_steps, 1 - self.n_obs_steps + self.horizon))

    @property
    def reward_delta_indices(self) -> None:
        """BC doesn't use rewards."""
        return None

    
    
