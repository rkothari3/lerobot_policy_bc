"""
Configuration for BC (Behavior Cloning) Policy
"""

from dataclasses import dataclass, field
from typing import Dict, Any, Optional


@dataclass
class BCConfig:
    """
    Configuration class for BC (Behavior Cloning) policy.
    
    This class holds all the hyperparameters and settings for the behavior cloning policy.
    """
    
    # Model architecture parameters
    input_channels: int = 3
    hidden_dim: int = 256
    num_layers: int = 4
    
    # Training parameters
    learning_rate: float = 1e-4
    batch_size: int = 32
    num_epochs: int = 100
    
    # Policy parameters
    action_dim: int = 7
    observation_dim: Optional[int] = None
    
    # Additional config
    use_dropout: bool = True
    dropout_rate: float = 0.1
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return {
            "input_channels": self.input_channels,
            "hidden_dim": self.hidden_dim,
            "num_layers": self.num_layers,
            "learning_rate": self.learning_rate,
            "batch_size": self.batch_size,
            "num_epochs": self.num_epochs,
            "action_dim": self.action_dim,
            "observation_dim": self.observation_dim,
            "use_dropout": self.use_dropout,
            "dropout_rate": self.dropout_rate,
        }
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "BCConfig":
        """Create configuration from dictionary."""
        return cls(**config_dict)
