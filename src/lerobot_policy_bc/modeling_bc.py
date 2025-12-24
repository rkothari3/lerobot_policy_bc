# modeling_bc.py
import torch
import torch.nn as nn
from typing import Dict, Any

from lerobot.policies.pretrained import PreTrainedPolicy
from .configuration_bc import BCPolicyConfig

class BCPolicy(PreTrainedPolicy):
    config_class = BCPolicyConfig
    name = "bc"

    # Model architecture
    def __init__(self, config: BCPolicyConfig, dataset_stats: Dict[str, Any] = None):
        super().__init__(config, dataset_stats)
        # CNN encoder
        self.cnn_encoder = nn.Sequential(
            # Layer 1: 3 channels -> 32 channels
            nn.Conv2d(in_channels=3 * config.n_obs_steps, out_channels=32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            # Layer 2: 32 channels -> 64 channels
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            # Layer 3: 64 channels -> 128 channels
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            # Layer 4: 128 channels -> 256 channels
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),

            # Flatten to vector 256 * 6 * 6 = 9216
            nn.Flatten(), 
        )
        # MLP Action head
        self.action_head = nn.Sequential(
            # Layer 1: Compress CNN features (9216) to hidden_dim (512)
            nn.Linear(in_features=9216, out_features=config.hidden_dim),
            nn.ReLU(), 
            # Layer 2: hidden_dim in and out; processes features
            nn.Linear(in_features=config.hidden_dim, out_features=config.hidden_dim),
            nn.ReLU(),
            # Layer 3: Output layer - predicts all actions
            nn.Linear(in_features=config.hidden_dim, out_features=config.horizon * 2)
            # No activation here! We want to predict raw action values.
        )
    
    # Forward: method defines how to use your model.
    def forward(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        # 1. Extract images
        images = batch["observation.image"]  # [B, n_obs_steps, C, H, W]
        
        # 2. Get batch size dynamically (don't hardcode!)
        batch_size = images.shape[0]
        
        # 3. Reshape: merge n_obs_steps with channels
        # [B, n_obs_steps, C, H, W] → [B, n_obs_steps*C, H, W]
        images = images.reshape(
            batch_size,
            self.config.n_obs_steps * 3,  # 2 * 3 = 6 channels
            96, 
            96
        )
        # Now: [B, 6, 96, 96]
        
        # 4. Pass through CNN encoder
        features = self.cnn_encoder(images)  # [B, 9216]
        
        # 5. Pass through action head
        actions = self.action_head(features)  # [B, horizon * 2] = [B, 32]
        
        # 6. Reshape to [B, horizon, action_dim]
        actions = actions.view(batch_size, self.config.horizon, 2)  # [B, 16, 2]
        
        # 7. Return as dictionary
        return {"action": actions}
    
    # Compute loss: computes the loss when predicting actions and comparing it with true actions.
    def compute_loss(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        # 1. Get predictions from forward pass
        predictions = self.forward(batch)  # Returns {"action": [B, horizon, 2]}
        predicted_actions = predictions["action"]  # [B, 16, 2]
        
        # 2. Get ground truth actions
        true_actions = batch["action"]  # [B, 16, 2]
        
        # 3. Calculate MSE loss
        loss = nn.functional.mse_loss(predicted_actions, true_actions)
        
        return loss
