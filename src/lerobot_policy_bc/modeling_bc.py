# modeling_bc.py
import torch
import torch.nn as nn
from typing import Dict, Any
from lerobot.policies.pretrained import PreTrainedPolicy
from lerobot.configs.types import FeatureType
from .configuration_bc import BCConfig


class BC(PreTrainedPolicy):
    config_class = BCConfig
    name = "bc"

    # Model architecture
    # Added **kwargs bc factory passes extra args
    def __init__(self, config: BCConfig, dataset_stats=None, **kwargs):
        super().__init__(config, dataset_stats)
        self.image_keys = [
            key for key, feat in self.config.input_features.items()
            if feat.type == FeatureType.VISUAL
        ]
        # State (robot/agent pos, joint angles, etc.)
        self.state_keys = [
            key for key, feat in self.config.input_features.items()
            if feat.type == FeatureType.STATE
        ]

        if not self.image_keys:
            raise ValueError("BC Policy requires at least one img input")

        self.output_dim = self.config.output_features["action"].shape[0]

        # CNN encoder
        self.cnn_encoder = self._build_cnn_encoder()

        with torch.no_grad():
            dummy_input = torch.zeros(1, 3, 96, 96)  # for pusht 96 x96 img
            cnn_output = self.cnn_encoder(dummy_input)
            cnn_output_dim = cnn_output.view(1, -1).shape[1]

        # Calc total feature dimension
        # Vision: cnn_output_dim * num_cameras * n_obs_steps
        # State: state_dim * n_obs_steps
        state_dim = sum(
            self.config.input_features[key].shape[0]
            for key in self.state_keys
        )
        total_feature_dim = (
            cnn_output_dim * len(self.image_keys) * self.config.n_obs_steps +
            state_dim * self.config.n_obs_steps
        )
        # MLP Action head
        self.mlp = nn.Sequential(
            nn.Linear(total_feature_dim, self.config.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.config.hidden_dim, self.config.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.config.hidden_dim, self.output_dim * self.config.horizon)
        )
        self.reset()

    def _build_cnn_encoder(self):
        """Build CNN encoder for visual observations."""
        return nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Flatten()
        )

    def _extract_features(self, batch):
        """Extract and process visual and state features from batch."""
        B = None
        all_features = []

        # Process images
        for key in self.image_keys:
            img = batch[key]
            if B is None:
                B = img.shape[0]

            # Handle shape variations
            if img.dim() == 4:  # [B, C, H, W]
                img = img.unsqueeze(1)  # [B, 1, C, H, W]

            # Ensure exactly n_obs_steps
            if img.shape[1] < self.config.n_obs_steps:
                padding_needed = self.config.n_obs_steps - img.shape[1]
                last_obs = img[:, -1:].repeat(1, padding_needed, 1, 1, 1)
                img = torch.cat([img, last_obs], dim=1)
            elif img.shape[1] > self.config.n_obs_steps:
                img = img[:, -self.config.n_obs_steps:]

            # Encode through CNN
            T, C, H, W = img.shape[1:]
            img_flat = img.reshape(B * T, C, H, W)
            encoded = self.cnn_encoder(img_flat)
            encoded = encoded.reshape(B, T, -1)
            all_features.append(encoded)

        # Process state features
        for key in self.state_keys:
            state = batch[key]

            # Handle shape variations
            if state.dim() == 2:  # [B, state_dim]
                state = state.unsqueeze(1)  # [B, 1, state_dim]

            # Ensure exactly n_obs_steps
            if state.shape[1] < self.config.n_obs_steps:
                padding_needed = self.config.n_obs_steps - state.shape[1]
                last_state = state[:, -1:].repeat(1, padding_needed, 1)
                state = torch.cat([state, last_state], dim=1)
            elif state.shape[1] > self.config.n_obs_steps:
                state = state[:, -self.config.n_obs_steps:]

            all_features.append(state)

        # Concatenate all features and flatten
        features = torch.cat(all_features, dim=-1)  # [B, T, total_features]
        features = features.reshape(B, -1)  # [B, T * total_features]
        return features

    # Forward: method defines how to use your model.
    def forward(self, batch, reduction="mean"):
        """Forward pass for training."""
        # Extract all features
        features = self._extract_features(batch)

        # Predict action sequence
        B = features.shape[0]
        action_pred = self.mlp(features)
        action_pred = action_pred.reshape(B, self.config.horizon, self.output_dim)

        # Compute loss
        action_target = batch["action"][:, :self.config.horizon]

        if reduction == "none":
            loss = nn.functional.mse_loss(action_pred, action_target, reduction='none')
            loss = loss.mean(dim=(1, 2))  # [B]
            return loss, {"action_pred": action_pred}
        else:
            loss = nn.functional.mse_loss(action_pred, action_target)
            return loss, {"action_pred": action_pred}

    def compute_loss(self, batch):
        """Compute loss (called by training loop)."""
        loss, _ = self.forward(batch)
        return loss

    def get_optim_params(self):
        """Return parameters for optimization."""
        return self.parameters()

    def reset(self):
        """Reset policy state. BC is stateless so this is a no-op."""
        pass

    def predict_action_chunk(self, batch):
        """Predict actions for inference/evaluation."""
        with torch.no_grad():
            features = self._extract_features(batch)
            B = features.shape[0]

            # Predict full action horizon
            action_pred = self.mlp(features)
            action_pred = action_pred.reshape(B, self.config.horizon, self.output_dim)

            # Return only n_action_steps for execution
            return action_pred[:, :self.config.n_action_steps]

    def select_action(self, batch):
        """Select single action for execution."""
        action_chunk = self.predict_action_chunk(batch)
        return action_chunk[:, 0]  # [B, action_dim]