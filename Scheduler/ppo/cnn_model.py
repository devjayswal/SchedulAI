"""
CNN-based Policy Architecture for Timetable Scheduling
This module provides a CNN-based policy that can better capture spatial relationships
in the timetable grid compared to simple MLP networks.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from gymnasium.spaces import Box
import numpy as np


class TimetableCNNExtractor(BaseFeaturesExtractor):
    """
    CNN feature extractor for timetable scheduling.
    Treats the timetable as a 2D grid where spatial relationships matter.
    """
    
    def __init__(self, observation_space: Box, features_dim: int = 256):
        """
        Initialize the CNN extractor.
        
        Args:
            observation_space: The observation space from the environment
            features_dim: The dimension of the output features
        """
        super(TimetableCNNExtractor, self).__init__(observation_space, features_dim)
        
        # Get dimensions from observation space
        # observation_space.shape = (num_flat_slots * num_classrooms,)
        total_slots = observation_space.shape[0]
        
        # We need to reshape this into a 2D grid
        # For now, we'll assume a reasonable grid size and pad if necessary
        # You can adjust these based on your specific timetable dimensions
        self.grid_height = 6  # Number of time slots per day
        self.grid_width = 8   # Number of classrooms (can be adjusted)
        
        # Calculate padding if needed
        self.padded_size = self.grid_height * self.grid_width
        if total_slots > self.padded_size:
            # If we have more slots than our grid, we need to increase grid size
            self.grid_width = (total_slots + self.grid_height - 1) // self.grid_height
            self.padded_size = self.grid_height * self.grid_width
        
        # CNN layers for feature extraction
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)  # 1 input channel (timetable grid)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        
        # Batch normalization for better training stability
        self.bn1 = nn.BatchNorm2d(32)
        self.bn2 = nn.BatchNorm2d(64)
        self.bn3 = nn.BatchNorm2d(128)
        
        # Dropout for regularization
        self.dropout = nn.Dropout2d(0.2)
        
        # Calculate the size after convolutions
        # After 3 conv layers with padding=1, the spatial dimensions remain the same
        conv_output_size = 128 * self.grid_height * self.grid_width
        
        # Fully connected layers to reduce to desired feature dimension
        self.fc1 = nn.Linear(conv_output_size, 512)
        self.fc2 = nn.Linear(512, features_dim)
        
        # Additional dropout for FC layers
        self.fc_dropout = nn.Dropout(0.3)
        
    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the CNN extractor.
        
        Args:
            observations: Input tensor of shape (batch_size, num_flat_slots * num_classrooms)
            
        Returns:
            Feature tensor of shape (batch_size, features_dim)
        """
        batch_size = observations.shape[0]
        
        # Reshape observations to 2D grid
        # Pad or truncate to fit our grid dimensions
        if observations.shape[1] < self.padded_size:
            # Pad with zeros
            padding = torch.zeros(batch_size, self.padded_size - observations.shape[1], 
                                device=observations.device, dtype=observations.dtype)
            observations = torch.cat([observations, padding], dim=1)
        elif observations.shape[1] > self.padded_size:
            # Truncate
            observations = observations[:, :self.padded_size]
        
        # Reshape to 2D grid: (batch_size, 1, height, width)
        x = observations.view(batch_size, 1, self.grid_height, self.grid_width)
        
        # Normalize the input to [0, 1] range for better CNN performance
        x = x.float() / (x.max() + 1e-8)  # Add small epsilon to avoid division by zero
        
        # First convolutional block
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.dropout(x)
        
        # Second convolutional block
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.dropout(x)
        
        # Third convolutional block
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.dropout(x)
        
        # Flatten for fully connected layers
        x = x.view(batch_size, -1)
        
        # Fully connected layers
        x = F.relu(self.fc1(x))
        x = self.fc_dropout(x)
        x = F.relu(self.fc2(x))
        
        return x


class TimetableCNNActorCritic(nn.Module):
    """
    CNN-based Actor-Critic network for PPO.
    This combines the CNN feature extractor with actor and critic heads.
    """
    
    def __init__(self, observation_space: Box, action_space, features_dim: int = 256):
        """
        Initialize the CNN Actor-Critic network.
        
        Args:
            observation_space: The observation space from the environment
            action_space: The action space from the environment
            features_dim: The dimension of the features from the CNN extractor
        """
        super(TimetableCNNActorCritic, self).__init__()
        
        # CNN feature extractor
        self.features_extractor = TimetableCNNExtractor(observation_space, features_dim)
        
        # Actor head (policy network)
        self.action_net = nn.Sequential(
            nn.Linear(features_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, action_space.nvec[0])  # Output size for course selection
        )
        
        # Critic head (value network)
        self.value_net = nn.Sequential(
            nn.Linear(features_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1)  # Single value output
        )
        
    def forward(self, observations: torch.Tensor) -> tuple:
        """
        Forward pass through the actor-critic network.
        
        Args:
            observations: Input tensor of shape (batch_size, observation_dim)
            
        Returns:
            Tuple of (action_logits, value)
        """
        # Extract features using CNN
        features = self.features_extractor(observations)
        
        # Get action logits and value
        action_logits = self.action_net(features)
        value = self.value_net(features)
        
        return action_logits, value


def create_cnn_policy_kwargs(observation_space: Box, action_space, features_dim: int = 256):
    """
    Create policy kwargs for CNN-based PPO.
    
    Args:
        observation_space: The observation space from the environment
        action_space: The action space from the environment
        features_dim: The dimension of the features from the CNN extractor
        
    Returns:
        Dictionary of policy kwargs for PPO
    """
    return {
        'features_extractor_class': TimetableCNNExtractor,
        'features_extractor_kwargs': {'features_dim': features_dim},
        'net_arch': [],  # We're using custom CNN, so no additional layers needed
        'activation_fn': torch.nn.ReLU,
        'normalize_images': False,  # We handle normalization in the CNN
    }


def create_cnn_hyperparams():
    """
    Create hyperparameters optimized for CNN-based PPO.
    
    Returns:
        Dictionary of hyperparameters
    """
    return {
        'learning_rate': 1e-4,  # Lower learning rate for CNN
        'n_steps': 2048,        # Larger buffer for better gradient estimates
        'batch_size': 128,      # Larger batch size for CNN training
        'n_epochs': 10,         # More epochs for CNN training
        'gamma': 0.99,
        'gae_lambda': 0.95,
        'clip_range': 0.2,
        'ent_coef': 0.01,
        'vf_coef': 0.5,         # Value function coefficient
        'max_grad_norm': 0.5,
        'tensorboard_log': "./tb_logs/",
        'device': "auto",       # Automatically use GPU if available, fallback to CPU
        'verbose': 1
    }


# Example usage function
def create_cnn_model(env, model_path="cnn_ppo_timetable.pth", **kwargs):
    """
    Create a CNN-based PPO model.
    
    Args:
        env: The environment
        model_path: Path to save/load the model
        **kwargs: Additional arguments for PPO
        
    Returns:
        CNN-based PPO model
    """
    from sb3_contrib import MaskablePPO
    
    # Get policy kwargs
    policy_kwargs = create_cnn_policy_kwargs(
        env.observation_space, 
        env.action_space,
        features_dim=kwargs.get('features_dim', 256)
    )
    
    # Get hyperparameters
    hyperparams = create_cnn_hyperparams()
    hyperparams.update(kwargs)
    
    # Ensure proper logging configuration
    if 'tensorboard_log' not in hyperparams:
        hyperparams['tensorboard_log'] = "./tb_logs/"
    if 'verbose' not in hyperparams:
        hyperparams['verbose'] = 1
    
    # Create model with proper policy
    model = MaskablePPO(
        "MlpPolicy",  # We'll override the feature extractor
        env,
        policy_kwargs=policy_kwargs,
        **hyperparams
    )
    
    # Initialize logger if not already done
    if not hasattr(model, 'logger') or model.logger is None:
        from stable_baselines3.common.logger import configure
        log_dir = hyperparams.get('tensorboard_log', "./tb_logs/")
        # Ensure log directory exists
        import os
        os.makedirs(log_dir, exist_ok=True)
        model.set_logger(configure(log_dir))
    
    return model
