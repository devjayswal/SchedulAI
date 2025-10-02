"""
CNN-optimized hyperparameters for PPO training
These hyperparameters are specifically tuned for CNN-based architectures
"""

import torch
from .cnn_model import create_cnn_policy_kwargs, create_cnn_hyperparams

# CNN-specific policy configuration
def get_cnn_policy_kwargs(observation_space, action_space, features_dim=256):
    """Get CNN policy kwargs optimized for timetable scheduling."""
    return create_cnn_policy_kwargs(observation_space, action_space, features_dim)

# CNN-optimized PPO hyperparameters
cnn_ppo_kwargs = dict(
    learning_rate=1e-4,        # Lower learning rate for CNN stability
    n_steps=2048,             # Larger buffer for better gradient estimates
    batch_size=128,           # Larger batch size for CNN training
    n_epochs=10,              # More epochs for CNN training
    gamma=0.99,
    gae_lambda=0.95,
    clip_range=0.2,
    ent_coef=0.01,
    vf_coef=0.5,              # Value function coefficient
    max_grad_norm=0.5,
    tensorboard_log="./tb_logs/",
    device="auto",             # Automatically use GPU if available, fallback to CPU
    verbose=1
)

# CNN training configuration
cnn_train_config = {
    "total_timesteps": 1000000,  # More timesteps for CNN training
    "log_interval": 2000,        # Less frequent logging for CNN
    "eval_freq": 10000,          # Less frequent evaluation
    "best_model_save_path": "models/cnn_best_model",
    "log_path": "logs/cnn/",
}

# CNN environment configuration (same as MLP but with CNN-specific settings)
cnn_env_config = {
    "num_courses": 5,
    "num_slots": 6,             # Keep original constraints
    "num_classrooms": 5,        # Increased from 3 to reduce constraint conflicts
    "n_envs": 2,                # Reduced for CNN (more memory intensive)
    "features_dim": 256,        # CNN feature dimension
    "grid_height": 6,           # Time slots per day
    "grid_width": 8,            # Classrooms (can be adjusted)
}

# Comparison hyperparameters (for benchmarking)
mlp_policy_kwargs = dict(
    net_arch=dict(pi=[64, 64], vf=[64, 64]),
    activation_fn=torch.nn.ReLU
)

mlp_ppo_kwargs = dict(
    learning_rate=3e-4,         # Higher learning rate for MLP
    n_steps=1024,              # Smaller buffer for MLP
    batch_size=64,             # Smaller batch size for MLP
    n_epochs=4,                # Fewer epochs for MLP
    gamma=0.99,
    gae_lambda=0.95,
    clip_range=0.2,
    ent_coef=0.01,
    max_grad_norm=0.5,
    tensorboard_log="./tb_logs/",
    device="cpu",
    verbose=1
)

mlp_train_config = {
    "total_timesteps": 500000,  # Fewer timesteps for MLP
    "log_interval": 1000,
    "eval_freq": 5000,
    "best_model_save_path": "models/mlp_best_model",
    "log_path": "logs/mlp/",
}

# Function to get configuration based on model type
def get_config(model_type="cnn"):
    """
    Get configuration based on model type.
    
    Args:
        model_type: Either "cnn" or "mlp"
        
    Returns:
        Dictionary containing policy_kwargs, ppo_kwargs, train_config, env_config
    """
    if model_type.lower() == "cnn":
        return {
            "policy_kwargs": None,  # Will be set dynamically based on env
            "ppo_kwargs": cnn_ppo_kwargs,
            "train_config": cnn_train_config,
            "env_config": cnn_env_config
        }
    else:
        return {
            "policy_kwargs": mlp_policy_kwargs,
            "ppo_kwargs": mlp_ppo_kwargs,
            "train_config": mlp_train_config,
            "env_config": cnn_env_config  # Same env config
        }

# CNN-specific training utilities
def get_cnn_model_path(job_id=None):
    """Get CNN model path, optionally with job ID."""
    if job_id:
        return f"cnn_ppo_timetable_{job_id}.pth"
    return "cnn_ppo_timetable.pth"

def get_mlp_model_path(job_id=None):
    """Get MLP model path, optionally with job ID."""
    if job_id:
        return f"mlp_ppo_timetable_{job_id}.pth"
    return "ppo_timetable.pth"

# CNN architecture comparison settings
CNN_ARCHITECTURES = {
    "small": {
        "features_dim": 128,
        "conv_channels": [16, 32, 64],
        "fc_layers": [256, 128]
    },
    "medium": {
        "features_dim": 256,
        "conv_channels": [32, 64, 128],
        "fc_layers": [512, 256]
    },
    "large": {
        "features_dim": 512,
        "conv_channels": [64, 128, 256],
        "fc_layers": [1024, 512]
    }
}

def get_cnn_architecture_config(arch_type="medium"):
    """Get CNN architecture configuration."""
    return CNN_ARCHITECTURES.get(arch_type, CNN_ARCHITECTURES["medium"])
