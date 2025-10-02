"""
Optimized Configuration for PPO Training

This configuration addresses the training issues identified:
1. Increased environment capacity to reduce constraint conflicts
2. Optimized training parameters for better learning
3. Enhanced reward shaping for better guidance

Usage: Import this instead of hyperparams.py for improved training
"""

import torch

# Optimized Policy Configuration
policy_kwargs = dict(
    net_arch=dict(pi=[128, 128], vf=[128, 128]),  # Larger networks for better learning
    activation_fn=torch.nn.ReLU
)

# Optimized PPO Parameters
ppo_kwargs = dict(
    learning_rate=3e-4,      # Increased for faster learning
    n_steps=1024,           # Reduced for more frequent updates
    batch_size=64,          # Reduced for better gradient estimates
    n_epochs=4,             # Reduced to prevent overfitting
    gamma=0.99,
    gae_lambda=0.95,
    clip_range=0.2,
    ent_coef=0.01,
    max_grad_norm=0.5,
    tensorboard_log="./tb_logs/",
    device="auto"  # Automatically use GPU if available, fallback to CPU
)

# Optimized Training Configuration
train_config = {
    "total_timesteps": 500000,
    "log_interval": 500,     # More frequent logging for better monitoring
    "eval_freq": 2500,       # More frequent evaluation
    "best_model_save_path": "models/best_model",
    "log_path": "logs/",
}

# Optimized Environment Configuration
env_config = {
    "num_courses": 5,
    "num_slots": 6,          # KEEP ORIGINAL - These are hard constraints from college schedule
    "num_classrooms": 5,     # Increased from 3 to reduce constraint conflicts
    "n_envs": 4
}

# Additional Training Options
training_options = {
    "use_curriculum_learning": True,
    "use_enhanced_rewards": True,
    "early_stopping_patience": 10,
    "save_frequency": 10000,
}
