"""
Enhanced Configuration for Improved PPO Training
This configuration file contains all the enhanced settings for better performance.
"""

import torch
import os

# Enhanced Training Configuration
ENHANCED_TRAINING_CONFIG = {
    # Environment settings
    "n_envs": 8,                    # Parallel environments for faster training
    "max_steps_per_episode": 150,   # Maximum steps per episode
    
    # Training parameters
    "total_timesteps": 1000000,     # Total training timesteps
    "log_interval": 1000,           # Logging interval
    "eval_freq": 10000,             # Evaluation frequency
    "save_freq": 50000,             # Model saving frequency
    
    # Model settings
    "use_enhanced_cnn": True,       # Use enhanced CNN architecture
    "use_curriculum_learning": True, # Enable curriculum learning
    "use_advanced_metrics": True,   # Enable advanced metrics tracking
    
    # Curriculum learning settings
    "curriculum_max_difficulty": 2, # Maximum difficulty level
    "curriculum_success_threshold": 0.6, # Success rate threshold for advancement
    "curriculum_episodes_per_level": 50, # Episodes before difficulty check
    "difficulty_start": 0,
    "difficulty_increase_threshold": 0.6,
    
    # Reward shaping settings
    "use_enhanced_rewards": True,   # Use enhanced reward shaping
    "reward_objectives": {
        "constraint_satisfaction": 0.4,
        "efficiency": 0.3,
        "fairness": 0.2,
        "preferences": 0.1
    },
    
    # Performance monitoring
    "performance_thresholds": {
        "min_success_rate": 0.3,
        "target_success_rate": 0.8,
        "max_constraint_violation_rate": 0.2,
        "min_utilization_efficiency": 0.6
    }
}

# Enhanced CNN Hyperparameters
ENHANCED_CNN_HYPERPARAMS = {
    "learning_rate": 0.0003,        # Learning rate
    "n_steps": 2048,               # Larger buffer for better gradient estimates
    "batch_size": 64,              # Batch size for CNN training
    "n_epochs": 10,                # Epochs for complex architecture
    "gamma": 0.99,
    "gae_lambda": 0.95,
    "clip_range": 0.2,
    "ent_coef": 0.01,              # Entropy coefficient
    "vf_coef": 0.5,
    "max_grad_norm": 0.5,
    "tensorboard_log": "./tb_logs/",
    "device": "auto",              # Automatically use GPU if available, fallback to CPU
    "verbose": 1
}

# Enhanced MLP Hyperparameters (for comparison)
ENHANCED_MLP_HYPERPARAMS = {
    "learning_rate": 3e-4,         # Higher learning rate for MLP
    "n_steps": 1024,              # Smaller buffer for MLP
    "batch_size": 64,             # Batch size for MLP
    "n_epochs": 4,                # Fewer epochs for MLP
    "gamma": 0.99,
    "gae_lambda": 0.95,
    "clip_range": 0.2,
    "ent_coef": 0.01,
    "vf_coef": 0.5,
    "max_grad_norm": 0.5,
    "tensorboard_log": "./tb_logs/",
    "device": "auto",
    "verbose": 1
}

# Enhanced Environment Configuration
ENHANCED_ENV_CONFIG = {
    "num_courses": 5,
    "num_slots": 6,                # Keep original constraints
    "num_classrooms": 5,           # Increased from 3 to reduce constraint conflicts
    "features_dim": 256,           # CNN feature dimension
    "grid_height": 6,              # Time slots per day
    "grid_width": 8,               # Classrooms (can be adjusted)
    "use_action_masking": True,    # Enable action masking
    "use_reward_shaping": True,    # Enable reward shaping
    "max_steps": 150,
}

# Advanced Metrics Configuration
METRICS_CONFIG = {
    "track_episode_metrics": True,
    "track_training_metrics": True,
    "track_performance_metrics": True,
    "save_metrics_frequency": 20000,  # Save metrics every N timesteps
    "plot_metrics": True,             # Generate metric plots
    "metrics_history_size": 1000,     # Size of metrics history
}

# Logging Configuration
LOGGING_CONFIG = {
    "log_level": "INFO",
    "log_to_file": True,
    "log_to_console": True,
    "log_format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    "log_rotation": True,
    "max_log_size": "10MB",
    "backup_count": 5
}

# Model Architecture Configurations
ARCHITECTURE_CONFIGS = {
    "enhanced_cnn": {
        "features_dim": 256,
        "conv_channels": [32, 64, 128],
        "residual_blocks": 3,
        "attention_heads": 8,
        "dropout_rate": 0.2,
        "use_batch_norm": True,
        "use_attention": True,
        "use_residual": True
    },
    "enhanced_mlp": {
        "hidden_sizes": [128, 128, 64],
        "activation": "relu",
        "dropout_rate": 0.1,
        "use_batch_norm": False
    }
}

# Performance Optimization Settings
PERFORMANCE_CONFIG = {
    "use_parallel_training": True,
    "n_parallel_envs": 8,
    "use_vectorized_envs": True,
    "use_gpu_if_available": True,   # Enable GPU usage when available
    "memory_efficient": True,
    "cache_observations": True,
    "prefetch_batches": True
}

# Curriculum Learning Settings
CURRICULUM_CONFIG = {
    "enabled": True,
    "initial_difficulty": 0,
    "max_difficulty": 2,
    "success_threshold": 0.6,
    "episodes_per_check": 20,
    "difficulty_increase_rate": 0.1,
    "difficulty_decrease_rate": 0.2,
    "min_episodes_before_advancement": 50,
    "curriculum_thresholds": {
        "easy_to_medium": 0.4,
        "medium_to_hard": 0.7
    }
}

# Reward Shaping Settings
REWARD_CONFIG = {
    "use_multi_objective": True,
    "use_curriculum_rewards": True,
    "use_adaptive_weights": True,
    "base_rewards": {
        "successful_placement": 15.0,
        "course_completion": 25.0,
        "episode_completion": 60.0,
        "constraint_violation": -8.0
    },
    "efficiency_rewards": {
        "utilization_bonus": 10.0,
        "compactness_bonus": 5.0,
        "distribution_bonus": 3.0
    },
    "fairness_rewards": {
        "faculty_balance": 8.0,
        "classroom_balance": 5.0,
        "time_balance": 3.0
    },
    "reward_scaling": {
        "exploration_phase": 1.2,
        "exploitation_phase": 1.0,
        "mastery_phase": 0.8
    }
}

# Early Stopping Configuration
EARLY_STOPPING_CONFIG = {
    "patience": 50000,
    "min_improvement": 0.01
}

def get_config(config_type: str = "training"):
    """
    Get enhanced configuration based on type.
    
    Args:
        config_type: Type of configuration to return
        
    Returns:
        Dictionary containing the requested configuration
    """
    configs = {
        "training": ENHANCED_TRAINING_CONFIG,
        "cnn_hyperparams": ENHANCED_CNN_HYPERPARAMS,
        "mlp_hyperparams": ENHANCED_MLP_HYPERPARAMS,
        "env": ENHANCED_ENV_CONFIG,
        "metrics": METRICS_CONFIG,
        "logging": LOGGING_CONFIG,
        "architecture": ARCHITECTURE_CONFIGS,
        "performance": PERFORMANCE_CONFIG,
        "curriculum": CURRICULUM_CONFIG,
        "reward": REWARD_CONFIG,
        "early_stopping": EARLY_STOPPING_CONFIG
    }
    
    return configs.get(config_type, ENHANCED_TRAINING_CONFIG)

def get_model_config(model_type: str = "enhanced_cnn"):
    """
    Get model-specific configuration.
    
    Args:
        model_type: Type of model ("enhanced_cnn" or "enhanced_mlp")
        
    Returns:
        Dictionary containing model configuration
    """
    if model_type == "enhanced_cnn":
        return {
            "hyperparams": ENHANCED_CNN_HYPERPARAMS,
            "architecture": ARCHITECTURE_CONFIGS["enhanced_cnn"],
            "use_enhanced_features": True
        }
    else:
        return {
            "hyperparams": ENHANCED_MLP_HYPERPARAMS,
            "architecture": ARCHITECTURE_CONFIGS["enhanced_mlp"],
            "use_enhanced_features": False
        }

def setup_enhanced_training_environment():
    """Setup environment for enhanced training."""
    # Create necessary directories
    directories = [
        "./tb_logs/",
        "./logs/",
        "./models/",
        "./logs/metrics/",
        "./logs/enhanced_training/"
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
    
    # Set up logging
    import logging
    logging.basicConfig(
        level=getattr(logging, LOGGING_CONFIG["log_level"]),
        format=LOGGING_CONFIG["log_format"],
        handlers=[
            logging.FileHandler("./logs/enhanced_training.log"),
            logging.StreamHandler()
        ]
    )
    
    print("Enhanced training environment setup complete!")

# Example usage
if __name__ == "__main__":
    # Setup environment
    setup_enhanced_training_environment()
    
    # Get configurations
    training_config = get_config("training")
    cnn_config = get_model_config("enhanced_cnn")
    
    print("Enhanced Training Configuration:")
    print(f"  - Parallel environments: {training_config['n_envs']}")
    print(f"  - Total timesteps: {training_config['total_timesteps']}")
    print(f"  - Use enhanced CNN: {training_config['use_enhanced_cnn']}")
    print(f"  - Use curriculum learning: {training_config['use_curriculum_learning']}")
    
    print("\nEnhanced CNN Configuration:")
    print(f"  - Learning rate: {cnn_config['hyperparams']['learning_rate']}")
    print(f"  - Batch size: {cnn_config['hyperparams']['batch_size']}")
    print(f"  - N epochs: {cnn_config['hyperparams']['n_epochs']}")
    print(f"  - Features dim: {cnn_config['architecture']['features_dim']}")