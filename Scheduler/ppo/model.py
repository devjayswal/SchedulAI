import torch
from stable_baselines3 import PPO
from sb3_contrib import MaskablePPO
from ppo.cnn_model import create_cnn_model, create_cnn_policy_kwargs, create_cnn_hyperparams

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def save_model(model, path: str = "ppo_timetable.pth"):
    """Save the model policy and optimizer for continuous learning."""
    torch.save({
        'policy_state_dict': model.policy.state_dict(),
        'optimizer_state_dict': model.policy.optimizer.state_dict() if model.policy.optimizer else None
    }, path)
    print(f"Model saved to {path}")

def load_model(env, path="ppo_timetable.pth", policy_kwargs=None, use_cnn=True, **ppo_kwargs):
    """Load a PPO model, restoring policy and optimizer if available."""
    
    if use_cnn:
        # Use CNN-based model
        model = create_cnn_model(env, path, **ppo_kwargs)
    else:
        # Use traditional MLP model
        model = PPO("MlpPolicy", env, policy_kwargs=policy_kwargs, **ppo_kwargs, verbose=1)
    
    try:
        checkpoint = torch.load(path)
        model.policy.load_state_dict(checkpoint['policy_state_dict'])
        
        if checkpoint.get('optimizer_state_dict'):
            model.policy.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        else:
            print("Warning: No optimizer state found. Training may start fresh.")

        print(f"Model loaded from {path}")
    except FileNotFoundError:
        print(f"No checkpoint found at {path}. Starting from scratch.")
    
    return model

def create_cnn_ppo_model(env, model_path="cnn_ppo_timetable.pth", **kwargs):
    """Create a CNN-based PPO model with optimized hyperparameters."""
    return create_cnn_model(env, model_path, **kwargs)

def create_mlp_ppo_model(env, policy_kwargs=None, **ppo_kwargs):
    """Create a traditional MLP-based PPO model."""
    return PPO("MlpPolicy", env, policy_kwargs=policy_kwargs, **ppo_kwargs, verbose=1)
