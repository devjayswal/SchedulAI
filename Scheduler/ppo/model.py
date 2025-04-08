import torch
from stable_baselines3 import PPO

def save_model(model, path="ppo_timetable.pth"):
    """Save the trained PPO model using PyTorch."""
    torch.save(model.policy.state_dict(), path)
    print(f"Model saved to {path}")

def load_model(env, path="ppo_timetable.pth"):
    """Load a trained PPO model from a file."""
    model = PPO("MlpPolicy", env, verbose=1)
    model.policy.load_state_dict(torch.load(path))
    print(f"Model loaded from {path}")
    return model
