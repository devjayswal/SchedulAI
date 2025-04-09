import torch
from stable_baselines3 import PPO

def save_model(model: PPO, path: str = "ppo_timetable.pth"):
    """Save the model policy and optimizer for continuous learning."""
    torch.save({
        'policy_state_dict': model.policy.state_dict(),
        'optimizer_state_dict': model.policy.optimizer.state_dict() if model.policy.optimizer else None
    }, path)
    print(f"Model saved to {path}")

def load_model(env, path="ppo_timetable.pth", policy_kwargs=None, **ppo_kwargs) -> PPO:
    """Load a PPO model, restoring policy and optimizer if available."""
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
