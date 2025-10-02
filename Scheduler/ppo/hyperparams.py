import torch
policy_kwargs = dict(
    net_arch=dict(pi=[64, 64], vf=[64, 64]) ,
    activation_fn=torch.nn.ReLU
)

ppo_kwargs = dict(
    learning_rate=3e-4,  # Increased for faster learning
    n_steps=1024,        # Reduced for more frequent updates
    batch_size=64,       # Reduced for better gradient estimates
    n_epochs=4,          # Reduced to prevent overfitting
    gamma=0.99,
    gae_lambda=0.95,
    clip_range=0.2,
    ent_coef=0.01,
    max_grad_norm=0.5,
    tensorboard_log="./tb_logs/",
    device="auto"  # Automatically use GPU if available, fallback to CPU
)

train_config = {
    "total_timesteps": 500000,  # Increased for better training
    "log_interval": 1000,  # Add this line
    "eval_freq": 5000,
    "best_model_save_path": "models/best_model",
    "log_path": "logs/",
}


env_config = {
    "num_courses": 5,
    "num_slots": 6,       # KEEP ORIGINAL - These are hard constraints from college schedule
    "num_classrooms": 5,  # Increased from 3 to reduce constraint conflicts
    "n_envs": 4
}
