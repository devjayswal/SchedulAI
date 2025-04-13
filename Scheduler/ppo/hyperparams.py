import torch
policy_kwargs = dict(
    net_arch=dict(pi=[64, 64], vf=[64, 64]) ,
    activation_fn=torch.nn.ReLU
)

ppo_kwargs = dict(
    learning_rate=1e-4,
    n_steps=2048,
    batch_size=256,
    n_epochs=10,
    gamma=0.99,
    gae_lambda=0.95,
    clip_range=0.2,
    ent_coef=0.01,
    max_grad_norm=0.5,
    tensorboard_log="./tb_logs/"
)

train_config = {
    "total_timesteps": 100000,  # Example value
    "log_interval": 1000,  # Add this line
    "eval_freq": 5000,
    "best_model_save_path": "models/best_model",
    "log_path": "logs/",
}


env_config = {
    "num_courses": 5,
    "num_slots": 6,
    "num_classrooms": 3,
    "n_envs": 4
}
