import torch
policy_kwargs = dict(
    net_arch=[dict(pi=[256, 256], vf=[256, 256])],
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
    "total_timesteps": 1_000_000,
    "eval_freq": 50_000,
    "best_model_save_path": "./best/",
    "log_path": "./logs/"
}

env_config = {
    "num_courses": 5,
    "num_slots": 6,
    "num_classrooms": 3,
    "n_envs": 4
}
