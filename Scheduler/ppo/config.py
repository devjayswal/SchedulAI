import gym
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import EvalCallback
from env import TimetableEnv
from model import load_model, save_model
from hyperparams import env_config, policy_kwargs, ppo_kwargs, train_config
from sb3_contrib import MaskablePPO

# 1) Create vectorized environment
env = make_vec_env(
    lambda: TimetableEnv(
        num_courses=env_config['num_courses'],
        num_slots=env_config['num_slots'],
        num_classrooms=env_config['num_classrooms']
    ),
    n_envs=env_config['n_envs']
)

# 2) Load existing model or initialize a new one
model = load_model(env, "ppo_timetable.pth", policy_kwargs=policy_kwargs, **ppo_kwargs)

# 3) Set up evaluation callback
eval_callback = EvalCallback(
    env,
    eval_freq=train_config['eval_freq'],
    best_model_save_path=train_config['best_model_save_path'],
    log_path=train_config['log_path'],
    deterministic=True,
    render=False
)

# 4) Train the model
model.learn(total_timesteps=train_config['total_timesteps'], callback=eval_callback)

# 5) Save the final model state
save_model(model, "ppo_timetable.pth")
