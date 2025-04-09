import torch
import asyncio
from env import TimetableEnv
from model import load_model, save_model
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import EvalCallback
from ppo.hyperparams import policy_kwargs, ppo_kwargs, train_config

async def run_training(data):
    """Load or initialize model and start PPO training with incoming data."""
    
    # Extract environment config from the data
    num_courses = sum(len(branch["courses"]) for branch in data["branches"])
    num_slots = len(data["time_slots"])
    num_classrooms = len(data["classrooms"])

    # Create the environment using the extracted data
    env = make_vec_env(lambda: TimetableEnv(num_courses, num_slots, num_classrooms, data), 
                       n_envs=1)  # Assuming single environment instance for now

    # Load or initialize model
    model = load_model(env, path="ppo_timetable.pth", policy_kwargs=policy_kwargs, **ppo_kwargs)

    eval_cb = EvalCallback(env, eval_freq=train_config["eval_freq"],
                           best_model_save_path=train_config["best_model_save_path"],
                           log_path=train_config["log_path"])

    for step in range(0, train_config["total_timesteps"], train_config["log_interval"]):
        model.learn(total_timesteps=train_config["log_interval"], callback=eval_cb)
        save_model(model, "ppo_timetable.pth")
        yield f"Trained up to {step + train_config['log_interval']} timesteps..."

    yield "Training complete."
