import os
import time
import torch
import asyncio
import numpy as np
import logging

from sb3_contrib import MaskablePPO
from sb3_contrib.common.wrappers import ActionMasker
from ppo.env import TimetableEnv
from ppo.model import load_model, save_model
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import EvalCallback
from ppo.hyperparams import policy_kwargs, ppo_kwargs, train_config

async def run_training(data, job_id):
    """Load or initialize MaskablePPO model and start training with logging per job."""

    # Set up logging for the specific job
    log_dir = f"logs/{job_id}"
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, "log.log")

    train_logger = logging.getLogger(f"ppo_{job_id}")
    train_logger.setLevel(logging.INFO)

    fh = logging.FileHandler(log_file)
    fh.setLevel(logging.INFO)
    sh = logging.StreamHandler()
    sh.setLevel(logging.INFO)

    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    fh.setFormatter(formatter)
    sh.setFormatter(formatter)

    if not train_logger.handlers:
        train_logger.addHandler(fh)
        train_logger.addHandler(sh)

    train_logger.info(f"Starting training for job {job_id}")

    # Define the action mask function
    def mask_fn(env: TimetableEnv) -> np.ndarray:
        return env.get_action_mask()


    # Create and wrap the environment with masking
    def make_env():
        return ActionMasker(TimetableEnv(data), mask_fn)

    env = make_vec_env(make_env, n_envs=1)

    # Load or initialize the model
    model = load_model(env, path="ppo_timetable.pth", policy_kwargs=policy_kwargs, **ppo_kwargs)
    if not isinstance(model, MaskablePPO):
        model = MaskablePPO("MlpPolicy", env, policy_kwargs=policy_kwargs, verbose=1, **ppo_kwargs)

    eval_cb = EvalCallback(
        env,
        eval_freq=train_config["eval_freq"],
        best_model_save_path=train_config["best_model_save_path"],
        log_path=train_config["log_path"],
    )

    # Training loop
    start_time = time.time()
    for step in range(0, train_config["total_timesteps"], train_config["log_interval"]):
        model.learn(total_timesteps=train_config["log_interval"], callback=eval_cb)

        elapsed_time = time.time() - start_time
        loss = model.logger.name_to_value.get('loss', 'N/A')
        reward = model.logger.name_to_value.get('ep_rew_mean', 'N/A')

        log_msg = (f"Step: {step + train_config['log_interval']} | "
                   f"Elapsed: {elapsed_time:.2f}s | "
                   f"Loss: {loss} | Reward: {reward}")
        train_logger.info(log_msg)
        yield log_msg

        save_model(model, "ppo_timetable.pth")

    train_logger.info("Training complete.")
    yield "Training complete."
