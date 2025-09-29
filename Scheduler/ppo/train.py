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
from utils.config_processor import config_processor
from utils.job_manager import update_progress, add_log

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
    add_log(job_id, "Starting training initialization")
    update_progress(job_id, phase="Initializing", percentage=5)
    
    # Use dynamic configuration if available
    if hasattr(data, 'env_config') and hasattr(data, 'training_config'):
        env_config = data.env_config
        training_config = data.training_config
        
        train_logger.info(f"Using dynamic configuration:")
        train_logger.info(f"  - Courses: {env_config.get('num_courses', 'N/A')}")
        train_logger.info(f"  - Time slots: {env_config.get('num_slots', 'N/A')}")
        train_logger.info(f"  - Classrooms: {env_config.get('num_classrooms', 'N/A')}")
        train_logger.info(f"  - Learning rate: {training_config.get('learning_rate', 'N/A')}")
        train_logger.info(f"  - Batch size: {training_config.get('batch_size', 'N/A')}")
        
        add_log(job_id, f"Dynamic config: {env_config.get('num_courses', 'N/A')} courses, {env_config.get('num_classrooms', 'N/A')} classrooms")
    else:
        train_logger.info("Using default configuration")
        add_log(job_id, "Using default configuration")

    # Define the action mask function
    def mask_fn(env: TimetableEnv) -> np.ndarray:
        return env.get_action_mask()


    # Create and wrap the environment with masking
    add_log(job_id, "Creating environment")
    update_progress(job_id, phase="Setting up environment", percentage=10)
    
    def make_env():
        return ActionMasker(TimetableEnv(data), mask_fn)

    env = make_vec_env(make_env, n_envs=1)

    # Load or initialize the model
    add_log(job_id, "Loading/initializing model")
    update_progress(job_id, phase="Loading model", percentage=15)
    
    model = load_model(env, path="ppo_timetable.pth", policy_kwargs=policy_kwargs, **ppo_kwargs)
    if not isinstance(model, MaskablePPO):
        model = MaskablePPO("MlpPolicy", env, policy_kwargs=policy_kwargs, verbose=1, **ppo_kwargs)
        add_log(job_id, "Initialized new model")
    else:
        add_log(job_id, "Loaded existing model")

    eval_cb = EvalCallback(
        env,
        eval_freq=train_config["eval_freq"],
        best_model_save_path=train_config["best_model_save_path"],
        log_path=train_config["log_path"],
    )

    # Training loop
    add_log(job_id, "Starting training loop")
    update_progress(job_id, phase="Training", percentage=20)
    
    start_time = time.time()
    total_steps = train_config["total_timesteps"]
    log_interval = train_config["log_interval"]
    
    for step in range(0, total_steps, log_interval):
        # Update progress
        current_step = step + log_interval
        progress_percentage = 20 + (current_step / total_steps) * 70  # 20-90% for training
        update_progress(job_id, current_step=current_step, total_steps=total_steps, 
                       phase="Training", percentage=progress_percentage)
        
        model.learn(total_timesteps=log_interval, callback=eval_cb)

        elapsed_time = time.time() - start_time
        loss = model.logger.name_to_value.get('loss', 'N/A')
        reward = model.logger.name_to_value.get('ep_rew_mean', 'N/A')

        log_msg = (f"Step: {current_step} | "
                   f"Elapsed: {elapsed_time:.2f}s | "
                   f"Loss: {loss} | Reward: {reward}")
        train_logger.info(log_msg)
        add_log(job_id, log_msg)
        yield log_msg

        save_model(model, "ppo_timetable.pth")

    add_log(job_id, "Training completed successfully")
    update_progress(job_id, phase="Completed", percentage=100)
    train_logger.info("Training complete.")
    yield "Training complete."
