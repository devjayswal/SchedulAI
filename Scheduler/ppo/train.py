import os
import time
import torch
import asyncio
from ppo.env import TimetableEnv
from ppo.model import load_model, save_model
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import EvalCallback
from ppo.hyperparams import policy_kwargs, ppo_kwargs, train_config
import logging

async def run_training(data, job_id):
    """Load or initialize model and start PPO training with logging per job."""
    
    # Create a log directory for this job
    log_dir = f"logs/{job_id}"
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, "log.log")
    train_logger = logging.getLogger(f"ppo_{job_id}")
    train_logger.setLevel(logging.INFO)

    fh = logging.FileHandler(log_file)
    fh.setLevel(logging.INFO)

    #StreamHandler for console output
    sh = logging.StreamHandler()
    sh.setLevel(logging.INFO)

    fmt = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    fh.setFormatter(fmt)
    sh.setFormatter(fmt)

    # avoid duplicate logs in the same file
    if not train_logger.handlers:
        train_logger.addHandler(fh)
        train_logger.addHandler(sh)
    
    train_logger.info(f"Starting training for job {job_id}")    


    # Create the environment
    env = make_vec_env(lambda: TimetableEnv( data), n_envs=1)

    # Load or initialize model
    model = load_model(env, path="ppo_timetable.pth", policy_kwargs=policy_kwargs, **ppo_kwargs)

    eval_cb = EvalCallback(env, eval_freq=train_config["eval_freq"],
                           best_model_save_path=train_config["best_model_save_path"],
                           log_path=train_config["log_path"])

    start_time = time.time()

    
    for step in range(0, train_config["total_timesteps"], train_config["log_interval"]):
        model.learn(total_timesteps=train_config["log_interval"], callback=eval_cb)
        
        elapsed_time = time.time() - start_time
        loss  = model.logger.name_to_value.get('loss', 'N/A')
        ep_rew_mean = model.logger.name_to_value.get('ep_rew_mean', 'N/A')
        log_msg = (f"Step: {step + train_config['log_interval']} | "
                    f"Elapsed: {elapsed_time:.2f}s | "
                    f"Loss: {model.logger.name_to_value.get('loss', 'N/A')} | "
                    f"Reward: {model.logger.name_to_value.get('ep_rew_mean', 'N/A')}")
        
        train_logger.info(log_msg)  # Log to file and console
        yield log_msg  # Send log message to _run_job()
        
        save_model(model, "ppo_timetable.pth")

    train_logger.info("Training complete.")
    # Save the final model
    yield "Training complete."
