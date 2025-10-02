"""
CNN-based PPO Training Script for Timetable Scheduling
This script provides training functionality specifically for CNN-based models.
"""

import os
import time
import torch
import asyncio
import numpy as np
import logging

from sb3_contrib import MaskablePPO
from sb3_contrib.common.wrappers import ActionMasker
from ppo.env import TimetableEnv
from ppo.model import create_cnn_ppo_model, save_model
from ppo.cnn_model import create_cnn_model
from ppo.reward_shaping import AdaptiveRewardSystem
from ppo.advanced_metrics import AdvancedMetrics, MetricsCallback
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import EvalCallback, BaseCallback
from ppo.cnn_hyperparams import get_config, get_cnn_model_path
from utils.config_processor import config_processor
from utils.job_manager import update_progress, add_log


class MetricsCallback(BaseCallback):
    """Custom callback to track training metrics."""
    
    def __init__(self, verbose=0):
        super(MetricsCallback, self).__init__(verbose)
        self.metrics = {
            'loss': 'N/A',
            'reward': 'N/A', 
            'policy_loss': 'N/A',
            'value_loss': 'N/A'
        }
        self.last_metrics = {}
    
    def _on_step(self) -> bool:
        # This will be called after each training step
        # We can access the model's internal metrics here
        return True
    
    def _on_rollout_end(self) -> None:
        # Called at the end of each rollout
        # Try to extract metrics from the model
        try:
            if hasattr(self.model, 'logger') and self.model.logger is not None:
                # Update metrics from logger with multiple fallback options
                logger_dict = self.model.logger.name_to_value
                
                # Try different possible metric names based on what we saw in the test
                self.metrics['loss'] = (
                    logger_dict.get('train/loss') or 
                    logger_dict.get('loss') or 'N/A'
                )
                self.metrics['reward'] = (
                    logger_dict.get('rollout/ep_rew_mean') or 
                    logger_dict.get('ep_rew_mean') or 'N/A'
                )
                self.metrics['policy_loss'] = (
                    logger_dict.get('train/policy_gradient_loss') or 
                    logger_dict.get('train/policy_loss') or 
                    logger_dict.get('policy_loss') or 'N/A'
                )
                self.metrics['value_loss'] = (
                    logger_dict.get('train/value_loss') or 
                    logger_dict.get('value_loss') or 'N/A'
                )
                
                # Store the last successful metrics
                if any(v != 'N/A' for v in self.metrics.values()):
                    self.last_metrics = self.metrics.copy()
                    
        except Exception as e:
            # If there's an error, keep the last known metrics
            if self.last_metrics:
                self.metrics = self.last_metrics.copy()
    
    def _on_training_end(self) -> None:
        # Called at the end of training - this is where we should capture final metrics
        try:
            if hasattr(self.model, 'logger') and self.model.logger is not None:
                logger_dict = self.model.logger.name_to_value
                
                # Update metrics from logger
                self.metrics['loss'] = logger_dict.get('train/loss', 'N/A')
                self.metrics['reward'] = logger_dict.get('rollout/ep_rew_mean', 'N/A')
                self.metrics['policy_loss'] = logger_dict.get('train/policy_gradient_loss', 'N/A')
                self.metrics['value_loss'] = logger_dict.get('train/value_loss', 'N/A')
                
                # Store the last successful metrics
                if any(v != 'N/A' for v in self.metrics.values()):
                    self.last_metrics = self.metrics.copy()
                    
        except Exception as e:
            # If there's an error, keep the last known metrics
            if self.last_metrics:
                self.metrics = self.last_metrics.copy()
    
    def get_metrics(self):
        """Get the current metrics."""
        return self.metrics.copy()


async def run_cnn_training(data, job_id):
    """Load or initialize CNN-based MaskablePPO model and start training with logging per job."""

    # Set up logging for the specific job
    log_dir = f"logs/{job_id}"
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, "cnn_log.log")

    train_logger = logging.getLogger(f"cnn_ppo_{job_id}")
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

    train_logger.info(f"Starting CNN training for job {job_id}")
    add_log(job_id, "Starting CNN training initialization")
    update_progress(job_id, phase="Initializing CNN", percentage=5)
    
    # Get CNN configuration
    config = get_config("cnn")
    cnn_ppo_kwargs = config["ppo_kwargs"]
    cnn_train_config = config["train_config"]
    cnn_env_config = config["env_config"]
    
    # Use dynamic configuration if available
    if hasattr(data, 'env_config') and hasattr(data, 'training_config'):
        env_config = data.env_config
        training_config = data.training_config
        
        train_logger.info(f"Using dynamic configuration for CNN:")
        train_logger.info(f"  - Courses: {env_config.get('num_courses', 'N/A')}")
        train_logger.info(f"  - Time slots: {env_config.get('num_slots', 'N/A')}")
        train_logger.info(f"  - Classrooms: {env_config.get('num_classrooms', 'N/A')}")
        train_logger.info(f"  - Learning rate: {training_config.get('learning_rate', 'N/A')}")
        train_logger.info(f"  - Batch size: {training_config.get('batch_size', 'N/A')}")
        
        # Update CNN config with dynamic values
        cnn_ppo_kwargs.update({
            'learning_rate': training_config.get('learning_rate', cnn_ppo_kwargs['learning_rate']),
            'batch_size': training_config.get('batch_size', cnn_ppo_kwargs['batch_size']),
            'n_steps': training_config.get('n_steps', cnn_ppo_kwargs['n_steps']),
        })
        
        add_log(job_id, f"CNN Dynamic config: {env_config.get('num_courses', 'N/A')} courses, {env_config.get('num_classrooms', 'N/A')} classrooms")
    else:
        train_logger.info("Using default CNN configuration")
        add_log(job_id, "Using default CNN configuration")

    # Define the action mask function
    def mask_fn(env: TimetableEnv) -> np.ndarray:
        return env.get_action_mask()

    # Create and wrap the environment with masking
    add_log(job_id, "Creating environment for CNN")
    update_progress(job_id, phase="Setting up CNN environment", percentage=10)
    
    def make_env():
        return ActionMasker(TimetableEnv(data), mask_fn)

    # Use parallel environments for faster training
    n_envs = 8  # Increased from 1 for better performance
    env = make_vec_env(make_env, n_envs=n_envs)

    # Create CNN model
    add_log(job_id, "Creating CNN model")
    update_progress(job_id, phase="Creating CNN model", percentage=15)
    
    model_path = get_cnn_model_path(job_id)
    # Use enhanced CNN model for better performance
    model = create_cnn_model(env, model_path, **cnn_ppo_kwargs)
    add_log(job_id, "CNN model created successfully")

    # Set up evaluation callback
    eval_cb = EvalCallback(
        env,
        eval_freq=cnn_train_config["eval_freq"],
        best_model_save_path=cnn_train_config["best_model_save_path"],
        log_path=cnn_train_config["log_path"],
        deterministic=True,
        render=False
    )
    
    # Set up metrics callback
    metrics_cb = MetricsCallback()

    # Training loop
    add_log(job_id, "Starting CNN training loop")
    update_progress(job_id, phase="CNN Training", percentage=20)
    
    start_time = time.time()
    total_steps = cnn_train_config["total_timesteps"]
    log_interval = cnn_train_config["log_interval"]
    
    train_logger.info(f"CNN Training Configuration:")
    train_logger.info(f"  - Total timesteps: {total_steps}")
    train_logger.info(f"  - Learning rate: {cnn_ppo_kwargs['learning_rate']}")
    train_logger.info(f"  - Batch size: {cnn_ppo_kwargs['batch_size']}")
    train_logger.info(f"  - N steps: {cnn_ppo_kwargs['n_steps']}")
    train_logger.info(f"  - N epochs: {cnn_ppo_kwargs['n_epochs']}")
    
    for step in range(0, total_steps, log_interval):
        # Update progress
        current_step = step + log_interval
        progress_percentage = 20 + (current_step / total_steps) * 70  # 20-90% for training
        update_progress(job_id, current_step=current_step, total_steps=total_steps, 
                       phase="CNN Training", percentage=progress_percentage)
        
        # Train the model with both callbacks
        from stable_baselines3.common.callbacks import CallbackList
        callback_list = CallbackList([eval_cb, metrics_cb])
        model.learn(total_timesteps=log_interval, callback=callback_list)

        elapsed_time = time.time() - start_time
        
        # Get training metrics from custom callback first, then fallback to logger
        try:
            # Try to get metrics from our custom callback
            callback_metrics = metrics_cb.get_metrics()
            loss = callback_metrics.get('loss', 'N/A')
            reward = callback_metrics.get('reward', 'N/A')
            policy_loss = callback_metrics.get('policy_loss', 'N/A')
            value_loss = callback_metrics.get('value_loss', 'N/A')
            
            # If callback metrics are still N/A, try logger directly
            if loss == 'N/A' and hasattr(model, 'logger') and model.logger is not None:
                # Use the correct metric names we found in the test
                loss = model.logger.name_to_value.get('train/loss', 'N/A')
                reward = model.logger.name_to_value.get('rollout/ep_rew_mean', 'N/A')
                policy_loss = model.logger.name_to_value.get('train/policy_gradient_loss', 'N/A')
                value_loss = model.logger.name_to_value.get('train/value_loss', 'N/A')
                
                # If still N/A, log available metrics for debugging
                if loss == 'N/A' and hasattr(model.logger, 'name_to_value'):
                    available_metrics = list(model.logger.name_to_value.keys())
                    train_logger.debug(f"Available metrics: {available_metrics}")
                    
        except Exception as e:
            train_logger.warning(f"Could not access logger metrics: {e}")
            loss = reward = policy_loss = value_loss = 'N/A'

        log_msg = (f"CNN Step: {current_step} | "
                   f"Elapsed: {elapsed_time:.2f}s | "
                   f"Loss: {loss} | Reward: {reward} | "
                   f"Policy Loss: {policy_loss} | Value Loss: {value_loss}")
        train_logger.info(log_msg)
        add_log(job_id, log_msg)
        yield log_msg

        # Save model checkpoint
        save_model(model, model_path)

    add_log(job_id, "CNN training completed successfully")
    update_progress(job_id, phase="CNN Completed", percentage=100)
    train_logger.info("CNN training complete.")
    yield "CNN training complete."


def test_cnn_model():
    """Test function to verify CNN model creation and basic functionality."""
    print("Testing CNN model creation...")
    
    try:
        # Create a simple test environment
        from models.Timetable import Timetable
        from models.Course import Course
        from models.Faculty import Faculty
        from models.Classroom import Classroom
        
        # Create minimal test data
        timetable = Timetable()
        timetable.days = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"]
        timetable.time_slots = ["09:00-10:00", "10:00-11:00", "11:00-12:00", "12:00-13:00", "14:00-15:00", "15:00-16:00"]
        
        # Add test courses
        faculty = Faculty("F001", "Test Faculty", "test@example.com")
        classroom = Classroom("C001", "Test Room", "theory", 50)
        
        for i in range(3):
            course = Course(f"C{i+1}", f"Course {i+1}", 3, "theory", faculty.short_name)
            timetable.courses.append(course)
        
        timetable.faculty = [faculty]
        timetable.classrooms = [classroom]
        
        # Create environment
        env = TimetableEnv(timetable)
        
        # Test CNN model creation
        model = create_cnn_ppo_model(env)
        print("[OK] CNN model created successfully")
        
        # Test model forward pass
        obs = env.reset()[0]
        action, _ = model.predict(obs, deterministic=True)
        print(f"[OK] CNN model prediction successful: action = {action}")
        
        # Test model training step
        model.learn(total_timesteps=100)
        print("[OK] CNN model training step successful")
        
        print("All CNN model tests passed!")
        return True
        
    except Exception as e:
        print(f"✗ CNN model test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    # Run tests
    test_cnn_model()
