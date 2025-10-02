"""
Improved Training Script for Timetable Scheduling
This script addresses the training issues by:
1. Using improved environment with better reward shaping
2. Implementing curriculum learning
3. Better hyperparameter tuning
4. Enhanced monitoring and logging
"""

import os
import time
import torch
import numpy as np
import logging
from typing import Dict, Any, Optional, Tuple
from datetime import datetime

from sb3_contrib import MaskablePPO
from sb3_contrib.common.wrappers import ActionMasker
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import EvalCallback, BaseCallback, CallbackList
from stable_baselines3.common.vec_env import VecMonitor

from ppo.env import TimetableEnv
from ppo.reward_shaping import create_improved_reward_function, RewardAnalyzer
from ppo.cnn_model import create_cnn_model
from utils.config_processor import config_processor
from utils.job_manager import update_progress, add_log


class ImprovedTrainingCallback(BaseCallback):
    """Enhanced callback for improved training monitoring."""
    
    def __init__(self, reward_system, verbose=1):
        super().__init__(verbose)
        self.reward_system = reward_system
        self.episode_rewards = []
        self.episode_lengths = []
        self.success_count = 0
        self.total_episodes = 0
        self.reward_analyzer = RewardAnalyzer()
        
    def _on_step(self) -> bool:
        """Called at each step."""
        return True
    
    def _on_rollout_end(self) -> None:
        """Called at the end of each rollout."""
        # Get episode information from the environment
        if hasattr(self.training_env, 'get_attr'):
            try:
                episode_infos = self.training_env.get_attr('get_performance_metrics')
                if episode_infos:
                    for info in episode_infos:
                        if info:
                            self._process_episode_info(info)
            except:
                pass
    
    def _process_episode_info(self, info: Dict):
        """Process episode information."""
        self.total_episodes += 1
        
        # Extract episode data
        episode_reward = info.get('total_reward', 0)
        episode_length = info.get('episode_length', 0)
        successful_placements = info.get('successful_placements', 0)
        constraint_violations = info.get('constraint_violations', 0)
        utilization_rate = info.get('utilization_rate', 0)
        
        # Determine success (simplified criteria)
        success = (successful_placements > 0 and 
                  constraint_violations < successful_placements and
                  utilization_rate > 0.3)
        
        if success:
            self.success_count += 1
        
        # Store data
        self.episode_rewards.append(episode_reward)
        self.episode_lengths.append(episode_length)
        self.reward_analyzer.add_episode(episode_reward, success, episode_length)
        
        # Update reward system
        self.reward_system.update_episode(success, episode_reward)
        
        # Log progress
        if self.total_episodes % 50 == 0:
            self._log_progress()
    
    def _log_progress(self):
        """Log training progress."""
        if not self.episode_rewards:
            return
            
        recent_rewards = self.episode_rewards[-50:]
        recent_lengths = self.episode_lengths[-50:]
        
        avg_reward = np.mean(recent_rewards)
        avg_length = np.mean(recent_lengths)
        success_rate = self.success_count / max(self.total_episodes, 1)
        
        # Get reward system metrics
        system_metrics = self.reward_system.get_system_metrics()
        
        print(f"\n=== Training Progress (Episode {self.total_episodes}) ===")
        print(f"Recent Avg Reward: {avg_reward:.2f}")
        print(f"Recent Avg Length: {avg_length:.2f}")
        print(f"Success Rate: {success_rate:.2%}")
        print(f"Difficulty Level: {system_metrics.get('difficulty_level', 0)}")
        print(f"Learning Phase: {system_metrics.get('learning_phase', 'unknown')}")
        print(f"Reward Scale: {system_metrics.get('reward_scale', 1.0):.2f}")
        
        # Get analysis
        analysis = self.reward_analyzer.get_analysis()
        if analysis:
            print(f"Reward Trend: {analysis.get('reward_trend', 'unknown')}")
            print(f"Success Trend: {analysis.get('success_trend', 'unknown')}")
    
    def get_training_summary(self) -> Dict:
        """Get current training summary."""
        if not self.episode_rewards:
            return {
                'episode_count': self.total_episodes,
                'avg_reward': 0,
                'avg_length': 0,
                'success_rate': 0
            }
            
        recent_rewards = self.episode_rewards[-50:] if len(self.episode_rewards) >= 50 else self.episode_rewards
        recent_lengths = self.episode_lengths[-50:] if len(self.episode_lengths) >= 50 else self.episode_lengths
        
        return {
            'episode_count': self.total_episodes,
            'avg_reward': np.mean(recent_rewards),
            'avg_length': np.mean(recent_lengths),
            'success_rate': self.success_count / max(self.total_episodes, 1)
        }


class ImprovedTrainingManager:
    """Manages improved training with better configuration."""
    
    def __init__(self, config_path: str = None):
        self.config = self._load_config(config_path)
        self.model = None
        self.env = None
        self.reward_system = None
        self.training_callback = None
        
        # Setup logging
        self._setup_logging()
        
        # Create directories
        os.makedirs("models/enhanced", exist_ok=True)
        os.makedirs("logs/enhanced_training", exist_ok=True)
        
    def _load_config(self, config_path: str) -> Dict:
        """Load training configuration."""
        default_config = {
            'total_timesteps': 500000,
            'learning_rate': 3e-4,
            'n_steps': 2048,
            'batch_size': 64,
            'n_epochs': 10,
            'gamma': 0.99,
            'gae_lambda': 0.95,
            'clip_range': 0.2,
            'ent_coef': 0.01,
            'vf_coef': 0.5,
            'max_grad_norm': 0.5,
            'n_envs': 4,
            'eval_freq': 10000,
            'save_freq': 50000,
            'log_interval': 1000,
            'use_cnn': True,
            'curriculum_learning': True,
            'difficulty_start': 0,
            'difficulty_increase_threshold': 0.6,
        }
        
        if config_path and os.path.exists(config_path):
            # Load from file if provided
            import json
            with open(config_path, 'r') as f:
                file_config = json.load(f)
                default_config.update(file_config)
        
        return default_config
    
    def _setup_logging(self):
        """Setup logging for training."""
        log_dir = "logs/enhanced_training"
        os.makedirs(log_dir, exist_ok=True)
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(os.path.join(log_dir, 'training.log')),
                logging.StreamHandler()
            ]
        )
        
        self.logger = logging.getLogger("ImprovedTraining")
    
    def create_environment(self, timetable_data: Dict) -> TimetableEnv:
        """Create improved environment."""
        # Process timetable data
        timetable = config_processor(timetable_data)
        
        # Create environment
        env = TimetableEnv(
            timetable=timetable,
            max_steps=self.config.get('max_steps', 100),
            difficulty_level=self.config.get('difficulty_start', 0)
        )
        
        # Create reward system
        reward_func, reward_system = create_improved_reward_function(env)
        self.reward_system = reward_system
        
        # Wrap with action masker
        env = ActionMasker(env, mask_fn=env.get_action_mask)
        
        return env
    
    def create_model(self, env) -> MaskablePPO:
        """Create improved PPO model."""
        # Model configuration
        model_config = {
            'learning_rate': self.config['learning_rate'],
            'n_steps': self.config['n_steps'],
            'batch_size': self.config['batch_size'],
            'n_epochs': self.config['n_epochs'],
            'gamma': self.config['gamma'],
            'gae_lambda': self.config['gae_lambda'],
            'clip_range': self.config['clip_range'],
            'ent_coef': self.config['ent_coef'],
            'vf_coef': self.config['vf_coef'],
            'max_grad_norm': self.config['max_grad_norm'],
            'verbose': 1,
            'tensorboard_log': "logs/enhanced_training/tensorboard",
        }
        
        # Use CNN if specified
        if self.config.get('use_cnn', True):
            policy_kwargs = {
                'features_extractor_class': create_cnn_model,
                'features_extractor_kwargs': {
                    'num_courses': env.num_courses,
                    'num_slots': env.num_flat_slots,
                    'num_classrooms': env.num_classrooms
                }
            }
            model_config['policy_kwargs'] = policy_kwargs
        
        # Create model
        model = MaskablePPO(
            "MlpPolicy" if not self.config.get('use_cnn', True) else "CnnPolicy",
            env,
            **model_config
        )
        
        return model
    
    def train(self, timetable_data: Dict, model_path: str = None) -> str:
        """Train the model with improved configuration."""
        self.logger.info("Starting improved training...")
        
        # Create environment
        self.env = self.create_environment(timetable_data)
        
        # Create vectorized environment
        n_envs = self.config.get('n_envs', 4)
        if n_envs > 1:
            vec_env = make_vec_env(
                lambda: self.create_environment(timetable_data),
                n_envs=n_envs
            )
            vec_env = VecMonitor(vec_env, "logs/enhanced_training/monitor")
        else:
            vec_env = self.env
        
        # Create model
        self.model = self.create_model(vec_env)
        
        # Load existing model if provided
        if model_path and os.path.exists(model_path):
            self.model = self.model.load(model_path, env=vec_env)
            self.logger.info(f"Loaded model from {model_path}")
        
        # Create callbacks
        callbacks = []
        
        # Training callback
        self.training_callback = ImprovedTrainingCallback(self.reward_system)
        callbacks.append(self.training_callback)
        
        # Evaluation callback
        eval_env = self.create_environment(timetable_data)
        eval_callback = EvalCallback(
            eval_env,
            best_model_save_path="models/enhanced/",
            log_path="logs/enhanced_training/eval/",
            eval_freq=self.config['eval_freq'],
            deterministic=True,
            render=False
        )
        callbacks.append(eval_callback)
        
        # Start training
        start_time = time.time()
        
        try:
            self.model.learn(
                total_timesteps=self.config['total_timesteps'],
                callback=CallbackList(callbacks),
                progress_bar=True
            )
            
            training_time = time.time() - start_time
            self.logger.info(f"Training completed in {training_time:.2f} seconds")
            
        except KeyboardInterrupt:
            self.logger.info("Training interrupted by user")
        except Exception as e:
            self.logger.error(f"Training failed: {e}")
            raise
        
        # Save final model
        final_model_path = f"models/enhanced/enhanced_model_{int(time.time())}.zip"
        self.model.save(final_model_path)
        self.logger.info(f"Model saved to {final_model_path}")
        
        # Save training results
        self._save_training_results()
        
        return final_model_path
    
    def _save_training_results(self):
        """Save training results and analysis."""
        if not self.training_callback:
            return
            
        results = {
            'config': self.config,
            'total_episodes': self.training_callback.total_episodes,
            'success_count': self.training_callback.success_count,
            'final_success_rate': self.training_callback.success_count / max(self.training_callback.total_episodes, 1),
            'reward_analysis': self.training_callback.reward_analyzer.get_analysis(),
            'system_metrics': self.reward_system.get_system_metrics() if self.reward_system else {},
            'timestamp': datetime.now().isoformat()
        }
        
        # Save to file
        import json
        results_path = f"logs/enhanced_training/training_results_{int(time.time())}.json"
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        self.logger.info(f"Training results saved to {results_path}")
        
        # Plot rewards if possible
        try:
            plot_path = f"logs/enhanced_training/reward_plot_{int(time.time())}.png"
            self.training_callback.reward_analyzer.plot_rewards(plot_path)
            self.logger.info(f"Reward plot saved to {plot_path}")
        except Exception as e:
            self.logger.warning(f"Could not create reward plot: {e}")
    
    def evaluate_model(self, model_path: str, timetable_data: Dict, 
                      num_episodes: int = 10) -> Dict:
        """Evaluate the trained model."""
        self.logger.info(f"Evaluating model from {model_path}")
        
        # Create environment
        eval_env = self.create_environment(timetable_data)
        
        # Load model
        model = MaskablePPO.load(model_path, env=eval_env)
        
        # Run evaluation
        episode_rewards = []
        episode_lengths = []
        success_count = 0
        
        for episode in range(num_episodes):
            obs = eval_env.reset()
            done = False
            episode_reward = 0
            episode_length = 0
            
            while not done:
                action_masks = eval_env.action_masks()
                action, _ = model.predict(obs, action_masks=action_masks, deterministic=True)
                obs, reward, done, info = eval_env.step(action)
                episode_reward += reward
                episode_length += 1
            
            episode_rewards.append(episode_reward)
            episode_lengths.append(episode_length)
            
            # Check success
            if info.get('success', False):
                success_count += 1
            
            self.logger.info(f"Episode {episode + 1}: Reward={episode_reward:.2f}, "
                           f"Length={episode_length}, Success={info.get('success', False)}")
        
        # Calculate metrics
        eval_results = {
            'num_episodes': num_episodes,
            'avg_reward': np.mean(episode_rewards),
            'std_reward': np.std(episode_rewards),
            'avg_length': np.mean(episode_lengths),
            'success_rate': success_count / num_episodes,
            'episode_rewards': episode_rewards,
            'episode_lengths': episode_lengths
        }
        
        self.logger.info(f"Evaluation Results:")
        self.logger.info(f"  Average Reward: {eval_results['avg_reward']:.2f} ± {eval_results['std_reward']:.2f}")
        self.logger.info(f"  Average Length: {eval_results['avg_length']:.2f}")
        self.logger.info(f"  Success Rate: {eval_results['success_rate']:.2%}")
        
        return eval_results


async def run_training(data, job_id: str, use_cnn: bool = True, 
                      n_envs: int = 8, total_timesteps: int = 1000000):
    """Run enhanced training with parallel environments and advanced features."""
    
    # Set up logging
    log_dir = f"logs/{job_id}"
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, "enhanced_training.log")
    
    train_logger = logging.getLogger(f"enhanced_training_{job_id}")
    train_logger.setLevel(logging.INFO)
    
    # Clear existing handlers
    train_logger.handlers.clear()
    
    # File handler
    fh = logging.FileHandler(log_file)
    fh.setLevel(logging.INFO)
    
    # Console handler
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    
    # Formatter
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)
    
    train_logger.addHandler(fh)
    train_logger.addHandler(ch)
    
    train_logger.info(f"Starting enhanced training for job {job_id}")
    train_logger.info(f"Configuration: CNN={use_cnn}, N_ENVS={n_envs}, TIMESTEPS={total_timesteps}")
    
    from utils.job_manager import update_progress, add_log
    add_log(job_id, "Starting enhanced training initialization")
    update_progress(job_id, phase="Initializing Enhanced Training", percentage=5)
    
    # Create training manager
    manager = ImprovedTrainingManager()
    
    # Update manager config with provided parameters
    manager.config.update({
        'total_timesteps': total_timesteps,
        'n_envs': n_envs,
        'use_cnn': use_cnn
    })
    
    # Create environment
    add_log(job_id, "Creating enhanced environment")
    update_progress(job_id, phase="Setting up environment", percentage=10)
    
    # Process timetable data
    timetable_data = config_processor(data) if hasattr(data, '__dict__') else data
    
    # Create environment
    env = manager.create_environment(timetable_data)
    
    # Create vectorized environment
    if n_envs > 1:
        vec_env = make_vec_env(
            lambda: manager.create_environment(timetable_data),
            n_envs=n_envs
        )
        vec_env = VecMonitor(vec_env, f"logs/{job_id}/monitor")
    else:
        vec_env = env
    
    # Create model
    add_log(job_id, "Creating enhanced model")
    update_progress(job_id, phase="Creating enhanced model", percentage=15)
    
    model = manager.create_model(vec_env)
    
    # Create callbacks
    callbacks = []
    
    # Training callback
    training_callback = ImprovedTrainingCallback(manager.reward_system)
    callbacks.append(training_callback)
    
    # Evaluation callback
    eval_env = manager.create_environment(timetable_data)
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=f"models/enhanced/enhanced_best_model_{job_id}",
        log_path=f"logs/{job_id}/eval/",
        eval_freq=manager.config['eval_freq'],
        deterministic=True,
        render=False
    )
    callbacks.append(eval_callback)
    
    # Training configuration
    log_interval = manager.config.get('log_interval', 1000)
    start_time = time.time()
    
    train_logger.info(f"Enhanced Training Configuration:")
    train_logger.info(f"  - Total timesteps: {total_timesteps}")
    train_logger.info(f"  - Parallel environments: {n_envs}")
    train_logger.info(f"  - Log interval: {log_interval}")
    train_logger.info(f"  - Model type: {'Enhanced CNN' if use_cnn else 'Enhanced MLP'}")
    
    # Training loop
    add_log(job_id, "Starting enhanced training loop")
    update_progress(job_id, phase="Enhanced Training", percentage=20)
    
    try:
        for step in range(0, total_timesteps, log_interval):
            # Update progress
            current_step = step + log_interval
            progress_percentage = 20 + (current_step / total_timesteps) * 70
            update_progress(job_id, current_step=current_step, total_steps=total_timesteps,
                           phase="Enhanced Training", percentage=progress_percentage)
            
            # Train the model
            model.learn(total_timesteps=log_interval, callback=CallbackList(callbacks))
            
            # Get training summary
            training_summary = training_callback.get_training_summary()
            elapsed_time = time.time() - start_time
            
            # Log training progress
            log_msg = (
                f"Enhanced Step: {current_step} | "
                f"Elapsed: {elapsed_time:.2f}s | "
                f"Episodes: {training_callback.total_episodes} | "
                f"Success Rate: {training_callback.success_count / max(training_callback.total_episodes, 1):.2%}"
            )
            
            train_logger.info(log_msg)
            add_log(job_id, log_msg)
            yield log_msg
            
            # Save model checkpoint
            model_path = f"models/enhanced/enhanced_model_{job_id}_{current_step}.zip"
            model.save(model_path)
            
    except Exception as e:
        error_msg = f"Training failed: {str(e)}"
        train_logger.error(error_msg)
        add_log(job_id, error_msg)
        yield error_msg
        raise
    
    # Final metrics and model saving
    add_log(job_id, "Enhanced training completed successfully")
    update_progress(job_id, phase="Enhanced Training Completed", percentage=100)
    
    # Save final model
    final_model_path = f"models/enhanced/enhanced_model_{job_id}_final.zip"
    model.save(final_model_path)
    train_logger.info(f"Final model saved to {final_model_path}")
    
    # Generate final summary
    final_summary = {
        'total_episodes': training_callback.total_episodes,
        'success_count': training_callback.success_count,
        'final_success_rate': training_callback.success_count / max(training_callback.total_episodes, 1),
        'training_time': time.time() - start_time
    }
    
    train_logger.info("=" * 60)
    train_logger.info("ENHANCED TRAINING COMPLETED")
    train_logger.info("=" * 60)
    train_logger.info(f"Total Episodes: {final_summary['total_episodes']}")
    train_logger.info(f"Final Success Rate: {final_summary['final_success_rate']:.2%}")
    train_logger.info(f"Training Time: {final_summary['training_time']:.2f} seconds")
    train_logger.info("=" * 60)
    
    yield "Enhanced training completed successfully!"


def main():
    """Main function for improved training."""
    # Example usage
    manager = ImprovedTrainingManager()
    
    # Example timetable data (you would load this from your actual data)
    timetable_data = {
        # Your timetable configuration here
    }
    
    # Train model
    model_path = manager.train(timetable_data)
    
    # Evaluate model
    eval_results = manager.evaluate_model(model_path, timetable_data)
    
    print("Training completed successfully!")
    print(f"Model saved to: {model_path}")
    print(f"Evaluation results: {eval_results}")


if __name__ == "__main__":
    main()
