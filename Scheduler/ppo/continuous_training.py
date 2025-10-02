"""
Continuous Training System for Timetable Scheduling
This module implements a continuous training approach where models are trained once
and then continue learning from subsequent timetables without starting from scratch.
"""

import os
import time
import torch
import asyncio
import numpy as np
import logging
from typing import Dict, Any, Optional, Tuple
from datetime import datetime

from sb3_contrib import MaskablePPO
from sb3_contrib.common.wrappers import ActionMasker
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import EvalCallback, BaseCallback, CallbackList

from ppo.env import TimetableEnv
from ppo.cnn_model import create_cnn_model
from ppo.reward_shaping import AdaptiveRewardSystem, create_improved_reward_function
from ppo.advanced_metrics import AdvancedMetrics, MetricsCallback
from ppo.model import save_model, load_model
from utils.config_processor import config_processor
from utils.job_manager import update_progress, add_log
from utils.model_manager import get_model_manager
from utils.gpu_utils import get_gpu_manager, optimize_config_for_gpu


class ContinuousTrainingManager:
    """Manages continuous training across multiple timetables."""
    
    def __init__(self, base_model_path: str = "models/continuous/continuous_model.pth"):
        self.base_model_path = base_model_path
        self.current_model = None
        self.training_history = []
        self.global_metrics = AdvancedMetrics(log_dir="logs/continuous_training")
        self.is_first_training = True
        self.model_manager = get_model_manager()
        
        # Create models directory structure if it doesn't exist
        os.makedirs("models/continuous", exist_ok=True)
        os.makedirs("models/enhanced", exist_ok=True)
        os.makedirs("models/legacy", exist_ok=True)
        os.makedirs("models/checkpoints", exist_ok=True)
        os.makedirs("logs/continuous_training", exist_ok=True)
        
        # Setup logging
        self.setup_logging()
    
    def setup_logging(self):
        """Setup continuous training logging."""
        self.logger = logging.getLogger("ContinuousTraining")
        self.logger.setLevel(logging.INFO)
        
        # File handler
        log_file = "logs/continuous_training/continuous_training.log"
        fh = logging.FileHandler(log_file)
        fh.setLevel(logging.INFO)
        
        # Console handler
        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)
        
        # Formatter
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        fh.setFormatter(formatter)
        ch.setFormatter(formatter)
        
        if not self.logger.handlers:
            self.logger.addHandler(fh)
            self.logger.addHandler(ch)
    
    def get_model_path(self, job_id: str = None) -> str:
        """Get the appropriate model path for continuous training."""
        if job_id:
            return f"models/continuous/continuous_model_{job_id}.pth"
        return self.base_model_path
    
    def load_or_create_model(self, env, job_id: str, use_cnn: bool = True, 
                           n_envs: int = 8, **kwargs) -> MaskablePPO:
        """Load existing model or create new one for continuous training."""
        
        model_path = self.get_model_path(job_id)
        
        if self.is_first_training or not os.path.exists(self.base_model_path):
            # First training or no base model exists
            self.logger.info("Creating new model for continuous training")
            add_log(job_id, "Creating new continuous training model")
            
            if use_cnn:
                model = create_cnn_model(env, model_path)
                self.logger.info("Created new Enhanced CNN model")
            else:
                from ppo.model import create_mlp_ppo_model
                from ppo.cnn_hyperparams import get_config
                config = get_config("mlp")
                model = create_mlp_ppo_model(env, **config["ppo_kwargs"])
                self.logger.info("Created new Enhanced MLP model")
            
            self.is_first_training = False
            
        else:
            # Load existing model for continuous training
            self.logger.info(f"Loading existing model from {self.base_model_path}")
            add_log(job_id, "Loading existing model for continuous training")
            
            try:
                if use_cnn:
                    model = create_cnn_model(env, model_path)
                    # Load the base model weights
                    checkpoint = torch.load(self.base_model_path)
                    model.policy.load_state_dict(checkpoint['policy_state_dict'])
                    if checkpoint.get('optimizer_state_dict'):
                        model.policy.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                else:
                    from ppo.model import create_mlp_ppo_model
                    from ppo.cnn_hyperparams import get_config
                    config = get_config("mlp")
                    model = create_mlp_ppo_model(env, **config["ppo_kwargs"])
                    # Load the base model weights
                    checkpoint = torch.load(self.base_model_path)
                    model.policy.load_state_dict(checkpoint['policy_state_dict'])
                    if checkpoint.get('optimizer_state_dict'):
                        model.policy.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                
                self.logger.info("Successfully loaded existing model for continuous training")
                add_log(job_id, "Successfully loaded existing model")
                
            except Exception as e:
                self.logger.warning(f"Failed to load existing model: {e}. Creating new model.")
                add_log(job_id, f"Failed to load existing model: {e}. Creating new model.")
                
                if use_cnn:
                    model = create_cnn_model(env, model_path)
                else:
                    from ppo.model import create_mlp_ppo_model
                    from ppo.cnn_hyperparams import get_config
                    config = get_config("mlp")
                    model = create_mlp_ppo_model(env, **config["ppo_kwargs"])
        
        self.current_model = model
        return model
    
    def save_continuous_model(self, model: MaskablePPO, job_id: str = None):
        """Save the model for continuous training with metadata."""
        model_path = self.get_model_path(job_id)
        
        # Prepare metadata
        metadata = {
            "job_id": job_id,
            "training_type": "continuous",
            "model_type": "enhanced_cnn" if hasattr(model, 'policy') and 'cnn' in str(type(model.policy)).lower() else "enhanced_mlp",
            "episode_count": getattr(self, 'episode_count', 0),
            "training_steps": getattr(self, 'training_steps', 0),
            "performance_metrics": getattr(self.global_metrics, 'get_summary', lambda: {})(),
            "is_base_model": job_id is None
        }
        
        # Save job-specific model with metadata
        self.model_manager.save_model_with_metadata(model, model_path, metadata)
        
        # Update base model for next training
        base_metadata = metadata.copy()
        base_metadata["is_base_model"] = True
        self.model_manager.save_model_with_metadata(model, self.base_model_path, base_metadata)
        
        # Create checkpoint
        self.model_manager.create_checkpoint(model, job_id or "base", "continuous")
        
        self.logger.info(f"Saved continuous model to {model_path} and base model to {self.base_model_path}")
    
    def should_continue_training(self, metrics: AdvancedMetrics, 
                               min_success_rate: float = 0.8) -> Tuple[bool, str]:
        """Determine if training should continue based on performance."""
        performance_status = metrics.get_performance_status()
        
        if performance_status['overall_performance'] == 'excellent':
            return False, "Training completed - excellent performance achieved"
        elif performance_status['overall_performance'] == 'good':
            # Check if we have enough episodes for a good assessment
            if metrics.episode_count >= 50:
                return False, "Training completed - good performance achieved"
            else:
                return True, "Continuing training - need more episodes for assessment"
        else:
            return True, "Continuing training - performance needs improvement"
    
    def get_adaptive_timesteps(self, base_timesteps: int, performance_status: Dict) -> int:
        """Adapt training timesteps based on performance."""
        if performance_status['overall_performance'] == 'excellent':
            return min(base_timesteps, 50000)  # Reduce training for excellent performance
        elif performance_status['overall_performance'] == 'good':
            return base_timesteps  # Standard training
        else:
            return int(base_timesteps * 1.5)  # Increase training for poor performance
    
    def get_fastest_training_method(self) -> Tuple[bool, bool]:
        """Determine the fastest training method based on system capabilities."""
        # Check if CNN is available and faster
        try:
            import torch
            if torch.cuda.is_available():
                # GPU available - CNN is faster
                return True, True  # use_cnn=True, use_enhanced=True
            else:
                # CPU only - MLP might be faster for small problems
                return False, True  # use_cnn=False, use_enhanced=True
        except:
            # Fallback to enhanced MLP
            return False, True


class ContinuousTrainingCallback(BaseCallback):
    """Enhanced callback for continuous training with conflict detection."""
    
    def __init__(self, metrics_tracker: AdvancedMetrics, reward_shaper: AdaptiveRewardSystem,
                 continuous_manager: ContinuousTrainingManager, job_id: str,
                 verbose: int = 0):
        super(ContinuousTrainingCallback, self).__init__(verbose)
        self.metrics_tracker = metrics_tracker
        self.reward_shaper = reward_shaper
        self.continuous_manager = continuous_manager
        self.job_id = job_id
        self.episode_rewards = []
        self.episode_lengths = []
        self.current_episode_reward = 0
        self.current_episode_length = 0
        self.episode_count = 0
        self.conflict_free_episodes = 0
        self.best_timetable = None
        self.best_reward = float('-inf')
        
    def _on_step(self) -> bool:
        """Called at each step."""
        # Track episode progress
        if len(self.locals.get('rewards', [])) > 0:
            self.current_episode_reward += self.locals['rewards'][0]
            self.current_episode_length += 1
        
        # Check if episode is done
        if len(self.locals.get('dones', [])) > 0 and self.locals['dones'][0]:
            self._on_episode_end()
        
        return True
    
    def _on_episode_end(self):
        """Called at the end of each episode."""
        if self.current_episode_length > 0:
            # Determine success (positive reward indicates success)
            success = self.current_episode_reward > 0
            
            # Check if this is a conflict-free timetable
            is_conflict_free = self.current_episode_reward > 50  # Threshold for conflict-free
            if is_conflict_free:
                self.conflict_free_episodes += 1
                
                # Store best timetable
                if self.current_episode_reward > self.best_reward:
                    self.best_reward = self.current_episode_reward
                    self.best_timetable = {
                        'reward': self.current_episode_reward,
                        'episode': self.episode_count,
                        'timestamp': datetime.now().isoformat(),
                        'job_id': self.job_id
                    }
            
            # Update metrics
            self.metrics_tracker.end_episode(
                success=success,
                courses_completed=int(self.current_episode_reward / 10),
                total_courses=5
            )
            
            # Update reward shaper with episode results
            self.reward_shaper.update_episode(success, self.current_episode_reward)
            
            # Store episode data
            self.episode_rewards.append(self.current_episode_reward)
            self.episode_lengths.append(self.current_episode_length)
            self.episode_count += 1
            
            # Reset for next episode
            self.current_episode_reward = 0
            self.current_episode_length = 0
    
    def get_training_summary(self) -> Dict:
        """Get current training summary."""
        return {
            'episode_count': self.episode_count,
            'conflict_free_episodes': self.conflict_free_episodes,
            'conflict_free_rate': self.conflict_free_episodes / max(self.episode_count, 1),
            'avg_reward': np.mean(self.episode_rewards) if self.episode_rewards else 0,
            'avg_length': np.mean(self.episode_lengths) if self.episode_lengths else 0,
            'difficulty_level': self.reward_shaper.reward_shaper.difficulty_level,
            'success_rate': np.mean([r > 0 for r in self.episode_rewards[-50:]]) if len(self.episode_rewards) >= 50 else 0,
            'best_timetable': self.best_timetable
        }


async def run_continuous_training(data, job_id: str, use_cnn: bool = None, 
                                n_envs: int = 8, total_timesteps: int = 1000000):
    """Run continuous training with model persistence and conflict detection."""
    
    # Initialize GPU manager and log GPU status
    gpu_manager = get_gpu_manager()
    if gpu_manager.gpu_info["available"]:
        add_log(job_id, f"GPU training enabled: {gpu_manager.gpu_info['device_name']}")
    else:
        add_log(job_id, "GPU not available, using CPU training")
    
    # Initialize continuous training manager
    continuous_manager = ContinuousTrainingManager()
    
    # Auto-detect fastest training method if not specified
    if use_cnn is None:
        use_cnn, use_enhanced = continuous_manager.get_fastest_training_method()
        train_logger = logging.getLogger(f"continuous_training_{job_id}")
        train_logger.info(f"Auto-detected fastest method: CNN={use_cnn}, Enhanced={use_enhanced}")
    else:
        use_enhanced = True  # Always use enhanced features with continuous training
    
    # Set up logging
    log_dir = f"logs/{job_id}"
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, "continuous_training.log")
    
    train_logger = logging.getLogger(f"continuous_training_{job_id}")
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
    
    train_logger.info(f"Starting continuous training for job {job_id}")
    train_logger.info(f"Configuration: CNN={use_cnn}, N_ENVS={n_envs}, TIMESTEPS={total_timesteps}")
    
    add_log(job_id, "Starting continuous training initialization")
    update_progress(job_id, phase="Initializing Continuous Training", percentage=5)
    
    # Initialize metrics tracker
    metrics_tracker = AdvancedMetrics(log_dir=f"logs/{job_id}/metrics")
    
    # Create enhanced reward function first
    reward_shaper = AdaptiveRewardSystem(
        data.num_courses if hasattr(data, 'num_courses') else 5,
        data.num_flat_slots if hasattr(data, 'num_flat_slots') else 30,
        data.num_classrooms if hasattr(data, 'num_classrooms') else 5
    )
    
    add_log(job_id, "Enhanced reward system initialized with AdaptiveRewardSystem")
    train_logger.info("Enhanced reward system initialized with AdaptiveRewardSystem")
    
    # Define action mask function
    def mask_fn(env: TimetableEnv) -> np.ndarray:
        return env.get_action_mask()
    
    # Create parallel environments
    add_log(job_id, f"Creating {n_envs} parallel environments")
    update_progress(job_id, phase="Setting up parallel environments", percentage=10)
    
    def make_env():
        # Use the enhanced TimetableEnv directly (now it's the default env.py)
        return ActionMasker(TimetableEnv(data), mask_fn)
    
    env = make_vec_env(make_env, n_envs=n_envs)
    
    # Load or create model for continuous training
    add_log(job_id, "Loading/creating model for continuous training")
    update_progress(job_id, phase="Loading model for continuous training", percentage=15)
    
    model = continuous_manager.load_or_create_model(env, job_id, use_cnn, n_envs)
    
    # Set up callbacks
    eval_cb = EvalCallback(
        env,
        eval_freq=10000,
        best_model_save_path=f"models/continuous/continuous_best_model_{job_id}",
        log_path=f"logs/{job_id}/eval",
        deterministic=True,
        render=False
    )
    
    continuous_cb = ContinuousTrainingCallback(metrics_tracker, reward_shaper, 
                                             continuous_manager, job_id)
    metrics_cb = MetricsCallback(metrics_tracker)
    
    callback_list = CallbackList([eval_cb, continuous_cb, metrics_cb])
    
    # Training configuration
    log_interval = 5000
    start_time = time.time()
    
    train_logger.info(f"Continuous Training Configuration:")
    train_logger.info(f"  - Total timesteps: {total_timesteps}")
    train_logger.info(f"  - Parallel environments: {n_envs}")
    train_logger.info(f"  - Log interval: {log_interval}")
    train_logger.info(f"  - Model type: {'Enhanced CNN' if use_cnn else 'Enhanced MLP'}")
    train_logger.info(f"  - Continuous training: {'First training' if continuous_manager.is_first_training else 'Continuing from previous model'}")
    
    # Training loop
    add_log(job_id, "Starting continuous training loop")
    update_progress(job_id, phase="Continuous Training", percentage=20)
    
    for step in range(0, total_timesteps, log_interval):
        # Update progress
        current_step = step + log_interval
        progress_percentage = 20 + (current_step / total_timesteps) * 70
        update_progress(job_id, current_step=current_step, total_steps=total_timesteps,
                       phase="Continuous Training", percentage=progress_percentage)
        
        # Train the model
        model.learn(total_timesteps=log_interval, callback=callback_list)
        
        # Get training summary
        training_summary = continuous_cb.get_training_summary()
        elapsed_time = time.time() - start_time
        
        # Get performance status
        performance_status = metrics_tracker.get_performance_status()
        
        # Check if we should continue training
        should_continue, continue_reason = continuous_manager.should_continue_training(metrics_tracker)
        
        # Log training progress
        log_msg = (
            f"Continuous Step: {current_step} | "
            f"Elapsed: {elapsed_time:.2f}s | "
            f"Episodes: {training_summary['episode_count']} | "
            f"Conflict-free: {training_summary['conflict_free_episodes']} | "
            f"Conflict-free Rate: {training_summary['conflict_free_rate']:.2%} | "
            f"Avg Reward: {training_summary['avg_reward']:.2f} | "
            f"Success Rate: {training_summary['success_rate']:.2%} | "
            f"Performance: {performance_status['overall_performance']} | "
            f"Continue: {should_continue}"
        )
        
        train_logger.info(log_msg)
        add_log(job_id, log_msg)
        yield log_msg
        
        # Save model checkpoint
        continuous_manager.save_continuous_model(model, job_id)
        
        # Save metrics periodically
        if current_step % (log_interval * 4) == 0:
            metrics_file = metrics_tracker.save_metrics(f"continuous_metrics_checkpoint_{current_step}.json")
            train_logger.info(f"Metrics saved to {metrics_file}")
        
        # Check for conflict-free timetable
        if training_summary['best_timetable']:
            best_msg = (
                f"Best conflict-free timetable found! "
                f"Reward: {training_summary['best_timetable']['reward']:.2f}, "
                f"Episode: {training_summary['best_timetable']['episode']}"
            )
            train_logger.info(best_msg)
            add_log(job_id, best_msg)
            yield best_msg
        
        # Early stopping if excellent performance achieved
        if not should_continue and performance_status['overall_performance'] == 'excellent':
            train_logger.info(f"Early stopping: {continue_reason}")
            add_log(job_id, f"Early stopping: {continue_reason}")
            yield f"Early stopping: {continue_reason}"
            break
        
        # Print warnings and recommendations
        if performance_status['warnings']:
            for warning in performance_status['warnings']:
                train_logger.warning(f"Performance Warning: {warning}")
        
        if performance_status['recommendations']:
            for recommendation in performance_status['recommendations']:
                train_logger.info(f"Recommendation: {recommendation}")
    
    # Final metrics and model saving
    add_log(job_id, "Continuous training completed successfully")
    update_progress(job_id, phase="Continuous Training Completed", percentage=100)
    
    # Save final metrics
    final_metrics_file = metrics_tracker.save_metrics("continuous_final_metrics.json")
    train_logger.info(f"Final metrics saved to {final_metrics_file}")
    
    # Save final model
    continuous_manager.save_continuous_model(model, job_id)
    train_logger.info(f"Final continuous model saved")
    
    # Generate final summary
    final_summary = metrics_tracker.get_summary()
    performance_status = metrics_tracker.get_performance_status()
    training_summary = continuous_cb.get_training_summary()
    
    train_logger.info("=" * 60)
    train_logger.info("CONTINUOUS TRAINING COMPLETED")
    train_logger.info("=" * 60)
    train_logger.info(f"Total Episodes: {continuous_cb.episode_count}")
    train_logger.info(f"Conflict-free Episodes: {training_summary['conflict_free_episodes']}")
    train_logger.info(f"Conflict-free Rate: {training_summary['conflict_free_rate']:.2%}")
    train_logger.info(f"Final Success Rate: {training_summary['success_rate']:.2%}")
    train_logger.info(f"Final Difficulty Level: {training_summary['difficulty_level']}")
    train_logger.info(f"Overall Performance: {performance_status['overall_performance']}")
    train_logger.info(f"Training Time: {elapsed_time:.2f} seconds")
    
    if training_summary['best_timetable']:
        train_logger.info(f"Best Timetable Reward: {training_summary['best_timetable']['reward']:.2f}")
        yield f"Best conflict-free timetable found with reward: {training_summary['best_timetable']['reward']:.2f}"
    
    if performance_status['success_indicators']:
        train_logger.info("Success Indicators:")
        for indicator in performance_status['success_indicators']:
            train_logger.info(f"  [OK] {indicator}")
    
    if performance_status['warnings']:
        train_logger.info("Warnings:")
        for warning in performance_status['warnings']:
            train_logger.info(f"  [WARNING] {warning}")
    
    if performance_status['recommendations']:
        train_logger.info("Recommendations:")
        for recommendation in performance_status['recommendations']:
            train_logger.info(f"  [TIP] {recommendation}")
    
    train_logger.info("=" * 60)
    
    # Generate the actual timetable using the trained model
    yield "Generating final timetable..."
    generated_timetable = await generate_timetable_with_model(model, data, job_id)
    
    if generated_timetable:
        yield f"Timetable generated successfully with {generated_timetable.get('total_sessions', 0)} sessions"
    else:
        yield "Warning: Could not generate timetable"
    
    yield "Continuous training completed successfully!"


async def generate_timetable_with_model(model, data, job_id: str):
    """Generate the actual timetable using the trained model."""
    try:
        from utils.database import db
        from bson import ObjectId
        
        # Create a single environment for generation
        def make_env():
            return ActionMasker(TimetableEnv(data), lambda env: env.get_action_mask())
        
        env = make_vec_env(make_env, n_envs=1)
        
        # Generate timetable
        obs = env.reset()
        done = False
        total_reward = 0
        steps = 0
        max_steps = 200  # Prevent infinite loops
        
        timetable_entries = []
        
        while not done and steps < max_steps:
            # Get action from trained model
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, info = env.step(action)
            
            total_reward += reward[0] if isinstance(reward, np.ndarray) else reward
            steps += 1
            
            # Extract timetable information from environment state
            if hasattr(env.envs[0], 'env') and hasattr(env.envs[0].env, 'timetable'):
                # Get the current state of the timetable
                current_timetable = env.envs[0].env.timetable
                if current_timetable:
                    # Convert timetable to serializable format
                    timetable_data = {
                        'job_id': job_id,
                        'total_reward': float(total_reward),
                        'steps_taken': steps,
                        'courses': [],
                        'faculty': [],
                        'classrooms': [],
                        'branches': [],
                        'time_slots': current_timetable.time_slots,
                        'days': current_timetable.days,
                        'timetables': {}
                    }
                    
                    # Extract course information
                    for course in current_timetable.courses:
                        timetable_data['courses'].append({
                            'subject_code': course.subject_code,
                            'subject_name': course.subject_name,
                            'credits': course.credits,
                            'faculty_id': course.faculty_id,
                            'course_type': getattr(course, 'course_type', 'theory')
                        })
                    
                    # Extract faculty information
                    for faculty in current_timetable.faculty:
                        timetable_data['faculty'].append({
                            'short_name': faculty.short_name,
                            'full_name': faculty.full_name,
                            'department': getattr(faculty, 'department', ''),
                            'email': getattr(faculty, 'email', '')
                        })
                    
                    # Extract classroom information
                    for classroom in current_timetable.classrooms:
                        timetable_data['classrooms'].append({
                            'room_number': classroom.room_number,
                            'capacity': classroom.capacity,
                            'room_type': getattr(classroom, 'room_type', 'general')
                        })
                    
                    # Extract branch information
                    for branch in current_timetable.branches:
                        branch_data = {
                            'branch_name': branch.branch_name,
                            'semester': branch.semester,
                            'courses': [course.subject_code for course in branch.courses]
                        }
                        timetable_data['branches'].append(branch_data)
                    
                    # Extract timetable entries
                    for branch_sem, class_timetable in current_timetable.timetables.items():
                        timetable_data['timetables'][branch_sem] = []
                        for entry in class_timetable.entries:
                            timetable_data['timetables'][branch_sem].append({
                                'course_code': entry.course_code,
                                'faculty': entry.faculty,
                                'day': entry.day,
                                'time_slot': entry.time_slot,
                                'classroom': entry.classroom,
                                'course_type': getattr(entry, 'course_type', 'theory')
                            })
                    
                    # Save to database
                    timetable_collection = db["GeneratedTimetables"]
                    result = timetable_collection.insert_one(timetable_data)
                    
                    return {
                        'timetable_id': str(result.inserted_id),
                        'total_sessions': len([entry for entries in timetable_data['timetables'].values() for entry in entries]),
                        'total_reward': total_reward,
                        'steps_taken': steps
                    }
        
        return None
        
    except Exception as e:
        print(f"Error generating timetable: {e}")
        return None


async def run_continuous_training_sync(data, job_id: str, use_cnn: bool = True, 
                               n_envs: int = 8, total_timesteps: int = 1000000):
    """Synchronous wrapper for continuous training."""
    results = []
    async for msg in run_continuous_training(data, job_id, use_cnn, n_envs, total_timesteps):
        results.append(msg)
    return results


# Example usage and testing
def test_continuous_training():
    """Test continuous training functionality."""
    print("Testing Continuous Training...")
    
    try:
        # Create test data
        from models.Timetable import Timetable
        from models.Course import Course
        from models.Faculty import Faculty
        from models.Classroom import Classroom
        
        timetable = Timetable()
        timetable.days = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"]
        timetable.time_slots = ["09:00-10:00", "10:00-11:00", "11:00-12:00", "12:00-13:00", "14:00-15:00", "15:00-16:00"]
        
        faculty = Faculty("F001", "Test Faculty", "test@example.com")
        classroom = Classroom("C001", "Test Room", "theory")
        
        for i in range(3):
            course = Course(f"C{i+1}", f"Course {i+1}", 3, "theory", faculty.short_name)
            timetable.courses.append(course)
        
        timetable.faculty = [faculty]
        timetable.classrooms = [classroom]
        
        # Test continuous training (short run)
        print("Running short continuous training test...")
        results = asyncio.run(run_continuous_training_sync(
            timetable, "test_continuous_job", use_cnn=True, n_envs=2, total_timesteps=1000
        ))
        
        print(f"Continuous training test completed with {len(results)} log messages")
        
        return True
        
    except Exception as e:
        print(f"Continuous training test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    # Run tests
    test_continuous_training()
