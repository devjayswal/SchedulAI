"""
Model Comparison Script
Compare MLP vs CNN performance on timetable scheduling task
"""

import os
import time
import torch
import numpy as np
import logging
from datetime import datetime

from sb3_contrib import MaskablePPO
from sb3_contrib.common.wrappers import ActionMasker
from ppo.env import TimetableEnv
from ppo.model import create_cnn_ppo_model, create_mlp_ppo_model, save_model
from stable_baselines3.common.env_util import make_vec_env
from ppo.cnn_hyperparams import get_config, get_cnn_model_path, get_mlp_model_path
from utils.config_processor import config_processor


def create_test_environment():
    """Create a test environment for model comparison."""
    from models.Timetable import Timetable
    from models.Course import Course
    from models.Faculty import Faculty
    from models.Classroom import Classroom
    from models.Branch import Branch
    
    # Create test timetable
    timetable = Timetable()
    timetable.days = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"]
    timetable.time_slots = ["09:00-10:00", "10:00-11:00", "11:00-12:00", "12:00-13:00", "14:00-15:00", "15:00-16:00"]
    
    # Add test faculty
    faculty1 = Faculty("F001", "Dr. Smith", "smith@example.com")
    faculty2 = Faculty("F002", "Dr. Johnson", "johnson@example.com")
    faculty3 = Faculty("F003", "Dr. Brown", "brown@example.com")
    timetable.faculty = [faculty1, faculty2, faculty3]
    
    # Add test classrooms
    classroom1 = Classroom("C001", "Room 101", "theory", 50)
    classroom2 = Classroom("C002", "Room 102", "theory", 40)
    classroom3 = Classroom("C003", "Lab 201", "lab", 30)
    timetable.classrooms = [classroom1, classroom2, classroom3]
    
    # Add test courses
    courses = [
        Course("C001", "Mathematics", 3, "theory", faculty1.short_name),
        Course("C002", "Physics", 3, "theory", faculty2.short_name),
        Course("C003", "Chemistry Lab", 2, "lab", faculty3.short_name),
        Course("C004", "Computer Science", 3, "theory", faculty1.short_name),
        Course("C005", "Data Structures", 3, "theory", faculty2.short_name),
    ]
    timetable.courses = courses
    
    # Add test branch
    branch = Branch("CS", "Computer Science", 3)
    branch.courses = courses
    timetable.branches = [branch]
    
    # Initialize timetables
    timetable.initialize_timetables()
    
    return timetable


def train_model(model, env, model_name, timesteps=10000, log_interval=1000):
    """Train a model and return training metrics."""
    print(f"\nTraining {model_name} model...")
    
    start_time = time.time()
    rewards = []
    losses = []
    
    for step in range(0, timesteps, log_interval):
        model.learn(total_timesteps=log_interval)
        
        # Get metrics
        try:
            reward = model.logger.name_to_value.get('ep_rew_mean', 0)
            loss = model.logger.name_to_value.get('loss', 0)
            rewards.append(reward)
            losses.append(loss)
        except:
            rewards.append(0)
            losses.append(0)
        
        elapsed = time.time() - start_time
        print(f"  {model_name} Step {step + log_interval}: Reward={reward:.2f}, Loss={loss:.4f}, Time={elapsed:.1f}s")
    
    total_time = time.time() - start_time
    return {
        'model_name': model_name,
        'total_time': total_time,
        'final_reward': rewards[-1] if rewards else 0,
        'avg_reward': np.mean(rewards) if rewards else 0,
        'final_loss': losses[-1] if losses else 0,
        'avg_loss': np.mean(losses) if losses else 0,
        'rewards': rewards,
        'losses': losses
    }


def evaluate_model(model, env, model_name, num_episodes=10):
    """Evaluate a trained model."""
    print(f"\nEvaluating {model_name} model...")
    
    episode_rewards = []
    episode_lengths = []
    successful_episodes = 0
    
    for episode in range(num_episodes):
        obs = env.reset()[0]
        total_reward = 0
        steps = 0
        done = False
        
        while not done and steps < 100:  # Max 100 steps per episode
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, truncated, info = env.step(action)
            total_reward += reward
            steps += 1
            
            if truncated:
                done = True
        
        episode_rewards.append(total_reward)
        episode_lengths.append(steps)
        
        if total_reward > 0:  # Consider positive reward as successful
            successful_episodes += 1
        
        print(f"  Episode {episode + 1}: Reward={total_reward:.2f}, Steps={steps}")
    
    return {
        'model_name': model_name,
        'avg_reward': np.mean(episode_rewards),
        'std_reward': np.std(episode_rewards),
        'avg_length': np.mean(episode_lengths),
        'success_rate': successful_episodes / num_episodes,
        'episode_rewards': episode_rewards,
        'episode_lengths': episode_lengths
    }


def compare_models():
    """Compare MLP vs CNN models."""
    print("=" * 60)
    print("MODEL COMPARISON: MLP vs CNN")
    print("=" * 60)
    
    # Create test environment
    timetable = create_test_environment()
    
    # Define action mask function
    def mask_fn(env: TimetableEnv) -> np.ndarray:
        return env.get_action_mask()
    
    def make_env():
        return ActionMasker(TimetableEnv(timetable), mask_fn)
    
    env = make_vec_env(make_env, n_envs=1)
    
    # Get configurations
    mlp_config = get_config("mlp")
    cnn_config = get_config("cnn")
    
    # Training parameters
    training_timesteps = 20000  # Reduced for faster comparison
    evaluation_episodes = 5
    
    results = {}
    
    # Train and evaluate MLP model
    print("\n" + "=" * 40)
    print("MLP MODEL")
    print("=" * 40)
    
    mlp_model = create_mlp_ppo_model(
        env, 
        policy_kwargs=mlp_config["policy_kwargs"],
        **mlp_config["ppo_kwargs"]
    )
    
    mlp_training_results = train_model(
        mlp_model, env, "MLP", 
        timesteps=training_timesteps, 
        log_interval=5000
    )
    
    mlp_eval_results = evaluate_model(mlp_model, env, "MLP", evaluation_episodes)
    results['mlp'] = {**mlp_training_results, **mlp_eval_results}
    
    # Save MLP model
    save_model(mlp_model, get_mlp_model_path("comparison"))
    
    # Train and evaluate CNN model
    print("\n" + "=" * 40)
    print("CNN MODEL")
    print("=" * 40)
    
    cnn_model = create_cnn_ppo_model(env, **cnn_config["ppo_kwargs"])
    
    cnn_training_results = train_model(
        cnn_model, env, "CNN", 
        timesteps=training_timesteps, 
        log_interval=5000
    )
    
    cnn_eval_results = evaluate_model(cnn_model, env, "CNN", evaluation_episodes)
    results['cnn'] = {**cnn_training_results, **cnn_eval_results}
    
    # Save CNN model
    save_model(cnn_model, get_cnn_model_path("comparison"))
    
    # Print comparison results
    print("\n" + "=" * 60)
    print("COMPARISON RESULTS")
    print("=" * 60)
    
    print(f"{'Metric':<20} {'MLP':<15} {'CNN':<15} {'Winner':<10}")
    print("-" * 60)
    
    metrics = [
        ('Training Time (s)', 'total_time'),
        ('Final Reward', 'final_reward'),
        ('Avg Reward', 'avg_reward'),
        ('Final Loss', 'final_loss'),
        ('Avg Loss', 'avg_loss'),
        ('Eval Avg Reward', 'avg_reward'),
        ('Success Rate', 'success_rate'),
        ('Avg Episode Length', 'avg_length')
    ]
    
    for metric_name, metric_key in metrics:
        mlp_value = results['mlp'].get(metric_key, 0)
        cnn_value = results['cnn'].get(metric_key, 0)
        
        if 'time' in metric_name.lower() or 'loss' in metric_name.lower():
            # Lower is better
            winner = "MLP" if mlp_value < cnn_value else "CNN"
        else:
            # Higher is better
            winner = "MLP" if mlp_value > cnn_value else "CNN"
        
        print(f"{metric_name:<20} {mlp_value:<15.4f} {cnn_value:<15.4f} {winner:<10}")
    
    # Save results to file
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = f"model_comparison_{timestamp}.txt"
    
    with open(results_file, 'w') as f:
        f.write("Model Comparison Results\n")
        f.write("=" * 60 + "\n")
        f.write(f"Timestamp: {timestamp}\n")
        f.write(f"Training Timesteps: {training_timesteps}\n")
        f.write(f"Evaluation Episodes: {evaluation_episodes}\n\n")
        
        for model_name, model_results in results.items():
            f.write(f"{model_name.upper()} MODEL RESULTS:\n")
            f.write("-" * 30 + "\n")
            for key, value in model_results.items():
                if isinstance(value, (list, np.ndarray)):
                    f.write(f"{key}: {len(value)} values\n")
                else:
                    f.write(f"{key}: {value}\n")
            f.write("\n")
    
    print(f"\nDetailed results saved to: {results_file}")
    print("\nComparison complete!")
    
    return results


if __name__ == "__main__":
    compare_models()
