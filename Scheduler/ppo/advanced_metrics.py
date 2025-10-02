"""
Advanced Metrics Tracking System for PPO Timetable Training
This module provides comprehensive metrics tracking and analysis.
"""

import numpy as np
import torch
import json
import os
from datetime import datetime
from typing import Dict, List, Any, Optional
from collections import defaultdict, deque
import logging
from stable_baselines3.common.callbacks import BaseCallback


class AdvancedMetrics:
    """Advanced metrics tracking for timetable scheduling."""
    
    def __init__(self, log_dir: str = "logs/metrics"):
        self.log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)
        
        # Initialize metrics storage
        self.metrics = {
            # Training metrics
            'episode_rewards': deque(maxlen=1000),
            'episode_lengths': deque(maxlen=1000),
            'training_losses': deque(maxlen=1000),
            'value_losses': deque(maxlen=1000),
            'policy_losses': deque(maxlen=1000),
            'entropy_losses': deque(maxlen=1000),
            
            # Performance metrics
            'success_rates': deque(maxlen=100),
            'constraint_violation_rates': deque(maxlen=100),
            'utilization_efficiency': deque(maxlen=100),
            'faculty_load_balance': deque(maxlen=100),
            'classroom_balance': deque(maxlen=100),
            'time_distribution_score': deque(maxlen=100),
            
            # Learning metrics
            'learning_rates': deque(maxlen=100),
            'gradient_norms': deque(maxlen=100),
            'kl_divergences': deque(maxlen=100),
            'clip_fractions': deque(maxlen=100),
            
            # Environment metrics
            'courses_completed': deque(maxlen=100),
            'courses_failed': deque(maxlen=100),
            'average_episode_time': deque(maxlen=100),
            'action_mask_efficiency': deque(maxlen=100),
        }
        
        # Episode-level tracking
        self.current_episode_metrics = {}
        self.episode_count = 0
        
        # Performance thresholds
        self.performance_thresholds = {
            'min_success_rate': 0.3,
            'target_success_rate': 0.8,
            'max_constraint_violation_rate': 0.2,
            'min_utilization_efficiency': 0.6
        }
        
        # Setup logging
        self.setup_logging()
    
    def setup_logging(self):
        """Setup metrics logging."""
        self.logger = logging.getLogger("AdvancedMetrics")
        self.logger.setLevel(logging.INFO)
        
        # File handler
        log_file = os.path.join(self.log_dir, "metrics.log")
        fh = logging.FileHandler(log_file)
        fh.setLevel(logging.INFO)
        
        # Formatter
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        fh.setFormatter(formatter)
        
        if not self.logger.handlers:
            self.logger.addHandler(fh)
    
    def start_episode(self):
        """Start tracking a new episode."""
        self.current_episode_metrics = {
            'reward': 0,
            'steps': 0,
            'constraint_violations': 0,
            'successful_placements': 0,
            'courses_completed': 0,
            'start_time': datetime.now(),
            'action_mask_usage': [],
            'faculty_utilization': defaultdict(int),
            'classroom_utilization': defaultdict(int),
            'time_slot_usage': defaultdict(int)
        }
    
    def update_step(self, reward: float, action: Any, action_mask: np.ndarray, 
                   state_info: Dict, constraint_violation: bool = False):
        """Update metrics for a single step."""
        self.current_episode_metrics['reward'] += reward
        self.current_episode_metrics['steps'] += 1
        
        if constraint_violation:
            self.current_episode_metrics['constraint_violations'] += 1
        
        if reward > 0:
            self.current_episode_metrics['successful_placements'] += 1
        
        # Track action mask efficiency
        mask_usage = np.sum(action_mask) / len(action_mask)
        self.current_episode_metrics['action_mask_usage'].append(mask_usage)
        
        # Track resource utilization
        if 'faculty_id' in state_info:
            self.current_episode_metrics['faculty_utilization'][state_info['faculty_id']] += 1
        if 'classroom_id' in state_info:
            self.current_episode_metrics['classroom_utilization'][state_info['classroom_id']] += 1
        if 'time_slot' in state_info:
            self.current_episode_metrics['time_slot_usage'][state_info['time_slot']] += 1
    
    def end_episode(self, success: bool, courses_completed: int, total_courses: int):
        """End episode and update metrics."""
        self.episode_count += 1
        
        # Calculate episode metrics
        episode_reward = self.current_episode_metrics['reward']
        episode_length = self.current_episode_metrics['steps']
        episode_time = (datetime.now() - self.current_episode_metrics['start_time']).total_seconds()
        
        # Success rate
        success_rate = 1.0 if success else 0.0
        
        # Constraint violation rate
        constraint_violation_rate = (
            self.current_episode_metrics['constraint_violations'] / 
            max(episode_length, 1)
        )
        
        # Utilization efficiency
        utilization_efficiency = self._calculate_utilization_efficiency()
        
        # Faculty load balance
        faculty_balance = self._calculate_faculty_balance()
        
        # Classroom balance
        classroom_balance = self._calculate_classroom_balance()
        
        # Time distribution score
        time_distribution = self._calculate_time_distribution_score()
        
        # Action mask efficiency
        action_mask_efficiency = np.mean(self.current_episode_metrics['action_mask_usage']) if self.current_episode_metrics['action_mask_usage'] else 0
        
        # Update metrics
        self.metrics['episode_rewards'].append(episode_reward)
        self.metrics['episode_lengths'].append(episode_length)
        self.metrics['success_rates'].append(success_rate)
        self.metrics['constraint_violation_rates'].append(constraint_violation_rate)
        self.metrics['utilization_efficiency'].append(utilization_efficiency)
        self.metrics['faculty_load_balance'].append(faculty_balance)
        self.metrics['classroom_balance'].append(classroom_balance)
        self.metrics['time_distribution_score'].append(time_distribution)
        self.metrics['courses_completed'].append(courses_completed)
        self.metrics['courses_failed'].append(total_courses - courses_completed)
        self.metrics['average_episode_time'].append(episode_time)
        self.metrics['action_mask_efficiency'].append(action_mask_efficiency)
        
        # Log episode summary
        self.logger.info(
            f"Episode {self.episode_count}: Reward={episode_reward:.2f}, "
            f"Length={episode_length}, Success={success}, "
            f"Constraint Violations={constraint_violation_rate:.3f}, "
            f"Utilization={utilization_efficiency:.3f}"
        )
    
    def update_training_metrics(self, training_info: Dict):
        """Update training-specific metrics."""
        if 'loss' in training_info:
            self.metrics['training_losses'].append(training_info['loss'])
        if 'value_loss' in training_info:
            self.metrics['value_losses'].append(training_info['value_loss'])
        if 'policy_loss' in training_info:
            self.metrics['policy_losses'].append(training_info['policy_loss'])
        if 'entropy_loss' in training_info:
            self.metrics['entropy_losses'].append(training_info['entropy_loss'])
        if 'learning_rate' in training_info:
            self.metrics['learning_rates'].append(training_info['learning_rate'])
        if 'grad_norm' in training_info:
            self.metrics['gradient_norms'].append(training_info['grad_norm'])
        if 'kl_div' in training_info:
            self.metrics['kl_divergences'].append(training_info['kl_div'])
        if 'clip_frac' in training_info:
            self.metrics['clip_fractions'].append(training_info['clip_frac'])
    
    def _calculate_utilization_efficiency(self) -> float:
        """Calculate resource utilization efficiency."""
        total_placements = self.current_episode_metrics['successful_placements']
        if total_placements == 0:
            return 0.0
        
        # Calculate how well resources are utilized
        faculty_utilization = len(self.current_episode_metrics['faculty_utilization'])
        classroom_utilization = len(self.current_episode_metrics['classroom_utilization'])
        time_utilization = len(self.current_episode_metrics['time_slot_usage'])
        
        # Normalize by total possible resources (this would need to be passed from environment)
        # For now, return a simple efficiency measure
        return min(1.0, total_placements / 20.0)  # Assume max 20 placements is 100% efficient
    
    def _calculate_faculty_balance(self) -> float:
        """Calculate faculty load balancing score."""
        faculty_utilization = self.current_episode_metrics['faculty_utilization']
        if not faculty_utilization:
            return 0.0
        
        loads = list(faculty_utilization.values())
        if len(loads) <= 1:
            return 1.0
        
        # Calculate coefficient of variation (lower is better)
        mean_load = np.mean(loads)
        std_load = np.std(loads)
        
        if mean_load == 0:
            return 0.0
        
        cv = std_load / mean_load
        # Convert to balance score (0-1, higher is better)
        balance_score = max(0.0, 1.0 - cv)
        return balance_score
    
    def _calculate_classroom_balance(self) -> float:
        """Calculate classroom utilization balance score."""
        classroom_utilization = self.current_episode_metrics['classroom_utilization']
        if not classroom_utilization:
            return 0.0
        
        loads = list(classroom_utilization.values())
        if len(loads) <= 1:
            return 1.0
        
        # Calculate coefficient of variation
        mean_load = np.mean(loads)
        std_load = np.std(loads)
        
        if mean_load == 0:
            return 0.0
        
        cv = std_load / mean_load
        balance_score = max(0.0, 1.0 - cv)
        return balance_score
    
    def _calculate_time_distribution_score(self) -> float:
        """Calculate time slot distribution score."""
        time_usage = self.current_episode_metrics['time_slot_usage']
        if not time_usage:
            return 0.0
        
        # Calculate how evenly distributed the time slots are
        usage_counts = list(time_usage.values())
        if len(usage_counts) <= 1:
            return 1.0
        
        # Calculate Gini coefficient (lower is better for even distribution)
        sorted_counts = sorted(usage_counts)
        n = len(sorted_counts)
        cumsum = np.cumsum(sorted_counts)
        gini = (n + 1 - 2 * np.sum(cumsum) / cumsum[-1]) / n if cumsum[-1] > 0 else 0
        
        # Convert to distribution score (0-1, higher is better)
        distribution_score = max(0.0, 1.0 - gini)
        return distribution_score
    
    def get_summary(self) -> Dict:
        """Get summary statistics for all metrics."""
        summary = {}
        
        for metric_name, values in self.metrics.items():
            if values:
                summary[metric_name] = {
                    'mean': float(np.mean(values)),
                    'std': float(np.std(values)),
                    'min': float(np.min(values)),
                    'max': float(np.max(values)),
                    'count': len(values),
                    'recent_mean': float(np.mean(list(values)[-10:])) if len(values) >= 10 else float(np.mean(values))
                }
            else:
                summary[metric_name] = {
                    'mean': 0.0, 'std': 0.0, 'min': 0.0, 'max': 0.0, 'count': 0, 'recent_mean': 0.0
                }
        
        return summary
    
    def get_performance_status(self) -> Dict:
        """Get current performance status and recommendations."""
        summary = self.get_summary()
        
        status = {
            'overall_performance': 'unknown',
            'recommendations': [],
            'warnings': [],
            'success_indicators': []
        }
        
        # Check success rate
        if 'success_rates' in summary and summary['success_rates']['recent_mean'] > 0:
            recent_success_rate = summary['success_rates']['recent_mean']
            
            if recent_success_rate >= self.performance_thresholds['target_success_rate']:
                status['overall_performance'] = 'excellent'
                status['success_indicators'].append(f"High success rate: {recent_success_rate:.2%}")
            elif recent_success_rate >= self.performance_thresholds['min_success_rate']:
                status['overall_performance'] = 'good'
                status['success_indicators'].append(f"Moderate success rate: {recent_success_rate:.2%}")
            else:
                status['overall_performance'] = 'poor'
                status['warnings'].append(f"Low success rate: {recent_success_rate:.2%}")
                status['recommendations'].append("Consider adjusting reward shaping or reducing problem difficulty")
        
        # Check constraint violations
        if 'constraint_violation_rates' in summary:
            violation_rate = summary['constraint_violation_rates']['recent_mean']
            if violation_rate > self.performance_thresholds['max_constraint_violation_rate']:
                status['warnings'].append(f"High constraint violation rate: {violation_rate:.2%}")
                status['recommendations'].append("Improve action masking or increase constraint penalties")
        
        # Check utilization efficiency
        if 'utilization_efficiency' in summary:
            utilization = summary['utilization_efficiency']['recent_mean']
            if utilization < self.performance_thresholds['min_utilization_efficiency']:
                status['warnings'].append(f"Low utilization efficiency: {utilization:.2%}")
                status['recommendations'].append("Consider adding efficiency rewards to the reward function")
        
        # Check learning progress
        if 'episode_rewards' in summary and len(self.metrics['episode_rewards']) >= 20:
            recent_rewards = list(self.metrics['episode_rewards'])[-20:]
            if np.mean(recent_rewards) > np.mean(list(self.metrics['episode_rewards'])[-40:-20]):
                status['success_indicators'].append("Rewards are improving over time")
            else:
                status['warnings'].append("Rewards may be stagnating")
                status['recommendations'].append("Consider adjusting learning rate or exploration strategy")
        
        return status
    
    def save_metrics(self, filename: str = None):
        """Save metrics to file."""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"metrics_{timestamp}.json"
        
        filepath = os.path.join(self.log_dir, filename)
        
        # Convert deques to lists for JSON serialization
        metrics_dict = {}
        for key, values in self.metrics.items():
            # Convert numpy types to Python native types for JSON serialization
            metrics_dict[key] = [float(v) if hasattr(v, 'item') else v for v in list(values)]
        
        # Add summary and status
        metrics_dict['summary'] = self.get_summary()
        metrics_dict['performance_status'] = self.get_performance_status()
        metrics_dict['episode_count'] = self.episode_count
        metrics_dict['timestamp'] = datetime.now().isoformat()
        
        with open(filepath, 'w') as f:
            json.dump(metrics_dict, f, indent=2)
        
        self.logger.info(f"Metrics saved to {filepath}")
        return filepath
    
    def plot_metrics(self, save_path: str = None):
        """Plot metrics (requires matplotlib)."""
        try:
            import matplotlib.pyplot as plt
            
            # Create subplots
            fig, axes = plt.subplots(3, 3, figsize=(18, 12))
            axes = axes.flatten()
            
            # Plot key metrics
            key_metrics = [
                'episode_rewards', 'success_rates', 'constraint_violation_rates',
                'utilization_efficiency', 'faculty_load_balance', 'classroom_balance',
                'time_distribution_score', 'episode_lengths', 'action_mask_efficiency'
            ]
            
            for i, metric_name in enumerate(key_metrics):
                if i < len(axes) and self.metrics[metric_name]:
                    values = list(self.metrics[metric_name])
                    axes[i].plot(values)
                    axes[i].set_title(metric_name.replace('_', ' ').title())
                    axes[i].set_xlabel('Episode')
                    axes[i].set_ylabel('Value')
                    axes[i].grid(True)
                    
                    # Add moving average
                    if len(values) >= 10:
                        window_size = min(20, len(values) // 5)
                        moving_avg = np.convolve(values, np.ones(window_size)/window_size, mode='valid')
                        axes[i].plot(range(window_size-1, len(values)), moving_avg, 
                                   color='red', alpha=0.7, label=f'MA({window_size})')
                        axes[i].legend()
            
            plt.tight_layout()
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.show()
            
        except ImportError:
            self.logger.warning("Matplotlib not available for plotting")
        except Exception as e:
            self.logger.error(f"Error plotting metrics: {e}")


class MetricsCallback(BaseCallback):
    """Callback for integrating metrics with training."""
    
    def __init__(self, metrics_tracker: AdvancedMetrics, verbose: int = 0):
        super(MetricsCallback, self).__init__(verbose)
        self.metrics_tracker = metrics_tracker
        self.episode_started = False
    
    def init_callback(self, model):
        """Initialize the callback with the model."""
        super().init_callback(model)
        self.metrics_tracker.start_episode()
        self.episode_started = True
    
    def _on_step(self) -> bool:
        """Called at each step."""
        # Track episode progress
        if len(self.locals.get('rewards', [])) > 0:
            reward = self.locals['rewards'][0]
            action = self.locals.get('actions', [None])[0]
            
            # Get action mask from environment if available
            action_mask = np.ones(100)  # Default mask, should be updated from env
            state_info = {}  # Default state info
            
            self.metrics_tracker.update_step(reward, action, action_mask, state_info)
        
        # Check if episode is done
        if len(self.locals.get('dones', [])) > 0 and self.locals['dones'][0]:
            self._on_episode_end()
        
        return True
    
    def _on_episode_end(self):
        """Called at the end of each episode."""
        if self.episode_started:
            # Determine success (positive reward indicates success)
            success = self.metrics_tracker.current_episode_metrics.get('reward', 0) > 0
            courses_completed = self.metrics_tracker.current_episode_metrics.get('successful_placements', 0)
            total_courses = 5  # Default, should be passed from environment
            
            self.metrics_tracker.end_episode(success, courses_completed, total_courses)
            
            # Start new episode
            self.metrics_tracker.start_episode()
    
    def custom_on_episode_start(self):
        """Called at the start of each episode."""
        self.metrics_tracker.start_episode()
        self.episode_started = True
    
    def custom_on_step(self, reward: float, action: Any, action_mask: np.ndarray, 
                state_info: Dict, constraint_violation: bool = False):
        """Called for each step."""
        if self.episode_started:
            self.metrics_tracker.update_step(reward, action, action_mask, 
                                            state_info, constraint_violation)
    
    def custom_on_episode_end(self, success: bool, courses_completed: int, total_courses: int):
        """Called at the end of each episode."""
        if self.episode_started:
            self.metrics_tracker.end_episode(success, courses_completed, total_courses)
            self.episode_started = False
    
    def custom_on_training_update(self, training_info: Dict):
        """Called after training updates."""
        self.metrics_tracker.update_training_metrics(training_info)
