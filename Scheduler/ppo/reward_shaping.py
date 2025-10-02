"""
Improved Reward Shaping for Timetable Environment
This module provides a more effective reward system that addresses the training issues:
1. Positive guidance for learning
2. Curriculum learning support
3. Better reward scaling
4. Incremental learning rewards
"""

import numpy as np
from collections import deque
from typing import Dict, List, Tuple, Any


class ImprovedRewardShaper:
    """Improved reward shaping with better learning guidance."""
    
    def __init__(self, num_courses: int, num_slots: int, num_classrooms: int):
        self.num_courses = num_courses
        self.num_slots = num_slots
        self.num_classrooms = num_classrooms
        
        # Reward components
        self.base_rewards = {
            'successful_placement': 10.0,      # Reward for valid placement
            'course_completion': 25.0,         # Reward for completing a course
            'episode_completion': 100.0,       # Reward for completing all courses
            'efficiency_bonus': 5.0,           # Bonus for good utilization
            'progress_bonus': 2.0,             # Bonus for making progress
        }
        
        self.penalties = {
            'constraint_violation': -5.0,      # Reduced penalty for violations
            'invalid_action': -2.0,            # Small penalty for invalid actions
            'inaction': -1.0,                  # Small penalty for no action
            'slot_occupied': -1.0,             # Small penalty for occupied slots
        }
        
        # Curriculum learning parameters
        self.difficulty_level = 0
        self.episode_count = 0
        self.success_count = 0
        self.recent_rewards = deque(maxlen=100)
        self.recent_successes = deque(maxlen=50)
        
        # Adaptive parameters
        self.reward_scale = 1.0
        self.learning_phase = 'exploration'  # exploration, exploitation, mastery
        
    def calculate_reward(self, action_result: Dict, state_info: Dict, 
                        episode_info: Dict = None) -> float:
        """Calculate improved reward with better guidance."""
        reward = 0.0
        
        # 1. Base action reward
        if action_result.get('successful_placement', False):
            reward += self.base_rewards['successful_placement']
            
        # 2. Course completion bonus
        if action_result.get('course_completed', False):
            reward += self.base_rewards['course_completion']
            
        # 3. Episode completion bonus
        if action_result.get('episode_completed', False):
            reward += self.base_rewards['episode_completion']
            
        # 4. Progress bonus (reward for making progress)
        if episode_info:
            completed_courses = episode_info.get('completed_courses', 0)
            total_courses = episode_info.get('total_courses', self.num_courses)
            progress = completed_courses / total_courses
            reward += progress * self.base_rewards['progress_bonus']
            
        # 5. Efficiency bonus
        utilization = state_info.get('utilization_rate', 0.0)
        if utilization > 0.5:  # Reward good utilization
            reward += (utilization - 0.5) * self.base_rewards['efficiency_bonus']
            
        # 6. Constraint violation penalty (reduced)
        if action_result.get('constraint_violation', False):
            reward += self.penalties['constraint_violation']
            
        # 7. Invalid action penalty
        if action_result.get('invalid_action', False):
            reward += self.penalties['invalid_action']
            
        # 8. Inaction penalty
        if action_result.get('inaction', False):
            reward += self.penalties['inaction']
            
        # 9. Slot occupied penalty
        if action_result.get('slot_occupied', False):
            reward += self.penalties['slot_occupied']
        
        # Apply curriculum learning scaling
        reward = self._apply_curriculum_scaling(reward)
        
        # Store for adaptive learning
        self.recent_rewards.append(reward)
        
        return reward
    
    def _apply_curriculum_scaling(self, reward: float) -> float:
        """Apply curriculum learning scaling to rewards."""
        if self.difficulty_level == 0:
            # Easy mode: More generous rewards
            return reward * 1.5
        elif self.difficulty_level == 1:
            # Medium mode: Standard rewards
            return reward * 1.0
        else:
            # Hard mode: Stricter rewards
            return reward * 0.8
    
    def update_episode(self, episode_success: bool, episode_reward: float):
        """Update reward shaper with episode results."""
        self.episode_count += 1
        if episode_success:
            self.success_count += 1
        self.recent_successes.append(episode_success)
        
        # Update learning phase
        self._update_learning_phase()
        
        # Update difficulty level
        self._update_difficulty_level()
        
        # Update reward scaling
        self._update_reward_scaling()
    
    def _update_learning_phase(self):
        """Update learning phase based on recent performance."""
        if len(self.recent_successes) < 20:
            return
            
        recent_success_rate = np.mean(list(self.recent_successes))
        recent_avg_reward = np.mean(list(self.recent_rewards)) if self.recent_rewards else 0
        
        if recent_success_rate > 0.8 and recent_avg_reward > 50:
            self.learning_phase = 'mastery'
        elif recent_success_rate > 0.5 or recent_avg_reward > 20:
            self.learning_phase = 'exploitation'
        else:
            self.learning_phase = 'exploration'
    
    def _update_difficulty_level(self):
        """Update difficulty level based on performance."""
        if len(self.recent_successes) < 30:
            return
            
        success_rate = np.mean(list(self.recent_successes))
        
        if success_rate > 0.7 and self.difficulty_level < 2:
            self.difficulty_level += 1
            print(f"Difficulty increased to level {self.difficulty_level}")
        elif success_rate < 0.3 and self.difficulty_level > 0:
            self.difficulty_level -= 1
            print(f"Difficulty decreased to level {self.difficulty_level}")
    
    def _update_reward_scaling(self):
        """Update reward scaling based on learning phase."""
        if self.learning_phase == 'exploration':
            self.reward_scale = 1.2  # More generous rewards for exploration
        elif self.learning_phase == 'exploitation':
            self.reward_scale = 1.0  # Standard rewards
        else:  # mastery
            self.reward_scale = 0.8  # Stricter rewards for mastery
    
    def get_curriculum_reward(self, episode_count: int, base_reward: float,
                            action_result: Dict, state_info: Dict,
                            episode_info: Dict = None) -> float:
        """Get curriculum-adjusted reward."""
        # Calculate base reward
        reward = self.calculate_reward(action_result, state_info, episode_info)
        
        # Apply curriculum scaling
        reward *= self.reward_scale
        
        # Add exploration bonus for early episodes
        if episode_count < 100:
            reward *= 1.1  # 10% bonus for early exploration
            
        return reward
    
    def get_performance_metrics(self) -> Dict:
        """Get current performance metrics."""
        recent_success_rate = np.mean(list(self.recent_successes)) if self.recent_successes else 0
        recent_avg_reward = np.mean(list(self.recent_rewards)) if self.recent_rewards else 0
        
        return {
            'difficulty_level': self.difficulty_level,
            'learning_phase': self.learning_phase,
            'episode_count': self.episode_count,
            'success_count': self.success_count,
            'success_rate': recent_success_rate,
            'avg_reward': recent_avg_reward,
            'reward_scale': self.reward_scale
        }


class AdaptiveRewardSystem:
    """Adaptive reward system that adjusts based on training progress."""
    
    def __init__(self, num_courses: int, num_slots: int, num_classrooms: int):
        self.reward_shaper = ImprovedRewardShaper(num_courses, num_slots, num_classrooms)
        self.performance_history = deque(maxlen=1000)
        self.adaptation_threshold = 0.1  # 10% improvement threshold
        
    def get_reward(self, action_result: Dict, state_info: Dict, 
                  episode_info: Dict = None) -> float:
        """Get adaptive reward."""
        reward = self.reward_shaper.get_curriculum_reward(
            self.reward_shaper.reward_shaper.episode_count,
            action_result.get('base_reward', 0),
            action_result,
            state_info,
            episode_info
        )
        
        # Store performance data
        self.performance_history.append({
            'reward': reward,
            'success': action_result.get('episode_completed', False),
            'episode_info': episode_info
        })
        
        return reward
    
    def update_episode(self, episode_success: bool, episode_reward: float):
        """Update the reward system with episode results."""
        self.reward_shaper.update_episode(episode_success, episode_reward)
        
        # Check for performance stagnation
        self._check_stagnation()
    
    def _check_stagnation(self):
        """Check for performance stagnation and adapt if needed."""
        if len(self.performance_history) < 100:
            return
            
        # Get recent performance
        recent_performance = list(self.performance_history)[-100:]
        recent_success_rate = np.mean([p['success'] for p in recent_performance])
        recent_avg_reward = np.mean([p['reward'] for p in recent_performance])
        
        # Get older performance for comparison
        if len(self.performance_history) >= 200:
            older_performance = list(self.performance_history)[-200:-100]
            older_success_rate = np.mean([p['success'] for p in older_performance])
            older_avg_reward = np.mean([p['reward'] for p in older_performance])
            
            # Check for stagnation
            success_improvement = recent_success_rate - older_success_rate
            reward_improvement = recent_avg_reward - older_avg_reward
            
            if (abs(success_improvement) < self.adaptation_threshold and 
                abs(reward_improvement) < self.adaptation_threshold):
                # Performance is stagnating, try to adapt
                self._adapt_rewards()
    
    def _adapt_rewards(self):
        """Adapt reward parameters to break stagnation."""
        # Increase exploration bonus
        if self.reward_shaper.reward_shaper.learning_phase == 'exploration':
            self.reward_shaper.reward_shaper.reward_scale *= 1.1
        # Decrease difficulty if stuck
        elif self.reward_shaper.reward_shaper.difficulty_level > 0:
            self.reward_shaper.reward_shaper.difficulty_level -= 1
            print(f"Adapted: Reduced difficulty to level {self.reward_shaper.reward_shaper.difficulty_level}")
    
    def get_system_metrics(self) -> Dict:
        """Get comprehensive system metrics."""
        base_metrics = self.reward_shaper.get_performance_metrics()
        
        if self.performance_history:
            recent_performance = list(self.performance_history)[-100:]
            base_metrics.update({
                'recent_success_rate': np.mean([p['success'] for p in recent_performance]),
                'recent_avg_reward': np.mean([p['reward'] for p in recent_performance]),
                'total_episodes': len(self.performance_history)
            })
        
        return base_metrics


def create_improved_reward_function(env):
    """Create improved reward function for environment."""
    reward_system = AdaptiveRewardSystem(
        env.num_courses,
        env.num_flat_slots,
        env.num_classrooms
    )
    
    def improved_reward(action_result, state_info, episode_info=None):
        return reward_system.get_reward(action_result, state_info, episode_info)
    
    return improved_reward, reward_system


class RewardAnalyzer:
    """Analyze reward patterns and provide insights."""
    
    def __init__(self):
        self.reward_history = []
        self.success_history = []
        self.episode_lengths = []
        
    def add_episode(self, total_reward: float, success: bool, episode_length: int):
        """Add episode data for analysis."""
        self.reward_history.append(total_reward)
        self.success_history.append(success)
        self.episode_lengths.append(episode_length)
        
    def get_analysis(self) -> Dict:
        """Get reward analysis."""
        if not self.reward_history:
            return {}
            
        return {
            'avg_reward': np.mean(self.reward_history),
            'reward_std': np.std(self.reward_history),
            'success_rate': np.mean(self.success_history),
            'avg_episode_length': np.mean(self.episode_lengths),
            'reward_trend': self._calculate_trend(self.reward_history),
            'success_trend': self._calculate_trend(self.success_history),
            'total_episodes': len(self.reward_history)
        }
    
    def _calculate_trend(self, data: List[float]) -> str:
        """Calculate trend in data."""
        if len(data) < 10:
            return 'insufficient_data'
            
        # Simple linear trend
        x = np.arange(len(data))
        y = np.array(data)
        slope = np.polyfit(x, y, 1)[0]
        
        if slope > 0.1:
            return 'improving'
        elif slope < -0.1:
            return 'declining'
        else:
            return 'stable'
    
    def plot_rewards(self, save_path: str = None):
        """Plot reward history."""
        try:
            import matplotlib.pyplot as plt
            
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
            
            # Plot rewards
            ax1.plot(self.reward_history)
            ax1.set_title('Episode Rewards Over Time')
            ax1.set_xlabel('Episode')
            ax1.set_ylabel('Total Reward')
            ax1.grid(True)
            
            # Plot success rate (moving average)
            if len(self.success_history) > 10:
                window_size = min(50, len(self.success_history) // 10)
                success_ma = np.convolve(self.success_history, 
                                       np.ones(window_size)/window_size, 
                                       mode='valid')
                ax2.plot(success_ma)
                ax2.set_title(f'Success Rate (Moving Average, window={window_size})')
                ax2.set_xlabel('Episode')
                ax2.set_ylabel('Success Rate')
                ax2.grid(True)
            
            plt.tight_layout()
            if save_path:
                plt.savefig(save_path)
            plt.show()
            
        except ImportError:
            print("Matplotlib not available for plotting")
