"""
Enhanced Reward Shaping for Timetable Environment

This module provides improved reward functions to guide the PPO agent
toward better scheduling solutions without disrupting existing training.
"""

import numpy as np

class RewardShaper:
    """Enhanced reward shaping for better agent learning."""
    
    def __init__(self, num_courses, num_slots, num_classrooms):
        self.num_courses = num_courses
        self.num_slots = num_slots
        self.num_classrooms = num_classrooms
        
        # Reward weights
        self.weights = {
            'successful_placement': 10.0,    # Reward for successful course placement
            'constraint_violation': -5.0,    # Penalty for constraint violations
            'progress_bonus': 2.0,           # Bonus for making progress
            'efficiency_bonus': 1.0,         # Bonus for efficient scheduling
            'completion_bonus': 20.0,        # Large bonus for completing courses
            'episode_completion': 50.0,      # Bonus for completing all courses
        }
    
    def calculate_reward(self, action_result, state_info):
        """
        Calculate enhanced reward based on action result and state information.
        
        Args:
            action_result: Result from environment step
            state_info: Dictionary with state information
        
        Returns:
            Enhanced reward value
        """
        reward = 0.0
        
        # Base reward from environment
        base_reward = action_result.get('base_reward', 0)
        reward += base_reward
        
        # Successful placement bonus
        if action_result.get('successful_placement', False):
            reward += self.weights['successful_placement']
        
        # Progress bonus (courses completed)
        completed_courses = state_info.get('completed_courses', 0)
        reward += completed_courses * self.weights['progress_bonus']
        
        # Efficiency bonus (utilization rate)
        utilization = state_info.get('utilization_rate', 0.0)
        reward += utilization * self.weights['efficiency_bonus']
        
        # Course completion bonus
        if action_result.get('course_completed', False):
            reward += self.weights['completion_bonus']
        
        # Episode completion bonus
        if action_result.get('episode_completed', False):
            reward += self.weights['episode_completion']
        
        return reward
    
    def get_curriculum_reward(self, episode_count, base_reward):
        """
        Apply curriculum learning by adjusting rewards based on training progress.
        
        Args:
            episode_count: Current episode number
            base_reward: Base reward from environment
        
        Returns:
            Curriculum-adjusted reward
        """
        # Gradually increase difficulty and adjust rewards
        if episode_count < 100:
            # Early episodes: more forgiving
            return base_reward * 1.5
        elif episode_count < 500:
            # Mid training: standard rewards
            return base_reward
        else:
            # Late training: stricter rewards
            return base_reward * 0.8

def create_enhanced_reward_function(env):
    """
    Create an enhanced reward function wrapper for the environment.
    
    Args:
        env: The TimetableEnv instance
    
    Returns:
        Enhanced reward function
    """
    shaper = RewardShaper(
        env.num_courses,
        env.num_flat_slots,
        env.num_classrooms
    )
    
    def enhanced_reward(action_result, state_info):
        return shaper.calculate_reward(action_result, state_info)
    
    return enhanced_reward
