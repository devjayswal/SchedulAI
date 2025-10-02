"""
Curriculum Learning Environment for Progressive Difficulty Training
This module provides a curriculum learning wrapper that gradually increases difficulty.
"""

import numpy as np
import gymnasium as gym
from gymnasium.spaces import MultiDiscrete, Box
from typing import Dict, Any, List, Tuple
from collections import defaultdict

from ppo.env import TimetableEnv
from models.Timetable import Timetable
from models.Course import Course
from models.Faculty import Faculty
from models.Classroom import Classroom
from models.Branch import Branch


class CurriculumTimetableEnv(TimetableEnv):
    """Curriculum learning wrapper for TimetableEnv with progressive difficulty."""
    
    def __init__(self, base_timetable: Timetable, difficulty_level: int = 0, 
                 max_difficulty: int = 4, success_threshold: float = 0.8):
        """
        Initialize curriculum environment.
        
        Args:
            base_timetable: Base timetable configuration
            difficulty_level: Current difficulty level (0=easiest, max_difficulty=hardest)
            max_difficulty: Maximum difficulty level
            success_threshold: Success rate threshold for advancing difficulty
        """
        self.base_timetable = base_timetable
        self.difficulty_level = difficulty_level
        self.max_difficulty = max_difficulty
        self.success_threshold = success_threshold
        
        # Create modified timetable based on difficulty
        self.modified_timetable = self._create_difficulty_timetable()
        
        # Initialize base environment
        super().__init__(self.modified_timetable)
        
        # Curriculum tracking
        self.episode_count = 0
        self.success_count = 0
        self.recent_successes = []
        self.difficulty_history = [difficulty_level]
        
    def _create_difficulty_timetable(self) -> Timetable:
        """Create timetable with difficulty-adjusted parameters."""
        # Deep copy base timetable
        timetable = Timetable()
        timetable.days = self.base_timetable.days.copy()
        timetable.time_slots = self.base_timetable.time_slots.copy()
        
        # Adjust parameters based on difficulty level
        if self.difficulty_level == 0:
            # Easiest: More resources, fewer constraints
            timetable = self._create_easy_timetable(timetable)
        elif self.difficulty_level == 1:
            # Easy: Slightly reduced resources
            timetable = self._create_easy_timetable(timetable)
            timetable = self._reduce_resources(timetable, 0.9)
        elif self.difficulty_level == 2:
            # Medium: Standard resources
            timetable = self._copy_base_timetable(timetable)
        elif self.difficulty_level == 3:
            # Hard: Reduced resources, more constraints
            timetable = self._copy_base_timetable(timetable)
            timetable = self._reduce_resources(timetable, 0.8)
            timetable = self._add_constraints(timetable)
        else:
            # Hardest: Minimal resources, maximum constraints
            timetable = self._copy_base_timetable(timetable)
            timetable = self._reduce_resources(timetable, 0.7)
            timetable = self._add_constraints(timetable)
            timetable = self._add_advanced_constraints(timetable)
        
        return timetable
    
    def _create_easy_timetable(self, timetable: Timetable) -> Timetable:
        """Create easy difficulty timetable with extra resources."""
        # Add extra faculty
        extra_faculty = []
        for i in range(2):  # Add 2 extra faculty
            faculty = Faculty(f"F_EXTRA_{i+1}", f"Extra Faculty {i+1}", f"extra{i+1}@example.com")
            extra_faculty.append(faculty)
        timetable.faculty = self.base_timetable.faculty + extra_faculty
        
        # Add extra classrooms
        extra_classrooms = []
        for i in range(3):  # Add 3 extra classrooms
            classroom = Classroom(f"C_EXTRA_{i+1}", f"Extra Room {i+1}", "theory", 50)
            extra_classrooms.append(classroom)
        timetable.classrooms = self.base_timetable.classrooms + extra_classrooms
        
        # Copy courses and branches
        timetable.courses = self.base_timetable.courses.copy()
        timetable.branches = self.base_timetable.branches.copy()
        
        # Initialize timetables
        timetable.initialize_timetables()
        
        return timetable
    
    def _copy_base_timetable(self, timetable: Timetable) -> Timetable:
        """Copy base timetable exactly."""
        timetable.faculty = self.base_timetable.faculty.copy()
        timetable.classrooms = self.base_timetable.classrooms.copy()
        timetable.courses = self.base_timetable.courses.copy()
        timetable.branches = self.base_timetable.branches.copy()
        timetable.initialize_timetables()
        return timetable
    
    def _reduce_resources(self, timetable: Timetable, reduction_factor: float) -> Timetable:
        """Reduce available resources by the given factor."""
        # Reduce faculty
        num_faculty_to_keep = max(1, int(len(timetable.faculty) * reduction_factor))
        timetable.faculty = timetable.faculty[:num_faculty_to_keep]
        
        # Reduce classrooms
        num_classrooms_to_keep = max(1, int(len(timetable.classrooms) * reduction_factor))
        timetable.classrooms = timetable.classrooms[:num_classrooms_to_keep]
        
        # Reassign courses to remaining faculty
        for i, course in enumerate(timetable.courses):
            faculty_index = i % len(timetable.faculty)
            course.faculty_id = timetable.faculty[faculty_index].short_name
        
        return timetable
    
    def _add_constraints(self, timetable: Timetable) -> Timetable:
        """Add additional constraints for higher difficulty."""
        # Add faculty availability constraints (some faculty unavailable certain days)
        for faculty in timetable.faculty:
            if not hasattr(faculty, 'unavailable_days'):
                # Randomly make some faculty unavailable on certain days
                unavailable_days = np.random.choice(
                    timetable.days, 
                    size=np.random.randint(1, 3), 
                    replace=False
                )
                faculty.unavailable_days = set(unavailable_days)
        
        # Add classroom maintenance constraints
        for classroom in timetable.classrooms:
            if not hasattr(classroom, 'maintenance_slots'):
                # Randomly assign maintenance slots
                maintenance_slots = np.random.choice(
                    timetable.time_slots,
                    size=np.random.randint(1, 2),
                    replace=False
                )
                classroom.maintenance_slots = set(maintenance_slots)
        
        return timetable
    
    def _add_advanced_constraints(self, timetable: Timetable) -> Timetable:
        """Add advanced constraints for highest difficulty."""
        # Add course prerequisites
        for i, course in enumerate(timetable.courses):
            if i > 0 and not hasattr(course, 'prerequisites'):
                # Make some courses have prerequisites
                if np.random.random() < 0.3:  # 30% chance
                    course.prerequisites = [timetable.courses[i-1].subject_code]
        
        # Add faculty specialization constraints
        for faculty in timetable.faculty:
            if not hasattr(faculty, 'specializations'):
                # Randomly assign specializations
                available_courses = [c.subject_code for c in timetable.courses]
                specializations = np.random.choice(
                    available_courses,
                    size=np.random.randint(1, min(3, len(available_courses))),
                    replace=False
                )
                faculty.specializations = set(specializations)
        
        return timetable
    
    def step(self, action):
        """Take a step with curriculum-specific reward adjustments."""
        obs, reward, done, truncated, info = super().step(action)
        
        # Adjust reward based on difficulty level
        difficulty_multiplier = 1.0 + (self.difficulty_level * 0.2)  # 20% bonus per level
        reward *= difficulty_multiplier
        
        # Add curriculum-specific info
        info['difficulty_level'] = self.difficulty_level
        info['curriculum_progress'] = self._get_curriculum_progress()
        
        return obs, reward, done, truncated, info
    
    def reset(self, seed=None, options=None):
        """Reset environment and track episode."""
        obs, info = super().reset(seed=seed, options=options)
        
        # Track episode
        self.episode_count += 1
        
        # Add curriculum info
        info['difficulty_level'] = self.difficulty_level
        info['curriculum_progress'] = self._get_curriculum_progress()
        
        return obs, info
    
    def _get_curriculum_progress(self) -> Dict:
        """Get current curriculum learning progress."""
        recent_success_rate = (
            np.mean(self.recent_successes[-20:]) if len(self.recent_successes) >= 20
            else np.mean(self.recent_successes) if self.recent_successes else 0.0
        )
        
        return {
            'difficulty_level': self.difficulty_level,
            'max_difficulty': self.max_difficulty,
            'episode_count': self.episode_count,
            'success_count': self.success_count,
            'recent_success_rate': recent_success_rate,
            'ready_for_advancement': recent_success_rate >= self.success_threshold,
            'difficulty_history': self.difficulty_history.copy()
        }
    
    def update_difficulty(self, episode_success: bool):
        """Update difficulty level based on recent performance."""
        self.recent_successes.append(episode_success)
        if episode_success:
            self.success_count += 1
        
        # Keep only recent successes (last 50 episodes)
        if len(self.recent_successes) > 50:
            self.recent_successes = self.recent_successes[-50:]
        
        # Check if ready for advancement
        if len(self.recent_successes) >= 20:  # Need at least 20 episodes
            recent_success_rate = np.mean(self.recent_successes[-20:])
            
            # Advance difficulty if success rate is high enough
            if (self.difficulty_level < self.max_difficulty and 
                recent_success_rate >= self.success_threshold):
                self.difficulty_level += 1
                self.difficulty_history.append(self.difficulty_level)
                self._update_timetable_for_difficulty()
                print(f"Curriculum: Advanced to difficulty level {self.difficulty_level}")
            
            # Decrease difficulty if success rate is too low
            elif (self.difficulty_level > 0 and 
                  recent_success_rate < self.success_threshold * 0.5):
                self.difficulty_level -= 1
                self.difficulty_history.append(self.difficulty_level)
                self._update_timetable_for_difficulty()
                print(f"Curriculum: Decreased to difficulty level {self.difficulty_level}")
    
    def _update_timetable_for_difficulty(self):
        """Update the timetable when difficulty changes."""
        # Create new timetable with updated difficulty
        self.modified_timetable = self._create_difficulty_timetable()
        
        # Update environment parameters
        self.timetable = self.modified_timetable
        self.num_courses = len(self.timetable.courses)
        self.num_classrooms = len(self.timetable.classrooms)
        
        # Update action and observation spaces
        self.action_space = MultiDiscrete((
            self.num_courses + 1,
            self.num_flat_slots,
            self.num_classrooms,
        ))
        self.observation_space = Box(
            low=0,
            high=self.num_courses,
            shape=(self.num_flat_slots * self.num_classrooms,),
            dtype=np.int32,
        )
        
        # Reset environment state
        self.reset()
    
    def get_difficulty_info(self) -> Dict:
        """Get detailed difficulty information."""
        return {
            'current_difficulty': self.difficulty_level,
            'max_difficulty': self.max_difficulty,
            'episode_count': self.episode_count,
            'success_count': self.success_count,
            'success_rate': self.success_count / max(self.episode_count, 1),
            'recent_success_rate': (
                np.mean(self.recent_successes[-20:]) if len(self.recent_successes) >= 20
                else np.mean(self.recent_successes) if self.recent_successes else 0.0
            ),
            'difficulty_history': self.difficulty_history.copy(),
            'ready_for_advancement': (
                len(self.recent_successes) >= 20 and 
                np.mean(self.recent_successes[-20:]) >= self.success_threshold
            ),
            'timetable_info': {
                'num_courses': self.num_courses,
                'num_faculty': len(self.timetable.faculty),
                'num_classrooms': self.num_classrooms,
                'num_days': self.num_days,
                'num_slots_per_day': self.num_slots_per_day
            }
        }


class CurriculumWrapper:
    """Wrapper for managing curriculum learning across multiple environments."""
    
    def __init__(self, base_timetable: Timetable, n_envs: int = 4, 
                 max_difficulty: int = 4, success_threshold: float = 0.8):
        self.base_timetable = base_timetable
        self.n_envs = n_envs
        self.max_difficulty = max_difficulty
        self.success_threshold = success_threshold
        
        # Create curriculum environments
        self.envs = []
        for i in range(n_envs):
            # Start with different difficulty levels for diversity
            initial_difficulty = min(i, max_difficulty)
            env = CurriculumTimetableEnv(
                base_timetable, 
                difficulty_level=initial_difficulty,
                max_difficulty=max_difficulty,
                success_threshold=success_threshold
            )
            self.envs.append(env)
        
        # Track overall curriculum progress
        self.global_episode_count = 0
        self.global_success_count = 0
    
    def step(self, actions):
        """Step all environments."""
        results = []
        for i, (env, action) in enumerate(zip(self.envs, actions)):
            obs, reward, done, truncated, info = env.step(action)
            results.append((obs, reward, done, truncated, info))
            
            # Update global tracking
            self.global_episode_count += 1
            if reward > 0:  # Positive reward indicates success
                self.global_success_count += 1
            
            # Update difficulty for this environment
            if done:
                episode_success = reward > 0
                env.update_difficulty(episode_success)
        
        return results
    
    def reset(self):
        """Reset all environments."""
        results = []
        for env in self.envs:
            obs, info = env.reset()
            results.append((obs, info))
        return results
    
    def get_curriculum_summary(self) -> Dict:
        """Get summary of curriculum learning progress."""
        env_summaries = []
        for i, env in enumerate(self.envs):
            env_summaries.append({
                'env_id': i,
                'difficulty_info': env.get_difficulty_info()
            })
        
        return {
            'global_episode_count': self.global_episode_count,
            'global_success_count': self.global_success_count,
            'global_success_rate': self.global_success_count / max(self.global_episode_count, 1),
            'environment_summaries': env_summaries,
            'average_difficulty': np.mean([env.difficulty_level for env in self.envs]),
            'max_difficulty_reached': max([env.difficulty_level for env in self.envs]),
            'environments_at_max_difficulty': sum([1 for env in self.envs if env.difficulty_level == self.max_difficulty])
        }


def create_curriculum_environment(base_timetable: Timetable, difficulty_level: int = 0) -> CurriculumTimetableEnv:
    """Create a curriculum learning environment."""
    return CurriculumTimetableEnv(base_timetable, difficulty_level)


def create_curriculum_wrapper(base_timetable: Timetable, n_envs: int = 4) -> CurriculumWrapper:
    """Create a curriculum learning wrapper for multiple environments."""
    return CurriculumWrapper(base_timetable, n_envs)


# Example usage and testing
def test_curriculum_learning():
    """Test curriculum learning functionality."""
    print("Testing Curriculum Learning...")
    
    try:
        # Create test timetable
        from models.Timetable import Timetable
        from models.Course import Course
        from models.Faculty import Faculty
        from models.Classroom import Classroom
        
        timetable = Timetable()
        timetable.days = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"]
        timetable.time_slots = ["09:00-10:00", "10:00-11:00", "11:00-12:00", "12:00-13:00", "14:00-15:00", "15:00-16:00"]
        
        faculty = Faculty("F001", "Test Faculty", "test@example.com")
        classroom = Classroom("C001", "Test Room", "theory", 50)
        
        for i in range(3):
            course = Course(f"C{i+1}", f"Course {i+1}", 3, "theory", faculty.short_name)
            timetable.courses.append(course)
        
        timetable.faculty = [faculty]
        timetable.classrooms = [classroom]
        
        # Test curriculum environment
        curriculum_env = create_curriculum_environment(timetable, difficulty_level=0)
        print(f"✓ Curriculum environment created at difficulty level {curriculum_env.difficulty_level}")
        
        # Test environment step
        obs, info = curriculum_env.reset()
        print(f"✓ Environment reset successful, difficulty: {info['difficulty_level']}")
        
        # Test difficulty progression
        for _ in range(5):
            curriculum_env.update_difficulty(episode_success=True)
        
        print(f"✓ Difficulty progression test successful, new level: {curriculum_env.difficulty_level}")
        
        # Test curriculum wrapper
        curriculum_wrapper = create_curriculum_wrapper(timetable, n_envs=2)
        print(f"✓ Curriculum wrapper created with {len(curriculum_wrapper.envs)} environments")
        
        summary = curriculum_wrapper.get_curriculum_summary()
        print(f"✓ Curriculum summary generated: {summary['global_episode_count']} episodes")
        
        print("All curriculum learning tests passed!")
        return True
        
    except Exception as e:
        print(f"✗ Curriculum learning test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    # Run tests
    test_curriculum_learning()
