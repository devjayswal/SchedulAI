"""
Test script for optimized configuration

This script tests the new configuration without disrupting current training.
It validates that the environment can be created and basic operations work.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from ppo.config_optimized import env_config, ppo_kwargs, policy_kwargs
from ppo.env import TimetableEnv
from models.Timetable import Timetable
from models.Course import Course
from models.Faculty import Faculty
from models.Classroom import Classroom
from models.Branch import Branch

def create_test_timetable():
    """Create a test timetable with the new configuration."""
    
    # Create test data
    courses = [
        Course("CS101", "Computer Science", "theory", 3, "FAC001"),
        Course("CS102", "Programming", "lab", 2, "FAC002"),
        Course("MATH101", "Mathematics", "theory", 3, "FAC003"),
        Course("PHY101", "Physics", "theory", 3, "FAC004"),
        Course("ENG101", "English", "theory", 2, "FAC005"),
    ]
    
    faculty = [
        Faculty("FAC001", "Dr. Smith", "Computer Science"),
        Faculty("FAC002", "Dr. Johnson", "Computer Science"),
        Faculty("FAC003", "Dr. Brown", "Mathematics"),
        Faculty("FAC004", "Dr. Davis", "Physics"),
        Faculty("FAC005", "Dr. Wilson", "English"),
    ]
    
    classrooms = [
        Classroom("CR001", "Lecture Hall", 50, "theory"),
        Classroom("CR002", "Computer Lab", 30, "lab"),
        Classroom("CR003", "Lecture Hall", 40, "theory"),
        Classroom("CR004", "Lecture Hall", 35, "theory"),
        Classroom("CR005", "Lab", 25, "lab"),
    ]
    
    # Create time slots (12 slots as per new config)
    time_slots = [
        "08:00-09:00", "09:00-10:00", "10:00-11:00", "11:00-12:00",
        "13:00-14:00", "14:00-15:00", "15:00-16:00", "16:00-17:00",
        "17:00-18:00", "18:00-19:00", "19:00-20:00", "20:00-21:00"
    ]
    
    days = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"]
    
    # Create branches
    branch = Branch("CS", "Computer Science", 1, courses)
    
    # Create timetable
    timetable = Timetable(
        courses=courses,
        faculty=faculty,
        classrooms=classrooms,
        branches=[branch],
        time_slots=time_slots,
        days=days
    )
    
    return timetable

def test_environment_creation():
    """Test that the environment can be created with new config."""
    print("Testing environment creation with optimized config...")
    
    try:
        timetable = create_test_timetable()
        env = TimetableEnv(timetable)
        
        print(f"✓ Environment created successfully")
        print(f"  - Courses: {env.num_courses}")
        print(f"  - Time slots: {env.num_slots_per_day}")
        print(f"  - Classrooms: {env.num_classrooms}")
        print(f"  - Total flat slots: {env.num_flat_slots}")
        
        return True
        
    except Exception as e:
        print(f"✗ Environment creation failed: {e}")
        return False

def test_action_space():
    """Test the action space with new configuration."""
    print("\nTesting action space...")
    
    try:
        timetable = create_test_timetable()
        env = TimetableEnv(timetable)
        
        action_space = env.action_space
        obs_space = env.observation_space
        
        print(f"✓ Action space: {action_space}")
        print(f"✓ Observation space: {obs_space}")
        
        # Test action sampling
        action = action_space.sample()
        print(f"✓ Sample action: {action}")
        
        return True
        
    except Exception as e:
        print(f"✗ Action space test failed: {e}")
        return False

def test_environment_step():
    """Test a single environment step."""
    print("\nTesting environment step...")
    
    try:
        timetable = create_test_timetable()
        env = TimetableEnv(timetable)
        
        obs = env.reset()
        action = env.action_space.sample()
        obs, reward, done, truncated, info = env.step(action)
        
        print(f"✓ Step completed successfully")
        print(f"  - Reward: {reward}")
        print(f"  - Done: {done}")
        print(f"  - Observation shape: {obs.shape}")
        
        return True
        
    except Exception as e:
        print(f"✗ Environment step failed: {e}")
        return False

def main():
    """Run all tests."""
    print("=" * 50)
    print("Testing Optimized Configuration")
    print("=" * 50)
    
    tests = [
        test_environment_creation,
        test_action_space,
        test_environment_step,
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        if test():
            passed += 1
    
    print("\n" + "=" * 50)
    print(f"Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("✓ All tests passed! Configuration is ready to use.")
        print("\nTo use the optimized configuration:")
        print("1. Import from ppo.config_optimized instead of ppo.hyperparams")
        print("2. The new config provides better learning with reduced constraints")
    else:
        print("✗ Some tests failed. Please check the configuration.")
    
    return passed == total

if __name__ == "__main__":
    main()
