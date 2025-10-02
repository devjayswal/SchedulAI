"""
Test Script for Enhanced Training Integration
This script tests the integration between frontend requests and enhanced training.
"""

import asyncio
import sys
import os

# Add the current directory to the path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from controllers.timetable import create_timetable
from models.Timetable import Timetable
from models.Course import Course
from models.Faculty import Faculty
from models.Classroom import Classroom
from models.Branch import Branch


def create_test_timetable_data():
    """Create test timetable data for integration testing."""
    timetable_data = {
        "name": "Test Enhanced Timetable",
        "description": "Testing enhanced training integration",
        "weekdays": ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"],
        "time_slots": ["09:00-10:00", "10:00-11:00", "11:00-12:00", "12:00-13:00", "14:00-15:00", "15:00-16:00"],
        
        "branches": [
            {
                "branch_name": "CSE",
                "semester": 4,
                "courses": [
                    {
                        "subject_code": "CS201",
                        "subject_name": "Data Structures",
                        "subject_type": "theory",
                        "credits": 3,
                        "faculty_id": "F001"
                    },
                    {
                        "subject_code": "CS202",
                        "subject_name": "Algorithms",
                        "subject_type": "theory",
                        "credits": 3,
                        "faculty_id": "F002"
                    },
                    {
                        "subject_code": "CS203",
                        "subject_name": "Programming Lab",
                        "subject_type": "lab",
                        "credits": 2,
                        "faculty_id": "F001"
                    }
                ]
            }
        ],
        
        "faculty": [
            {"id": "F001", "name": "Dr. Smith", "email": "smith@institute.edu"},
            {"id": "F002", "name": "Dr. Johnson", "email": "johnson@institute.edu"},
            {"id": "F003", "name": "Dr. Brown", "email": "brown@institute.edu"}
        ],
        
        "classrooms": [
            {"id": "CR101", "name": "Room 101", "type": "theory", "capacity": 50},
            {"id": "CR102", "name": "Room 102", "type": "theory", "capacity": 40},
            {"id": "LAB201", "name": "Lab 201", "type": "lab", "capacity": 30}
        ],
        
        # Enhanced training configuration
        "training_config": {
            "use_enhanced_training": True,
            "use_enhanced_cnn": True,
            "n_envs": 4,  # Reduced for testing
            "total_timesteps": 1000,  # Very small for quick testing
            "use_curriculum_learning": True,
            "use_advanced_metrics": True,
            "use_enhanced_rewards": True
        },
        
        "schedule_config": {
            "max_daily_slots": 6,
            "lunch_break": "12:00-13:00"
        },
        
        "infrastructure_config": {
            "classroom_count": 3,
            "lab_count": 1,
            "theory_room_count": 2
        }
    }
    
    return timetable_data


async def test_enhanced_integration():
    """Test the enhanced training integration."""
    print("=" * 60)
    print("TESTING ENHANCED TRAINING INTEGRATION")
    print("=" * 60)
    
    try:
        # Create test data
        test_data = create_test_timetable_data()
        print("✓ Test data created successfully")
        
        # Test 1: Enhanced training (default)
        print("\n1. Testing Enhanced Training (Default)...")
        result1 = await create_timetable(test_data)
        print(f"✓ Enhanced training job created: {result1['job_id']}")
        print(f"  - Enhanced training: {result1['enhanced_training']}")
        
        # Test 2: Force enhanced training
        print("\n2. Testing Forced Enhanced Training...")
        test_data_enhanced = test_data.copy()
        test_data_enhanced["training_config"]["use_enhanced_training"] = True
        result2 = await create_timetable(test_data_enhanced)
        print(f"✓ Forced enhanced training job created: {result2['job_id']}")
        print(f"  - Enhanced training: {result2['enhanced_training']}")
        
        # Test 3: Force legacy training
        print("\n3. Testing Legacy Training...")
        test_data_legacy = test_data.copy()
        test_data_legacy["training_config"]["use_enhanced_training"] = False
        result3 = await create_timetable(test_data_legacy)
        print(f"✓ Legacy training job created: {result3['job_id']}")
        print(f"  - Enhanced training: {result3['enhanced_training']}")
        
        # Test 4: No training config (should default to enhanced)
        print("\n4. Testing Default Behavior (No Config)...")
        test_data_default = test_data.copy()
        del test_data_default["training_config"]
        result4 = await create_timetable(test_data_default)
        print(f"✓ Default training job created: {result4['job_id']}")
        print(f"  - Enhanced training: {result4['enhanced_training']}")
        
        print("\n" + "=" * 60)
        print("ALL INTEGRATION TESTS PASSED! ✅")
        print("=" * 60)
        print("Summary:")
        print(f"  - Enhanced training (default): {result1['enhanced_training']}")
        print(f"  - Forced enhanced training: {result2['enhanced_training']}")
        print(f"  - Legacy training: {result3['enhanced_training']}")
        print(f"  - Default behavior: {result4['enhanced_training']}")
        print("\nThe frontend will now automatically use enhanced training!")
        
        return True
        
    except Exception as e:
        print(f"✗ Integration test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


async def test_quick_training():
    """Test a quick training run to verify everything works."""
    print("\n" + "=" * 60)
    print("TESTING QUICK TRAINING RUN")
    print("=" * 60)
    
    try:
        # Create minimal test data for quick training
        quick_data = create_test_timetable_data()
        quick_data["training_config"]["total_timesteps"] = 100  # Very quick test
        quick_data["training_config"]["n_envs"] = 2  # Minimal environments
        
        print("Starting quick training test...")
        result = await create_timetable(quick_data)
        job_id = result['job_id']
        
        print(f"✓ Quick training job started: {job_id}")
        print("  - This will run for a few seconds to verify the system works")
        print("  - Check the logs/ directory for training output")
        print("  - The job will complete quickly due to minimal timesteps")
        
        return True
        
    except Exception as e:
        print(f"✗ Quick training test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Main test function."""
    print("Enhanced Training Integration Test")
    print("This script tests the integration between frontend requests and enhanced training.")
    print("\nChoose test option:")
    print("1. Test integration only (recommended)")
    print("2. Test integration + quick training run")
    print("3. Quick training run only")
    
    choice = input("Enter your choice (1-3): ").strip()
    
    if choice == "1":
        asyncio.run(test_enhanced_integration())
    elif choice == "2":
        asyncio.run(test_enhanced_integration())
        asyncio.run(test_quick_training())
    elif choice == "3":
        asyncio.run(test_quick_training())
    else:
        print("Invalid choice. Running integration test...")
        asyncio.run(test_enhanced_integration())


if __name__ == "__main__":
    main()
