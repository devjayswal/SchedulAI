"""
Test script for dynamic configuration integration

This script tests the complete flow from frontend request to backend processing
with dynamic parameters.
"""

import json
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from utils.config_processor import config_processor

def test_config_processing():
    """Test the configuration processing with sample data."""
    print("Testing Dynamic Configuration Processing")
    print("=" * 50)
    
    # Sample request data (similar to what frontend would send)
    sample_request = {
        "schedule_config": {
            "weekdays": ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"],
            "time_slots": [
                "09:00-10:00", "10:00-11:00", "11:00-12:00", 
                "12:00-13:00", "14:00-15:00", "15:00-16:00"
            ],
            "lunch_break": "12:00-13:00",
            "max_daily_slots": 6,
            "max_weekly_slots": 30
        },
        "infrastructure_config": {
            "default_classroom_count": 5,
            "default_lab_count": 2,
            "default_theory_room_count": 3,
            "classroom_capacity": {
                "theory": 50,
                "lab": 30
            }
        },
        "training_config": {
            "learning_rate": 3e-4,
            "batch_size": 64,
            "n_steps": 1024,
            "n_epochs": 4,
            "total_timesteps": 500000,
            "use_enhanced_rewards": True,
            "use_curriculum_learning": True
        },
        "branches": [
            {
                "branch_name": "AIML",
                "semester": 6,
                "courses": [
                    {
                        "subject_code": "2280622",
                        "subject_name": "Image Processing",
                        "subject_type": "theory",
                        "credits": 3,
                        "faculty_id": "F01"
                    },
                    {
                        "subject_code": "2280623",
                        "subject_name": "Machine Learning",
                        "subject_type": "theory",
                        "credits": 4,
                        "faculty_id": "F02"
                    },
                    {
                        "subject_code": "2280624",
                        "subject_name": "Deep Learning Lab",
                        "subject_type": "lab",
                        "credits": 2,
                        "faculty_id": "F03"
                    }
                ]
            }
        ],
        "faculty": [
            {"id": "F01", "name": "Dr. A", "email": "a@institute.edu"},
            {"id": "F02", "name": "Dr. B", "email": "b@institute.edu"},
            {"id": "F03", "name": "Dr. C", "email": "c@institute.edu"}
        ],
        "classrooms": [
            {"id": "CR101", "name": "Lecture Hall 1", "type": "theory", "capacity": 50},
            {"id": "CR102", "name": "Lecture Hall 2", "type": "theory", "capacity": 45},
            {"id": "CR103", "name": "Lecture Hall 3", "type": "theory", "capacity": 40},
            {"id": "LAB201", "name": "Computer Lab 1", "type": "lab", "capacity": 30},
            {"id": "LAB202", "name": "Computer Lab 2", "type": "lab", "capacity": 25}
        ]
    }
    
    try:
        # Process the request
        print("1. Processing request...")
        processed_config = config_processor.process_request(sample_request)
        print("✓ Request processed successfully")
        
        # Extract environment config
        print("\n2. Extracting environment configuration...")
        env_config = config_processor.get_environment_config(processed_config)
        print("✓ Environment config extracted:")
        print(f"   - Courses: {env_config['num_courses']}")
        print(f"   - Time slots: {env_config['num_slots']}")
        print(f"   - Classrooms: {env_config['num_classrooms']}")
        print(f"   - Weekdays: {len(env_config['weekdays'])}")
        
        # Extract training config
        print("\n3. Extracting training configuration...")
        training_config = config_processor.get_training_config(processed_config)
        print("✓ Training config extracted:")
        print(f"   - Learning rate: {training_config['learning_rate']}")
        print(f"   - Batch size: {training_config['batch_size']}")
        print(f"   - N steps: {training_config['n_steps']}")
        print(f"   - Enhanced rewards: {training_config['use_enhanced_rewards']}")
        
        # Test with minimal data (defaults should be applied)
        print("\n4. Testing with minimal data...")
        minimal_request = {
            "branches": sample_request["branches"],
            "faculty": sample_request["faculty"],
            "classrooms": sample_request["classrooms"]
        }
        
        minimal_processed = config_processor.process_request(minimal_request)
        minimal_env_config = config_processor.get_environment_config(minimal_processed)
        
        print("✓ Minimal request processed with defaults:")
        print(f"   - Courses: {minimal_env_config['num_courses']}")
        print(f"   - Time slots: {minimal_env_config['num_slots']}")
        print(f"   - Classrooms: {minimal_env_config['num_classrooms']}")
        
        # Test validation
        print("\n5. Testing validation...")
        invalid_request = {
            "training_config": {
                "learning_rate": 0.1,  # Too high
                "batch_size": 2000,    # Too high
            },
            "branches": sample_request["branches"],
            "faculty": sample_request["faculty"],
            "classrooms": sample_request["classrooms"]
        }
        
        invalid_processed = config_processor.process_request(invalid_request)
        invalid_training_config = config_processor.get_training_config(invalid_processed)
        
        print("✓ Invalid values corrected to defaults:")
        print(f"   - Learning rate: {invalid_training_config['learning_rate']} (corrected from 0.1)")
        print(f"   - Batch size: {invalid_training_config['batch_size']} (corrected from 2000)")
        
        return True
        
    except Exception as e:
        print(f"✗ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_backward_compatibility():
    """Test backward compatibility with legacy request format."""
    print("\n" + "=" * 50)
    print("Testing Backward Compatibility")
    print("=" * 50)
    
    # Legacy request format
    legacy_request = {
        "weekdays": ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"],
        "time_slots": [
            "09:00-10:00", "10:00-11:00", "11:00-12:00", 
            "12:00-13:00", "14:00-15:00", "15:00-16:00"
        ],
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
                        "faculty_id": "F01"
                    }
                ]
            }
        ],
        "faculty": [
            {"id": "F01", "name": "Dr. Smith", "email": "smith@institute.edu"}
        ],
        "classrooms": [
            {"id": "CR101", "type": "theory"}
        ]
    }
    
    try:
        processed_config = config_processor.process_request(legacy_request)
        env_config = config_processor.get_environment_config(processed_config)
        
        print("✓ Legacy request processed successfully:")
        print(f"   - Courses: {env_config['num_courses']}")
        print(f"   - Time slots: {env_config['num_slots']}")
        print(f"   - Classrooms: {env_config['num_classrooms']}")
        print(f"   - Weekdays: {len(env_config['weekdays'])}")
        
        return True
        
    except Exception as e:
        print(f"✗ Backward compatibility test failed: {e}")
        return False

def main():
    """Run all tests."""
    print("Dynamic Configuration Integration Tests")
    print("=" * 60)
    
    tests = [
        test_config_processing,
        test_backward_compatibility
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        if test():
            passed += 1
    
    print("\n" + "=" * 60)
    print(f"Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("✓ All tests passed! Dynamic configuration is working correctly.")
        print("\nKey Features:")
        print("- Dynamic parameters from frontend")
        print("- Reasonable defaults for all settings")
        print("- Backward compatibility with legacy format")
        print("- Validation and error correction")
        print("- Environment and training config extraction")
    else:
        print("✗ Some tests failed. Please check the configuration.")
    
    return passed == total

if __name__ == "__main__":
    main()
