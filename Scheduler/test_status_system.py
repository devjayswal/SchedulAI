"""
Test script for the status monitoring system

This script tests the complete status monitoring functionality including
job creation, progress tracking, and status endpoints.
"""

import asyncio
import json
import sys
import os
import time
import requests
from datetime import datetime

# Add the current directory to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from utils.job_manager import (
    create_job, set_status, update_progress, add_log, 
    get_job_summary, get_all_jobs, cleanup_job
)

def test_job_manager():
    """Test the job manager functionality."""
    print("Testing Job Manager")
    print("=" * 40)
    
    try:
        # Create a test job
        job_id = create_job("test_timetable", {"test": "data"})
        print(f"✓ Created job: {job_id}")
        
        # Set status to running
        set_status(job_id, "running")
        print(f"✓ Set status to running")
        
        # Update progress
        update_progress(job_id, current_step=100, total_steps=500, phase="Training", percentage=20)
        print(f"✓ Updated progress")
        
        # Add some logs
        add_log(job_id, "Starting training process")
        add_log(job_id, "Processing data...")
        add_log(job_id, "Training model...")
        print(f"✓ Added logs")
        
        # Get job summary
        summary = get_job_summary(job_id)
        print(f"✓ Retrieved job summary")
        print(f"  - Status: {summary['status']}")
        print(f"  - Progress: {summary['progress']['percentage']}%")
        print(f"  - Logs: {len(summary['recent_logs'])} entries")
        
        # Simulate progress updates
        for i in range(1, 6):
            time.sleep(0.1)  # Small delay
            progress = i * 20
            update_progress(job_id, current_step=i*100, total_steps=500, percentage=progress)
            add_log(job_id, f"Training step {i*100} completed")
        
        # Complete the job
        set_status(job_id, "completed")
        update_progress(job_id, percentage=100, phase="Completed")
        add_log(job_id, "Training completed successfully")
        
        # Get final summary
        final_summary = get_job_summary(job_id)
        print(f"✓ Job completed")
        print(f"  - Final status: {final_summary['status']}")
        print(f"  - Final progress: {final_summary['progress']['percentage']}%")
        
        # Cleanup
        cleanup_job(job_id)
        print(f"✓ Job cleaned up")
        
        return True
        
    except Exception as e:
        print(f"✗ Job manager test failed: {e}")
        return False

def test_status_endpoints():
    """Test the status API endpoints."""
    print("\nTesting Status API Endpoints")
    print("=" * 40)
    
    try:
        # Create a test job
        job_id = create_job("api_test", {"api": "test"})
        set_status(job_id, "running")
        update_progress(job_id, current_step=50, total_steps=200, percentage=25)
        add_log(job_id, "API test started")
        
        base_url = "http://127.0.0.1:8000"
        
        # Test job status endpoint
        response = requests.get(f"{base_url}/status/job/{job_id}")
        if response.status_code == 200:
            data = response.json()
            print(f"✓ Job status endpoint working")
            print(f"  - Status: {data['status']}")
            print(f"  - Progress: {data['progress']['percentage']}%")
        else:
            print(f"✗ Job status endpoint failed: {response.status_code}")
            return False
        
        # Test progress endpoint
        response = requests.get(f"{base_url}/status/job/{job_id}/progress")
        if response.status_code == 200:
            data = response.json()
            print(f"✓ Progress endpoint working")
            print(f"  - Phase: {data['current_phase']}")
            print(f"  - Percentage: {data['percentage']}%")
        else:
            print(f"✗ Progress endpoint failed: {response.status_code}")
            return False
        
        # Test logs endpoint
        response = requests.get(f"{base_url}/status/job/{job_id}/logs")
        if response.status_code == 200:
            data = response.json()
            print(f"✓ Logs endpoint working")
            print(f"  - Log entries: {len(data['logs'])}")
        else:
            print(f"✗ Logs endpoint failed: {response.status_code}")
            return False
        
        # Test all jobs endpoint
        response = requests.get(f"{base_url}/status/jobs")
        if response.status_code == 200:
            data = response.json()
            print(f"✓ All jobs endpoint working")
            print(f"  - Total jobs: {data['total']}")
        else:
            print(f"✗ All jobs endpoint failed: {response.status_code}")
            return False
        
        # Test active jobs endpoint
        response = requests.get(f"{base_url}/status/jobs/active")
        if response.status_code == 200:
            data = response.json()
            print(f"✓ Active jobs endpoint working")
            print(f"  - Active jobs: {data['total']}")
        else:
            print(f"✗ Active jobs endpoint failed: {response.status_code}")
            return False
        
        # Cleanup
        cleanup_job(job_id)
        print(f"✓ API test completed and cleaned up")
        
        return True
        
    except requests.exceptions.ConnectionError:
        print("✗ Could not connect to API server. Make sure the server is running.")
        return False
    except Exception as e:
        print(f"✗ Status endpoints test failed: {e}")
        return False

def test_progress_simulation():
    """Test realistic progress simulation."""
    print("\nTesting Progress Simulation")
    print("=" * 40)
    
    try:
        # Create a realistic training job
        job_id = create_job("training_simulation", {
            "courses": 5,
            "classrooms": 5,
            "time_slots": 6
        })
        
        set_status(job_id, "running")
        add_log(job_id, "Starting timetable generation")
        
        # Simulate realistic training phases
        phases = [
            ("Initializing", 5, "Setting up environment and model"),
            ("Data Processing", 15, "Processing course and classroom data"),
            ("Training", 70, "Running PPO training algorithm"),
            ("Optimization", 15, "Optimizing timetable constraints"),
            ("Finalization", 5, "Generating final timetable")
        ]
        
        current_percentage = 0
        
        for phase_name, phase_duration, phase_description in phases:
            add_log(job_id, f"Starting phase: {phase_name}")
            update_progress(job_id, phase=phase_name, percentage=current_percentage)
            
            # Simulate progress within the phase
            for step in range(1, phase_duration + 1):
                time.sleep(0.05)  # Small delay for realism
                current_percentage = min(100, current_percentage + (100 / sum(p[1] for p in phases)))
                update_progress(job_id, percentage=current_percentage)
                
                if step % 5 == 0:  # Log every 5 steps
                    add_log(job_id, f"{phase_name}: {step}/{phase_duration} - {phase_description}")
            
            add_log(job_id, f"Completed phase: {phase_name}")
        
        # Complete the job
        set_status(job_id, "completed")
        update_progress(job_id, percentage=100, phase="Completed")
        add_log(job_id, "Timetable generation completed successfully")
        
        # Get final summary
        summary = get_job_summary(job_id)
        print(f"✓ Progress simulation completed")
        print(f"  - Final status: {summary['status']}")
        print(f"  - Final progress: {summary['progress']['percentage']}%")
        print(f"  - Total logs: {len(summary['recent_logs'])}")
        print(f"  - Elapsed time: {summary['elapsed_time']:.2f} seconds")
        
        # Cleanup
        cleanup_job(job_id)
        print(f"✓ Simulation cleaned up")
        
        return True
        
    except Exception as e:
        print(f"✗ Progress simulation failed: {e}")
        return False

def test_error_handling():
    """Test error handling scenarios."""
    print("\nTesting Error Handling")
    print("=" * 40)
    
    try:
        # Test non-existent job
        summary = get_job_summary("non_existent_job")
        if "error" in summary:
            print(f"✓ Non-existent job handled correctly")
        else:
            print(f"✗ Non-existent job not handled correctly")
            return False
        
        # Test invalid job ID
        try:
            set_status("invalid_id", "running")
            print(f"✗ Invalid job ID not handled correctly")
            return False
        except:
            print(f"✓ Invalid job ID handled correctly")
        
        # Test cleanup of non-existent job
        try:
            cleanup_job("non_existent_job")
            print(f"✓ Cleanup of non-existent job handled correctly")
        except:
            print(f"✗ Cleanup of non-existent job not handled correctly")
            return False
        
        return True
        
    except Exception as e:
        print(f"✗ Error handling test failed: {e}")
        return False

def main():
    """Run all status system tests."""
    print("Status Monitoring System Tests")
    print("=" * 60)
    
    tests = [
        ("Job Manager", test_job_manager),
        ("Status Endpoints", test_status_endpoints),
        ("Progress Simulation", test_progress_simulation),
        ("Error Handling", test_error_handling)
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print(f"\n{test_name}")
        print("-" * len(test_name))
        if test_func():
            passed += 1
            print(f"✓ {test_name} test passed")
        else:
            print(f"✗ {test_name} test failed")
    
    print("\n" + "=" * 60)
    print(f"Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("✓ All tests passed! Status monitoring system is working correctly.")
        print("\nKey Features Verified:")
        print("- Job creation and management")
        print("- Progress tracking and updates")
        print("- Log management")
        print("- Status API endpoints")
        print("- Real-time progress simulation")
        print("- Error handling")
    else:
        print("✗ Some tests failed. Please check the status monitoring system.")
    
    return passed == total

if __name__ == "__main__":
    main()
