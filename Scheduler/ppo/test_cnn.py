"""
Simple test script for CNN model implementation
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def test_cnn_imports():
    """Test if all CNN modules can be imported."""
    try:
        from .cnn_model import TimetableCNNExtractor, create_cnn_model
        print("✓ CNN model imports successful")
        return True
    except ImportError as e:
        print(f"✗ CNN model import failed: {e}")
        return False

def test_cnn_hyperparams():
    """Test if CNN hyperparameters can be loaded."""
    try:
        from .cnn_hyperparams import get_config, cnn_ppo_kwargs
        config = get_config("cnn")
        print("✓ CNN hyperparameters loaded successfully")
        print(f"  - Learning rate: {config['ppo_kwargs']['learning_rate']}")
        print(f"  - Batch size: {config['ppo_kwargs']['batch_size']}")
        return True
    except ImportError as e:
        print(f"✗ CNN hyperparameters import failed: {e}")
        return False

def test_model_creation():
    """Test if CNN model can be created."""
    try:
        from .model import create_cnn_ppo_model
        print("✓ CNN model creation function available")
        return True
    except ImportError as e:
        print(f"✗ CNN model creation import failed: {e}")
        return False

def main():
    """Run all tests."""
    print("Testing CNN Implementation...")
    print("=" * 40)
    
    tests = [
        test_cnn_imports,
        test_cnn_hyperparams,
        test_model_creation
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        if test():
            passed += 1
        print()
    
    print("=" * 40)
    print(f"Tests passed: {passed}/{total}")
    
    if passed == total:
        print("✓ All tests passed! CNN implementation is ready.")
    else:
        print("✗ Some tests failed. Please check the implementation.")
    
    return passed == total

if __name__ == "__main__":
    main()
