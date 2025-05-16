#!/usr/bin/env python3
import time
import os
import sys
from hardware_controller import HardwareController
# Check if we're running on a Raspberry Pi
IS_RASPBERRY_PI = os.uname().machine.startswith('arm') or os.uname().machine == 'aarch64'

def test_servo_independently():
    """Independent test for servo motor using HardwareController"""
    print("=== Servo Motor Independent Test ===")
    
    try:

        
      # Initialize the controller
        print("Initializing hardware controller...")
        controller = HardwareController()
        
        # Test basic functionality
        print("\nRunning servo test sequence:")
        
        # Test 1: Lock position
        print("Test 1: Moving servo to LOCK position")
        controller.control_lock(True)
        time.sleep(2)  # Allow time to observe movement
        
        # Test 2: Unlock position
        print("Test 2: Moving servo to UNLOCK position")
        controller.control_lock(False)
        time.sleep(2)  # Allow time to observe movement
        
        # Test 3: Multiple cycles
        print("Test 3: Running 3 lock/unlock cycles")
        for i in range(3):
            print(f"  Cycle {i+1}: Locking")
            controller.control_lock(True)
            time.sleep(1)
            print(f"  Cycle {i+1}: Unlocking")
            controller.control_lock(False)
            time.sleep(1)
        
        # Test 4: Hold positions
        print("Test 4: Testing position hold")
        print("  Moving to LOCK position and holding for 3 seconds")
        controller.control_lock(True)
        time.sleep(3)
        print("  Moving to UNLOCK position and holding for 3 seconds")
        controller.control_lock(False)
        time.sleep(3)
        
        # Final position - decide where you want to leave the servo
        print("Setting final position to LOCK")
        controller.control_lock(True)
        
        print("\nServo motor test completed successfully!")
    
    except Exception as e:
        print(f"Error during servo test: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        # Clean up
        try:
            print("\nCleaning up resources...")
            controller.cleanup()
            print("Cleanup complete")
            
            # Remove temporary file if it was created
            if os.path.exists('temp_hardware_control.py'):
                os.remove('temp_hardware_control.py')
                if os.path.exists('temp_hardware_control.pyc'):
                    os.remove('temp_hardware_control.pyc')
                if os.path.exists('__pycache__/temp_hardware_control.cpython-*.pyc'):
                    import glob
                    for f in glob.glob('__pycache__/temp_hardware_control.cpython-*.pyc'):
                        os.remove(f)
        except:
            print("Note: Cleanup encountered some issues")

if __name__ == "__main__":
    test_servo_independently()
