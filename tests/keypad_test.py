#!/usr/bin/env python3
import time
import sys
import os
import threading

# Import HardwareController directly
try:
    from hardware_controller import HardwareController
except ImportError:
    print("ERROR: Unable to import HardwareController. Make sure the module is in the current directory or PYTHONPATH.")
    sys.exit(1)

def keypad_listener(controller):
    """Background thread to continuously listen for keypad input"""
    print("Keypad listener started. Press keys on the keypad...")
    keys_pressed = 0
    
    while True:
        key = controller.read_keypad()
        if key:
            keys_pressed += 1
            print(f"\n[{keys_pressed}] Key pressed: {key}")
            
            # Show on display if available
            controller.display_message(f"Key pressed:\n{key}")
            
            # Exit if # is pressed 3 times
            if key == '#' and keys_pressed > 10:
                print("\nExit sequence detected (#). Stopping test...")
                break
                
        # Small visual indicator that we're still running
        sys.stdout.write('.')
        sys.stdout.flush()
        time.sleep(0.1)

def main():
    print("Keypad Test using HardwareController")
    print("====================================")
    
    try:
        # Initialize hardware controller
        print("Initializing hardware controller...")
        controller = HardwareController()
        
        # Display startup message
        controller.display_message("Keypad Test\nPress any key")
        
        # Start keypad listener in background thread
        listener_thread = threading.Thread(target=keypad_listener, args=(controller,))
        listener_thread.daemon = True
        listener_thread.start()
        
        # Motion sensor test
        print("\nMotion sensor test")
        print("------------------")
        print("Move in front of the sensor if available")

        for i in range(10):
            motion_detected = controller.read_motion()
            print(f"Motion detected: {motion_detected}")
            
            if motion_detected:
                controller.display_message("Motion detected!")
            
            time.sleep(1)
        
        # Lock test
        print("\nLock test")
        print("---------")
        print("Testing lock mechanism...")
        
        controller.display_message("Testing lock...")
        controller.control_lock(True)  # Lock
        print("Lock engaged")
        time.sleep(2)
        
        controller.control_lock(False)  # Unlock
        print("Lock disengaged")
        controller.display_message("Lock disengaged")
        time.sleep(2)
        
        controller.control_lock(True)  # Lock again
        print("Lock re-engaged")
        time.sleep(1)
        
        # Main loop - wait for listener thread or user interrupt
        print("\nPress Ctrl+C to stop the test")
        while listener_thread.is_alive():
            time.sleep(0.5)
            
    except KeyboardInterrupt:
        print("\nTest terminated by user")
    except Exception as e:
        print(f"\nError during test: {e}")
    finally:
        print("\nCleaning up hardware resources...")
        try:
            controller.cleanup()
        except:
            pass
        print("Test completed")

if __name__ == "__main__":
    main()