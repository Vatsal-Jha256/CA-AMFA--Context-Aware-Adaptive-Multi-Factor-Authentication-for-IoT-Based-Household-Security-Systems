import time
import os
import logging
from hardware_controller import HardwareController
from adaptive_mfa import AdaptiveMFA

def main():
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler("homelock.log"),
            logging.StreamHandler()
        ]
    )
    logger = logging.getLogger("HomeLock")
    
    # Define working keys based on test
    working_keys = ['0', '4', '5', '6', '7', '9', '*', '#']
    
    try:
        # Initialize hardware
        hw = HardwareController()
        hw.display_message("System Starting")
        time.sleep(1)
        
        # Initialize MFA system
        mfa = AdaptiveMFA(hw)
        
        # Main loop
        while True:
            hw.display_message("Enter option:\n4:Login 5:Register")
            
            # Get initial choice - wait for valid input
            choice = None
            while choice not in ['4', '5']:
                choice = hw.read_keypad()
                if choice is not None and choice not in ['4', '5']:
                    hw.display_message("Invalid choice\nUse 4 or 5")
                    time.sleep(1)
                    hw.display_message("Enter option:\n4:Login 5:Register")
                time.sleep(0.1)  # Add small delay to prevent CPU hogging

            if choice == '4':  # Login flow
                hw.display_message("Enter ID\nor press #")
                
                # Get username
                username = ""
                while True:
                    key = hw.read_keypad()
                    if key:
                        if key == '#':  # Submit
                            break
                        elif key == '*':  # Backspace
                            if username:
                                username = username[:-1]
                        elif key in working_keys:
                            username += key
                        hw.display_message(f"Enter ID:\n{username}")
                    time.sleep(0.1)
                
                if username:
                    # Check if user exists
                    user = mfa.db.get_user(username)
                    if not user:
                        hw.display_message(f"User {username}\nnot found")
                        time.sleep(2)
                        continue
                    
                    # Attempt authentication
                    success = mfa.authenticate_user(username)
                    logger.info(f"Auth attempt for {username}: {'success' if success else 'failed'}")
            
            elif choice == '5':  # Registration flow
                hw.display_message("New User\nEnter ID:")
                
                # Get new username with validation
                username = ""
                while True:
                    key = hw.read_keypad()
                    if key:
                        if key == '#':  # Submit
                            if len(username) >= 2:  # Minimum username length
                                break
                            else:
                                hw.display_message("ID too short\nMin 2 digits")
                                time.sleep(1)
                                hw.display_message(f"New User ID:\n{username}")
                        elif key == '*':  # Backspace
                            if username:
                                username = username[:-1]
                        elif key in working_keys:
                            username += key
                        hw.display_message(f"New User ID:\n{username}")
                    time.sleep(0.1)

                if not username:
                    continue
                    
                # Check if user already exists
                if mfa.db.get_user(username):
                    hw.display_message(f"User {username}\nalready exists")
                    time.sleep(2)
                    continue
                
                # Get password with validation
                hw.display_message("Create password:")
                password = ""
                while True:
                    key = hw.read_keypad()
                    if key:
                        if key == '#':  # Submit
                            if len(password) >= 4:  # Minimum password length
                                break
                            else:
                                hw.display_message("Password too short\nMin 4 digits")
                                time.sleep(1)
                                hw.display_message("Create password:\n" + "*" * len(password))
                        elif key == '*':  # Backspace
                            if password:
                                password = password[:-1]
                        elif key in working_keys:
                            password += key
                        hw.display_message("Create password:\n" + "*" * len(password))
                    time.sleep(0.1)
                
                # Confirm password
                hw.display_message("Confirm password:")
                confirm_password = ""
                while True:
                    key = hw.read_keypad()
                    if key:
                        if key == '#':  # Submit
                            break
                        elif key == '*':  # Backspace
                            if confirm_password:
                                confirm_password = confirm_password[:-1]
                        elif key in working_keys:
                            confirm_password += key
                        hw.display_message("Confirm password:\n" + "*" * len(confirm_password))
                    time.sleep(0.1)
                
                # Check if passwords match
                if password != confirm_password:
                    hw.display_message("Passwords\ndon't match")
                    time.sleep(2)
                    continue

                # Ask about face registration
                hw.display_message("Register face?\n6:Yes 7:No")
                face_choice = None
                timeout_start = time.time()
                while face_choice not in ['6', '7'] and time.time() - timeout_start < 10:
                    face_choice = hw.read_keypad()
                    time.sleep(0.1)
                
                # Enroll user with or without face
                if face_choice == '6':  # With face
                    success = mfa.enroll_user(username, password, capture_face=True)
                elif face_choice == '7':  # Without face
                    success = mfa.enroll_user(username, password, capture_face=False)
                else:  # Default to no face if timeout
                    hw.display_message("No selection\nFace skipped")
                    time.sleep(1)
                    success = mfa.enroll_user(username, password, capture_face=False)
                
                if success:
                    # QR code is already displayed in the enroll_user function
                    hw.display_message("Scan QR with\nAuth app")
                    time.sleep(5)
                    hw.display_message("Registration\ncomplete!")
                    logger.info(f"New user registered: {username}")
                else:
                    hw.display_message("Registration\nfailed")
                    logger.error(f"Failed to register user: {username}")
                
                time.sleep(2)
            
    except KeyboardInterrupt:
        logger.info("System shutdown requested")
    except Exception as e:
        logger.error(f"System error: {e}", exc_info=True)
    finally:
        # Clean shutdown
        if 'mfa' in locals():
            mfa.shutdown()
        if 'hw' in locals():
            hw.cleanup()
        logger.info("System shutdown complete")

if __name__ == "__main__":
    main()
