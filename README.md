# CA-AMFA: Context-Aware Adaptive Multi-Factor Authentication for IoT-Based Household Security Systems

This repository contains the implementation of a Context-Aware Adaptive Multi-Factor Authentication (CA-AMFA) system for IoT-based household security systems. The system dynamically adjusts authentication requirements based on risk assessment using contextual bandit algorithms.

## Overview

The CA-AMFA system leverages a set of risk factors including time-based patterns, user behavior, network conditions, motion detection, and failed login attempts to calculate a risk score. Based on this score, it dynamically selects the appropriate authentication methods (password, OTP, facial recognition) to balance security and usability.

## Hardware Requirements

- Raspberry Pi 3B (or compatible)
- Raspberry Pi Camera Module v1.3
- SSD1306 OLED Display
- Tower Pro MG995 Servo Motor for lock control
- 4×4 Matrix Keypad
- PC for facial recognition processing

## Software Setup

### 1. Install Required Packages

```bash
pip install -r requirements.txt
```

### 2. Enable I2C and Camera Interface

```bash
sudo raspi-config
# Select Interface Options > I2C > Enable
# Select Interface Options > Camera > Enable
```

### 3. Hardware Connections

Refer to the physical connections section in the hardware_setup.md file for detailed wiring instructions.

## Project Structure

- `ContextualBandits/`: Implementation of various bandit algorithms
  - `ContextualBandit.py`: Base class for bandit algorithms
  - `EpsilonGreedyBandit.py`: Epsilon-Greedy implementation
  - `ThompsonSamplingBandit.py`: Thompson Sampling implementation
  - `UCBBandit.py`: Upper Confidence Bound implementation
- `RiskAssessment/`: Risk factor implementations
  - `FailedAttemptsRiskFactor.py`: Risk based on failed login attempts
  - `MotionActivityRiskFactor.py`: Risk based on motion detection
  - `NetworkBasedRiskFactor.py`: Risk based on network conditions
  - `TimeBasedRiskFactor.py`: Risk based on time of day
  - `UserBehaviorRiskFactor.py`: Risk based on user behavior patterns
- `database/`: Database management
  - `db.py`: SQLite database interface
- `models/`: Serialized model files
- `results/`: Performance metrics and evaluation results
- `simulation/`: Simulation framework for testing
- `tests/`: Component test scripts
- `utils/`: Utility functions
- `adaptive_mfa.py`: Main implementation of the adaptive MFA system
- `hardware_controller.py`: Interface for hardware components
- `main.py`: Entry point for the application

## Running the System

### Main Application

To run the main application on the Raspberry Pi:

```bash
python main.py
```

### Testing Components

To test individual hardware components:

```bash
# Test the camera
python tests/test_camera.py

# Test the OLED display
python tests/oled_test.py

# Test the keypad
python tests/keypad_test.py

# Test the servo motor
python tests/servo_test.py

# Test QR code generation (for OTP setup)
python tests/test_qr.py
```

### Running Simulations

To evaluate the system using simulations:

```bash
# Compare bandit algorithms with fixed-weight approach
python simulation/compare_bandit_fixed.py

# Run a security simulation
python simulation/security_sim.py

# Run comprehensive evaluation
python test.py
```

## Usage

1. **Registration**: Use the keypad to register a new user (option 5)
2. **Authentication**: Log in with your credentials (option 4)
3. **Risk Assessment**: The system will dynamically determine which authentication factors are required based on the calculated risk score
4. **Authentication Factors**:
   - PIN entry via keypad
   - OTP verification (generate during registration, verify during login)
   - Facial recognition (capture from Pi Camera, processed on connected PC)

## License

[MIT License](LICENSE) 
