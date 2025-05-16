````markdown
# CA-AMFA: Context-Aware Adaptive Multi-Factor Authentication for IoT-Based Household Security Systems

**Context-Aware Adaptive Multi-Factor Authentication (CA-AMFA)** dynamically adjusts authentication requirements based on real-time risk assessment using contextual bandit algorithms. This repository contains both the hardware and software implementations needed to deploy CA-AMFA on a Raspberry Pi–based security system.

---

## Table of Contents

1. [Overview](#overview)
2. [Features](#features)
3. [Hardware Requirements](#hardware-requirements)
4. [Software Setup](#software-setup)
5. [Project Structure](#project-structure)
6. [Running the System](#running-the-system)
7. [Usage](#usage)
8. [License](#license)

---

## Overview

CA-AMFA leverages a variety of risk factors, including:

- **Time-based patterns**
- **User behavior**
- **Network conditions**
- **Motion detection**
- **Failed login attempts**

A risk score is calculated from these factors, and the system dynamically selects one or more authentication methods (PIN/password, OTP, facial recognition) to balance security and usability.

---

## Features

- **Adaptive Authentication**: Chooses authentication methods based on real-time risk.
- **Modular Bandit Algorithms**: Easily switch between ε-Greedy, Thompson Sampling, and UCB implementations.
- **Simulation Framework**: Run Monte Carlo simulations to compare bandit policies.
- **Plug-and-Play Hardware**: Interfaces for camera, keypad, OLED display, and servo lock.

---

## Hardware Requirements

- **Raspberry Pi 3B** (or compatible)
- **Raspberry Pi Camera Module v1.3**
- **SSD1306 OLED Display**
- **Tower Pro MG995 Servo Motor** (for lock control)
- **4×4 Matrix Keypad**
- **PC** (for offloading facial recognition processing)

Refer to `hardware_setup.md` for detailed wiring diagrams and pin assignments.

---

## Software Setup

1. **Clone the Repository**

   ```bash
   git clone https://github.com/yourusername/CA-AMFA.git
   cd CA-AMFA
````

2. **Create and Activate a Virtual Environment**

   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```

3. **Install Dependencies**

   ```bash
   pip install -r requirements.txt
   ```

4. **Enable Interfaces on the Raspberry Pi**

   ```bash
   sudo raspi-config
   # Select Interface Options:
   # - I2C > Enable
   # - Camera > Enable
   ```

---

## Project Structure

```text
CA-AMFA/
├── ContextualBandits/           # Bandit algorithm implementations
│   ├── ContextualBandit.py      # Base class
│   ├── EpsilonGreedyBandit.py   # ε-Greedy
│   ├── ThompsonSamplingBandit.py# Thompson Sampling
│   └── UCBBandit.py             # Upper Confidence Bound
├── RiskAssessment/              # Risk factor modules
│   ├── FailedAttemptsRiskFactor.py
│   ├── MotionActivityRiskFactor.py
│   ├── NetworkBasedRiskFactor.py
│   ├── TimeBasedRiskFactor.py
│   └── UserBehaviorRiskFactor.py
├── database/                    # SQLite interface
│   └── db.py
├── models/                      # Serialized model files
├── results/                     # Evaluation metrics & plots
├── simulation/                  # Simulation scripts
├── tests/                       # Unit tests for components
├── utils/                       # Utility functions
├── adaptive_mfa.py              # Core adaptive MFA logic
├── hardware_controller.py       # Hardware abstraction layer
├── main.py                      # Entry point for live system
└── run_simulation.py            # CLI for simulations
```

---

## Running the System

### Main Application (on Raspberry Pi)

```bash
python main.py
```

### Simulations and Evaluations

* **Basic run with reproducible seed**

  ```bash
  python run_simulation.py --seed 42
  ```

* **Multiple runs (default seed increment)**

  ```bash
  python run_simulation.py --runs 10 --seed 42
  ```

* **Specific seeds for reproducibility**

  ```bash
  python run_simulation.py --seed_list "42,123,456,789,1010"
  ```

* **Set confidence level for statistical intervals**

  ```bash
  python run_simulation.py --runs 10 --confidence 0.99
  ```

* **Generate boxplots to visualize variability**

  ```bash
  python run_simulation.py --runs 10 --boxplots
  ```

* **Compare against a different baseline method**

  ```bash
  python run_simulation.py --runs 10 --compare_with thompson
  ```

---

## Usage

1. **Registration**: Use the keypad menu (option 5) to register a new user.
2. **Login**: Select "Authenticate" (option 4) and enter your PIN.
3. **Risk Assessment**: The system calculates a risk score and prompts for additional factors if needed.
4. **Authentication Factors**:

   * **PIN** entry via keypad
   * **OTP** verification (QR code generated at registration)
   * **Facial Recognition** (capture via Pi Camera, processed on PC)

---

## License

This project is licensed under the **MIT License**. See [LICENSE](LICENSE) for details.

```
```
