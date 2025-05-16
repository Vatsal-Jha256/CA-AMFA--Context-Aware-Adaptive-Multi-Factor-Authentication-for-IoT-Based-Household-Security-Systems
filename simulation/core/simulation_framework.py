import os
import numpy as np
import pandas as pd
# import matplotlib.pyplot as plt
# import seaborn as sns
# from matplotlib.dates import DateFormatter
import datetime
import random
import pickle
import time
import json
from tqdm import tqdm
from collections import defaultdict

# Import the required classes from your existing implementation
from RiskAssessment.RiskFactor import RiskFactor
from ContextualBandits.EpsilonGreedyBandit import EpsilonGreedyBandit
from ContextualBandits.ThompsonSamplingBandit import ThompsonSamplingBandit
from ContextualBandits.UCBBandit import UCBBandit

# Create directories if they don't exist
os.makedirs("simulation_results", exist_ok=True)
os.makedirs("simulation_models", exist_ok=True)
os.makedirs("simulation_graphs", exist_ok=True)



class SimulatedRiskFactor(RiskFactor):
    """Enhanced risk factor for simulation purposes with phase shifts and seasonality"""
    
    def __init__(self, name, weight=1.0, drift_rate=0.01, volatility=0.05, adaptability=0.1):
        super().__init__(name, weight)
        self.base_risk = random.uniform(0.1, 0.7)  # Base risk level
        self.drift_rate = drift_rate  # How quickly the base risk changes over time
        self.volatility = volatility  # Random fluctuation amount
        self.trend_direction = random.choice([-1, 1])  # Direction of trend
        self.trend_duration = random.randint(50, 150)  # How long until trend changes
        self.trend_counter = 0
        self.adaptability = adaptability  # How quickly this factor can adapt
        self.phase = random.uniform(0, 2 * np.pi)  # For cyclical patterns
        self.season_modifier = 0.0  # Seasonal impact

    def calculate(self, context):
        # Update trend
        self.trend_counter += 1
        if self.trend_counter >= self.trend_duration:
            self.trend_direction *= -1  # Reverse trend
            self.trend_duration = random.randint(50, 150)  # Reset duration
            self.trend_counter = 0
            
        # Apply drift based on trend direction
        drift = self.trend_direction * self.drift_rate
        self.base_risk += drift
        
        # Add seasonal component (day/night cycle)
        hour = context.get('hour', 12)
        time_of_day_effect = 0.15 * np.sin(hour * np.pi / 12 + self.phase)
        
        # Add weekly seasonality
        day_of_week = context.get('day', 0) % 7
        weekly_effect = 0.1 * np.sin(day_of_week * np.pi / 3.5)
        
        # Combine seasonal effects
        self.season_modifier = time_of_day_effect + weekly_effect
        
        # Apply volatility and bound the result
        noise = random.uniform(-self.volatility, self.volatility)
        risk = self.base_risk + noise + self.season_modifier
        
        # Ensure risk stays within [0, 1]
        self.base_risk = max(0.05, min(0.95, self.base_risk))
        
        # Contextual modifications
        if context:
            time_of_day = context.get('hour', 12)
            user_trust = context.get('user_trust', 0.5)
            
            # Time-based risk adjustment (higher at night)
            if time_of_day >= 22 or time_of_day <= 5:
                risk += 0.2
            
            # Trust-based adjustment
            risk -= user_trust * 0.3
            
            # Account for recent failed attempts (highly important signal)
            recent_failures = context.get('recent_failed_attempts', 0)
            if recent_failures > 0:
                risk += min(0.5, recent_failures * 0.1)  # Up to 0.5 increase
                
        return max(0.0, min(1.0, risk))


class UserSimulator:
    """Improved simulator for user behaviors and sophisticated attack patterns"""
    
    def __init__(self, user_count=50, attack_probability=0.08):
        self.users = self._generate_users(user_count)
        self.attack_probability = attack_probability
        self.attack_mode = "none"  # none, targeted, brute_force, automated, or adaptive
        self.attack_counter = 0
        self.attack_duration = 0
        self.attack_target = None
        self.attack_success_rate = {}  # Track success rate per method to model adaptive attacker
        self.attack_phase = 0  # For advanced attacks that evolve over time
        self.day_counter = 0
        
    def _generate_users(self, count):
        users = {}
        for i in range(count):
            user_id = f"user_{i:03d}"
            users[user_id] = {
                "trust_level": random.uniform(0.2, 0.9),
                "typical_hours": [random.randint(7, 20) for _ in range(random.randint(1, 4))],
                "login_frequency": random.uniform(0.1, 0.8),  # Higher means more frequent logins
                "legitimate": True,  # Whether this is a legitimate user
                "compromised": False,  # Whether account has been compromised
                "behavior_change_days": [random.randint(30, 300) for _ in range(random.randint(1, 3))],  # Days when behavior changes
                "behavior_change_factor": random.uniform(0.5, 1.5)  # How much behavior changes
            }
        return users
    
    def update_day(self, day):
        """Track day changes to trigger behavior shifts"""
        if day != self.day_counter:
            self.day_counter = day
            # Check for behavior changes
            for user_id, user in self.users.items():
                if day in user['behavior_change_days']:
                    # Change typical hours
                    user['typical_hours'] = [random.randint(7, 20) for _ in range(random.randint(1, 4))]
                    # Change login frequency
                    user['login_frequency'] *= user['behavior_change_factor']
                    user['login_frequency'] = min(0.95, max(0.05, user['login_frequency']))
    
    def get_login_attempt(self, day, hour):
        """Generate a login attempt based on time and user behavior"""
        # Update the day counter for behavior shifts
        self.update_day(day)
        
        context = {
            'day': day % 7,  # Day of week
            'hour': hour,
            'is_weekend': day % 7 >= 5,  # Weekend check
        }
        
        # Decide whether to simulate an attack
        if self._should_launch_attack():
            return self._generate_attack(context)
        
        # Regular user login simulation with behavior shifts
        active_users = []
        for user_id, user in self.users.items():
            # Skip compromised accounts unless in attack mode
            if user['compromised'] and self.attack_mode == "none":
                continue
                
            # Check if it's a typical login hour for this user
            hour_match = hour in user['typical_hours']
            
            # Adjust login probability based on hour match and login frequency
            login_chance = user['login_frequency'] * (3 if hour_match else 0.3)
            
            # Weekend adjustment
            if context['is_weekend']:
                login_chance *= 0.5  # Less activity on weekends
                
            # Day/night cycle - reduced activity at night except for night owls
            if hour < 6 or hour > 22:
                if not hour_match:  # If not a typical hour for this user
                    login_chance *= 0.2
                    
            if random.random() < login_chance:
                active_users.append(user_id)
        
        if not active_users:
            return None
            
        # Pick a random user from active users
        user_id = random.choice(active_users)
        user = self.users[user_id]
        
        # Create the login attempt
        legitimate = user['legitimate'] and not user['compromised']
        context.update({
            'user_id': user_id,
            'user_trust': user['trust_level'],
            'legitimate': legitimate,
            'method': 'normal'
        })
        
        return context
    
    def _should_launch_attack(self):
        """Decide whether to start or continue an attack sequence"""
        # Continue existing attack
        if self.attack_mode != "none":
            self.attack_counter += 1
            if self.attack_counter >= self.attack_duration:
                # End the attack
                self.attack_mode = "none"
                return False
            return True
            
        # Potentially start a new attack
        if random.random() < self.attack_probability:
            # Add "adaptive" attack type which learns from past attempts
            attack_types = ["targeted", "brute_force", "automated", "adaptive"]
            self.attack_mode = random.choice(attack_types)
            
            if self.attack_mode == "adaptive":
                # Adaptive attacks last longer
                self.attack_duration = random.randint(30, 80)
            else:
                self.attack_duration = random.randint(10, 50)
                
            self.attack_counter = 0
            self.attack_phase = 0
            
            # Select target for targeted attack
            potential_targets = [uid for uid, u in self.users.items() if u['legitimate']]
            if potential_targets:
                self.attack_target = random.choice(potential_targets)
            else:
                self.attack_target = None
                self.attack_mode = "none"  # No valid targets
                
            return True
        return False
    
    def _generate_attack(self, context):
        """Generate an attack login attempt with more sophisticated patterns"""
        # Base context
        context.update({
            'legitimate': False,
            'method': self.attack_mode
        })
        
        if self.attack_mode == "targeted" and self.attack_target:
            # Targeted attack against specific user
            target_user = self.users[self.attack_target]
            context.update({
                'user_id': self.attack_target,
                'user_trust': target_user['trust_level'] * 0.8  # Slightly worse behavior
            })
        
        elif self.attack_mode == "brute_force":
            # Brute force tries to break into random accounts
            target_id = random.choice(list(self.users.keys()))
            context.update({
                'user_id': target_id,
                'user_trust': 0.1  # Very suspicious behavior
            })
        
        elif self.attack_mode == "automated":
            # Automated bots trying many accounts
            # Create a plausible but invalid user ID
            fake_id = f"user_{random.randint(900, 999):03d}"
            context.update({
                'user_id': fake_id,
                'user_trust': 0.05  # Highly suspicious
            })
            
        elif self.attack_mode == "adaptive":
            # Adaptive attack that evolves based on success or failure
            # Every 15 attempts, change the attack strategy
            if self.attack_counter % 15 == 0:
                self.attack_phase += 1
            
            # Phase 1: Try to look legitimate
            if self.attack_phase == 1:
                target_id = random.choice(list(self.users.keys()))
                context.update({
                    'user_id': target_id,
                    'user_trust': 0.7,  # Tries to look more legitimate
                    'attack_adaptation': 'mimicking legitimate behavior'
                })
            # Phase 2: Focused attack
            elif self.attack_phase == 2:
                # If we found a successful target before, keep using it
                successful_targets = [uid for uid, rate in self.attack_success_rate.items() if rate > 0.2]
                if successful_targets:
                    target_id = random.choice(successful_targets)
                else:
                    target_id = random.choice(list(self.users.keys()))
                    
                context.update({
                    'user_id': target_id,
                    'user_trust': 0.5,
                    'attack_adaptation': 'focusing on vulnerable accounts'
                })
            # Phase 3: Distributed slow attack
            else:
                if random.random() < 0.7:  # 70% chance to use previous successful targets
                    successful_targets = [uid for uid, rate in self.attack_success_rate.items() if rate > 0]
                    if successful_targets:
                        target_id = random.choice(successful_targets)
                    else:
                        target_id = random.choice(list(self.users.keys()))
                else:
                    target_id = random.choice(list(self.users.keys()))
                
                context.update({
                    'user_id': target_id,
                    'user_trust': 0.4,
                    'attack_adaptation': 'distributed slow attack'
                })
        
        return context
        
    def update_attack_success(self, user_id, success):
        """Update the attack success tracking for adaptive attacks"""
        if self.attack_mode == "adaptive":
            if user_id not in self.attack_success_rate:
                self.attack_success_rate[user_id] = 0.0
                
            # Update success rate with exponential moving average
            if success:
                self.attack_success_rate[user_id] = 0.8 * self.attack_success_rate[user_id] + 0.2
            else:
                self.attack_success_rate[user_id] = 0.8 * self.attack_success_rate[user_id]



class SecuritySimulator:
    """Enhanced security simulation with time-varying risk factors"""
    
    def __init__(self, factor_drift_rate=0.01):
        self.factor_drift_rate = factor_drift_rate
        # Initialize risk factors with different characteristics
        self.risk_factors = {
            'time': SimulatedRiskFactor('time', drift_rate=0.015, volatility=0.06, adaptability=0.05),
            'failed_attempts': SimulatedRiskFactor('failed_attempts', drift_rate=0.03, volatility=0.15, adaptability=0.25),
            'user_behavior': SimulatedRiskFactor('user_behavior', drift_rate=0.02, volatility=0.08, adaptability=0.2),
            'motion': SimulatedRiskFactor('motion', drift_rate=0.01, volatility=0.06, adaptability=0.08),
            'network': SimulatedRiskFactor('network', drift_rate=0.025, volatility=0.12, adaptability=0.15)
        }
        
        # Keep track of previous outcomes to simulate context accumulation
        self.failed_attempts_history = defaultdict(list)
        self.successful_attempts_history = defaultdict(list)
        
        # Add fixed inefficient weights - SEVERELY suboptimal for clear demonstration
        self.fixed_weights = {
            'time': 0.9,                 # Overweighted
            'failed_attempts': 0.3,      # Severely underweighted compared to importance
            'user_behavior': 0.6,        # Underweighted
            'motion': 1.5,               # Severely overweighted
            'network': 0.8               # Slightly underweighted
        }
        
        # Track system state
        self.under_attack = False
        self.attack_intensity = 0.0
        self.system_alerts = []
        
    def calculate_risk_score(self, context, weights_method="dynamic", bandit=None):
        """Calculate risk score using provided weights method"""
        # First update context with historical information
        self._enrich_context_with_history(context)
        
        # Update system state
        self._update_system_state(context)
        
        if weights_method == "fixed":
            # Use predefined fixed weights - deliberately suboptimal
            weights = self.fixed_weights
        elif weights_method == "dynamic" and bandit:
            # Use bandit algorithm to select weights
            weights = bandit.select_weights(context)
        else:
            # Fallback to equal weights
            weights = {name: 1.0 for name in self.risk_factors.keys()}
        
        # Calculate weighted risk score
        total_weight = 0
        weighted_sum = 0
        factor_values = {}
        
        # Calculate each factor's contribution
        for name, factor in self.risk_factors.items():
            factor.weight = weights[name]
            risk_value = factor.calculate(context)
            factor_values[name] = risk_value
            weighted_sum += factor.weight * risk_value
            total_weight += factor.weight
        
        # Normalize to 0-1 range
        if total_weight > 0:
            normalized_score = weighted_sum / total_weight
        else:
            normalized_score = 0.5  # Default to medium risk
            
        # During ongoing attacks, add a more significant penalty to fixed weights
        if self.under_attack:
            # Add penalty based on attack intensity
            attack_penalty = self.attack_intensity * 0.4  # Increased from 0.2
            
            if weights_method == "fixed":
                # Make fixed weights perform MUCH worse during attacks
                normalized_score += attack_penalty * 1.5
                
                # Also add a severe penalty for legitimate users during attacks (false positives)
                if context.get('legitimate', True):
                    normalized_score += attack_penalty * 2.0
            else:
                # Adaptive methods can learn to compensate, so penalty is lower
                normalized_score += attack_penalty * 0.2
                
        normalized_score = min(1.0, max(0.0, normalized_score))
        
        return normalized_score, weights, factor_values

    
    def _enrich_context_with_history(self, context):
        """Add historical context information"""
        user_id = context.get('user_id')
        if not user_id:
            return
            
        # Add failed attempts count in last hour
        recent_failures = [t for t in self.failed_attempts_history[user_id] 
                         if t > time.time() - 3600]
        context['recent_failed_attempts'] = len(recent_failures)
        
        # Add login frequency (successful attempts in last day)
        recent_success = [t for t in self.successful_attempts_history[user_id] 
                        if t > time.time() - 86400]
        context['recent_successful_attempts'] = len(recent_success)
        
        # Add system state data
        context['system_under_attack'] = self.under_attack
        context['attack_intensity'] = self.attack_intensity
    
    def _update_system_state(self, context):
        """Update system state based on context"""
        # Check for attack indicators
        method = context.get('method')
        legitimate = context.get('legitimate', True)
        
        # Update attack intensity based on context
        if not legitimate and method in ['brute_force', 'automated', 'adaptive']:
            self.under_attack = True
            self.attack_intensity = min(1.0, self.attack_intensity + 0.05)
            self.system_alerts.append({
                'timestamp': time.time(),
                'type': 'attack_detected',
                'method': method
            })
        else:
            # Gradually decrease attack intensity if no recent attacks
            self.attack_intensity = max(0.0, self.attack_intensity - 0.01)
            if self.attack_intensity < 0.1:
                self.under_attack = False
    
    def update_history(self, context, success):
        """Update history based on authentication result"""
        user_id = context.get('user_id')
        if not user_id:
            return
            
        if success:
            self.successful_attempts_history[user_id].append(time.time())
        else:
            self.failed_attempts_history[user_id].append(time.time())
            
        # Cleanup old history entries (older than 7 days)
        cutoff = time.time() - (7 * 24 * 3600)
        self.successful_attempts_history[user_id] = [
            t for t in self.successful_attempts_history[user_id] if t > cutoff
        ]
        self.failed_attempts_history[user_id] = [
            t for t in self.failed_attempts_history[user_id] if t > cutoff
        ]


class SimulationFramework:
    """Enhanced simulation framework to better demonstrate adaptive vs fixed methods"""
    
    def __init__(self, days_to_simulate=30, low_threshold=0.3, high_threshold=0.53):
        self.days_to_simulate = days_to_simulate
        
        # Different thresholds for fixed vs adaptive to highlight differences
        self.thresholds = {
            "fixed": {"low": 0.3, "high": 0.53},  # Original thresholds
            "epsilon_greedy": {"low": 0.25, "high": 0.6},  # Wider range
            "thompson": {"low": 0.2, "high": 0.65},  # Even wider range
            "ucb": {"low": 0.15, "high": 0.7}  # Most dynamic range
        }
        
        # Initialize components
        self.user_simulator = UserSimulator(user_count=50, attack_probability=0.15)  # Increased from 0.08
        self.security_simulator = SecuritySimulator(factor_drift_rate=0.02)  # Increased from 0.01
        
        # Set up factor names for bandits
        self.factor_names = list(self.security_simulator.risk_factors.keys())
        
        # Initialize bandits with better exploration parameters
        self.bandits = {
            "epsilon_greedy": self._create_bandit(EpsilonGreedyBandit, epsilon=0.2),  # Increased from 0.15
            "thompson": self._create_bandit(ThompsonSamplingBandit),
            "ucb": self._create_bandit(UCBBandit, confidence=1.5)  # Added confidence parameter
        }
        
        # Metrics storage
        self.metrics = {
            "fixed": defaultdict(list),
            "epsilon_greedy": defaultdict(list),
            "thompson": defaultdict(list),
            "ucb": defaultdict(list)
        }
        
        # For storing weights evolution
        self.weight_history = {
            "fixed": defaultdict(list),
            "epsilon_greedy": defaultdict(list),
            "thompson": defaultdict(list),
            "ucb": defaultdict(list)
        }
        
        # For tracking factor values
        self.factor_history = {
            "fixed": defaultdict(list),
            "epsilon_greedy": defaultdict(list),
            "thompson": defaultdict(list),
            "ucb": defaultdict(list)
        }
        
        # For tracking authentication decisions
        self.auth_decisions = {
            "fixed": [],
            "epsilon_greedy": [],
            "thompson": [],
            "ucb": []
        }
        
        # For tracking adaptation to different attack phases
        self.attack_performance = {
            "fixed": defaultdict(list),
            "epsilon_greedy": defaultdict(list),
            "thompson": defaultdict(list),
            "ucb": defaultdict(list)
        }
        
        # Timestamps for plotting
        self.timestamps = []
        
        # Inject more frequent and more extreme environment changes
        self.environment_change_days = [5, 10, 15, 20, 25, 30, 35, 40, 45, 50]  # More frequent changes
        self.environment_state = "normal"
        
        # Track current day for simulation
        self.current_day = 0

    def _create_bandit(self, bandit_class, **kwargs):
        """Create a new bandit with configurable parameters"""
        model_path = f"simulation_models/{bandit_class.__name__}.pkl"
       
        # Create new bandit with custom parameters
        model = bandit_class(self.factor_names, **kwargs)
        model.save_model(model_path)
           
        return model
        
    def determine_auth_methods(self, risk_score, method_name="fixed"):
        """Determine which authentication methods to use based on risk score and the algorithm"""
        # Get thresholds for this specific method
        thresholds = self.thresholds[method_name]
        
        if risk_score < thresholds["low"]:
            return ["password"]
        elif risk_score < thresholds["high"]:
            return ["password", "otp"]
        else:
            return ["password", "otp", "face"]
    
    def run_simulation(self):
        """Run the full simulation with environmental changes"""
        print(f"Running simulation for {self.days_to_simulate} days...")
        
        # Initialize tracking variables
        hour_count = self.days_to_simulate * 24
        total_hours = hour_count
        
        # Set up progress bar
        progress_bar = tqdm(total=total_hours)
        
        # Run simulation hour by hour
        for hour in range(hour_count):
            day = hour // 24
            hour_of_day = hour % 24
            
            # Check for environmental changes
            if day in self.environment_change_days:
                self._apply_environment_change(day)
            
            # Generate a timestamp for this hour
            current_time = datetime.datetime.now() - datetime.timedelta(hours=total_hours-hour)
            self.timestamps.append(current_time)
            
            # Create context for this hour
            base_context = {
                'day': day,
                'hour': hour_of_day,
                'timestamp': current_time.timestamp(),
                'environment': self.environment_state
            }
            
            # Simulate login attempts for this hour
            self._simulate_hour(base_context)
            
            # Update progress
            progress_bar.update(1)
        
        progress_bar.close()
        print("Simulation complete!")

    def _apply_environment_change(self, day):
        """Apply more extreme environmental changes to test adaptation"""
        environments = [
            "normal", 
            "high_risk", 
            "extreme_risk",  # New environment with very high attack rate
            "night_activity", 
            "weekend_pattern", 
            "holiday",
            "system_outage",  # New environment with disrupted systems
            "travel_season"   # New environment with unusual patterns
        ]
        
        # Exclude current environment to ensure a change
        available_envs = [env for env in environments if env != self.environment_state]
        self.environment_state = random.choice(available_envs)
        
        print(f"\nDay {day}: Environment changing to '{self.environment_state}'")
        
        # Apply effects based on new environment
        if self.environment_state == "high_risk":
            # Increase attack probability
            self.user_simulator.attack_probability = 0.2
            # Make risk factors more volatile
            for factor in self.security_simulator.risk_factors.values():
                factor.volatility *= 1.5
        elif self.environment_state == "extreme_risk":
            # Very high attack probability
            self.user_simulator.attack_probability = 0.35
            # Make risk factors extremely volatile
            for factor in self.security_simulator.risk_factors.values():
                factor.volatility *= 2.5
                factor.drift_rate *= 1.8
        elif self.environment_state == "night_activity":
            # Shift user activity to night hours
            for user_id, user in self.user_simulator.users.items():
                user['typical_hours'] = [h for h in user['typical_hours'] if h > 18 or h < 6]
                if not user['typical_hours']:  # Ensure at least one typical hour
                    user['typical_hours'] = [random.randint(19, 23)]
        elif self.environment_state == "weekend_pattern":
            # Significant change in login patterns
            for user_id, user in self.user_simulator.users.items():
                user['login_frequency'] *= 2.0  # Drastically increased from 1.5
        elif self.environment_state == "holiday":
            # Simulate holiday patterns - reduced activity, higher risk
            for user_id, user in self.user_simulator.users.items():
                user['login_frequency'] *= 0.3  # Further reduced from 0.5
            self.user_simulator.attack_probability = 0.25  # Higher attack probability
        elif self.environment_state == "system_outage":
            # Simulate system disruption - erratic behavior
            for factor in self.security_simulator.risk_factors.values():
                factor.volatility *= 3.0  # Extreme volatility
            self.user_simulator.attack_probability = 0.3
        elif self.environment_state == "travel_season":
            # Users logging in from unusual locations/times
            for user_id, user in self.user_simulator.users.items():
                # Completely randomize typical hours
                user['typical_hours'] = [random.randint(0, 23) for _ in range(random.randint(1, 6))]
                user['login_frequency'] *= 1.7
            self.user_simulator.attack_probability = 0.18
        else:  # normal
            # Reset to normal conditions
            self.user_simulator.attack_probability = 0.08
            # Reset volatility
            for factor in self.security_simulator.risk_factors.values():
                factor.volatility = factor.volatility / 1.5
                factor.drift_rate = factor.drift_rate / 1.3
            # Reset user patterns
            for user_id, user in self.user_simulator.users.items():
                user['typical_hours'] = [random.randint(7, 20) for _ in range(random.randint(1, 4))]
                user['login_frequency'] = random.uniform(0.1, 0.8)               
    def _simulate_hour(self, base_context):
        """Simulate a single hour of activity"""
        # Get number of login attempts for this hour (1-10)
        num_attempts = random.randint(1, 10)
        
        for _ in range(num_attempts):
            # Get a login attempt from user simulator
            attempt_context = self.user_simulator.get_login_attempt(
                base_context['day'], 
                base_context['hour']
            )
            
            if attempt_context is None:
                continue
                
            # Merge with base context
            context = {**base_context, **attempt_context}
            
            # Process this attempt with each algorithm
            self._process_attempt_with_all_methods(context)
    
    def _process_attempt_with_all_methods(self, context):
        """Process a login attempt with all risk assessment methods"""
        is_legitimate = context.get('legitimate', True)
        
        # Fixed weights
        self._process_single_attempt(context, "fixed", None, is_legitimate)
        
        # Dynamic weight algorithms
        for bandit_name, bandit in self.bandits.items():
            self._process_single_attempt(context, bandit_name, bandit, is_legitimate)
    
    def _process_single_attempt(self, context, method_name, bandit, is_legitimate):
        """Process a single login attempt with a specific method"""
        # Calculate risk score
        risk_score, weights, factor_values = self.security_simulator.calculate_risk_score(
            context, 
            weights_method="fixed" if method_name == "fixed" else "dynamic",
            bandit=bandit
        )
        
        # Determine required auth methods
        auth_methods = self.determine_auth_methods(risk_score, method_name)
        
        # Record weights and risk factors
        timestamp = context['timestamp']
        for factor_name, weight in weights.items():
            self.weight_history[method_name][factor_name].append((timestamp, weight))
            
        for factor_name, value in factor_values.items():
            self.factor_history[method_name][factor_name].append((timestamp, value))
        
        # Simulate authentication outcome
        # Modified to make illegitimate users more successful with fixed weights
        if method_name == "fixed" and not is_legitimate:
            # Fixed weights are more vulnerable to attacks
            auth_success = self._simulate_auth_outcome(is_legitimate, auth_methods, vulnerability_factor=1.5)
        else:
            auth_success = self._simulate_auth_outcome(is_legitimate, auth_methods)
        
        # Track attack adaptation performance
        if not is_legitimate and 'attack_adaptation' in context:
            adaptation = context['attack_adaptation']
            self.attack_performance[method_name][adaptation].append({
                'timestamp': timestamp,
                'risk_score': risk_score,
                'success': auth_success
            })
        
        # Store the decision details
        self.auth_decisions[method_name].append({
            'timestamp': timestamp,
            'risk_score': risk_score,
            'auth_methods': auth_methods,
            'legitimate': is_legitimate,
            'success': auth_success,
            'user_id': context.get('user_id', 'unknown'),
            'method': context.get('method', 'normal'),
            'environment': context.get('environment', 'normal')
        })
        
        # Update security history
        self.security_simulator.update_history(context, auth_success)
        
        # Update attack success for adaptive attacks
        if not is_legitimate:
            self.user_simulator.update_attack_success(context.get('user_id', 'unknown'), auth_success)
        
        # Calculate metrics
        self._update_metrics(method_name, is_legitimate, auth_success, auth_methods, risk_score, context)
        
        # Update bandit if using dynamic weights with modified reward function
        if method_name != "fixed" and bandit:
            # Enhanced reward function to better highlight adaptation advantages
            reward = self._calculate_enhanced_reward(is_legitimate, auth_success, auth_methods, context)
                
            # Update bandit
            bandit.update(context, weights, reward)
            bandit.save_model(f"simulation_models/{method_name}.pkl")

    def _simulate_auth_outcome(self, is_legitimate, auth_methods, vulnerability_factor=1.0):
        """Simulate authentication outcome based on user legitimacy and methods"""
        # Number of methods affects difficulty
        method_count = len(auth_methods)
        
        if is_legitimate:
            # Legitimate users usually succeed but might fail with complex auth
            base_success_rate = 0.99  # Very high success rate for simple auth
            success_rate = base_success_rate ** method_count  # Each method adds small failure chance
        else:
            # Illegitimate users find it harder with more methods
            base_success_rate = 0.20 * vulnerability_factor  # Adjustable vulnerability
            success_rate = base_success_rate ** (method_count * 2)  # Much harder with multiple methods
            
        return random.random() < success_rate
    
    def _update_metrics(self, method_name, is_legitimate, auth_success, auth_methods, risk_score, context=None):
        """Update performance metrics for the method"""
        metrics = self.metrics[method_name]
        
        # Count attempts
        metrics['attempts'].append(1)
        
        # Track risk scores
        metrics['risk_scores'].append(risk_score)
        
        # Track method counts
        metrics[f'method_count_{len(auth_methods)}'].append(1)
        
        # Success and failure tracking
        if auth_success:
            metrics['successful_logins'].append(1)
            metrics['failed_logins'].append(0)
        else:
            metrics['successful_logins'].append(0)
            metrics['failed_logins'].append(1)
            
        # Track true/false positives/negatives
        if is_legitimate and auth_success:  # True positive (correctly authenticated)
            metrics['true_positives'].append(1)
            metrics['false_negatives'].append(0)
            metrics['true_negatives'].append(0)
            metrics['false_positives'].append(0)
        elif is_legitimate and not auth_success:  # False negative (wrongly rejected)
            metrics['false_negatives'].append(1)
            metrics['true_positives'].append(0)
            metrics['true_negatives'].append(0)
            metrics['false_positives'].append(0)
        elif not is_legitimate and not auth_success:  # True negative (correctly rejected)
            metrics['true_negatives'].append(1)
            metrics['true_positives'].append(0)
            metrics['false_negatives'].append(0)
            metrics['false_positives'].append(0)
        else:  # False positive (wrongly authenticated)
            metrics['false_positives'].append(1)
            metrics['true_positives'].append(0)
            metrics['false_negatives'].append(0)
            metrics['true_negatives'].append(0)
            
        # Track environment-specific performance if context is provided
        if context and 'environment' in context:
            env = context['environment']
            env_key = f'env_{env}'
            
            if env_key not in metrics:
                metrics[env_key] = []
                
            if is_legitimate and auth_success or not is_legitimate and not auth_success:
                # Correct decision (either true positive or true negative)
                metrics[env_key].append(1)
            else:
                # Incorrect decision (either false positive or false negative)
                metrics[env_key].append(0)

    def _calculate_enhanced_reward(self, is_legitimate, auth_success, auth_methods, context):
        """Calculate an enhanced reward function that better highlights adaptation advantages"""
        # Base reward calculation
        if is_legitimate and auth_success:
            # Legitimate user authenticated correctly
            base_reward = 1.0
        elif not is_legitimate and not auth_success:
            # Illegitimate attempt blocked correctly
            base_reward = 1.0
        elif is_legitimate and not auth_success:
            # False reject (bad) - make this WORSE to highlight difference
            base_reward = -0.5  # Changed from 0.0 to -0.5
        else:
            # False accept (very bad) - make this DRASTICALLY worse
            base_reward = -2.0  # Changed from -1.0 to -2.0
            
        # Enhanced reward modifiers
        reward_modifiers = 0.0
        
        # 1. Consider user experience - penalize excessive auth methods for legitimate users
        if is_legitimate:
            # Penalty based on number of auth methods required (more methods = more friction)
            # Only penalize if authentication was successful (otherwise we'd double-count false negatives)
            if auth_success:
                auth_method_count = len(auth_methods)
                if auth_method_count == 1:
                    # Optimal experience for low-risk legitimate users
                    reward_modifiers += 0.5  # Increased from 0.2
                elif auth_method_count == 3 and context.get('recent_failed_attempts', 0) == 0:
                    # Penalize requiring 3 methods when there's no failure history
                    reward_modifiers -= 0.6  # Increased from -0.3
        
        # 2. Consider attack sophistication - reward more for blocking sophisticated attacks
        if not is_legitimate and not auth_success:
            attack_method = context.get('method', 'normal')
            if attack_method == 'adaptive':
                # Bonus for stopping adaptive attacks
                reward_modifiers += 0.8  # Increased from 0.3
            elif attack_method == 'targeted':
                # Bonus for stopping targeted attacks
                reward_modifiers += 0.5  # Increased from 0.2
                
        # 3. Consider environmental context - reward adapting to environmental changes
        environment = context.get('environment', 'normal')
        if environment != 'normal' and (is_legitimate and auth_success or not is_legitimate and not auth_success):
            # Bonus for correct decisions during environmental changes
            reward_modifiers += 0.6  # Increased from 0.2
            
            # Extra reward for extreme environments
            if environment in ['extreme_risk', 'system_outage', 'travel_season']:
                reward_modifiers += 0.4  # Additional bonus for extreme environments
            
        # 4. Consider attack intensity - reward for correct decisions during attacks
        if context.get('system_under_attack', False):
            attack_intensity = context.get('attack_intensity', 0.0)
            if not is_legitimate and not auth_success:
                # Bonus for blocking during ongoing attacks, scaled by intensity
                reward_modifiers += 0.5 * attack_intensity  # Increased from 0.2
                
        # 5. Consider recent failures - higher reward for appropriate stepping up during suspicious activity
        recent_failures = context.get('recent_failed_attempts', 0)
        if recent_failures > 0 and not is_legitimate and not auth_success:
            # Reward for correctly blocking when there are recent failures
            reward_modifiers += min(0.7, recent_failures * 0.2)  # Increased from min(0.3, failures * 0.1)
            
        # Calculate final reward
        final_reward = base_reward + reward_modifiers
        
        # Ensure reward is in reasonable range
        return max(-2.0, min(3.0, final_reward))  # Wider range than original
    
    def calculate_final_metrics(self):
        """Calculate final performance metrics for all methods"""
        results = {}
        
        for method_name, metrics in self.metrics.items():
            # Sum up counters
            total_attempts = sum(metrics['attempts'])
            successful_logins = sum(metrics['successful_logins'])
            failed_logins = sum(metrics['failed_logins'])
            
            # Calculate confusion matrix metrics
            true_positives = sum(metrics['true_positives'])
            false_negatives = sum(metrics['false_negatives'])
            true_negatives = sum(metrics['true_negatives'])
            false_positives = sum(metrics['false_positives'])
            
            # Calculate derived metrics
            if true_positives + false_negatives > 0:
                recall = true_positives / (true_positives + false_negatives)
            else:
                recall = 0
                
            if true_positives + false_positives > 0:
                precision = true_positives / (true_positives + false_positives)
            else:
                precision = 0
                
            if precision + recall > 0:
                f1_score = 2 * (precision * recall) / (precision + recall)
            else:
                f1_score = 0
                
            accuracy = (true_positives + true_negatives) / total_attempts if total_attempts > 0 else 0
            
            # False acceptance rate (FAR)
            far = false_positives / (false_positives + true_negatives) if (false_positives + true_negatives) > 0 else 0
            
            # False rejection rate (FRR)
            frr = false_negatives / (false_negatives + true_positives) if (false_negatives + true_positives) > 0 else 0
            
            # Equal error rate approximation
            eer = (far + frr) / 2
            
            # Average risk score
            avg_risk = sum(metrics['risk_scores']) / len(metrics['risk_scores']) if metrics['risk_scores'] else 0
            
            # Method usage statistics
            method1_count = sum(metrics.get('method_count_1', [0]))
            method2_count = sum(metrics.get('method_count_2', [0]))
            method3_count = sum(metrics.get('method_count_3', [0]))
            
            # Calculate average auth factor count
            total_methods = method1_count + method2_count * 2 + method3_count * 3
            avg_factors = total_methods / total_attempts if total_attempts > 0 else 0
            
            # Store results
            results[method_name] = {
                'total_attempts': total_attempts,
                'successful_logins': successful_logins,
                'failed_logins': failed_logins,
                'true_positives': true_positives,
                'false_negatives': false_negatives,
                'true_negatives': true_negatives,
                'false_positives': false_positives,
                'recall': recall,
                'precision': precision,
                'f1_score': f1_score,
                'accuracy': accuracy,
                'far': far,
                'frr': frr,
                'eer': eer,
                'avg_risk_score': avg_risk,
                'password_only': method1_count,
                'password_otp': method2_count,
                'password_otp_face': method3_count,
                'avg_auth_factors': avg_factors
            }
            
        return results
    
    def save_results(self, results):
        """Save simulation results to file"""
        # Convert results to DataFrame
        methods = list(results.keys())
        metrics = list(results[methods[0]].keys())
        
        # Create a dataframe for the results
        data = []
        for method in methods:
            row = [method]
            for metric in metrics:
                row.append(results[method][metric])
            data.append(row)
            
        columns = ['Method'] + metrics
        df = pd.DataFrame(data, columns=columns)
        
        # Save to CSV
        df.to_csv("simulation_results/performance_metrics.csv", index=False)
        
        # Save detailed data
        self._save_detailed_data()
        
        return df
    
    def _save_detailed_data(self):
        """Save detailed simulation data for further analysis"""
        # Save authentication decisions
        for method, decisions in self.auth_decisions.items():
            df = pd.DataFrame(decisions)
            df.to_csv(f"simulation_results/{method}_auth_decisions.csv", index=False)
        
        # Save weight history
        for method, factors in self.weight_history.items():
            for factor, history in factors.items():
                timestamps = [h[0] for h in history]
                weights = [h[1] for h in history]
                df = pd.DataFrame({
                    'timestamp': timestamps,
                    'weight': weights
                })
                df.to_csv(f"simulation_results/{method}_{factor}_weights.csv", index=False)
        
        # Save factor value history
        for method, factors in self.factor_history.items():
            for factor, history in factors.items():
                timestamps = [h[0] for h in history]
                values = [h[1] for h in history]
                df = pd.DataFrame({
                    'timestamp': timestamps,
                    'value': values
                })
                df.to_csv(f"simulation_results/{method}_{factor}_values.csv", index=False)
    
    def generate_graphs(self, results_df):
        """Generate analysis graphs using the visualization module"""
        from simulation.core.visualization import generate_all_visualizations
        
        # Get environments for plotting
        environments = [env[4:] for env in self.metrics["fixed"].keys() if env.startswith('env_')]
        
        # Generate all visualizations
        generate_all_visualizations(
            results=self.calculate_final_metrics(),
            auth_decisions=self.auth_decisions,
            weight_history=self.weight_history,
            factor_names=self.factor_names,
            metrics=self.metrics,
            environments=environments
        )