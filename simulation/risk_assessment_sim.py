import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import os
import logging
from typing import Dict, List, Tuple
import random
from scipy import stats

from RiskAssessment.TimeBasedRiskFactor import TimeBasedRiskFactor
from RiskAssessment.FailedAttemptsRiskFactor import FailedAttemptsRiskFactor
from RiskAssessment.UserBehaviorRiskFactor import UserBehaviorRiskFactor
from RiskAssessment.MotionActivityRiskFactor import MotionActivityRiskFactor
from RiskAssessment.NetworkBasedRiskFactor import NetworkBasedRiskFactor

from ContextualBandits.EpsilonGreedyBandit import EpsilonGreedyBandit
from ContextualBandits.ThompsonSamplingBandit import ThompsonSamplingBandit
from ContextualBandits.UCBBandit import UCBBandit

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("simulation.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("RiskAssessmentSim")

class RiskAssessmentSimulation:
    def __init__(self, duration_hours: int = 24, time_step_minutes: int = 15):
        self.duration_hours = duration_hours
        self.time_step_minutes = time_step_minutes
        self.results_dir = "results"
        os.makedirs(self.results_dir, exist_ok=True)
        
        # Initialize risk factors
        self.risk_factors = {
            "time": TimeBasedRiskFactor(weight=1.0),
            "failed_attempts": FailedAttemptsRiskFactor(weight=1.5),
            "user_behavior": UserBehaviorRiskFactor(weight=1.2),
            "motion": MotionActivityRiskFactor(weight=0.8),
            "network": NetworkBasedRiskFactor(weight=0.5)
        }
        
        # Initialize bandits
        factor_names = list(self.risk_factors.keys())
        self.bandit_algorithms = {
            "fixed_weights": self._create_fixed_weights(factor_names),
            "epsilon_greedy": EpsilonGreedyBandit(factor_names),
            "thompson": ThompsonSamplingBandit(factor_names),
            "ucb": UCBBandit(factor_names)
        }
        
        # Simulation metrics
        self.metrics = {
            "risk_scores": [],
            "auth_methods": [],
            "detection_rates": [],
            "weights": [],
            "timestamps": [],
            "algorithms": [],
            "attack_detected": [],
            "false_positives": [],
            "false_negatives": []
        }
        
        # Context drift parameters
        self.drift_rate = 0.05
        self.current_drift = 0.0
        
    def _create_fixed_weights(self, factor_names: List[str]) -> Dict[str, float]:
        """Create fixed weights for baseline comparison"""
        return {name: 1.0 for name in factor_names}
        
    def _generate_context(self, timestamp: datetime) -> Dict:
        """Generate context with drift"""
        # Base context
        context = {
            "hour": timestamp.hour,
            "day_of_week": timestamp.weekday(),
            "failed_attempts": random.randint(0, 3),
            "user_location": random.choice(["home", "office", "public"]),
            "network_type": random.choice(["wifi", "cellular", "vpn"]),
            "motion_level": random.uniform(0, 1)
        }
        
        # Apply context drift
        if random.random() < self.drift_rate:
            self.current_drift += 0.1
            context["motion_level"] = min(1.0, context["motion_level"] + self.current_drift)
            context["network_type"] = random.choice(["wifi", "cellular", "vpn", "unknown"])
            
        return context
        
    def _calculate_risk_score(self, context: Dict, algorithm: str) -> Tuple[float, Dict[str, float]]:
        """Calculate risk score using selected algorithm"""
        if algorithm == "fixed_weights":
            weights = self.bandit_algorithms[algorithm]
        else:
            weights = self.bandit_algorithms[algorithm].select_weights(context)
            
        total_weight = 0
        weighted_sum = 0
        
        for name, factor in self.risk_factors.items():
            factor.weight = weights[name]
            risk_value = factor.get_normalized_value(context)
            weighted_sum += factor.weight * risk_value
            total_weight += factor.weight
            
        normalized_score = weighted_sum / total_weight if total_weight > 0 else 0.5
        
        return normalized_score, weights
        
    def _determine_auth_method(self, risk_score: float) -> str:
        """Determine authentication method based on risk score"""
        if risk_score < 0.3:
            return "password"
        elif risk_score < 0.6:
            return "password_otp"
        else:
            return "password_otp_face"
            
    def _simulate_attack(self, context: Dict) -> bool:
        """Simulate attack attempt"""
        # Higher attack probability during off-hours
        if 0 <= context["hour"] < 6:
            return random.random() < 0.2
        return random.random() < 0.05
        
    def run_simulation(self):
        """Run the complete simulation"""
        current_time = datetime.now()
        end_time = current_time + timedelta(hours=self.duration_hours)
        
        while current_time < end_time:
            context = self._generate_context(current_time)
            is_attack = self._simulate_attack(context)
            
            for algo_name in self.bandit_algorithms.keys():
                risk_score, weights = self._calculate_risk_score(context, algo_name)
                auth_method = self._determine_auth_method(risk_score)
                
                # Record metrics
                self.metrics["risk_scores"].append(risk_score)
                self.metrics["auth_methods"].append(auth_method)
                self.metrics["weights"].append(weights)
                self.metrics["timestamps"].append(current_time)
                self.metrics["algorithms"].append(algo_name)
                
                # Record attack detection
                attack_detected = risk_score > 0.7  # High risk score indicates attack
                self.metrics["attack_detected"].append(attack_detected)
                
                # Record false positives/negatives
                if is_attack and not attack_detected:
                    self.metrics["false_negatives"].append(1)
                elif not is_attack and attack_detected:
                    self.metrics["false_positives"].append(1)
                else:
                    self.metrics["false_negatives"].append(0)
                    self.metrics["false_positives"].append(0)
                
                # Update bandit if not fixed weights
                if algo_name != "fixed_weights" and not is_attack:
                    self.bandit_algorithms[algo_name].update(context, weights, 1.0)
                    
            current_time += timedelta(minutes=self.time_step_minutes)
            
    def plot_results(self):
        """Generate publication-ready plots"""
        # Set publication settings
        plt.style.use('seaborn-whitegrid')
        plt.rcParams.update({
            'font.family': 'Arial',
            'font.size': 12,
            'figure.dpi': 300
        })
        
        # 1. Risk Score Evolution
        self._plot_risk_evolution()
        
        # 2. Authentication Method Distribution
        self._plot_auth_distribution()
        
        # 3. Weight Convergence
        self._plot_weight_convergence()
        
        # 4. Security vs Usability Tradeoff
        self._plot_security_usability()
        
        # 5. Performance Comparison with Fixed Weights
        self._plot_performance_comparison()
        
        # 6. Attack Detection Analysis
        self._plot_attack_detection()
        
    def _plot_risk_evolution(self):
        """Plot risk score evolution over time"""
        fig, ax = plt.subplots(figsize=(12, 6))
        
        for algo in self.bandit_algorithms.keys():
            mask = [a == algo for a in self.metrics["algorithms"]]
            risk_scores = np.array(self.metrics["risk_scores"])[mask]
            timestamps = np.array(self.metrics["timestamps"])[mask]
            
            ax.plot(timestamps, risk_scores, label=algo, alpha=0.7)
            
        ax.set_xlabel("Time")
        ax.set_ylabel("Risk Score")
        ax.set_title("Risk Score Evolution")
        ax.legend()
        plt.tight_layout()
        plt.savefig(f"{self.results_dir}/risk_evolution.pdf")
        plt.close()
        
    def _plot_auth_distribution(self):
        """Plot authentication method distribution"""
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Create DataFrame for easier plotting
        df = pd.DataFrame({
            'algorithm': self.metrics["algorithms"],
            'auth_method': self.metrics["auth_methods"]
        })
        
        # Plot grouped bar chart
        auth_counts = df.groupby(['algorithm', 'auth_method']).size().unstack()
        auth_counts.plot(kind='bar', ax=ax)
        
        ax.set_xlabel("Algorithm")
        ax.set_ylabel("Count")
        ax.set_title("Authentication Method Distribution by Algorithm")
        plt.tight_layout()
        plt.savefig(f"{self.results_dir}/auth_distribution.pdf")
        plt.close()
        
    def _plot_weight_convergence(self):
        """Plot weight convergence for each risk factor"""
        fig, axes = plt.subplots(len(self.risk_factors), 1, figsize=(12, 15))
        
        for i, (factor_name, _) in enumerate(self.risk_factors.items()):
            for algo in self.bandit_algorithms.keys():
                if algo != "fixed_weights":
                    mask = [a == algo for a in self.metrics["algorithms"]]
                    weights = [w[factor_name] for w in np.array(self.metrics["weights"])[mask]]
                    timestamps = np.array(self.metrics["timestamps"])[mask]
                    
                    axes[i].plot(timestamps, weights, label=algo, alpha=0.7)
                    
            axes[i].set_title(f"{factor_name} Weight Convergence")
            axes[i].legend()
            
        plt.tight_layout()
        plt.savefig(f"{self.results_dir}/weight_convergence.pdf")
        plt.close()
        
    def _plot_security_usability(self):
        """Plot security vs usability tradeoff"""
        fig, ax = plt.subplots(figsize=(10, 6))
        
        for algo in self.bandit_algorithms.keys():
            mask = [a == algo for a in self.metrics["algorithms"]]
            risk_scores = np.array(self.metrics["risk_scores"])[mask]
            auth_methods = np.array(self.metrics["auth_methods"])[mask]
            
            # Calculate security and usability scores
            security_score = np.mean(risk_scores)
            usability_score = 1 - len([m for m in auth_methods if m == "password_otp_face"]) / len(auth_methods)
            
            ax.scatter(security_score, usability_score, label=algo, s=100)
            
        ax.set_xlabel("Security Score")
        ax.set_ylabel("Usability Score")
        ax.set_title("Security vs Usability Tradeoff")
        ax.legend()
        plt.tight_layout()
        plt.savefig(f"{self.results_dir}/security_usability.pdf")
        plt.close()
        
    def _plot_performance_comparison(self):
        """Plot performance comparison with fixed weights"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # Convert metrics to DataFrame for easier analysis
        df = pd.DataFrame({
            'algorithm': self.metrics["algorithms"],
            'risk_score': self.metrics["risk_scores"],
            'auth_method': self.metrics["auth_methods"],
            'false_positive': self.metrics["false_positives"],
            'false_negative': self.metrics["false_negatives"]
        })
        
        # 1. Average Risk Score Comparison
        avg_risk = df.groupby('algorithm')['risk_score'].mean()
        avg_risk.plot(kind='bar', ax=axes[0,0])
        axes[0,0].set_title("Average Risk Score")
        axes[0,0].set_ylabel("Risk Score")
        
        # 2. False Positive Rate
        fp_rate = df.groupby('algorithm')['false_positive'].mean()
        fp_rate.plot(kind='bar', ax=axes[0,1])
        axes[0,1].set_title("False Positive Rate")
        axes[0,1].set_ylabel("Rate")
        
        # 3. False Negative Rate
        fn_rate = df.groupby('algorithm')['false_negative'].mean()
        fn_rate.plot(kind='bar', ax=axes[1,0])
        axes[1,0].set_title("False Negative Rate")
        axes[1,0].set_ylabel("Rate")
        
        # 4. Authentication Method Distribution
        auth_dist = pd.crosstab(df['algorithm'], df['auth_method'], normalize='index')
        auth_dist.plot(kind='bar', stacked=True, ax=axes[1,1])
        axes[1,1].set_title("Authentication Method Distribution")
        axes[1,1].set_ylabel("Proportion")
        
        plt.tight_layout()
        plt.savefig(f"{self.results_dir}/performance_comparison.pdf")
        plt.close()
        
    def _plot_attack_detection(self):
        """Plot attack detection analysis"""
        fig, axes = plt.subplots(2, 1, figsize=(12, 10))
        
        # Convert metrics to DataFrame
        df = pd.DataFrame({
            'algorithm': self.metrics["algorithms"],
            'attack_detected': self.metrics["attack_detected"],
            'false_positive': self.metrics["false_positives"],
            'false_negative': self.metrics["false_negatives"]
        })
        
        # 1. Attack Detection Rate
        detection_rate = df.groupby('algorithm')['attack_detected'].mean()
        detection_rate.plot(kind='bar', ax=axes[0])
        axes[0].set_title("Attack Detection Rate")
        axes[0].set_ylabel("Detection Rate")
        
        # 2. Error Analysis
        error_rates = pd.DataFrame({
            'False Positive': df.groupby('algorithm')['false_positive'].mean(),
            'False Negative': df.groupby('algorithm')['false_negative'].mean()
        })
        error_rates.plot(kind='bar', ax=axes[1])
        axes[1].set_title("Error Rates")
        axes[1].set_ylabel("Rate")
        
        plt.tight_layout()
        plt.savefig(f"{self.results_dir}/attack_detection.pdf")
        plt.close()

if __name__ == "__main__":
    # Run simulation
    sim = RiskAssessmentSimulation(duration_hours=24, time_step_minutes=15)
    sim.run_simulation()
    sim.plot_results() 