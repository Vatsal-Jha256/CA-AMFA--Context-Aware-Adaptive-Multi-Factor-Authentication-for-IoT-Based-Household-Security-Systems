#!/usr/bin/env python3
"""
Compare Bandit vs Fixed Weight Risk Assessment

This script compares the performance of the contextual bandit algorithm
with a fixed-weight risk assessment approach using improved methodology
for fair and meaningful comparison.
"""
#TODO: potentially change exploration rate over time and test other bandit algorithms
import os
import sys
import sqlite3
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import json
import argparse
from tqdm import tqdm
from sklearn.model_selection import KFold
from sklearn.metrics import roc_curve, auc, precision_recall_curve, f1_score, confusion_matrix, average_precision_score
import seaborn as sns

# Add the parent directory to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.core.contextual_bandits import RiskAssessmentBandit
from app.core.risk_assessment import calculate_risk_score

# Risk level thresholds
RISK_THRESHOLDS = {
    'low': 0.3,
    'medium': 0.6,
    'high': 0.8
}

# Fixed weights for comparison
FIXED_WEIGHTS = {
    'time': 0.2,
    'location': 0.3,
    'history': 0.25,
    'device': 0.15,
    'motion': 0.1
}

def connect_to_db():
    """Connect to the SQLite database."""
    db_path = os.path.join('instance', 'site.db')
    return sqlite3.connect(db_path)

def generate_synthetic_data(num_samples=10000, noise_level=0.1, concept_drift=False):
    """
    Generate synthetic data for evaluation with improved methodology.
    
    Args:
        num_samples: Number of samples to generate
        noise_level: Level of noise to add to the data (0.0 to 1.0)
        concept_drift: Whether to simulate concept drift
        
    Returns:
        pandas.DataFrame: Generated synthetic data
    """
    conn = connect_to_db()
    
    # Get user IDs from the database
    cursor = conn.cursor()
    cursor.execute("SELECT id FROM users")
    user_ids = [row[0] for row in cursor.fetchall()]
    
    if not user_ids:
        user_ids = [1, 2, 3, 4, 5]  # Default user IDs if none found
    
    # Define user behavior profiles with more realistic patterns
    user_profiles = {
        user_id: {
            'usual_hours': (np.random.uniform(0.1, 0.3), np.random.uniform(0.6, 0.8)),  # Work hours
            'usual_locations': [f"192.168.1.{i}" for i in range(100, 110)],  # Office network
            'usual_devices': ['Mozilla/5.0 (Windows NT 10.0; Win64; x64)', 'Mozilla/5.0 (Macintosh; Intel Mac OS X)'],
            'travel_frequency': np.random.uniform(0.1, 0.3),  # How often they work remotely
            'device_change_frequency': np.random.uniform(0.1, 0.3),  # How often they use different devices
            'risk_tolerance': np.random.uniform(0.7, 0.9),  # How much risk is acceptable for this user
            'login_frequency': np.random.uniform(5, 20),  # Average logins per day
            'weekend_activity': np.random.uniform(0.1, 0.5),  # Probability of weekend activity
            'holiday_activity': np.random.uniform(0.05, 0.3),  # Probability of holiday activity
            'multi_factor_auth': np.random.choice([True, False], p=[0.8, 0.2]),  # Whether user uses MFA
            'security_training': np.random.uniform(0.5, 1.0)  # User's security awareness level
        } for user_id in user_ids
    }
    
    # Define base scenarios with more complex patterns
    base_scenarios = [
        {
            'name': 'normal_access',
            'weight': 0.5,
            'factors': {
                'time': (0.1, 0.4),      # Normal hours
                'location': (0.1, 0.5),   # Known locations
                'history': (0.1, 0.4),    # Good history
                'device': (0.1, 0.5),     # Known devices
                'motion': (0.2, 0.6)      # Variable motion
            },
            'risk_correlation': 0.3,      # Low correlation with risk
            'seasonal_factor': 0.1,       # Slight seasonal variation
            'weekly_pattern': True,       # Follow weekly patterns
            'user_specific': True         # Varies by user profile
        },
        {
            'name': 'remote_work',
            'weight': 0.2,
            'factors': {
                'time': (0.2, 0.5),      # Flexible hours
                'location': (0.3, 0.7),   # Various locations
                'history': (0.1, 0.4),    # Good history
                'device': (0.2, 0.6),     # Multiple devices
                'motion': (0.2, 0.5)      # Normal motion
            },
            'risk_correlation': 0.4,      # Medium correlation with risk
            'seasonal_factor': 0.2,       # Moderate seasonal variation
            'weekly_pattern': True,       # Follow weekly patterns
            'user_specific': True         # Varies by user profile
        },
        {
            'name': 'suspicious_behavior',
            'weight': 0.15,
            'factors': {
                'time': (0.4, 0.8),      # Odd hours
                'location': (0.3, 0.7),   # Less known locations
                'history': (0.3, 0.7),    # Some failures
                'device': (0.3, 0.7),     # Less known devices
                'motion': (0.3, 0.7)      # Suspicious motion
            },
            'risk_correlation': 0.6,      # Medium correlation with risk
            'seasonal_factor': 0.3,       # Strong seasonal variation
            'weekly_pattern': False,      # No weekly pattern
            'user_specific': True         # Varies by user profile
        },
        {
            'name': 'attack_pattern',
            'weight': 0.15,
            'factors': {
                'time': (0.6, 0.9),      # Very odd hours
                'location': (0.7, 0.9),   # Unknown locations
                'history': (0.6, 0.9),    # Bad history
                'device': (0.7, 0.9),     # Unknown devices
                'motion': (0.6, 0.9)      # Very suspicious motion
            },
            'risk_correlation': 0.8,      # High correlation with risk
            'seasonal_factor': 0.4,       # Very strong seasonal variation
            'weekly_pattern': False,      # No weekly pattern
            'user_specific': False        # Same for all users
        }
    ]
    
    # Define advanced attack patterns
    attack_patterns = [
        {
            'name': 'brute_force',
            'weight': 0.3,
            'factors': {
                'time': (0.7, 0.95),     # Very odd hours
                'location': (0.8, 0.95),  # Unknown locations
                'history': (0.9, 0.95),   # Multiple failures
                'device': (0.7, 0.9),     # Unknown devices
                'motion': (0.6, 0.8)      # Suspicious motion
            },
            'burst_factor': 0.9,          # High burst probability
            'time_compression': 0.8,       # Compressed time between attempts
            'location_diversity': 0.7       # Diverse locations
        },
        {
            'name': 'credential_stuffing',
            'weight': 0.3,
            'factors': {
                'time': (0.6, 0.9),      # Odd hours
                'location': (0.7, 0.9),   # Various locations
                'history': (0.8, 0.95),   # Multiple failures
                'device': (0.8, 0.95),    # Various devices
                'motion': (0.7, 0.9)      # Suspicious motion
            },
            'burst_factor': 0.7,          # Medium burst probability
            'time_compression': 0.6,       # Moderate time compression
            'location_diversity': 0.9      # Very diverse locations
        },
        {
            'name': 'targeted_attack',
            'weight': 0.2,
            'factors': {
                'time': (0.5, 0.8),      # Odd hours
                'location': (0.6, 0.85),  # Semi-known locations
                'history': (0.7, 0.9),    # Some failures
                'device': (0.6, 0.85),    # Semi-known devices
                'motion': (0.5, 0.75)     # Somewhat suspicious motion
            },
            'burst_factor': 0.5,          # Low burst probability
            'time_compression': 0.4,       # Low time compression
            'location_diversity': 0.5      # Moderate location diversity
        },
        {
            'name': 'advanced_persistent_threat',
            'weight': 0.2,
            'factors': {
                'time': (0.4, 0.7),      # Semi-odd hours
                'location': (0.5, 0.8),   # Semi-known locations
                'history': (0.6, 0.85),   # Few failures
                'device': (0.5, 0.8),     # Semi-known devices
                'motion': (0.4, 0.7)      # Slightly suspicious motion
            },
            'burst_factor': 0.3,          # Very low burst probability
            'time_compression': 0.2,       # Very low time compression
            'location_diversity': 0.3      # Low location diversity
        }
    ]
    
    # Define holidays and special dates
    holidays = [
        (1, 1),   # New Year's Day
        (12, 25), # Christmas
        (7, 4),   # Independence Day
        (10, 31), # Halloween
        (11, 25), # Thanksgiving
    ]
    
    # Generate data
    data = []
    for i in tqdm(range(num_samples), desc="Generating synthetic data"):
        # Select a user and their profile
        user_id = np.random.choice(user_ids)
        profile = user_profiles[user_id]
        
        # Determine if this is an attack attempt
        is_attack = np.random.random() < 0.15  # 15% chance of attack
        
        if is_attack:
            # Select an attack pattern
            attack_pattern = np.random.choice(attack_patterns, p=[p['weight'] for p in attack_patterns])
            scenario = {
                'name': attack_pattern['name'],
                'factors': attack_pattern['factors'],
                'risk_correlation': 0.9,  # High correlation for attacks
                'seasonal_factor': 0.5,   # Strong seasonal variation
                'weekly_pattern': False,  # No weekly pattern
                'user_specific': False    # Same for all users
            }
            
            # Apply attack-specific modifications
            burst_probability = attack_pattern['burst_factor']
            time_compression = attack_pattern['time_compression']
            location_diversity = attack_pattern['location_diversity']
        else:
            # Select a normal scenario based on weights
            scenario = np.random.choice(base_scenarios, p=[s['weight'] for s in base_scenarios])
            burst_probability = 0.1
            time_compression = 0.1
            location_diversity = 0.1
        
        # Calculate time-based factors
        current_time = datetime.now() - timedelta(days=np.random.randint(0, 365))
        hour = current_time.hour / 24.0  # Normalize to [0,1]
        day_of_week = current_time.weekday() / 7.0  # Normalize to [0,1]
        day_of_year = current_time.timetuple().tm_yday / 365.0  # Normalize to [0,1]
        month = current_time.month
        day = current_time.day
        
        # Check if it's a holiday
        is_holiday = (month, day) in holidays
        
        # Check if it's a weekend
        is_weekend = day_of_week >= 0.7
        
        # Apply burst pattern for attacks
        if is_attack and np.random.random() < burst_probability:
            # Generate multiple samples in a short time period
            burst_samples = np.random.randint(3, 10)
            for _ in range(burst_samples):
                # Compress time between attempts
                hour_offset = np.random.uniform(0, 0.1 * time_compression)
                current_hour = (hour + hour_offset) % 1.0
                
                # Generate sample with compressed time
                sample = _generate_sample(
                    user_id, profile, scenario, current_time, current_hour, 
                    day_of_week, day_of_year, is_holiday, is_weekend, 
                    noise_level, concept_drift, i, num_samples, location_diversity,
                    attack_patterns  # Pass attack_patterns to _generate_sample
                )
                data.append(sample)
            
            # Skip the rest of the loop for burst samples
            continue
        
        # Generate a single sample
        sample = _generate_sample(
            user_id, profile, scenario, current_time, hour, 
            day_of_week, day_of_year, is_holiday, is_weekend, 
            noise_level, concept_drift, i, num_samples, location_diversity,
            attack_patterns  # Pass attack_patterns to _generate_sample
        )
        data.append(sample)
    
    # Convert to DataFrame
    df = pd.DataFrame(data)
    
    conn.close()
    return df

def _generate_sample(user_id, profile, scenario, current_time, hour, day_of_week, day_of_year, 
                    is_holiday, is_weekend, noise_level, concept_drift, sample_index, num_samples, 
                    location_diversity, attack_patterns):
    """Helper function to generate a single sample."""
    # Generate correlated factors with user profile influence
    base_random = np.random.random()  # Base randomness for correlation
    factors = {}
    
    for factor, (min_val, max_val) in scenario['factors'].items():
        # Mix base randomness with independent randomness based on correlation
        corr = scenario['risk_correlation']
        independent_random = np.random.random()
        factor_value = corr * base_random + (1 - corr) * independent_random
        
        # Apply user profile influence if scenario is user-specific
        if scenario.get('user_specific', True):
            if factor == 'time':
                # Consider user's usual hours
                usual_start, usual_end = profile['usual_hours']
                
                # Adjust for holidays and weekends
                if is_holiday and np.random.random() < profile['holiday_activity']:
                    # More random hours on holidays
                    factor_value = factor_value * 0.9 + np.random.random() * 0.1
                elif is_weekend and np.random.random() < profile['weekend_activity']:
                    # Different hours on weekends
                    factor_value = factor_value * 0.8 + np.random.uniform(0.1, 0.4) * 0.2
                elif scenario['weekly_pattern']:
                    # Higher probability of access during usual hours on weekdays
                    if day_of_week < 0.7:  # Weekday
                        factor_value = factor_value * 0.7 + usual_start * 0.3
                    else:  # Weekend
                        factor_value = factor_value * 0.9 + usual_start * 0.1
                else:
                    factor_value = factor_value * 0.8 + usual_start * 0.2
                    
            elif factor == 'location':
                # Consider user's travel frequency
                if np.random.random() < profile['travel_frequency']:
                    # More likely to be remote
                    factor_value = factor_value * 0.8 + 0.2
                
                # Apply location diversity for attacks
                if location_diversity > 0.1:
                    factor_value = factor_value * (1 - location_diversity) + np.random.random() * location_diversity
                    
            elif factor == 'device':
                # Consider user's device change frequency
                if np.random.random() < profile['device_change_frequency']:
                    # More likely to use different device
                    factor_value = factor_value * 0.8 + 0.2
                
                # Consider user's security training
                if profile['security_training'] > 0.8:
                    # More security-aware users are less likely to use unknown devices
                    factor_value = factor_value * 0.7 + 0.3 * (1 - profile['security_training'])
        
        # Add seasonal variation
        seasonal_variation = scenario['seasonal_factor'] * np.sin(2 * np.pi * day_of_year)
        factor_value = factor_value + seasonal_variation
        
        # Scale to range and add noise
        factor_value = min_val + factor_value * (max_val - min_val)
        noise = np.random.normal(0, noise_level)
        factor_value = np.clip(factor_value + noise, 0, 1)
        
        factors[factor] = factor_value
    
    # Calculate true risk score using a complex function
    # that doesn't directly match either approach
    base_risk = (
        0.25 * np.sin(np.pi * factors['time']) +
        0.2 * np.exp(factors['location']) / np.e +
        0.3 * np.power(factors['history'], 1.5) +
        0.15 * np.sqrt(factors['device']) +
        0.1 * (1 - np.cos(np.pi * factors['motion']))
    )
    
    # Add concept drift if enabled
    if concept_drift:
        # Add both short-term and long-term drift
        short_term_drift = 0.1 * np.sin(2 * np.pi * sample_index / (num_samples/10))  # 10 cycles
        long_term_drift = 0.05 * np.sin(2 * np.pi * sample_index / num_samples)  # 1 cycle
        
        # Add user-specific drift
        user_drift = 0.03 * np.sin(2 * np.pi * user_id / len(profile))
        
        # Add seasonal drift
        seasonal_drift = 0.07 * np.sin(2 * np.pi * day_of_year)
        
        base_risk = np.clip(base_risk + short_term_drift + long_term_drift + user_drift + seasonal_drift, 0, 1)
    
    # Add metadata
    sample = {
        'user_id': user_id,
        'timestamp': current_time.isoformat(),
        'scenario': scenario['name'],
        'true_risk_score': base_risk,
        'hour_of_day': hour,
        'day_of_week': day_of_week,
        'day_of_year': day_of_year,
        'is_holiday': is_holiday,
        'is_weekend': is_weekend,
        'is_attack': scenario['name'] in [p['name'] for p in attack_patterns],
        'attack_type': scenario['name'] if scenario['name'] in [p['name'] for p in attack_patterns] else 'none',
        **factors
    }
    
    return sample

def train_bandit(data_df, bandit, learning_rate=0.001, exploration_rate=0.3, regularization=0.1, min_weight=0.1):
    """
    Train the bandit on the synthetic data with improved methodology.
    
    Args:
        data_df: DataFrame containing synthetic data
        bandit: RiskAssessmentBandit instance
        learning_rate: Learning rate for the bandit
        exploration_rate: Initial exploration rate
        regularization: Regularization strength to prevent extreme weights
        min_weight: Minimum allowed weight for any factor
        
    Returns:
        dict: Training metrics
    """
    metrics = {
        'accuracy': [],
        'avg_reward': [],
        'exploration_rate': [],
        'weight_entropy': []  # Measure of weight distribution diversity
    }
    
    # Process each sample
    for _, row in tqdm(data_df.iterrows(), total=len(data_df), desc="Training bandit"):
        # Create factors dictionary
        factors = {
            'time': row['time'],
            'location': row['location'],
            'history': row['history'],
            'device': row['device'],
            'motion': row['motion']
        }
        
        # Get context features
        context_features = bandit.get_context_features(factors)
        
        # Select action
        action = bandit.select_action(context_features)
        
        # Calculate base reward based on risk prediction
        if row['true_risk_score'] < RISK_THRESHOLDS['low']:
            base_reward = 1.0 if action == 'low' else -0.5
        elif row['true_risk_score'] < RISK_THRESHOLDS['medium']:
            base_reward = 1.0 if action == 'medium' else -0.5
        else:  # high
            base_reward = 1.0 if action == 'high' else -0.5
        
        # Add regularization term to encourage balanced weights
        current_weights = bandit.get_weights()
        weight_values = np.array(list(current_weights.values()))
        
        # Calculate entropy-based regularization
        entropy = -np.sum(weight_values * np.log(weight_values + 1e-10))
        entropy_reward = regularization * entropy
        
        # Add diversity reward based on weight distribution
        ideal_weight = 1.0 / len(current_weights)  # Equal distribution
        weight_deviation = np.mean(np.abs(weight_values - ideal_weight))
        diversity_reward = regularization * (1.0 - weight_deviation)
        
        # Combined reward with stronger emphasis on diversity
        reward = base_reward + entropy_reward + 2.0 * diversity_reward
        
        # Update bandit
        bandit.update(context_features, action, reward)
        
        # Update metrics
        current_metrics = bandit.get_performance_metrics(window_size=100)
        metrics['accuracy'].append(current_metrics['accuracy'])
        metrics['avg_reward'].append(current_metrics['avg_reward'])
        metrics['exploration_rate'].append(current_metrics['exploration_rate'])
        
        # Calculate weight entropy
        updated_weights = bandit.get_weights()
        weight_values = np.array(list(updated_weights.values()))
        entropy = -np.sum(weight_values * np.log(weight_values + 1e-10))
        metrics['weight_entropy'].append(entropy)
    
    return metrics

def evaluate_bandit(data_df, bandit, threshold=0.5):
    """
    Evaluate the bandit's performance with enhanced metrics.
    
    Args:
        data_df: DataFrame containing test data
        bandit: Trained bandit model
        threshold: Risk score threshold for classification
        
    Returns:
        dict: Evaluation metrics
    """
    # Get predictions
    predictions = []
    true_scores = []
    attack_types = []
    is_attacks = []
    timestamps = []
    user_ids = []
    
    for _, row in data_df.iterrows():
        # Get context features
        context = bandit.get_context_features({
            'time': row['time'],
            'location': row['location'],
            'history': row['history'],
            'device': row['device'],
            'motion': row['motion']
        })
        
        # Get prediction
        risk_score = bandit._calculate_risk_score(context)
        predictions.append(risk_score)
        true_scores.append(row['true_risk_score'])
        attack_types.append(row['attack_type'])
        is_attacks.append(row['is_attack'])
        timestamps.append(row['timestamp'])
        user_ids.append(row['user_id'])
    
    predictions = np.array(predictions)
    true_scores = np.array(true_scores)
    attack_types = np.array(attack_types)
    is_attacks = np.array(is_attacks)
    timestamps = np.array(timestamps)
    user_ids = np.array(user_ids)
    
    # Convert to binary predictions
    pred_labels = (predictions >= threshold).astype(int)
    true_labels = (true_scores >= threshold).astype(int)
    
    # Basic metrics
    tn, fp, fn, tp = confusion_matrix(true_labels, pred_labels).ravel()
    fpr = fp / (fp + tn)
    fnr = fn / (fn + tp)
    accuracy = (tp + tn) / len(predictions)
    
    # Calculate ROC curve and AUC
    fpr_curve, tpr_curve, thresholds = roc_curve(true_labels, predictions)
    roc_auc = auc(fpr_curve, tpr_curve)
    
    # Calculate precision-recall curve
    precision, recall, pr_thresholds = precision_recall_curve(true_labels, predictions)
    avg_precision = average_precision_score(true_labels, predictions)
    
    # Calculate F1 score
    f1 = f1_score(true_labels, pred_labels)
    
    # Calculate attack-specific metrics
    attack_metrics = {}
    for attack_type in np.unique(attack_types):
        if attack_type == 'none':
            continue
            
        attack_mask = attack_types == attack_type
        if np.sum(attack_mask) == 0:
            continue
            
        attack_tp = np.sum((pred_labels == 1) & (true_labels == 1) & attack_mask)
        attack_fp = np.sum((pred_labels == 1) & (true_labels == 0) & attack_mask)
        attack_fn = np.sum((pred_labels == 0) & (true_labels == 1) & attack_mask)
        attack_tn = np.sum((pred_labels == 0) & (true_labels == 0) & attack_mask)
        
        attack_metrics[attack_type] = {
            'true_positives': int(attack_tp),
            'false_positives': int(attack_fp),
            'false_negatives': int(attack_fn),
            'true_negatives': int(attack_tn),
            'precision': attack_tp / (attack_tp + attack_fp) if (attack_tp + attack_fp) > 0 else 0,
            'recall': attack_tp / (attack_tp + attack_fn) if (attack_tp + attack_fn) > 0 else 0,
            'f1_score': 2 * attack_tp / (2 * attack_tp + attack_fp + attack_fn) if (2 * attack_tp + attack_fp + attack_fn) > 0 else 0
        }
    
    # Calculate temporal metrics
    timestamps = pd.to_datetime(timestamps)
    timestamps_series = pd.Series(timestamps)
    hourly_metrics = {}
    for hour in range(24):
        hour_mask = timestamps_series.dt.hour == hour
        if np.sum(hour_mask) == 0:
            continue
            
        hour_tp = np.sum((pred_labels == 1) & (true_labels == 1) & hour_mask)
        hour_fp = np.sum((pred_labels == 1) & (true_labels == 0) & hour_mask)
        hour_fn = np.sum((pred_labels == 0) & (true_labels == 1) & hour_mask)
        hour_tn = np.sum((pred_labels == 0) & (true_labels == 0) & hour_mask)
        
        hourly_metrics[hour] = {
            'true_positives': int(hour_tp),
            'false_positives': int(hour_fp),
            'false_negatives': int(hour_fn),
            'true_negatives': int(hour_tn),
            'accuracy': (hour_tp + hour_tn) / np.sum(hour_mask) if np.sum(hour_mask) > 0 else 0
        }
    
    # Calculate user-specific metrics
    user_metrics = {}
    for user_id in np.unique(user_ids):
        user_mask = user_ids == user_id
        if np.sum(user_mask) == 0:
            continue
            
        user_tp = np.sum((pred_labels == 1) & (true_labels == 1) & user_mask)
        user_fp = np.sum((pred_labels == 1) & (true_labels == 0) & user_mask)
        user_fn = np.sum((pred_labels == 0) & (true_labels == 1) & user_mask)
        user_tn = np.sum((pred_labels == 0) & (true_labels == 0) & user_mask)
        
        user_metrics[user_id] = {
            'true_positives': int(user_tp),
            'false_positives': int(user_fp),
            'false_negatives': int(user_fn),
            'true_negatives': int(user_tn),
            'accuracy': (user_tp + user_tn) / np.sum(user_mask) if np.sum(user_mask) > 0 else 0
        }
    
    # Calculate burst detection metrics
    burst_metrics = {}
    for window_size in [5, 10, 15, 30]:  # minutes
        # Group predictions by time windows
        time_windows = pd.DataFrame({
            'timestamp': timestamps,
            'prediction': pred_labels,
            'true_label': true_labels,
            'is_attack': is_attacks
        })
        time_windows['window'] = time_windows['timestamp'].dt.floor(f'{window_size}min')
        
        # Calculate burst metrics for each window
        window_metrics = []
        for window, group in time_windows.groupby('window'):
            if len(group) < 2:  # Skip single-sample windows
                continue
                
            window_metrics.append({
                'window_start': window,
                'samples': len(group),
                'attacks': np.sum(group['is_attack']),
                'detected_attacks': np.sum((group['prediction'] == 1) & (group['is_attack'])),
                'false_alarms': np.sum((group['prediction'] == 1) & (~group['is_attack']))
            })
        
        if window_metrics:
            window_metrics = pd.DataFrame(window_metrics)
            burst_metrics[f'{window_size}min'] = {
                'total_windows': len(window_metrics),
                'attack_windows': np.sum(window_metrics['attacks'] > 0),
                'detected_attack_windows': np.sum((window_metrics['attacks'] > 0) & (window_metrics['detected_attacks'] > 0)),
                'false_alarm_windows': np.sum(window_metrics['false_alarms'] > 0),
                'avg_samples_per_window': window_metrics['samples'].mean(),
                'detection_rate': np.sum((window_metrics['attacks'] > 0) & (window_metrics['detected_attacks'] > 0)) / np.sum(window_metrics['attacks'] > 0) if np.sum(window_metrics['attacks'] > 0) > 0 else 0,
                'false_alarm_rate': np.sum(window_metrics['false_alarms'] > 0) / np.sum(window_metrics['attacks'] == 0) if np.sum(window_metrics['attacks'] == 0) > 0 else 0
            }
    
    # Calculate weight stability metrics
    weight_history = bandit.get_weight_history()
    if weight_history:
        weight_stability = {
            'time': np.std([w['time'] for w in weight_history]),
            'location': np.std([w['location'] for w in weight_history]),
            'history': np.std([w['history'] for w in weight_history]),
            'device': np.std([w['device'] for w in weight_history]),
            'motion': np.std([w['motion'] for w in weight_history])
        }
    else:
        weight_stability = None
    
    return {
        'total_samples': len(predictions),
        'legitimate_samples': np.sum(~is_attacks),
        'attack_samples': np.sum(is_attacks),
        'true_positives': int(tp),
        'false_positives': int(fp),
        'true_negatives': int(tn),
        'false_negatives': int(fn),
        'false_positive_rate': float(fpr),
        'false_negative_rate': float(fnr),
        'accuracy': float(accuracy),
        'auc': float(roc_auc),
        'avg_precision': float(avg_precision),
        'f1_score': float(f1),
        'attack_metrics': attack_metrics,
        'hourly_metrics': hourly_metrics,
        'user_metrics': user_metrics,
        'burst_metrics': burst_metrics,
        'weight_stability': weight_stability,
        'roc_curve': {
            'fpr': fpr_curve.tolist(),
            'tpr': tpr_curve.tolist(),
            'thresholds': thresholds.tolist()
        },
        'pr_curve': {
            'precision': precision.tolist(),
            'recall': recall.tolist(),
            'thresholds': pr_thresholds.tolist()
        },
        'confusion_matrix': confusion_matrix(true_labels, pred_labels)
    }

def evaluate_fixed_weights(data_df, weights):
    """
    Evaluate the fixed-weight approach on the synthetic data with improved metrics.
    
    Args:
        data_df: DataFrame containing synthetic data
        weights: Dictionary of fixed weights
        
    Returns:
        dict: Evaluation metrics
    """
    results = {
        'true_positives': 0,
        'false_positives': 0,
        'true_negatives': 0,
        'false_negatives': 0,
        'legitimate_samples': 0,
        'attack_samples': 0,
        'predictions': [],
        'true_scores': []
    }
    
    # Process each sample
    for _, row in data_df.iterrows():
        # Create factors dictionary
        factors = {
            'time': row['time'],
            'location': row['location'],
            'history': row['history'],
            'device': row['device'],
            'motion': row['motion']
        }
        
        # Calculate risk score using fixed weights
        risk_score = calculate_risk_score_with_weights(factors, weights)
        
        # Determine risk level based on thresholds
        if risk_score < RISK_THRESHOLDS['low']:
            action = 'low'
        elif risk_score < RISK_THRESHOLDS['medium']:
            action = 'medium'
        else:
            action = 'high'
        
        # Store prediction and true score for ROC curve
        results['predictions'].append(risk_score)
        results['true_scores'].append(row['true_risk_score'])
        
        # Update results
        if row['true_risk_score'] >= RISK_THRESHOLDS['high']:
            results['attack_samples'] += 1
            if action == 'high':
                results['true_positives'] += 1
            else:
                results['false_negatives'] += 1
        else:
            results['legitimate_samples'] += 1
            if action == 'high':
                results['false_positives'] += 1
            else:
                results['true_negatives'] += 1
    
    # Calculate metrics
    total_samples = len(data_df)
    fpr = results['false_positives'] / results['legitimate_samples'] if results['legitimate_samples'] > 0 else 0
    fnr = results['false_negatives'] / results['attack_samples'] if results['attack_samples'] > 0 else 0
    accuracy = (results['true_positives'] + results['true_negatives']) / total_samples
    
    # Calculate ROC curve and AUC
    fpr_curve, tpr_curve, _ = roc_curve(
        [1 if score >= RISK_THRESHOLDS['high'] else 0 for score in results['true_scores']],
        results['predictions']
    )
    auc_score = auc(fpr_curve, tpr_curve)
    
    # Calculate Precision-Recall curve
    precision_curve, recall_curve, _ = precision_recall_curve(
        [1 if score >= RISK_THRESHOLDS['high'] else 0 for score in results['true_scores']],
        results['predictions']
    )
    
    # Calculate F1 score
    predictions_binary = [1 if score >= RISK_THRESHOLDS['high'] else 0 for score in results['predictions']]
    true_binary = [1 if score >= RISK_THRESHOLDS['high'] else 0 for score in results['true_scores']]
    f1 = f1_score(true_binary, predictions_binary)
    
    # Calculate confusion matrix
    cm = confusion_matrix(true_binary, predictions_binary)
    
    return {
        'total_samples': total_samples,
        'legitimate_samples': results['legitimate_samples'],
        'attack_samples': results['attack_samples'],
        'true_positives': results['true_positives'],
        'false_positives': results['false_positives'],
        'true_negatives': results['true_negatives'],
        'false_negatives': results['false_negatives'],
        'false_positive_rate': fpr,
        'false_negative_rate': fnr,
        'accuracy': accuracy,
        'auc': auc_score,
        'f1_score': f1,
        'roc_curve': {
            'fpr': fpr_curve.tolist(),
            'tpr': tpr_curve.tolist()
        },
        'pr_curve': {
            'precision': precision_curve.tolist(),
            'recall': recall_curve.tolist()
        },
        'confusion_matrix': cm
    }

def calculate_risk_score_with_weights(factors, weights):
    """
    Calculate risk score using fixed weights.
    
    Args:
        factors: Dictionary of risk factors
        weights: Dictionary of weights
        
    Returns:
        float: Calculated risk score
    """
    # Calculate weighted sum
    weighted_sum = sum(weights[key] * factors.get(key, 0) for key in weights)
    
    # Add epsilon and ensure risk is between 0 and 1
    total_risk = min(1.0, max(0.0, weighted_sum + 0.01))
    
    return total_risk

def plot_comparison(bandit_metrics, fixed_metrics, save_path='comparison.png'):
    """
    Plot comparison between bandit and fixed-weight approaches.
    
    Args:
        bandit_metrics: Dictionary containing bandit metrics
        fixed_metrics: Dictionary containing fixed-weight metrics
        save_path: Path to save the plot
    """
    plt.figure(figsize=(15, 10))
    
    # Plot accuracy
    plt.subplot(3, 1, 1)
    plt.plot(bandit_metrics['accuracy'], label='Bandit Accuracy')
    plt.axhline(y=fixed_metrics['accuracy'], color='r', linestyle='--', label='Fixed Weight Accuracy')
    plt.title('Comparison: Bandit vs Fixed Weight')
    plt.ylabel('Accuracy')
    plt.legend()
    
    # Plot average reward
    plt.subplot(3, 1, 2)
    plt.plot(bandit_metrics['avg_reward'], label='Bandit Average Reward')
    plt.axhline(y=fixed_metrics['avg_reward'], color='r', linestyle='--', label='Fixed Weight Average Reward')
    plt.ylabel('Average Reward')
    plt.legend()
    
    # Plot weight entropy (diversity)
    plt.subplot(3, 1, 3)
    plt.plot(bandit_metrics['weight_entropy'], label='Bandit Weight Entropy')
    plt.axhline(y=0, color='r', linestyle='--', label='Fixed Weight (No Adaptation)')
    plt.xlabel('Iterations')
    plt.ylabel('Weight Entropy (Higher = More Balanced)')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def plot_roc_curves(bandit_eval, fixed_eval, save_path='roc_curves.png'):
    """
    Plot ROC curves for both approaches.
    
    Args:
        bandit_eval: Bandit evaluation results
        fixed_eval: Fixed-weight evaluation results
        save_path: Path to save the plot
    """
    plt.figure(figsize=(10, 8))
    
    # Plot ROC curves
    fpr_bandit = bandit_eval['roc_curve']['fpr']
    tpr_bandit = bandit_eval['roc_curve']['tpr']
    fpr_fixed = fixed_eval['roc_curve']['fpr']
    tpr_fixed = fixed_eval['roc_curve']['tpr']
    
    plt.plot(fpr_bandit, tpr_bandit, label=f'Bandit (AUC = {bandit_eval["auc"]:.3f})')
    plt.plot(fpr_fixed, tpr_fixed, label=f'Fixed Weight (AUC = {fixed_eval["auc"]:.3f})')
    
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curves: Bandit vs Fixed Weight')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def plot_pr_curves(bandit_eval, fixed_eval, save_path='pr_curves.png'):
    """
    Plot precision-recall curves for both approaches.
    
    Args:
        bandit_eval: Bandit evaluation results
        fixed_eval: Fixed-weight evaluation results
        save_path: Path to save the plot
    """
    plt.figure(figsize=(10, 8))
    
    # Plot PR curves
    precision_bandit = bandit_eval['pr_curve']['precision']
    recall_bandit = bandit_eval['pr_curve']['recall']
    precision_fixed = fixed_eval['pr_curve']['precision']
    recall_fixed = fixed_eval['pr_curve']['recall']
    
    plt.plot(recall_bandit, precision_bandit, label=f'Bandit (F1 = {bandit_eval["f1_score"]:.3f})')
    plt.plot(recall_fixed, precision_fixed, label=f'Fixed Weight (F1 = {fixed_eval["f1_score"]:.3f})')
    
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curves: Bandit vs Fixed Weight')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def plot_confusion_matrices(bandit_eval, fixed_eval, save_path='confusion_matrices.png'):
    """
    Plot confusion matrices for both approaches.
    
    Args:
        bandit_eval: Bandit evaluation results
        fixed_eval: Fixed-weight evaluation results
        save_path: Path to save the plot
    """
    plt.figure(figsize=(15, 6))
    
    # Plot bandit confusion matrix
    plt.subplot(1, 2, 1)
    sns.heatmap(bandit_eval['confusion_matrix'], annot=True, fmt='d', cmap='Blues')
    plt.title('Bandit Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    
    # Plot fixed weight confusion matrix
    plt.subplot(1, 2, 2)
    sns.heatmap(fixed_eval['confusion_matrix'], annot=True, fmt='d', cmap='Blues')
    plt.title('Fixed Weight Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def plot_weights_comparison(bandit, fixed_weights, save_path='weights_comparison.png'):
    """
    Plot comparison of bandit weights vs fixed weights.
    
    Args:
        bandit: RiskAssessmentBandit instance
        fixed_weights: Dictionary of fixed weights
        save_path: Path to save the plot
    """
    # Get final bandit weights
    bandit_weights = bandit.get_weights()
    
    # Create DataFrame for comparison
    comparison = pd.DataFrame({
        'Factor': list(bandit_weights.keys()),
        'Bandit Weights': list(bandit_weights.values()),
        'Fixed Weights': [fixed_weights[factor] for factor in bandit_weights.keys()]
    })
    
    # Plot comparison
    plt.figure(figsize=(10, 6))
    x = np.arange(len(comparison))
    width = 0.35
    
    plt.bar(x - width/2, comparison['Bandit Weights'], width, label='Bandit Weights')
    plt.bar(x + width/2, comparison['Fixed Weights'], width, label='Fixed Weights')
    
    plt.xlabel('Risk Factors')
    plt.ylabel('Weight Value')
    plt.title('Comparison: Bandit vs Fixed Weights')
    plt.xticks(x, comparison['Factor'])
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def cross_validate(data_df, n_splits=5, learning_rate=0.001, exploration_rate=0.3, regularization=0.1, min_weight=0.05):
    """
    Perform k-fold cross-validation to evaluate both approaches.
    
    Args:
        data_df: DataFrame containing synthetic data
        n_splits: Number of folds for cross-validation
        learning_rate: Learning rate for the bandit
        exploration_rate: Initial exploration rate
        regularization: Regularization strength
        min_weight: Minimum allowed weight for any factor
        
    Returns:
        dict: Cross-validation results
    """
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    
    bandit_results = []
    fixed_results = []
    
    for fold, (train_idx, test_idx) in enumerate(kf.split(data_df)):
        print(f"\nFold {fold + 1}/{n_splits}")
        
        # Split data
        train_data = data_df.iloc[train_idx]
        test_data = data_df.iloc[test_idx]
        
        # Initialize bandit
        bandit = RiskAssessmentBandit(
            learning_rate=learning_rate,
            exploration_rate=exploration_rate
        )
        
        # Train bandit
        train_bandit(train_data, bandit, learning_rate, exploration_rate, regularization, min_weight)
        
        # Evaluate both approaches
        bandit_eval = evaluate_bandit(test_data, bandit)
        fixed_eval = evaluate_fixed_weights(test_data, FIXED_WEIGHTS)
        
        # Store results
        bandit_results.append(bandit_eval['metrics'])
        fixed_results.append(fixed_eval['metrics'])
        
        print(f"Bandit Accuracy: {bandit_eval['metrics']['accuracy']:.4f}, AUC: {bandit_eval['metrics']['auc']:.4f}")
        print(f"Fixed Weight Accuracy: {fixed_eval['metrics']['accuracy']:.4f}, AUC: {fixed_eval['metrics']['auc']:.4f}")
    
    # Calculate average results
    avg_bandit = {
        'accuracy': np.mean([r['accuracy'] for r in bandit_results]),
        'auc': np.mean([r['auc'] for r in bandit_results]),
        'f1': np.mean([r['f1'] for r in bandit_results]),
        'fpr': np.mean([r['fpr'] for r in bandit_results]),
        'fnr': np.mean([r['fnr'] for r in bandit_results])
    }
    
    avg_fixed = {
        'accuracy': np.mean([r['accuracy'] for r in fixed_results]),
        'auc': np.mean([r['auc'] for r in fixed_results]),
        'f1': np.mean([r['f1'] for r in fixed_results]),
        'fpr': np.mean([r['fpr'] for r in fixed_results]),
        'fnr': np.mean([r['fnr'] for r in fixed_results])
    }
    
    return {
        'bandit': avg_bandit,
        'fixed': avg_fixed,
        'fold_results': {
            'bandit': bandit_results,
            'fixed': fixed_results
        }
    }

def main():
    """Main function."""
    parser = argparse.ArgumentParser(description='Compare Bandit vs Fixed Weight Risk Assessment')
    parser.add_argument('--runs', type=int, default=10, help='Number of training runs')
    parser.add_argument('--samples', type=int, default=10000, help='Number of samples per run')
    parser.add_argument('--learning-rate', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--exploration-rate', type=float, default=0.3, help='Initial exploration rate')
    parser.add_argument('--regularization', type=float, default=0.1, help='Regularization strength')
    parser.add_argument('--min-weight', type=float, default=0.1, help='Minimum allowed weight for any factor')
    parser.add_argument('--noise-level', type=float, default=0.1, help='Noise level in synthetic data')
    parser.add_argument('--concept-drift', action='store_true', help='Enable concept drift simulation')
    parser.add_argument('--cross-validate', action='store_true', help='Perform cross-validation')
    parser.add_argument('--n-folds', type=int, default=5, help='Number of folds for cross-validation')
    parser.add_argument('--weights-file', type=str, default='instance/bandit_weights.json', help='Path to save weights')
    parser.add_argument('--fixed-weights', type=str, default=None, help='Path to fixed weights file')
    args = parser.parse_args()
    
    # Load fixed weights if provided
    fixed_weights = FIXED_WEIGHTS
    if args.fixed_weights and os.path.exists(args.fixed_weights):
        try:
            with open(args.fixed_weights, 'r') as f:
                fixed_weights = json.load(f)
                print(f"Loaded fixed weights from {args.fixed_weights}")
        except (json.JSONDecodeError, FileNotFoundError):
            print(f"Could not load fixed weights from {args.fixed_weights}. Using default weights.")
    
    # Generate synthetic data
    print("Generating synthetic data...")
    data_df = generate_synthetic_data(
        num_samples=args.samples,
        noise_level=args.noise_level,
        concept_drift=args.concept_drift
    )
    
    if args.cross_validate:
        # Perform cross-validation
        print(f"Performing {args.n_folds}-fold cross-validation...")
        cv_results = cross_validate(
            data_df,
            n_splits=args.n_folds,
            learning_rate=args.learning_rate,
            exploration_rate=args.exploration_rate,
            regularization=args.regularization,
            min_weight=args.min_weight
        )
        
        # Print cross-validation results
        print("\nCross-Validation Results:")
        print(f"Bandit - Accuracy: {cv_results['bandit']['accuracy']:.4f}, AUC: {cv_results['bandit']['auc']:.4f}, F1: {cv_results['bandit']['f1']:.4f}")
        print(f"Fixed Weight - Accuracy: {cv_results['fixed']['accuracy']:.4f}, AUC: {cv_results['fixed']['auc']:.4f}, F1: {cv_results['fixed']['f1']:.4f}")
        print(f"Improvement - Accuracy: {(cv_results['bandit']['accuracy'] - cv_results['fixed']['accuracy']) / cv_results['fixed']['accuracy'] * 100:.2f}%, AUC: {(cv_results['bandit']['auc'] - cv_results['fixed']['auc']) / cv_results['fixed']['auc'] * 100:.2f}%")
        
        # Save cross-validation results
        with open('cv_results.json', 'w') as f:
            json.dump(cv_results, f, indent=2)
        
        return
    
    # Initialize bandit
    bandit = RiskAssessmentBandit(
        learning_rate=args.learning_rate,
        exploration_rate=args.exploration_rate,
        weights_file=args.weights_file
    )
    
    # Training loop
    all_metrics = {
        'accuracy': [],
        'avg_reward': [],
        'exploration_rate': [],
        'weight_entropy': []
    }
    
    # Fixed weight metrics
    fixed_metrics = {
        'accuracy': 0.0,
        'avg_reward': 0.0,
        'exploration_rate': 0.0
    }
    
    for run in range(args.runs):
        print(f"\nRun {run + 1}/{args.runs}")
        
        # Train bandit
        metrics = train_bandit(data_df, bandit, args.learning_rate, args.exploration_rate, args.regularization, args.min_weight)
        
        # Update overall metrics
        for key in all_metrics:
            if key in metrics:
                all_metrics[key].extend(metrics[key])
        
        # Evaluate bandit
        bandit_evaluation = evaluate_bandit(data_df, bandit)
        
        # Evaluate fixed weights
        fixed_evaluation = evaluate_fixed_weights(data_df, fixed_weights)
        
        # Update fixed metrics (average across runs)
        fixed_metrics['accuracy'] = (fixed_metrics['accuracy'] * run + fixed_evaluation['accuracy']) / (run + 1)
        fixed_metrics['avg_reward'] = (fixed_metrics['avg_reward'] * run + fixed_evaluation['accuracy']) / (run + 1)
        
        # Print evaluation results
        print("\nBandit Evaluation Results:")
        print(f"Total Samples: {len(data_df)}")
        print(f"Legitimate Samples: {bandit_evaluation['legitimate_samples']}")
        print(f"Attack Samples: {bandit_evaluation['attack_samples']}")
        print(f"True Positives: {bandit_evaluation['true_positives']}")
        print(f"False Positives: {bandit_evaluation['false_positives']}")
        print(f"True Negatives: {bandit_evaluation['true_negatives']}")
        print(f"False Negatives: {bandit_evaluation['false_negatives']}")
        print(f"False Positive Rate: {bandit_evaluation['false_positive_rate']:.4f}")
        print(f"False Negative Rate: {bandit_evaluation['false_negative_rate']:.4f}")
        print(f"Accuracy: {bandit_evaluation['accuracy']:.4f}")
        print(f"AUC: {bandit_evaluation['auc']:.4f}")
        print(f"F1 Score: {bandit_evaluation['f1_score']:.4f}")
        
        print("\nFixed Weight Evaluation Results:")
        print(f"Total Samples: {len(data_df)}")
        print(f"Legitimate Samples: {fixed_evaluation['legitimate_samples']}")
        print(f"Attack Samples: {fixed_evaluation['attack_samples']}")
        print(f"True Positives: {fixed_evaluation['true_positives']}")
        print(f"False Positives: {fixed_evaluation['false_positives']}")
        print(f"True Negatives: {fixed_evaluation['true_negatives']}")
        print(f"False Negatives: {fixed_evaluation['false_negatives']}")
        print(f"False Positive Rate: {fixed_evaluation['false_positive_rate']:.4f}")
        print(f"False Negative Rate: {fixed_evaluation['false_negative_rate']:.4f}")
        print(f"Accuracy: {fixed_evaluation['accuracy']:.4f}")
        print(f"AUC: {fixed_evaluation['auc']:.4f}")
        print(f"F1 Score: {fixed_evaluation['f1_score']:.4f}")
        
        # Save weights
        bandit.save_weights()
    
    # Plot results
    plot_comparison(all_metrics, fixed_metrics)
    plot_weights_comparison(bandit, fixed_weights)
    
    # Plot ROC and PR curves for the final evaluation
    plot_roc_curves(bandit_evaluation, fixed_evaluation)
    plot_pr_curves(bandit_evaluation, fixed_evaluation)
    plot_confusion_matrices(bandit_evaluation, fixed_evaluation)
    
    # Print final comparison
    print("\nFinal Comparison:")
    print(f"Bandit Accuracy: {bandit_evaluation['accuracy']:.4f}")
    print(f"Fixed Weight Accuracy: {fixed_metrics['accuracy']:.4f}")
    print(f"Improvement: {(bandit_evaluation['accuracy'] - fixed_metrics['accuracy']) / fixed_metrics['accuracy'] * 100:.2f}%")
    print(f"Bandit AUC: {bandit_evaluation['auc']:.4f}")
    print(f"Fixed Weight AUC: {fixed_evaluation['auc']:.4f}")
    print(f"AUC Improvement: {(bandit_evaluation['auc'] - fixed_evaluation['auc']) / fixed_evaluation['auc'] * 100:.2f}%")
    
    # Print final weights
    print("\nFinal Bandit Weights:")
    for factor, weight in bandit.get_weights().items():
        print(f"{factor}: {weight:.4f}")
    
    print("\nFixed Weights:")
    for factor, weight in fixed_weights.items():
        print(f"{factor}: {weight:.4f}")

if __name__ == '__main__':
    main() 