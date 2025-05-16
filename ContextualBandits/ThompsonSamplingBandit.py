import numpy as np
import datetime
from collections import defaultdict
from ContextualBandits.ContextualBandit import ContextualBandit

# Define a function outside the class to use with defaultdict
def default_one():
    return 1.0

class ThompsonSamplingBandit(ContextualBandit):
    """Thompson Sampling contextual bandit implementation"""
    
    def __init__(self, factor_names):
        super().__init__(factor_names)
        # For each factor and context, track alpha and beta parameters of Beta distribution
        self.alpha = {name: defaultdict(default_one) for name in factor_names}
        self.beta = {name: defaultdict(default_one) for name in factor_names}
    
    def _context_to_key(self, context):
        """Convert context to a string key for lookup"""
        # Simplified context binning
        day_of_week = datetime.datetime.now().weekday()
        hour_bin = datetime.datetime.now().hour // 4  # 6 bins of 4 hours each
        return f"{day_of_week}_{hour_bin}"
    
    def select_weights(self, context):
        context_key = self._context_to_key(context)
        # Sample from Beta distribution for each factor
        weights = {}
        for name in self.factor_names:
            # Sample from Beta(alpha, beta) and scale to appropriate weight range
            sample = np.random.beta(self.alpha[name][context_key], self.beta[name][context_key])
            # Scale to range [0.1, 3.0]
            weights[name] = 0.1 + sample * 2.9
        return weights
    
    def update(self, context, selected_weights, reward):
        context_key = self._context_to_key(context)
        # Binary reward signal (0 or 1)
        success = 1 if reward > 0.5 else 0
        # Update for each factor
        for name in self.factor_names:
            if success:
                self.alpha[name][context_key] += 1
            else:
                self.beta[name][context_key] += 1
