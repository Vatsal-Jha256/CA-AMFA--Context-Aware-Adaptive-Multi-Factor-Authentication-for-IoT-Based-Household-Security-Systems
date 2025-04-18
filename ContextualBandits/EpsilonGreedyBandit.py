import random 
import datetime
from collections import defaultdict
from ContextualBandits.ContextualBandit import ContextualBandit
class EpsilonGreedyBandit(ContextualBandit):
    """Epsilon-greedy contextual bandit implementation"""
    
    def __init__(self, factor_names, epsilon=0.1, learning_rate=0.05):
        super().__init__(factor_names)
        self.epsilon = epsilon  # Exploration rate
        self.learning_rate = learning_rate
        self.q_values = {name: defaultdict(float) for name in factor_names}
        self.counts = {name: defaultdict(int) for name in factor_names}
        
    def _context_to_key(self, context):
        """Convert context to a string key for lookup"""
        # Simplified context binning - in practice, you might want more sophisticated binning
        day_of_week = datetime.datetime.now().weekday()
        hour_bin = datetime.datetime.now().hour // 4  # 6 bins of 4 hours each
        return f"{day_of_week}_{hour_bin}"
        
    def select_weights(self, context):
        context_key = self._context_to_key(context)
        
        # With probability epsilon, explore (random weights)
        if random.random() < self.epsilon:
            return {name: random.uniform(0.5, 1.5) for name in self.factor_names}
        
        # Otherwise, use current best weights
        return {name: self.q_values[name][context_key] 
                if self.counts[name][context_key] > 0 
                else 1.0 
                for name in self.factor_names}
        
    def update(self, context, selected_weights, reward):
        context_key = self._context_to_key(context)
        
        # Update Q-values for each factor
        for name in self.factor_names:
            self.counts[name][context_key] += 1
            
            # Update rule: Q = Q + alpha * (R - Q)
            current_q = self.q_values[name][context_key]
            self.q_values[name][context_key] += self.learning_rate * (reward - current_q)
            
            # Ensure weights stay within reasonable bounds
            self.q_values[name][context_key] = max(0.1, min(3.0, self.q_values[name][context_key]))