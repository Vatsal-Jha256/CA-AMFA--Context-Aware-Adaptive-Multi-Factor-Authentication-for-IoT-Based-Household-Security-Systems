from ContextualBandits.ContextualBandit import ContextualBandit
from collections import defaultdict
import datetime
import numpy as np 

class UCBBandit(ContextualBandit):
    """Upper Confidence Bound contextual bandit implementation"""
    
    def __init__(self, factor_names, confidence=1.0):
        super().__init__(factor_names)
        self.confidence = confidence
        self.q_values = {name: defaultdict(float) for name in factor_names}
        self.counts = {name: defaultdict(int) for name in factor_names}
        self.total_counts = defaultdict(int)
        
    def _context_to_key(self, context):
        """Convert context to a string key for lookup"""
        day_of_week = datetime.datetime.now().weekday()
        hour_bin = datetime.datetime.now().hour // 4
        return f"{day_of_week}_{hour_bin}"
        
    def select_weights(self, context):
        context_key = self._context_to_key(context)
        self.total_counts[context_key] += 1
        
        weights = {}
        for name in self.factor_names:
            # If never tried, use default weight with high uncertainty
            if self.counts[name][context_key] == 0:
                weights[name] = 1.0
                continue
                
            # UCB formula: Q(a) + c * sqrt(ln(t) / N(a))
            exploitation = self.q_values[name][context_key]
            exploration = self.confidence * np.sqrt(
                np.log(self.total_counts[context_key]) / self.counts[name][context_key]
            )
            
            # Bound the weight to reasonable range
            weights[name] = max(0.1, min(3.0, exploitation + exploration))
            
        return weights
        
    def update(self, context, selected_weights, reward):
        context_key = self._context_to_key(context)
        
        # Update values for each factor
        for name in self.factor_names:
            self.counts[name][context_key] += 1
            
            # Incremental average update
            old_q = self.q_values[name][context_key]
            self.q_values[name][context_key] += (reward - old_q) / self.counts[name][context_key]