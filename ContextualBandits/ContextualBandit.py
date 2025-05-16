
from abc import ABC, abstractmethod
import pickle
import logging
import os 
logger = logging.getLogger(__name__)

class ContextualBandit(ABC):
    """Abstract base class for contextual bandit algorithms"""
    
    def __init__(self, factor_names):
        self.factor_names=factor_names
        self.weights = {name: 1.0 for name in factor_names}  # Initial equal weights
        
    @abstractmethod
    def select_weights(self, context):
        """Select weights for the current context"""
        pass
        
    @abstractmethod
    def update(self, context, selected_weights, reward):
        """Update the model based on feedback"""
        pass
        
    def save_model(self, filename):
        with open(filename, 'wb') as f:
            pickle.dump(self, f)
        # Verify successful write
        if os.path.getsize(filename) == 0:
            raise IOError(f"Failed to write model to {filename}")
    
    @classmethod
    def load_model(cls, filename):
        """Safe model loading with corruption handling"""
        if not os.path.exists(filename):
            return None
        try:
            with open(filename, 'rb') as f:
                return pickle.load(f)
        except (FileNotFoundError, EOFError, pickle.UnpicklingError) as e:
            print(f"Model load failed: {e}, creating new model")
            os.remove(filename)  # Delete corrupted file
            return None
