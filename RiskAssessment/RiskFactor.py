
from abc import abstractmethod
import logging



class RiskFactor:
    """Base class for risk factors that contribute to the overall risk score"""
    
    def __init__(self, name, weight=1.0):
        self.name = name
        self.weight = weight
        
    @abstractmethod
    def calculate(self, context):
        """Calculate the risk factor value given the current context"""
        pass
    
    def get_normalized_value(self, context):
        """Get the normalized risk value (0.0-1.0)"""
        return self.calculate(context)
# Add this method to the RiskFactor base class
    def get_raw_value(self, context):
        """Get the raw (non-normalized) risk value"""
        return self.calculate(context)
