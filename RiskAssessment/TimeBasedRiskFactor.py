import random 
import datetime 
from RiskAssessment.RiskFactor import RiskFactor
class TimeBasedRiskFactor(RiskFactor):
    """Risk factor based on time of day"""
    
    def __init__(self, name="time_of_day", weight=1.0, high_risk_hours=None):
        super().__init__(name, weight)
        # Default high risk hours (night time: 10PM-6AM)
        self.high_risk_hours = high_risk_hours or list(range(22, 24)) + list(range(0, 6))
    
    def calculate(self, context):
        current_hour = datetime.datetime.now().hour
        # Higher risk during unusual hours
        if current_hour in self.high_risk_hours:
            return random.uniform(0.7, 1.0)  # High risk during unusual hours
        elif current_hour in [7, 8, 17, 18, 19]:  # Common home entry/exit times
            return random.uniform(0.1, 0.3)  # Lower risk during normal hours
        else:
            return random.uniform(0.3, 0.6)  # Medium risk during other times