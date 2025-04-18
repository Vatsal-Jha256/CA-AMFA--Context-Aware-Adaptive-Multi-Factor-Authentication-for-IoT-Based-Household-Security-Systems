import time 
from RiskAssessment.RiskFactor import RiskFactor

class FailedAttemptsRiskFactor(RiskFactor):
    """Risk factor based on recent failed login attempts"""
    
    def __init__(self, name="failed_attempts", weight=1.5, window_seconds=3600):
        super().__init__(name, weight)
        self.failed_attempts = []
        self.window_seconds = window_seconds  # 1 hour window by default
        
    def record_failed_attempt(self):
        """Record a failed login attempt"""
        self.failed_attempts.append(time.time())
        # Clean up old attempts
        self.clean_old_attempts()
        
    def clean_old_attempts(self):
        """Remove attempts older than the window"""
        cutoff_time = time.time() - self.window_seconds
        self.failed_attempts = [t for t in self.failed_attempts if t > cutoff_time]
        
    def calculate(self, context):
        self.clean_old_attempts()
        recent_count = len(self.failed_attempts)
        
        # Normalize based on thresholds
        if recent_count == 0:
            return 0.0
        elif recent_count == 1:
            return 0.3
        elif recent_count == 2:
            return 0.6
        else:
            return min(1.0, 0.7 + (recent_count - 2) * 0.1)  # Increases with more failures