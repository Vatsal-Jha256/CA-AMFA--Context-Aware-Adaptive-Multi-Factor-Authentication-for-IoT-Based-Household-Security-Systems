import time 
from RiskAssessment.RiskFactor import RiskFactor

class MotionActivityRiskFactor(RiskFactor):
    """Risk factor based on motion detection activity"""
    
    def __init__(self, name="motion_activity", weight=0.8, window_seconds=300):
        super().__init__(name, weight)
        self.motion_events = []
        self.window_seconds = window_seconds  # 5 minutes window by default
        
    def record_motion(self):
        """Record a motion detection event"""
        self.motion_events.append(time.time())
        # Clean up old events
        self.clean_old_events()
        
    def clean_old_events(self):
        """Remove events older than the window"""
        cutoff_time = time.time() - self.window_seconds
        self.motion_events = [t for t in self.motion_events if t > cutoff_time]
        
    def calculate(self, context):
        self.clean_old_events()
        recent_count = len(self.motion_events)
        
        # Normalize - more activity could indicate suspicious behavior
        if recent_count == 0:
            return 0.1  # Very low activity is slightly suspicious
        elif recent_count < 3: 
            return 0.2  # Some activity is normal
        elif recent_count < 8:
            return 0.4  # Moderate activity
        else:
            return min(1.0, 0.5 + (recent_count - 7) * 0.05)  # High activity is more suspicious