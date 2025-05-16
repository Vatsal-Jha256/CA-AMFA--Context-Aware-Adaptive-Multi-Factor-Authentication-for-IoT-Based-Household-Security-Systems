from RiskAssessment.RiskFactor import RiskFactor
import json
import time 
import datetime
import random
from collections import defaultdict
import logging
# Set up logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

class UserBehaviorRiskFactor(RiskFactor):
    """Risk factor based on user behavior patterns"""
    
    def __init__(self, name="user_behavior", weight=1.2):
        super().__init__(name, weight)
        self.access_history = defaultdict(list)  # {user_id: [timestamps]}
        self.load_history()
        
    def load_history(self):
        """Load access history from file if available"""
        try:
            with open('access_history.json', 'r') as f:
                self.access_history = defaultdict(list, json.load(f))
        except (FileNotFoundError, json.JSONDecodeError):
            logger.info("No existing access history found or file corrupted. Starting fresh.")
    
    def save_history(self):
        """Save access history to file"""
        with open('access_history.json', 'w') as f:
            json.dump(self.access_history, f)
            
    def record_access(self, user_id):
        """Record a successful access by user"""
        timestamp = time.time()
        self.access_history[user_id].append(timestamp)
        # Keep only last 100 entries per user
        if len(self.access_history[user_id]) > 100:
            self.access_history[user_id] = self.access_history[user_id][-100:]
        self.save_history()
        
    def calculate(self, context):
        user_id = context.get('user_id')
        
        if not user_id or user_id not in self.access_history or len(self.access_history[user_id]) < 5:
            return 0.7  # Higher risk for new/unknown users
            
        # Analyze access patterns
        hour_now = datetime.datetime.now().hour
        day_now = datetime.datetime.now().weekday()
        
        # Check if current hour and day match past behavior
        hour_matches = 0
        day_matches = 0
        
        for timestamp in self.access_history[user_id][-20:]:  # Check last 20 accesses
            dt = datetime.datetime.fromtimestamp(timestamp)
            if abs(dt.hour - hour_now) <= 1:  # Within 1 hour window
                hour_matches += 1
            if dt.weekday() == day_now:
                day_matches += 1
                
        hour_frequency = hour_matches / min(20, len(self.access_history[user_id]))
        day_frequency = day_matches / min(20, len(self.access_history[user_id]))
        
        # Calculate risk based on pattern matching
        if hour_frequency > 0.3 and day_frequency > 0.3:
            return random.uniform(0.1, 0.3)  # Low risk - matches typical pattern
        elif hour_frequency > 0.2 or day_frequency > 0.2:
            return random.uniform(0.3, 0.6)  # Medium risk - partial match
        else:
            return random.uniform(0.6, 0.9)  # High risk - unusual pattern