import random
import json 
import logging
from RiskAssessment.RiskFactor import RiskFactor
# Set up logging    
logger = logging.getLogger(__name__)
class NetworkBasedRiskFactor(RiskFactor):
    """Risk factor based on network conditions"""
    
    def __init__(self, name="network_factor", weight=0.5):
        super().__init__(name, weight)
        self.known_devices = set()
        self.load_known_devices()
        
    def load_known_devices(self):
        """Load known network devices from file"""
        try:
            with open('known_devices.json', 'r') as f:
                self.known_devices = set(json.load(f))
        except (FileNotFoundError, json.JSONDecodeError):
            logger.info("No known devices file found. Starting fresh.")
    
    def save_known_devices(self):
        """Save known devices to file"""
        with open('known_devices.json', 'w') as f:
            json.dump(list(self.known_devices), f)
    
    def add_known_device(self, device_id):
        """Add a device to known devices list"""
        self.known_devices.add(device_id)
        self.save_known_devices()
        
    def calculate(self, context):
        # Check if connecting device is known
        device_id = context.get('device_id')
        if not device_id:
            return 0.5  # Neutral if no device info
        
        if device_id in self.known_devices:
            return random.uniform(0.1, 0.3)  # Low risk for known device
        else:
            return random.uniform(0.7, 0.9)  # High risk for unknown device