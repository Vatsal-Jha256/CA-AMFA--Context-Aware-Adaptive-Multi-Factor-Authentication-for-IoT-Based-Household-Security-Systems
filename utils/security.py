import bcrypt
import pyotp
import requests
import numpy as np
import cv2
import base64
import pickle

class SecurityUtils:
    def __init__(self, db, face_server_url="http://172.20.10.2:5000"):
        """Initialize security utilities with database connection
        
        Args:
            db: Database instance for persistent storage
            face_server_url: URL of the face recognition server
        """
        # Store database reference
        self.db = db
        
        # Face recognition server URL
        self.face_server_url = face_server_url
        
        # Rate limiting configuration
        self.max_attempts = 3
        self.lockout_duration = 300  # 5 minutes
        
    def hash_password(self, password):
        """Hash password using bcrypt"""
        return bcrypt.hashpw(password.encode(), bcrypt.gensalt())
    
    def verify_password(self, password, hashed):
        """Verify password against hash"""
        return bcrypt.checkpw(password.encode(), hashed)
    
    def generate_totp_secret(self):
        """Generate new TOTP secret"""
        return pyotp.random_base32()
    def get_totp_qr(self, username, secret, issuer="AdaptiveMFA"):
        """Generate QR code provisioning URI for TOTP setup"""
        totp = pyotp.TOTP(secret)
        return totp.provisioning_uri(name=username, issuer_name=issuer)
    
    def verify_totp(self, secret, token):
        """Verify TOTP token"""
        totp = pyotp.TOTP(secret)
        return totp.verify(token)
    
    def get_face_encoding(self, image):
        """Get face encoding by sending to server"""
        # Convert image to JPEG format
        _, img_encoded = cv2.imencode('.jpg', image)
        
        # Base64 encode the image
        img_base64 = base64.b64encode(img_encoded).decode('utf-8')
        
        # Send to server
        try:
            response = requests.post(
                f"{self.face_server_url}/face/encode",
                json={'image': img_base64},
                timeout=10
            )
            
            if not response.ok:
                print(f"Server error: {response.status_code}")
                return None
            
            data = response.json()
            if not data.get('success'):
                print(f"Face encoding failed: {data.get('error')}")
                return None
            
            # Decode and deserialize the face encoding
            encoding_base64 = data['face_encoding']
            encoding_bytes = base64.b64decode(encoding_base64)
            encoding = pickle.loads(encoding_bytes)
            
            return encoding
        except Exception as e:
            print(f"Error connecting to face server: {e}")
            return None
    def compare_faces(self, encoding1, encoding2, tolerance=0.6):
        """Compare face encodings using server"""
        if encoding1 is None or encoding2 is None:
            return False
        
        try:
            # Serialize the encodings
            encoding1_bytes = pickle.dumps(encoding1)
            encoding2_bytes = pickle.dumps(encoding2)
            
            # Base64 encode for JSON transmission
            encoding1_base64 = base64.b64encode(encoding1_bytes).decode('utf-8')
            encoding2_base64 = base64.b64encode(encoding2_bytes).decode('utf-8')
            
            # Send to server
            response = requests.post(
                f"{self.face_server_url}/face/compare",
                json={
                    'encoding1': encoding1_base64,
                    'encoding2': encoding2_base64,
                    'tolerance': tolerance
                },
                timeout=10
            )
            
            if not response.ok:
                print(f"Server error: {response.status_code}")
                return False
            
            data = response.json()
            if not data.get('success'):
                print(f"Face comparison failed")
                return False
            
            return data.get('match', False)
        except Exception as e:
            print(f"Error connecting to face server: {e}")
            return False
    
    def check_rate_limit(self, username):
        """Check if user is rate limited using database records
        
        Returns:
            bool: True if user can attempt authentication, False if rate limited
        """
        # Get recent failed attempts from database
        recent_failures = self.db.get_failed_attempts_count(username, seconds=self.lockout_duration)
        
        # Check if user is locked out
        return recent_failures < self.max_attempts
    def record_attempt(self, username, success, risk_score=None, methods_used=None, device_info=None):
        """Record an authentication attempt in the database"""
        self.db.record_auth_attempt(
            username=username,
            success=success,
            risk_score=risk_score,
            methods_used=methods_used,
            device_info=device_info
        )
