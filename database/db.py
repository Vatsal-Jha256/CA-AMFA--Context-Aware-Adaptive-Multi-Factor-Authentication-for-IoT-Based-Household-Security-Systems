# database.py - Database implementation for AdaptiveMFA system

import sqlite3
import json
import os
import logging
from contextlib import contextmanager

logger = logging.getLogger("AdaptiveMFA.Database")

class Database:
    """Database handler for the AdaptiveMFA system"""
    
    def __init__(self, db_path="security_system.db"):
        """Initialize database connection and create tables if needed"""
        self.db_path = db_path
        self._create_tables()
        logger.info(f"Database initialized at {db_path}")
    
    @contextmanager
    def get_connection(self):
        """Context manager for database connections"""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row  # Return rows as dictionaries
        try:
            yield conn
        finally:
            conn.close()
    
    def _create_tables(self):
        """Create necessary database tables if they don't exist"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            
            # Users table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS users (
                    username TEXT PRIMARY KEY,
                    password_hash TEXT NOT NULL,
                    otp_secret TEXT NOT NULL,
                    face_encoding BLOB,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    last_login TIMESTAMP,
                    login_count INTEGER DEFAULT 0
                )
            ''')
            
            # Authentication attempts table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS auth_attempts (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    username TEXT NOT NULL,
                    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    success BOOLEAN NOT NULL,
                    risk_score REAL,
                    methods_used TEXT,
                    ip_address TEXT,
                    device_info TEXT,
                    FOREIGN KEY (username) REFERENCES users(username)
                )
            ''')
            
            # Risk factors history
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS risk_factor_history (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    factor_name TEXT NOT NULL,
                    raw_value REAL,
                    normalized_value REAL,
                    weight REAL,
                    context TEXT
                )
            ''')
            
            # System metrics
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS system_metrics (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    bandit_algorithm TEXT,
                    attempts INTEGER,
                    successful_logins INTEGER,
                    failed_logins INTEGER,
                    accuracy REAL,
                    false_authentication_rate REAL,
                    false_rejection_rate REAL,
                    avg_latency REAL,
                    cpu_percent REAL,
                    memory_percent REAL
                )
            ''')
            
            conn.commit()
    
    def get_user(self, username):
        """Get user data by username"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT * FROM users WHERE username = ?", 
                (username,)
            )
            user = cursor.fetchone()
            
            if user:
                # Convert SQLite Row to dict
                user_dict = dict(user)
                # Convert BLOB face_encoding back to numpy array if exists
                if user_dict["face_encoding"]:
                    user_dict["face_encoding"] = json.loads(user_dict["face_encoding"])
                return user_dict
            return None
    
    def get_all_users(self):
        """Get all users"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT username FROM users")
            return [row["username"] for row in cursor.fetchall()]
    
    def create_user(self, username, password_hash, otp_secret, face_encoding=None):
        """Create a new user"""
        # Convert face_encoding to JSON string if exists
        face_encoding_blob = json.dumps(face_encoding.tolist()) if face_encoding is not None else None
        
        with self.get_connection() as conn:
            cursor = conn.cursor()
            try:
                cursor.execute(
                    "INSERT INTO users (username, password_hash, otp_secret, face_encoding) VALUES (?, ?, ?, ?)",
                    (username, password_hash, otp_secret, face_encoding_blob)
                )
                conn.commit()
                logger.info(f"Created user {username}")
                return True
            except sqlite3.IntegrityError:
                logger.warning(f"Failed to create user {username}: already exists")
                return False
    
    def update_user(self, username, **kwargs):
        """Update user information"""
        if "face_encoding" in kwargs and kwargs["face_encoding"] is not None:
            kwargs["face_encoding"] = json.dumps(kwargs["face_encoding"].tolist())
        
        with self.get_connection() as conn:
            cursor = conn.cursor()
            
            # Dynamically build the update query
            fields = []
            values = []
            for field, value in kwargs.items():
                fields.append(f"{field} = ?")
                values.append(value)
            
            if not fields:
                return False
            
            query = f"UPDATE users SET {', '.join(fields)} WHERE username = ?"
            values.append(username)
            
            cursor.execute(query, values)
            conn.commit()
            return cursor.rowcount > 0
    
    def update_login(self, username):
        """Update last login time and count for a user"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "UPDATE users SET last_login = CURRENT_TIMESTAMP, login_count = login_count + 1 WHERE username = ?",
                (username,)
            )
            conn.commit()
    
    def delete_user(self, username):
        """Delete a user"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("DELETE FROM users WHERE username = ?", (username,))
            conn.commit()
            return cursor.rowcount > 0
    
    def record_auth_attempt(self, username, success, risk_score=None, methods_used=None, ip_address=None, device_info=None):
        """Record an authentication attempt"""
        methods_str = json.dumps(methods_used) if methods_used else None
        device_info_str = json.dumps(device_info) if device_info else None
        
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """INSERT INTO auth_attempts 
                   (username, success, risk_score, methods_used, ip_address, device_info)
                   VALUES (?, ?, ?, ?, ?, ?)""",
                (username, success, risk_score, methods_str, ip_address, device_info_str)
            )
            conn.commit()
    
    def get_recent_auth_attempts(self, username, limit=10):
        """Get recent authentication attempts for a user"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT * FROM auth_attempts WHERE username = ? ORDER BY timestamp DESC LIMIT ?",
                (username, limit)
            )
            return [dict(row) for row in cursor.fetchall()]
    
    def get_failed_attempts_count(self, username, seconds=300):
        """Get the count of failed attempts within the specified time window"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """SELECT COUNT(*) as count FROM auth_attempts 
                   WHERE username = ? AND success = 0 AND 
                   timestamp > datetime('now', ?||' seconds')""",
                (username, -seconds)
            )
            return cursor.fetchone()["count"]
    
    def record_risk_factor(self, factor_name, raw_value, normalized_value, weight, context=None):
        """Record risk factor values for analysis"""
        context_str = json.dumps(context) if context else None
        
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """INSERT INTO risk_factor_history 
                   (factor_name, raw_value, normalized_value, weight, context)
                   VALUES (?, ?, ?, ?, ?)""",
                (factor_name, raw_value, normalized_value, weight, context_str)
            )
            conn.commit()
    
    def save_metrics(self, metrics):
        """Save system metrics"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            
            # Extract fields from metrics dict
            fields = list(metrics.keys())
            placeholders = ", ".join(["?"] * len(fields))
            values = [metrics[field] for field in fields]
            
            query = f"INSERT INTO system_metrics ({', '.join(fields)}) VALUES ({placeholders})"
            cursor.execute(query, values)
            conn.commit()
    
    def migrate_from_json(self, json_file="users.json"):
        """Migrate users from JSON file to database"""
        if not os.path.exists(json_file):
            logger.warning(f"JSON file {json_file} not found")
            return True
        
        try:
            with open(json_file, 'r') as f:
                users = json.load(f)
            
            with self.get_connection() as conn:
                cursor = conn.cursor()
                for username, user_data in users.items():
                    # Extract user data
                    password_hash = user_data.get("password")
                    otp_secret = user_data.get("otp_secret")
                    face_encoding = user_data.get("face_encoding")
                    
                    # Convert face_encoding to JSON string if exists
                    face_encoding_blob = json.dumps(face_encoding) if face_encoding else None
                    
                    # Check if user already exists
                    cursor.execute("SELECT 1 FROM users WHERE username = ?", (username,))
                    if cursor.fetchone():
                        logger.info(f"User {username} already exists in database, skipping")
                        continue
                    
                    # Insert user
                    cursor.execute(
                        "INSERT INTO users (username, password_hash, otp_secret, face_encoding) VALUES (?, ?, ?, ?)",
                        (username, password_hash, otp_secret, face_encoding_blob)
                    )
                
                conn.commit()
                logger.info(f"Migrated {len(users)} users from JSON to database")
            return True
        except Exception as e:
            logger.error(f"Failed to migrate users from JSON: {e}")
            return False
