import os
import sys
import time
import subprocess
from datetime import datetime
import paho.mqtt.client as mqtt
import threading

# Check if we're running on a Raspberry Pi
IS_RASPBERRY_PI = os.uname().machine.startswith('arm') or os.uname().machine == 'aarch64'
if IS_RASPBERRY_PI:
    import RPi.GPIO as GPIO
    try:
        import board
        import busio
        from adafruit_ssd1306 import SSD1306_I2C
        # Import PIL for image processing if available
        try:
            from PIL import Image, ImageDraw, ImageFont
        except ImportError:
            print("PIL not installed. Image compression won't be available.")
    except ImportError:
        print("Warning: Adafruit libraries not installed. OLED display may not work.")
else:
    print("Not running on Raspberry Pi - using mock hardware controller")
    # Create a mock hardware controller class
    class MockHardwareController:
        def __init__(self):
            print("Mock hardware controller initialized")
            
        def connect_mqtt(self, broker, port):
            print(f"Mock MQTT connection to {broker}:{port}")
            
        def read_motion(self):
            return False
            
        def control_lock(self, engage):
            print(f"Mock lock {'engaged' if engage else 'disengaged'}")
            
        def read_keypad(self):
            # Simulate keypad input
            import random
            if random.random() < 0.1:  # 10% chance of returning a key
                return random.choice(['1', '2', '3', '4', '5', '6', '7', '8', '9', '0', '#', '*'])
            return None
            
        def display_message(self, message):
            print(f"Mock display: {message}")
            
        def capture_image(self):
            print("Mock camera: Image captured")
            return "mock_image_path.jpg"
            
        def cleanup(self):
            print("Mock cleanup performed")
    
    # Create a mock controller instance
    mock_controller = MockHardwareController()
import logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("hardware.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("Hardware")

class HardwareController:
    def __init__(self):
        # Camera configuration
        self.camera_enabled = False
        self.image_capture_interval = 15  # seconds
        self.raw_dir = "/home/pi/captured_images/raw" if IS_RASPBERRY_PI else "./captured_images/raw"
        self.compressed_dir = "/home/pi/captured_images/compressed" if IS_RASPBERRY_PI else "./captured_images/compressed"
        self.camera_thread = None
        self.camera_running = False
        self.pc_ip = "172.20.10.8"  # Default PC IP
        self.upload_url = f"http://{self.pc_ip}:5000/upload"
        self.max_image_size = 640  # Maximum width or height for compression
        
        # Create directories with 0o755 permissions
        try:
            os.makedirs(self.raw_dir, exist_ok=True)
            os.makedirs(self.compressed_dir, exist_ok=True)
            os.chmod(self.raw_dir, 0o755)  # Ensure write permissions
            os.chmod(self.compressed_dir, 0o755)
        except Exception as e:
            logger.error(f"Directory creation failed: {str(e)}")
            raise
        if IS_RASPBERRY_PI:
            # Initialize GPIO
            GPIO.setmode(GPIO.BCM)
            GPIO.setwarnings(False)
            
            # PIR Motion Sensor
            self.PIR_PIN = 17
            GPIO.setup(self.PIR_PIN, GPIO.IN)
            
            # Servo Motor
            self.SERVO_PIN = 18
            GPIO.setup(self.SERVO_PIN, GPIO.OUT)
            self.servo = GPIO.PWM(self.SERVO_PIN, 50)  # 50Hz frequency
            self.servo.start(0)
            
            # Keypad setup - using the pins from your setup guide
            self.KEYPAD_ROWS = [23, 24, 25, 8]
            self.KEYPAD_COLS = [7, 12, 16, 20]
            self.KEYPAD_KEYS = [
                ['1', '2', '3', 'A'],
                ['4', '5', '6', 'B'],
                ['7', '8', '9', 'C'],
                ['*', '0', '#', 'D']
            ]
            
            # Setup row pins as outputs
            for row in self.KEYPAD_ROWS:
                GPIO.setup(row, GPIO.OUT)
                GPIO.output(row, GPIO.HIGH)
            
            # Setup column pins as inputs with pull-up
            for col in self.KEYPAD_COLS:
                GPIO.setup(col, GPIO.IN, pull_up_down=GPIO.PUD_UP)
            
            # OLED Display setup
            try:
                i2c = busio.I2C(board.SCL, board.SDA)
                self.display = SSD1306_I2C(128, 32, i2c)
                self.display.fill(0)
                self.display.show()
                self.display_available = True
            except Exception as e:
                print(f"OLED Display initialization failed: {e}")
                print("Using the correct I2C bus: SCL=Pin 5, SDA=Pin 3")
                self.display_available = False
                
        else:
            # Use mock controller
            self.controller = mock_controller
            self.display_available = False
        
        # MQTT Setup
        self.mqtt_client = mqtt.Client()
        self.mqtt_client.on_connect = self.on_connect
        self.mqtt_client.on_message = self.on_message
        
    def on_connect(self, client, userdata, flags, rc):
        print(f"Connected with result code {rc}")
        client.subscribe("security/status")
        client.subscribe("security/camera/control")
        
    def on_message(self, client, userdata, msg):
        print(f"{msg.topic} {str(msg.payload)}")
        if msg.topic == "security/camera/control":
            command = msg.payload.decode()
            if command == "start":
                self.start_camera_capture()
            elif command == "stop":
                self.stop_camera_capture()
            elif command.startswith("interval:"):
                try:
                    interval = int(command.split(":")[1])
                    self.image_capture_interval = interval
                    print(f"Camera capture interval set to {interval} seconds")
                except ValueError:
                    print("Invalid interval value")

    def connect_mqtt(self, broker="localhost", port=1883):
        try:
            self.mqtt_client.connect(broker, port, 60)
            self.mqtt_client.loop_start()
            print(f"MQTT connected to {broker}:{port}")
        except Exception as e:
            print(f"MQTT connection failed: {e}")
            
    def read_motion(self):
        if IS_RASPBERRY_PI:
            return GPIO.input(self.PIR_PIN)
        else:
            return self.controller.read_motion()
        
    def control_lock(self, engage):
        if IS_RASPBERRY_PI:
            if engage:
                self.servo.ChangeDutyCycle(7.5)  # 90 degrees
            else:
                self.servo.ChangeDutyCycle(2.5)  # 0 degrees
            time.sleep(0.5)  # Give servo time to move
            self.servo.ChangeDutyCycle(0)  # Stop servo jitter
        else:
            self.controller.control_lock(engage)
        
    def read_keypad(self):
        if IS_RASPBERRY_PI:
            key = None
            for row_idx, row in enumerate(self.KEYPAD_ROWS):
                GPIO.output(row, GPIO.LOW)  # Set current row to LOW
                
                for col_idx, col in enumerate(self.KEYPAD_COLS):
                    if GPIO.input(col) == GPIO.LOW:  # Key is pressed
                        key = self.KEYPAD_KEYS[row_idx][col_idx]
                        
                        # Wait for key release
                        while GPIO.input(col) == GPIO.LOW:
                            time.sleep(0.01)
                            
                GPIO.output(row, GPIO.HIGH)  # Set current row back to HIGH
            return key
        else:
            return self.controller.read_keypad()
        
    def display_message(self, message):
        if IS_RASPBERRY_PI and self.display_available:
            try:
                self.display.fill(0)
                # Create blank image for drawing
                image = Image.new("1", (self.display.width, self.display.height))
                draw = ImageDraw.Draw(image)
                
                # Load default font
                font = ImageFont.load_default()
                
                # Split message into lines
                lines = message.split('\n')
                for i, line in enumerate(lines[:4]):  # Display up to 4 lines
                    draw.text((0, i*8), line, font=font, fill=255)
                
                # Display image
                self.display.image(image)
                self.display.show()
            except Exception as e:
                print(f"Display error: {e}")
        else:
            if not IS_RASPBERRY_PI:
                self.controller.display_message(message)
            else:
                print(f"Display message (OLED unavailable): {message}")
    
    def capture_image(self):
        """Capture a single image using libcamera-jpeg"""
        if not IS_RASPBERRY_PI:
            return self.controller.capture_image()
            
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        raw_path = f"{self.raw_dir}/image_{timestamp}.jpg"
        compressed_path = f"{self.compressed_dir}/image_{timestamp}_compressed.jpg"
        
        try:
            # Capture image using libcamera-jpeg
            subprocess.run(["libcamera-jpeg", "-o", raw_path, "-t", "1000", "--nopreview"], 
                          check=True, stderr=subprocess.PIPE)
            print(f"? Captured {raw_path}")
            
            # Compress and resize the image
            self.resize_and_compress_image(raw_path, compressed_path)
            
            # Notify via MQTT
            self.mqtt_client.publish("security/camera/image", compressed_path)
            
            return compressed_path
        except subprocess.CalledProcessError as e:
            print(f"Camera capture failed: {e.stderr.decode() if e.stderr else e}")
            return None
        except Exception as e:
            print(f"Error capturing image: {e}")
            return None

    def resize_and_compress_image(self, input_path, output_path):
        """Resize and compress an image using PIL"""
        if not IS_RASPBERRY_PI or 'Image' not in globals():
            print("Image compression not available")
            return
            
        try:
            with Image.open(input_path) as img:
                img.thumbnail((self.max_image_size, self.max_image_size))  # preserve aspect ratio
                img.save(output_path, "JPEG", quality=85)
                print(f"? Compressed and resized: {output_path}")
        except Exception as e:
            print(f"Error compressing image: {e}")
    
    def send_image_to_pc(self, filepath):
        """Send image to PC server"""
        if not os.path.exists(filepath):
            print(f"Error: File not found: {filepath}")
            return
            
        import requests
        
        with open(filepath, 'rb') as f:
            files = {'file': (os.path.basename(filepath), f)}
            try:
                res = requests.post(self.upload_url, files=files)
                print(f"? Sent {filepath}, server responded: {res.text}")
            except Exception as e:
                print(f"? Failed to send image: {e}")
    
    def camera_loop(self):
        """Background loop for periodically capturing images"""
        while self.camera_running:
            image_path = self.capture_image()
            if image_path:
                self.send_image_to_pc(image_path)
            
            # Wait for the next capture interval
            start_time = time.time()
            while self.camera_running and (time.time() - start_time) < self.image_capture_interval:
                time.sleep(0.5)  # Check for stop condition every 0.5 seconds
    
    def start_camera_capture(self, pc_ip=None, interval=None):
        """Start periodic camera capture in a separate thread"""
        if pc_ip:
            self.pc_ip = pc_ip
            self.upload_url = f"http://{self.pc_ip}:5000/upload"
            
        if interval:
            self.image_capture_interval = interval
            
        if not self.camera_running:
            self.camera_running = True
            self.camera_thread = threading.Thread(target=self.camera_loop)
            self.camera_thread.daemon = True
            self.camera_thread.start()
            print(f"Camera capture started with {self.image_capture_interval}s interval")
            
            # Publish status
            self.mqtt_client.publish("security/camera/status", "running")
            
            # Show on display if available
            self.display_message("Camera active\nCapturing images")
    
    def stop_camera_capture(self):
        """Stop the camera capture thread"""
        if self.camera_running:
            self.camera_running = False
            if self.camera_thread:
                self.camera_thread.join(timeout=2.0)
            print("Camera capture stopped")
            
            # Publish status
            self.mqtt_client.publish("security/camera/status", "stopped")
            
            # Show on display if available
            self.display_message("Camera inactive")
        
    def display_qr_code(self, uri):
        """Display QR code for TOTP setup with fallbacks"""
        try:
            # First extract and display the secret key (most important part)
            import re
            secret_match = re.search(r'secret=([A-Z0-9]+)', uri)
            secret = secret_match.group(1) if secret_match else uri
            
            # Display the secret in chunks if needed
            self.display_message(f"TOTP Secret:\n{secret[:16]}")
            time.sleep(2)
            
            if len(secret) > 16:
                self.display_message(f"TOTP Secret cont:\n{secret[16:]}")
                time.sleep(2)
            
            # Try to show QR on PC if available
            try:
                import qrcode
                from PIL import Image
                
                # Generate QR code
                qr = qrcode.QRCode(
                    version=1,
                    error_correction=qrcode.constants.ERROR_CORRECT_L,
                    box_size=10,
                    border=4,
                )
                qr.add_data(uri)
                qr.make(fit=True)
                qr_img = qr.make_image(fill_color="black", back_color="white")
                
                # Save to temporary file
                import tempfile
                with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
                    qr_path = tmp.name
                    qr_img.save(qr_path)
                
                # Try to display on PC
                if os.name == 'nt':  # Windows
                    os.startfile(qr_path)
                elif os.name == 'posix':  # macOS/Linux
                    if os.system(f"which xdg-open > /dev/null") == 0:
                        os.system(f"xdg-open {qr_path} &")
                    elif os.system(f"which display > /dev/null") == 0:
                        os.system(f"display {qr_path} &")
                
                print(f"QR code saved to: {qr_path}")
                print("You can scan this with your authenticator app")
                
            except ImportError:
                print("QR code generation requires qrcode[pil] package")
                print("Install with: pip install qrcode[pil]")
            except Exception as e:
                print(f"Couldn't display QR: {e}")
                
        except Exception as e:
            print(f"Error handling QR code: {e}")
            # Final fallback - just show the secret
            self.display_message("TOTP Setup:\nEnter secret\nmanually")
            time.sleep(1)
            self.display_message(f"Secret:\n{secret}")
            time.sleep(3)

    def update_risk_display(self, username, risk_score):
        """Update OLED with current risk score"""
        self.hardware.display_risk_score(risk_score, self.low_threshold, self.high_threshold)
        time.sleep(1.5)  # Show risk score for 1.5 seconds

    def display_risk_score(self, risk_score, threshold_low=0.3, threshold_high=0.7):
        """Display risk score with visual indicator on OLED"""
        if not IS_RASPBERRY_PI or not self.display_available:
            print(f"Risk score: {risk_score:.2f}")
            return
        
        try:
            self.display.fill(0)
            
            # Create blank image for drawing
            image = Image.new("1", (self.display.width, self.display.height))
            draw = ImageDraw.Draw(image)
            
            # Load default font
            font = ImageFont.load_default()
            
            # Draw risk score text
            draw.text((0, 0), f"Risk Score: {risk_score:.2f}", font=font, fill=255)
            
            # Calculate bar length based on risk score
            bar_width = int(risk_score * (self.display.width - 2))
            
            # Draw progress bar outline
            draw.rectangle([(0, 16), (self.display.width - 1, 24)], outline=255, fill=0)
            
            # Draw progress bar fill
            if bar_width > 0:
                draw.rectangle([(1, 17), (bar_width, 23)], outline=255, fill=255)
            
            # Draw risk level text
            if risk_score < threshold_low:
                risk_text = "LOW RISK"
            elif risk_score < threshold_high:
                risk_text = "MEDIUM RISK"
            else:
                risk_text = "HIGH RISK"
            
            draw.text((0, 25), risk_text, font=font, fill=255)
            
            # Display image
            self.display.image(image)
            self.display.show()
            
        except Exception as e:
            print(f"Display error: {e}")
            print(f"Risk score: {risk_score:.2f}")

    def cleanup(self):
        """Clean up resources"""
        # Stop camera if running
        self.stop_camera_capture()
        
        if IS_RASPBERRY_PI:
            try:
                self.servo.stop()
            except:
                pass
            GPIO.cleanup()
        else:
            self.controller.cleanup()
        
        try:
            self.mqtt_client.loop_stop()
            self.mqtt_client.disconnect()
        except:
            pass
