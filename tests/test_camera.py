#!/usr/bin/env python3
import os
import time
import sys
import subprocess
from datetime import datetime

try:
    from PIL import Image
    HAVE_PIL = True
except ImportError:
    HAVE_PIL = False
    print("PIL not installed. Install with: pip install Pillow")

# Save directories
raw_dir = "./captured_images/raw"
compressed_dir = "./captured_images/compressed"
os.makedirs(raw_dir, exist_ok=True)
os.makedirs(compressed_dir, exist_ok=True)

# PC details (can be changed by arguments)
PC_IP = "172.20.10.8"  # Default PC IP
UPLOAD_URL = f"http://{PC_IP}:5000/upload"

# Resize config
MAX_SIZE = 640  # max width or height

def resize_and_compress_image(input_path, output_path):
    """Resize and compress an image"""
    if not HAVE_PIL:
        print("PIL not available. Skipping compression.")
        return False
        
    try:
        with Image.open(input_path) as img:
            img.thumbnail((MAX_SIZE, MAX_SIZE))  # preserve aspect ratio
            img.save(output_path, "JPEG", quality=85)
            print(f"? Compressed and resized: {output_path}")
            return True
    except Exception as e:
        print(f"Error compressing image: {e}")
        return False

def send_image_to_pc(filepath, upload_url):
    """Send image to PC server"""
    try:
        import requests
    except ImportError:
        print("Requests not installed. Install with: pip install requests")
        return False
        
    try:
        with open(filepath, 'rb') as f:
            files = {'file': (os.path.basename(filepath), f)}
            res = requests.post(upload_url, files=files)
            print(f"? Sent {filepath}, server responded: {res.text}")
            return True
    except Exception as e:
        print(f"? Failed to send image: {e}")
        return False

def capture_image():
    """Capture a single image"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    raw_path = f"{raw_dir}/image_{timestamp}.jpg"
    compressed_path = f"{compressed_dir}/image_{timestamp}_compressed.jpg"
    
    try:
        # Capture image using libcamera-jpeg
        print("Capturing image...")
        subprocess.run(["libcamera-jpeg", "-o", raw_path, "-t", "1000", "--nopreview"], 
                      check=True, stderr=subprocess.PIPE)
        print(f"? Captured {raw_path}")
        
        # Compress if PIL is available
        if HAVE_PIL:
            resize_and_compress_image(raw_path, compressed_path)
            return compressed_path
        else:
            return raw_path
    except subprocess.CalledProcessError as e:
        print(f"Camera capture failed: {e.stderr.decode() if e.stderr else e}")
        return None
    except Exception as e:
        print(f"Error capturing image: {e}")
        return None

def main():
    if len(sys.argv) > 1:
        global PC_IP, UPLOAD_URL
        PC_IP = sys.argv[1]
        UPLOAD_URL = f"http://{PC_IP}:5000/upload"
        print(f"Using PC IP: {PC_IP}")
    
    print("Camera Test Script")
    print("1. Capture a single image")
    print("2. Start continuous capture (15s interval)")
    print("3. Exit")
    
    choice = input("Enter choice (1-3): ")
    
    if choice == "1":
        image_path = capture_image()
        if image_path:
            upload = input("Do you want to upload the image to PC? (y/n): ")
            if upload.lower() == 'y':
                send_image_to_pc(image_path, UPLOAD_URL)
    
    elif choice == "2":
        interval = input("Enter capture interval in seconds (default: 15): ")
        try:
            interval = int(interval) if interval else 15
        except ValueError:
            interval = 15
            
        print(f"Starting continuous capture every {interval} seconds...")
        print("Press Ctrl+C to stop")
        
        try:
            count = 0
            while True:
                count += 1
                print(f"\nCapture #{count}")
                image_path = capture_image()
                if image_path:
                    upload = input("Upload this image? (y/n/a for always, s for stop): ")
                    if upload.lower() == 'y':
                        send_image_to_pc(image_path, UPLOAD_URL)
                    elif upload.lower() == 'a':
                        send_image_to_pc(image_path, UPLOAD_URL)
                        print("Auto-uploading all future images")
                        auto_upload = True
                    elif upload.lower() == 's':
                        break
                        
                print(f"Waiting {interval} seconds...")
                time.sleep(interval)
        except KeyboardInterrupt:
            print("\nContinuous capture stopped")
    
    print("Camera test completed")

if __name__ == "__main__":
    main()
