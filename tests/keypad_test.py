import RPi.GPIO as GPIO
import time

# Set up GPIO
GPIO.setmode(GPIO.BCM)
GPIO.setwarnings(False)

# Use the pins from your setup guide
ROWS = [23, 24, 25, 8]
COLS = [7, 12, 16, 20]

KEYS = [
    ['1', '2', '3', 'A'],
    ['4', '5', '6', 'B'],
    ['7', '8', '9', 'C'],
    ['*', '0', '#', 'D']
]

# Set up row pins as outputs
for row in ROWS:
    GPIO.setup(row, GPIO.OUT)
    GPIO.output(row, GPIO.HIGH)

# Set up column pins as inputs with pull-up
for col in COLS:
    GPIO.setup(col, GPIO.IN, pull_up_down=GPIO.PUD_UP)

print("Keypad Test Running. Press keys on your keypad.")
print("Press Ctrl+C to exit.")

try:
    while True:
        # Scan rows and columns
        for row_idx, row in enumerate(ROWS):
            GPIO.output(row, GPIO.LOW)  # Set current row to LOW
            
            for col_idx, col in enumerate(COLS):
                if GPIO.input(col) == GPIO.LOW:  # Key is pressed
                    key = KEYS[row_idx][col_idx]
                    print(f"Key pressed: {key} at position ({row_idx},{col_idx})")
                    
                    # Wait for key release
                    while GPIO.input(col) == GPIO.LOW:
                        time.sleep(0.01)
            
            GPIO.output(row, GPIO.HIGH)  # Set current row back to HIGH
        
        time.sleep(0.1)  # Short delay between scans

except KeyboardInterrupt:
    print("\nExiting test")
finally:
    GPIO.cleanup()
