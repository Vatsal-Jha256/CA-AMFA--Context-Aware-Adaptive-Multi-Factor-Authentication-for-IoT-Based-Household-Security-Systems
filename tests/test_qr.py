def test_full_qr_display():
    """Test function for displaying a QR code"""
    from hardware_controller import HardwareController
    
    # Initialize the hardware controller
    hardware = HardwareController()
    
    # Sample TOTP URI (replace with your actual TOTP URI if needed)
    totp_uri = "otpauth://totp/SecuritySystem:admin@example.com?secret=ABCDEF67&issuer=SecuritySystem"
    
    print("\nTesting QR code display...")
    print(f"URI to display: {totp_uri}")
    
    # Extract secret for manual entry
    import re
    secret_match = re.search(r'secret=([A-Z0-9]+)', totp_uri)
    secret = secret_match.group(1) if secret_match else "NO_SECRET_FOUND"
    print(f"\nManual entry secret: {secret}")
    
    # Call the display_qr_code method
    hardware.display_qr_code(totp_uri)
    
    # For Raspberry Pi, we'll save to a known location
    qr_path = "/home/pi/totp_qr.png"
    try:
        import qrcode
        qr = qrcode.QRCode(
            version=1,
            error_correction=qrcode.constants.ERROR_CORRECT_L,
            box_size=10,
            border=4,
        )
        qr.add_data(totp_uri)
        qr.make(fit=True)
        qr_img = qr.make_image(fill_color="black", back_color="white")
        qr_img.save(qr_path)
        print(f"\nQR code saved to: {qr_path}")
        print("You can:")
        print("1. View it on your Pi with: sudo apt install fim && fim -a {qr_path}")
        print("2. SCP it to your computer: scp pi@your_pi_ip:{qr_path} .")
        print("3. Enter the secret manually: {secret}")
    except Exception as e:
        print(f"\nCouldn't save QR code: {e}")
        print("Please enter the secret manually: {secret}")
    
    # Clean up hardware resources
    hardware.cleanup()
    print("\nTest completed.")

if __name__ == "__main__":
    test_full_qr_display()
