#!/usr/bin/env python3
"""
Thorlabs Power Meter Diagnostic Tool
This script checks if your Thorlabs power meter is properly recognized by your system
and works on both Windows and Raspberry Pi.
"""

import sys
import os
import time
import platform
import subprocess
import logging

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('powermeter_diagnostic.log')
    ]
)

def print_and_log(message):
    """Print message to console and log it"""
    print(message)
    logging.info(message)

print_and_log(f"Starting Thorlabs Power Meter diagnostic on {platform.system()} {platform.release()}")
print_and_log(f"Python version: {platform.python_version()}")

# Check if we're on Windows or Raspberry Pi
IS_WINDOWS = platform.system() == 'Windows'
IS_RASPBERRY_PI = platform.system() == 'Linux' and os.path.exists('/sys/firmware/devicetree/base/model') and 'Raspberry Pi' in open('/sys/firmware/devicetree/base/model').read()

print_and_log(f"Running on Windows: {IS_WINDOWS}")
print_and_log(f"Running on Raspberry Pi: {IS_RASPBERRY_PI}")

# Step 1: Check for required packages
print_and_log("\n=== Step 1: Checking for required packages ===")

required_packages = ['pyvisa', 'pyvisa-py', 'ThorlabsPM100']
missing_packages = []

for package in required_packages:
    try:
        __import__(package.replace('-', '_'))
        print_and_log(f"[OK] {package} is installed")
    except ImportError:
        print_and_log(f"[MISSING] {package} is NOT installed")
        missing_packages.append(package)

if missing_packages:
    print_and_log("\nInstalling missing packages...")
    for package in missing_packages:
        try:
            subprocess.check_call([sys.executable, '-m', 'pip', 'install', package])
            print_and_log(f"[OK] Successfully installed {package}")
        except subprocess.CalledProcessError:
            print_and_log(f"[ERROR] Failed to install {package}")

# Now import the required packages
try:
    import pyvisa
    import ThorlabsPM100
    print_and_log("\nSuccessfully imported required packages")
except ImportError as e:
    print_and_log(f"\nFailed to import required packages: {e}")
    sys.exit(1)

# Step 2: Check USB devices
print_and_log("\n=== Step 2: Checking USB devices ===")

if IS_WINDOWS:
    try:
        # On Windows, use PowerShell to list USB devices
        result = subprocess.run(['powershell', '-Command', "Get-PnpDevice | Where-Object {$_.Class -eq 'USB'} | Format-Table -Property FriendlyName,Status,DeviceID"], 
                               capture_output=True, text=True)
        print_and_log("USB devices detected by Windows:")
        print_and_log(result.stdout)
        
        # Look for Thorlabs in the output
        if "Thorlabs" in result.stdout:
            print_and_log("[FOUND] Thorlabs device found in USB devices list")
        else:
            print_and_log("[NOT FOUND] No Thorlabs device found in USB devices list")
    except Exception as e:
        print_and_log(f"Error listing USB devices: {e}")
else:
    try:
        # On Linux/Raspberry Pi, use lsusb
        result = subprocess.run(['lsusb'], capture_output=True, text=True)
        print_and_log("USB devices detected by Linux:")
        print_and_log(result.stdout)
        
        # Look for Thorlabs in the output (vendor ID 1313)
        if "1313" in result.stdout or "Thorlabs" in result.stdout:
            print_and_log("[FOUND] Thorlabs device found in USB devices list")
        else:
            print_and_log("[NOT FOUND] No Thorlabs device found in USB devices list")
    except Exception as e:
        print_and_log(f"Error listing USB devices: {e}")

# Step 3: Check VISA resources
print_and_log("\n=== Step 3: Checking VISA resources ===")

# Try with default backend
try:
    rm = pyvisa.ResourceManager()
    backend_type = "Default"
    print_and_log(f"Using {backend_type} VISA backend")
except ValueError:
    # If that fails, try with the py backend
    try:
        rm = pyvisa.ResourceManager('@py')
        backend_type = "PyVISA-py"
        print_and_log(f"Using {backend_type} VISA backend")
    except Exception as e:
        print_and_log(f"Error initializing ResourceManager: {e}")
        rm = None

if rm:
    try:
        # List available resources
        resources = rm.list_resources()
        print_and_log(f"Available VISA resources: {resources}")
        
        # Check for Thorlabs identifiers
        thorlabs_identifiers = [
            "USB0::0x1313::0x8078",  # Common PM100D identifier
            "USB0::0x1313::0x8072",  # Alternative identifier
            "USB0::0x1313::0x8070",  # PM100A identifier
            "USB::0x1313",           # Generic Thorlabs identifier
            "USB0::1313::8078",      # Alternative format
            "ASRL"                   # Serial port (some PM100D use this)
        ]
        
        found_thorlabs = False
        for resource in resources:
            for identifier in thorlabs_identifiers:
                if identifier in resource:
                    print_and_log(f"[FOUND] Found Thorlabs power meter: {resource}")
                    found_thorlabs = True
                    break
        
        if not found_thorlabs:
            print_and_log("[NOT FOUND] No Thorlabs power meter found in VISA resources")
    except Exception as e:
        print_and_log(f"Error listing resources: {e}")

# Step 4: Try to connect to common Thorlabs addresses
print_and_log("\n=== Step 4: Trying to connect to power meter ===")

if rm:
    # Common addresses for Thorlabs power meters
    common_addresses = [
        "USB0::0x1313::0x8078::P0000000::INSTR",
        "USB0::0x1313::0x8078::P0005750::INSTR",
        "USB0::0x1313::0x8078::P0000001::INSTR",
        "USB0::1313::8078::INSTR",  # Original format from the script
        "ASRL1::INSTR",
        "ASRL2::INSTR"
    ]
    
    connected = False
    successful_address = None
    
    for addr in common_addresses:
        try:
            print_and_log(f"Trying to connect to {addr}...")
            device = rm.open_resource(addr, term_chars='\n', timeout=1000)
            
            # Try to get device identity
            try:
                idn = device.query("*IDN?")
                print_and_log(f"[SUCCESS] Successfully connected to device: {addr}")
                print_and_log(f"Device identity: {idn}")
                successful_address = addr
                
                # Try to read power
                try:
                    power = device.query("MEAS:POW?")
                    print_and_log(f"Current power reading: {power} W")
                except Exception as e:
                    print_and_log(f"Error reading power: {e}")
                
                connected = True
            except Exception as e:
                print_and_log(f"Could not query device identity: {e}")
            
            device.close()
            
            if connected:
                break
        except Exception as e:
            print_and_log(f"Failed to connect to {addr}: {e}")

# Step 5: Try with ThorlabsPM100 library
print_and_log("\n=== Step 5: Testing with ThorlabsPM100 library ===")

if rm and not connected:
    # Try again with ThorlabsPM100 library
    for addr in common_addresses:
        try:
            print_and_log(f"Trying to connect to {addr} with ThorlabsPM100...")
            inst = rm.open_resource(addr, term_chars='\n', timeout=1000)
            power_meter = ThorlabsPM100.ThorlabsPM100(inst)
            
            # Try to read power
            try:
                power = power_meter.read
                print_and_log(f"[SUCCESS] Successfully connected with ThorlabsPM100")
                print_and_log(f"Current power reading: {power} W")
                connected = True
                successful_address = addr
            except Exception as e:
                print_and_log(f"Error reading power: {e}")
            
            inst.close()
            
            if connected:
                break
        except Exception as e:
            print_and_log(f"Failed to connect with ThorlabsPM100: {e}")

# Step 6: Summary and recommendations
print_and_log("\n=== Summary and Recommendations ===")

if connected:
    print_and_log("[SUCCESS] Successfully connected to Thorlabs power meter!")
    print_and_log("\nRecommendations for your code:")
    print_and_log(f"1. Use the {backend_type} VISA backend")
    print_and_log(f"2. Use the resource address: {successful_address}")
    print_and_log(f"3. Make sure to set term_chars='\\n' and timeout=1000 when opening the resource")
    
    # Generate code snippet
    code_snippet = f"""
# Example code to connect to your power meter:
import pyvisa
import ThorlabsPM100

# Initialize VISA resource manager
rm = pyvisa.ResourceManager()  # or pyvisa.ResourceManager('@py') if using PyVISA-py

# Connect to the power meter
inst = rm.open_resource("{successful_address}", term_chars='\\n', timeout=1000)
power_meter = ThorlabsPM100.ThorlabsPM100(inst)

# Read power
power = power_meter.read
print(f"Current power: {{power}} W")
"""
    print_and_log("\nCode snippet:")
    print_and_log(code_snippet)
    
    # Write code snippet to file
    with open('powermeter_connection_example.py', 'w') as f:
        f.write(code_snippet)
    print_and_log("Example code saved to powermeter_connection_example.py")
    
else:
    print_and_log("[FAILED] Could not connect to Thorlabs power meter")
    print_and_log("\nPossible issues and solutions:")
    print_and_log("1. Make sure the power meter is properly connected and powered on")
    print_and_log("2. Check if the appropriate drivers are installed")
    print_and_log("3. On Windows, try installing the Thorlabs Optical Power Monitor software")
    print_and_log("4. On Raspberry Pi, make sure you have the necessary permissions to access USB devices")
    print_and_log("5. Try running this script with administrator/sudo privileges")

print_and_log("\nDiagnostic complete. Check powermeter_diagnostic.log for details.")

if __name__ == "__main__":
    # Keep the console window open on Windows
    if IS_WINDOWS:
        input("\nPress Enter to exit...")
