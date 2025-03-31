import pyvisa
import logging
import sys

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('powermeter_detection.log')
    ]
)

def find_thorlabs_devices():
    """Find all Thorlabs devices connected to the system."""
    logging.info("Searching for Thorlabs power meters...")
    
    # Try with default backend
    try:
        logging.info("Trying default VISA backend...")
        rm = pyvisa.ResourceManager()
        backend_type = "Default"
    except ValueError:
        # If that fails, try with the py backend
        try:
            logging.info("Trying PyVISA-py backend...")
            rm = pyvisa.ResourceManager('@py')
            backend_type = "PyVISA-py"
        except Exception as e:
            logging.error(f"Failed to initialize PyVISA: {e}")
            return
    
    logging.info(f"Using {backend_type} backend")
    
    # List all available resources
    try:
        resources = rm.list_resources()
        logging.info(f"Found {len(resources)} resources: {resources}")
    except Exception as e:
        logging.error(f"Failed to list resources: {e}")
        return
    
    # Common Thorlabs identifiers
    thorlabs_identifiers = [
        "USB0::0x1313::0x8078",  # Common PM100D identifier
        "USB0::0x1313::0x8072",  # Alternative identifier
        "USB0::0x1313::0x8070",  # PM100A identifier
        "USB::0x1313",           # Generic Thorlabs identifier
        "ASRL"                   # Serial port (some PM100D use this)
    ]
    
    # Check each resource
    found_devices = []
    for resource in resources:
        # Check if it matches any Thorlabs identifier
        is_thorlabs = False
        for identifier in thorlabs_identifiers:
            if identifier in resource:
                is_thorlabs = True
                break
        
        if is_thorlabs:
            logging.info(f"Potential Thorlabs device found: {resource}")
            try:
                # Try to open the device and get its identity
                device = rm.open_resource(resource, timeout=1000)
                try:
                    idn = device.query("*IDN?").strip()
                    logging.info(f"Device identity: {idn}")
                    found_devices.append((resource, idn))
                except Exception as e:
                    logging.warning(f"Could not query device identity for {resource}: {e}")
                    found_devices.append((resource, "Unknown (could not query identity)"))
                finally:
                    device.close()
            except Exception as e:
                logging.warning(f"Could not open resource {resource}: {e}")
                found_devices.append((resource, f"Unknown (could not open: {e})"))
    
    # Try some common addresses even if they weren't listed
    common_addresses = [
        "USB0::0x1313::0x8078::P0000000::INSTR",
        "USB0::0x1313::0x8078::P0005750::INSTR",
        "USB0::0x1313::0x8078::P0000001::INSTR",
        "ASRL1::INSTR",
        "ASRL2::INSTR"
    ]
    
    for addr in common_addresses:
        if addr not in resources:
            logging.info(f"Trying common address: {addr}")
            try:
                device = rm.open_resource(addr, timeout=1000)
                try:
                    idn = device.query("*IDN?").strip()
                    logging.info(f"Device found at {addr}: {idn}")
                    found_devices.append((addr, idn))
                except Exception as e:
                    logging.debug(f"Could not query device at {addr}: {e}")
                finally:
                    device.close()
            except Exception as e:
                logging.debug(f"Could not open resource {addr}: {e}")
    
    return found_devices

def main():
    """Main function to find and display Thorlabs devices."""
    print("Searching for Thorlabs power meters...")
    print("This may take a few moments...")
    print("Check the powermeter_detection.log file for detailed information.")
    print("-" * 50)
    
    devices = find_thorlabs_devices()
    
    if not devices:
        print("No Thorlabs power meters found.")
        print("Please check that your device is connected and powered on.")
        print("Also ensure that the appropriate drivers are installed.")
        return
    
    print(f"Found {len(devices)} potential Thorlabs devices:")
    print("-" * 50)
    
    for i, (resource, identity) in enumerate(devices, 1):
        print(f"Device {i}:")
        print(f"  Resource name: {resource}")
        print(f"  Identity: {identity}")
        print("-" * 50)
    
    print("\nTo use a specific device in your code, update the PowerMeter initialization in sony_pm100d.py:")
    print('power_meter = PowerMeter(resource_name="<resource_name>", simulation_mode=False)')
    print("\nReplace <resource_name> with the resource name of your device.")

if __name__ == "__main__":
    main()
