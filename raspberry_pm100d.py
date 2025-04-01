"""
Raspberry Pi Compatible Coupling Optimization
This module provides interfaces for SLM control and power meter reading on Raspberry Pi.
"""

import numpy as np
import time
import random
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import sys
import os
import platform
import subprocess
from typing import Tuple, List, Optional, Callable
import logging
import tkinter as tk
import usb.core
import usb.util
import struct

# Determine if we're on Raspberry Pi
IS_RASPBERRY_PI = platform.system() == 'Linux' and os.path.exists('/sys/firmware/devicetree/base/model') and 'Raspberry Pi' in open('/sys/firmware/devicetree/base/model').read()

# For SLM display
try:
    import cv2
except ImportError:
    print("OpenCV not found. Installing...")
    import pip
    pip.main(['install', 'opencv-python'])
    import cv2

# For SLM control - assuming a basic interface
# You may need to modify this based on your specific SLM model
class SLM:
    def __init__(self, resolution: Tuple[int, int] = (800, 600), simulation_mode: bool = False, display_position: int = 0):
        """
        Initialize the SLM with given resolution.
        
        Args:
            resolution: Tuple of (width, height) for the SLM display
            simulation_mode: If True, run in simulation mode without hardware
            display_position: X position to display the pattern window (0 for primary display on Raspberry Pi)
        """
        self.resolution = resolution
        self.phase_mask = np.zeros(resolution, dtype=np.uint8)
        self.connected = False
        self.simulation_mode = simulation_mode
        self.display_position = display_position
        self.window_name = "SLM Phase Mask"
        self.display_window_created = False
        self.initialize()
        
    def initialize(self):
        """Connect to the SLM hardware."""
        try:
            if self.simulation_mode:
                self.connected = True
                logging.info("SLM initialized in simulation mode")
                
                # Create a frameless window for displaying the phase mask
                cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL | cv2.WINDOW_GUI_NORMAL)
                cv2.setWindowProperty(self.window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
                cv2.resizeWindow(self.window_name, self.resolution[0], self.resolution[1])
                
                # On Raspberry Pi, we don't need to move the window as we'll use the primary display
                if not IS_RASPBERRY_PI:
                    cv2.moveWindow(self.window_name, self.display_position, 0)
                
                self.display_window_created = True
                
                # Display initial blank phase mask
                self._update_display()
                
                return
                
            # Placeholder for actual SLM initialization code
            # This would typically involve connecting to the SLM via its API
            # For example: self.slm_device = SLMLibrary.connect()
            self.connected = True
            logging.info("SLM initialized successfully")
            
            # Create a frameless window for displaying the phase mask
            cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL | cv2.WINDOW_GUI_NORMAL)
            cv2.setWindowProperty(self.window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
            cv2.resizeWindow(self.window_name, self.resolution[0], self.resolution[1])
            
            # On Raspberry Pi, we don't need to move the window as we'll use the primary display
            if not IS_RASPBERRY_PI:
                cv2.moveWindow(self.window_name, self.display_position, 0)
                
            self.display_window_created = True
            
            # Display initial blank phase mask
            self._update_display()
            
        except Exception as e:
            logging.error(f"Failed to initialize SLM: {e}")
            self.connected = False
    
    def _update_display(self):
        """Update the display window with the current phase mask."""
        if not self.display_window_created:
            return
            
        try:
            # Display the phase mask
            cv2.imshow(self.window_name, self.phase_mask)
            cv2.waitKey(1)  # Update the window (1ms wait)
            
            # Account for 60Hz refresh rate (approximately 16.67ms per frame)
            # Wait for at least one refresh cycle to ensure the display is updated
            time.sleep(0.02)  # 20ms, slightly longer than one refresh cycle at 60Hz
        except Exception as e:
            logging.error(f"Error updating display: {e}")
    
    def apply_phase_mask(self, phase_mask: np.ndarray):
        """
        Apply a phase mask to the SLM.
        
        Args:
            phase_mask: 2D numpy array with phase values (0-255)
        """
        if not self.connected:
            logging.warning("SLM not connected. Cannot apply phase mask.")
            return False
            
        # Ensure the phase mask has the correct dimensions
        if phase_mask.shape != self.resolution:
            phase_mask = self.resize_mask(phase_mask)
            
        self.phase_mask = phase_mask
        
        # Update the display window
        self._update_display()
        
        if not self.simulation_mode:
            # Placeholder for actual SLM update code
            # For example: self.slm_device.update_display(self.phase_mask)
            pass
        
        return True
    
    def resize_mask(self, mask: np.ndarray) -> np.ndarray:
        """Resize the mask to match SLM resolution."""
        from scipy.ndimage import zoom
        
        # Calculate zoom factors
        zoom_factors = (self.resolution[0] / mask.shape[0], 
                        self.resolution[1] / mask.shape[1])
        
        # Resize the mask
        resized_mask = zoom(mask, zoom_factors, order=1)
        
        return resized_mask
    
    def get_random_mask(self) -> np.ndarray:
        """Generate a random phase mask."""
        return np.random.randint(0, 256, self.resolution, dtype=np.uint8)
    
    def get_zernike_mask(self, coefficients: List[float]) -> np.ndarray:
        """
        Generate a phase mask based on Zernike polynomials.
        
        Args:
            coefficients: List of coefficients for Zernike polynomials
            
        Returns:
            2D numpy array with phase values
        """
        # Placeholder for Zernike polynomial implementation
        # This would typically involve calculating Zernike modes and combining them
        # For simplicity, we'll just return a random mask for now
        return self.get_random_mask()
    
    def perturb_mask(self, mask: np.ndarray, magnitude: float = 0.1) -> np.ndarray:
        """
        Create a slightly modified version of the given mask.
        
        Args:
            mask: The original phase mask
            magnitude: The magnitude of the perturbation (0-1)
            
        Returns:
            A perturbed version of the mask
        """
        # Scale the magnitude to a reasonable range (0-255 * magnitude)
        scale = 255 * magnitude
        
        # Generate random perturbations
        perturbation = np.random.normal(0, scale, mask.shape)
        
        # Apply perturbations and clip to valid range
        new_mask = np.clip(mask + perturbation, 0, 255).astype(np.uint8)
        
        return new_mask
        
    def close(self):
        """Close the SLM and any open windows."""
        if self.display_window_created:
            try:
                cv2.destroyWindow(self.window_name)
                self.display_window_created = False
            except Exception as e:
                logging.error(f"Error closing display window: {e}")
                
        # Add any other cleanup code for the actual SLM hardware here
        self.connected = False
        logging.info("SLM closed")
    
    def save_pattern(self, filename: str, save_npy: bool = True, save_image: bool = True):
        """
        Save the current phase mask to a file.
        
        Args:
            filename: Base filename to save to (without extension)
            save_npy: Whether to save as a NumPy array (.npy)
            save_image: Whether to save as an image (.png)
        
        Returns:
            List of saved filenames
        """
        saved_files = []
        
        if not self.connected:
            logging.warning("SLM not connected. Cannot save pattern.")
            return saved_files
        
        try:
            # Save as NumPy array
            if save_npy:
                npy_filename = f"{filename}.npy"
                np.save(npy_filename, self.phase_mask)
                saved_files.append(npy_filename)
                logging.info(f"Saved phase mask to {npy_filename}")
            
            # Save as image
            if save_image:
                img_filename = f"{filename}.png"
                cv2.imwrite(img_filename, self.phase_mask)
                saved_files.append(img_filename)
                logging.info(f"Saved phase mask image to {img_filename}")
                
                # Also save a colorized version for better visualization
                colorized = cv2.applyColorMap(self.phase_mask, cv2.COLORMAP_JET)
                color_img_filename = f"{filename}_colorized.png"
                cv2.imwrite(color_img_filename, colorized)
                saved_files.append(color_img_filename)
                logging.info(f"Saved colorized phase mask to {color_img_filename}")
            
            return saved_files
        
        except Exception as e:
            logging.error(f"Error saving phase mask: {e}")
            return saved_files


class PowerMeter:
    def __init__(self, vendor_id: int = 0x1313, product_id: int = 0x8078):
        """
        Initialize the Thorlabs power meter using direct USB communication.
        This is optimized for Raspberry Pi where PyVISA might not work well.
        
        Args:
            vendor_id: USB vendor ID for Thorlabs (default: 0x1313)
            product_id: USB product ID for the power meter (default: 0x8078 for PM100D)
        """
        self.vendor_id = vendor_id
        self.product_id = product_id
        self.device = None
        self.connected = False
        self.wavelength = 650.0  # Default wavelength in nm
        
        # Low-pass filter parameters
        self.filter_enabled = True
        self.filter_window_size = 5  # Number of samples to average
        self.filter_buffer = []  # Buffer for recent power readings
        self.filter_alpha = 0.2  # Alpha for exponential moving average (alternative filter)
        self.last_filtered_value = None  # Last filtered value for exponential filter
        
        # Simulation parameters for testing without hardware
        self.simulation_mode = False
        self.simulation_base_power = 1.0e-3  # 1 mW
        self.simulation_noise = 0.02  # 2% noise
        self.simulation_drift = 0.0001  # 0.01% drift per second
        self.simulation_start_time = time.time()
        
        # Try to connect to the power meter
        self.connect()
    
    def connect(self, device_info=None):
        """
        Connect to the power meter using PyUSB.
        
        Args:
            device_info: Optional dictionary with device information
                        If provided, will use the device_info to connect
        
        Returns:
            True if connected successfully, False otherwise
        """
        if self.connected:
            return True
            
        try:
            # If in simulation mode, just return success
            if self.simulation_mode:
                self.connected = True
                logging.info("Connected to simulated power meter")
                return True
                
            # If device_info is provided, use it to connect
            if device_info and 'device' in device_info:
                self.device = device_info['device']
                self.connected = True
                logging.info(f"Connected to {device_info.get('model', 'Thorlabs Power Meter')} (SN: {device_info.get('serial', 'Unknown')})")
                return True
            
            # Find the device by vendor and product ID
            self.device = usb.core.find(idVendor=self.vendor_id, idProduct=self.product_id)
            
            if self.device is None:
                logging.warning(f"No Thorlabs power meter found with VID:PID {hex(self.vendor_id)}:{hex(self.product_id)}")
                return False
            
            # Set the active configuration. With no arguments, the first configuration will be used
            try:
                self.device.set_configuration()
            except usb.core.USBError as e:
                # On Linux, we might need to detach the kernel driver first
                if IS_RASPBERRY_PI or platform.system() == 'Linux':
                    for cfg in self.device:
                        for intf in cfg:
                            if self.device.is_kernel_driver_active(intf.bInterfaceNumber):
                                try:
                                    self.device.detach_kernel_driver(intf.bInterfaceNumber)
                                except usb.core.USBError as e2:
                                    logging.warning(f"Could not detach kernel driver: {e2}")
                    # Try again to set configuration
                    self.device.set_configuration()
            
            # Get the first interface
            cfg = self.device.get_active_configuration()
            interface_number = cfg[(0, 0)].bInterfaceNumber
            
            # Claim the interface
            usb.util.claim_interface(self.device, interface_number)
            
            # Find the endpoints
            ep_out = None
            ep_in = None
            
            for ep in cfg[(0, 0)]:
                if ep.bEndpointAddress & 0x80:  # Direction IN
                    ep_in = ep
                else:  # Direction OUT
                    ep_out = ep
            
            if not ep_in or not ep_out:
                logging.error("Could not find endpoints")
                return False
                
            self.ep_in = ep_in
            self.ep_out = ep_out
            self.interface_number = interface_number
            
            self.connected = True
            logging.info(f"Connected to Thorlabs power meter (VID:PID {hex(self.vendor_id)}:{hex(self.product_id)})")
            
            # Initialize the device with default settings
            self.set_wavelength(self.wavelength)
            
            return True
            
        except Exception as e:
            logging.error(f"Error connecting to power meter: {e}")
            return False
    
    def disconnect(self):
        """Disconnect from the power meter"""
        if not self.connected:
            return True
            
        try:
            if self.simulation_mode:
                self.connected = False
                logging.info("Disconnected from simulated power meter")
                return True
                
            # Release the interface
            usb.util.release_interface(self.device, self.interface_number)
            
            # On Linux, we might want to reattach the kernel driver
            if IS_RASPBERRY_PI or platform.system() == 'Linux':
                try:
                    self.device.attach_kernel_driver(self.interface_number)
                except usb.core.USBError as e:
                    logging.warning(f"Could not reattach kernel driver: {e}")
            
            self.connected = False
            logging.info("Disconnected from power meter")
            return True
            
        except Exception as e:
            logging.error(f"Error disconnecting from power meter: {e}")
            return False
    
    def find_devices(self):
        """
        Find all connected Thorlabs power meter devices.
        
        Returns:
            List of dictionaries with device information
        """
        devices = []
        
        try:
            # Find all devices with Thorlabs vendor ID
            found_devices = usb.core.find(idVendor=self.vendor_id, find_all=True)
            
            for dev in found_devices:
                try:
                    # Get device information
                    product_id = dev.idProduct
                    manufacturer = usb.util.get_string(dev, dev.iManufacturer) if dev.iManufacturer else "Unknown"
                    product = usb.util.get_string(dev, dev.iProduct) if dev.iProduct else "Unknown"
                    serial = usb.util.get_string(dev, dev.iSerialNumber) if dev.iSerialNumber else "Unknown"
                    
                    # Add to the list
                    devices.append({
                        'device': dev,
                        'vendor_id': self.vendor_id,
                        'product_id': product_id,
                        'model': product,
                        'manufacturer': manufacturer,
                        'serial': serial
                    })
                except Exception as e:
                    logging.warning(f"Error getting device information: {e}")
            
            return devices
            
        except Exception as e:
            logging.error(f"Error finding devices: {e}")
            return []
    
    def _send_command(self, command):
        """
        Send a SCPI command to the power meter.
        
        Args:
            command: SCPI command string
        
        Returns:
            True if successful, False otherwise
        """
        if not self.connected or self.simulation_mode:
            return False
            
        try:
            # Add termination character if not present
            if not command.endswith('\n'):
                command += '\n'
                
            # Convert to bytes and send
            self.ep_out.write(command.encode('ascii'))
            return True
            
        except Exception as e:
            logging.error(f"Error sending command: {e}")
            return False
    
    def _query(self, command):
        """
        Send a query command and read the response.
        
        Args:
            command: SCPI query command string
        
        Returns:
            Response string or None if error
        """
        if not self.connected:
            return None
            
        if self.simulation_mode:
            # Simulate responses for common queries
            if command.strip().upper() == "MEAS:POW?":
                return str(self._simulate_power_reading())
            elif command.startswith("WAV?"):
                return str(self.wavelength)
            else:
                return "0.0"  # Default response
        
        try:
            # Send the command
            if not self._send_command(command):
                return None
                
            # Read the response (with timeout)
            response = self.ep_in.read(64, timeout=1000).tobytes().decode('ascii').strip()
            return response
            
        except Exception as e:
            logging.error(f"Error querying device: {e}")
            return None
    
    def _simulate_power_reading(self):
        """
        Simulate a power reading for testing without hardware.
        
        Returns:
            Simulated power value in watts
        """
        # Calculate time-based drift
        elapsed_time = time.time() - self.simulation_start_time
        drift = self.simulation_drift * elapsed_time
        
        # Add random noise
        noise = random.uniform(-self.simulation_noise, self.simulation_noise) * self.simulation_base_power
        
        # Calculate power with drift and noise
        power = self.simulation_base_power * (1.0 + drift + noise)
        
        return power
    
    def read_power(self) -> float:
        """
        Read the current power measurement.
        
        Returns:
            Power reading in watts, or -1 if error
        """
        if not self.connected:
            logging.warning("Device not connected")
            return -1
            
        try:
            if self.simulation_mode:
                power_value = self._simulate_power_reading()
            else:
                # Send the measurement query
                response = self._query("MEAS:POW?")
                
                if response is None:
                    return -1
                    
                power_value = float(response)
            
            # Apply low-pass filtering if enabled
            if self.filter_enabled:
                power_value = self._apply_filter(power_value)
                
            return power_value
            
        except Exception as e:
            logging.error(f"Error reading power: {e}")
            return -1
    
    def _apply_filter(self, power_value: float) -> float:
        """
        Apply a low-pass filter to the power reading.
        
        Args:
            power_value: The raw power reading
            
        Returns:
            Filtered power value
        """
        # Moving average filter
        self.filter_buffer.append(power_value)
        
        # Keep buffer at the specified window size
        if len(self.filter_buffer) > self.filter_window_size:
            self.filter_buffer.pop(0)
        
        # Calculate the average
        filtered_value = sum(self.filter_buffer) / len(self.filter_buffer)
        
        return filtered_value
    
    def set_filter_params(self, enabled: bool = True, window_size: int = 5, alpha: float = 0.2):
        """
        Set the parameters for the low-pass filter.
        
        Args:
            enabled: Whether to enable filtering
            window_size: Number of samples to use in the moving average
            alpha: Weight for the exponential moving average (0-1)
                  Lower values give more weight to past readings (smoother)
        """
        self.filter_enabled = enabled
        
        # Update window size if changed
        if window_size != self.filter_window_size:
            self.filter_window_size = max(1, window_size)  # Ensure at least 1
            self.filter_buffer = self.filter_buffer[-self.filter_window_size:] if self.filter_buffer else []
        
        # Update alpha for exponential filter
        self.filter_alpha = max(0.0, min(1.0, alpha))  # Clamp between 0 and 1
        
        logging.info(f"Power meter filter: enabled={enabled}, window_size={window_size}, alpha={alpha}")
    
    def set_wavelength(self, wavelength: float):
        """
        Set the wavelength for power correction.
        
        Args:
            wavelength: Wavelength in nanometers
        
        Returns:
            True if successful, False otherwise
        """
        if not self.connected:
            logging.warning("Device not connected")
            return False
            
        try:
            self.wavelength = wavelength
            
            if not self.simulation_mode:
                # Send the wavelength command
                return self._send_command(f"WAV {wavelength}")
            
            logging.info(f"Wavelength set to {wavelength} nm")
            return True
            
        except Exception as e:
            logging.error(f"Error setting wavelength: {e}")
            return False
    
    def set_averaging(self, count: int):
        """
        Set the number of measurements to average.
        
        Args:
            count: Number of measurements to average (1-10000)
        
        Returns:
            True if successful, False otherwise
        """
        if not self.connected:
            logging.warning("Device not connected")
            return False
            
        try:
            if not self.simulation_mode:
                # Send the averaging command
                return self._send_command(f"SENS:AVER:COUN {count}")
            
            logging.info(f"Averaging set to {count}")
            return True
            
        except Exception as e:
            logging.warning(f"Error setting averaging: {e}")
            return False
    
    def set_auto_range(self, enabled: bool = True):
        """
        Enable or disable auto-ranging.
        
        Args:
            enabled: True to enable auto-ranging, False to disable
        
        Returns:
            True if successful, False otherwise
        """
        if not self.connected:
            logging.warning("Device not connected")
            return False
            
        try:
            if not self.simulation_mode:
                # Send the auto-range command
                return self._send_command(f"SENS:POW:DC:RANG:AUTO {1 if enabled else 0}")
            
            logging.info(f"Auto-range set to {enabled}")
            return True
            
        except Exception as e:
            logging.warning(f"Error setting auto-range: {e}")
            return False
    
    def set_simulation_mode(self, enabled: bool = True, base_power: float = 1.0e-3, noise: float = 0.02, drift: float = 0.0001):
        """
        Enable or disable simulation mode for testing without hardware.
        
        Args:
            enabled: True to enable simulation mode, False to disable
            base_power: Base power level for simulation (watts)
            noise: Noise level as a fraction of base power
            drift: Drift rate per second as a fraction of base power
        
        Returns:
            True if successful, False otherwise
        """
        self.simulation_mode = enabled
        
        if enabled:
            self.simulation_base_power = base_power
            self.simulation_noise = noise
            self.simulation_drift = drift
            self.simulation_start_time = time.time()
            self.connected = True
            logging.info(f"Simulation mode enabled: base_power={base_power}W, noise={noise*100}%, drift={drift*100}%/s")
        else:
            # Try to connect to real hardware
            self.connected = False
            self.connect()
            
        return True

class SimulatedAnnealing:
    def __init__(self, 
                 slm: SLM, 
                 power_meter: PowerMeter,
                 initial_temperature: float = 100.0,
                 cooling_rate: float = 0.95,
                 iterations_per_temp: int = 10,
                 min_temperature: float = 0.1,
                 perturbation_scale: float = 0.1):
        """
        Initialize the simulated annealing optimizer.
        
        Args:
            slm: SLM object for controlling the spatial light modulator
            power_meter: PowerMeter object for reading power measurements
            initial_temperature: Starting temperature for annealing
            cooling_rate: Rate at which temperature decreases (0-1)
            iterations_per_temp: Number of iterations at each temperature
            min_temperature: Stopping temperature
            perturbation_scale: Scale of perturbations to the phase mask (0-1)
        """
        self.slm = slm
        self.power_meter = power_meter
        self.initial_temperature = initial_temperature
        self.cooling_rate = cooling_rate
        self.iterations_per_temp = iterations_per_temp
        self.min_temperature = min_temperature
        self.perturbation_scale = perturbation_scale
        
        # For tracking progress
        self.best_mask = None
        self.best_power = -float('inf')
        self.power_history = []
        self.temperature_history = []
        
        # Flag to control stopping the optimization
        self.stop_requested = False
    
    def objective_function(self) -> float:
        """
        Measure the current power as the objective function.
        Higher power means better coupling efficiency.
        
        Returns:
            Power reading in watts
        """
        power = self.power_meter.read_power()
        return power
    
    def stop(self):
        """Request to stop the optimization process."""
        self.stop_requested = True
        logging.info("Stop requested for optimization")
    
    def run_optimization(self, 
                         initial_mask: Optional[np.ndarray] = None,
                         callback: Optional[Callable] = None) -> Tuple[np.ndarray, float, List[float]]:
        """
        Run the simulated annealing optimization.
        
        Args:
            initial_mask: Starting phase mask (if None, a random mask is used)
            callback: Optional callback function called after each iteration
            
        Returns:
            Tuple of (best_mask, best_power, power_history)
        """
        # Reset the stop flag
        self.stop_requested = False
        
        # Reset tracking variables
        self.power_history = []
        self.temperature_history = []
        self.best_power = -float('inf')
        self.best_mask = None
        
        # Initialize temperature
        temperature = self.initial_temperature
        
        # Create an initial solution if not provided
        if initial_mask is None:
            current_mask = self.slm.get_random_mask()
        else:
            current_mask = initial_mask.copy()
        
        # Apply the initial mask
        self.slm.apply_phase_mask(current_mask)
        
        # Evaluate the initial solution
        current_power = self.objective_function()
        
        # Initialize the best solution
        self.best_mask = current_mask.copy()
        self.best_power = current_power
        
        # Add the initial power to the history
        self.power_history.append(current_power)
        
        # Main simulated annealing loop
        iteration = 0
        
        while temperature > self.min_temperature and not self.stop_requested:
            # Record the current temperature
            self.temperature_history.append(temperature)
            
            # Multiple iterations at each temperature
            for i in range(self.iterations_per_temp):
                # Check if stop was requested
                if self.stop_requested:
                    logging.info("Optimization stopped by user")
                    break
                    
                iteration += 1
                
                # Generate a neighboring solution by perturbing the current mask
                neighbor_mask = self.slm.perturb_mask(current_mask, self.perturbation_scale)
                
                # Apply the new mask
                self.slm.apply_phase_mask(neighbor_mask)
                
                # Evaluate the new solution
                neighbor_power = self.objective_function()
                
                # Calculate the change in power
                delta_power = neighbor_power - current_power
                
                # Decide whether to accept the new solution
                if delta_power > 0:  # Better solution, always accept
                    current_mask = neighbor_mask.copy()
                    current_power = neighbor_power
                    
                    # Update the best solution if needed
                    if current_power > self.best_power:
                        self.best_mask = current_mask.copy()
                        self.best_power = current_power
                        logging.info(f"New best power: {self.best_power:.6f} W")
                else:
                    # Worse solution, accept with probability based on temperature
                    acceptance_probability = np.exp(delta_power / temperature)
                    
                    if random.random() < acceptance_probability:
                        current_mask = neighbor_mask.copy()
                        current_power = neighbor_power
                        logging.info(f"Accepted worse solution with probability {acceptance_probability:.4f}")
                
                # Record the current power
                self.power_history.append(current_power)
                
                # Call the callback function if provided
                if callback is not None:
                    callback(iteration, temperature, current_power, self.best_power)
            
            # Cool down
            temperature *= self.cooling_rate
            logging.info(f"Temperature: {temperature:.4f}, Best power: {self.best_power:.6f} W")
        
        # Ensure the best mask is applied at the end
        self.slm.apply_phase_mask(self.best_mask)
        
        # Log the reason for stopping
        if self.stop_requested:
            logging.info("Optimization stopped by user request")
        else:
            logging.info("Optimization completed normally (reached minimum temperature)")
        
        return self.best_mask, self.best_power, self.power_history
    
    def plot_progress(self, save_path: Optional[str] = None):
        """
        Plot the optimization progress.
        
        Args:
            save_path: Optional path to save the plot
        """
        plt.figure(figsize=(12, 8))
        
        # Plot power history
        plt.subplot(2, 1, 1)
        plt.plot(self.power_history)
        plt.title('Power vs. Iteration')
        plt.xlabel('Iteration')
        plt.ylabel('Power (W)')
        plt.grid(True)
        
        # Plot temperature history
        plt.subplot(2, 1, 2)
        iterations_per_temp_point = len(self.power_history) // len(self.temperature_history)
        x_temp = [i * iterations_per_temp_point for i in range(len(self.temperature_history))]
        plt.plot(x_temp, self.temperature_history)
        plt.title('Temperature vs. Iteration')
        plt.xlabel('Iteration')
        plt.ylabel('Temperature')
        plt.grid(True)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path)
        
        plt.show()


def live_optimization_plot(optimizer):
    """
    Create a live plot of the optimization progress.
    
    Args:
        optimizer: SimulatedAnnealing object
    """
    # Create the figure and axes
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
    
    # Initialize empty plots
    line1, = ax1.plot([], [], lw=2)
    line2, = ax2.plot([], [], lw=2)
    
    # Set up the axes
    ax1.set_title('Power vs. Iteration')
    ax1.set_xlabel('Iteration')
    ax1.set_ylabel('Power (W)')
    ax1.grid(True)
    
    ax2.set_title('Temperature vs. Iteration')
    ax2.set_xlabel('Iteration')
    ax2.set_ylabel('Temperature')
    ax2.grid(True)
    
    # Function to update the plot
    def update(frame):
        # Update power history plot
        x1 = list(range(len(optimizer.power_history)))
        y1 = optimizer.power_history
        line1.set_data(x1, y1)
        ax1.relim()
        ax1.autoscale_view()
        
        # Update temperature history plot
        if optimizer.temperature_history:
            iterations_per_temp_point = max(1, len(optimizer.power_history) // len(optimizer.temperature_history))
            x2 = [i * iterations_per_temp_point for i in range(len(optimizer.temperature_history))]
            y2 = optimizer.temperature_history
            line2.set_data(x2, y2)
            ax2.relim()
            ax2.autoscale_view()
        
        return line1, line2
    
    # Create the animation
    ani = FuncAnimation(fig, update, interval=500, blit=True)
    
    plt.tight_layout()
    plt.show()


def main():
    """Main function to run the coupling optimization."""
    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler('raspberry_coupling_optimization.log')
        ]
    )
    
    logging.info("Starting Raspberry Pi coupling optimization")
    
    # Check if running on Raspberry Pi
    if IS_RASPBERRY_PI:
        logging.info("Running on Raspberry Pi")
        display_position = 0  # Use primary display on Raspberry Pi
    else:
        logging.info("Not running on Raspberry Pi, using simulation mode")
        display_position = 1280  # Secondary display on non-Pi systems
    
    # Initialize the SLM
    slm = SLM(resolution=(800, 600), simulation_mode=False, display_position=display_position)
    
    if not slm.connected:
        logging.error("Failed to connect to SLM. Exiting.")
        return
    
    # Initialize the power meter with direct USB communication
    power_meter = PowerMeter()
    
    # If not connected, try simulation mode
    if not power_meter.connected:
        logging.warning("Failed to connect to power meter. Using simulation mode.")
        power_meter.set_simulation_mode(True)
    
    # Initialize the simulated annealing optimizer
    optimizer = SimulatedAnnealing(
        slm=slm,
        power_meter=power_meter,
        initial_temperature=100.0,
        cooling_rate=0.95,
        iterations_per_temp=10,
        min_temperature=0.1,
        perturbation_scale=0.1
    )
    
    # Launch the GUI
    try:
        # Import the GUI module
        from coupling_gui import CouplingOptimizerGUI
        
        # Create the GUI
        root = tk.Tk()
        app = CouplingOptimizerGUI(root, optimizer)
        
        # Start the GUI main loop
        app.run()
        
    except Exception as e:
        logging.error(f"Error launching GUI: {e}")
        
        # Fall back to command-line mode if GUI fails
        run_command_line_mode(optimizer)
    
    # Close the SLM and any open windows
    slm.close()


def run_command_line_mode(optimizer):
    """Run in command-line mode without GUI."""
    logging.info("Running in command-line mode")
    
    # Ask for optimization parameters
    print("\nCoupling Optimization - Command Line Mode")
    print("----------------------------------------")
    
    # Set parameters
    initial_temp = float(input("Initial temperature (default: 100.0): ") or "100.0")
    cooling_rate = float(input("Cooling rate (0-1, default: 0.95): ") or "0.95")
    iterations = int(input("Iterations per temperature (default: 10): ") or "10")
    min_temp = float(input("Minimum temperature (default: 0.1): ") or "0.1")
    perturbation = float(input("Perturbation scale (0-1, default: 0.1): ") or "0.1")
    
    # Update optimizer parameters
    optimizer.initial_temperature = initial_temp
    optimizer.cooling_rate = cooling_rate
    optimizer.iterations_per_temp = iterations
    optimizer.min_temperature = min_temp
    optimizer.perturbation_scale = perturbation
    
    # Start optimization
    print("\nStarting optimization...")
    
    # Run the optimization
    best_mask, best_power, power_history = optimizer.run_optimization()
    
    # Show results
    print(f"\nOptimization complete!")
    print(f"Best power: {best_power:.6f} W")
    
    # Save the best mask
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    filename = f"best_mask_{timestamp}"
    saved_files = optimizer.slm.save_pattern(filename)
    
    print(f"Best mask saved to: {', '.join(saved_files)}")
    
    # Plot the results
    optimizer.plot_progress(f"optimization_progress_{timestamp}.png")


def install_dependencies():
    """Install required dependencies for Raspberry Pi."""
    if not IS_RASPBERRY_PI:
        logging.info("Not running on Raspberry Pi, skipping dependency installation")
        return
        
    logging.info("Installing dependencies for Raspberry Pi")
    
    try:
        # Create a shell script to install dependencies
        script_content = """#!/bin/bash
echo "Installing dependencies for Thorlabs power meter on Raspberry Pi"

# Update package lists
sudo apt-get update

# Install required packages
sudo apt-get install -y python3-pip python3-numpy python3-matplotlib python3-opencv
sudo apt-get install -y python3-usb python3-tk libusb-1.0-0-dev

# Install Python packages
pip3 install pyusb numpy matplotlib opencv-python

# Set up udev rules for Thorlabs devices
echo 'SUBSYSTEM=="usb", ATTRS{idVendor}=="1313", MODE="0666"' | sudo tee /etc/udev/rules.d/99-thorlabs.rules

# Reload udev rules
sudo udevadm control --reload-rules
sudo udevadm trigger

echo "Installation complete!"
"""
        
        # Write the script to a file
        with open("install_thorlabs_pi.sh", "w") as f:
            f.write(script_content)
        
        # Make the script executable
        os.chmod("install_thorlabs_pi.sh", 0o755)
        
        # Run the script
        subprocess.run(["./install_thorlabs_pi.sh"], check=True)
        
        logging.info("Dependencies installed successfully")
        
    except Exception as e:
        logging.error(f"Error installing dependencies: {e}")


if __name__ == "__main__":
    # Check if we need to install dependencies
    if len(sys.argv) > 1 and sys.argv[1] == "--install":
        install_dependencies()
    else:
        main()
