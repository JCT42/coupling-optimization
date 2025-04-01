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
import pyvisa

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
    def __init__(self, resolution: Tuple[int, int] = (800, 600), simulation_mode: bool = False, display_position: int = 1280):
        """
        Initialize the SLM with given resolution.
        
        Args:
            resolution: Tuple of (width, height) for the SLM display
            simulation_mode: If True, run in simulation mode without hardware
            display_position: X position to display the pattern window
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
                
                # Create a window for displaying the phase mask
                cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
                cv2.moveWindow(self.window_name, self.display_position, 0)
                cv2.resizeWindow(self.window_name, self.resolution[0], self.resolution[1])
                
                # Remove window decorations but don't use fullscreen to maintain exact dimensions
                cv2.setWindowProperty(self.window_name, cv2.WND_PROP_AUTOSIZE, cv2.WINDOW_AUTOSIZE)
                
                self.display_window_created = True
                
                # Display initial blank phase mask
                self._update_display()
                
                return
                
            # Placeholder for actual SLM initialization code
            # This would typically involve connecting to the SLM via its API
            # For example: self.slm_device = SLMLibrary.connect()
            self.connected = True
            logging.info("SLM initialized successfully")
            
            # Create a window for displaying the phase mask
            cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
            cv2.moveWindow(self.window_name, self.display_position, 0)
            cv2.resizeWindow(self.window_name, self.resolution[0], self.resolution[1])
            
            # Remove window decorations but don't use fullscreen to maintain exact dimensions
            cv2.setWindowProperty(self.window_name, cv2.WND_PROP_AUTOSIZE, cv2.WINDOW_AUTOSIZE)
            
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
            # Make sure the phase mask has the correct dimensions
            if self.phase_mask.shape != self.resolution:
                logging.warning(f"Phase mask shape {self.phase_mask.shape} doesn't match resolution {self.resolution}. Resizing.")
                self.phase_mask = self.resize_mask(self.phase_mask)
            
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
    def __init__(self, resource_name: Optional[str] = None):
        """
        Initialize the Thorlabs PM100D power meter.
        
        Args:
            resource_name: VISA resource name for the power meter
                          (if None, will attempt to find it automatically)
        """
        self.rm = pyvisa.ResourceManager()
        self.device = None
        self.connected = False
        self.wavelength = 650.0  # Default wavelength in nm
        
        # Print all available resources to help with debugging
        try:
            resources = self.rm.list_resources()
            logging.info(f"Available VISA resources: {resources}")
        except Exception as e:
            logging.error(f"Error listing resources: {e}")
        
        # Try to connect to the PM100D
        try:
            if resource_name:
                self.device = self.rm.open_resource(resource_name)
                self.connected = True
                logging.info(f"Connected to PM100D using provided resource: {resource_name}")
            else:
                # Try multiple resource formats in order of likelihood
                resource_formats = [
                    'USB0::1313::8078::INSTR',  # Format from thorlabs_power_meter.py
                    'USB0::0x1313::0x8078::INSTR',  # Alternative format with hex
                    'USB0::0x1313::0x8078::P0000000::INSTR',  # With serial number
                    'USB0::0x1313::0x8078::P0005750::INSTR',  # Alternative serial
                    'ASRL1::INSTR',  # Serial port 1
                    'ASRL2::INSTR'   # Serial port 2
                ]
                
                for fmt in resource_formats:
                    try:
                        logging.info(f"Trying to connect with: {fmt}")
                        self.device = self.rm.open_resource(fmt)
                        self.connected = True
                        logging.info(f"Connected to PM100D using: {fmt}")
                        break
                    except Exception as e:
                        logging.warning(f"Failed to connect with {fmt}: {e}")
                
                if not self.connected:
                    # Try to find a Thorlabs device in the available resources
                    for resource in resources:
                        if '1313' in resource:  # Thorlabs vendor ID
                            try:
                                logging.info(f"Trying to connect with found resource: {resource}")
                                self.device = self.rm.open_resource(resource)
                                self.connected = True
                                logging.info(f"Connected to PM100D using: {resource}")
                                break
                            except Exception as e:
                                logging.warning(f"Failed to connect with {resource}: {e}")
        except Exception as e:
            logging.error(f"Error connecting to device: {e}")
    
    def find_devices(self):
        """Find all connected Thorlabs power meter devices"""
        try:
            devices = []
            
            # Use PyVISA to find devices
            for resource in self.rm.list_resources():
                devices.append({
                    'device': resource,
                    'product_id': 0,  # Not available through PyVISA
                    'model': "Thorlabs Power Meter",  # Will be updated after connection
                    'serial': resource,
                    'resource_name': resource
                })
                
            return devices
        except Exception as e:
            logging.error(f"Error finding devices: {e}")
            return []
    
    def connect(self, device_info):
        """Connect to a specific power meter"""
        try:
            # Try to connect to the device
            self.device = self.rm.open_resource(device_info['resource_name'])
            self.connected = True
            logging.info(f"Connected to {device_info['model']} (SN: {device_info['serial']})")
            return True
        except Exception as e:
            logging.error(f"Error connecting to device: {e}")
            return False
    
    def disconnect(self):
        """Disconnect from the power meter"""
        if self.device:
            self.device.close()
            self.connected = False
            logging.info("Disconnected from power meter")
    
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
            # Using PyVISA interface - exactly as in thorlabs_power_meter.py
            power = self.device.query("MEAS:POW?")  # Send query to get power measurement
            return float(power)
        except Exception as e:
            logging.error(f"Error reading power: {e}")
            return -1
    
    def set_wavelength(self, wavelength: float):
        """
        Set the wavelength for power correction.
        
        Args:
            wavelength: Wavelength in nanometers
        """
        if not self.connected:
            logging.warning("Device not connected")
            return False
            
        try:
            self.device.write(f"WAV {wavelength}")  # Command to set wavelength
            self.wavelength = wavelength
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
        """
        if not self.connected:
            logging.warning("Device not connected")
            return False
            
        try:
            # Try to set averaging using the standard SCPI command
            self.device.write(f"SENS:AVER:COUN {count}")
            logging.info(f"Averaging set to {count}")
            return True
        except Exception as e:
            logging.warning(f"Error setting averaging: {e}")
            # If the command fails, just log a warning but don't fail
            return False
    
    def set_auto_range(self, enabled: bool = True):
        """
        Enable or disable auto-ranging.
        
        Args:
            enabled: True to enable auto-ranging, False to disable
        """
        if not self.connected:
            logging.warning("Device not connected")
            return False
            
        try:
            # Try to set auto-range using the standard SCPI command
            self.device.write(f"SENS:POW:DC:RANG:AUTO {1 if enabled else 0}")
            logging.info(f"Auto-range set to {enabled}")
            return True
        except Exception as e:
            logging.warning(f"Error setting auto-range: {e}")
            # If the command fails, just log a warning but don't fail
            return False


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
            logging.FileHandler('coupling_optimization.log')
        ]
    )
    
    logging.info("Starting coupling optimization")
    
    # Initialize the SLM
    slm = SLM(resolution=(800, 600), simulation_mode=False, display_position=1280)  # Adjust resolution to match your SLM
    
    if not slm.connected:
        logging.error("Failed to connect to SLM. Exiting.")
        return
    
    # Initialize the power meter with the exact resource name found on the system
    power_meter = PowerMeter(resource_name="USB0::0x1313::0x8078::P0043233::INSTR")
    
    if not power_meter.connected:
        logging.error("Failed to connect to power meter. Exiting.")
        return
    
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
    
    # Define a callback function for live updates
    def callback(iteration, temperature, current_power, best_power):
        if iteration % 10 == 0:  # Update every 10 iterations
            print(f"Iteration: {iteration}, Temp: {temperature:.4f}, "
                  f"Current: {current_power:.6f} W, Best: {best_power:.6f} W")
    
    # Run the optimization with live plotting
    import threading
    
    # Start the live plot in a separate thread
    plot_thread = threading.Thread(target=live_optimization_plot, args=(optimizer,))
    plot_thread.daemon = True
    plot_thread.start()
    
    # Run the optimization
    best_mask, best_power, power_history = optimizer.run_optimization(callback=callback)
    
    # Print the results
    logging.info(f"Optimization complete")
    logging.info(f"Best power: {best_power:.6f} W")
    
    # Save the best mask
    np.save('best_phase_mask.npy', best_mask)
    logging.info("Best phase mask saved to best_phase_mask.npy")
    
    # Plot the final results
    optimizer.plot_progress(save_path='optimization_results.png')
    
    logging.info("Optimization results saved to optimization_results.png")


if __name__ == "__main__":
    main()