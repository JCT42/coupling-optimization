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
import threading
import cv2

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

# For Raspberry Pi, we may need to set the display environment variable
if IS_RASPBERRY_PI:
    # Check if DISPLAY is set
    if "DISPLAY" not in os.environ:
        os.environ["DISPLAY"] = ":0"  # Set to default display
    logging.info(f"Running on Raspberry Pi, DISPLAY set to: {os.environ.get('DISPLAY')}")

# For SLM control - assuming a basic interface
# You may need to modify this based on your specific SLM model
class SLM:
    def __init__(self, resolution: Tuple[int, int] = (800, 600), simulation_mode: bool = False, display_position: int = 1280):
        """
        Initialize the SLM with given resolution.
        
        Args:
            resolution: Tuple of (width, height) for the SLM display, defaults to (800, 600)
            simulation_mode: If True, run in simulation mode without hardware
            display_position: X position to display the pattern window
        """
        # Always use 800x600 for the SLM
        self.resolution = (800, 600)  # width=800, height=600
        self.phase_mask = np.zeros((800, 600), dtype=np.uint8)  # width=800, height=600
        self.connected = False
        self.simulation_mode = simulation_mode
        self.display_position = display_position
        self.window_name = "SLM Phase Mask"
        self.display_window_created = False
        
        # Initialize the SLM
        self.initialize()
        
    def initialize(self):
        """Connect to the SLM hardware."""
        try:
            if self.simulation_mode:
                self.connected = True
                logging.info("SLM initialized in simulation mode")
                
                # Create a window for displaying the phase mask using OpenCV
                self._create_display_window()
                
                # Display initial blank phase mask
                self._update_display()
                
                return
                
            # Placeholder for actual SLM initialization code
            # This would typically involve connecting to the SLM via its API
            # For example: self.slm_device = SLMLibrary.connect()
            self.connected = True
            logging.info("SLM initialized successfully")
            
            # Create a window for displaying the phase mask
            self._create_display_window()
            
            # Display initial blank phase mask
            self._update_display()
            
        except Exception as e:
            logging.error(f"Failed to initialize SLM: {e}")
            self.connected = False
    
    def _create_display_window(self):
        """Create a window for displaying the phase mask."""
        try:
            # Create a named window with normal size
            cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
            
            # Resize the window to match the SLM resolution
            cv2.resizeWindow(self.window_name, self.resolution[0], self.resolution[1])
            
            # Move the window to the specified position
            cv2.moveWindow(self.window_name, self.display_position, 0)
            
            # On Raspberry Pi, set the window to stay on top
            if IS_RASPBERRY_PI:
                try:
                    cv2.setWindowProperty(self.window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
                except Exception as e:
                    logging.warning(f"Could not set fullscreen property: {e}")
            
            self.display_window_created = True
            logging.info(f"Created display window: {self.window_name}")
        except Exception as e:
            logging.error(f"Error creating display window: {e}")
            self.display_window_created = False
    
    def _update_display(self):
        """Update the display window with the current phase mask."""
        if not self.display_window_created:
            logging.warning("Display window not created. Cannot update display.")
            return
            
        try:
            # Display the phase mask
            cv2.imshow(self.window_name, self.phase_mask)
            
            # Use a longer waitKey time on Raspberry Pi
            if IS_RASPBERRY_PI:
                cv2.waitKey(10)  # 10ms wait for Raspberry Pi
            else:
                cv2.waitKey(1)  # 1ms wait for other platforms
            
            # Account for 60Hz refresh rate (approximately 16.67ms per frame)
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
            
        # Ensure the phase mask has the correct dimensions (800x600)
        if phase_mask.shape != (800, 600):  # width=800, height=600
            logging.info(f"Resizing phase mask from {phase_mask.shape} to (800, 600)")
            phase_mask = cv2.resize(phase_mask, (800, 600), interpolation=cv2.INTER_LINEAR)
            
        self.phase_mask = phase_mask
        
        # Update the display window
        self._update_display()
        
        if not self.simulation_mode:
            # Placeholder for actual SLM update code
            # For example: self.slm_device.update_display(self.phase_mask)
            pass
        
        return True
    
    def resize_mask(self, mask: np.ndarray) -> np.ndarray:
        """Resize the mask to match SLM resolution (800x600)."""
        # Always resize to 800x600 regardless of self.resolution
        return cv2.resize(mask, (800, 600), interpolation=cv2.INTER_LINEAR)
    
    def get_random_mask(self) -> np.ndarray:
        """Generate a random phase mask."""
        return np.random.randint(0, 256, (800, 600), dtype=np.uint8)  # width=800, height=600
    
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
        
    def close(self):
        """Close the SLM and any open windows."""
        # Close OpenCV window
        if self.display_window_created:
            try:
                cv2.destroyWindow(self.window_name)
                self.display_window_created = False
            except Exception as e:
                logging.error(f"Error closing display window: {e}")
                
        # Add any other cleanup code for the actual SLM hardware here
        self.connected = False
        logging.info("SLM closed")

class PowerMeter:
    def __init__(self, resource_name: Optional[str] = None):
        """
        Initialize the power meter.
        
        Args:
            resource_name: VISA resource name for the power meter
        """
        self.resource_name = resource_name
        self.connected = False
        self.resource_manager = None
        self.device = None
        self.wavelength = 1550  # Default wavelength in nm
        self.averaging_count = 10  # Default averaging count
        
        # Connect to the power meter
        self.connect()
        
    def connect(self):
        """Connect to the power meter."""
        try:
            # Initialize the VISA resource manager
            self.resource_manager = pyvisa.ResourceManager()
            
            # If no resource name is provided, try to find the power meter
            if self.resource_name is None:
                resources = self.resource_manager.list_resources()
                logging.info(f"Available resources: {resources}")
                
                # Look for Thorlabs PM100D
                for resource in resources:
                    if "USB" in resource:
                        try:
                            device = self.resource_manager.open_resource(resource)
                            idn = device.query("*IDN?")
                            if "PM100" in idn:
                                self.resource_name = resource
                                self.device = device
                                logging.info(f"Found power meter at {resource}: {idn}")
                                break
                            else:
                                device.close()
                        except Exception as e:
                            logging.debug(f"Error checking resource {resource}: {e}")
            
            # If we have a resource name, connect to it
            if self.resource_name is not None and self.device is None:
                self.device = self.resource_manager.open_resource(self.resource_name)
                idn = self.device.query("*IDN?")
                logging.info(f"Connected to power meter: {idn}")
            
            # Check if we're connected
            if self.device is not None:
                self.connected = True
                
                # Configure the power meter
                self.configure()
            else:
                logging.warning("No power meter found.")
                
        except Exception as e:
            logging.error(f"Error connecting to power meter: {e}")
            self.connected = False
    
    def configure(self):
        """Configure the power meter."""
        if not self.connected:
            return
            
        try:
            # Set the wavelength
            self.set_wavelength(self.wavelength)
            
            # Set the averaging count
            self.set_averaging(self.averaging_count)
            
            # Set to power measurement mode
            self.device.write("CONF:POW")
            
            # Set auto-range
            self.device.write("POW:RANG:AUTO 1")
            
            logging.info("Power meter configured")
            
        except Exception as e:
            logging.error(f"Error configuring power meter: {e}")
    
    def set_wavelength(self, wavelength: float):
        """
        Set the wavelength for power measurements.
        
        Args:
            wavelength: Wavelength in nm
        """
        if not self.connected:
            return
            
        try:
            # Convert to meters (PM100D uses meters)
            wavelength_m = wavelength * 1e-9
            
            # Set the wavelength
            self.device.write(f"SENS:CORR:WAV {wavelength_m}")
            
            # Store the wavelength
            self.wavelength = wavelength
            
            logging.info(f"Wavelength set to {wavelength} nm")
            
        except Exception as e:
            logging.error(f"Error setting wavelength: {e}")
    
    def set_averaging(self, count: int):
        """
        Set the averaging count for power measurements.
        
        Args:
            count: Number of samples to average
        """
        if not self.connected:
            return
            
        try:
            # Set the averaging count
            self.device.write(f"SENS:AVER:COUN {count}")
            
            # Enable averaging
            self.device.write("SENS:AVER:STAT 1")
            
            # Store the averaging count
            self.averaging_count = count
            
            logging.info(f"Averaging count set to {count}")
            
        except Exception as e:
            logging.error(f"Error setting averaging: {e}")
    
    def get_power(self) -> float:
        """
        Get the current power reading.
        
        Returns:
            Power in watts
        """
        if not self.connected:
            # Return a random value in simulation mode
            return random.uniform(0.0, 1.0) * 1e-3
            
        try:
            # Read the power
            power = float(self.device.query("MEAS:POW?"))
            
            return power
            
        except Exception as e:
            logging.error(f"Error reading power: {e}")
            
            # Try to reconnect
            self.connect()
            
            # Return 0 on error
            return 0.0
    
    def close(self):
        """Close the connection to the power meter."""
        if self.connected:
            try:
                self.device.close()
                logging.info("Power meter connection closed")
            except Exception as e:
                logging.error(f"Error closing power meter connection: {e}")
            
            self.connected = False

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
        power = self.power_meter.get_power()
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