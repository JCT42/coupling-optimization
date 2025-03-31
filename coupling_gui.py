import tkinter as tk
from tkinter import ttk
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
import threading
import time
import cv2
from PIL import Image, ImageTk
import logging

class CouplingOptimizerGUI:
    def __init__(self, root, optimizer=None):
        """
        Initialize the GUI for the coupling optimizer.
        
        Args:
            root: Tkinter root window
            optimizer: SimulatedAnnealing optimizer instance (can be set later)
        """
        self.root = root
        self.root.title("Coupling Optimization")
        self.root.geometry("1200x800")
        self.optimizer = optimizer
        
        # Set up logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.StreamHandler(),
                logging.FileHandler('coupling_optimization.log')
            ]
        )
        
        # Variables for optimization parameters
        self.initial_temperature = tk.DoubleVar(value=100.0)
        self.cooling_rate = tk.DoubleVar(value=0.95)
        self.iterations_per_temp = tk.IntVar(value=10)
        self.min_temperature = tk.DoubleVar(value=0.1)
        self.perturbation_scale = tk.DoubleVar(value=0.1)
        self.wavelength = tk.DoubleVar(value=1550)
        self.averaging_count = tk.IntVar(value=10)
        
        # Variables for status
        self.current_power = tk.DoubleVar(value=0.0)
        self.best_power = tk.DoubleVar(value=0.0)
        self.current_temperature = tk.DoubleVar(value=0.0)
        self.iteration_count = tk.IntVar(value=0)
        self.optimization_running = False
        self.optimization_thread = None
        
        # Create the main layout
        self.create_layout()
        
        # Initialize plots
        self.initialize_plots()
        
        # Update power display periodically
        self.update_power_display()
        
    def create_layout(self):
        """Create the main GUI layout."""
        # Create a notebook with tabs
        notebook = ttk.Notebook(self.root)
        notebook.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Main tab
        main_frame = ttk.Frame(notebook)
        notebook.add(main_frame, text="Main")
        
        # Parameters tab
        params_frame = ttk.Frame(notebook)
        notebook.add(params_frame, text="Parameters")
        
        # Advanced tab
        advanced_frame = ttk.Frame(notebook)
        notebook.add(advanced_frame, text="Advanced")
        
        # Set up the main tab
        self.setup_main_tab(main_frame)
        
        # Set up the parameters tab
        self.setup_parameters_tab(params_frame)
        
        # Set up the advanced tab
        self.setup_advanced_tab(advanced_frame)
        
    def setup_main_tab(self, parent):
        """Set up the main tab with visualization and controls."""
        # Create left and right frames
        left_frame = ttk.Frame(parent)
        left_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        right_frame = ttk.Frame(parent)
        right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Left frame - SLM pattern display
        pattern_frame = ttk.LabelFrame(left_frame, text="SLM Pattern")
        pattern_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        self.pattern_canvas = tk.Canvas(pattern_frame, bg="black", width=400, height=400)
        self.pattern_canvas.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Right frame - Power meter and controls
        # Power meter display
        power_frame = ttk.LabelFrame(right_frame, text="Power Meter")
        power_frame.pack(fill=tk.X, padx=5, pady=5)
        
        # Current power display with large font
        current_power_frame = ttk.Frame(power_frame)
        current_power_frame.pack(fill=tk.X, padx=5, pady=5)
        
        ttk.Label(current_power_frame, text="Current Power:").pack(side=tk.LEFT, padx=5)
        ttk.Label(current_power_frame, textvariable=self.current_power, 
                 font=("Arial", 24, "bold")).pack(side=tk.LEFT, padx=5)
        ttk.Label(current_power_frame, text="W").pack(side=tk.LEFT)
        
        # Best power display
        best_power_frame = ttk.Frame(power_frame)
        best_power_frame.pack(fill=tk.X, padx=5, pady=5)
        
        ttk.Label(best_power_frame, text="Best Power:").pack(side=tk.LEFT, padx=5)
        ttk.Label(best_power_frame, textvariable=self.best_power,
                 font=("Arial", 18)).pack(side=tk.LEFT, padx=5)
        ttk.Label(best_power_frame, text="W").pack(side=tk.LEFT)
        
        # Status display
        status_frame = ttk.LabelFrame(right_frame, text="Optimization Status")
        status_frame.pack(fill=tk.X, padx=5, pady=5)
        
        # Temperature display
        temp_frame = ttk.Frame(status_frame)
        temp_frame.pack(fill=tk.X, padx=5, pady=2)
        
        ttk.Label(temp_frame, text="Temperature:").pack(side=tk.LEFT, padx=5)
        ttk.Label(temp_frame, textvariable=self.current_temperature).pack(side=tk.LEFT, padx=5)
        
        # Iteration display
        iter_frame = ttk.Frame(status_frame)
        iter_frame.pack(fill=tk.X, padx=5, pady=2)
        
        ttk.Label(iter_frame, text="Iteration:").pack(side=tk.LEFT, padx=5)
        ttk.Label(iter_frame, textvariable=self.iteration_count).pack(side=tk.LEFT, padx=5)
        
        # Control buttons
        control_frame = ttk.Frame(right_frame)
        control_frame.pack(fill=tk.X, padx=5, pady=10)
        
        self.start_button = ttk.Button(control_frame, text="Start Optimization", 
                                      command=self.start_optimization)
        self.start_button.pack(side=tk.LEFT, padx=5, pady=5, fill=tk.X, expand=True)
        
        self.stop_button = ttk.Button(control_frame, text="Stop Optimization", 
                                     command=self.stop_optimization, state=tk.DISABLED)
        self.stop_button.pack(side=tk.LEFT, padx=5, pady=5, fill=tk.X, expand=True)
        
        # Power history plot
        plot_frame = ttk.LabelFrame(right_frame, text="Power History")
        plot_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        self.power_plot_frame = ttk.Frame(plot_frame)
        self.power_plot_frame.pack(fill=tk.BOTH, expand=True)
        
    def setup_parameters_tab(self, parent):
        """Set up the parameters tab with adjustable optimization parameters."""
        # Create a frame for the parameters
        params_frame = ttk.Frame(parent)
        params_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Simulated Annealing Parameters
        sa_frame = ttk.LabelFrame(params_frame, text="Simulated Annealing Parameters")
        sa_frame.pack(fill=tk.X, padx=5, pady=5)
        
        # Initial Temperature
        temp_frame = ttk.Frame(sa_frame)
        temp_frame.pack(fill=tk.X, padx=5, pady=5)
        
        ttk.Label(temp_frame, text="Initial Temperature:").grid(row=0, column=0, sticky=tk.W, padx=5, pady=2)
        ttk.Entry(temp_frame, textvariable=self.initial_temperature, width=10).grid(row=0, column=1, padx=5, pady=2)
        ttk.Scale(temp_frame, from_=1.0, to=500.0, variable=self.initial_temperature, 
                 orient=tk.HORIZONTAL).grid(row=0, column=2, sticky=(tk.W, tk.E), padx=5, pady=2)
        
        # Cooling Rate
        cool_frame = ttk.Frame(sa_frame)
        cool_frame.pack(fill=tk.X, padx=5, pady=5)
        
        ttk.Label(cool_frame, text="Cooling Rate:").grid(row=0, column=0, sticky=tk.W, padx=5, pady=2)
        ttk.Entry(cool_frame, textvariable=self.cooling_rate, width=10).grid(row=0, column=1, padx=5, pady=2)
        ttk.Scale(cool_frame, from_=0.5, to=0.99, variable=self.cooling_rate, 
                 orient=tk.HORIZONTAL).grid(row=0, column=2, sticky=(tk.W, tk.E), padx=5, pady=2)
        
        # Iterations per Temperature
        iter_frame = ttk.Frame(sa_frame)
        iter_frame.pack(fill=tk.X, padx=5, pady=5)
        
        ttk.Label(iter_frame, text="Iterations per Temperature:").grid(row=0, column=0, sticky=tk.W, padx=5, pady=2)
        ttk.Entry(iter_frame, textvariable=self.iterations_per_temp, width=10).grid(row=0, column=1, padx=5, pady=2)
        ttk.Scale(iter_frame, from_=1, to=50, variable=self.iterations_per_temp, 
                 orient=tk.HORIZONTAL).grid(row=0, column=2, sticky=(tk.W, tk.E), padx=5, pady=2)
        
        # Minimum Temperature
        min_temp_frame = ttk.Frame(sa_frame)
        min_temp_frame.pack(fill=tk.X, padx=5, pady=5)
        
        ttk.Label(min_temp_frame, text="Minimum Temperature:").grid(row=0, column=0, sticky=tk.W, padx=5, pady=2)
        ttk.Entry(min_temp_frame, textvariable=self.min_temperature, width=10).grid(row=0, column=1, padx=5, pady=2)
        ttk.Scale(min_temp_frame, from_=0.001, to=1.0, variable=self.min_temperature, 
                 orient=tk.HORIZONTAL).grid(row=0, column=2, sticky=(tk.W, tk.E), padx=5, pady=2)
        
        # Perturbation Scale
        perturb_frame = ttk.Frame(sa_frame)
        perturb_frame.pack(fill=tk.X, padx=5, pady=5)
        
        ttk.Label(perturb_frame, text="Perturbation Scale:").grid(row=0, column=0, sticky=tk.W, padx=5, pady=2)
        ttk.Entry(perturb_frame, textvariable=self.perturbation_scale, width=10).grid(row=0, column=1, padx=5, pady=2)
        ttk.Scale(perturb_frame, from_=0.01, to=0.5, variable=self.perturbation_scale, 
                 orient=tk.HORIZONTAL).grid(row=0, column=2, sticky=(tk.W, tk.E), padx=5, pady=2)
        
        # Power Meter Parameters
        pm_frame = ttk.LabelFrame(params_frame, text="Power Meter Parameters")
        pm_frame.pack(fill=tk.X, padx=5, pady=5)
        
        # Wavelength
        wl_frame = ttk.Frame(pm_frame)
        wl_frame.pack(fill=tk.X, padx=5, pady=5)
        
        ttk.Label(wl_frame, text="Wavelength (nm):").grid(row=0, column=0, sticky=tk.W, padx=5, pady=2)
        ttk.Entry(wl_frame, textvariable=self.wavelength, width=10).grid(row=0, column=1, padx=5, pady=2)
        ttk.Scale(wl_frame, from_=500, to=2000, variable=self.wavelength, 
                 orient=tk.HORIZONTAL).grid(row=0, column=2, sticky=(tk.W, tk.E), padx=5, pady=2)
        
        # Averaging Count
        avg_frame = ttk.Frame(pm_frame)
        avg_frame.pack(fill=tk.X, padx=5, pady=5)
        
        ttk.Label(avg_frame, text="Averaging Count:").grid(row=0, column=0, sticky=tk.W, padx=5, pady=2)
        ttk.Entry(avg_frame, textvariable=self.averaging_count, width=10).grid(row=0, column=1, padx=5, pady=2)
        ttk.Scale(avg_frame, from_=1, to=100, variable=self.averaging_count, 
                 orient=tk.HORIZONTAL).grid(row=0, column=2, sticky=(tk.W, tk.E), padx=5, pady=2)
        
        # Apply Parameters button
        apply_frame = ttk.Frame(params_frame)
        apply_frame.pack(fill=tk.X, padx=5, pady=10)
        
        ttk.Button(apply_frame, text="Apply Parameters", 
                  command=self.apply_parameters).pack(padx=5, pady=5)
        
    def setup_advanced_tab(self, parent):
        """Set up the advanced tab with additional controls and visualizations."""
        # Create a frame for advanced options
        advanced_frame = ttk.Frame(parent)
        advanced_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Temperature history plot
        temp_plot_frame = ttk.LabelFrame(advanced_frame, text="Temperature History")
        temp_plot_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        self.temp_plot_frame = ttk.Frame(temp_plot_frame)
        self.temp_plot_frame.pack(fill=tk.BOTH, expand=True)
        
        # Save/Load configuration
        config_frame = ttk.LabelFrame(advanced_frame, text="Configuration")
        config_frame.pack(fill=tk.X, padx=5, pady=5)
        
        button_frame = ttk.Frame(config_frame)
        button_frame.pack(fill=tk.X, padx=5, pady=5)
        
        ttk.Button(button_frame, text="Save Configuration", 
                  command=self.save_configuration).pack(side=tk.LEFT, padx=5, pady=5)
        
        ttk.Button(button_frame, text="Load Configuration", 
                  command=self.load_configuration).pack(side=tk.LEFT, padx=5, pady=5)
        
        # Reset button
        reset_frame = ttk.Frame(advanced_frame)
        reset_frame.pack(fill=tk.X, padx=5, pady=10)
        
        ttk.Button(reset_frame, text="Reset Optimization", 
                  command=self.reset_optimization).pack(padx=5, pady=5)
        
    def initialize_plots(self):
        """Initialize the matplotlib plots."""
        # Power history plot
        self.power_fig = Figure(figsize=(5, 3), dpi=100)
        self.power_ax = self.power_fig.add_subplot(111)
        self.power_ax.set_title('Power vs. Iteration')
        self.power_ax.set_xlabel('Iteration')
        self.power_ax.set_ylabel('Power (W)')
        self.power_ax.grid(True)
        
        self.power_canvas = FigureCanvasTkAgg(self.power_fig, master=self.power_plot_frame)
        self.power_canvas.draw()
        self.power_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # Temperature history plot
        self.temp_fig = Figure(figsize=(5, 3), dpi=100)
        self.temp_ax = self.temp_fig.add_subplot(111)
        self.temp_ax.set_title('Temperature vs. Iteration')
        self.temp_ax.set_xlabel('Iteration')
        self.temp_ax.set_ylabel('Temperature')
        self.temp_ax.grid(True)
        
        self.temp_canvas = FigureCanvasTkAgg(self.temp_fig, master=self.temp_plot_frame)
        self.temp_canvas.draw()
        self.temp_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # Initialize plot data
        self.power_data = []
        self.temp_data = []
        self.iteration_data = []
        
    def update_power_display(self):
        """Update the power display periodically."""
        if self.optimizer and hasattr(self.optimizer, 'power_meter'):
            # Update current power if not running optimization
            if not self.optimization_running:
                power = self.optimizer.power_meter.read_power()
                self.current_power.set(f"{power:.6f}")
            
            # Update pattern display if available
            if hasattr(self.optimizer, 'slm') and hasattr(self.optimizer.slm, 'phase_mask'):
                self.update_pattern_display(self.optimizer.slm.phase_mask)
        
        # Schedule the next update
        self.root.after(500, self.update_power_display)
        
    def update_pattern_display(self, phase_mask):
        """Update the SLM pattern display."""
        try:
            # Resize the phase mask for display
            display_size = (400, 400)
            resized_mask = cv2.resize(phase_mask, display_size)
            
            # Convert to PIL Image
            pil_image = Image.fromarray(resized_mask)
            
            # Convert to PhotoImage
            tk_image = ImageTk.PhotoImage(image=pil_image)
            
            # Update canvas
            self.pattern_canvas.create_image(0, 0, anchor=tk.NW, image=tk_image)
            self.pattern_canvas.image = tk_image  # Keep a reference to prevent garbage collection
        except Exception as e:
            logging.error(f"Error updating pattern display: {e}")
        
    def update_plots(self):
        """Update the plots with current data."""
        if not self.power_data:
            return
            
        # Update power history plot
        self.power_ax.clear()
        self.power_ax.plot(self.iteration_data, self.power_data, 'b-')
        self.power_ax.set_title('Power vs. Iteration')
        self.power_ax.set_xlabel('Iteration')
        self.power_ax.set_ylabel('Power (W)')
        self.power_ax.grid(True)
        self.power_canvas.draw()
        
        # Update temperature history plot
        if self.temp_data:
            self.temp_ax.clear()
            # We may have fewer temperature points than iterations
            temp_iterations = list(range(0, len(self.iteration_data), 
                                        max(1, len(self.iteration_data) // len(self.temp_data))))[:len(self.temp_data)]
            self.temp_ax.plot(temp_iterations, self.temp_data, 'r-')
            self.temp_ax.set_title('Temperature vs. Iteration')
            self.temp_ax.set_xlabel('Iteration')
            self.temp_ax.set_ylabel('Temperature')
            self.temp_ax.grid(True)
            self.temp_canvas.draw()
        
    def optimization_callback(self, iteration, temperature, current_power, best_power):
        """Callback function for the optimization process."""
        # Update GUI variables
        self.current_power.set(f"{current_power:.6f}")
        self.best_power.set(f"{best_power:.6f}")
        self.current_temperature.set(f"{temperature:.4f}")
        self.iteration_count.set(iteration)
        
        # Update plot data
        self.power_data.append(current_power)
        self.iteration_data.append(iteration)
        
        # We get temperature updates less frequently
        if temperature not in self.temp_data:
            self.temp_data.append(temperature)
        
        # Update plots periodically to avoid too frequent updates
        if iteration % 5 == 0:
            self.root.after(0, self.update_plots)
        
    def start_optimization(self):
        """Start the optimization process."""
        if not self.optimizer:
            logging.error("Optimizer not initialized")
            return
            
        if self.optimization_running:
            return
            
        # Apply current parameters
        self.apply_parameters()
        
        # Start optimization in a separate thread
        self.optimization_running = True
        self.optimization_thread = threading.Thread(target=self.run_optimization)
        self.optimization_thread.daemon = True
        self.optimization_thread.start()
        
        # Update button states
        self.start_button.config(state=tk.DISABLED)
        self.stop_button.config(state=tk.NORMAL)
        
    def stop_optimization(self):
        """Stop the optimization process."""
        if not self.optimization_running:
            return
            
        self.optimization_running = False
        
        # Update button states
        self.stop_button.config(state=tk.DISABLED)
        self.start_button.config(state=tk.NORMAL)
        
    def run_optimization(self):
        """Run the optimization process."""
        try:
            # Reset plot data
            self.power_data = []
            self.temp_data = []
            self.iteration_data = []
            
            # Run the optimization
            best_mask, best_power, power_history = self.optimizer.run_optimization(
                callback=self.optimization_callback
            )
            
            # Update final results
            self.best_power.set(f"{best_power:.6f}")
            logging.info(f"Optimization complete. Best power: {best_power:.6f} W")
            
            # Save the best mask
            np.save('best_phase_mask.npy', best_mask)
            logging.info("Best phase mask saved to best_phase_mask.npy")
            
        except Exception as e:
            logging.error(f"Error during optimization: {e}")
        finally:
            # Ensure we reset the running flag and update buttons
            self.optimization_running = False
            self.root.after(0, lambda: self.stop_button.config(state=tk.DISABLED))
            self.root.after(0, lambda: self.start_button.config(state=tk.NORMAL))
        
    def apply_parameters(self):
        """Apply the current parameter values to the optimizer."""
        if not self.optimizer:
            return
            
        # Update simulated annealing parameters
        self.optimizer.initial_temperature = self.initial_temperature.get()
        self.optimizer.cooling_rate = self.cooling_rate.get()
        self.optimizer.iterations_per_temp = self.iterations_per_temp.get()
        self.optimizer.min_temperature = self.min_temperature.get()
        self.optimizer.perturbation_scale = self.perturbation_scale.get()
        
        # Update power meter parameters if available
        if hasattr(self.optimizer, 'power_meter'):
            self.optimizer.power_meter.set_wavelength(self.wavelength.get())
            self.optimizer.power_meter.set_averaging(self.averaging_count.get())
            
        logging.info("Parameters applied")
        
    def save_configuration(self):
        """Save the current configuration to a file."""
        config = {
            'initial_temperature': self.initial_temperature.get(),
            'cooling_rate': self.cooling_rate.get(),
            'iterations_per_temp': self.iterations_per_temp.get(),
            'min_temperature': self.min_temperature.get(),
            'perturbation_scale': self.perturbation_scale.get(),
            'wavelength': self.wavelength.get(),
            'averaging_count': self.averaging_count.get()
        }
        
        try:
            np.save('coupling_config.npy', config)
            logging.info("Configuration saved to coupling_config.npy")
        except Exception as e:
            logging.error(f"Error saving configuration: {e}")
        
    def load_configuration(self):
        """Load configuration from a file."""
        try:
            config = np.load('coupling_config.npy', allow_pickle=True).item()
            
            self.initial_temperature.set(config.get('initial_temperature', 100.0))
            self.cooling_rate.set(config.get('cooling_rate', 0.95))
            self.iterations_per_temp.set(config.get('iterations_per_temp', 10))
            self.min_temperature.set(config.get('min_temperature', 0.1))
            self.perturbation_scale.set(config.get('perturbation_scale', 0.1))
            self.wavelength.set(config.get('wavelength', 1550))
            self.averaging_count.set(config.get('averaging_count', 10))
            
            logging.info("Configuration loaded from coupling_config.npy")
        except Exception as e:
            logging.error(f"Error loading configuration: {e}")
        
    def reset_optimization(self):
        """Reset the optimization process."""
        # Stop any running optimization
        self.stop_optimization()
        
        # Reset variables
        self.current_power.set(0.0)
        self.best_power.set(0.0)
        self.current_temperature.set(0.0)
        self.iteration_count.set(0)
        
        # Reset plot data
        self.power_data = []
        self.temp_data = []
        self.iteration_data = []
        
        # Update plots
        self.update_plots()
        
        logging.info("Optimization reset")
        
    def set_optimizer(self, optimizer):
        """Set the optimizer instance."""
        self.optimizer = optimizer
        
    def run(self):
        """Run the main event loop."""
        self.root.mainloop()


if __name__ == "__main__":
    # This is just for testing the GUI independently
    root = tk.Tk()
    app = CouplingOptimizerGUI(root)
    app.run()
