"""
Coupling Efficiency Optimization using Zernike Polynomials and SPGD/SPSA

This application implements a real-time feedback loop to optimize coupling efficiency
into a single-mode fiber using:
- SPGD or SPSA as the optimization algorithm
- Zernike polynomials as the phase basis
- A GUI to visualize coupling power and optimization progress
- A power meter connection to the Thorlabs PM100D
"""

import numpy as np
import time
import logging
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import threading
import json
import os
from datetime import datetime
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

# Import our modules
from sony_pm100d import PowerMeter, SLM
from zernike_polynomials import ZernikeBasis
from spgd_optimizer import SPGDOptimizer
from coupling_gui import CouplingOptimizerGUI

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('coupling_optimization.log')
    ]
)

class CouplingOptimizationApp:
    """Main application for coupling efficiency optimization."""
    
    def __init__(self):
        """Initialize the application."""
        # Create the root window
        self.root = tk.Tk()
        self.root.title("Coupling Efficiency Optimization")
        self.root.geometry("1200x800")
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)
        
        # Initialize components
        self.initialize_components()
        
        # Create the GUI
        self.gui = CouplingOptimizerGUI(self.root)
        
        # Set up the optimizer in the GUI
        self.setup_optimizer()
        
        # Add additional controls for Zernike and SPGD parameters
        self.add_zernike_controls()
        
    def initialize_components(self):
        """Initialize the SLM, power meter, and optimizer components."""
        try:
            # Initialize the SLM
            self.slm = SLM(resolution=(800, 600), simulation_mode=False)
            logging.info("SLM initialized")
            
            # Initialize the power meter
            self.power_meter = PowerMeter()
            if self.power_meter.connected:
                logging.info("Power meter connected")
            else:
                logging.warning("Power meter not connected. Check connections and try again.")
            
            # Initialize the Zernike basis
            self.zernike_basis = ZernikeBasis(resolution=(800, 600), num_modes=15)
            logging.info(f"Zernike basis initialized with {self.zernike_basis.num_modes} modes")
            
            # Initialize the optimizer
            self.optimizer = SPGDOptimizer(
                slm=self.slm,
                power_meter=self.power_meter,
                zernike_basis=self.zernike_basis,
                num_coefficients=15,
                learning_rate=0.1,
                perturbation_size=0.05,
                max_iterations=1000,
                convergence_threshold=1e-5,
                algorithm="spgd"
            )
            logging.info("SPGD optimizer initialized")
            
        except Exception as e:
            logging.error(f"Error initializing components: {e}")
            messagebox.showerror("Initialization Error", f"Error initializing components: {e}")
    
    def setup_optimizer(self):
        """Set up the optimizer in the GUI."""
        if hasattr(self, 'gui') and hasattr(self, 'optimizer'):
            self.gui.set_optimizer(self.optimizer)
            logging.info("Optimizer set in GUI")
    
    def add_zernike_controls(self):
        """Add Zernike and SPGD parameter controls to the GUI."""
        # Create a new tab for Zernike controls
        if not hasattr(self.gui, 'notebook'):
            logging.warning("GUI notebook not found")
            return
        
        zernike_frame = ttk.Frame(self.gui.notebook)
        self.gui.notebook.add(zernike_frame, text="Zernike Controls")
        
        # Create variables for Zernike parameters
        self.num_modes_var = tk.IntVar(value=15)
        self.algorithm_var = tk.StringVar(value="spgd")
        self.learning_rate_var = tk.DoubleVar(value=0.1)
        self.perturbation_size_var = tk.DoubleVar(value=0.05)
        self.max_iterations_var = tk.IntVar(value=1000)
        self.convergence_threshold_var = tk.DoubleVar(value=1e-5)
        
        # Create the layout
        main_frame = ttk.Frame(zernike_frame)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Algorithm selection
        algo_frame = ttk.LabelFrame(main_frame, text="Algorithm")
        algo_frame.pack(fill=tk.X, padx=5, pady=5)
        
        ttk.Radiobutton(algo_frame, text="SPGD", variable=self.algorithm_var, 
                       value="spgd", command=self.update_parameter_ranges).pack(side=tk.LEFT, padx=20, pady=5)
        ttk.Radiobutton(algo_frame, text="SPSA", variable=self.algorithm_var, 
                       value="spsa", command=self.update_parameter_ranges).pack(side=tk.LEFT, padx=20, pady=5)
        
        # Number of Zernike modes
        modes_frame = ttk.Frame(main_frame)
        modes_frame.pack(fill=tk.X, padx=5, pady=5)
        
        ttk.Label(modes_frame, text="Number of Zernike Modes:").grid(row=0, column=0, sticky=tk.W, padx=5, pady=2)
        ttk.Entry(modes_frame, textvariable=self.num_modes_var, width=10).grid(row=0, column=1, padx=5, pady=2)
        ttk.Scale(modes_frame, from_=5, to=30, variable=self.num_modes_var, 
                 orient=tk.HORIZONTAL).grid(row=0, column=2, sticky=(tk.W, tk.E), padx=5, pady=2)
        
        # Learning rate
        lr_frame = ttk.Frame(main_frame)
        lr_frame.pack(fill=tk.X, padx=5, pady=5)
        
        ttk.Label(lr_frame, text="Learning Rate:").grid(row=0, column=0, sticky=tk.W, padx=5, pady=2)
        self.lr_entry = ttk.Entry(lr_frame, textvariable=self.learning_rate_var, width=10)
        self.lr_entry.grid(row=0, column=1, padx=5, pady=2)
        self.lr_scale = ttk.Scale(lr_frame, from_=0.01, to=0.5, variable=self.learning_rate_var, 
                 orient=tk.HORIZONTAL)
        self.lr_scale.grid(row=0, column=2, sticky=(tk.W, tk.E), padx=5, pady=2)
        
        # Perturbation size
        pert_frame = ttk.Frame(main_frame)
        pert_frame.pack(fill=tk.X, padx=5, pady=5)
        
        ttk.Label(pert_frame, text="Perturbation Size:").grid(row=0, column=0, sticky=tk.W, padx=5, pady=2)
        self.pert_entry = ttk.Entry(pert_frame, textvariable=self.perturbation_size_var, width=10)
        self.pert_entry.grid(row=0, column=1, padx=5, pady=2)
        self.pert_scale = ttk.Scale(pert_frame, from_=0.01, to=0.2, variable=self.perturbation_size_var, 
                 orient=tk.HORIZONTAL)
        self.pert_scale.grid(row=0, column=2, sticky=(tk.W, tk.E), padx=5, pady=2)
        
        # Max iterations
        iter_frame = ttk.Frame(main_frame)
        iter_frame.pack(fill=tk.X, padx=5, pady=5)
        
        ttk.Label(iter_frame, text="Max Iterations:").grid(row=0, column=0, sticky=tk.W, padx=5, pady=2)
        ttk.Entry(iter_frame, textvariable=self.max_iterations_var, width=10).grid(row=0, column=1, padx=5, pady=2)
        ttk.Scale(iter_frame, from_=100, to=5000, variable=self.max_iterations_var, 
                 orient=tk.HORIZONTAL).grid(row=0, column=2, sticky=(tk.W, tk.E), padx=5, pady=2)
        
        # Convergence threshold
        conv_frame = ttk.Frame(main_frame)
        conv_frame.pack(fill=tk.X, padx=5, pady=5)
        
        ttk.Label(conv_frame, text="Convergence Threshold:").grid(row=0, column=0, sticky=tk.W, padx=5, pady=2)
        ttk.Entry(conv_frame, textvariable=self.convergence_threshold_var, width=10).grid(row=0, column=1, padx=5, pady=2)
        ttk.Scale(conv_frame, from_=1e-7, to=1e-3, variable=self.convergence_threshold_var, 
                 orient=tk.HORIZONTAL).grid(row=0, column=2, sticky=(tk.W, tk.E), padx=5, pady=2)
        
        # Apply button
        apply_frame = ttk.Frame(main_frame)
        apply_frame.pack(fill=tk.X, padx=5, pady=10)
        
        ttk.Button(apply_frame, text="Apply Zernike Parameters", 
                  command=self.apply_zernike_parameters).pack(padx=5, pady=5)
        
        # Zernike coefficients display
        coef_frame = ttk.LabelFrame(main_frame, text="Zernike Coefficients")
        coef_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Create a figure for the coefficients plot
        self.coef_fig = Figure(figsize=(6, 4), dpi=100)
        self.coef_ax = self.coef_fig.add_subplot(111)
        self.coef_ax.set_title('Zernike Coefficients')
        self.coef_ax.set_xlabel('Mode Index')
        self.coef_ax.set_ylabel('Coefficient Value')
        self.coef_ax.grid(True)
        
        self.coef_canvas = FigureCanvasTkAgg(self.coef_fig, master=coef_frame)
        self.coef_canvas.draw()
        self.coef_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # Add buttons for loading/saving coefficients
        button_frame = ttk.Frame(main_frame)
        button_frame.pack(fill=tk.X, padx=5, pady=5)
        
        ttk.Button(button_frame, text="Load Coefficients", 
                  command=self.load_coefficients).pack(side=tk.LEFT, padx=5, pady=5)
        ttk.Button(button_frame, text="Save Coefficients", 
                  command=self.save_coefficients).pack(side=tk.LEFT, padx=5, pady=5)
        ttk.Button(button_frame, text="Reset Coefficients", 
                  command=self.reset_coefficients).pack(side=tk.LEFT, padx=5, pady=5)
        ttk.Button(button_frame, text="Update Plot", 
                  command=self.update_coefficient_plot).pack(side=tk.LEFT, padx=5, pady=5)
    
    def update_parameter_ranges(self):
        """Update parameter ranges based on the selected algorithm."""
        algorithm = self.algorithm_var.get()
        
        if algorithm == "spgd":
            # SPGD typically uses smaller learning rates and perturbation sizes
            self.learning_rate_var.set(0.1)
            self.perturbation_size_var.set(0.05)
            self.lr_scale.configure(from_=0.01, to=0.5)
            self.pert_scale.configure(from_=0.01, to=0.2)
        else:  # SPSA
            # SPSA can use larger learning rates and perturbation sizes
            self.learning_rate_var.set(0.2)
            self.perturbation_size_var.set(0.1)
            self.lr_scale.configure(from_=0.05, to=1.0)
            self.pert_scale.configure(from_=0.05, to=0.5)
        
        # Update the entries
        self.lr_entry.update()
        self.pert_entry.update()
    
    def apply_zernike_parameters(self):
        """Apply the Zernike and SPGD parameters to the optimizer."""
        if not hasattr(self, 'optimizer'):
            logging.warning("Optimizer not initialized")
            return
        
        try:
            # Update the optimizer parameters
            self.optimizer.algorithm = self.algorithm_var.get()
            self.optimizer.learning_rate = self.learning_rate_var.get()
            self.optimizer.perturbation_size = self.perturbation_size_var.get()
            self.optimizer.max_iterations = self.max_iterations_var.get()
            self.optimizer.convergence_threshold = self.convergence_threshold_var.get()
            
            # Update the number of coefficients if changed
            new_num_modes = self.num_modes_var.get()
            if new_num_modes != self.optimizer.num_coefficients:
                # Ensure the Zernike basis has enough modes
                if new_num_modes > self.zernike_basis.num_modes:
                    self.zernike_basis = ZernikeBasis(
                        resolution=self.slm.resolution, 
                        num_modes=new_num_modes
                    )
                    self.optimizer.zernike_basis = self.zernike_basis
                
                # Update the number of coefficients
                old_coefficients = self.optimizer.coefficients
                self.optimizer.num_coefficients = new_num_modes
                
                # Resize the coefficients array
                if len(old_coefficients) > new_num_modes:
                    self.optimizer.coefficients = old_coefficients[:new_num_modes]
                else:
                    self.optimizer.coefficients = np.pad(
                        old_coefficients, 
                        (0, new_num_modes - len(old_coefficients))
                    )
            
            logging.info(f"Applied Zernike parameters: algorithm={self.algorithm_var.get()}, "
                         f"num_modes={new_num_modes}, learning_rate={self.learning_rate_var.get()}, "
                         f"perturbation_size={self.perturbation_size_var.get()}, "
                         f"max_iterations={self.max_iterations_var.get()}, "
                         f"convergence_threshold={self.convergence_threshold_var.get()}")
            
            # Update the coefficient plot
            self.update_coefficient_plot()
            
            messagebox.showinfo("Parameters Applied", "Zernike parameters applied successfully")
        except Exception as e:
            logging.error(f"Error applying Zernike parameters: {e}")
            messagebox.showerror("Parameter Error", f"Error applying parameters: {e}")
    
    def update_coefficient_plot(self):
        """Update the Zernike coefficients plot."""
        if not hasattr(self, 'optimizer') or not hasattr(self, 'coef_ax'):
            return
        
        try:
            # Clear the plot
            self.coef_ax.clear()
            
            # Plot the coefficients
            x = np.arange(1, len(self.optimizer.coefficients) + 1)
            self.coef_ax.bar(x, self.optimizer.coefficients)
            
            # Set labels and title
            self.coef_ax.set_title('Zernike Coefficients')
            self.coef_ax.set_xlabel('Mode Index')
            self.coef_ax.set_ylabel('Coefficient Value')
            self.coef_ax.set_xticks(x)
            self.coef_ax.grid(True)
            
            # Update the canvas
            self.coef_canvas.draw()
        except Exception as e:
            logging.error(f"Error updating coefficient plot: {e}")
    
    def load_coefficients(self):
        """Load Zernike coefficients from a file."""
        if not hasattr(self, 'optimizer'):
            logging.warning("Optimizer not initialized")
            return
        
        try:
            filename = filedialog.askopenfilename(
                title="Load Zernike Coefficients",
                filetypes=[("NumPy Files", "*.npy"), ("All Files", "*.*")]
            )
            
            if filename:
                success = self.optimizer.load_coefficients(filename)
                if success:
                    messagebox.showinfo("Load Successful", f"Coefficients loaded from {filename}")
                    self.update_coefficient_plot()
                else:
                    messagebox.showerror("Load Error", "Failed to load coefficients")
        except Exception as e:
            logging.error(f"Error loading coefficients: {e}")
            messagebox.showerror("Load Error", f"Error loading coefficients: {e}")
    
    def save_coefficients(self):
        """Save current Zernike coefficients to a file."""
        if not hasattr(self, 'optimizer'):
            logging.warning("Optimizer not initialized")
            return
        
        try:
            filename = filedialog.asksaveasfilename(
                title="Save Zernike Coefficients",
                defaultextension=".npy",
                filetypes=[("NumPy Files", "*.npy"), ("All Files", "*.*")]
            )
            
            if filename:
                np.save(filename, self.optimizer.coefficients)
                messagebox.showinfo("Save Successful", f"Coefficients saved to {filename}")
        except Exception as e:
            logging.error(f"Error saving coefficients: {e}")
            messagebox.showerror("Save Error", f"Error saving coefficients: {e}")
    
    def reset_coefficients(self):
        """Reset Zernike coefficients to zero."""
        if not hasattr(self, 'optimizer'):
            logging.warning("Optimizer not initialized")
            return
        
        try:
            self.optimizer.coefficients = np.zeros(self.optimizer.num_coefficients)
            self.optimizer.apply_phase_mask(self.optimizer.coefficients)
            self.update_coefficient_plot()
            messagebox.showinfo("Reset Successful", "Coefficients reset to zero")
        except Exception as e:
            logging.error(f"Error resetting coefficients: {e}")
            messagebox.showerror("Reset Error", f"Error resetting coefficients: {e}")
    
    def on_close(self):
        """Handle window close event."""
        try:
            # Stop any running optimization
            if hasattr(self, 'optimizer') and self.optimizer.running:
                self.optimizer.stop()
                time.sleep(0.5)  # Give it time to stop
            
            # Close the SLM
            if hasattr(self, 'slm'):
                self.slm.close()
            
            # Close the power meter
            if hasattr(self, 'power_meter') and self.power_meter.connected:
                self.power_meter.disconnect()
            
            logging.info("Application closed")
            self.root.destroy()
        except Exception as e:
            logging.error(f"Error during shutdown: {e}")
            self.root.destroy()
    
    def run(self):
        """Run the application."""
        self.root.mainloop()


if __name__ == "__main__":
    try:
        app = CouplingOptimizationApp()
        app.run()
    except Exception as e:
        logging.error(f"Application error: {e}")
        messagebox.showerror("Application Error", f"Error: {e}")
