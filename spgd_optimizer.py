"""
SPGD/SPSA Optimization for Coupling Efficiency

This module implements the Stochastic Parallel Gradient Descent (SPGD) and
Stochastic Perturbation Stochastic Approximation (SPSA) algorithms for
optimizing coupling efficiency into a single-mode fiber using Zernike polynomials.
"""

import numpy as np
import time
import logging
from typing import List, Tuple, Optional, Callable, Dict, Any
import threading
import json
from datetime import datetime

from zernike_polynomials import ZernikeBasis

class SPGDOptimizer:
    """
    Stochastic Parallel Gradient Descent (SPGD) optimizer for coupling efficiency.
    """
    
    def __init__(self, 
                 slm,
                 power_meter,
                 zernike_basis: ZernikeBasis,
                 num_coefficients: int = 15,
                 learning_rate: float = 0.1,
                 perturbation_size: float = 0.05,
                 max_iterations: int = 1000,
                 convergence_threshold: float = 1e-5,
                 algorithm: str = "spgd"):
        """
        Initialize the SPGD optimizer.
        
        Args:
            slm: SLM object for controlling the spatial light modulator
            power_meter: PowerMeter object for reading power measurements
            zernike_basis: ZernikeBasis object for generating phase patterns
            num_coefficients: Number of Zernike coefficients to optimize
            learning_rate: Learning rate for the optimization algorithm
            perturbation_size: Size of the random perturbations
            max_iterations: Maximum number of iterations
            convergence_threshold: Convergence threshold for stopping
            algorithm: Optimization algorithm to use ("spgd" or "spsa")
        """
        self.slm = slm
        self.power_meter = power_meter
        self.zernike_basis = zernike_basis
        self.num_coefficients = min(num_coefficients, zernike_basis.num_modes)
        self.learning_rate = learning_rate
        self.perturbation_size = perturbation_size
        self.max_iterations = max_iterations
        self.convergence_threshold = convergence_threshold
        self.algorithm = algorithm.lower()
        
        # Initialize coefficients to zero
        self.coefficients = np.zeros(self.num_coefficients)
        
        # For tracking progress
        self.best_coefficients = np.copy(self.coefficients)
        self.best_power = -float('inf')
        self.power_history = []
        self.coefficient_history = []
        
        # Flag to control stopping the optimization
        self.stop_requested = False
        self.running = False
        
        # Logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.StreamHandler(),
                logging.FileHandler('spgd_optimization.log')
            ]
        )
    
    def measure_power(self) -> float:
        """
        Measure the current power as the objective function.
        Higher power means better coupling efficiency.
        
        Returns:
            Power reading in watts, or -1 if error
        """
        if not self.power_meter.connected:
            logging.warning("Power meter not connected")
            return -1.0
        
        try:
            power = self.power_meter.read_power()
            return power
        except Exception as e:
            logging.error(f"Error reading power: {e}")
            return -1.0
    
    def apply_phase_mask(self, coefficients: np.ndarray) -> bool:
        """
        Generate and apply a phase mask to the SLM based on Zernike coefficients.
        
        Args:
            coefficients: Array of Zernike coefficients
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Generate the phase mask
            phase_mask = self.zernike_basis.generate_phase_mask(coefficients)
            
            # Apply to the SLM
            return self.slm.apply_phase_mask(phase_mask)
        except Exception as e:
            logging.error(f"Error applying phase mask: {e}")
            return False
    
    def spgd_step(self) -> Tuple[np.ndarray, float]:
        """
        Perform one step of the SPGD algorithm.
        
        Returns:
            Tuple of (updated_coefficients, current_power)
        """
        # Generate random perturbations
        delta = np.random.normal(0, 1, self.num_coefficients)
        delta = delta / np.linalg.norm(delta) * self.perturbation_size
        
        # Apply positive perturbation
        coefficients_plus = self.coefficients + delta
        self.apply_phase_mask(coefficients_plus)
        time.sleep(0.05)  # Small delay for SLM to settle
        power_plus = self.measure_power()
        
        # Apply negative perturbation
        coefficients_minus = self.coefficients - delta
        self.apply_phase_mask(coefficients_minus)
        time.sleep(0.05)  # Small delay for SLM to settle
        power_minus = self.measure_power()
        
        # Calculate gradient estimate
        gradient = (power_plus - power_minus) / (2 * self.perturbation_size)
        
        # Update coefficients
        self.coefficients = self.coefficients + self.learning_rate * gradient * delta
        
        # Apply updated coefficients
        self.apply_phase_mask(self.coefficients)
        time.sleep(0.05)  # Small delay for SLM to settle
        current_power = self.measure_power()
        
        return self.coefficients, current_power
    
    def spsa_step(self) -> Tuple[np.ndarray, float]:
        """
        Perform one step of the SPSA algorithm.
        
        Returns:
            Tuple of (updated_coefficients, current_power)
        """
        # Generate random perturbation direction (Bernoulli distribution)
        delta = np.random.choice([-1, 1], size=self.num_coefficients)
        
        # Apply positive perturbation
        coefficients_plus = self.coefficients + self.perturbation_size * delta
        self.apply_phase_mask(coefficients_plus)
        time.sleep(0.05)  # Small delay for SLM to settle
        power_plus = self.measure_power()
        
        # Apply negative perturbation
        coefficients_minus = self.coefficients - self.perturbation_size * delta
        self.apply_phase_mask(coefficients_minus)
        time.sleep(0.05)  # Small delay for SLM to settle
        power_minus = self.measure_power()
        
        # Calculate gradient estimate
        gradient = (power_plus - power_minus) / (2 * self.perturbation_size)
        
        # Update coefficients - SPSA uses element-wise division by delta
        gradient_estimate = gradient / delta
        self.coefficients = self.coefficients + self.learning_rate * gradient_estimate
        
        # Apply updated coefficients
        self.apply_phase_mask(self.coefficients)
        time.sleep(0.05)  # Small delay for SLM to settle
        current_power = self.measure_power()
        
        return self.coefficients, current_power
    
    def stop(self):
        """Request to stop the optimization process."""
        logging.info("Stop requested")
        self.stop_requested = True
    
    def run_optimization(self, 
                         initial_coefficients: Optional[np.ndarray] = None,
                         callback: Optional[Callable] = None) -> Tuple[np.ndarray, float, List[float]]:
        """
        Run the optimization algorithm.
        
        Args:
            initial_coefficients: Starting Zernike coefficients (if None, zeros are used)
            callback: Optional callback function called after each iteration
            
        Returns:
            Tuple of (best_coefficients, best_power, power_history)
        """
        if self.running:
            logging.warning("Optimization already running")
            return self.best_coefficients, self.best_power, self.power_history
        
        self.running = True
        self.stop_requested = False
        
        # Initialize coefficients
        if initial_coefficients is not None:
            self.coefficients = np.copy(initial_coefficients[:self.num_coefficients])
        else:
            self.coefficients = np.zeros(self.num_coefficients)
        
        # Reset tracking variables
        self.best_coefficients = np.copy(self.coefficients)
        self.best_power = -float('inf')
        self.power_history = []
        self.coefficient_history = []
        
        # Apply initial coefficients
        self.apply_phase_mask(self.coefficients)
        time.sleep(0.1)  # Small delay for SLM to settle
        
        # Measure initial power
        current_power = self.measure_power()
        self.power_history.append(current_power)
        self.coefficient_history.append(np.copy(self.coefficients))
        
        if current_power > self.best_power:
            self.best_power = current_power
            self.best_coefficients = np.copy(self.coefficients)
        
        # Main optimization loop
        iteration = 0
        prev_power = current_power
        
        logging.info(f"Starting {self.algorithm.upper()} optimization with {self.num_coefficients} Zernike modes")
        
        try:
            while iteration < self.max_iterations and not self.stop_requested:
                # Perform one step of the selected algorithm
                if self.algorithm == "spgd":
                    self.coefficients, current_power = self.spgd_step()
                else:  # SPSA
                    self.coefficients, current_power = self.spsa_step()
                
                # Update tracking variables
                self.power_history.append(current_power)
                self.coefficient_history.append(np.copy(self.coefficients))
                
                if current_power > self.best_power:
                    self.best_power = current_power
                    self.best_coefficients = np.copy(self.coefficients)
                
                # Check for convergence
                power_change = abs(current_power - prev_power) / max(abs(prev_power), 1e-10)
                if power_change < self.convergence_threshold and iteration > 10:
                    logging.info(f"Converged after {iteration+1} iterations")
                    break
                
                prev_power = current_power
                iteration += 1
                
                # Call the callback function if provided
                if callback:
                    callback(iteration, current_power, self.best_power, self.coefficients)
                
                # Log progress periodically
                if iteration % 10 == 0:
                    logging.info(f"Iteration {iteration}: Power = {current_power:.6f} W, Best = {self.best_power:.6f} W")
        
        except Exception as e:
            logging.error(f"Error during optimization: {e}")
        finally:
            # Apply the best coefficients
            self.apply_phase_mask(self.best_coefficients)
            
            # Save the results
            self.save_results()
            
            self.running = False
            logging.info(f"Optimization completed: Best power = {self.best_power:.6f} W")
        
        return self.best_coefficients, self.best_power, self.power_history
    
    def save_results(self):
        """Save the optimization results to files."""
        try:
            # Save best coefficients
            np.save('best_zernike_coefficients.npy', self.best_coefficients)
            
            # Save optimization history
            history = {
                'power_history': self.power_history,
                'coefficient_history': [coef.tolist() for coef in self.coefficient_history],
                'best_power': float(self.best_power),
                'best_coefficients': self.best_coefficients.tolist(),
                'algorithm': self.algorithm,
                'num_coefficients': self.num_coefficients,
                'learning_rate': self.learning_rate,
                'perturbation_size': self.perturbation_size,
                'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            }
            
            with open('optimization_history.json', 'w') as f:
                json.dump(history, f, indent=2)
            
            logging.info("Optimization results saved")
        except Exception as e:
            logging.error(f"Error saving results: {e}")
    
    def load_coefficients(self, filename: str) -> bool:
        """
        Load Zernike coefficients from a file.
        
        Args:
            filename: Path to the coefficients file
            
        Returns:
            True if successful, False otherwise
        """
        try:
            loaded_coefficients = np.load(filename)
            
            # Ensure the correct number of coefficients
            if len(loaded_coefficients) >= self.num_coefficients:
                self.coefficients = loaded_coefficients[:self.num_coefficients]
            else:
                # Pad with zeros if needed
                self.coefficients = np.pad(loaded_coefficients, 
                                          (0, self.num_coefficients - len(loaded_coefficients)))
            
            # Apply the loaded coefficients
            self.apply_phase_mask(self.coefficients)
            
            logging.info(f"Loaded coefficients from {filename}")
            return True
        except Exception as e:
            logging.error(f"Error loading coefficients: {e}")
            return False
