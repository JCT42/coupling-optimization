"""
Zernike Polynomials Implementation for SLM Phase Patterns

This module provides functions to generate Zernike polynomials on a grid,
which can be used to create phase patterns for a Spatial Light Modulator (SLM).
"""

import numpy as np
from typing import List, Tuple, Optional
import logging

def cart_to_polar(x: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Convert Cartesian coordinates to polar coordinates.
    
    Args:
        x: X coordinates
        y: Y coordinates
        
    Returns:
        Tuple of (r, theta) arrays
    """
    r = np.sqrt(x**2 + y**2)
    theta = np.arctan2(y, x)
    return r, theta

def create_grid(size: Tuple[int, int]) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Create a normalized grid for Zernike polynomial calculation.
    
    Args:
        size: Tuple of (height, width) for the grid
        
    Returns:
        Tuple of (x, y, r, theta) arrays where:
        - x, y are normalized Cartesian coordinates (-1 to 1)
        - r, theta are the corresponding polar coordinates
    """
    height, width = size
    
    # Create normalized x and y coordinates
    x = np.linspace(-1, 1, width)
    y = np.linspace(-1, 1, height)
    
    # Create a meshgrid
    x_grid, y_grid = np.meshgrid(x, y)
    
    # Convert to polar coordinates
    r_grid, theta_grid = cart_to_polar(x_grid, y_grid)
    
    return x_grid, y_grid, r_grid, theta_grid

def factorial(n: int) -> int:
    """
    Calculate factorial of n.
    
    Args:
        n: Integer to calculate factorial of
        
    Returns:
        n!
    """
    if n == 0 or n == 1:
        return 1
    else:
        return n * factorial(n - 1)

def zernike_radial(n: int, m: int, r: np.ndarray) -> np.ndarray:
    """
    Calculate the radial component of the Zernike polynomial.
    
    Args:
        n: Radial degree
        m: Azimuthal degree (|m| <= n and n-|m| is even)
        r: Radial coordinate array (0 to 1)
        
    Returns:
        Radial component of the Zernike polynomial
    """
    if (n - abs(m)) % 2 != 0:
        return np.zeros_like(r)
    
    R = np.zeros_like(r)
    
    for k in range((n - abs(m)) // 2 + 1):
        coef = (-1)**k * factorial(n - k)
        coef /= factorial(k) * factorial((n + abs(m)) // 2 - k) * factorial((n - abs(m)) // 2 - k)
        R += coef * r**(n - 2*k)
    
    return R

def zernike_polynomial(n: int, m: int, r: np.ndarray, theta: np.ndarray) -> np.ndarray:
    """
    Calculate the Zernike polynomial.
    
    Args:
        n: Radial degree
        m: Azimuthal degree (|m| <= n and n-|m| is even)
        r: Radial coordinate array (0 to 1)
        theta: Angular coordinate array
        
    Returns:
        Zernike polynomial Z_n^m
    """
    if abs(m) > n:
        raise ValueError(f"Invalid Zernike indices: |m| = {abs(m)} > n = {n}")
    
    # Set values outside the unit circle to zero
    mask = r <= 1.0
    r_masked = np.copy(r)
    r_masked[~mask] = 0.0
    
    # Calculate the radial component
    R = zernike_polynomial_radial(n, abs(m), r_masked)
    
    # Calculate the angular component
    if m > 0:
        Z = R * np.cos(m * theta)
    elif m < 0:
        Z = R * np.sin(abs(m) * theta)
    else:
        Z = R
    
    # Apply the mask
    Z[~mask] = 0.0
    
    return Z

def zernike_polynomial_radial(n: int, m: int, r: np.ndarray) -> np.ndarray:
    """
    Calculate the radial component of the Zernike polynomial.
    
    Args:
        n: Radial degree
        m: Azimuthal degree (|m| <= n and n-|m| is even)
        r: Radial coordinate array (0 to 1)
        
    Returns:
        Radial component of the Zernike polynomial
    """
    if (n - m) % 2 != 0:
        return np.zeros_like(r)
    
    R = np.zeros_like(r)
    
    for k in range((n - m) // 2 + 1):
        coef = (-1)**k * factorial(n - k)
        coef /= factorial(k) * factorial((n + m) // 2 - k) * factorial((n - m) // 2 - k)
        R += coef * r**(n - 2*k)
    
    return R

def noll_to_zernike(j: int) -> Tuple[int, int]:
    """
    Convert Noll index j to Zernike indices (n, m).
    
    Args:
        j: Noll index (1-based)
        
    Returns:
        Tuple of (n, m) Zernike indices
    """
    if j < 1:
        raise ValueError("Noll index j must be >= 1")
    
    n = 0
    j1 = j - 1
    while j1 >= n + 1:
        n += 1
        j1 -= n
    
    m = (-1)**(j % 2) * ((n % 2) + 2 * int((j1 + ((n + 1) % 2)) / 2))
    
    return n, m

def zernike_name(j: int) -> str:
    """
    Get the name of the Zernike mode for a given Noll index.
    
    Args:
        j: Noll index (1-based)
        
    Returns:
        Name of the Zernike mode
    """
    names = {
        1: "Piston",
        2: "Tip",
        3: "Tilt",
        4: "Defocus",
        5: "Oblique Astigmatism",
        6: "Vertical Astigmatism",
        7: "Vertical Coma",
        8: "Horizontal Coma",
        9: "Vertical Trefoil",
        10: "Oblique Trefoil",
        11: "Primary Spherical",
        12: "Vertical Secondary Astigmatism",
        13: "Oblique Secondary Astigmatism",
        14: "Vertical Quadrafoil",
        15: "Oblique Quadrafoil"
    }
    
    return names.get(j, f"Z{j}")

class ZernikeBasis:
    """Class to generate and manage Zernike polynomial basis for SLM patterns."""
    
    def __init__(self, resolution: Tuple[int, int], num_modes: int = 15):
        """
        Initialize the Zernike basis.
        
        Args:
            resolution: Tuple of (height, width) for the SLM
            num_modes: Number of Zernike modes to use (default: 15)
        """
        self.resolution = resolution
        self.num_modes = num_modes
        self.basis = []
        
        # Create the coordinate grids
        self.x, self.y, self.r, self.theta = create_grid(resolution)
        
        # Generate the Zernike basis
        self.generate_basis()
        
    def generate_basis(self):
        """Generate the Zernike polynomial basis."""
        self.basis = []
        
        for j in range(1, self.num_modes + 1):
            n, m = noll_to_zernike(j)
            Z = zernike_polynomial(n, m, self.r, self.theta)
            
            # Normalize the Zernike mode
            if np.any(Z != 0):
                Z = Z / np.sqrt(np.sum(Z**2))
            
            self.basis.append(Z)
            logging.info(f"Generated Zernike mode {j}: {zernike_name(j)} (n={n}, m={m})")
        
    def get_mode(self, j: int) -> np.ndarray:
        """
        Get a specific Zernike mode.
        
        Args:
            j: Mode index (1-based)
            
        Returns:
            The Zernike mode as a 2D array
        """
        if j < 1 or j > self.num_modes:
            raise ValueError(f"Mode index {j} out of range (1-{self.num_modes})")
        
        return self.basis[j-1]
    
    def combine_modes(self, coefficients: List[float]) -> np.ndarray:
        """
        Combine Zernike modes using the given coefficients.
        
        Args:
            coefficients: List of coefficients for each mode
            
        Returns:
            Combined Zernike modes as a 2D array
        """
        if len(coefficients) > self.num_modes:
            coefficients = coefficients[:self.num_modes]
        elif len(coefficients) < self.num_modes:
            coefficients = coefficients + [0.0] * (self.num_modes - len(coefficients))
        
        # Combine the modes
        combined = np.zeros(self.resolution)
        for j, coef in enumerate(coefficients, 1):
            combined += coef * self.get_mode(j)
        
        return combined
    
    def phase_to_slm(self, phase: np.ndarray) -> np.ndarray:
        """
        Convert phase values to SLM grayscale values.
        
        Args:
            phase: Phase array in radians
            
        Returns:
            Grayscale values for SLM (0-255)
        """
        # Map phase from [-π, π] to [0, 255]
        slm_values = ((phase + np.pi) / (2 * np.pi) * 255).astype(np.uint8)
        return slm_values
    
    def generate_phase_mask(self, coefficients: List[float]) -> np.ndarray:
        """
        Generate a phase mask for the SLM based on Zernike coefficients.
        
        Args:
            coefficients: List of coefficients for each Zernike mode
            
        Returns:
            Phase mask as grayscale values (0-255)
        """
        # Combine the Zernike modes
        phase = self.combine_modes(coefficients)
        
        # Scale the phase to [-π, π]
        phase_max = np.max(np.abs(phase))
        if phase_max > 0:
            phase = phase / phase_max * np.pi
        
        # Convert to SLM grayscale values
        return self.phase_to_slm(phase)
