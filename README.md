# Coupling Optimization System

A system for optimizing fiber coupling efficiency using a Thorlabs PM100D power meter and a Spatial Light Modulator (SLM).

## Components

- **SLM Control**: Uses OpenCV for visualization at position 1280
- **Power Meter Interface**: Connects to Thorlabs PM100D using PyVISA
- **Optimization Algorithm**: Uses simulated annealing to maximize coupling efficiency
- **GUI Interface**: Provides real-time visualization and parameter controls

## Files

- `sony_pm100d.py`: Main implementation with SLM control and power meter interface
- `thorlabs_power_meter.py`: Alternative power meter interface implementation
- `coupling_gui.py`: GUI for the optimization system
- `find_powermeter.py`: Utility to detect connected power meters
- `check_powermeter.py`: Diagnostic tool for power meter connections

## Requirements

- Python 3.x
- PyVISA
- PyVISA-py backend
- ThorlabsPM100
- OpenCV
- NumPy
- Matplotlib
- Tkinter

## Usage

1. Connect the Thorlabs PM100D power meter to your computer
2. Connect the SLM to your computer
3. Run `python sony_pm100d.py` to start the optimization
4. Adjust parameters as needed in the GUI

## Notes

- SLM resolution is set to 800x600
- The system will automatically detect the power meter using the appropriate resource name
- If hardware is not detected, the system will provide appropriate error messages
