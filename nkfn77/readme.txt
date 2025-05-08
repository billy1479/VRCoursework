# RenderPy Project

## Required Packages

To run this project, ensure you have the following Python packages installed:

1. **numpy** - For numerical computations.
2. **matplotlib** - For rendering visualizations.
3. **Pillow** - For image processing.
4. **PyOpenGL** - For OpenGL bindings in Python.
5. **Opencv** - For image processing.

## Installation Instructions

You can install the required packages using pip:

```
pip install numpy matplotlib Pillow PyOpenGL opencv-python
```

Ensure you have Python 3.6 or higher installed on your system.

## Additional Notes

- For GPU rendering, ensure your system has the necessary OpenGL drivers installed.
- If you encounter issues, verify your Python environment and package versions.

## Running Instructions

This project can be ran via 'python3 render.py' while in the root directory. 

This will then render a simulation of the headset scene with an alpha value of 1 for pure gyroscopic DRF. This will be saved as headset_simulation_gyro_only.mp4

Following this, a simulation of the headset scene with an alpha value of 0.9, allowing for full fusion of the sensor data correction. This will be saved as headset_simulation_sensor_fusion.mp4