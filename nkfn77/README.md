# PLEASE NOTE THIS IS AN UPDATED VERSION TO THE ONE FOUND ON GITHUB!!!

# RenderPy VR Headset Simulation

This project simulates a VR headset and multiple headsets interacting on a virtual floor, using real IMU (Inertial Measurement Unit) data for realistic orientation and sensor fusion. The simulation is rendered in Python and produces video outputs demonstrating both pure gyroscope and sensor fusion tracking.

## Features

- **3D rendering** of a VR headset and multiple headsets on a virtual floor.
- **IMU data integration**: Uses real sensor data to drive headset orientation.
- **Sensor fusion**: Compare pure gyroscope (alpha=1.0) and fused (alpha=0.9) orientation tracking.
- **Physics simulation**: Headsets slide and collide on the floor with realistic physics.
- **Depth of field effect** and lighting.
- **Video recording**: Automatically saves simulation as `.mp4` files.

## Requirements

- Python 3.6 or higher
- numpy
- matplotlib
- Pillow
- PyOpenGL
- opencv-python
- pygame

Install all dependencies with:
```
pip install numpy matplotlib Pillow PyOpenGL opencv-python pygame
```

## Usage

1. Place your IMU data as `IMUdata.csv` in the project root.
2. Run the simulation:
   ```
   python3 render.py
   ```
3. The simulation will automatically run two scenarios:
   - **Gyroscope only** (alpha=1.0): Output saved as `headset_simulation_gyro_only.mp4`
   - **Sensor fusion** (alpha=0.9): Output saved as `headset_simulation_sensor_fusion.mp4`

## Controls

- **B**: Toggle blur (depth of field)
- **R**: Reset simulation
- **P**: Pause/resume
- **V**: Start/stop recording
- **ESC**: Quit

## Output

- Video files are saved in the project directory after each simulation run.

## Notes

- Ensure your system supports OpenGL for best performance.
- If you encounter issues, verify your Python environment and package versions.

## Modules
### Image.py
Contains an image class capable of generating an image and exporting it to a PNG. Images are implemented as a buffer of 32-bit RGBA pixel color data stored in a byte array. This modules uses `zlib` and `struct` for compressing and packing PNG data. 

### Shape.py
Classes representing points, lines and triangles. Each has a `draw()` method for drawing that shape in an Image. Anti-aliased lines are drawn using [Wu's Line Drawing Algorithm](https://en.wikipedia.org/wiki/Xiaolin_Wu's_line_algorithm). Triangles are drawn by iterating over a bounding box and calculating barycentric coordinates to smoothly interpolate color over the shape.

### Model.py
Class with functions for reading a `.obj` file into a stored model, retrieving vertices, faces, properties of the model.

### Render.py
Implements the rendering pipeline. Loads a model, transforms vertices, computes shading, rasterizes faces into triangles, and outputs the final image to a PNG.

## Milestones
**12/29/17** – Hidden surface removal using Z-buffer

![cola](https://raw.githubusercontent.com/ecann/RenderPy/master/images/cola_depth_comparison.png)

**12/23/17 (1 year later 😛)** – Smooth shading using vertex normals

![cow](https://raw.githubusercontent.com/ecann/RenderPy/master/images/smoothcow.png)

**12/24/16** – Simple n\*l flat shading

![cow](https://raw.githubusercontent.com/ecann/RenderPy/master/images/cow.png)

**9/13/16** – Load model from .obj file, rasterize and render with triangles

![cow](https://raw.githubusercontent.com/ecann/RenderPy/master/images/discocow.png)

**9/11/16** – Draw triangles

![cow](https://raw.githubusercontent.com/ecann/RenderPy/master/images/triangle.png)

**8/27/16** – Draw lines with Wu's Algorithm

**8/19/16** – Image module wrapping a color buffer, PNG writer
