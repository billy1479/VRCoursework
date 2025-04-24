#!/usr/bin/env python3
"""
Headset IMU Filtering Comparison

This standalone script compares different filter methods for IMU data:
- Pure gyroscope integration (alpha=1.0)
- Complementary filter with different alpha values

It creates a visualization of a 3D model controlled by these different filtering approaches
and saves comparison videos.
"""

import pygame
import math
from vector import Vector
from image import Image, Color
from model import Model, DeadReckoningFilter, SensorDataParser
from shape import Point, Triangle
from video_recorder import VideoRecorder

def getVertexNormal(vertIndex, faceNormalsByVertex):
    """Compute vertex normals by averaging the normals of adjacent faces"""
    normal = Vector(0, 0, 0)
    for adjNormal in faceNormalsByVertex[vertIndex]:
        normal = normal + adjNormal
    return normal / len(faceNormalsByVertex[vertIndex])

def getPerspectiveProjection(x, y, z, width, height):
    """Apply perspective projection to convert 3D coordinates to screen coordinates"""
    from model import Matrix4, Vec4
    
    # Set up perspective parameters
    fov = math.pi / 3.0  # 60-degree field of view
    aspect = width / height
    near = 0.1     # Near clipping plane
    far = 100.0    # Far clipping plane
    
    # Create the perspective matrix
    perspective_matrix = Matrix4.perspective(fov, aspect, near, far)
    
    # Create a vector in homogeneous coordinates
    point = Vec4(x, y, z, 1.0)
    
    # Apply perspective transformation
    projected = perspective_matrix.multiply(point)
    
    # Perform perspective division
    normalized = projected.perspectiveDivide()
    
    # Convert to screen coordinates
    screenX = int((normalized.x + 1.0) * width / 2.0)
    screenY = int((normalized.y + 1.0) * height / 2.0)
    
    return screenX, screenY

def load_csv_data(file_path):
    """Load and parse IMU data from CSV file"""
    parser = SensorDataParser(file_path)
    sensor_data = parser.parse()
    print(f"Loaded {len(sensor_data)} sensor data entries from {file_path}")
    return sensor_data

def compare_filter_methods(csv_contents, model, width, height):
    """
    Compare and visualize the difference between pure gyroscope-based tracking 
    and gyroscope+accelerometer tracking with different alpha values.
    
    Args:
        csv_contents: List of sensor data readings
        model: 3D model to visualize
        width, height: Dimensions for rendering
    """
    # Create a video recorder
    recorder = VideoRecorder(width, height, fps=30, output_dir="comparison_results")
    
    # Define alpha values to test (1.0 = gyro only, lower values use more accelerometer)
    # We use extreme values to make differences more apparent
    alpha_values = [1.0, 0.5, 0.1]
    
    # We'll run a subset of the data for quicker testing
    test_duration = min(len(csv_contents), 6000)  # Limit to 300 samples
    
    print("Running comparison of filter methods...")
    
    # For each alpha value, record a video
    for alpha in alpha_values:
        # Create a new filter with this alpha value
        filter_name = "gyro_only" if alpha == 1.0 else f"complementary_alpha_{alpha}"
        print(f"\nTesting {filter_name}")
        
        dr_filter = DeadReckoningFilter(alpha=alpha)
        
        # Calibrate using first 100 samples (assuming device is at rest)
        dr_filter.calibrate(csv_contents[:100])
        
        # Create pygame for visualization
        pygame.init()
        screen = pygame.display.set_mode((width, height))
        pygame.display.set_caption(f"Filter Comparison - {filter_name}")
        buffer_surface = pygame.Surface((width, height))
        
        # Create image buffer
        image = Image(width, height, Color(255, 255, 255, 255))
        
        # Init z-buffer
        zBuffer = [-float('inf')] * width * height
        
        # Reset model
        model.normalizeGeometry()
        model.setPosition(0, 0, -12)
        
        # Create pygame clock for timing
        clock = pygame.time.Clock()
        
        # Start recording
        recorder.start_recording()
        
        # Run the simulation
        for i in range(test_duration):
            # Handle timing
            delta_time = clock.tick(60) / 1000.0
            
            # Get current sensor data and update filter
            current_sensor_data = csv_contents[i]
            position, orientation = dr_filter.update(current_sensor_data)
            
            # Convert quaternion to Euler angles
            roll, pitch, yaw = dr_filter.get_euler_angles()
            
            # Update model rotation - convert to degrees
            # Note: Model.setRotation expects radians but we're explicitly
            # converting for clarity in debugging
            model.setRotation(roll, pitch, yaw)
            
            # Reset image and z-buffer for new frame
            image = Image(width, height, Color(255, 255, 255, 255))
            zBuffer = [-float('inf')] * width * height
            
            # Calculate face normals
            faceNormals = {}
            for face in model.faces:
                p0 = model.getTransformedVertex(face[0])
                p1 = model.getTransformedVertex(face[1])
                p2 = model.getTransformedVertex(face[2])
                faceNormal = (p2-p0).cross(p1-p0).normalize()

                for j in face:
                    if j not in faceNormals:
                        faceNormals[j] = []

                    faceNormals[j].append(faceNormal)

            # Calculate vertex normals
            vertexNormals = []
            for vertIndex in range(len(model.vertices)):
                if vertIndex in faceNormals:
                    vertNorm = getVertexNormal(vertIndex, faceNormals)
                    vertexNormals.append(vertNorm)
                else:
                    vertexNormals.append(Vector(0, 0, 1))  # Default normal

            # Render all faces for this frame
            for face in model.faces:
                p0 = model.getTransformedVertex(face[0])
                p1 = model.getTransformedVertex(face[1])
                p2 = model.getTransformedVertex(face[2])
                n0, n1, n2 = [vertexNormals[i] for i in face]

                # Define the light direction
                lightDir = Vector(0, 0, -1)

                # Set to true if face should be culled
                cull = False

                # Transform vertices and calculate lighting
                transformedPoints = []
                for p, n in zip([p0, p1, p2], [n0, n1, n2]):
                    intensity = n * lightDir

                    if intensity < 0:
                        cull = True
                        
                    screenX, screenY = getPerspectiveProjection(p.x, p.y, p.z, width, height)
                    
                    transformedPoints.append(Point(screenX, screenY, p.z, Color(intensity*255, intensity*255, intensity*255, 255)))

                if not cull:
                    Triangle(transformedPoints[0], transformedPoints[1], transformedPoints[2]).draw_faster(image, zBuffer)

            # Draw alpha value and filter type on the screen
            # This requires adding text to the pygame surface
            pixel_array = pygame.surfarray.pixels3d(buffer_surface)
            for y in range(height):
                for x in range(width):
                    flipY = (height - y - 1)
                    index = (flipY * width + x) * 4 + flipY + 1
                    
                    r = image.buffer[index]
                    g = image.buffer[index + 1]
                    b = image.buffer[index + 2]
                    
                    pixel_array[x, y] = (r, g, b)
            
            del pixel_array
            
            # Add text overlay
            font = pygame.font.SysFont('Arial', 24)
            text = font.render(f"Filter: {filter_name}", True, (255, 0, 0))
            buffer_surface.blit(text, (10, 10))
            
            # Add orientation values
            orientation_text = font.render(f"Roll: {math.degrees(roll):.1f}° Pitch: {math.degrees(pitch):.1f}° Yaw: {math.degrees(yaw):.1f}°", True, (255, 0, 0))
            buffer_surface.blit(orientation_text, (10, 40))
            
            # Update display
            screen.blit(buffer_surface, (0, 0))
            pygame.display.flip()
            
            # Capture frame for video
            recorder.capture_frame(screen)
            
            # Handle events
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    recorder.stop_recording()
                    return
        
        # Stop recording and save video
        recorder.stop_recording()
        recorder.save_video(f"{filter_name}.mp4")
        recorder.clear_frames()
        
        # Clean up pygame
        pygame.quit()
    
    # Generate comparison video showing side-by-side
    print("Creating side-by-side comparison video...")
    create_side_by_side_comparison(alpha_values)

def create_side_by_side_comparison(alpha_values):
    """
    Create a side-by-side comparison video using FFmpeg
    
    Args:
        alpha_values: List of alpha values used in the comparison
    """
    try:
        import subprocess
        
        # Build the filter complex for side-by-side comparison
        videos = []
        filter_complex = ""
        
        for i, alpha in enumerate(alpha_values):
            filter_name = "gyro_only" if alpha == 1.0 else f"complementary_alpha_{alpha}"
            videos.append(f"comparison_results/{filter_name}.mp4")
        
        # Create FFmpeg command for side-by-side layout
        if len(videos) == 2:
            # Simple side-by-side for 2 videos
            cmd = f"ffmpeg -i {videos[0]} -i {videos[1]} -filter_complex \"[0:v]setpts=PTS-STARTPTS[left];[1:v]setpts=PTS-STARTPTS[right];[left][right]hstack=inputs=2[v]\" -map \"[v]\" comparison_results/side_by_side.mp4"
        elif len(videos) == 3:
            # Grid layout for 3 videos (2x2 with one blank)
            cmd = f"""ffmpeg -i {videos[0]} -i {videos[1]} -i {videos[2]} -filter_complex "
                [0:v]setpts=PTS-STARTPTS,scale=iw/2:ih/2[top_left];
                [1:v]setpts=PTS-STARTPTS,scale=iw/2:ih/2[top_right];
                [2:v]setpts=PTS-STARTPTS,scale=iw/2:ih/2[bottom_left];
                [top_left][top_right]hstack=inputs=2[top];
                [bottom_left]pad=iw*2:ih[bottom];
                [top][bottom]vstack=inputs=2[v]" -map "[v]" comparison_results/grid_comparison.mp4"""
        else:
            # Create a 2x2 grid for 4 videos
            cmd = f"""ffmpeg -i {videos[0]} -i {videos[1]} -i {videos[2]} -filter_complex "
                [0:v]setpts=PTS-STARTPTS,scale=iw/2:ih/2[top_left];
                [1:v]setpts=PTS-STARTPTS,scale=iw/2:ih/2[top_right];
                [2:v]setpts=PTS-STARTPTS,scale=iw/2:ih/2[bottom_left];
                [top_left][top_right]hstack=inputs=2[top];
                [bottom_left]pad=iw*2:ih[bottom];
                [top][bottom]vstack=inputs=2[v]" -map "[v]" comparison_results/grid_comparison.mp4"""
        
        print(f"Running FFmpeg command: {cmd}")
        subprocess.run(cmd, shell=True, check=True)
        print("Comparison video created successfully!")
        
    except Exception as e:
        print(f"Error creating comparison video: {e}")
        print("You may need to install FFmpeg or manually create the comparison")

def fix_dead_reckoning_filter():
    """
    Apply the fix to DeadReckoningFilter in model.py
    This function patches the DeadReckoningFilter.update method to fix the
    accelerometer orientation calculation issue.
    
    The main issue is that the original code replaces the orientation with 
    the tilt correction quaternion instead of applying the tilt as a relative correction.
    """
    from model import Quaternion, Vector
    # Get the original update method
    original_update = DeadReckoningFilter.update
    
    # Define a fixed version that correctly applies accelerometer correction
    def fixed_update(self, sensor_data):
        """
        Update position and orientation based on sensor readings with fixed
        accelerometer correction.
        
        This fixed version correctly applies the tilt quaternion to the current
        orientation instead of replacing it, making the accelerometer correction
        actually have an effect.
        """
        # Initialize time on first update
        if self.last_time is None:
            self.last_time = sensor_data.time
            return self.position, self.orientation
    
        # Calculate time delta
        dt = sensor_data.time - self.last_time
        self.last_time = sensor_data.time
        
        # Skip if dt is too small (prevent division by zero)
        if dt < 0.001:
            return self.position, self.orientation
            
        # Bias-corrected angular velocity
        gyro_x = sensor_data.gyroscope[0] - self.gyro_bias[0]
        gyro_y = sensor_data.gyroscope[1] - self.gyro_bias[1]
        gyro_z = sensor_data.gyroscope[2] - self.gyro_bias[2]
        
        # -------------------
        # GYROSCOPE INTEGRATION
        # -------------------
        
        # Convert gyroscope readings to quaternion rate of change
        q_dot = Quaternion(
            0.5 * (-self.orientation.x * gyro_x - self.orientation.y * gyro_y - self.orientation.z * gyro_z),
            0.5 * (self.orientation.w * gyro_x + self.orientation.y * gyro_z - self.orientation.z * gyro_y),
            0.5 * (self.orientation.w * gyro_y - self.orientation.x * gyro_z + self.orientation.z * gyro_x),
            0.5 * (self.orientation.w * gyro_z + self.orientation.x * gyro_y - self.orientation.y * gyro_x)
        )
        
        # Integrate orientation using first-order approximation
        gyro_orientation = Quaternion(
            self.orientation.w + q_dot.w * dt,
            self.orientation.x + q_dot.x * dt,
            self.orientation.y + q_dot.y * dt,
            self.orientation.z + q_dot.z * dt
        )
        
        # Normalize quaternion
        gyro_magnitude = math.sqrt(
            gyro_orientation.w**2 + 
            gyro_orientation.x**2 + 
            gyro_orientation.y**2 + 
            gyro_orientation.z**2
        )
        
        gyro_orientation.w /= gyro_magnitude
        gyro_orientation.x /= gyro_magnitude
        gyro_orientation.y /= gyro_magnitude
        gyro_orientation.z /= gyro_magnitude
        
        # -------------------
        # ACCELEROMETER TILT CORRECTION
        # -------------------
        
        # Normalize accelerometer data to get gravity direction
        accel_x, accel_y, accel_z = sensor_data.accelerometer
        accel_magnitude = math.sqrt(accel_x**2 + accel_y**2 + accel_z**2)
        
        # Only apply correction if acceleration is close to gravity
        gravity_threshold = 0.2  # 20% tolerance
        if abs(accel_magnitude - 9.8) < (9.8 * gravity_threshold):
            # Normalize accelerometer data to get unit vector in direction of gravity
            accel_x /= accel_magnitude
            accel_y /= accel_magnitude
            accel_z /= accel_magnitude
            
            # The measured acceleration vector in the sensor frame
            accel_body = Vector(accel_x, accel_y, accel_z)
            
            # The reference gravity vector in the global frame (pointing down)
            gravity_world = Vector(0, 0, -1)
            
            # Convert body acceleration to world frame using current orientation
            accel_body_quat = Quaternion(0, accel_x, accel_y, accel_z)
            orientation_conj = self._quaternion_conjugate(self.orientation)
            accel_world_quat = self._quaternion_multiply(
                self._quaternion_multiply(self.orientation, accel_body_quat),
                orientation_conj
            )
            accel_world = Vector(accel_world_quat.x, accel_world_quat.y, accel_world_quat.z)
            
            # Find the rotation axis (perpendicular to both vectors)
            tilt_axis = accel_world.cross(gravity_world)
            tilt_axis_magnitude = tilt_axis.norm()
            
            # Find the angle between the measured gravity and world up vector
            if tilt_axis_magnitude > 0.001:  # Avoid normalizing zero vector
                tilt_axis = tilt_axis / tilt_axis_magnitude
                
                # Calculate the cosine of the angle between vectors
                cos_angle = accel_world * gravity_world
                # Clamp to valid range to avoid floating point errors
                cos_angle = max(min(cos_angle, 1.0), -1.0)
                # Get the angle
                tilt_angle = math.acos(cos_angle)
                
                # Create a quaternion for the tilt correction
                half_angle = tilt_angle * 0.5
                tilt_quat = Quaternion(
                    math.cos(half_angle),
                    tilt_axis.x * math.sin(half_angle),
                    tilt_axis.y * math.sin(half_angle),
                    tilt_axis.z * math.sin(half_angle)
                )
                
                # FIX: Apply tilt correction to current orientation instead of replacing it
                accel_orientation = self._quaternion_multiply(tilt_quat, self.orientation)
            else:
                accel_orientation = gyro_orientation
        else:
            # If acceleration is not close to gravity, skip tilt correction
            accel_orientation = gyro_orientation
        
        # -------------------
        # COMPLEMENTARY FILTER
        # -------------------
        
        # Apply complementary filter to fuse gyroscope and accelerometer estimations
        self.orientation = self._quaternion_slerp(gyro_orientation, accel_orientation, self.alpha)
        
        # Double integrate acceleration to get position
        # First integration: velocity
        self.velocity_x = self.velocity_x + accel_x * dt if hasattr(self, 'velocity_x') else accel_x * dt
        self.velocity_y = self.velocity_y + accel_y * dt if hasattr(self, 'velocity_y') else accel_y * dt
        self.velocity_z = self.velocity_z + accel_z * dt if hasattr(self, 'velocity_z') else accel_z * dt
        
        # Second integration: position
        self.position.x += self.velocity_x * dt
        self.position.y += self.velocity_y * dt
        self.position.z += self.velocity_z * dt
        
        # Apply complementary filter with magnetometer for yaw correction
        if hasattr(sensor_data, 'magnetometer'):
            self._apply_magnetometer_correction(sensor_data.magnetometer)
        
        return self.position, self.orientation
    
    # Replace the original method with our fixed version
    DeadReckoningFilter.update = fixed_update
    print("Applied fix to DeadReckoningFilter.update method")

def main():
    """Main entry point for the headset comparison program"""
    # Define dimensions
    width = 512
    height = 512

    # Import necessary components from model.py
    from model import DeadReckoningFilter, Quaternion, Matrix4, Vec4
    
    # Fix the dead reckoning filter
    fix_dead_reckoning_filter()
    
    # Load the CSV data - use the relative path
    csv_contents = load_csv_data("../IMUData.csv")
    
    # Debug the first few IMU readings
    # debug_imu_data(csv_contents)
    
    # Try to load the model
    try:
        model = Model('data/headset.obj')
    except:
        print("Error loading headset model. Using backup path...")
        try:
            model = Model('../data/headset.obj')
        except Exception as e:
            print(f"Could not load model: {e}")
            print("Creating a simple cube model instead")
            # Create a simple cube model
            from vector import Vector
            
            class SimpleModel:
                def __init__(self):
                    # Define vertices for a simple cube
                    self.vertices = [
                        Vector(-1, -1, -1),  # 0
                        Vector(1, -1, -1),   # 1
                        Vector(1, 1, -1),    # 2
                        Vector(-1, 1, -1),   # 3
                        Vector(-1, -1, 1),   # 4
                        Vector(1, -1, 1),    # 5
                        Vector(1, 1, 1),     # 6
                        Vector(-1, 1, 1)     # 7
                    ]
                    
                    # Define faces (triangulated)
                    self.faces = [
                        [0, 1, 2], [0, 2, 3],  # Front
                        [1, 5, 6], [1, 6, 2],  # Right
                        [5, 4, 7], [5, 7, 6],  # Back
                        [4, 0, 3], [4, 3, 7],  # Left
                        [3, 2, 6], [3, 6, 7],  # Top
                        [4, 5, 1], [4, 1, 0]   # Bottom
                    ]
                    
                    self.scale = [1, 1, 1]
                    self.rot = [0, 0, 0]
                    self.trans = [0, 0, -12]
                    self.transform = Matrix4()
                    self.updateTransform()
                
                def normalizeGeometry(self):
                    pass  # Already normalized
                
                def setPosition(self, x, y, z):
                    self.trans = [x, y, z]
                    self.updateTransform()
                
                def setRotation(self, x, y, z):
                    self.rot = [x, y, z]
                    self.updateTransform()
                
                def updateTransform(self):
                    # Start with scaling
                    scale_matrix = Matrix4.scaling(self.scale[0])
                    
                    # Apply rotations
                    rot_x = Matrix4.rotation_x(self.rot[0])
                    rot_y = Matrix4.rotation_y(self.rot[1])
                    rot_z = Matrix4.rotation_z(self.rot[2])
                    
                    # Combine rotations using matrix multiplication
                    rotation = rot_z.multiply_matrix(rot_y.multiply_matrix(rot_x))
                    
                    # Apply translation
                    trans = Matrix4.translation(*self.trans)
                    
                    # Combine all transformations
                    self.transform = trans.multiply_matrix(rotation.multiply_matrix(scale_matrix))
                
                def getTransformedVertex(self, index):
                    vertex = self.vertices[index]
                    # Transform the vertex using our transformation matrix
                    transformed = self.transform.multiply_vector(vertex)
                    # Return the transformed vertex's x, y, z components
                    return Vector(transformed.x, transformed.y, transformed.z)
            
            model = SimpleModel()
    
    # Run the comparison with real data
    compare_filter_methods(csv_contents, model, width, height)
    
    print("Comparison complete. Videos saved in comparison_results/ directory")

if __name__ == "__main__":
    main()