#!/usr/bin/env python3
"""
Simultaneous Three Headset Alpha Comparison

This script creates a side-by-side visualization of three headsets using different
alpha values processing the same IMU data simultaneously to visualize drift in real-time.
It uses the existing rendering code from render.py and other modules.
"""

from matplotlib import pyplot as plt
from image import Image, Color
from model import DeadReckoningFilter, Model, Quaternion, Matrix4, Vec4, SensorData, SensorDataParser
from shape import Point, Line, Triangle
from vector import Vector
from video_recorder import VideoRecorder
import pygame
import numpy as np
import math
import os
import sys

# Define rendering window dimensions
SCREEN_WIDTH = 1536  # Divisible by 3 to have equal space for each headset
SCREEN_HEIGHT = 720
MODEL_WIDTH = SCREEN_WIDTH // 3
MODEL_HEIGHT = SCREEN_HEIGHT

def load_csv_data(file_path):
    """Load and parse IMU data from CSV file"""
    parser = SensorDataParser(file_path)
    sensor_data = parser.parse()
    print(f"Loaded {len(sensor_data)} sensor data entries from {file_path}")
    return sensor_data

def getPerspectiveProjection(x, y, z, width, height):
    """Apply perspective projection to convert 3D coordinates to screen coordinates"""
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

def getVertexNormal(vertIndex, faceNormalsByVertex):
    """Compute vertex normals by averaging the normals of adjacent faces"""
    normal = Vector(0, 0, 0)
    for adjNormal in faceNormalsByVertex[vertIndex]:
        normal = normal + adjNormal
    return normal / len(faceNormalsByVertex[vertIndex])

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


def render_three_models(csv_contents, model_path="data/headset.obj", output_video="three_alpha_comparison.mp4"):
    """
    Renders three instances of a model side by side with different alpha values
    to compare drift in real-time.
    """
    # Create output directory
    output_dir = "alpha_comparison"
    os.makedirs(output_dir, exist_ok=True)
    
    # Initialize pygame
    pygame.init()
    screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
    pygame.display.set_caption("Three Alpha Values Comparison")
    
    # Create a video recorder
    recorder = VideoRecorder(SCREEN_WIDTH, SCREEN_HEIGHT, fps=30, output_dir=output_dir)
    
    # Try to load the model
    try:
        # Try to load the model from provided path
        model = Model(model_path)
        # Create three copies of the model for each filter
        models = [Model(model_path) for _ in range(3)]
        for m in models:
            m.normalizeGeometry()
            m.setPosition(0, 0, -12)
    except Exception as e:
        print(f"Error loading model from {model_path}: {e}")
        
        # Try alternative path
        try:
            alt_path = "../data/headset.obj"
            print(f"Trying alternative path: {alt_path}")
            models = [Model(alt_path) for _ in range(3)]
            for m in models:
                m.normalizeGeometry()
                m.setPosition(0, 0, -12)
        except Exception as e2:
            print(f"Error loading model from alternative path: {e2}")
            print("Creating simple cube models instead")
            
            # Create simple cube models
            class SimpleCubeModel:
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
            
            models = [SimpleCubeModel() for _ in range(3)]
    
    # Define alpha values for comparison (pure gyro, balanced, heavy accel correction)
    alpha_values = [1.0, 0.5, 0.1]
    
    # Create filters for each alpha value
    filters = []
    for alpha in alpha_values:
        dr_filter = DeadReckoningFilter(alpha=alpha)
        dr_filter.calibrate(csv_contents[:100])  # Calibrate with first 100 samples
        filters.append(dr_filter)
    
    # Init images and z-buffers for rendering
    images = [Image(MODEL_WIDTH, MODEL_HEIGHT, Color(255, 255, 255, 255)) for _ in range(3)]
    z_buffers = [[-float('inf')] * MODEL_WIDTH * MODEL_HEIGHT for _ in range(3)]
    
    # Create pygame clock for timing
    clock = pygame.time.Clock()
    
    # Buffer surface for drawing
    main_surface = pygame.Surface((SCREEN_WIDTH, SCREEN_HEIGHT))
    
    # Capture frequency - adjust to control video length
    capture_frequency = 1  # Capture every frame
    
    # Font for text overlays
    pygame.font.init()
    font = pygame.font.SysFont('Arial', 24)
    label_font = pygame.font.SysFont('Arial', 36)
    
    # Start recording
    recorder.start_recording()
    
    # Use the full dataset
    test_duration = len(csv_contents)
    print(f"Processing {test_duration} samples")
    
    # Process in batches
    batch_size = 2000
    batch_count = (test_duration + batch_size - 1) // batch_size
    
    # Set up for tracking orientation differences
    orientations_history = {alpha: [] for alpha in alpha_values}
    
    # Main processing loop
    running = True
    for batch in range(batch_count):
        if not running:
            break
            
        start_idx = batch * batch_size
        end_idx = min(start_idx + batch_size, test_duration)
        
        print(f"Processing batch {batch+1}/{batch_count} (samples {start_idx} to {end_idx-1})")
        
        for i in range(start_idx, end_idx):
            # Handle timing
            delta_time = clock.tick(60) / 1000.0
            
            # Get current sensor data
            current_sensor_data = csv_contents[i]
            
            # Clear the screen
            main_surface.fill((255, 255, 255))
            
            # Update and render each model
            current_orientations = []
            
            for j, (dr_filter, model, alpha) in enumerate(zip(filters, models, alpha_values)):
                # Update filter with current sensor data
                position, orientation = dr_filter.update(current_sensor_data)
                
                # Convert quaternion to Euler angles
                roll, pitch, yaw = dr_filter.get_euler_angles()
                current_orientations.append((roll, pitch, yaw))
                
                # Update model rotation
                model.setRotation(roll, pitch, yaw)
                
                # Reset image and z-buffer for new frame
                images[j] = Image(MODEL_WIDTH, MODEL_HEIGHT, Color(255, 255, 255, 255))
                z_buffers[j] = [-float('inf')] * MODEL_WIDTH * MODEL_HEIGHT
                
                # Calculate face normals
                faceNormals = {}
                for face in model.faces:
                    p0 = model.getTransformedVertex(face[0])
                    p1 = model.getTransformedVertex(face[1])
                    p2 = model.getTransformedVertex(face[2])
                    faceNormal = (p2-p0).cross(p1-p0).normalize()

                    for idx in face:
                        if idx not in faceNormals:
                            faceNormals[idx] = []

                        faceNormals[idx].append(faceNormal)

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
                    n0, n1, n2 = [vertexNormals[idx] for idx in face]

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
                            
                        screenX, screenY = getPerspectiveProjection(p.x, p.y, p.z, MODEL_WIDTH, MODEL_HEIGHT)
                        
                        transformedPoints.append(Point(screenX, screenY, p.z, Color(intensity*255, intensity*255, intensity*255, 255)))

                    if not cull:
                        Triangle(transformedPoints[0], transformedPoints[1], transformedPoints[2]).draw_faster(images[j], z_buffers[j])
                
                # Copy the rendered image to the main surface
                offset_x = j * MODEL_WIDTH
                
                # Copy image data to pygame surface
                for y in range(MODEL_HEIGHT):
                    for x in range(MODEL_WIDTH):
                        flipY = (MODEL_HEIGHT - y - 1)
                        index = (flipY * MODEL_WIDTH + x) * 4 + flipY + 1
                        
                        r = images[j].buffer[index]
                        g = images[j].buffer[index + 1]
                        b = images[j].buffer[index + 2]
                        
                        main_surface.set_at((x + offset_x, y), (r, g, b))
                
                # Add label for this filter
                label_text = f"Alpha: {alpha}"
                if alpha == 1.0:
                    label_text = "Gyro Only (α=1.0)"
                label = label_font.render(label_text, True, (0, 0, 0))
                main_surface.blit(label, (offset_x + 20, 20))
                
                # Add orientation values
                roll_deg, pitch_deg, yaw_deg = math.degrees(roll), math.degrees(pitch), math.degrees(yaw)
                roll_text = font.render(f"Roll: {roll_deg:.1f}°", True, (255, 0, 0))
                pitch_text = font.render(f"Pitch: {pitch_deg:.1f}°", True, (0, 128, 0))
                yaw_text = font.render(f"Yaw: {yaw_deg:.1f}°", True, (0, 0, 255))
                
                main_surface.blit(roll_text, (offset_x + 20, 70))
                main_surface.blit(pitch_text, (offset_x + 20, 100))
                main_surface.blit(yaw_text, (offset_x + 20, 130))
                
                # Store orientation for analysis
                orientations_history[alpha].append((roll_deg, pitch_deg, yaw_deg))
            
            # Calculate orientation differences between models
            if len(current_orientations) > 1:
                gyro_roll, gyro_pitch, gyro_yaw = current_orientations[0]  # Gyro only (α=1.0)
                
                # Calculate difference between pure gyro and middle alpha
                diff_text = font.render("Difference from Gyro-only:", True, (0, 0, 0))
                main_surface.blit(diff_text, (SCREEN_WIDTH // 2 - 150, MODEL_HEIGHT - 120))
                
                for j in range(1, len(alpha_values)):
                    alpha = alpha_values[j]
                    roll, pitch, yaw = current_orientations[j]
                    
                    # Calculate differences
                    roll_diff = math.degrees(roll - gyro_roll)
                    pitch_diff = math.degrees(pitch - gyro_pitch)
                    yaw_diff = math.degrees(yaw - gyro_yaw)
                    
                    # Display differences
                    diff_pos_x = j * MODEL_WIDTH + 20
                    roll_diff_text = font.render(f"Roll Δ: {roll_diff:.1f}°", True, (255, 0, 0))
                    pitch_diff_text = font.render(f"Pitch Δ: {pitch_diff:.1f}°", True, (0, 128, 0))
                    yaw_diff_text = font.render(f"Yaw Δ: {yaw_diff:.1f}°", True, (0, 0, 255))
                    
                    main_surface.blit(roll_diff_text, (diff_pos_x, MODEL_HEIGHT - 90))
                    main_surface.blit(pitch_diff_text, (diff_pos_x, MODEL_HEIGHT - 60))
                    main_surface.blit(yaw_diff_text, (diff_pos_x, MODEL_HEIGHT - 30))
            
            # Add progress indicator
            progress_text = font.render(f"Sample: {i}/{test_duration} ({i/test_duration*100:.1f}%)", True, (0, 0, 0))
            main_surface.blit(progress_text, (SCREEN_WIDTH // 2 - 120, 5))
            
            # Update display
            screen.blit(main_surface, (0, 0))
            pygame.display.flip()
            
            # Capture frame for video
            if i % capture_frequency == 0:
                recorder.capture_frame(screen)
            
            # Handle events
            for event in pygame.event.get():
                if event.type == pygame.QUIT or (event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE):
                    running = False
                    break
        
        # Save intermediate video part every batch
        if batch < batch_count - 1 and running:
            recorder.stop_recording()
            part_name = f"part{batch+1}.mp4"
            print(f"Saving intermediate part: {part_name}")
            recorder.save_video(part_name)
            recorder.clear_frames()
            recorder.start_recording()
    
    # Stop recording and save final video
    recorder.stop_recording()
    recorder.save_video(output_video)
    
    # Plot the orientation differences over time
    plot_orientation_differences(orientations_history, output_dir)
    
    # Clean up pygame
    pygame.quit()
    
    # Combine video parts if there are multiple
    try:
        import glob
        import subprocess
        
        # Look for part files
        part_files = sorted(glob.glob(f"{output_dir}/part*.mp4"))
        
        if len(part_files) > 1:
            print(f"Combining {len(part_files)} video parts...")
            
            # Create a file list for ffmpeg
            with open(f"{output_dir}/parts.txt", "w") as f:
                for part in part_files:
                    f.write(f"file '{os.path.basename(part)}'\n")
            
            # Use ffmpeg to concatenate the parts
            cmd = f"ffmpeg -f concat -safe 0 -i {output_dir}/parts.txt -c copy {output_dir}/full_{output_video}"
            subprocess.run(cmd, shell=True, check=True)
            
            print(f"Combined video saved as full_{output_video}")
    except Exception as e:
        print(f"Error combining video parts: {e}")
    
    print(f"Comparison complete! Video saved as {output_dir}/{output_video}")

def plot_orientation_differences(orientations_history, output_dir):
    """
    Plot the orientation differences between different alpha values over time.
    """
    try:
        alphas = list(orientations_history.keys())
        if len(alphas) < 2:
            return
            
        # Use the pure gyro as reference
        gyro_alpha = 1.0
        gyro_data = orientations_history[gyro_alpha]
        
        # Create time axis
        time_points = list(range(len(gyro_data)))
        
        # Plot differences for other alphas
        plt.figure(figsize=(12, 10))
        
        # Plot roll differences
        plt.subplot(3, 1, 1)
        for alpha in alphas:
            if alpha == gyro_alpha:
                continue
                
            alpha_data = orientations_history[alpha]
            # Ensure same length
            min_len = min(len(gyro_data), len(alpha_data))
            
            # Calculate roll differences
            roll_diffs = [alpha_data[i][0] - gyro_data[i][0] for i in range(min_len)]
            
            plt.plot(time_points[:min_len], roll_diffs, label=f'α={alpha}')
        
        plt.title('Roll Difference from Gyro-only (α=1.0)')
        plt.ylabel('Degrees')
        plt.grid(True)
        plt.legend()
        
        # Plot pitch differences
        plt.subplot(3, 1, 2)
        for alpha in alphas:
            if alpha == gyro_alpha:
                continue
                
            alpha_data = orientations_history[alpha]
            # Ensure same length
            min_len = min(len(gyro_data), len(alpha_data))
            
            # Calculate pitch differences
            pitch_diffs = [alpha_data[i][1] - gyro_data[i][1] for i in range(min_len)]
            
            plt.plot(time_points[:min_len], pitch_diffs, label=f'α={alpha}')
        
        plt.title('Pitch Difference from Gyro-only (α=1.0)')
        plt.ylabel('Degrees')
        plt.grid(True)
        plt.legend()
        
        # Plot yaw differences
        plt.subplot(3, 1, 3)
        for alpha in alphas:
            if alpha == gyro_alpha:
                continue
                
            alpha_data = orientations_history[alpha]
            # Ensure same length
            min_len = min(len(gyro_data), len(alpha_data))
            
            # Calculate yaw differences
            yaw_diffs = [alpha_data[i][2] - gyro_data[i][2] for i in range(min_len)]
            
            plt.plot(time_points[:min_len], yaw_diffs, label=f'α={alpha}')
        
        plt.title('Yaw Difference from Gyro-only (α=1.0)')
        plt.xlabel('Sample Number')
        plt.ylabel('Degrees')
        plt.grid(True)
        plt.legend()
        
        plt.tight_layout()
        plt.savefig(f"{output_dir}/orientation_differences.png")
        plt.close()
        
        print(f"Orientation difference plots saved to {output_dir}/orientation_differences.png")
    except Exception as e:
        print(f"Error creating orientation plots: {e}")

def main():
    # Check command line arguments for IMU data file path
    if len(sys.argv) > 1:
        imu_data_path = sys.argv[1]
    else:
        imu_data_path = "../IMUData.csv"
    
    # Check for model path
    if len(sys.argv) > 2:
        model_path = sys.argv[2]
    else:
        model_path = "data/headset.obj"
    
    print(f"Using IMU data from: {imu_data_path}")
    print(f"Using model from: {model_path}")
    
    # Apply fix to dead reckoning filter
    # fix_dead_reckoning_filter()
    
    # Load IMU data
    try:
        csv_contents = load_csv_data(imu_data_path)
        
        # Print some debug info
        print(f"Loaded {len(csv_contents)} samples from IMU data")
        for i in range(min(5, len(csv_contents))):
            data = csv_contents[i]
            print(f"Sample {i}: Time={data.time:.3f}, Gyro=({data.gyroscope[0]:.3f}, {data.gyroscope[1]:.3f}, {data.gyroscope[2]:.3f})")
        
        # Render the comparison
        render_three_models(csv_contents, model_path, "three_alpha_comparison.mp4")
        
    except Exception as e:
        print(f"Error loading IMU data: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())