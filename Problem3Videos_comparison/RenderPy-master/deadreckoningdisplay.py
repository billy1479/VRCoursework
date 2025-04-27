#!/usr/bin/env python3
"""
Headset IMU Visualization

This script visualizes a headset's movement using IMU data. It loads a headset model
and animates it according to the orientation data from the IMU dataset.
"""

import pygame
import math
import os
import sys
from image import Image, Color
from model import DeadReckoningFilter, Model, Quaternion, Matrix4, Vec4, SensorDataParser
from shape import Triangle, Point
from vector import Vector
from video_recorder import VideoRecorder

# Define rendering window dimensions
SCREEN_WIDTH = 800
SCREEN_HEIGHT = 600

def load_csv_data(file_path):
    """Load and parse IMU data from CSV file"""
    parser = SensorDataParser(file_path)
    sensor_data = parser.parse()
    print(f"Loaded {len(sensor_data)} sensor data entries from {file_path}")
    return sensor_data

def quaternion_to_euler(q):
    """Convert quaternion to Euler angles (roll, pitch, yaw)"""
    # Roll (x-axis rotation)
    sinr_cosp = 2.0 * (q.w * q.x + q.y * q.z)
    cosr_cosp = 1.0 - 2.0 * (q.x * q.x + q.y * q.y)
    roll = math.atan2(sinr_cosp, cosr_cosp)
    
    # Pitch (y-axis rotation)
    sinp = 2.0 * (q.w * q.y - q.z * q.x)
    if abs(sinp) >= 1:
        pitch = math.copysign(math.pi / 2, sinp)  # Use 90 degrees if out of range
    else:
        pitch = math.asin(sinp)
        
    # Yaw (z-axis rotation)
    siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
    cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
    yaw = math.atan2(siny_cosp, cosy_cosp)
    
    return roll, pitch, yaw

def render_model(model, image, z_buffer, width, height):
    """Render the 3D model"""
    # Define the light direction for shading
    light_dir = Vector(0.5, 0.5, -1).normalize()
    
    # Define perspective projection parameters
    fov = math.pi / 3.0  # 60-degree field of view
    aspect = width / height
    near = 0.1
    far = 100.0
    
    # Create perspective projection matrix
    perspective_matrix = Matrix4.perspective(fov, aspect, near, far)
    
    # Calculate face normals (for lighting and backface culling)
    face_normals = {}
    for face in model.faces:
        p0 = model.getTransformedVertex(face[0])
        p1 = model.getTransformedVertex(face[1])
        p2 = model.getTransformedVertex(face[2])
        face_normal = (p2-p0).cross(p1-p0).normalize()

        for i in face:
            if i not in face_normals:
                face_normals[i] = []
            face_normals[i].append(face_normal)
    
    # Calculate vertex normals
    vertex_normals = []
    for vert_index in range(len(model.vertices)):
        if vert_index in face_normals:
            # Average the face normals connected to this vertex
            vert_norm = Vector(0, 0, 0)
            for normal in face_normals[vert_index]:
                vert_norm = vert_norm + normal
            if len(face_normals[vert_index]) > 0:
                vert_norm = vert_norm / len(face_normals[vert_index])
                vertex_normals.append(vert_norm.normalize())
            else:
                vertex_normals.append(Vector(0, 0, 1))
        else:
            # If vertex isn't used in any face
            vertex_normals.append(Vector(0, 0, 1))
    
    # Render all faces
    for face in model.faces:
        if len(face) != 3 or max(face) >= len(model.vertices):
            continue  # Skip invalid faces
            
        # Get transformed vertices
        p0 = model.getTransformedVertex(face[0])
        p1 = model.getTransformedVertex(face[1])
        p2 = model.getTransformedVertex(face[2])
        
        # Get vertex normals
        if face[0] < len(vertex_normals) and face[1] < len(vertex_normals) and face[2] < len(vertex_normals):
            n0, n1, n2 = [vertex_normals[i] for i in face]
        else:
            # Use face normal as fallback
            face_normal = (p2-p0).cross(p1-p0).normalize()
            n0 = n1 = n2 = face_normal
        
        # Backface culling - skip faces facing away from the camera
        if (p0.z + p1.z + p2.z) / 3.0 > 0:  # Simple check for faces behind the camera
            continue
        
        # Calculate lighting for each vertex
        intensity0 = max(0.2, n0 * light_dir)  # Add ambient light (0.2)
        intensity1 = max(0.2, n1 * light_dir)
        intensity2 = max(0.2, n2 * light_dir)
        
        # Apply perspective projection to vertices
        v0 = perspective_matrix.multiply_vector(Vec4(p0.x, p0.y, p0.z))
        v1 = perspective_matrix.multiply_vector(Vec4(p1.x, p1.y, p1.z))
        v2 = perspective_matrix.multiply_vector(Vec4(p2.x, p2.y, p2.z))
        
        # Perspective divide
        v0 = v0.perspectiveDivide()
        v1 = v1.perspectiveDivide()
        v2 = v2.perspectiveDivide()
        
        # Convert to screen coordinates
        screen_x0 = int((v0.x + 1.0) * width / 2.0)
        screen_y0 = int((v0.y + 1.0) * height / 2.0)
        screen_x1 = int((v1.x + 1.0) * width / 2.0)
        screen_y1 = int((v1.y + 1.0) * height / 2.0)
        screen_x2 = int((v2.x + 1.0) * width / 2.0)
        screen_y2 = int((v2.y + 1.0) * height / 2.0)
        
        # Create triangle points with lighting
        p0 = Point(screen_x0, screen_y0, v0.z, Color(int(intensity0*200), int(intensity0*200), int(intensity0*200), 255))
        p1 = Point(screen_x1, screen_y1, v1.z, Color(int(intensity1*200), int(intensity1*200), int(intensity1*200), 255))
        p2 = Point(screen_x2, screen_y2, v2.z, Color(int(intensity2*200), int(intensity2*200), int(intensity2*200), 255))
        
        # Draw the triangle
        Triangle(p0, p1, p2).draw_faster(image, z_buffer)

def update_display(image, screen):
    """Update the Pygame display with the rendered image"""
    # Create a new surface for the rendered image
    surface = pygame.Surface((image.width, image.height), pygame.SRCALPHA)
    
    # Convert the image buffer to pygame format
    for y in range(image.height):
        for x in range(image.width):
            # Calculate the index in the buffer
            flipY = (image.height - y - 1)  # Flip Y coordinate to match Image class convention
            index = (flipY * image.width + x) * 4 + flipY + 1  # +1 for the null byte at start of row
            
            # Extract RGBA values from the buffer
            r = image.buffer[index]
            g = image.buffer[index + 1]
            b = image.buffer[index + 2]
            a = image.buffer[index + 3]
            
            # Set the pixel on the surface
            if a > 0:  # Only draw non-transparent pixels
                surface.set_at((x, y), (r, g, b, a))
    
    # Blit the rendered image onto the screen
    screen.blit(surface, (0, 0))

def draw_text(surface, text, position, font, color):
    """Draw text on the surface centered at the given position"""
    text_surface = font.render(text, True, color)
    text_rect = text_surface.get_rect()
    text_rect.center = position
    surface.blit(text_surface, text_rect)

def visualize_headset_movement(csv_file_path, model_path="data/headset.obj", output_video="headset_movement.mp4"):
    """
    Visualize headset movement using IMU data
    
    Args:
        csv_file_path: Path to the IMU data CSV file
        model_path: Path to the 3D model file
        output_video: Filename for output video
    """
    # Create output directory
    output_dir = "output"
    os.makedirs(output_dir, exist_ok=True)
    
    # Initialize pygame
    pygame.init()
    screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
    pygame.display.set_caption("Headset IMU Visualization")
    
    # Create video recorder
    recorder = VideoRecorder(SCREEN_WIDTH, SCREEN_HEIGHT, fps=30, output_dir=output_dir)
    recorder.start_recording()
    
    # Load the headset model
    try:
        print(f"Loading model from {model_path}")
        model = Model(model_path)
        model.normalizeGeometry()
        # Position the model much farther away to make it appear much smaller
        model.setPosition(0, 0, -50)
        # Apply a smaller scale factor
        model.scale = [0.3, 0.3, 0.3]
        model.updateTransform()
    except Exception as e:
        print(f"Error loading model from {model_path}: {e}")
        
        # Try with alternate path
        try:
            alt_path = "data/headset.obj"
            print(f"Trying alternate path: {alt_path}")
            model = Model(alt_path)
            model.normalizeGeometry()
            model.setPosition(0, 0, -6)
            model.scale = [0.6, 0.6, 0.6]
            model.updateTransform()
        except Exception as e2:
            print(f"Error loading model from alternate path: {e2}")
            print("Failed to load headset model")
            return None
    
    # Load sensor data
    sensor_data = load_csv_data(csv_file_path)
    
    # Create the dead reckoning filter
    dr_filter = DeadReckoningFilter()
    
    # Calibrate filter using first 100 samples (assuming device at rest)
    if len(sensor_data) > 100:
        dr_filter.calibrate(sensor_data[:100])
        print("Filter calibrated")
    
    # Initialize z-buffer for depth testing
    z_buffer = [-float('inf')] * SCREEN_WIDTH * SCREEN_HEIGHT
    
    # Setup clock for timing
    clock = pygame.time.Clock()
    running = True
    
    # Define background color
    background_color = (255, 255, 255)  # White background
    
    # Font for minimal text overlays
    pygame.font.init()
    small_font = pygame.font.SysFont('Arial', 16)
    
    # Use all frames from the dataset
    max_frames = len(sensor_data)  # Process all available samples
    
    # Speed up factor - process multiple IMU samples per frame
    speed_factor = 1  # Process 5 samples per frame - makes simulation 5x faster
    
    # Main visualization loop
    current_data_index = 0
    frame_count = 0
    
    print(f"Processing all {max_frames} samples with speed factor {speed_factor}x")
    
    while running and current_data_index < max_frames:
        # Handle events
        for event in pygame.event.get():
            if event.type == pygame.QUIT or (event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE):
                running = False
        
        # Get current sensor data
        current_sensor_data = sensor_data[current_data_index]
        
        # Update the filter - returns a quaternion
        orientation = dr_filter.update(current_sensor_data)
        
        # Process multiple samples per frame to speed up the simulation
        for _ in range(speed_factor - 1):
            current_data_index += 1
            if current_data_index >= max_frames:
                break
            if current_data_index < len(sensor_data):
                next_data = sensor_data[current_data_index]
                orientation = dr_filter.update(next_data)
        
        # Convert quaternion to Euler angles
        roll, pitch, yaw = quaternion_to_euler(orientation)
        
        # Clear screen
        screen.fill(background_color)
        
        # Add debug information
        # Convert rotation to degrees for display
        roll_deg = math.degrees(roll)
        pitch_deg = math.degrees(pitch)
        yaw_deg = math.degrees(yaw)
        
        # Get raw sensor data for debugging
        gyro_x, gyro_y, gyro_z = current_sensor_data.gyroscope
        accel_x, accel_y, accel_z = current_sensor_data.accelerometer
        
        # Create text for debug info
        roll_text = small_font.render(f"Roll: {roll_deg:.2f}°", True, (255, 0, 0))
        pitch_text = small_font.render(f"Pitch: {pitch_deg:.2f}°", True, (0, 128, 0))
        yaw_text = small_font.render(f"Yaw: {yaw_deg:.2f}°", True, (0, 0, 255))
        
        # Sensor data text
        gyro_text = small_font.render(f"Gyro: X:{gyro_x:.3f} Y:{gyro_y:.3f} Z:{gyro_z:.3f}", True, (128, 0, 128))
        accel_text = small_font.render(f"Accel: X:{accel_x:.3f} Y:{accel_y:.3f} Z:{accel_z:.3f}", True, (0, 128, 128))
        
        sample_text = small_font.render(f"Sample: {current_data_index}/{max_frames}", True, (0, 0, 0))
        time_text = small_font.render(f"Time: {current_sensor_data.time:.2f}s", True, (0, 0, 0))
        
        # Position and draw text
        screen.blit(roll_text, (10, 10))
        screen.blit(pitch_text, (10, 30))
        screen.blit(yaw_text, (10, 50))
        screen.blit(gyro_text, (10, 80))
        screen.blit(accel_text, (10, 100))
        screen.blit(sample_text, (10, 130))
        screen.blit(time_text, (10, 150))
        
        # Reset image and z-buffer for 3D rendering
        image = Image(SCREEN_WIDTH, SCREEN_HEIGHT, Color(0, 0, 0, 0))  # Transparent background
        z_buffer = [-float('inf')] * SCREEN_WIDTH * SCREEN_HEIGHT
        
        # Apply rotation to the model
        model.setRotation(roll, pitch, yaw)
        
        # Render the model
        render_model(model, image, z_buffer, SCREEN_WIDTH, SCREEN_HEIGHT)
        
        # Convert image buffer to pygame surface and display
        update_display(image, screen)
        
        # Add minimal text overlay - just a progress indicator at the bottom
        progress = current_data_index / max_frames * 100
        progress_text = small_font.render(f"Progress: {progress:.1f}%", True, (0, 0, 0))
        screen.blit(progress_text, (10, SCREEN_HEIGHT - 25))
        
        # Update display
        pygame.display.flip()
        
        # Capture frame for video
        recorder.capture_frame(screen)
        
        # Control frame rate
        clock.tick(30)
        
        # Move to next data point
        current_data_index += 1
        frame_count += 1
    
    # Stop recording and save video
    recorder.stop_recording()
    print("Recording stopped, saving video...")
    
    # Try to save video using OpenCV first
    output_path = os.path.join(output_dir, output_video)
    video_saved = recorder.save_video(output_video)
    
    # If OpenCV fails, try FFmpeg
    if not video_saved:
        print("Trying FFmpeg for video generation...")
        output_path = recorder.generate_ffmpeg_video()
        
    # Clean up
    pygame.quit()
    
    if output_path and os.path.exists(output_path):
        print(f"Visualization complete! Video saved to {output_path}")
        return output_path
    else:
        print("Warning: Video may not have been saved properly")
        return None

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
    
    try:
        # Run the visualization
        output_path = visualize_headset_movement(imu_data_path, model_path, "headset_movement.mp4")
        
        if output_path:
            print(f"Video saved to: {output_path}")
            return 0
        else:
            print("Visualization failed")
            return 1
            
    except Exception as e:
        print(f"Error during visualization: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    sys.exit(main())