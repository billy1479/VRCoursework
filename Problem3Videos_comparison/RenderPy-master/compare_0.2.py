#!/usr/bin/env python3
"""
Headset Implementation Comparison

This script visualizes two headsets side by side, one using gyroscope-only tracking
and the other using gravity-corrected tracking, to show the drift in real-time.
"""

import pygame
import math
import os
import sys
import importlib.util

# Dynamic imports for the two implementations
def import_module_from_file(file_path, module_name):
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

# Import the two model implementations
model_gyro = import_module_from_file("model_gyro.py", "model_gyro")
model_grav = import_module_from_file("model_grav.py", "model_grav")

# Import other necessary modules
from image import Image, Color
from vector import Vector
from shape import Triangle, Point
from video_recorder import VideoRecorder

# Define rendering window dimensions
SCREEN_WIDTH = 1200  # Wide enough for two headsets side by side
SCREEN_HEIGHT = 600
MODEL_WIDTH = SCREEN_WIDTH // 2  # Half the screen width for each model
MODEL_HEIGHT = SCREEN_HEIGHT

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

def render_model(model, image, z_buffer, width, height, matrix4_class, vec4_class):
    """Render the 3D model"""
    # Define the light direction for shading
    light_dir = Vector(0.5, 0.5, -1).normalize()
    
    # Define perspective projection parameters
    fov = math.pi / 3.0  # 60-degree field of view
    aspect = width / height
    near = 0.1
    far = 100.0
    
    # Create perspective projection matrix
    perspective_matrix = matrix4_class.perspective(fov, aspect, near, far)
    
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
        v0 = perspective_matrix.multiply_vector(vec4_class(p0.x, p0.y, p0.z))
        v1 = perspective_matrix.multiply_vector(vec4_class(p1.x, p1.y, p1.z))
        v2 = perspective_matrix.multiply_vector(vec4_class(p2.x, p2.y, p2.z))
        
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

def update_display(image, screen, x_offset=0):
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
    
    # Blit the rendered image onto the screen at the specified offset
    screen.blit(surface, (x_offset, 0))

def draw_text(surface, text, position, font, color):
    """Draw text on the surface centered at the given position"""
    text_surface = font.render(text, True, color)
    text_rect = text_surface.get_rect()
    text_rect.center = position
    surface.blit(text_surface, text_rect)

def draw_divider(surface, width, height):
    """Draw a vertical divider line between the two models"""
    pygame.draw.line(surface, (100, 100, 100), (width//2, 0), (width//2, height), 2)

def compare_implementations(csv_file_path, model_path="data/headset.obj", output_video="headset_comparison.mp4"):
    """
    Visualize two headsets side by side using different tracking implementations
    
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
    pygame.display.set_caption("Headset Implementation Comparison")
    
    # Create video recorder
    recorder = VideoRecorder(SCREEN_WIDTH, SCREEN_HEIGHT, fps=30, output_dir=output_dir)
    recorder.start_recording()
    
    # Load the headset model twice - one for each implementation
    try:
        print(f"Loading model from {model_path}")
        model_gyro_instance = model_gyro.Model(model_path)
        model_gyro_instance.normalizeGeometry()
        model_gyro_instance.setPosition(0, 0, -50)
        model_gyro_instance.scale = [0.3,0.3,0.3]
        model_gyro_instance.updateTransform()
        
        model_grav_instance = model_grav.Model(model_path)
        model_grav_instance.normalizeGeometry()
        model_grav_instance.setPosition(0, 0, -50)
        model_grav_instance.scale = [0.3,0.3,0.3]
        model_grav_instance.updateTransform()
    except Exception as e:
        print(f"Error loading model from {model_path}: {e}")
        
        # Try with alternate path
        try:
            alt_path = "data/headset.obj"
            print(f"Trying alternate path: {alt_path}")
            model_gyro_instance = model_gyro.Model(alt_path)
            model_gyro_instance.normalizeGeometry()
            model_gyro_instance.setPosition(0, 0, -15)
            model_gyro_instance.scale = [0.6, 0.6, 0.6]
            model_gyro_instance.updateTransform()
            
            model_grav_instance = model_grav.Model(alt_path)
            model_grav_instance.normalizeGeometry()
            model_grav_instance.setPosition(0, 0, -15)
            model_grav_instance.scale = [0.6, 0.6, 0.6]
            model_grav_instance.updateTransform()
        except Exception as e2:
            print(f"Error loading model from alternate path: {e2}")
            print("Failed to load headset model")
            return None
    
    # Load sensor data using the parser from the first module (they're identical)
    parser = model_gyro.SensorDataParser(csv_file_path)
    sensor_data = parser.parse()
    print(f"Loaded {len(sensor_data)} sensor data entries")
    
    # Create the dead reckoning filters from each implementation
    gyro_filter = model_gyro.DeadReckoningFilter()  # Gyroscope-only implementation
    grav_filter = model_grav.DeadReckoningFilter(alpha=0.2)  # Gravity-corrected implementation
    
    # Make sure both filters have the same initial state
    print("Initializing filters with identical starting orientations")
    
    # Calibrate filters using first 100 samples (assuming device at rest)
    if len(sensor_data) > 100:
        gyro_filter.calibrate(sensor_data[:100])
        grav_filter.calibrate(sensor_data[:100])
        print("Filters calibrated with identical bias values")
    
    # Initialize z-buffers for depth testing
    z_buffer_gyro = [-float('inf')] * MODEL_WIDTH * MODEL_HEIGHT
    z_buffer_grav = [-float('inf')] * MODEL_WIDTH * MODEL_HEIGHT
    
    # Setup clock for timing
    clock = pygame.time.Clock()
    running = True
    
    # Define background color
    background_color = (255, 255, 255)  # White background
    
    # Font for text overlays
    pygame.font.init()
    small_font = pygame.font.SysFont('Arial', 16)
    medium_font = pygame.font.SysFont('Arial', 24)
    large_font = pygame.font.SysFont('Arial', 32)
    
    # Use all samples from the dataset
    max_frames = len(sensor_data)
    
    # Use a more moderate speed factor to ensure accurate comparison
    speed_factor = 1  # Process samples at regular speed for more accurate comparison
    
    # Main visualization loop
    current_data_index = 0
    frame_count = 0
    
    print(f"Processing all {max_frames} samples with speed factor {speed_factor}x")
    
    # Initialize angle differences for tracking drift
    roll_diff_history = []
    pitch_diff_history = []
    yaw_diff_history = []
    max_diff_history = 100  # Keep track of the last 100 differences
    
    while running and current_data_index < max_frames:
        # Handle events
        for event in pygame.event.get():
            if event.type == pygame.QUIT or (event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE):
                running = False
        
        # Get current sensor data
        current_sensor_data = sensor_data[current_data_index]
        
        # Update both filters
        gyro_orientation = gyro_filter.update(current_sensor_data)
        grav_orientation = grav_filter.update(current_sensor_data)
        
        # Process multiple samples per frame to speed up the simulation
        for _ in range(speed_factor - 1):
            current_data_index += 1
            if current_data_index >= max_frames:
                break
            if current_data_index < len(sensor_data):
                next_data = sensor_data[current_data_index]
                # Make sure both filters process exactly the same data
                gyro_orientation = gyro_filter.update(next_data)
                grav_orientation = grav_filter.update(next_data)
        
        # Convert quaternions to Euler angles
        gyro_roll, gyro_pitch, gyro_yaw = quaternion_to_euler(gyro_orientation)
        grav_roll, grav_pitch, grav_yaw = quaternion_to_euler(grav_orientation)
        
        # Calculate differences between implementations
        roll_diff = math.degrees(abs(grav_roll - gyro_roll))
        pitch_diff = math.degrees(abs(grav_pitch - gyro_pitch))
        yaw_diff = math.degrees(abs(grav_yaw - gyro_yaw))
        
        # Keep track of differences for display
        roll_diff_history.append(roll_diff)
        pitch_diff_history.append(pitch_diff)
        yaw_diff_history.append(yaw_diff)
        
        # Limit history size
        if len(roll_diff_history) > max_diff_history:
            roll_diff_history.pop(0)
            pitch_diff_history.pop(0)
            yaw_diff_history.pop(0)
        
        # Calculate current average differences
        avg_roll_diff = sum(roll_diff_history) / len(roll_diff_history)
        avg_pitch_diff = sum(pitch_diff_history) / len(pitch_diff_history)
        avg_yaw_diff = sum(yaw_diff_history) / len(yaw_diff_history)
        
        # Clear screen
        screen.fill(background_color)
        
        # Draw divider between models
        draw_divider(screen, SCREEN_WIDTH, SCREEN_HEIGHT)
        
        # Reset images and z-buffers for 3D rendering
        image_gyro = Image(MODEL_WIDTH, MODEL_HEIGHT, Color(0, 0, 0, 0))  # Transparent background
        image_grav = Image(MODEL_WIDTH, MODEL_HEIGHT, Color(0, 0, 0, 0))  # Transparent background
        z_buffer_gyro = [-float('inf')] * MODEL_WIDTH * MODEL_HEIGHT
        z_buffer_grav = [-float('inf')] * MODEL_WIDTH * MODEL_HEIGHT
        
        # Apply rotations to the models
        model_gyro_instance.setRotation(gyro_roll, gyro_pitch, gyro_yaw)
        model_grav_instance.setRotation(grav_roll, grav_pitch, grav_yaw)
        
        # Render the models
        render_model(model_gyro_instance, image_gyro, z_buffer_gyro, MODEL_WIDTH, MODEL_HEIGHT, 
                    model_gyro.Matrix4, model_gyro.Vec4)
        render_model(model_grav_instance, image_grav, z_buffer_grav, MODEL_WIDTH, MODEL_HEIGHT, 
                    model_grav.Matrix4, model_grav.Vec4)
        
        # Convert image buffers to pygame surfaces and display
        update_display(image_gyro, screen, 0)
        update_display(image_grav, screen, MODEL_WIDTH)
        
        # Add titles for each implementation
        draw_text(screen, "Gyroscope Only", (MODEL_WIDTH//2, 30), large_font, (0, 0, 0))
        draw_text(screen, "Gravity Corrected", (MODEL_WIDTH + MODEL_WIDTH//2, 30), large_font, (0, 0, 0))
        
        # Add orientation values for both implementations
        gyro_roll_deg, gyro_pitch_deg, gyro_yaw_deg = math.degrees(gyro_roll), math.degrees(gyro_pitch), math.degrees(gyro_yaw)
        grav_roll_deg, grav_pitch_deg, grav_yaw_deg = math.degrees(grav_roll), math.degrees(grav_pitch), math.degrees(grav_yaw)
        
        # Show gyro implementation values
        draw_text(screen, f"Roll: {gyro_roll_deg:.2f}°", (MODEL_WIDTH//2, 70), small_font, (255, 0, 0))
        draw_text(screen, f"Pitch: {gyro_pitch_deg:.2f}°", (MODEL_WIDTH//2, 90), small_font, (0, 128, 0))
        draw_text(screen, f"Yaw: {gyro_yaw_deg:.2f}°", (MODEL_WIDTH//2, 110), small_font, (0, 0, 255))
        
        # Show gravity-corrected implementation values
        draw_text(screen, f"Roll: {grav_roll_deg:.2f}°", (MODEL_WIDTH + MODEL_WIDTH//2, 70), small_font, (255, 0, 0))
        draw_text(screen, f"Pitch: {grav_pitch_deg:.2f}°", (MODEL_WIDTH + MODEL_WIDTH//2, 90), small_font, (0, 128, 0))
        draw_text(screen, f"Yaw: {grav_yaw_deg:.2f}°", (MODEL_WIDTH + MODEL_WIDTH//2, 110), small_font, (0, 0, 255))
        
        # Show sensor data
        gyro_x, gyro_y, gyro_z = current_sensor_data.gyroscope
        accel_x, accel_y, accel_z = current_sensor_data.accelerometer
        
        # Display sensor values
        draw_text(screen, f"Gyro: X:{gyro_x:.3f} Y:{gyro_y:.3f} Z:{gyro_z:.3f}", (SCREEN_WIDTH//2, 150), small_font, (128, 0, 128))
        draw_text(screen, f"Accel: X:{accel_x:.3f} Y:{accel_y:.3f} Z:{accel_z:.3f}", (SCREEN_WIDTH//2, 170), small_font, (0, 128, 128))
        
        # Show difference information
        diff_title = "Implementation Differences (degrees)"
        draw_text(screen, diff_title, (SCREEN_WIDTH//2, SCREEN_HEIGHT - 120), medium_font, (0, 0, 0))
        
        draw_text(screen, f"Roll Diff: {roll_diff:.2f}° (Avg: {avg_roll_diff:.2f}°)", (SCREEN_WIDTH//2, SCREEN_HEIGHT - 90), small_font, (255, 0, 0))
        draw_text(screen, f"Pitch Diff: {pitch_diff:.2f}° (Avg: {avg_pitch_diff:.2f}°)", (SCREEN_WIDTH//2, SCREEN_HEIGHT - 70), small_font, (0, 128, 0))
        draw_text(screen, f"Yaw Diff: {yaw_diff:.2f}° (Avg: {avg_yaw_diff:.2f}°)", (SCREEN_WIDTH//2, SCREEN_HEIGHT - 50), small_font, (0, 0, 255))
        
        # Add time and progress information
        progress = current_data_index / max_frames * 100
        time_text = f"Time: {current_sensor_data.time:.2f}s"
        progress_text = f"Progress: {progress:.1f}%"
        
        draw_text(screen, time_text, (SCREEN_WIDTH//2, SCREEN_HEIGHT - 150), medium_font, (0, 0, 0))
        draw_text(screen, progress_text, (SCREEN_WIDTH//2, SCREEN_HEIGHT - 25), small_font, (0, 0, 0))
        
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
        # Run the comparison
        output_path = compare_implementations(imu_data_path, model_path, "headset_comparison_0.2.mp4")
        
        if output_path:
            print(f"Video saved to: {output_path}")
            print("\nNote: You may need to adjust the alpha parameter in model_grav.DeadReckoningFilter")
            print("to see different levels of gravity correction influence.")
            print("A higher alpha (closer to 1.0) means more gyroscope influence.")
            print("A lower alpha (closer to 0.0) means more accelerometer influence.")
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