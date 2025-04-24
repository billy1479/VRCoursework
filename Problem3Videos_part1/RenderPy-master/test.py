import pygame
import math
import os
from image import Color, Image
from model import Model, DeadReckoningFilter, SensorDataParser, Quaternion, Matrix4, Vec4
from shape import Triangle, Point
from vector import Vector
from video_recorder import VideoRecorder

def load_csv_data(file_path):
    """Load and parse sensor data from CSV file"""
    parser = SensorDataParser(file_path)
    sensor_data = parser.parse()
    print(f"Loaded {len(sensor_data)} sensor data entries")
    return sensor_data

def visualize_headset_movement(csv_file_path, output_video="headset_movement.mp4"):
    """
    Visualize headset movement using IMU data
    
    Args:
        csv_file_path: Path to the IMU data CSV file
        output_video: Filename for output video
    """
    # Initialize pygame
    pygame.init()
    
    # Set up display parameters
    width, height = 800, 600
    screen = pygame.display.set_mode((width, height))
    pygame.display.set_caption("Headset Movement Visualization")
    
    # Create video recorder
    recorder = VideoRecorder(width, height, fps=30)
    recorder.start_recording()
    
    # Load the headset model
    try:
        model = Model('data/headset.obj')
        model.normalizeGeometry()
        # Position the model in front of the camera
        model.setPosition(0, 0, -4)
    except FileNotFoundError:
        print(f"Headset model not found at data/headset.obj")
        return
    
    # Load sensor data
    sensor_data = load_csv_data(csv_file_path)
    
    # Create the dead reckoning filter
    dr_filter = DeadReckoningFilter()
    
    # Calibrate filter using first 100 samples (assuming device at rest)
    if len(sensor_data) > 100:
        dr_filter.calibrate(sensor_data[:100])
        print("Filter calibrated")
    
    # Initialize z-buffer for depth testing
    z_buffer = [-float('inf')] * width * height
    
    # Setup clock for timing
    clock = pygame.time.Clock()
    running = True
    frame_count = 0
    
    # Define colors for visualization
    background_color = (240, 240, 240)  # Light gray background
    grid_color = (200, 200, 200)  # Lighter gray for grid
    
    # Track orientation over time for visualization
    orientation_history = []
    max_history = 100  # Number of historical positions to show
    
    # Font for text
    pygame.font.init()
    font = pygame.font.SysFont('Arial', 16)
    
    # Main visualization loop
    current_data_index = 0
    
    # Set maximum frames to process (to ensure it doesn't run too long)
    max_frames = min(len(sensor_data), 2000)  # Process up to 2000 frames
    
    while running and current_data_index < max_frames:
        # Handle events
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
        
        # Get current sensor data
        current_sensor_data = sensor_data[current_data_index]
        
        # Update the filter - returns a quaternion
        orientation = dr_filter.update(current_sensor_data)
        
        # Convert quaternion to Euler angles
        roll, pitch, yaw = quaternion_to_euler(orientation)
        
        # Store history for trail visualization
        orientation_history.append((roll, pitch, yaw))
        
        # Limit history size
        if len(orientation_history) > max_history:
            orientation_history.pop(0)
        
        # Clear screen
        screen.fill(background_color)
        
        # Draw reference grid
        draw_grid(screen, width, height)
        
        # Draw orientation history trail
        draw_orientation_trail(screen, orientation_history, (0, 128, 255), width, height)
        
        # Reset image and z-buffer for 3D rendering
        image = Image(width, height, Color(0, 0, 0, 0))  # Transparent background
        z_buffer = [-float('inf')] * width * height
        
        # Apply rotation to the model
        model.setRotation(roll, pitch, yaw)
        
        # Render the model
        render_model(model, image, z_buffer, width, height)
        
        # Convert image buffer to pygame surface and display
        update_display(image, screen)
        
        # Add text overlays - show current orientation values
        roll_deg, pitch_deg, yaw_deg = math.degrees(roll), math.degrees(pitch), math.degrees(yaw)
        info_text = f"Roll: {roll_deg:.1f}° Pitch: {pitch_deg:.1f}° Yaw: {yaw_deg:.1f}°"
        draw_text(screen, info_text, (width//2, height-30), font, (0, 0, 0))
        
        # Add time counter
        time_text = f"Time: {current_sensor_data.time:.2f}s"
        draw_text(screen, time_text, (width//2, 20), font, (0, 0, 0))
        
        # Add frame counter
        frame_text = f"Frame: {current_data_index}/{max_frames}"
        draw_text(screen, frame_text, (width//2, 50), font, (0, 0, 0))
        
        # Capture frame for video
        recorder.capture_frame(screen)
        
        # Update display
        pygame.display.flip()
        
        # Control frame rate
        clock.tick(30)
        
        # Move to next data point
        current_data_index += 1
        frame_count += 1
    
    # Stop recording and save video
    recorder.stop_recording()
    output_path = recorder.save_video(output_video)
    if not output_path:
        output_path = recorder.generate_ffmpeg_video()
    
    # Clean up
    pygame.quit()
    print(f"Video saved to {output_path}")
    return output_path

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

def draw_grid(surface, width, height):
    """Draw a reference grid on the screen"""
    color = (200, 200, 200)  # Light gray
    
    # Draw horizontal lines
    for y in range(0, height, 50):
        pygame.draw.line(surface, color, (0, y), (width, y), 1)
    
    # Draw vertical lines
    for x in range(0, width, 50):
        pygame.draw.line(surface, color, (x, 0), (x, height), 1)
    
    # Draw center lines darker
    center_color = (150, 150, 150)
    pygame.draw.line(surface, center_color, (width//2, 0), (width//2, height), 2)
    pygame.draw.line(surface, center_color, (0, height//2), (width, height//2), 2)

def draw_orientation_trail(surface, history, color, width, height):
    """Draw the orientation history as trails"""
    if len(history) < 2:
        return
    
    # Scale factors to convert rotations to pixel coordinates
    scale_x = width // 8
    scale_y = height // 8
    center_x = width // 2
    center_y = height // 2
    
    # Draw points and lines to show orientation history
    points = []
    for i, (roll, pitch, yaw) in enumerate(history):
        x = center_x + roll * scale_x
        y = center_y - pitch * scale_y  # Invert Y for screen coordinates
        alpha = min(255, 100 + 155 * (i / len(history)))  # Fade older points
        point_color = (color[0], color[1], color[2], int(alpha))
        points.append((int(x), int(y)))
    
    # Draw connecting lines
    for i in range(1, len(points)):
        pygame.draw.line(surface, color, points[i-1], points[i], 2)
    
    # Draw circle at the latest point
    if points:
        pygame.draw.circle(surface, (255, 0, 0), points[-1], 4)

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

if __name__ == "__main__":
    # Run the visualization
    csv_file_path = "../IMUData.csv"
    output_video = "headset_movement.mp4"
    
    try:
        visualize_headset_movement(csv_file_path, output_video)
    except Exception as e:
        print(f"Error during visualization: {e}")
        import traceback
        traceback.print_exc()