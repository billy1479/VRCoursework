import pygame
from image import Image, Color
from model import Model, DeadReckoningFilter, SensorDataParser, Quaternion
from vector import Vector
import math
import time
import os

# Import the VideoRecorder from the separate file
from video_recorder import VideoRecorder

class ComparativeHeadsetScene:
    """
    A 3D scene that compares two orientation filtering methods:
    - Left headset: Using only gyroscope data
    - Right headset: Using gyroscope + gravity-based tilt correction with accelerometer data
    
    This allows visualization of drift and correction differences between methods.
    """
    def __init__(self, width=800, height=600, csv_path="IMUData.csv", auto_record=True):
        # Initialize pygame
        pygame.init()
        self.width = width
        self.height = height
        self.screen = pygame.display.set_mode((width, height))
        pygame.display.set_caption("Orientation Filter Comparison - Gyro Only vs. Gravity Correction")
        
        # Create buffer surface for rendering
        self.buffer_surface = pygame.Surface((width, height))
        
        # Image and Z-buffer for rendering
        self.image = Image(width, height, Color(20, 20, 40, 255))  # Dark blue background
        self.zBuffer = [-float('inf')] * width * height
        
        # Fixed camera position with good view
        self.camera_pos = Vector(0, 10, -30)
        self.camera_target = Vector(0, 5, 0) 
        self.light_dir = Vector(0.5, -1, -0.5).normalize()

        # IMU playback speed multiplier
        self.imu_playback_speed = 3  # Process 3 samples per frame
        
        # Frame counter
        self.frame_count = 0
        self.fps_history = []
        
        # Load sensor data for the headsets
        self.csv_path = csv_path
        self.load_sensor_data()
        
        # Create scene objects
        self.gyro_only_headset = None  # Left headset - gyro only
        self.full_filter_headset = None  # Right headset - gyro + accel
        self.setup_scene()
        
        # Debug and control flags
        self.show_debug = True
        self.paused = False
        self.show_orientation_data = True
        
        # Font for info display
        self.font = pygame.font.SysFont('Arial', 16)
        
        # Set up video recording
        self.video_recorder = VideoRecorder(width, height, fps=30)
        self.is_recording = False
        self.auto_record = auto_record
        self.recording_finished = False

    def load_sensor_data(self):
        """Load and preprocess sensor data from CSV file"""
        try:
            parser = SensorDataParser(self.csv_path)
            self.sensor_data = parser.parse()
            print(f"Loaded {len(self.sensor_data)} sensor data points")
            
            # Create two separate dead reckoning filters
            
            # 1. Gyroscope-only filter (no accelerometer correction)
            self.gyro_only_filter = DeadReckoningFilter(alpha=1.0)  # alpha=1.0 means ignore accelerometer
            
            # 2. Full filter with gravity-based tilt correction
            self.full_filter = DeadReckoningFilter(alpha=0.98)  # 0.98 means 98% gyro, 2% accel
            
            # Calibrate both filters using the same initial samples
            self.gyro_only_filter.calibrate(self.sensor_data[:min(100, len(self.sensor_data))])
            self.full_filter.calibrate(self.sensor_data[:min(100, len(self.sensor_data))])
            
            self.current_data_index = 0
            
            # Store Euler angles for both filters for display
            self.gyro_only_angles = [0, 0, 0]
            self.full_filter_angles = [0, 0, 0]
            
        except Exception as e:
            print(f"Error loading sensor data: {e}")
            print("Using fallback rotation pattern instead")
            self.sensor_data = None
            self.gyro_only_filter = None
            self.full_filter = None
            
    def perspective_projection(self, x, y, z, width=None, height=None):
        """Convert 3D world coordinates to 2D screen coordinates with perspective"""
        # Use class width/height if not provided
        if width is None:
            width = self.width
        if height is None:
            height = self.height
            
        # Calculate vector from camera to point
        to_point = Vector(
            x - self.camera_pos.x,
            y - self.camera_pos.y,
            z - self.camera_pos.z
        )
        
        # Calculate camera orientation vectors
        forward = Vector(
            self.camera_target.x - self.camera_pos.x,
            self.camera_target.y - self.camera_pos.y,
            self.camera_target.z - self.camera_pos.z
        ).normalize()
        
        # Define camera's up vector (world up)
        world_up = Vector(0, 1, 0)
        
        # Calculate camera's right vector (perpendicular to forward and up)
        right = forward.cross(world_up).normalize()
        
        # Calculate true up vector (perpendicular to forward and right)
        up = right.cross(forward).normalize()
        
        # Project point onto camera orientation vectors
        right_component = to_point.x * right.x + to_point.y * right.y + to_point.z * right.z
        up_component = to_point.x * up.x + to_point.y * up.y + to_point.z * up.z
        forward_component = to_point.x * forward.x + to_point.y * forward.y + to_point.z * forward.z
        
        # Skip if point is behind camera
        if forward_component < 0.1:
            return -1, -1
        
        # Apply perspective projection
        fov = math.pi / 3.0  # 60 degrees
        aspect = width / height
        
        # Convert to NDC coordinates (-1 to 1)
        x_ndc = right_component / (forward_component * math.tan(fov/2) * aspect)
        y_ndc = up_component / (forward_component * math.tan(fov/2))
        
        # Convert to screen coordinates
        screen_x = int((x_ndc + 1.0) * width / 2.0)
        screen_y = int((-y_ndc + 1.0) * height / 2.0)  # Flip Y
        
        return screen_x, screen_y
        
    def setup_scene(self):
        """Set up the scene with two headsets - one for each filter method"""
        # Create left headset (gyro only)
        gyro_model = Model('data/headset.obj')
        gyro_model.normalizeGeometry()
        
        # Position it on the left side - further away from center
        gyro_position = Vector(-15, 5, -10)
        gyro_model.setPosition(gyro_position.x, gyro_position.y, gyro_position.z)
        
        # Set scale - smaller size
        gyro_model.scale = [1.2, 1.2, 1.2]
        gyro_model.updateTransform()
        
        # Red color for gyro-only headset
        gyro_model.diffuse_color = (255, 50, 50)  # Red color
        
        # Store the model
        self.gyro_only_headset = {
            "model": gyro_model,
            "rotation": [0, 0, 0],
            "position": gyro_position,
            "label": "Gyroscope Only"
        }
        
        # Create right headset (gyro + accelerometer)
        full_model = Model('data/headset.obj')
        full_model.normalizeGeometry()
        
        # Position it on the right side - further away from center
        full_position = Vector(15, 5, -10)
        full_model.setPosition(full_position.x, full_position.y, full_position.z)
        
        # Set scale - smaller size
        full_model.scale = [1.2, 1.2, 1.2]
        full_model.updateTransform()
        
        # Green color for full filter headset
        full_model.diffuse_color = (50, 255, 50)  # Green color
        
        # Store the model
        self.full_filter_headset = {
            "model": full_model,
            "rotation": [0, 0, 0],
            "position": full_position,
            "label": "Gyro + Gravity Correction"
        }
        
        # Load the floor object
        try:
            floor_model = Model('data/floor.obj')
            print("Floor model loaded successfully")
        except Exception as e:
            print(f"Error loading floor model: {e}")
            # Create a simple floor plane as fallback
            floor_model = Model('data/headset.obj')  # Use headset as base
            floor_model.scale = [30.0, 0.1, 30.0]   # Flatten it
            
        floor_model.normalizeGeometry()
        floor_model.setPosition(0, -5, -10)  # Adjust position as needed
        floor_model.scale = [40.0, 1.0, 40.0]  # Adjust scale as needed
        floor_model.updateTransform()
        floor_model.diffuse_color = (100, 100, 100)  # Gray color for the floor
        
        # Store the floor model
        self.floor_model = floor_model

    def update_headsets(self, dt):
        """Update both headsets with their respective filters"""
        if self.sensor_data and self.gyro_only_filter and self.full_filter:
            # Use sensor data for rotation
            if self.current_data_index < len(self.sensor_data):
                # Process multiple samples per frame for smoother animation
                for _ in range(self.imu_playback_speed):
                    if self.current_data_index >= len(self.sensor_data):
                        # Check if we've reached the end of the dataset
                        if self.auto_record and self.is_recording and not self.recording_finished:
                            self.stop_recording()
                            self.recording_finished = True
                            print("IMU dataset completed, stopping recording")
                        break
                        
                    sensor_data = self.sensor_data[self.current_data_index]
                    self.current_data_index += 1
                    
                    # ---- UPDATE GYRO-ONLY HEADSET (LEFT) ----
                    # Use the old update method that only uses gyroscope
                    _, gyro_orientation = self.gyro_only_filter.update_old(sensor_data)
                    
                    # Convert quaternion to Euler angles for model rotation
                    roll, pitch, yaw = self.gyro_only_filter.get_euler_angles()
                    self.gyro_only_angles = [roll, pitch, yaw]
                    
                    # Apply rotation to model
                    self.gyro_only_headset["model"].setRotation(roll, pitch, yaw)
                    self.gyro_only_headset["rotation"] = [roll, pitch, yaw]
                    
                    # Ensure position is updated from model's transform
                    gyro_model = self.gyro_only_headset["model"]
                    self.gyro_only_headset["position"] = Vector(
                        gyro_model.trans[0], 
                        gyro_model.trans[1], 
                        gyro_model.trans[2]
                    )
                    
                    # ---- UPDATE FULL FILTER HEADSET (RIGHT) ----
                    # Use the updated method that uses both gyroscope and accelerometer
                    _, full_orientation = self.full_filter.update(sensor_data)
                    
                    # Convert quaternion to Euler angles for model rotation
                    roll, pitch, yaw = self.full_filter.get_euler_angles()
                    self.full_filter_angles = [roll, pitch, yaw]
                    
                    # Apply rotation to model
                    self.full_filter_headset["model"].setRotation(roll, pitch, yaw)
                    self.full_filter_headset["rotation"] = [roll, pitch, yaw]
                    
                    # Ensure position is updated from model's transform
                    full_model = self.full_filter_headset["model"]
                    self.full_filter_headset["position"] = Vector(
                        full_model.trans[0], 
                        full_model.trans[1], 
                        full_model.trans[2]
                    )
            else:
                # Reset to beginning of data when we reach the end
                self.current_data_index = 0
                
                # If auto-recording is enabled, stop recording when dataset is finished
                if self.auto_record and self.is_recording and not self.recording_finished:
                    self.stop_recording()
                    self.recording_finished = True
                    print("IMU dataset completed, stopping recording")
        else:
            # Fallback: simple rotation pattern if no IMU data
            self.gyro_only_headset["rotation"][0] += dt * 1.0  # Roll
            self.gyro_only_headset["rotation"][1] += dt * 1.5  # Pitch
            self.gyro_only_headset["rotation"][2] += dt * 0.8  # Yaw
            
            # Apply rotation to model
            gyro_model = self.gyro_only_headset["model"]
            gyro_model.setRotation(
                self.gyro_only_headset["rotation"][0],
                self.gyro_only_headset["rotation"][1],
                self.gyro_only_headset["rotation"][2]
            )
            
            # Same rotations for full filter headset but with slight difference
            self.full_filter_headset["rotation"][0] += dt * 0.9  # Roll
            self.full_filter_headset["rotation"][1] += dt * 1.4  # Pitch
            self.full_filter_headset["rotation"][2] += dt * 0.7  # Yaw
            
            # Apply rotation to model
            full_model = self.full_filter_headset["model"]
            full_model.setRotation(
                self.full_filter_headset["rotation"][0],
                self.full_filter_headset["rotation"][1],
                self.full_filter_headset["rotation"][2]
            )
    
    def render_scene(self):
        """Render the scene with both headsets and the floor"""
        # Clear image and z-buffer for new frame
        self.image = Image(self.width, self.height, Color(20, 20, 40, 255))
        self.zBuffer = [-float('inf')] * self.width * self.height
        
        # First render the floor
        if self.floor_model:
            self.render_model(self.floor_model)
        
        # Render both headsets
        if self.gyro_only_headset:
            self.render_model(self.gyro_only_headset["model"])
        
        if self.full_filter_headset:
            self.render_model(self.full_filter_headset["model"])
        
        # Update display
        self.update_display(self.image)
        
        # Draw headset labels and other information
        self.draw_labels()
        
        # Draw debug info
        if self.show_debug:
            self.draw_debug_info()
        
        # Capture frame for video if recording
        if self.is_recording:
            self.video_recorder.capture_frame(self.screen)
    
    def render_model(self, model_obj):
        """Render a 3D model with lighting"""
        # Handle models with or without a 'model' attribute
        model = getattr(model_obj, 'model', model_obj)

        # Precalculate transformed vertices
        transformed_vertices = []
        for i in range(len(model.vertices)):
            transformed_vertex = model.getTransformedVertex(i)
            transformed_vertices.append(transformed_vertex)

        # Calculate face normals
        faceNormals = {}
        for face in model.faces:
            p0 = transformed_vertices[face[0]]
            p1 = transformed_vertices[face[1]]
            p2 = transformed_vertices[face[2]]
            faceNormal = (p2 - p0).cross(p1 - p0).normalize()

            for i in face:
                if i not in faceNormals:
                    faceNormals[i] = []
                faceNormals[i].append(faceNormal)

        # Calculate vertex normals
        vertexNormals = []
        for vertIndex in range(len(model.vertices)):
            if vertIndex in faceNormals:
                normal = Vector(0, 0, 0)
                for adjNormal in faceNormals[vertIndex]:
                    normal = normal + adjNormal
                vertexNormals.append(normal / len(faceNormals[vertIndex]))
            else:
                vertexNormals.append(Vector(0, 1, 0))  # Default normal

        # Get model color
        model_color = getattr(model, 'diffuse_color', (255, 255, 255))

        # Render all faces
        for face in model.faces:
            p0 = transformed_vertices[face[0]]
            p1 = transformed_vertices[face[1]]
            p2 = transformed_vertices[face[2]]
            n0, n1, n2 = [vertexNormals[i] for i in face]

            # Apply backface culling except for floor
            is_floor = (model == self.floor_model)
            if not is_floor:  # Only apply backface culling to non-floor objects
                avg_normal = (n0 + n1 + n2) / 3
                view_dir = Vector(
                    self.camera_pos.x - (p0.x + p1.x + p2.x) / 3,
                    self.camera_pos.y - (p0.y + p1.y + p2.y) / 3,
                    self.camera_pos.z - (p0.z + p1.z + p2.z) / 3
                ).normalize()
                if avg_normal * view_dir <= 0:
                    continue

            # Create points with lighting
            from shape import Triangle, Point
            triangle_points = []
            for p, n in zip([p0, p1, p2], [n0, n1, n2]):
                screenX, screenY = self.perspective_projection(p.x, p.y, p.z)

                # Skip if offscreen
                if screenX < 0 or screenY < 0 or screenX >= self.width or screenY >= self.height:
                    continue

                # Calculate lighting intensity
                intensity = max(0.2, n * self.light_dir)

                # Apply lighting to model color
                r, g, b = model_color
                color = Color(
                    int(r * intensity),
                    int(g * intensity),
                    int(b * intensity),
                    255
                )

                # Create point
                point = Point(screenX, screenY, p.z, color)
                triangle_points.append(point)

            # Draw the triangle if all points are valid
            if len(triangle_points) == 3:
                Triangle(
                    triangle_points[0],
                    triangle_points[1],
                    triangle_points[2]
                ).draw_faster(self.image, self.zBuffer)

    def update_display(self, image):
        """Update the display with current image buffer"""
        # Convert image buffer to pygame surface
        for y in range(self.height):
            for x in range(self.width):
                # Calculate buffer index
                flipY = (self.height - y - 1)
                idx = (flipY * self.width + x) * 4 + flipY + 1  # +1 for null byte
                
                # Extract RGB values
                if idx + 2 < len(image.buffer):
                    r = image.buffer[idx]
                    g = image.buffer[idx + 1]
                    b = image.buffer[idx + 2]
                    
                    # Set pixel on screen
                    self.screen.set_at((x, y), (r, g, b))

    def draw_labels(self):
        """Draw labels for the two headsets to identify them"""
        if self.gyro_only_headset:
            # Project the position to screen space
            pos = self.gyro_only_headset["position"]
            screen_x, screen_y = self.perspective_projection(pos.x, pos.y - 3, pos.z)
            
            if screen_x > 0 and screen_y > 0:
                # Draw label with red background
                label = self.font.render(self.gyro_only_headset["label"], True, (255, 255, 255))
                label_rect = label.get_rect(center=(screen_x, screen_y + 30))
                
                # Draw background
                bg_rect = label_rect.copy()
                bg_rect.inflate_ip(10, 10)
                pygame.draw.rect(self.screen, (150, 0, 0), bg_rect)
                pygame.draw.rect(self.screen, (255, 255, 255), bg_rect, 1)
                
                # Draw text
                self.screen.blit(label, label_rect)
        
        if self.full_filter_headset:
            # Project the position to screen space
            pos = self.full_filter_headset["position"]
            screen_x, screen_y = self.perspective_projection(pos.x, pos.y - 3, pos.z)
            
            if screen_x > 0 and screen_y > 0:
                # Draw label with green background
                label = self.font.render(self.full_filter_headset["label"], True, (255, 255, 255))
                label_rect = label.get_rect(center=(screen_x, screen_y + 30))
                
                # Draw background
                bg_rect = label_rect.copy()
                bg_rect.inflate_ip(10, 10)
                pygame.draw.rect(self.screen, (0, 150, 0), bg_rect)
                pygame.draw.rect(self.screen, (255, 255, 255), bg_rect, 1)
                
                # Draw text
                self.screen.blit(label, label_rect)
    
    def draw_debug_info(self):
        """Draw debug information on screen including IMU dataset progress and orientation angles"""
        # Calculate FPS
        if len(self.fps_history) > 0:
            fps = len(self.fps_history) / sum(self.fps_history)
        else:
            fps = 0
            
        # Display FPS
        fps_text = self.font.render(f"FPS: {int(fps)}", True, (255, 255, 255))
        self.screen.blit(fps_text, (10, 10))
        
        # Display IMU dataset progress
        if hasattr(self, 'sensor_data') and self.sensor_data:
            # Calculate progress as percentage
            progress_percent = (self.current_data_index / len(self.sensor_data)) * 100
            
            # Display textual progress info
            imu_text = self.font.render(
                f"IMU Data: {self.current_data_index}/{len(self.sensor_data)} ({progress_percent:.1f}%)",
                True, (255, 255, 255)
            )
            self.screen.blit(imu_text, (10, 35))
            
            # Draw progress bar
            progress_bar_width = 200
            progress_bar_height = 10
            x_pos = 10
            y_pos = 60
            
            # Draw background bar (empty portion)
            pygame.draw.rect(
                self.screen,
                (80, 80, 80),  # Dark gray
                (x_pos, y_pos, progress_bar_width, progress_bar_height)
            )
            
            # Draw filled portion based on progress
            filled_width = int(progress_bar_width * (self.current_data_index / len(self.sensor_data)))
            pygame.draw.rect(
                self.screen,
                (0, 200, 100),  # Green
                (x_pos, y_pos, filled_width, progress_bar_height)
            )
            
            # Draw border around progress bar
            pygame.draw.rect(
                self.screen,
                (200, 200, 200),  # Light gray
                (x_pos, y_pos, progress_bar_width, progress_bar_height),
                1  # Border width
            )
        
        # Display orientation data for both headsets
        if self.show_orientation_data:
            y_pos = 90
            
            # Gyroscope only data (left headset)
            if hasattr(self, 'gyro_only_angles'):
                roll, pitch, yaw = self.gyro_only_angles
                angles_text = self.font.render(
                    f"Gyro Only - Roll: {math.degrees(roll):.1f}°, Pitch: {math.degrees(pitch):.1f}°, Yaw: {math.degrees(yaw):.1f}°",
                    True, (255, 100, 100)
                )
                self.screen.blit(angles_text, (10, y_pos))
                y_pos += 25
            
            # Full filter data (right headset)
            if hasattr(self, 'full_filter_angles'):
                roll, pitch, yaw = self.full_filter_angles
                angles_text = self.font.render(
                    f"With Gravity Correction - Roll: {math.degrees(roll):.1f}°, Pitch: {math.degrees(pitch):.1f}°, Yaw: {math.degrees(yaw):.1f}°",
                    True, (100, 255, 100)
                )
                self.screen.blit(angles_text, (10, y_pos))
                y_pos += 25
                
            # Add difference between methods
            if hasattr(self, 'gyro_only_angles') and hasattr(self, 'full_filter_angles'):
                roll_diff = math.degrees(self.gyro_only_angles[0] - self.full_filter_angles[0])
                pitch_diff = math.degrees(self.gyro_only_angles[1] - self.full_filter_angles[1])
                yaw_diff = math.degrees(self.gyro_only_angles[2] - self.full_filter_angles[2])
                
                diff_text = self.font.render(
                    f"Difference - Roll: {abs(roll_diff):.1f}°, Pitch: {abs(pitch_diff):.1f}°, Yaw: {abs(yaw_diff):.1f}°",
                    True, (255, 255, 100)
                )
                self.screen.blit(diff_text, (10, y_pos))
        
        # Display recording status
        if self.is_recording:
            rec_text = self.font.render(
                f"RECORDING [{len(self.video_recorder.frames)} frames]",
                True, (255, 0, 0)
            )
            self.screen.blit(rec_text, (self.width - 300, 10))
        
        # Display controls at bottom of screen
        controls_text = self.font.render(
            "O: Toggle orientation data | R: Reset | P: Pause | V: Record | ESC: Quit",
            True, (200, 200, 200)
        )
        self.screen.blit(controls_text, (10, self.height - 30))
    
    def handle_events(self):
        """Handle user input events"""
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                # If recording is active, save the video before quitting
                if self.is_recording:
                    self.stop_recording()
                return False
            
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    # If recording is active, save the video before quitting
                    if self.is_recording:
                        self.stop_recording()
                    return False
                
                elif event.key == pygame.K_o:
                    # Toggle orientation data display
                    self.show_orientation_data = not self.show_orientation_data
                    print(f"Orientation data display: {'On' if self.show_orientation_data else 'Off'}")
                
                elif event.key == pygame.K_r:
                    # Reset filters and restart from beginning of data
                    if self.sensor_data:
                        # Re-create filters
                        self.gyro_only_filter = DeadReckoningFilter(alpha=1.0)
                        self.full_filter = DeadReckoningFilter(alpha=0.98)
                        
                        # Recalibrate
                        self.gyro_only_filter.calibrate(self.sensor_data[:min(100, len(self.sensor_data))])
                        self.full_filter.calibrate(self.sensor_data[:min(100, len(self.sensor_data))])
                        
                        # Reset index
                        self.current_data_index = 0
                        print("Filters reset and data restarted")
                
                elif event.key == pygame.K_p:
                    # Pause/resume simulation
                    self.paused = not self.paused
                    print(f"Simulation {'Paused' if self.paused else 'Resumed'}")
                
                elif event.key == pygame.K_d:
                    # Toggle debug visualization
                    self.show_debug = not self.show_debug
                    print(f"Debug Info: {'Enabled' if self.show_debug else 'Disabled'}")
                
                elif event.key == pygame.K_v:
                    # Toggle video recording
                    if not self.is_recording:
                        self.start_recording()
                    else:
                        self.stop_recording()
        
        return True
    
    def start_recording(self):
        """Start recording the simulation"""
        self.is_recording = True
        self.video_recorder.start_recording()
        print("Started recording video")
    
    def stop_recording(self):
        """Stop recording and save the video file"""
        if not self.is_recording:
            return
            
        self.is_recording = False
        self.video_recorder.stop_recording()
        
        # Save the video
        try:
            # Try to save with OpenCV first
            file_path = self.video_recorder.save_video(filename="orientation_filter_comparison.mp4")
            if file_path:
                print(f"Video saved to {file_path}")
            else:
                print("Failed to save video with OpenCV, saving frames as images...")
                # Fall back to saving frames as images
                frames_dir = self.video_recorder.save_frames_as_images()
                print(f"Frames saved to {frames_dir}")
        except Exception as e:
            print(f"Error saving video: {e}")
    
    def run(self):
        """Main loop to run the simulation"""
        clock = pygame.time.Clock()
        running = True
        
        print("Orientation Filter Comparison - Gyroscope Only vs. Gravity Correction")
        print("--------------------------------------------------------")
        print("Controls:")
        print("  O: Toggle orientation data display")
        print("  R: Reset filters and restart from beginning")
        print("  P: Pause/resume simulation")
        print("  D: Toggle debug info")
        print("  V: Start/stop video recording")
        print("  ESC: Quit")
        
        # Auto-start recording if enabled
        if self.auto_record:
            self.start_recording()
        
        while running:
            dt = min(clock.tick(60) / 1000.0, 0.1)  # Cap at 0.1s to prevent physics jumps
            
            # Track frame time for FPS calculation
            self.fps_history.append(dt)
            if len(self.fps_history) > 20:
                self.fps_history.pop(0)
            
            # Handle events
            running = self.handle_events()
            
            # Exit if auto-recording was enabled and recording has finished
            if self.auto_record and self.recording_finished:
                # Give a short delay to ensure the recording is properly saved
                time.sleep(0.5)
                print("Auto-recording complete. Exiting simulation.")
                running = False
                break
            
            # Skip updates if paused
            if not self.paused:
                # Update headsets with their respective filters
                self.update_headsets(dt)
            
            # Render the scene
            self.render_scene()
            
            # Update display
            pygame.display.flip()
            
            # Increment frame counter
            self.frame_count += 1
        
        # Clean up
        pygame.quit()
        print(f"Simulation ended after {self.frame_count} frames")

def run_orientation_filter_comparison():
    """
    Start a comparison scene showing the difference between orientation filtering methods.
    """
    try:
        # Try to use a local path first
        scene = ComparativeHeadsetScene(csv_path="../IMUData.csv")
    except Exception as e:
        # If that fails, try using a relative path with parent directory
        try:
            scene = ComparativeHeadsetScene(csv_path="../IMUData.csv")
        except Exception as e2:
            print(f"Error loading IMU data: {e2}")
            print("Continuing with fallback rotation pattern")
            scene = ComparativeHeadsetScene(csv_path="")
    
    scene.run()

# Run this function to start the simulation
if __name__ == "__main__":
    run_orientation_filter_comparison()