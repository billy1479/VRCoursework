import os
import time
import math
import pygame
import random
from vector import Vector
from model import Model, SensorDataParser, SensorData
from quaternion import Quaternion
from dead_reckoning_filter import DeadReckoningFilter
from color_support import ColoredModel
from image import Image, Color
from video_recorder import VideoRecorder

class GyroscopeOnlyFilter(DeadReckoningFilter):
    """Modified filter that uses ONLY gyroscope data, ignoring accelerometer and magnetometer"""
    
    def update(self, sensor_data):
        """Override to use only gyroscope integration"""
        if self.last_time is None:
            self.last_time = sensor_data.time
            return self.orientation

        dt = sensor_data.time - self.last_time
        self.last_time = sensor_data.time
        
        if dt <= 0 or dt > 0.1:
            dt = 0.01  # Use reasonable default
            
        # Extract gyroscope data and apply bias correction
        gyro_x = sensor_data.gyroscope[0] - self.gyro_bias[0]
        gyro_y = sensor_data.gyroscope[1] - self.gyro_bias[1]
        gyro_z = sensor_data.gyroscope[2] - self.gyro_bias[2]
        
        self.rotation_rate = Vector(gyro_x, gyro_y, gyro_z)
        
        # Gyroscope integration only
        rotation_quat = Quaternion(0, gyro_x, gyro_y, gyro_z)
        
        q_dot = self._quaternion_multiply(self.orientation, rotation_quat)
        q_dot.w *= 0.5
        q_dot.x *= 0.5
        q_dot.y *= 0.5
        q_dot.z *= 0.5
        
        # Update orientation using only gyroscope data
        self.orientation = Quaternion(
            self.orientation.w + q_dot.w * dt,
            self.orientation.x + q_dot.x * dt,
            self.orientation.y + q_dot.y * dt,
            self.orientation.z + q_dot.z * dt
        )
        self.orientation.normalize()
        
        return self.orientation

class ComparativeHeadsetRenderer:
    def __init__(self, width=1000, height=600):
        # Initialize pygame
        pygame.init()
        self.width = width
        self.height = height
        self.screen = pygame.display.set_mode((width, height))
        pygame.display.set_caption("Headset Orientation Comparison: Gyro-Only vs Full Filter")
        
        # Setup camera and lighting
        self.camera_pos = Vector(0, 5, -10)
        self.camera_target = Vector(0, 0, 0)
        self.light_dir = Vector(0.5, -1, -0.5).normalize()
        
        # Split screen in half for comparison
        self.left_half = pygame.Surface((width//2, height))
        self.right_half = pygame.Surface((width//2, height))
        
        # Initialize video recorder
        self.video_recorder = VideoRecorder(width, height, fps=30, output_dir="output")
        self.is_recording = False
        
        # Load IMU data
        self.sensor_data = self.load_imu_data()
        
        # Create two filters
        self.gyro_only_filter = GyroscopeOnlyFilter(alpha=0.98)
        self.full_filter = DeadReckoningFilter(alpha=0.95, beta=0.05, mag_weight=0.02)
        
        # Create the headset models
        self.gyro_headset = self.create_headset_model((255, 0, 0))  # Red
        self.full_headset = self.create_headset_model((0, 255, 0))  # Green
        
        # Tracking variables
        self.current_data_index = 0
        self.paused = False
        self.font = pygame.font.SysFont('Arial', 18)

    def load_imu_data(self):
        """Load IMU data from CSV file"""
        csv_path = "./IMUdata.csv"
        
        # Fall back to a relative path if the direct path doesn't work
        if not os.path.exists(csv_path):
            csv_path = "../IMUdata.csv"
            
        if not os.path.exists(csv_path):
            print(f"Warning: IMU data file not found at {csv_path}")
            print("Using synthetic IMU data instead")
            return self.generate_synthetic_imu_data()
            
        try:
            parser = SensorDataParser(csv_path)
            data = parser.parse()
            print(f"Loaded {len(data)} IMU data points")
            return data
        except Exception as e:
            print(f"Error loading IMU data: {e}")
            print("Using synthetic IMU data instead")
            return self.generate_synthetic_imu_data()
    
    def generate_synthetic_imu_data(self, num_samples=500):
        """Generate synthetic IMU data for testing"""
        data = []
        for i in range(num_samples):
            time = i / 30.0  # 30 Hz data
            
            # Create oscillating rotation pattern
            gyro_x = math.sin(time * 0.5) * 0.3
            gyro_y = math.cos(time * 0.8) * 0.2
            gyro_z = math.sin(time * 0.3) * 0.4
            
            # Add some noise for realism
            gyro_x += random.uniform(-0.05, 0.05)
            gyro_y += random.uniform(-0.05, 0.05)
            gyro_z += random.uniform(-0.05, 0.05)
            
            # Generate accelerometer data (gravity + small noise)
            accel_x = random.uniform(-0.2, 0.2)
            accel_y = random.uniform(-0.2, 0.2)
            accel_z = 9.81 + random.uniform(-0.1, 0.1)
            
            # Generate magnetometer data (earth's field + small noise)
            mag_x = 1.0 + random.uniform(-0.1, 0.1)
            mag_y = 0.0 + random.uniform(-0.1, 0.1)
            mag_z = 0.0 + random.uniform(-0.1, 0.1)
            
            sample = SensorData(
                time=time,
                gyroscope=(gyro_x, gyro_y, gyro_z),
                accelerometer=(accel_x, accel_y, accel_z),
                magnetometer=(mag_x, mag_y, mag_z)
            )
            data.append(sample)
            
        return data

    def create_headset_model(self, color):
        """Create a headset model with the specified color"""
        try:
            model = Model('./data/headset.obj')
            model.normalizeGeometry()
            colored_model = ColoredModel(model, diffuse_color=color)
            return colored_model
        except Exception as e:
            print(f"Error loading headset model: {e}")
            # Create a simple placeholder model
            model = Model('./headset.obj')
            model.normalizeGeometry()
            colored_model = ColoredModel(model, diffuse_color=color)
            return colored_model

    def perspective_projection(self, x, y, z, width=None, height=None):
        """Project 3D coordinates to 2D screen space"""
        if width is None:
            width = self.width // 2
        if height is None:
            height = self.height
        
        # Calculate vector from camera to point
        to_point = Vector(
            x - self.camera_pos.x,
            y - self.camera_pos.y,
            z - self.camera_pos.z
        )
        
        # Camera orientation vectors
        forward = Vector(
            self.camera_target.x - self.camera_pos.x,
            self.camera_target.y - self.camera_pos.y,
            self.camera_target.z - self.camera_pos.z
        ).normalize()
        
        world_up = Vector(0, 1, 0)
        right = forward.cross(world_up).normalize()
        up = right.cross(forward).normalize()
        
        # Project point onto camera vectors
        right_comp = to_point * right
        up_comp = to_point * up
        forward_comp = to_point * forward
        
        if forward_comp < 0.1:
            return -1, -1
        
        # Apply perspective projection
        fov = math.pi / 3.0
        aspect = width / height
        
        x_ndc = right_comp / (forward_comp * math.tan(fov/2) * aspect)
        y_ndc = up_comp / (forward_comp * math.tan(fov/2))
        
        # Convert to screen coordinates
        screen_x = int((x_ndc + 1.0) * width / 2.0)
        screen_y = int((-y_ndc + 1.0) * height / 2.0)
        
        return screen_x, screen_y

    def render_model(self, surface, model, quaternion=None, offset_x=0):
        """Render a model with the given quaternion rotation"""
        # Apply rotation if provided
        if quaternion:
            model.model.setQuaternionRotation(quaternion)
        
        # Get transformed vertices
        transformed_vertices = []
        for i in range(len(model.model.vertices)):
            vertex = model.model.getTransformedVertex(i)
            transformed_vertices.append(vertex)
        
        # Calculate face normals
        face_normals = {}
        for face_idx, face in enumerate(model.model.faces):
            v0 = transformed_vertices[face[0]]
            v1 = transformed_vertices[face[1]]
            v2 = transformed_vertices[face[2]]
            
            edge1 = v1 - v0
            edge2 = v2 - v0
            normal = edge1.cross(edge2).normalize()
            
            for i in face:
                if i not in face_normals:
                    face_normals[i] = []
                face_normals[i].append(normal)
        
        # Calculate vertex normals
        vertex_normals = []
        for vert_idx in range(len(model.model.vertices)):
            if vert_idx in face_normals:
                normal = Vector(0, 0, 0)
                for face_normal in face_normals[vert_idx]:
                    normal = normal + face_normal
                vertex_normals.append(normal.normalize())
            else:
                vertex_normals.append(Vector(0, 1, 0))
        
        # Get diffuse color
        if hasattr(model, 'diffuse_color'):
            diffuse_color = model.diffuse_color
        else:
            diffuse_color = (200, 200, 200)
        
        # Draw each face
        for face in model.model.faces:
            v0 = transformed_vertices[face[0]]
            v1 = transformed_vertices[face[1]]
            v2 = transformed_vertices[face[2]]
            
            n0 = vertex_normals[face[0]]
            n1 = vertex_normals[face[1]]
            n2 = vertex_normals[face[2]]
            
            # Backface culling
            avg_normal = (n0 + n1 + n2).normalize()
            view_dir = Vector(
                self.camera_pos.x - (v0.x + v1.x + v2.x) / 3,
                self.camera_pos.y - (v0.y + v1.y + v2.y) / 3,
                self.camera_pos.z - (v0.z + v1.z + v2.z) / 3
            ).normalize()
            
            if avg_normal * view_dir <= 0:
                continue
            
            # Project vertices to screen space
            screen_points = []
            for v, n in zip([v0, v1, v2], [n0, n1, n2]):
                screen_x, screen_y = self.perspective_projection(v.x, v.y, v.z)
                
                if screen_x < 0 or screen_y < 0:
                    continue
                
                # Apply lighting
                intensity = max(0.2, n * self.light_dir)
                r, g, b = diffuse_color
                color = (
                    int(r * intensity),
                    int(g * intensity),
                    int(b * intensity)
                )
                
                screen_points.append((screen_x + offset_x, screen_y, color))
            
            # Draw the triangle if all points are visible
            if len(screen_points) == 3:
                pygame.draw.polygon(surface, screen_points[0][2], 
                                   [(p[0], p[1]) for p in screen_points])

    def update_filters(self):
        """Update both filters with the current IMU data"""
        if self.current_data_index < len(self.sensor_data):
            sensor_data = self.sensor_data[self.current_data_index]
            
            # Update both filters
            gyro_orientation = self.gyro_only_filter.update(sensor_data)
            full_orientation = self.full_filter.update(sensor_data)
            
            self.current_data_index += 1
            return gyro_orientation, full_orientation
        
        return None, None

    def calibrate_filters(self):
        """Calibrate both filters using the first few samples"""
        if len(self.sensor_data) > 0:
            calibration_samples = min(100, len(self.sensor_data))
            self.gyro_only_filter.calibrate(self.sensor_data[:calibration_samples])
            self.full_filter.calibrate(self.sensor_data[:calibration_samples])
            print("Filters calibrated")

    def render_text(self, x, y, text, color=(255, 255, 255)):
        """Render text at the specified position"""
        text_surface = self.font.render(text, True, color)
        self.screen.blit(text_surface, (x, y))

    def start_recording(self):
        """Start video recording"""
        self.is_recording = True
        self.video_recorder.start_recording()
        print("Started recording video")

    def stop_recording(self):
        """Stop and save video recording"""
        if not self.is_recording:
            return
        
        self.is_recording = False
        self.video_recorder.stop_recording()
        
        # Save regular video
        video_path = self.video_recorder.save_video("headset_comparison.mp4")
        print(f"Video saved to: {video_path}")
        
        # Try to generate higher quality video with ffmpeg
        ffmpeg_path = self.video_recorder.generate_ffmpeg_video(quality="high")
        if ffmpeg_path:
            print(f"High quality video saved to: {ffmpeg_path}")

    def run(self):
        """Main simulation loop"""
        clock = pygame.time.Clock()
        running = True
        
        # Calibrate filters before starting
        self.calibrate_filters()
        
        # Start recording immediately
        self.start_recording()
        
        print("Headset Orientation Comparison")
        print("Controls: P (pause/resume), R (reset), ESC (quit)")
        
        while running:
            # Handle events
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        running = False
                    elif event.key == pygame.K_p:
                        self.paused = not self.paused
                    elif event.key == pygame.K_r:
                        # Reset simulation
                        self.current_data_index = 0
                        self.gyro_only_filter = GyroscopeOnlyFilter(alpha=0.98)
                        self.full_filter = DeadReckoningFilter(alpha=0.95, beta=0.05, mag_weight=0.02)
                        self.calibrate_filters()
            
            # Clear surfaces
            self.screen.fill((20, 20, 40))
            self.left_half.fill((20, 20, 40))
            self.right_half.fill((20, 20, 40))
            
            # Update filters if not paused
            if not self.paused:
                gyro_orientation, full_orientation = self.update_filters()
                if gyro_orientation and full_orientation:
                    # Render models
                    self.render_model(self.left_half, self.gyro_headset, gyro_orientation, 0)
                    self.render_model(self.right_half, self.full_headset, full_orientation, 0)
            else:
                # Continue showing last orientation during pause
                self.render_model(self.left_half, self.gyro_headset, None, 0)
                self.render_model(self.right_half, self.full_headset, None, 0)
            
            # Blit surfaces to screen
            self.screen.blit(self.left_half, (0, 0))
            self.screen.blit(self.right_half, (self.width // 2, 0))
            
            # Draw dividing line
            pygame.draw.line(self.screen, (255, 255, 255), 
                            (self.width // 2, 0), (self.width // 2, self.height), 2)
            
            # Display labels
            self.render_text(10, 10, "Gyroscope Only", (255, 100, 100))
            self.render_text(self.width // 2 + 10, 10, "Full Filter (Gyro + Accel + Mag)", (100, 255, 100))
            
            # Show progress
            progress = min(100, (self.current_data_index / len(self.sensor_data)) * 100)
            self.render_text(10, self.height - 30, 
                          f"Progress: {self.current_data_index}/{len(self.sensor_data)} ({progress:.1f}%)")
            
            # Show controls
            self.render_text(self.width - 350, self.height - 30, 
                          "P: Pause, R: Reset, ESC: Quit")
            
            # Show recording status
            if self.is_recording:
                self.render_text(self.width - 150, 10, "RECORDING", (255, 0, 0))
            
            # Update display
            pygame.display.flip()
            
            # Capture frame for video if recording
            if self.is_recording:
                self.video_recorder.capture_frame(self.screen)
            
            # Cap framerate
            clock.tick(30)
            
            # End simulation when all data is processed
            if self.current_data_index >= len(self.sensor_data) and not self.paused:
                time.sleep(2)  # Allow viewing final state
                self.stop_recording()
                time.sleep(1)  # Short delay after recording ends
                running = False
        
        # Make sure recording is stopped when closing
        if self.is_recording:
            self.stop_recording()
        
        # Clean up
        pygame.quit()
        print("Simulation ended")

if __name__ == "__main__":
    renderer = ComparativeHeadsetRenderer()
    renderer.run()