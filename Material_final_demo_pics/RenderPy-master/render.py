import random
import pygame
import os
import time
import math
import numpy as np
from image import Image, Color
from vector import Vector
from model import Model, DeadReckoningFilter, SensorDataParser, CollisionObject, Quaternion
from shape import Triangle, Point
from color_support import ColoredModel
from motion_blur import MotionBlurEffect
from video_recorder import VideoRecorder

class HeadsetSimulation:
    def __init__(self, width=800, height=600):
        # Initialize pygame
        pygame.init()
        self.width = width
        self.height = height
        self.screen = pygame.display.set_mode((width, height))
        pygame.display.set_caption("VR Headset Simulation")
        
        # Image and Z-buffer
        self.image = Image(width, height, Color(20, 20, 40, 255))
        self.zBuffer = [-float('inf')] * width * height
        
        # Camera and lighting
        self.camera_pos = Vector(0, 20, -50)
        self.camera_target = Vector(0, 5, -15)
        self.light_dir = Vector(0.5, -1, -0.5).normalize()
        
        # Motion blur
        self.motion_blur = MotionBlurEffect(blur_strength=0.7)
        self.blur_enabled = True
        
        # Video recorder
        self.video_recorder = VideoRecorder(width, height, fps=30)
        self.is_recording = False
        
        # Load IMU data
        self.load_imu_data()
        
        # Setup scene objects
        self.main_headset = None
        self.floor_headsets = []
        self.setup_scene()
        
        # Physics settings - Reduced friction significantly for longer movement
        self.friction_coefficient = 0.94  # Changed from 0.98 to 0.995 (much less friction)
        self.accumulator = 0
        
        # Target frames for 27 seconds at 30fps
        self.target_frames = 1000
        
        # Display
        self.font = pygame.font.SysFont('Arial', 18)
        self.frame_count = 0
        self.fps_history = []
        self.paused = False
        self.imu_progress = 0

    def create_synthetic_imu_data(self):
        """Create synthetic IMU data if no real data available"""
        print("Creating synthetic IMU data")
        from model import SensorData
        
        self.sensor_data = []
        for i in range(300):
            time_val = i / 30.0
            gyro_x = math.sin(time_val * 0.8) * 0.3
            gyro_y = math.cos(time_val * 0.5) * 0.2
            gyro_z = math.sin(time_val * 0.3) * 0.15
            
            self.sensor_data.append(
                SensorData(
                    time=time_val,
                    gyroscope=(gyro_x, gyro_y, gyro_z),
                    accelerometer=(0.1, 9.8, 0.2),
                    magnetometer=(0.5, 0.2, 0.8)
                )
            )
        
        self.dr_filter = DeadReckoningFilter(alpha=0.98)
        self.current_data_index = 0

    def setup_scene(self):
        """Set up the scene with main headset, floor headsets, and floor object"""
        # Main floating headset
        model = Model('./data/headset.obj')
        model.normalizeGeometry()
        
        # Position further away from the camera
        position = Vector(0, 15, -20)
        model.setPosition(position.x, position.y, position.z)
        
        # Make the headset smaller
        model.scale = [0.5, 0.5, 0.5]
        model.updateTransform()
        
        self.main_headset = {
            "model": ColoredModel(model, diffuse_color=(255, 215, 0)),
            "rotation": [0, 0, 0],
            "position": position
        }
        
        # Load the floor object
        try:
            floor_model = Model('./data/floor.obj')
            floor_model.normalizeGeometry()
            
            # Position the floor under the headsets
            floor_model.setPosition(0, 0, -20)
            
            # Scale the floor to cover the entire movement area
            floor_model.scale = [60.0, 1.0, 40.0]
            floor_model.updateTransform()
            
            # Create colored version of the floor
            self.floor_object = ColoredModel(floor_model, diffuse_color=(150, 150, 150))
            
            print("Floor model loaded successfully")
        except Exception as e:
            print(f"Error loading floor model: {e}")
            self.floor_object = None
        
        # Create floor headsets
        self.floor_headsets = self.create_floor_headsets()

    def create_floor_headsets(self):
        """Create sliding headsets that will collide on the floor with higher initial velocities"""
        headsets = []
        
        # Colors for different headsets
        colors = [
            (255, 0, 0),    # Red
            (0, 255, 0),    # Green
            (0, 0, 255),    # Blue
            (255, 255, 0),  # Yellow
            (255, 0, 255),  # Magenta
            (0, 255, 255),  # Cyan
            (255, 165, 0),  # Orange
            (128, 0, 128)   # Purple
        ]
        
        # Circle of headsets moving inward - Higher velocities
        num_circle = 8
        circle_radius = 20
        for i in range(num_circle):
            angle = (i / num_circle) * 2 * math.pi
            pos = Vector(
                circle_radius * math.cos(angle),
                1,
                circle_radius * math.sin(angle) - 10
            )
            
            # Set higher velocity for longer movement
            speed = 3 + (i % 3)  # Increased speed
            vel = Vector(
                -math.cos(angle) * speed,
                0,
                -math.sin(angle) * speed
            )
            
            model = Model('./data/headset.obj')
            model.normalizeGeometry()
            model.setPosition(pos.x, pos.y, pos.z)
            
            # Create a colored model for this headset
            colored_model = ColoredModel(model, diffuse_color=colors[i % len(colors)])
            
            # Create collision object
            headsets.append(CollisionObject(colored_model, pos, vel, radius=1.0))
        
        # Triangle formation of headsets with small initial velocities
        triangle_size = 3
        start_z = -5
        color_index = 0
        for row in range(triangle_size):
            for col in range(row + 1):
                pos = Vector(
                    (col - row/2) * 2.5,
                    1,
                    start_z + row * 2.5
                )
                
                # Add small random velocities so they're not completely stationary
                vel = Vector(
                    (random.random() - 0.5) * 0.5,  # Small random x velocity
                    0,
                    (random.random() - 0.5) * 0.5   # Small random z velocity
                )
                
                model = Model('./data/headset.obj')
                model.normalizeGeometry()
                model.setPosition(pos.x, pos.y, pos.z)
                
                colored_model = ColoredModel(model, diffuse_color=colors[color_index % len(colors)])
                color_index += 1
                
                headsets.append(CollisionObject(colored_model, pos, vel, radius=1.0))
        
        # Add "cue ball" white headsets from different angles
        # Main "cue ball" from behind
        pos = Vector(0, 1, -25)
        vel = Vector(0, 0, 5.5)  # Increased speed
        
        model = Model('./data/headset.obj')
        model.normalizeGeometry()
        model.setPosition(pos.x, pos.y, pos.z)
        
        colored_model = ColoredModel(model, diffuse_color=(255, 255, 255))
        headsets.append(CollisionObject(colored_model, pos, vel, radius=1.0))
        
        # Additional "cue balls" from sides to create more interesting collisions
        # From left
        pos = Vector(-18, 1, -15)
        vel = Vector(3, 0, 0)  # Moving right
        
        model = Model('./data/headset.obj')
        model.normalizeGeometry()
        model.setPosition(pos.x, pos.y, pos.z)
        
        colored_model = ColoredModel(model, diffuse_color=(220, 220, 255))  # Slightly blue-tinted white
        headsets.append(CollisionObject(colored_model, pos, vel, radius=1.0))
        
        # From right (will enter scene later)
        pos = Vector(18, 1, -10)
        vel = Vector(-2.5, 0, -1)  # Moving left and slightly back
        
        model = Model('./data/headset.obj')
        model.normalizeGeometry()
        model.setPosition(pos.x, pos.y, pos.z)
        
        colored_model = ColoredModel(model, diffuse_color=(255, 220, 220))  # Slightly red-tinted white
        headsets.append(CollisionObject(colored_model, pos, vel, radius=1.0))
        
        return headsets

    def perspective_projection(self, x, y, z, width=None, height=None):
        """Project 3D coordinates to 2D screen space"""
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

    def update_floor_physics(self, dt):
        """Update physics for floor headsets with more elastic collisions"""
        # Fixed timestep for consistent physics
        fixed_dt = 1/60.0
        self.accumulator += dt
        
        # Define boundary walls
        boundary = {
            'min_x': -30.0,
            'max_x': 30.0,
            'min_z': -40.0,
            'max_z': 0.0,
            'bounce_factor': 0.9  # Increased from 0.8 to 0.9 (more elastic bounces)
        }
        
        while self.accumulator >= fixed_dt:
            # Clear collision records
            for headset in self.floor_headsets:
                headset.clear_collision_history()
            
            # Check collisions between headsets
            for i in range(len(self.floor_headsets)):
                for j in range(i + 1, len(self.floor_headsets)):
                    if self.floor_headsets[i].check_collision(self.floor_headsets[j]):
                        self.floor_headsets[i].resolve_collision(self.floor_headsets[j])
            
            # Apply floor constraints and friction
            for headset in self.floor_headsets:
                # Update position based on velocity
                headset.update(fixed_dt)
                
                # Apply friction when on floor, but only if moving fast enough
                if headset.position.y - headset.radius <= 0.01:
                    headset.position.y = headset.radius
        
                    # Apply friction to horizontal velocity components
                    horizontal_speed_squared = (
                        headset.velocity.x**2 + 
                        headset.velocity.z**2
                    )
                    
                    # Only apply friction if moving horizontally
                    if horizontal_speed_squared > 0.001:
                        # Apply friction by reducing horizontal velocity
                        headset.velocity.x *= self.friction_coefficient
                        headset.velocity.z *= self.friction_coefficient
                        
                        # Recalculate horizontal speed after applying friction
                        new_horizontal_speed_squared = (
                            headset.velocity.x**2 + 
                            headset.velocity.z**2
                        )
                        
                        # Stop completely if very slow after friction is applied
                        if new_horizontal_speed_squared < 0.025:
                            headset.velocity.x = 0
                            headset.velocity.z = 0
                
                # Apply boundary constraints
                if headset.position.x - headset.radius < boundary['min_x']:
                    headset.position.x = boundary['min_x'] + headset.radius
                    headset.velocity.x = -headset.velocity.x * boundary['bounce_factor']
                    
                    # Add small random variation to z velocity on x boundary collision
                    headset.velocity.z += (random.random() - 0.5) * 0.5
                    
                elif headset.position.x + headset.radius > boundary['max_x']:
                    headset.position.x = boundary['max_x'] - headset.radius
                    headset.velocity.x = -headset.velocity.x * boundary['bounce_factor']
                    
                    # Add small random variation to z velocity on x boundary collision
                    headset.velocity.z += (random.random() - 0.5) * 0.5
                
                if headset.position.z - headset.radius < boundary['min_z']:
                    headset.position.z = boundary['min_z'] + headset.radius
                    headset.velocity.z = -headset.velocity.z * boundary['bounce_factor']
                    
                    # Add small random variation to x velocity on z boundary collision
                    headset.velocity.x += (random.random() - 0.5) * 0.5
                    
                elif headset.position.z + headset.radius > boundary['max_z']:
                    headset.position.z = boundary['max_z'] - headset.radius
                    headset.velocity.z = -headset.velocity.z * boundary['bounce_factor']
                    
                    # Add small random variation to x velocity on z boundary collision
                    headset.velocity.x += (random.random() - 0.5) * 0.5
                
                # Update model position
                headset.model.model.setPosition(headset.position.x, headset.position.y, headset.position.z)
            
            # Every 60 frames (about 1 second), add small random impulses to keep things moving
            if self.frame_count % 60 == 0 and self.frame_count < self.target_frames * 0.7:
                for headset in self.floor_headsets:
                    # Only add impulses to slow headsets
                    speed_sq = headset.velocity.x**2 + headset.velocity.z**2
                    if speed_sq < 2.0:
                        # Add small random impulse
                        headset.velocity.x += (random.random() - 0.5) * 0.8
                        headset.velocity.z += (random.random() - 0.5) * 0.8
            
            self.accumulator -= fixed_dt

    def render_model(self, model_obj):
        """Render a 3D model with lighting"""
        # Get the actual model (handle both direct models and ColoredModel objects)
        model = getattr(model_obj, 'model', model_obj)
        
        # Precalculate transformed vertices
        transformed_vertices = []
        for i in range(len(model.vertices)):
            vertex = model.getTransformedVertex(i)
            transformed_vertices.append(vertex)
        
        # Calculate face normals and vertex normals
        face_normals = {}
        for face in model.faces:
            v0 = transformed_vertices[face[0]]
            v1 = transformed_vertices[face[1]]
            v2 = transformed_vertices[face[2]]
            
            edge1 = Vector(v1.x - v0.x, v1.y - v0.y, v1.z - v0.z)
            edge2 = Vector(v2.x - v0.x, v2.y - v0.y, v2.z - v0.z)
            normal = edge1.cross(edge2).normalize()
            
            for i in face:
                if i not in face_normals:
                    face_normals[i] = []
                face_normals[i].append(normal)
        
        vertex_normals = []
        for vert_idx in range(len(model.vertices)):
            if vert_idx in face_normals:
                normal = Vector(0, 0, 0)
                for face_normal in face_normals[vert_idx]:
                    normal = normal + face_normal
                vertex_normals.append(normal.normalize())
            else:
                vertex_normals.append(Vector(0, 1, 0))
        
        # Get model color
        if hasattr(model_obj, 'diffuse_color'):
            model_color = model_obj.diffuse_color
        elif hasattr(model, 'diffuse_color'):
            model_color = model.diffuse_color
        else:
            model_color = (200, 200, 200)
        
        # Render faces
        for face in model.faces:
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
            
            # Create triangle points
            triangle_points = []
            for v, n in zip([v0, v1, v2], [n0, n1, n2]):
                screen_x, screen_y = self.perspective_projection(v.x, v.y, v.z)
                
                if screen_x < 0 or screen_y < 0 or screen_x >= self.width or screen_y >= self.height:
                    continue
                
                # Calculate lighting
                intensity = max(0.2, n * self.light_dir)
                
                r, g, b = model_color
                color = Color(
                    int(r * intensity),
                    int(g * intensity),
                    int(b * intensity),
                    255
                )
                
                point = Point(int(screen_x), int(screen_y), v.z, color)
                point.normal = n
                triangle_points.append(point)
            
            # Render triangle if all points are valid
            if len(triangle_points) == 3:
                # Create a fixed version of Triangle.draw_faster that properly handles integers
                self.draw_triangle(triangle_points[0], triangle_points[1], triangle_points[2])

    def draw_triangle(self, p0, p1, p2):
        """Draw a triangle with properly converted integer coordinates"""
        # Create a Triangle instance
        tri = Triangle(p0, p1, p2)
        
        # Calculate bounding box with integer conversion
        ymin = max(min(p0.y, p1.y, p2.y), 0)
        ymax = min(max(p0.y, p1.y, p2.y), self.image.height - 1)
        
        # Convert to integers for range()
        ymin_int = int(ymin)
        ymax_int = int(ymax)
        
        # Iterate over scan lines
        for y in range(ymin_int, ymax_int + 1):
            x_values = []
            
            # Find intersections with edges
            for edge_start, edge_end in [(p0, p1), (p1, p2), (p2, p0)]:
                if (edge_start.y <= y <= edge_end.y) or (edge_end.y <= y <= edge_start.y):
                    if edge_end.y == edge_start.y:  # Skip horizontal edges
                        continue
                    
                    if edge_start.y == y:
                        x_values.append(edge_start.x)
                    elif edge_end.y == y:
                        x_values.append(edge_end.x)
                    else:
                        t = (y - edge_start.y) / (edge_end.y - edge_start.y)
                        x = edge_start.x + t * (edge_end.x - edge_start.x)
                        x_values.append(x)
            
            if len(x_values) > 0:
                # Sort and convert to integers
                if len(x_values) == 1:
                    x_start = x_values[0]
                    x_end = x_start
                else:
                    x_start, x_end = sorted(x_values)[:2]
                
                x_start_int = max(int(x_start), 0)
                x_end_int = min(int(x_end), self.image.width - 1)
                
                # Draw horizontal span
                for x in range(x_start_int, x_end_int + 1):
                    point = Point(x, y, color=None)
                    in_triangle, color, z_value = tri.contains_point(point)
                    
                    if in_triangle:
                        # Perform z-buffer check
                        buffer_index = y * self.image.width + x
                        if buffer_index < len(self.zBuffer) and self.zBuffer[buffer_index] < z_value:
                            self.zBuffer[buffer_index] = z_value
                            self.image.setPixel(x, y, color)

    def render_floor(self):
        """Render a simple floor grid"""
        size = 30
        height = 0
        
        # Floor corners
        corners = [
            Vector(-size, height, -size - 10),
            Vector(size, height, -size - 10),
            Vector(size, height, size - 10),
            Vector(-size, height, size - 10)
        ]
        
        # Create floor points
        p0 = Point(corners[0].x, corners[0].y, corners[0].z)
        p1 = Point(corners[1].x, corners[1].y, corners[1].z)
        p2 = Point(corners[2].x, corners[2].y, corners[2].z)
        p3 = Point(corners[3].x, corners[3].y, corners[3].z)
        
        # Set normal and color
        normal = Vector(0, 1, 0)
        for p in [p0, p1, p2, p3]:
            p.normal = normal
            p.color = Color(100, 100, 100, 255)
        
        # Render floor triangles
        self.draw_triangle(p0, p1, p2)
        self.draw_triangle(p0, p2, p3)

    def render_scene(self):
        """Render all scene elements"""
        # Clear image and z-buffer
        self.image = Image(self.width, self.height, Color(20, 20, 40, 255))
        self.zBuffer = [-float('inf')] * self.width * self.height
        
        # Render floor object first (so it's behind everything else)
        if hasattr(self, 'floor_object') and self.floor_object:
            self.render_model(self.floor_object)
        else:
            # Render simple floor if floor model failed to load
            self.render_floor()
        
        # Render main headset
        self.render_model(self.main_headset["model"])
        
        # Render floor headsets
        for headset in self.floor_headsets:
            self.render_model(headset.model)
        
        # Apply motion blur if enabled
        if self.blur_enabled:
            # Use per_object_velocity_blur from your MotionBlurEffect class
            final_image = self.motion_blur.per_object_velocity_blur(
                self.image,
                self.floor_headsets,
                self.width,
                self.height,
                self.perspective_projection
            )
        else:
            final_image = self.image
        
        # Convert image to pygame surface
        for y in range(self.height):
            for x in range(self.width):
                idx = (final_image.height - y - 1) * final_image.width * 4 + x * 4 + (final_image.height - y - 1) + 1
                if idx + 2 < len(final_image.buffer):
                    r = final_image.buffer[idx]
                    g = final_image.buffer[idx + 1]
                    b = final_image.buffer[idx + 2]
                    self.screen.set_at((x, y), (r, g, b))
        
        # Draw boundary and debug info
        self.render_boundaries()
        self.draw_debug_info()
        
        # Update display
        pygame.display.flip()
        
        # Capture frame for video if recording
        if self.is_recording:
            self.video_recorder.capture_frame(self.screen)

    def render_boundaries(self):
        """Render boundary lines for the floor area"""
        boundary = {
            'min_x': -30.0,
            'max_x': 30.0,
            'min_z': -40.0,
            'max_z': 0.0,
            'height': 5.0
        }
        
        # Draw boundary edges
        wall_color = (100, 100, 220)
        
        # Floor corners (already on ground)
        floor_points = [
            (boundary['min_x'], 0, boundary['min_z']),
            (boundary['max_x'], 0, boundary['min_z']),
            (boundary['max_x'], 0, boundary['max_z']),
            (boundary['min_x'], 0, boundary['max_z'])
        ]
        
        # Project points to screen
        screen_points = []
        for point in floor_points:
            screen_point = self.perspective_projection(point[0], point[1], point[2])
            screen_points.append(screen_point)
        
        # Draw floor boundary
        for i in range(4):
            next_i = (i + 1) % 4
            if screen_points[i][0] >= 0 and screen_points[next_i][0] >= 0:
                pygame.draw.line(
                    self.screen,
                    wall_color,
                    screen_points[i],
                    screen_points[next_i],
                    1
                )

    def draw_debug_info(self):
        """Draw debug information on screen"""
        # Calculate FPS
        if len(self.fps_history) > 0:
            fps = len(self.fps_history) / sum(self.fps_history)
        else:
            fps = 0
        
        # Display FPS
        fps_text = self.font.render(f"FPS: {int(fps)}", True, (255, 255, 255))
        self.screen.blit(fps_text, (10, 10))
        
        # Display motion blur status
        blur_text = self.font.render(
            f"Motion Blur: {'ON' if self.blur_enabled else 'OFF'}", 
            True, (255, 255, 255)
        )
        self.screen.blit(blur_text, (10, 35))
        
        # Display IMU progress if using sensor data
        if hasattr(self, 'sensor_data') and self.sensor_data:
            progress = (self.current_data_index / len(self.sensor_data)) * 100
            imu_text = self.font.render(
                f"IMU Data: {self.current_data_index}/{len(self.sensor_data)} ({progress:.1f}%)",
                True, (255, 255, 255)
            )
            self.screen.blit(imu_text, (10, 60))
        
        # Display recording status
        if self.is_recording:
            rec_text = self.font.render(
                f"RECORDING [{len(self.video_recorder.frames)} frames]",
                True, (255, 0, 0)
            )
            self.screen.blit(rec_text, (self.width - 300, 10))
        
        # Display controls
        controls_text = self.font.render(
            "B: Toggle blur | R: Reset | P: Pause | V: Record | ESC: Quit",
            True, (200, 200, 200)
        )
        self.screen.blit(controls_text, (10, self.height - 30))

    def handle_events(self):
        """Handle user input events"""
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return False
            
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    return False
                
                elif event.key == pygame.K_b:
                    # Toggle motion blur
                    self.blur_enabled = not self.blur_enabled
                
                elif event.key == pygame.K_r:
                    # Reset scene
                    self.floor_headsets = self.create_floor_headsets()
                    if self.sensor_data:
                        self.current_data_index = 0
                
                elif event.key == pygame.K_p:
                    # Pause/resume
                    self.paused = not self.paused
                
                elif event.key == pygame.K_v:
                    # Toggle recording
                    if not self.is_recording:
                        self.start_recording()
                    else:
                        self.stop_recording()
        
        return True

    def start_recording(self):
        """Start video recording"""
        self.is_recording = True
        self.video_recorder.start_recording()
        print("Started recording")

    def stop_recording(self):
        """Stop and save video recording"""
        if not self.is_recording:
            return
        
        self.is_recording = False
        self.video_recorder.stop_recording()
        
        # Save video
        video_path = self.video_recorder.save_video("headset_simulation.mp4")
        print(f"Video saved to: {video_path}")
        
        # Try to generate higher quality video with ffmpeg
        ffmpeg_path = self.video_recorder.generate_ffmpeg_video(quality="high")
        if ffmpeg_path:
            print(f"High quality video saved to: {ffmpeg_path}")


    def load_imu_data(self):
        """Load IMU data for headset rotation"""
        imu_data_paths = ["../IMUdata.csv", "./IMUdata.csv"]
        self.sensor_data = None
        
        for path in imu_data_paths:
            try:
                if os.path.exists(path):
                    print(f"Found IMU data at {path}")
                    parser = SensorDataParser(path)
                    self.sensor_data = parser.parse()
                    print(f"Loaded {len(self.sensor_data)} IMU data points")
                    
                    # Calculate frames needed for 27 seconds at 30fps
                    self.target_frames = 810  # ~27 seconds at 30fps
                    
                    # Calculate IMU samples to process per frame
                    self.samples_per_frame = max(1, len(self.sensor_data) / self.target_frames)
                    print(f"Using ~{self.samples_per_frame:.2f} IMU samples per frame for 27-second video")
                    
                    # Create and calibrate filter
                    self.dr_filter = DeadReckoningFilter(alpha=0.98)
                    self.dr_filter.calibrate(self.sensor_data[:min(100, len(self.sensor_data))])
                    self.current_data_index = 0
                    return
            except Exception as e:
                print(f"Error loading IMU data from {path}: {e}")

    

    def update_main_headset(self, dt):
        """Update main headset orientation based on IMU data - one sample per frame"""
        if self.sensor_data and self.dr_filter:
            if self.current_data_index < len(self.sensor_data):
            # Calculate how many samples to process this frame
                samples_to_process = min(
                    math.ceil(self.samples_per_frame),  # Round up to ensure we use all samples
                    len(self.sensor_data) - self.current_data_index
                )
                
                # Process calculated number of samples
                for _ in range(samples_to_process):
                    sensor_data = self.sensor_data[self.current_data_index]
                    self.current_data_index += 1
                    orientation = self.dr_filter.update(sensor_data)
                
                # Apply the latest orientation
                self.main_headset["model"].model.setQuaternionRotation(orientation)
                
                # Store rotation angles
                roll, pitch, yaw = self.dr_filter.get_euler_angles()
                self.main_headset["rotation"] = [roll, pitch, yaw]
            
            # Calculate progress percentage for display
            self.imu_progress = min(100, (self.current_data_index / len(self.sensor_data)) * 100)
        else:
            # Simple rotation if no IMU data
            self.main_headset["rotation"][0] += dt * 1.0
            self.main_headset["rotation"][1] += dt * 1.5
            self.main_headset["rotation"][2] += dt * 0.8
            
            self.main_headset["model"].model.setRotation(
                self.main_headset["rotation"][0],
                self.main_headset["rotation"][1],
                self.main_headset["rotation"][2]
            )

    def run(self):
        """Main simulation loop"""
        clock = pygame.time.Clock()
        running = True
        
        print("VR Headset Simulation")
        print("Controls: B (blur), R (reset), P (pause), V (record), ESC (quit)")
        
        # Start recording automatically
        self.start_recording()
        
        while running:
            dt = min(clock.tick(60) / 1000.0, 0.1)
            
            # Track frame time for FPS
            self.fps_history.append(dt)
            if len(self.fps_history) > 20:
                self.fps_history.pop(0)
            
            # Handle events
            running = self.handle_events()
            
            # Skip updates if paused
            if not self.paused:
                # Update main headset
                self.update_main_headset(dt)
                
                # Update floor headsets
                self.update_floor_physics(dt)
            
            # Render the scene
            self.render_scene()
            
            # Increment frame counter
            self.frame_count += 1
            
            # Exit either when all IMU data is used or we reach target frames
            if (self.sensor_data and self.current_data_index >= len(self.sensor_data)) or \
            self.frame_count >= self.target_frames:
                if self.is_recording:
                    self.stop_recording()
                print(f"Simulation finished after {self.frame_count} frames")
                break
        
        # Clean up
        pygame.quit()
        print(f"Simulation ended after {self.frame_count} frames")

import pygame
import math
import os
from image import Image, Color
from vector import Vector
from model import Model, CollisionObject, Quaternion
from shape import Triangle, Point
from color_support import ColoredModel
from motion_blur import MotionBlurEffect
from video_recorder import VideoRecorder

# class MotionBlurHeadsetDemo:
#     """
#     A focused demonstration of motion blur with headsets flying through the air.
#     This class is optimized specifically for demonstrating the visual effect of motion blur.
#     """
#     def __init__(self, width=800, height=600, output_dir="motion_blur_demo"):
#         # Initialize pygame
#         pygame.init()
#         self.width = width
#         self.height = height
#         self.screen = pygame.display.set_mode((width, height))
#         pygame.display.set_caption("Motion Blur Headset Demo")
        
#         # Image and Z-buffer
#         self.image = Image(width, height, Color(20, 20, 40, 255))
#         self.zBuffer = [-float('inf')] * width * height
        
#         # Camera and lighting - closer to the action
#         self.camera_pos = Vector(0, 15, -40)
#         self.camera_target = Vector(0, 5, -10)
#         self.light_dir = Vector(0.5, -1, -0.5).normalize()
        
#         # Motion blur effect - higher strength for more pronounced effect
#         self.motion_blur = MotionBlurEffect(blur_strength=2.5) 
#         self.blur_enabled = True
        
#         # Video recording
#         self.output_dir = output_dir
#         if not os.path.exists(output_dir):
#             os.makedirs(output_dir)
            
#         if not os.path.exists(f"output/{output_dir}"):
#             os.makedirs(f"output/{output_dir}")
            
#         self.video_recorder = VideoRecorder(width, height, fps=5)
        
#         # Display
#         self.font = pygame.font.SysFont('Arial', 18)
#         self.frame_count = 0
#         self.split_view_mode = True  # Enable split view to compare with/without blur
        
#         # Setting up the headsets for the demo
#         self.flying_headsets = []
#         self.reference_markers = []
#         self.setup_scene()
    
#     def setup_scene(self):
#         """Set up the scene with multiple flying headsets at different speeds"""
#         # Create several flying headsets at different speeds and colors
        
#         # 1. Main fast headset (very fast for obvious blur)
#         model = Model('./data/headset.obj')
#         model.normalizeGeometry()
#         position = Vector(-30, 15, -20)
#         model.setPosition(position.x, position.y, position.z)
#         # Much higher velocity for very obvious blur effect
#         velocity = Vector(4.0, 0.0, 0.0)  # Increased from 1.0 to 4.0
#         colored_model = ColoredModel(model, diffuse_color=(255, 0, 0))  # Bright red
#         self.flying_headsets.append(CollisionObject(colored_model, position, velocity, radius=1.0))
        
#         # 2. Medium speed headset
#         model = Model('./data/headset.obj')
#         model.normalizeGeometry()
#         position = Vector(-30, 10, -15)
#         model.setPosition(position.x, position.y, position.z)
#         velocity = Vector(2.5, 0.0, 0.0)
#         colored_model = ColoredModel(model, diffuse_color=(0, 255, 0))  # Green
#         self.flying_headsets.append(CollisionObject(colored_model, position, velocity, radius=1.0))
        
#         # 3. Slower headset
#         model = Model('./data/headset.obj')
#         model.normalizeGeometry()
#         position = Vector(-30, 5, -10)
#         model.setPosition(position.x, position.y, position.z)
#         velocity = Vector(1.0, 0.0, 0.0)
#         colored_model = ColoredModel(model, diffuse_color=(0, 0, 255))  # Blue
#         self.flying_headsets.append(CollisionObject(colored_model, position, velocity, radius=1.0))
        
#         # Create vertical reference markers (stationary)
#         for z in range(-25, 0, 5):
#             for y in range(5, 20, 5):
#                 marker_position = Vector(0, y, z)
#                 marker_radius = 0.5
#                 self.reference_markers.append((marker_position, marker_radius))
        
#         # Create horizontal reference grid (helps show motion better)
#         grid_spacing = 5
#         for x in range(-30, 35, grid_spacing):
#             for z in range(-30, 5, grid_spacing):
#                 marker_position = Vector(x, 0.1, z)  # Just above floor
#                 marker_radius = 0.2
#                 self.reference_markers.append((marker_position, marker_radius))
        
#         # Load floor model
#         try:
#             floor_model = Model('./data/floor.obj')
#             floor_model.normalizeGeometry()
#             floor_model.setPosition(0, 0, -20)
#             floor_model.scale = [60.0, 1.0, 40.0]
#             floor_model.updateTransform()
#             self.floor_object = ColoredModel(floor_model, diffuse_color=(150, 150, 150))
            
#             # Add grid texture to floor
#             grid_lines = []
#             for x in range(-30, 35, 5):
#                 grid_lines.append(Vector(x, 0.05, -40))
#                 grid_lines.append(Vector(x, 0.05, 10))
#             for z in range(-40, 15, 5):
#                 grid_lines.append(Vector(-30, 0.05, z))
#                 grid_lines.append(Vector(30, 0.05, z))
                
#             print("Floor model loaded successfully")
#         except Exception as e:
#             print(f"Error loading floor model: {e}")
#             self.floor_object = None

#     def draw_triangle(self, p0, p1, p2, target_image=None, target_zbuffer=None):
#         """Draw a triangle with properly converted integer coordinates"""
#         # Use provided targets or default to self.image and self.zBuffer
#         target_image = target_image if target_image is not None else self.image
#         target_zbuffer = target_zbuffer if target_zbuffer is not None else self.zBuffer
        
#         # Create a Triangle instance
#         tri = Triangle(p0, p1, p2)
        
#         # Calculate bounding box with integer conversion
#         ymin = max(min(p0.y, p1.y, p2.y), 0)
#         ymax = min(max(p0.y, p1.y, p2.y), target_image.height - 1)
        
#         # Convert to integers for range()
#         ymin_int = int(ymin)
#         ymax_int = int(ymax)
        
#         # Iterate over scan lines
#         for y in range(ymin_int, ymax_int + 1):
#             x_values = []
            
#             # Find intersections with edges
#             for edge_start, edge_end in [(p0, p1), (p1, p2), (p2, p0)]:
#                 if (edge_start.y <= y <= edge_end.y) or (edge_end.y <= y <= edge_start.y):
#                     if edge_end.y == edge_start.y:  # Skip horizontal edges
#                         continue
                    
#                     if edge_start.y == y:
#                         x_values.append(edge_start.x)
#                     elif edge_end.y == y:
#                         x_values.append(edge_end.x)
#                     else:
#                         t = (y - edge_start.y) / (edge_end.y - edge_start.y)
#                         x = edge_start.x + t * (edge_end.x - edge_start.x)
#                         x_values.append(x)
            
#             if len(x_values) > 0:
#                 # Sort and convert to integers
#                 if len(x_values) == 1:
#                     x_start = x_values[0]
#                     x_end = x_start
#                 else:
#                     x_start, x_end = sorted(x_values)[:2]
                
#                 x_start_int = max(int(x_start), 0)
#                 x_end_int = min(int(x_end), target_image.width - 1)
                
#                 # Draw horizontal span
#                 for x in range(x_start_int, x_end_int + 1):
#                     point = Point(x, y, color=None)
#                     in_triangle, color, z_value = tri.contains_point(point)
                    
#                     if in_triangle:
#                         # Perform z-buffer check
#                         buffer_index = y * target_image.width + x
#                         if buffer_index < len(target_zbuffer) and target_zbuffer[buffer_index] < z_value:
#                             target_zbuffer[buffer_index] = z_value
#                             target_image.setPixel(x, y, color)
    
#     def perspective_projection(self, x, y, z, width=None, height=None):
#         """Project 3D coordinates to 2D screen space"""
#         if width is None:
#             width = self.width
#         if height is None:
#             height = self.height
        
#         # Calculate vector from camera to point
#         to_point = Vector(
#             x - self.camera_pos.x,
#             y - self.camera_pos.y,
#             z - self.camera_pos.z
#         )
        
#         # Camera orientation vectors
#         forward = Vector(
#             self.camera_target.x - self.camera_pos.x,
#             self.camera_target.y - self.camera_pos.y,
#             self.camera_target.z - self.camera_pos.z
#         ).normalize()
        
#         world_up = Vector(0, 1, 0)
#         right = forward.cross(world_up).normalize()
#         up = right.cross(forward).normalize()
        
#         # Project point onto camera vectors
#         right_comp = to_point * right
#         up_comp = to_point * up
#         forward_comp = to_point * forward
        
#         if forward_comp < 0.1:
#             return -1, -1
        
#         # Apply perspective projection
#         fov = math.pi / 3.0
#         aspect = width / height
        
#         x_ndc = right_comp / (forward_comp * math.tan(fov/2) * aspect)
#         y_ndc = up_comp / (forward_comp * math.tan(fov/2))
        
#         # Convert to screen coordinates
#         screen_x = int((x_ndc + 1.0) * width / 2.0)
#         screen_y = int((-y_ndc + 1.0) * height / 2.0)
        
#         return screen_x, screen_y
    
#     def render_floor(self, target_image=None, target_zbuffer=None):
#         """Render a simple floor grid with a checkered pattern"""
#         # Use provided targets or default to self.image and self.zBuffer
#         target_image = target_image if target_image is not None else self.image
#         target_zbuffer = target_zbuffer if target_zbuffer is not None else self.zBuffer
        
#         size = 30
#         height = 0
        
#         # Floor corners
#         corners = [
#             Vector(-size, height, -size - 10),
#             Vector(size, height, -size - 10),
#             Vector(size, height, size - 10),
#             Vector(-size, height, size - 10)
#         ]
        
#         # Create floor points
#         p0 = Point(corners[0].x, corners[0].y, corners[0].z)
#         p1 = Point(corners[1].x, corners[1].y, corners[1].z)
#         p2 = Point(corners[2].x, corners[2].y, corners[2].z)
#         p3 = Point(corners[3].x, corners[3].y, corners[3].z)
        
#         # Set normal and color
#         normal = Vector(0, 1, 0)
#         for p in [p0, p1, p2, p3]:
#             p.normal = normal
#             p.color = Color(100, 100, 100, 255)
        
#         # Render floor triangles
#         self.draw_triangle(p0, p1, p2, target_image=target_image, target_zbuffer=target_zbuffer)
#         self.draw_triangle(p0, p2, p3, target_image=target_image, target_zbuffer=target_zbuffer)
        
#         # Add grid lines for better motion reference
#         grid_spacing = 5
#         for i in range(-size, size + 1, grid_spacing):
#             # X grid lines
#             start_x = Vector(i, height + 0.01, -size - 10)  # Slightly above floor
#             end_x = Vector(i, height + 0.01, size - 10)
            
#             # Project to screen
#             start_screen_x, start_screen_y = self.perspective_projection(start_x.x, start_x.y, start_x.z)
#             end_screen_x, end_screen_y = self.perspective_projection(end_x.x, end_x.y, end_x.z)
            
#             if start_screen_x >= 0 and end_screen_x >= 0:
#                 # Create points for the line
#                 p_start = Point(start_screen_x, start_screen_y, start_x.z, Color(50, 50, 50, 255))
#                 p_end = Point(end_screen_x, end_screen_y, end_x.z, Color(50, 50, 50, 255))
                
#                 # Draw line using Bresenham's algorithm
#                 self.draw_line(p_start, p_end, target_image=target_image, target_zbuffer=target_zbuffer)
            
#             # Z grid lines
#             start_z = Vector(-size, height + 0.01, -size - 10 + i)
#             end_z = Vector(size, height + 0.01, -size - 10 + i)
            
#             # Project to screen
#             start_screen_x, start_screen_y = self.perspective_projection(start_z.x, start_z.y, start_z.z)
#             end_screen_x, end_screen_y = self.perspective_projection(end_z.x, end_z.y, end_z.z)
            
#             if start_screen_x >= 0 and end_screen_x >= 0:
#                 # Create points for the line
#                 p_start = Point(start_screen_x, start_screen_y, start_z.z, Color(50, 50, 50, 255))
#                 p_end = Point(end_screen_x, end_screen_y, end_z.z, Color(50, 50, 50, 255))
                
#                 # Draw line using Bresenham's algorithm
#                 self.draw_line(p_start, p_end, target_image=target_image, target_zbuffer=target_zbuffer)
    
#     def render_model(self, model_obj, target_image=None, target_zbuffer=None):
#         """Render a 3D model with lighting"""
#         # Use provided targets or default to self.image and self.zBuffer
#         target_image = target_image if target_image is not None else self.image
#         target_zbuffer = target_zbuffer if target_zbuffer is not None else self.zBuffer
        
#         # Get the actual model (handle both direct models and ColoredModel objects)
#         model = getattr(model_obj, 'model', model_obj)
        
#         # Precalculate transformed vertices
#         transformed_vertices = []
#         for i in range(len(model.vertices)):
#             vertex = model.getTransformedVertex(i)
#             transformed_vertices.append(vertex)
        
#         # Calculate face normals and vertex normals
#         face_normals = {}
#         for face in model.faces:
#             v0 = transformed_vertices[face[0]]
#             v1 = transformed_vertices[face[1]]
#             v2 = transformed_vertices[face[2]]
            
#             edge1 = Vector(v1.x - v0.x, v1.y - v0.y, v1.z - v0.z)
#             edge2 = Vector(v2.x - v0.x, v2.y - v0.y, v2.z - v0.z)
#             normal = edge1.cross(edge2).normalize()
            
#             for i in face:
#                 if i not in face_normals:
#                     face_normals[i] = []
#                 face_normals[i].append(normal)
        
#         vertex_normals = []
#         for vert_idx in range(len(model.vertices)):
#             if vert_idx in face_normals:
#                 normal = Vector(0, 0, 0)
#                 for face_normal in face_normals[vert_idx]:
#                     normal = normal + face_normal
#                 vertex_normals.append(normal.normalize())
#             else:
#                 vertex_normals.append(Vector(0, 1, 0))
        
#         # Get model color
#         if hasattr(model_obj, 'diffuse_color'):
#             model_color = model_obj.diffuse_color
#         elif hasattr(model, 'diffuse_color'):
#             model_color = model.diffuse_color
#         else:
#             model_color = (200, 200, 200)
        
#         # Render faces
#         for face in model.faces:
#             v0 = transformed_vertices[face[0]]
#             v1 = transformed_vertices[face[1]]
#             v2 = transformed_vertices[face[2]]
            
#             n0 = vertex_normals[face[0]]
#             n1 = vertex_normals[face[1]]
#             n2 = vertex_normals[face[2]]
            
#             # Backface culling
#             avg_normal = (n0 + n1 + n2).normalize()
#             view_dir = Vector(
#                 self.camera_pos.x - (v0.x + v1.x + v2.x) / 3,
#                 self.camera_pos.y - (v0.y + v1.y + v2.y) / 3,
#                 self.camera_pos.z - (v0.z + v1.z + v2.z) / 3
#             ).normalize()
            
#             if avg_normal * view_dir <= 0:
#                 continue
            
#             # Create triangle points
#             triangle_points = []
#             for v, n in zip([v0, v1, v2], [n0, n1, n2]):
#                 screen_x, screen_y = self.perspective_projection(v.x, v.y, v.z)
                
#                 if screen_x < 0 or screen_y < 0 or screen_x >= self.width or screen_y >= self.height:
#                     continue
                
#                 # Calculate lighting
#                 intensity = max(0.2, n * self.light_dir)
                
#                 r, g, b = model_color
#                 color = Color(
#                     int(r * intensity),
#                     int(g * intensity),
#                     int(b * intensity),
#                     255
#                 )
                
#                 point = Point(int(screen_x), int(screen_y), v.z, color)
#                 point.normal = n
#                 triangle_points.append(point)
            
#             # Render triangle if all points are valid
#             if len(triangle_points) == 3:
#                 self.draw_triangle(triangle_points[0], triangle_points[1], triangle_points[2], 
#                                   target_image=target_image, target_zbuffer=target_zbuffer)
    
#     def render_sphere(self, position, radius, color=(255, 255, 255), target_image=None, target_zbuffer=None):
#         """Render a simple sphere at the given position"""
#         # Use provided targets or default to self.image and self.zBuffer
#         target_image = target_image if target_image is not None else self.image
#         target_zbuffer = target_zbuffer if target_zbuffer is not None else self.zBuffer
        
#         # Project sphere center to screen
#         screen_x, screen_y = self.perspective_projection(position.x, position.y, position.z)
        
#         # Skip if off-screen
#         if screen_x < 0 or screen_y < 0 or screen_x >= self.width or screen_y >= self.height:
#             return
        
#         # Calculate projected radius (rough approximation)
#         distance = (position - self.camera_pos).length()
#         screen_radius = int(radius * 200 / distance)  # Simple projection
        
#         # Skip if too small
#         if screen_radius < 1:
#             return
        
#         # Draw sphere as a filled circle
#         for y in range(max(0, screen_y - screen_radius), min(self.height, screen_y + screen_radius + 1)):
#             for x in range(max(0, screen_x - screen_radius), min(self.width, screen_x + screen_radius + 1)):
#                 dx = x - screen_x
#                 dy = y - screen_y
#                 dist_sq = dx*dx + dy*dy
                
#                 if dist_sq <= screen_radius * screen_radius:
#                     # Calculate z using sphere equation
#                     t = 1.0 - dist_sq / (screen_radius * screen_radius)
#                     z = position.z - radius + 2 * radius * t
                    
#                     # Perform z-buffer check
#                     buffer_index = y * self.width + x
#                     if buffer_index < len(target_zbuffer) and target_zbuffer[buffer_index] < z:
#                         target_zbuffer[buffer_index] = z
                        
#                         # Simple lighting
#                         intensity = 0.5 + 0.5 * t  # Higher in center
#                         r, g, b = color
#                         lit_color = Color(
#                             int(r * intensity),
#                             int(g * intensity),
#                             int(b * intensity),
#                             255
#                         )
#                         target_image.setPixel(x, y, lit_color)
    
#     def render_scene(self):
#         """Render the scene with the flying headsets"""
#         # Clear image and z-buffer
#         self.image = Image(self.width, self.height, Color(20, 20, 40, 255))
#         self.zBuffer = [-float('inf')] * self.width * self.height
        
#         # Create a second buffer for split view comparison
#         if self.split_view_mode:
#             self.image_no_blur = Image(self.width, self.height, Color(20, 20, 40, 255))
#             self.zBuffer_no_blur = [-float('inf')] * self.width * self.height
        
#         # Render floor
#         if hasattr(self, 'floor_object') and self.floor_object:
#             self.render_model(self.floor_object)
#             if self.split_view_mode:
#                 self.render_model(self.floor_object, target_image=self.image_no_blur, target_zbuffer=self.zBuffer_no_blur)
#         else:
#             self.render_floor()
#             if self.split_view_mode:
#                 self.render_floor(target_image=self.image_no_blur, target_zbuffer=self.zBuffer_no_blur)
        
#         # Render reference markers (stationary objects)
#         for marker_pos, marker_radius in self.reference_markers:
#             # Draw simple spheres as markers
#             self.render_sphere(marker_pos, marker_radius, color=(200, 200, 200))
#             if self.split_view_mode:
#                 self.render_sphere(marker_pos, marker_radius, color=(200, 200, 200),
#                                  target_image=self.image_no_blur, target_zbuffer=self.zBuffer_no_blur)
        
#         # Render the flying headsets
#         for headset in self.flying_headsets:
#             self.render_model(headset.model)
#             if self.split_view_mode:
#                 self.render_model(headset.model, target_image=self.image_no_blur, target_zbuffer=self.zBuffer_no_blur)
        
#         # Apply motion blur for the main image
#         if self.blur_enabled:
#             blurred_image = self.motion_blur.per_object_velocity_blur(
#                 self.image,
#                 self.flying_headsets,  # All headsets
#                 self.width,
#                 self.height,
#                 self.perspective_projection
#             )
#         else:
#             blurred_image = self.image
        
#         # For split view, we use half of each image
#         if self.split_view_mode:
#             # Create combined image with a dividing line
#             final_image = Image(self.width, self.height, Color(20, 20, 40, 255))
            
#             # Left half: With motion blur
#             for y in range(self.height):
#                 for x in range(self.width // 2):
#                     idx_blur = (blurred_image.height - y - 1) * blurred_image.width * 4 + x * 4 + (blurred_image.height - y - 1) + 1
#                     idx_final = (final_image.height - y - 1) * final_image.width * 4 + x * 4 + (final_image.height - y - 1) + 1
                    
#                     if idx_blur + 2 < len(blurred_image.buffer) and idx_final + 2 < len(final_image.buffer):
#                         final_image.buffer[idx_final] = blurred_image.buffer[idx_blur]
#                         final_image.buffer[idx_final + 1] = blurred_image.buffer[idx_blur + 1]
#                         final_image.buffer[idx_final + 2] = blurred_image.buffer[idx_blur + 2]
#                         final_image.buffer[idx_final + 3] = blurred_image.buffer[idx_blur + 3]
            
#             # Right half: No motion blur
#             for y in range(self.height):
#                 for x in range(self.width // 2, self.width):
#                     idx_no_blur = (self.image_no_blur.height - y - 1) * self.image_no_blur.width * 4 + x * 4 + (self.image_no_blur.height - y - 1) + 1
#                     idx_final = (final_image.height - y - 1) * final_image.width * 4 + x * 4 + (final_image.height - y - 1) + 1
                    
#                     if idx_no_blur + 2 < len(self.image_no_blur.buffer) and idx_final + 2 < len(final_image.buffer):
#                         final_image.buffer[idx_final] = self.image_no_blur.buffer[idx_no_blur]
#                         final_image.buffer[idx_final + 1] = self.image_no_blur.buffer[idx_no_blur + 1]
#                         final_image.buffer[idx_final + 2] = self.image_no_blur.buffer[idx_no_blur + 2]
#                         final_image.buffer[idx_final + 3] = self.image_no_blur.buffer[idx_no_blur + 3]
#         else:
#             # Use full blurred image
#             final_image = blurred_image
        
#         # Convert image to pygame surface
#         for y in range(self.height):
#             for x in range(self.width):
#                 idx = (final_image.height - y - 1) * final_image.width * 4 + x * 4 + (final_image.height - y - 1) + 1
#                 if idx + 2 < len(final_image.buffer):
#                     r = final_image.buffer[idx]
#                     g = final_image.buffer[idx + 1]
#                     b = final_image.buffer[idx + 2]
#                     self.screen.set_at((x, y), (r, g, b))
        
#         # Draw a dividing line for split view
#         if self.split_view_mode:
#             pygame.draw.line(self.screen, (255, 255, 255), (self.width // 2, 0), (self.width // 2, self.height), 2)
        
#         # Draw information overlay
#         self.draw_overlay()
        
#         # Update display
#         pygame.display.flip()
    
#     def draw_line(self, p0, p1, target_image=None, target_zbuffer=None):
#         """Draw a simple line using Bresenham's algorithm"""
#         # Use provided targets or default to self.image and self.zBuffer
#         target_image = target_image if target_image is not None else self.image
#         target_zbuffer = target_zbuffer if target_zbuffer is not None else self.zBuffer
        
#         # Extract coordinates
#         x0, y0 = int(p0.x), int(p0.y)
#         x1, y1 = int(p1.x), int(p1.y)
        
#         # Get colors
#         color0 = p0.color
#         color1 = p1.color
        
#         # Calculate delta values
#         dx = abs(x1 - x0)
#         dy = abs(y1 - y0)
        
#         # Determine step direction
#         sx = 1 if x0 < x1 else -1
#         sy = 1 if y0 < y1 else -1
        
#         # Initialize error
#         err = dx - dy
        
#         # Bresenham's line algorithm
#         while True:
#             # Draw pixel if in bounds
#             if 0 <= x0 < target_image.width and 0 <= y0 < target_image.height:
#                 # Linear interpolation for z and color
#                 if dx > 0 or dy > 0:
#                     t = abs((x0 - int(p0.x)) / dx if dx > dy else (y0 - int(p0.y)) / dy) if (dx > 0 or dy > 0) else 0
#                 else:
#                     t = 0
                    
#                 # Interpolate z value
#                 z = p0.z * (1 - t) + p1.z * t
                
#                 # Z-buffer check
#                 buffer_index = y0 * target_image.width + x0
#                 if buffer_index < len(target_zbuffer) and target_zbuffer[buffer_index] < z:
#                     target_zbuffer[buffer_index] = z
                    
#                     # Interpolate color
#                     r = int(color0.r() * (1 - t) + color1.r() * t)
#                     g = int(color0.g() * (1 - t) + color1.g() * t)
#                     b = int(color0.b() * (1 - t) + color1.b() * t)
#                     a = int(color0.a() * (1 - t) + color1.a() * t)
                    
#                     target_image.setPixel(x0, y0, Color(r, g, b, a))
            
#             # Exit if we've reached the end point
#             if x0 == x1 and y0 == y1:
#                 break
                
#             # Update error and position
#             e2 = 2 * err
#             if e2 > -dy:
#                 err -= dy
#                 x0 += sx
#             if e2 < dx:
#                 err += dx
#                 y0 += sy
    
#     def draw_overlay(self):
#         """Draw information overlay on the screen"""
#         if self.split_view_mode:
#             # Draw labels for split view
#             left_label = self.font.render("WITH MOTION BLUR", True, (255, 255, 255))
#             right_label = self.font.render("NO MOTION BLUR", True, (255, 255, 255))
            
#             # Position labels in each half
#             left_x = (self.width // 4) - (left_label.get_width() // 2)
#             right_x = (self.width // 4 * 3) - (right_label.get_width() // 2)
            
#             self.screen.blit(left_label, (left_x, 10))
#             self.screen.blit(right_label, (right_x, 10))
#         else:
#             # Display blur status
#             blur_text = self.font.render(
#                 f"Motion Blur: {'ON' if self.blur_enabled else 'OFF'}", 
#                 True, (255, 255, 255)
#             )
#             self.screen.blit(blur_text, (10, 10))
        
#         # Display frame count
#         frame_text = self.font.render(f"Frame: {self.frame_count}", True, (255, 255, 255))
#         self.screen.blit(frame_text, (10, 40))
        
#         # Display headset speeds
#         y_pos = 70
#         for i, headset in enumerate(self.flying_headsets):
#             speed = headset.velocity.length()
#             color_name = "Red" if i == 0 else "Green" if i == 1 else "Blue"
#             speed_text = self.font.render(f"{color_name} Headset Speed: {speed:.1f}", True, (255, 255, 255))
#             self.screen.blit(speed_text, (10, y_pos))
#             y_pos += 25
        
#         # Display controls
#         controls_text = self.font.render(
#             "B: Toggle blur | S: Toggle split view | R: Reset | SPACE: Auto-toggle | ESC: Quit", 
#             True, (200, 200, 200)
#         )
#         self.screen.blit(controls_text, (10, self.height - 30))
    
#     def handle_events(self):
#         """Handle user input events"""
#         for event in pygame.event.get():
#             if event.type == pygame.QUIT:
#                 return False
            
#             elif event.type == pygame.KEYDOWN:
#                 if event.key == pygame.K_ESCAPE:
#                     return False
                
#                 elif event.key == pygame.K_b:
#                     # Toggle motion blur
#                     self.blur_enabled = not self.blur_enabled
#                     print(f"Motion blur: {'ON' if self.blur_enabled else 'OFF'}")
                
#                 elif event.key == pygame.K_s:
#                     # Toggle split view mode
#                     self.split_view_mode = not self.split_view_mode
#                     print(f"Split view mode: {'ON' if self.split_view_mode else 'OFF'}")
                
#                 elif event.key == pygame.K_r:
#                     # Reset demo
#                     self.setup_scene()
#                     self.frame_count = 0
#                     print("Demo reset")
                
#                 elif event.key == pygame.K_SPACE:
#                     # Toggle automatic blur switching
#                     self.auto_toggle = not self.auto_toggle if hasattr(self, 'auto_toggle') else True
#                     print(f"Auto toggle: {'ON' if hasattr(self, 'auto_toggle') and self.auto_toggle else 'OFF'}")
                
#                 elif event.key == pygame.K_UP:
#                     # Increase blur strength
#                     self.motion_blur.blur_strength = min(5.0, self.motion_blur.blur_strength + 0.5)
#                     print(f"Blur strength increased to {self.motion_blur.blur_strength}")
                
#                 elif event.key == pygame.K_DOWN:
#                     # Decrease blur strength
#                     self.motion_blur.blur_strength = max(0.5, self.motion_blur.blur_strength - 0.5)
#                     print(f"Blur strength decreased to {self.motion_blur.blur_strength}")
                
#                 elif event.key == pygame.K_1:
#                     # Speed up headsets
#                     for headset in self.flying_headsets:
#                         headset.velocity.x *= 1.25
#                     print("Headset speeds increased by 25%")
                
#                 elif event.key == pygame.K_2:
#                     # Slow down headsets
#                     for headset in self.flying_headsets:
#                         headset.velocity.x *= 0.8
#                     print("Headset speeds decreased by 20%")
        
#         return True
    
#     def update_headsets(self, dt):
#         """Update all flying headsets' positions and orientations"""
#         # Update each headset
#         for i, headset in enumerate(self.flying_headsets):
#             # Update position based on velocity
#             headset.update(dt)
            
#             # Add vertical oscillation (different for each headset)
#             amplitude = 3.0 - i * 0.5  # Less amplitude for slower headsets
#             frequency = 0.03 + i * 0.01  # Different frequency for each
#             headset.position.y = (15 - i * 5) + amplitude * math.sin(self.frame_count * frequency + i * math.pi/3)
            
#             # Add some Z oscillation too
#             z_amplitude = 1.5 - i * 0.3
#             z_frequency = 0.05 + i * 0.01
#             headset.position.z = -20 + i * 5 + z_amplitude * math.sin(self.frame_count * z_frequency + i * math.pi/3)
            
#             # Update model position
#             headset.model.model.setPosition(
#                 headset.position.x, 
#                 headset.position.y, 
#                 headset.position.z
#             )
            
#             # Add rotation for visual interest
#             rotation_speed = 0.05
#             headset.model.model.setRotation(
#                 self.frame_count * rotation_speed,
#                 self.frame_count * rotation_speed * 0.7,
#                 self.frame_count * rotation_speed * 0.5
#             )
            
#             # If headset reaches right side, reset to left
#             if headset.position.x > 30:
#                 headset.position.x = -30
#                 headset.model.model.setPosition(
#                     headset.position.x, 
#                     headset.position.y, 
#                     headset.position.z
#                 )
    
#     def run_demo(self, frames=240, toggle_interval=60):
#         """
#         Run the motion blur demo for the specified number of frames.
        
#         Args:
#             frames: Total number of frames to render
#             toggle_interval: Number of frames between blur toggles
#         """
#         print(f"Starting Enhanced Motion Blur Headset Demo - {frames} frames")
#         print("Controls:")
#         print("  B: Toggle motion blur on/off")
#         print("  S: Toggle split view (left: with blur, right: without blur)")
#         print("  SPACE: Toggle automatic blur switching")
#         print("  UP/DOWN: Increase/decrease blur strength")
#         print("  1/2: Increase/decrease headset speeds")
#         print("  R: Reset demo")
#         print("  ESC: Quit")
#         print(f"Motion blur will toggle every {toggle_interval} frames if auto-toggle is enabled")
        
#         # Start video recording
#         self.video_recorder.start_recording()
        
#         # Main demo loop
#         running = True
#         clock = pygame.time.Clock()
#         self.auto_toggle = True
        
#         # Save all frames option - can be expensive on disk space
#         save_all_frames = False
#         save_interval = 10  # Save every 10th frame to save disk space
        
#         while running and self.frame_count < frames:
#             # Limit to 30 FPS
#             dt = clock.tick(30) / 1000.0
            
#             # Handle events
#             running = self.handle_events()
#             if not running:
#                 break
            
#             # Update flying headsets
#             self.update_headsets(dt)
            
#             # Auto-toggle blur if enabled (only in non-split view mode)
#             if hasattr(self, 'auto_toggle') and self.auto_toggle and not self.split_view_mode and self.frame_count % toggle_interval == 0:
#                 self.blur_enabled = not self.blur_enabled
#                 print(f"Frame {self.frame_count}: Motion blur {'enabled' if self.blur_enabled else 'disabled'}")
            
#             # Render scene
#             self.render_scene()
            
#             # Capture frame for video
#             self.video_recorder.capture_frame(self.screen)
            
#             # Save individual frames (either all or at intervals)
#             if save_all_frames or self.frame_count % save_interval == 0:
#                 frame_path = os.path.join(self.output_dir, f"frame_{self.frame_count:04d}.png")
#                 pygame.image.save(self.screen, frame_path)
            
#             # Increment frame counter
#             self.frame_count += 1

#         self.video_recorder.stop_recording()
        
#         # Stop recording and save video
#         video_path = self.video_recorder.save_video(os.path.join(self.output_dir, "motion_blur_demo.mp4"))
#         print(f"Demo completed. Video saved to: {video_path}")
#         if save_all_frames:
#             print(f"All frames saved to: {self.output_dir}")
#         else:
#             print(f"Selected frames saved to: {self.output_dir} (every {save_interval}th frame)")
        
import pygame
import math
import os
from image import Image, Color
from vector import Vector
from model import Model, CollisionObject, Quaternion
from shape import Triangle, Point
from color_support import ColoredModel
from motion_blur import MotionBlurEffect
from video_recorder import VideoRecorder

# class MotionBlurHeadsetDemo:
#     """
#     A focused demonstration of motion blur with headsets flying through the air.
#     This class is optimized specifically for demonstrating the visual effect of motion blur.
#     """
#     def __init__(self, width=800, height=600, output_dir="motion_blur_demo"):
#         # Initialize pygame
#         pygame.init()
#         self.width = width
#         self.height = height
#         self.screen = pygame.display.set_mode((width, height))
#         pygame.display.set_caption("Motion Blur Headset Demo")
        
#         # Image and Z-buffer
#         self.image = Image(width, height, Color(20, 20, 40, 255))
#         self.zBuffer = [-float('inf')] * width * height
        
#         # Camera and lighting - moved back for a wider field of view
#         self.camera_pos = Vector(0, 18, -55)  # Moved back from z=-40 to z=-55
#         self.camera_target = Vector(0, 5, -10)
#         self.light_dir = Vector(0.5, -1, -0.5).normalize()
        
#         # Motion blur effect - higher strength for more pronounced effect
#         self.motion_blur = MotionBlurEffect(blur_strength=3.0)  # Increased from 2.5 to 3.0 for more visible effect
#         self.blur_enabled = True
        
#         # Video recording
#         self.output_dir = output_dir
#         if not os.path.exists(output_dir):
#             os.makedirs(output_dir)
        
#         # Display
#         self.font = pygame.font.SysFont('Arial', 18)
#         self.frame_count = 0
#         self.split_view_mode = True  # Enable split view to compare with/without blur
        
#         # Setting up the headsets for the demo
#         self.flying_headsets = []
#         self.reference_markers = []
#         self.setup_scene()
    
#     def setup_scene(self):
#         """Set up the scene with multiple flying headsets at different speeds"""
#         # Create several flying headsets at different speeds and colors
        
#         # Create a much wider area for headsets to travel
#         area_width = 80  # Increased from 30 to 80 (start at -40, end at +40)
        
#         # 1. Main fast headset (reduced speed for better visibility of blur)
#         model = Model('./data/headset.obj')
#         model.normalizeGeometry()
#         position = Vector(-40, 15, -35)  # Moved farther back from -20 to -35
#         model.setPosition(position.x, position.y, position.z)
#         # Reduced speed for better visualization of motion blur
#         velocity = Vector(1.0, 0.0, 0.0)  # Further reduced from 1.5 to 1.0
#         colored_model = ColoredModel(model, diffuse_color=(255, 0, 0))  # Bright red
#         self.flying_headsets.append(CollisionObject(colored_model, position, velocity, radius=1.0))
        
#         # 2. Medium speed headset
#         model = Model('./data/headset.obj')
#         model.normalizeGeometry()
#         position = Vector(-40, 10, -30)  # Moved farther back from -15 to -30
#         model.setPosition(position.x, position.y, position.z)
#         velocity = Vector(0.7, 0.0, 0.0)  # Reduced from 1.0 to 0.7
#         colored_model = ColoredModel(model, diffuse_color=(0, 255, 0))  # Green
#         self.flying_headsets.append(CollisionObject(colored_model, position, velocity, radius=1.0))
        
#         # 3. Slower headset
#         model = Model('./data/headset.obj')
#         model.normalizeGeometry()
#         position = Vector(-40, 5, -25)  # Moved farther back from -10 to -25
#         model.setPosition(position.x, position.y, position.z)
#         velocity = Vector(0.4, 0.0, 0.0)  # Reduced from 0.5 to 0.4
#         colored_model = ColoredModel(model, diffuse_color=(0, 0, 255))  # Blue
#         self.flying_headsets.append(CollisionObject(colored_model, position, velocity, radius=1.0))
        
#         # Create vertical reference markers (stationary)
#         for z in range(-40, 0, 10):  # Wider range for z
#             for y in range(5, 20, 5):
#                 marker_position = Vector(0, y, z)
#                 marker_radius = 0.5
#                 self.reference_markers.append((marker_position, marker_radius))
        
#         # Create horizontal reference grid (helps show motion better)
#         grid_spacing = 10  # Increased from 5 to 10
#         for x in range(-40, 45, grid_spacing):  # Wider grid
#             for z in range(-45, 5, grid_spacing):
#                 marker_position = Vector(x, 0.1, z)  # Just above floor
#                 marker_radius = 0.2
#                 self.reference_markers.append((marker_position, marker_radius))
        
#         # Load floor model
#         try:
#             floor_model = Model('./data/floor.obj')
#             floor_model.normalizeGeometry()
#             floor_model.setPosition(0, 0, -20)
#             floor_model.scale = [90.0, 1.0, 60.0]  # Increased from [60, 1, 40] to [90, 1, 60]
#             floor_model.updateTransform()
#             self.floor_object = ColoredModel(floor_model, diffuse_color=(150, 150, 150))
            
#             # Add grid texture to floor
#             grid_lines = []
#             for x in range(-40, 45, 10):  # Wider grid
#                 grid_lines.append(Vector(x, 0.05, -50))
#                 grid_lines.append(Vector(x, 0.05, 10))
#             for z in range(-50, 15, 10):
#                 grid_lines.append(Vector(-40, 0.05, z))
#                 grid_lines.append(Vector(40, 0.05, z))
                
#             print("Floor model loaded successfully")
#         except Exception as e:
#             print(f"Error loading floor model: {e}")
#             self.floor_object = None
    
#     def perspective_projection(self, x, y, z, width=None, height=None):
#         """Project 3D coordinates to 2D screen space"""
#         if width is None:
#             width = self.width
#         if height is None:
#             height = self.height
        
#         # Calculate vector from camera to point
#         to_point = Vector(
#             x - self.camera_pos.x,
#             y - self.camera_pos.y,
#             z - self.camera_pos.z
#         )
        
#         # Camera orientation vectors
#         forward = Vector(
#             self.camera_target.x - self.camera_pos.x,
#             self.camera_target.y - self.camera_pos.y,
#             self.camera_target.z - self.camera_pos.z
#         ).normalize()
        
#         world_up = Vector(0, 1, 0)
#         right = forward.cross(world_up).normalize()
#         up = right.cross(forward).normalize()
        
#         # Project point onto camera vectors
#         right_comp = to_point * right
#         up_comp = to_point * up
#         forward_comp = to_point * forward
        
#         if forward_comp < 0.1:
#             return -1, -1
        
#         # Apply perspective projection
#         fov = math.pi / 3.0
#         aspect = width / height
        
#         x_ndc = right_comp / (forward_comp * math.tan(fov/2) * aspect)
#         y_ndc = up_comp / (forward_comp * math.tan(fov/2))
        
#         # Convert to screen coordinates
#         screen_x = int((x_ndc + 1.0) * width / 2.0)
#         screen_y = int((-y_ndc + 1.0) * height / 2.0)
        
#         return screen_x, screen_y
    
#     def render_floor(self, target_image=None, target_zbuffer=None):
#         """Render a simple floor grid with a checkered pattern"""
#         # Use provided targets or default to self.image and self.zBuffer
#         target_image = target_image if target_image is not None else self.image
#         target_zbuffer = target_zbuffer if target_zbuffer is not None else self.zBuffer
        
#         size = 30
#         height = 0
        
#         # Floor corners
#         corners = [
#             Vector(-size, height, -size - 10),
#             Vector(size, height, -size - 10),
#             Vector(size, height, size - 10),
#             Vector(-size, height, size - 10)
#         ]
        
#         # Create floor points
#         p0 = Point(corners[0].x, corners[0].y, corners[0].z)
#         p1 = Point(corners[1].x, corners[1].y, corners[1].z)
#         p2 = Point(corners[2].x, corners[2].y, corners[2].z)
#         p3 = Point(corners[3].x, corners[3].y, corners[3].z)
        
#         # Set normal and color
#         normal = Vector(0, 1, 0)
#         for p in [p0, p1, p2, p3]:
#             p.normal = normal
#             p.color = Color(100, 100, 100, 255)
        
#         # Render floor triangles
#         self.draw_triangle(p0, p1, p2, target_image=target_image, target_zbuffer=target_zbuffer)
#         self.draw_triangle(p0, p2, p3, target_image=target_image, target_zbuffer=target_zbuffer)
        
#         # Add grid lines for better motion reference
#         grid_spacing = 5
#         for i in range(-size, size + 1, grid_spacing):
#             # X grid lines
#             start_x = Vector(i, height + 0.01, -size - 10)  # Slightly above floor
#             end_x = Vector(i, height + 0.01, size - 10)
            
#             # Project to screen
#             start_screen_x, start_screen_y = self.perspective_projection(start_x.x, start_x.y, start_x.z)
#             end_screen_x, end_screen_y = self.perspective_projection(end_x.x, end_x.y, end_x.z)
            
#             if start_screen_x >= 0 and end_screen_x >= 0:
#                 # Create points for the line
#                 p_start = Point(start_screen_x, start_screen_y, start_x.z, Color(50, 50, 50, 255))
#                 p_end = Point(end_screen_x, end_screen_y, end_x.z, Color(50, 50, 50, 255))
                
#                 # Draw line using Bresenham's algorithm
#                 self.draw_line(p_start, p_end, target_image=target_image, target_zbuffer=target_zbuffer)
            
#             # Z grid lines
#             start_z = Vector(-size, height + 0.01, -size - 10 + i)
#             end_z = Vector(size, height + 0.01, -size - 10 + i)
            
#             # Project to screen
#             start_screen_x, start_screen_y = self.perspective_projection(start_z.x, start_z.y, start_z.z)
#             end_screen_x, end_screen_y = self.perspective_projection(end_z.x, end_z.y, end_z.z)
            
#             if start_screen_x >= 0 and end_screen_x >= 0:
#                 # Create points for the line
#                 p_start = Point(start_screen_x, start_screen_y, start_z.z, Color(50, 50, 50, 255))
#                 p_end = Point(end_screen_x, end_screen_y, end_z.z, Color(50, 50, 50, 255))
                
#                 # Draw line using Bresenham's algorithm
#                 self.draw_line(p_start, p_end, target_image=target_image, target_zbuffer=target_zbuffer)
    
#     def render_model(self, model_obj, target_image=None, target_zbuffer=None):
#         """Render a 3D model with lighting"""
#         # Use provided targets or default to self.image and self.zBuffer
#         target_image = target_image if target_image is not None else self.image
#         target_zbuffer = target_zbuffer if target_zbuffer is not None else self.zBuffer
        
#         # Get the actual model (handle both direct models and ColoredModel objects)
#         model = getattr(model_obj, 'model', model_obj)
        
#         # Precalculate transformed vertices
#         transformed_vertices = []
#         for i in range(len(model.vertices)):
#             vertex = model.getTransformedVertex(i)
#             transformed_vertices.append(vertex)
        
#         # Calculate face normals and vertex normals
#         face_normals = {}
#         for face in model.faces:
#             v0 = transformed_vertices[face[0]]
#             v1 = transformed_vertices[face[1]]
#             v2 = transformed_vertices[face[2]]
            
#             edge1 = Vector(v1.x - v0.x, v1.y - v0.y, v1.z - v0.z)
#             edge2 = Vector(v2.x - v0.x, v2.y - v0.y, v2.z - v0.z)
#             normal = edge1.cross(edge2).normalize()
            
#             for i in face:
#                 if i not in face_normals:
#                     face_normals[i] = []
#                 face_normals[i].append(normal)
        
#         vertex_normals = []
#         for vert_idx in range(len(model.vertices)):
#             if vert_idx in face_normals:
#                 normal = Vector(0, 0, 0)
#                 for face_normal in face_normals[vert_idx]:
#                     normal = normal + face_normal
#                 vertex_normals.append(normal.normalize())
#             else:
#                 vertex_normals.append(Vector(0, 1, 0))
        
#         # Get model color
#         if hasattr(model_obj, 'diffuse_color'):
#             model_color = model_obj.diffuse_color
#         elif hasattr(model, 'diffuse_color'):
#             model_color = model.diffuse_color
#         else:
#             model_color = (200, 200, 200)
        
#         # Render faces
#         for face in model.faces:
#             v0 = transformed_vertices[face[0]]
#             v1 = transformed_vertices[face[1]]
#             v2 = transformed_vertices[face[2]]
            
#             n0 = vertex_normals[face[0]]
#             n1 = vertex_normals[face[1]]
#             n2 = vertex_normals[face[2]]
            
#             # Backface culling
#             avg_normal = (n0 + n1 + n2).normalize()
#             view_dir = Vector(
#                 self.camera_pos.x - (v0.x + v1.x + v2.x) / 3,
#                 self.camera_pos.y - (v0.y + v1.y + v2.y) / 3,
#                 self.camera_pos.z - (v0.z + v1.z + v2.z) / 3
#             ).normalize()
            
#             if avg_normal * view_dir <= 0:
#                 continue
            
#             # Create triangle points
#             triangle_points = []
#             for v, n in zip([v0, v1, v2], [n0, n1, n2]):
#                 screen_x, screen_y = self.perspective_projection(v.x, v.y, v.z)
                
#                 if screen_x < 0 or screen_y < 0 or screen_x >= self.width or screen_y >= self.height:
#                     continue
                
#                 # Calculate lighting
#                 intensity = max(0.2, n * self.light_dir)
                
#                 r, g, b = model_color
#                 color = Color(
#                     int(r * intensity),
#                     int(g * intensity),
#                     int(b * intensity),
#                     255
#                 )
                
#                 point = Point(int(screen_x), int(screen_y), v.z, color)
#                 point.normal = n
#                 triangle_points.append(point)
            
#             # Render triangle if all points are valid
#             if len(triangle_points) == 3:
#                 self.draw_triangle(triangle_points[0], triangle_points[1], triangle_points[2], 
#                                   target_image=target_image, target_zbuffer=target_zbuffer)
    
#     def render_sphere(self, position, radius, color=(255, 255, 255), target_image=None, target_zbuffer=None):
#         """Render a simple sphere at the given position"""
#         # Use provided targets or default to self.image and self.zBuffer
#         target_image = target_image if target_image is not None else self.image
#         target_zbuffer = target_zbuffer if target_zbuffer is not None else self.zBuffer
        
#         # Project sphere center to screen
#         screen_x, screen_y = self.perspective_projection(position.x, position.y, position.z)
        
#         # Skip if off-screen
#         if screen_x < 0 or screen_y < 0 or screen_x >= self.width or screen_y >= self.height:
#             return
        
#         # Calculate projected radius (rough approximation)
#         distance = (position - self.camera_pos).length()
#         screen_radius = int(radius * 200 / distance)  # Simple projection
        
#         # Skip if too small
#         if screen_radius < 1:
#             return
        
#         # Draw sphere as a filled circle
#         for y in range(max(0, screen_y - screen_radius), min(self.height, screen_y + screen_radius + 1)):
#             for x in range(max(0, screen_x - screen_radius), min(self.width, screen_x + screen_radius + 1)):
#                 dx = x - screen_x
#                 dy = y - screen_y
#                 dist_sq = dx*dx + dy*dy
                
#                 if dist_sq <= screen_radius * screen_radius:
#                     # Calculate z using sphere equation
#                     t = 1.0 - dist_sq / (screen_radius * screen_radius)
#                     z = position.z - radius + 2 * radius * t
                    
#                     # Perform z-buffer check
#                     buffer_index = y * self.width + x
#                     if buffer_index < len(target_zbuffer) and target_zbuffer[buffer_index] < z:
#                         target_zbuffer[buffer_index] = z
                        
#                         # Simple lighting
#                         intensity = 0.5 + 0.5 * t  # Higher in center
#                         r, g, b = color
#                         lit_color = Color(
#                             int(r * intensity),
#                             int(g * intensity),
#                             int(b * intensity),
#                             255
#                         )
#                         target_image.setPixel(x, y, lit_color)
    
#     def render_scene(self):
#         """Render the scene with the flying headsets"""
#         # Clear image and z-buffer
#         self.image = Image(self.width, self.height, Color(20, 20, 40, 255))
#         self.zBuffer = [-float('inf')] * self.width * self.height
        
#         # Create a second buffer for split view comparison
#         if self.split_view_mode:
#             self.image_no_blur = Image(self.width, self.height, Color(20, 20, 40, 255))
#             self.zBuffer_no_blur = [-float('inf')] * self.width * self.height
        
#         # Render floor
#         if hasattr(self, 'floor_object') and self.floor_object:
#             self.render_model(self.floor_object)
#             if self.split_view_mode:
#                 self.render_model(self.floor_object, target_image=self.image_no_blur, target_zbuffer=self.zBuffer_no_blur)
#         else:
#             self.render_floor()
#             if self.split_view_mode:
#                 self.render_floor(target_image=self.image_no_blur, target_zbuffer=self.zBuffer_no_blur)
        
#         # Render reference markers (stationary objects)
#         for marker_pos, marker_radius in self.reference_markers:
#             # Draw simple spheres as markers
#             self.render_sphere(marker_pos, marker_radius, color=(200, 200, 200))
#             if self.split_view_mode:
#                 self.render_sphere(marker_pos, marker_radius, color=(200, 200, 200),
#                                  target_image=self.image_no_blur, target_zbuffer=self.zBuffer_no_blur)
        
#         # Render the flying headsets
#         for headset in self.flying_headsets:
#             self.render_model(headset.model)
#             if self.split_view_mode:
#                 self.render_model(headset.model, target_image=self.image_no_blur, target_zbuffer=self.zBuffer_no_blur)
        
#         # Apply motion blur for the main image
#         if self.blur_enabled:
#             blurred_image = self.motion_blur.per_object_velocity_blur(
#                 self.image,
#                 self.flying_headsets,  # All headsets
#                 self.width,
#                 self.height,
#                 self.perspective_projection
#             )
#         else:
#             blurred_image = self.image
        
#         # For split view, we use half of each image
#         if self.split_view_mode:
#             # Create combined image with a dividing line
#             final_image = Image(self.width, self.height, Color(20, 20, 40, 255))
            
#             # Left half: With motion blur
#             for y in range(self.height):
#                 for x in range(self.width // 2):
#                     idx_blur = (blurred_image.height - y - 1) * blurred_image.width * 4 + x * 4 + (blurred_image.height - y - 1) + 1
#                     idx_final = (final_image.height - y - 1) * final_image.width * 4 + x * 4 + (final_image.height - y - 1) + 1
                    
#                     if idx_blur + 2 < len(blurred_image.buffer) and idx_final + 2 < len(final_image.buffer):
#                         final_image.buffer[idx_final] = blurred_image.buffer[idx_blur]
#                         final_image.buffer[idx_final + 1] = blurred_image.buffer[idx_blur + 1]
#                         final_image.buffer[idx_final + 2] = blurred_image.buffer[idx_blur + 2]
#                         final_image.buffer[idx_final + 3] = blurred_image.buffer[idx_blur + 3]
            
#             # Right half: No motion blur
#             for y in range(self.height):
#                 for x in range(self.width // 2, self.width):
#                     idx_no_blur = (self.image_no_blur.height - y - 1) * self.image_no_blur.width * 4 + x * 4 + (self.image_no_blur.height - y - 1) + 1
#                     idx_final = (final_image.height - y - 1) * final_image.width * 4 + x * 4 + (final_image.height - y - 1) + 1
                    
#                     if idx_no_blur + 2 < len(self.image_no_blur.buffer) and idx_final + 2 < len(final_image.buffer):
#                         final_image.buffer[idx_final] = self.image_no_blur.buffer[idx_no_blur]
#                         final_image.buffer[idx_final + 1] = self.image_no_blur.buffer[idx_no_blur + 1]
#                         final_image.buffer[idx_final + 2] = self.image_no_blur.buffer[idx_no_blur + 2]
#                         final_image.buffer[idx_final + 3] = self.image_no_blur.buffer[idx_no_blur + 3]
#         else:
#             # Use full blurred image
#             final_image = blurred_image
        
#         # Convert image to pygame surface
#         for y in range(self.height):
#             for x in range(self.width):
#                 idx = (final_image.height - y - 1) * final_image.width * 4 + x * 4 + (final_image.height - y - 1) + 1
#                 if idx + 2 < len(final_image.buffer):
#                     r = final_image.buffer[idx]
#                     g = final_image.buffer[idx + 1]
#                     b = final_image.buffer[idx + 2]
#                     self.screen.set_at((x, y), (r, g, b))
        
#         # Draw a dividing line for split view
#         if self.split_view_mode:
#             pygame.draw.line(self.screen, (255, 255, 255), (self.width // 2, 0), (self.width // 2, self.height), 2)
        
#         # Draw information overlay
#         self.draw_overlay()
        
#         # Update display
#         pygame.display.flip()
        
#     def draw_triangle(self, p0, p1, p2, target_image=None, target_zbuffer=None):
#         """Draw a triangle with properly converted integer coordinates"""
#         # Use provided targets or default to self.image and self.zBuffer
#         target_image = target_image if target_image is not None else self.image
#         target_zbuffer = target_zbuffer if target_zbuffer is not None else self.zBuffer
        
#         # Create a Triangle instance
#         tri = Triangle(p0, p1, p2)
        
#         # Calculate bounding box with integer conversion
#         ymin = max(min(p0.y, p1.y, p2.y), 0)
#         ymax = min(max(p0.y, p1.y, p2.y), target_image.height - 1)
        
#         # Convert to integers for range()
#         ymin_int = int(ymin)
#         ymax_int = int(ymax)
        
#         # Iterate over scan lines
#         for y in range(ymin_int, ymax_int + 1):
#             x_values = []
            
#             # Find intersections with edges
#             for edge_start, edge_end in [(p0, p1), (p1, p2), (p2, p0)]:
#                 if (edge_start.y <= y <= edge_end.y) or (edge_end.y <= y <= edge_start.y):
#                     if edge_end.y == edge_start.y:  # Skip horizontal edges
#                         continue
                    
#                     if edge_start.y == y:
#                         x_values.append(edge_start.x)
#                     elif edge_end.y == y:
#                         x_values.append(edge_end.x)
#                     else:
#                         t = (y - edge_start.y) / (edge_end.y - edge_start.y)
#                         x = edge_start.x + t * (edge_end.x - edge_start.x)
#                         x_values.append(x)
            
#             if len(x_values) > 0:
#                 # Sort and convert to integers
#                 if len(x_values) == 1:
#                     x_start = x_values[0]
#                     x_end = x_start
#                 else:
#                     x_start, x_end = sorted(x_values)[:2]
                
#                 x_start_int = max(int(x_start), 0)
#                 x_end_int = min(int(x_end), target_image.width - 1)
                
#                 # Draw horizontal span
#                 for x in range(x_start_int, x_end_int + 1):
#                     point = Point(x, y, color=None)
#                     in_triangle, color, z_value = tri.contains_point(point)
                    
#                     if in_triangle:
#                         # Perform z-buffer check
#                         buffer_index = y * target_image.width + x
#                         if buffer_index < len(target_zbuffer) and target_zbuffer[buffer_index] < z_value:
#                             target_zbuffer[buffer_index] = z_value
#                             target_image.setPixel(x, y, color)
    
#     def draw_line(self, p0, p1, target_image=None, target_zbuffer=None):
#         """Draw a simple line using Bresenham's algorithm"""
#         # Use provided targets or default to self.image and self.zBuffer
#         target_image = target_image if target_image is not None else self.image
#         target_zbuffer = target_zbuffer if target_zbuffer is not None else self.zBuffer
        
#         # Extract coordinates
#         x0, y0 = int(p0.x), int(p0.y)
#         x1, y1 = int(p1.x), int(p1.y)
        
#         # Get colors
#         color0 = p0.color
#         color1 = p1.color
        
#         # Calculate delta values
#         dx = abs(x1 - x0)
#         dy = abs(y1 - y0)
        
#         # Determine step direction
#         sx = 1 if x0 < x1 else -1
#         sy = 1 if y0 < y1 else -1
        
#         # Initialize error
#         err = dx - dy
        
#         # Bresenham's line algorithm
#         while True:
#             # Draw pixel if in bounds
#             if 0 <= x0 < target_image.width and 0 <= y0 < target_image.height:
#                 # Linear interpolation for z and color
#                 if dx > 0 or dy > 0:
#                     t = abs((x0 - int(p0.x)) / dx if dx > dy else (y0 - int(p0.y)) / dy) if (dx > 0 or dy > 0) else 0
#                 else:
#                     t = 0
                    
#                 # Interpolate z value
#                 z = p0.z * (1 - t) + p1.z * t
                
#                 # Z-buffer check
#                 buffer_index = y0 * target_image.width + x0
#                 if buffer_index < len(target_zbuffer) and target_zbuffer[buffer_index] < z:
#                     target_zbuffer[buffer_index] = z
                    
#                     # Interpolate color
#                     r = int(color0.r() * (1 - t) + color1.r() * t)
#                     g = int(color0.g() * (1 - t) + color1.g() * t)
#                     b = int(color0.b() * (1 - t) + color1.b() * t)
#                     a = int(color0.a() * (1 - t) + color1.a() * t)
                    
#                     target_image.setPixel(x0, y0, Color(r, g, b, a))
            
#             # Exit if we've reached the end point
#             if x0 == x1 and y0 == y1:
#                 break
                
#             # Update error and position
#             e2 = 2 * err
#             if e2 > -dy:
#                 err -= dy
#                 x0 += sx
#             if e2 < dx:
#                 err += dx
#                 y0 += sy
    
#     def draw_overlay(self):
#         """Draw information overlay on the screen"""
#         if self.split_view_mode:
#             # Draw labels for split view
#             left_label = self.font.render("WITH MOTION BLUR", True, (255, 255, 255))
#             right_label = self.font.render("NO MOTION BLUR", True, (255, 255, 255))
            
#             # Position labels in each half
#             left_x = (self.width // 4) - (left_label.get_width() // 2)
#             right_x = (self.width // 4 * 3) - (right_label.get_width() // 2)
            
#             self.screen.blit(left_label, (left_x, 10))
#             self.screen.blit(right_label, (right_x, 10))
#         else:
#             # Display blur status
#             blur_text = self.font.render(
#                 f"Motion Blur: {'ON' if self.blur_enabled else 'OFF'}", 
#                 True, (255, 255, 255)
#             )
#             self.screen.blit(blur_text, (10, 10))
        
#         # Display frame count
#         frame_text = self.font.render(f"Frame: {self.frame_count}", True, (255, 255, 255))
#         self.screen.blit(frame_text, (10, 40))
        
#         # Display headset speeds
#         y_pos = 70
#         for i, headset in enumerate(self.flying_headsets):
#             speed = headset.velocity.length()
#             color_name = "Red" if i == 0 else "Green" if i == 1 else "Blue"
#             speed_text = self.font.render(f"{color_name} Headset Speed: {speed:.1f}", True, (255, 255, 255))
#             self.screen.blit(speed_text, (10, y_pos))
#             y_pos += 25
        
#         # Display controls
#         controls_text = self.font.render(
#             "B: Toggle blur | S: Toggle split view | R: Reset | SPACE: Auto-toggle | ESC: Quit", 
#             True, (200, 200, 200)
#         )
#         self.screen.blit(controls_text, (10, self.height - 30))
    
#     def handle_events(self):
#         """Handle user input events"""
#         for event in pygame.event.get():
#             if event.type == pygame.QUIT:
#                 return False
            
#             elif event.type == pygame.KEYDOWN:
#                 if event.key == pygame.K_ESCAPE:
#                     return False
                
#                 elif event.key == pygame.K_b:
#                     # Toggle motion blur
#                     self.blur_enabled = not self.blur_enabled
#                     print(f"Motion blur: {'ON' if self.blur_enabled else 'OFF'}")
                
#                 elif event.key == pygame.K_s:
#                     # Toggle split view mode
#                     self.split_view_mode = not self.split_view_mode
#                     print(f"Split view mode: {'ON' if self.split_view_mode else 'OFF'}")
                
#                 elif event.key == pygame.K_r:
#                     # Reset demo
#                     self.setup_scene()
#                     self.frame_count = 0
#                     print("Demo reset")
                
#                 elif event.key == pygame.K_SPACE:
#                     # Toggle automatic blur switching
#                     self.auto_toggle = not self.auto_toggle if hasattr(self, 'auto_toggle') else True
#                     print(f"Auto toggle: {'ON' if hasattr(self, 'auto_toggle') and self.auto_toggle else 'OFF'}")
                
#                 elif event.key == pygame.K_UP:
#                     # Increase blur strength
#                     self.motion_blur.blur_strength = min(5.0, self.motion_blur.blur_strength + 0.5)
#                     print(f"Blur strength increased to {self.motion_blur.blur_strength}")
                
#                 elif event.key == pygame.K_DOWN:
#                     # Decrease blur strength
#                     self.motion_blur.blur_strength = max(0.5, self.motion_blur.blur_strength - 0.5)
#                     print(f"Blur strength decreased to {self.motion_blur.blur_strength}")
                
#                 elif event.key == pygame.K_1:
#                     # Speed up headsets
#                     for headset in self.flying_headsets:
#                         headset.velocity.x *= 1.25
#                     print("Headset speeds increased by 25%")
                
#                 elif event.key == pygame.K_2:
#                     # Slow down headsets
#                     for headset in self.flying_headsets:
#                         headset.velocity.x *= 0.8
#                     print("Headset speeds decreased by 20%")
        
#         return True
    
#     def update_headsets(self, dt):
#         """Update all flying headsets' positions and orientations"""
#         # Update each headset
#         for i, headset in enumerate(self.flying_headsets):
#             # Update position based on velocity
#             headset.update(dt)
            
#             # Add vertical oscillation (different for each headset)
#             amplitude = 3.0 - i * 0.5  # Less amplitude for slower headsets
#             frequency = 0.02 + i * 0.005  # Slower oscillation (reduced from 0.03 to 0.02)
#             headset.position.y = (15 - i * 5) + amplitude * math.sin(self.frame_count * frequency + i * math.pi/3)
            
#             # Add some Z oscillation too
#             z_amplitude = 1.5 - i * 0.3
#             z_frequency = 0.025 + i * 0.005  # Slower oscillation (reduced from 0.05 to 0.025)
#             base_z = -35 + i * 5  # Starting z position
#             headset.position.z = base_z + z_amplitude * math.sin(self.frame_count * z_frequency + i * math.pi/3)
            
#             # Update model position
#             headset.model.model.setPosition(
#                 headset.position.x, 
#                 headset.position.y, 
#                 headset.position.z
#             )
            
#             # Add rotation for visual interest (slower rotation)
#             rotation_speed = 0.03  # Reduced from 0.05 to 0.03
#             headset.model.model.setRotation(
#                 self.frame_count * rotation_speed,
#                 self.frame_count * rotation_speed * 0.7,
#                 self.frame_count * rotation_speed * 0.5
#             )
            
#             # If headset reaches right side, reset to left
#             if headset.position.x > 40:  # Increased from 30 to 40
#                 headset.position.x = -40  # Increased from -30 to -40
#                 headset.model.model.setPosition(
#                     headset.position.x, 
#                     headset.position.y, 
#                     headset.position.z
#                 )
    
#     def run_demo(self, frames=360, toggle_interval=90):
#         """
#         Run the motion blur demo for the specified number of frames.
        
#         Args:
#             frames: Total number of frames to render
#             toggle_interval: Number of frames between blur toggles
#         """
#         print(f"Starting Enhanced Motion Blur Headset Demo - {frames} frames")
#         print("Controls:")
#         print("  B: Toggle motion blur on/off")
#         print("  S: Toggle split view (left: with blur, right: without blur)")
#         print("  SPACE: Toggle automatic blur switching")
#         print("  UP/DOWN: Increase/decrease blur strength")
#         print("  1/2: Increase/decrease headset speeds")
#         print("  R: Reset demo")
#         print("  ESC: Quit")
#         print(f"Motion blur will toggle every {toggle_interval} frames if auto-toggle is enabled")
        
#         # Start video recording - even slower FPS for better visibility
#         self.video_recorder = VideoRecorder(self.width, self.height, fps=10)  # Reduced from 15 to 10 fps
#         self.video_recorder.start_recording()
        
#         # Main demo loop
#         running = True
#         clock = pygame.time.Clock()
#         self.auto_toggle = True
        
#         # Save all frames option - can be expensive on disk space
#         save_all_frames = False
#         save_interval = 5  # Save every 5th frame to show more gradual progression
        
#         while running and self.frame_count < frames:
#             # Limit to 10 FPS instead of 15 for even slower motion
#             target_fps = 10
#             dt = clock.tick(target_fps) / 1000.0
            
#             # Further slow down the physics for better visibility (reduce effective dt)
#             physics_slowdown = 0.3  # 30% speed physics (reduced from 0.5)
#             effective_dt = dt * physics_slowdown
            
#             # Handle events
#             running = self.handle_events()
#             if not running:
#                 break
            
#             # Update flying headsets with slower physics
#             self.update_headsets(effective_dt)
            
#             # Auto-toggle blur if enabled (only in non-split view mode)
#             if hasattr(self, 'auto_toggle') and self.auto_toggle and not self.split_view_mode and self.frame_count % toggle_interval == 0:
#                 self.blur_enabled = not self.blur_enabled
#                 print(f"Frame {self.frame_count}: Motion blur {'enabled' if self.blur_enabled else 'disabled'}")
            
#             # Render scene
#             self.render_scene()
            
#             # Capture frame for video
#             self.video_recorder.capture_frame(self.screen)
            
#             # Save individual frames (either all or at intervals)
#             if save_all_frames or self.frame_count % save_interval == 0:
#                 frame_path = os.path.join(self.output_dir, f"frame_{self.frame_count:04d}.png")
#                 pygame.image.save(self.screen, frame_path)
            
#             # Increment frame counter
#             self.frame_count += 1
        
#         # Stop recording and save video
#         video_path = self.video_recorder.save_video(os.path.join(self.output_dir, "motion_blur_demo.mp4"))
#         print(f"Demo completed. Video saved to: {video_path}")
#         if save_all_frames:
#             print(f"All frames saved to: {self.output_dir}")
#         else:
#             print(f"Selected frames saved to: {self.output_dir} (every {save_interval}th frame)")
        
#         # Try to generate higher quality video with ffmpeg - with even slower playback
#         try:
#             ffmpeg_path = self.video_recorder.generate_ffmpeg_video(
#                 output_path=os.path.join(self.output_dir, "motion_blur_demo_slow.mp4"),
#                 quality="high",
#                 extra_args=["-filter:v", "setpts=2.5*PTS"]  # 2.5x slower playback (increased from 2.0)
#             )
#             if ffmpeg_path:
#                 print(f"Slow-motion high quality video saved to: {ffmpeg_path}")
                
#             # Create an extra-slow version for frame-by-frame analysis
#             ffmpeg_path = self.video_recorder.generate_ffmpeg_video(
#                 output_path=os.path.join(self.output_dir, "motion_blur_demo_very_slow.mp4"),
#                 quality="high",
#                 extra_args=["-filter:v", "setpts=5.0*PTS"]  # 5x slower playback
#             )
#             if ffmpeg_path:
#                 print(f"Very slow-motion video saved to: {ffmpeg_path}")
#         except Exception as e:
#             print(f"Failed to generate slow-motion video: {e}")
        
#         # Clean up
#         pygame.quit()

import pygame
import math
import os
from image import Image, Color
from vector import Vector
from model import Model, CollisionObject, Quaternion
from shape import Triangle, Point
from color_support import ColoredModel
from motion_blur import MotionBlurEffect
from video_recorder import VideoRecorder

class MotionBlurHeadsetDemo:
    """
    A focused demonstration of motion blur with headsets flying through the air.
    This class is optimized specifically for demonstrating the visual effect of motion blur.
    """
    def __init__(self, width=800, height=600, output_dir="motion_blur_demo"):
        # Initialize pygame
        pygame.init()
        self.width = width
        self.height = height
        self.screen = pygame.display.set_mode((width, height))
        pygame.display.set_caption("Motion Blur Headset Demo")
        
        # Image and Z-buffer
        self.image = Image(width, height, Color(20, 20, 40, 255))
        self.zBuffer = [-float('inf')] * width * height
        
        # Camera and lighting - moved back for a wider field of view
        self.camera_pos = Vector(0, 18, -55)  # Moved back from z=-40 to z=-55
        self.camera_target = Vector(0, 5, -10)
        self.light_dir = Vector(0.5, -1, -0.5).normalize()
        
        # Motion blur effect - EXTREME blur strength for very obvious effect
        self.motion_blur = MotionBlurEffect(blur_strength=1.0, velocity_scale=2.0, max_samples=10)  # Dramatically increased from 3.0 to 10.0
        self.blur_enabled = True
        
        # Video recording
        self.output_dir = output_dir
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        # Display
        self.font = pygame.font.SysFont('Arial', 18)
        self.frame_count = 0
        self.split_view_mode = True  # Enable split view to compare with/without blur
        
        # Setting up the headsets for the demo
        self.flying_headsets = []
        self.reference_markers = []
        self.setup_scene()
    
    def setup_scene(self):
        """Set up the scene with multiple flying headsets at different speeds"""
        # Create several flying headsets at different speeds and colors
        
        # Create a much wider area for headsets to travel
        area_width = 80  # Increased from 30 to 80 (start at -40, end at +40)
        
        # 1. Main fast headset (much higher speed for dramatic blur)
        model = Model('./data/headset.obj')
        model.normalizeGeometry()
        model.scale = [0.2, 0.2, 0.2]  # Make headset twice as large
        position = Vector(-40, 25, -35)  # Moved farther back from -20 to -35
        model.setPosition(position.x, position.y, position.z)
        # MUCH higher speed for extremely visible motion blur
        velocity = Vector(5.0, 0.0, 0.0)  # Dramatically increased from 1.0 to 5.0
        colored_model = ColoredModel(model, diffuse_color=(255, 0, 0))  # Bright red
        self.flying_headsets.append(CollisionObject(colored_model, position, velocity, radius=2.0))
        
        # 2. Medium speed headset
        model = Model('./data/headset.obj')
        model.normalizeGeometry()
        model.scale = [0.2, 0.2, 0.2]  # Make headset larger
        position = Vector(-40, 15, -30)  # Moved farther back from -15 to -30
        model.setPosition(position.x, position.y, position.z)
        velocity = Vector(3.0, 0.0, 0.0)  # Increased from 0.7 to 3.0
        colored_model = ColoredModel(model, diffuse_color=(0, 255, 0))  # Green
        self.flying_headsets.append(CollisionObject(colored_model, position, velocity, radius=1.75))
        
        # 3. Slower headset
        model = Model('./data/headset.obj')
        model.normalizeGeometry()
        model.scale = [0.2, 0.2, 0.2]   # Make headset larger
        position = Vector(-40, 5, -25)  # Moved farther back from -10 to -25
        model.setPosition(position.x, position.y, position.z)
        velocity = Vector(1.0, 0.0, 0.0)  # Increased from 0.4 to 1.0
        colored_model = ColoredModel(model, diffuse_color=(0, 0, 255))  # Blue
        self.flying_headsets.append(CollisionObject(colored_model, position, velocity, radius=1.5))
        
        # Create vertical reference markers (stationary)
        for z in range(-40, 0, 10):  # Wider range for z
            for y in range(5, 20, 5):
                marker_position = Vector(0, y, z)
                marker_radius = 0.5
                self.reference_markers.append((marker_position, marker_radius))
        
        # Create horizontal reference grid (helps show motion better)
        grid_spacing = 10  # Increased from 5 to 10
        for x in range(-40, 45, grid_spacing):  # Wider grid
            for z in range(-45, 5, grid_spacing):
                marker_position = Vector(x, 0.1, z)  # Just above floor
                marker_radius = 0.2
                self.reference_markers.append((marker_position, marker_radius))
        
        # Load floor model
        try:
            floor_model = Model('./data/floor.obj')
            floor_model.normalizeGeometry()
            floor_model.setPosition(0, 0, -20)
            floor_model.scale = [90.0, 1.0, 60.0]  # Increased from [60, 1, 40] to [90, 1, 60]
            floor_model.updateTransform()
            self.floor_object = ColoredModel(floor_model, diffuse_color=(150, 150, 150))
            
            # Add grid texture to floor
            grid_lines = []
            for x in range(-40, 45, 10):  # Wider grid
                grid_lines.append(Vector(x, 0.05, -50))
                grid_lines.append(Vector(x, 0.05, 10))
            for z in range(-50, 15, 10):
                grid_lines.append(Vector(-40, 0.05, z))
                grid_lines.append(Vector(40, 0.05, z))
                
            print("Floor model loaded successfully")
        except Exception as e:
            print(f"Error loading floor model: {e}")
            self.floor_object = None
    
    def perspective_projection(self, x, y, z, width=None, height=None):
        """Project 3D coordinates to 2D screen space"""
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
    
    def render_floor(self, target_image=None, target_zbuffer=None):
        """Render a simple floor grid with a checkered pattern"""
        # Use provided targets or default to self.image and self.zBuffer
        target_image = target_image if target_image is not None else self.image
        target_zbuffer = target_zbuffer if target_zbuffer is not None else self.zBuffer
        
        size = 30
        height = 0
        
        # Floor corners
        corners = [
            Vector(-size, height, -size - 10),
            Vector(size, height, -size - 10),
            Vector(size, height, size - 10),
            Vector(-size, height, size - 10)
        ]
        
        # Create floor points
        p0 = Point(corners[0].x, corners[0].y, corners[0].z)
        p1 = Point(corners[1].x, corners[1].y, corners[1].z)
        p2 = Point(corners[2].x, corners[2].y, corners[2].z)
        p3 = Point(corners[3].x, corners[3].y, corners[3].z)
        
        # Set normal and color
        normal = Vector(0, 1, 0)
        for p in [p0, p1, p2, p3]:
            p.normal = normal
            p.color = Color(100, 100, 100, 255)
        
        # Render floor triangles
        self.draw_triangle(p0, p1, p2, target_image=target_image, target_zbuffer=target_zbuffer)
        self.draw_triangle(p0, p2, p3, target_image=target_image, target_zbuffer=target_zbuffer)
        
        # Add grid lines for better motion reference
        grid_spacing = 5
        for i in range(-size, size + 1, grid_spacing):
            # X grid lines
            start_x = Vector(i, height + 0.01, -size - 10)  # Slightly above floor
            end_x = Vector(i, height + 0.01, size - 10)
            
            # Project to screen
            start_screen_x, start_screen_y = self.perspective_projection(start_x.x, start_x.y, start_x.z)
            end_screen_x, end_screen_y = self.perspective_projection(end_x.x, end_x.y, end_x.z)
            
            if start_screen_x >= 0 and end_screen_x >= 0:
                # Create points for the line
                p_start = Point(start_screen_x, start_screen_y, start_x.z, Color(50, 50, 50, 255))
                p_end = Point(end_screen_x, end_screen_y, end_x.z, Color(50, 50, 50, 255))
                
                # Draw line using Bresenham's algorithm
                self.draw_line(p_start, p_end, target_image=target_image, target_zbuffer=target_zbuffer)
            
            # Z grid lines
            start_z = Vector(-size, height + 0.01, -size - 10 + i)
            end_z = Vector(size, height + 0.01, -size - 10 + i)
            
            # Project to screen
            start_screen_x, start_screen_y = self.perspective_projection(start_z.x, start_z.y, start_z.z)
            end_screen_x, end_screen_y = self.perspective_projection(end_z.x, end_z.y, end_z.z)
            
            if start_screen_x >= 0 and end_screen_x >= 0:
                # Create points for the line
                p_start = Point(start_screen_x, start_screen_y, start_z.z, Color(50, 50, 50, 255))
                p_end = Point(end_screen_x, end_screen_y, end_z.z, Color(50, 50, 50, 255))
                
                # Draw line using Bresenham's algorithm
                self.draw_line(p_start, p_end, target_image=target_image, target_zbuffer=target_zbuffer)
                
    def draw_triangle(self, p0, p1, p2, target_image=None, target_zbuffer=None):
        """Draw a triangle with properly converted integer coordinates"""
        # Use provided targets or default to self.image and self.zBuffer
        target_image = target_image if target_image is not None else self.image
        target_zbuffer = target_zbuffer if target_zbuffer is not None else self.zBuffer
        
        # Create a Triangle instance
        tri = Triangle(p0, p1, p2)
        
        # Calculate bounding box with integer conversion
        ymin = max(min(p0.y, p1.y, p2.y), 0)
        ymax = min(max(p0.y, p1.y, p2.y), target_image.height - 1)
        
        # Convert to integers for range()
        ymin_int = int(ymin)
        ymax_int = int(ymax)
        
        # Iterate over scan lines
        for y in range(ymin_int, ymax_int + 1):
            x_values = []
            
            # Find intersections with edges
            for edge_start, edge_end in [(p0, p1), (p1, p2), (p2, p0)]:
                if (edge_start.y <= y <= edge_end.y) or (edge_end.y <= y <= edge_start.y):
                    if edge_end.y == edge_start.y:  # Skip horizontal edges
                        continue
                    
                    if edge_start.y == y:
                        x_values.append(edge_start.x)
                    elif edge_end.y == y:
                        x_values.append(edge_end.x)
                    else:
                        t = (y - edge_start.y) / (edge_end.y - edge_start.y)
                        x = edge_start.x + t * (edge_end.x - edge_start.x)
                        x_values.append(x)
            
            if len(x_values) > 0:
                # Sort and convert to integers
                if len(x_values) == 1:
                    x_start = x_values[0]
                    x_end = x_start
                else:
                    x_start, x_end = sorted(x_values)[:2]
                
                x_start_int = max(int(x_start), 0)
                x_end_int = min(int(x_end), target_image.width - 1)
                
                # Draw horizontal span
                for x in range(x_start_int, x_end_int + 1):
                    point = Point(x, y, color=None)
                    in_triangle, color, z_value = tri.contains_point(point)
                    
                    if in_triangle:
                        # Perform z-buffer check
                        buffer_index = y * target_image.width + x
                        if buffer_index < len(target_zbuffer) and target_zbuffer[buffer_index] < z_value:
                            target_zbuffer[buffer_index] = z_value
                            target_image.setPixel(x, y, color)
    
    def render_model(self, model_obj, target_image=None, target_zbuffer=None):
        """Render a 3D model with lighting"""
        # Use provided targets or default to self.image and self.zBuffer
        target_image = target_image if target_image is not None else self.image
        target_zbuffer = target_zbuffer if target_zbuffer is not None else self.zBuffer
        
        # Get the actual model (handle both direct models and ColoredModel objects)
        model = getattr(model_obj, 'model', model_obj)
        
        # Precalculate transformed vertices
        transformed_vertices = []
        for i in range(len(model.vertices)):
            vertex = model.getTransformedVertex(i)
            transformed_vertices.append(vertex)
        
        # Calculate face normals and vertex normals
        face_normals = {}
        for face in model.faces:
            v0 = transformed_vertices[face[0]]
            v1 = transformed_vertices[face[1]]
            v2 = transformed_vertices[face[2]]
            
            edge1 = Vector(v1.x - v0.x, v1.y - v0.y, v1.z - v0.z)
            edge2 = Vector(v2.x - v0.x, v2.y - v0.y, v2.z - v0.z)
            normal = edge1.cross(edge2).normalize()
            
            for i in face:
                if i not in face_normals:
                    face_normals[i] = []
                face_normals[i].append(normal)
        
        vertex_normals = []
        for vert_idx in range(len(model.vertices)):
            if vert_idx in face_normals:
                normal = Vector(0, 0, 0)
                for face_normal in face_normals[vert_idx]:
                    normal = normal + face_normal
                vertex_normals.append(normal.normalize())
            else:
                vertex_normals.append(Vector(0, 1, 0))
        
        # Get model color
        if hasattr(model_obj, 'diffuse_color'):
            model_color = model_obj.diffuse_color
        elif hasattr(model, 'diffuse_color'):
            model_color = model.diffuse_color
        else:
            model_color = (200, 200, 200)
        
        # Render faces
        for face in model.faces:
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
            
            # Create triangle points
            triangle_points = []
            for v, n in zip([v0, v1, v2], [n0, n1, n2]):
                screen_x, screen_y = self.perspective_projection(v.x, v.y, v.z)
                
                if screen_x < 0 or screen_y < 0 or screen_x >= self.width or screen_y >= self.height:
                    continue
                
                # Calculate lighting
                intensity = max(0.2, n * self.light_dir)
                
                r, g, b = model_color
                color = Color(
                    int(r * intensity),
                    int(g * intensity),
                    int(b * intensity),
                    255
                )
                
                point = Point(int(screen_x), int(screen_y), v.z, color)
                point.normal = n
                triangle_points.append(point)
            
            # Render triangle if all points are valid
            if len(triangle_points) == 3:
                self.draw_triangle(triangle_points[0], triangle_points[1], triangle_points[2], 
                                  target_image=target_image, target_zbuffer=target_zbuffer)
    
    def render_sphere(self, position, radius, color=(255, 255, 255), target_image=None, target_zbuffer=None):
        """Render a simple sphere at the given position"""
        # Use provided targets or default to self.image and self.zBuffer
        target_image = target_image if target_image is not None else self.image
        target_zbuffer = target_zbuffer if target_zbuffer is not None else self.zBuffer
        
        # Project sphere center to screen
        screen_x, screen_y = self.perspective_projection(position.x, position.y, position.z)
        
        # Skip if off-screen
        if screen_x < 0 or screen_y < 0 or screen_x >= self.width or screen_y >= self.height:
            return
        
        # Calculate projected radius (rough approximation)
        distance = (position - self.camera_pos).length()
        screen_radius = int(radius * 200 / distance)  # Simple projection
        
        # Skip if too small
        if screen_radius < 1:
            return
        
        # Draw sphere as a filled circle
        for y in range(max(0, screen_y - screen_radius), min(self.height, screen_y + screen_radius + 1)):
            for x in range(max(0, screen_x - screen_radius), min(self.width, screen_x + screen_radius + 1)):
                dx = x - screen_x
                dy = y - screen_y
                dist_sq = dx*dx + dy*dy
                
                if dist_sq <= screen_radius * screen_radius:
                    # Calculate z using sphere equation
                    t = 1.0 - dist_sq / (screen_radius * screen_radius)
                    z = position.z - radius + 2 * radius * t
                    
                    # Perform z-buffer check
                    buffer_index = y * self.width + x
                    if buffer_index < len(target_zbuffer) and target_zbuffer[buffer_index] < z:
                        target_zbuffer[buffer_index] = z
                        
                        # Simple lighting
                        intensity = 0.5 + 0.5 * t  # Higher in center
                        r, g, b = color
                        lit_color = Color(
                            int(r * intensity),
                            int(g * intensity),
                            int(b * intensity),
                            255
                        )
                        target_image.setPixel(x, y, lit_color)
    
    def render_scene(self):
        """Render the scene with the flying headsets"""
        # Clear image and z-buffer
        self.image = Image(self.width, self.height, Color(20, 20, 40, 255))
        self.zBuffer = [-float('inf')] * self.width * self.height
        
        # Create a second buffer for split view comparison
        if self.split_view_mode:
            self.image_no_blur = Image(self.width, self.height, Color(20, 20, 40, 255))
            self.zBuffer_no_blur = [-float('inf')] * self.width * self.height
        
        # Render floor
        if hasattr(self, 'floor_object') and self.floor_object:
            self.render_model(self.floor_object)
            if self.split_view_mode:
                self.render_model(self.floor_object, target_image=self.image_no_blur, target_zbuffer=self.zBuffer_no_blur)
        else:
            self.render_floor()
            if self.split_view_mode:
                self.render_floor(target_image=self.image_no_blur, target_zbuffer=self.zBuffer_no_blur)
        
        # Render reference markers (stationary objects)
        for marker_pos, marker_radius in self.reference_markers:
            # Draw simple spheres as markers
            self.render_sphere(marker_pos, marker_radius, color=(200, 200, 200))
            if self.split_view_mode:
                self.render_sphere(marker_pos, marker_radius, color=(200, 200, 200),
                                 target_image=self.image_no_blur, target_zbuffer=self.zBuffer_no_blur)
        
        # Render the flying headsets
        for headset in self.flying_headsets:
            self.render_model(headset.model)
            if self.split_view_mode:
                self.render_model(headset.model, target_image=self.image_no_blur, target_zbuffer=self.zBuffer_no_blur)
        
        # Apply motion blur for the main image with EXTREME settings
        if self.blur_enabled:
            # Custom extreme blur for dramatic effect
            from motion_blur import MotionBlurEffect
            
            # Use a local instance with even higher settings for this frame
            extreme_blur = MotionBlurEffect(blur_strength=self.motion_blur.blur_strength)
            extreme_blur.velocity_scale = 2.0  # Double velocity scale (default is 1.0)
            extreme_blur.max_samples = 16  # Use more samples (default is usually 4-8)
            
            blurred_image = extreme_blur.per_object_velocity_blur(
                self.image,
                self.flying_headsets,  # All headsets
                self.width,
                self.height,
                self.perspective_projection
            )
        else:
            blurred_image = self.image
        
        # For split view, we use half of each image
        if self.split_view_mode:
            # Create combined image with a dividing line
            final_image = Image(self.width, self.height, Color(20, 20, 40, 255))
            
            # Left half: With motion blur
            for y in range(self.height):
                for x in range(self.width // 2):
                    idx_blur = (blurred_image.height - y - 1) * blurred_image.width * 4 + x * 4 + (blurred_image.height - y - 1) + 1
                    idx_final = (final_image.height - y - 1) * final_image.width * 4 + x * 4 + (final_image.height - y - 1) + 1
                    
                    if idx_blur + 2 < len(blurred_image.buffer) and idx_final + 2 < len(final_image.buffer):
                        final_image.buffer[idx_final] = blurred_image.buffer[idx_blur]
                        final_image.buffer[idx_final + 1] = blurred_image.buffer[idx_blur + 1]
                        final_image.buffer[idx_final + 2] = blurred_image.buffer[idx_blur + 2]
                        final_image.buffer[idx_final + 3] = blurred_image.buffer[idx_blur + 3]
            
            # Right half: No motion blur
            for y in range(self.height):
                for x in range(self.width // 2, self.width):
                    idx_no_blur = (self.image_no_blur.height - y - 1) * self.image_no_blur.width * 4 + x * 4 + (self.image_no_blur.height - y - 1) + 1
                    idx_final = (final_image.height - y - 1) * final_image.width * 4 + x * 4 + (final_image.height - y - 1) + 1
                    
                    if idx_no_blur + 2 < len(self.image_no_blur.buffer) and idx_final + 2 < len(final_image.buffer):
                        final_image.buffer[idx_final] = self.image_no_blur.buffer[idx_no_blur]
                        final_image.buffer[idx_final + 1] = self.image_no_blur.buffer[idx_no_blur + 1]
                        final_image.buffer[idx_final + 2] = self.image_no_blur.buffer[idx_no_blur + 2]
                        final_image.buffer[idx_final + 3] = self.image_no_blur.buffer[idx_no_blur + 3]
        else:
            # Use full blurred image
            final_image = blurred_image
        
        # Convert image to pygame surface
        for y in range(self.height):
            for x in range(self.width):
                idx = (final_image.height - y - 1) * final_image.width * 4 + x * 4 + (final_image.height - y - 1) + 1
                if idx + 2 < len(final_image.buffer):
                    r = final_image.buffer[idx]
                    g = final_image.buffer[idx + 1]
                    b = final_image.buffer[idx + 2]
                    self.screen.set_at((x, y), (r, g, b))
        
        # Draw a dividing line for split view
        if self.split_view_mode:
            pygame.draw.line(self.screen, (255, 255, 255), (self.width // 2, 0), (self.width // 2, self.height), 2)
        
        # Draw information overlay
        self.draw_overlay()
        
        # Update display
        pygame.display.flip()
    
    def draw_line(self, p0, p1, target_image=None, target_zbuffer=None):
        """Draw a simple line using Bresenham's algorithm"""
        # Use provided targets or default to self.image and self.zBuffer
        target_image = target_image if target_image is not None else self.image
        target_zbuffer = target_zbuffer if target_zbuffer is not None else self.zBuffer
        
        # Extract coordinates
        x0, y0 = int(p0.x), int(p0.y)
        x1, y1 = int(p1.x), int(p1.y)
        
        # Get colors
        color0 = p0.color
        color1 = p1.color
        
        # Calculate delta values
        dx = abs(x1 - x0)
        dy = abs(y1 - y0)
        
        # Determine step direction
        sx = 1 if x0 < x1 else -1
        sy = 1 if y0 < y1 else -1
        
        # Initialize error
        err = dx - dy
        
        # Bresenham's line algorithm
        while True:
            # Draw pixel if in bounds
            if 0 <= x0 < target_image.width and 0 <= y0 < target_image.height:
                # Linear interpolation for z and color
                if dx > 0 or dy > 0:
                    t = abs((x0 - int(p0.x)) / dx if dx > dy else (y0 - int(p0.y)) / dy) if (dx > 0 or dy > 0) else 0
                else:
                    t = 0
                    
                # Interpolate z value
                z = p0.z * (1 - t) + p1.z * t
                
                # Z-buffer check
                buffer_index = y0 * target_image.width + x0
                if buffer_index < len(target_zbuffer) and target_zbuffer[buffer_index] < z:
                    target_zbuffer[buffer_index] = z
                    
                    # Interpolate color
                    r = int(color0.r() * (1 - t) + color1.r() * t)
                    g = int(color0.g() * (1 - t) + color1.g() * t)
                    b = int(color0.b() * (1 - t) + color1.b() * t)
                    a = int(color0.a() * (1 - t) + color1.a() * t)
                    
                    target_image.setPixel(x0, y0, Color(r, g, b, a))
            
            # Exit if we've reached the end point
            if x0 == x1 and y0 == y1:
                break
                
            # Update error and position
            e2 = 2 * err
            if e2 > -dy:
                err -= dy
                x0 += sx
            if e2 < dx:
                err += dx
                y0 += sy
    
    def draw_overlay(self):
        """Draw information overlay on the screen"""
        if self.split_view_mode:
            # Draw labels for split view
            left_label = self.font.render("WITH MOTION BLUR", True, (255, 255, 255))
            right_label = self.font.render("NO MOTION BLUR", True, (255, 255, 255))
            
            # Position labels in each half
            left_x = (self.width // 4) - (left_label.get_width() // 2)
            right_x = (self.width // 4 * 3) - (right_label.get_width() // 2)
            
            self.screen.blit(left_label, (left_x, 10))
            self.screen.blit(right_label, (right_x, 10))
        else:
            # Display blur status
            blur_text = self.font.render(
                f"Motion Blur: {'ON' if self.blur_enabled else 'OFF'}", 
                True, (255, 255, 255)
            )
            self.screen.blit(blur_text, (10, 10))
        
        # Display frame count
        frame_text = self.font.render(f"Frame: {self.frame_count}", True, (255, 255, 255))
        self.screen.blit(frame_text, (10, 40))
        
        # Display headset speeds
        y_pos = 70
        for i, headset in enumerate(self.flying_headsets):
            speed = headset.velocity.length()
            color_name = "Red" if i == 0 else "Green" if i == 1 else "Blue"
            speed_text = self.font.render(f"{color_name} Headset Speed: {speed:.1f}", True, (255, 255, 255))
            self.screen.blit(speed_text, (10, y_pos))
            y_pos += 25
        
        # Display controls
        controls_text = self.font.render(
            "B: Toggle blur | S: Toggle split view | R: Reset | SPACE: Auto-toggle | ESC: Quit", 
            True, (200, 200, 200)
        )
        self.screen.blit(controls_text, (10, self.height - 30))
    
    def handle_events(self):
        """Handle user input events"""
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return False
            
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    return False
                
                elif event.key == pygame.K_b:
                    # Toggle motion blur
                    self.blur_enabled = not self.blur_enabled
                    print(f"Motion blur: {'ON' if self.blur_enabled else 'OFF'}")
                
                elif event.key == pygame.K_s:
                    # Toggle split view mode
                    self.split_view_mode = not self.split_view_mode
                    print(f"Split view mode: {'ON' if self.split_view_mode else 'OFF'}")
                
                elif event.key == pygame.K_r:
                    # Reset demo
                    self.setup_scene()
                    self.frame_count = 0
                    print("Demo reset")
                
                elif event.key == pygame.K_SPACE:
                    # Toggle automatic blur switching
                    self.auto_toggle = not self.auto_toggle if hasattr(self, 'auto_toggle') else True
                    print(f"Auto toggle: {'ON' if hasattr(self, 'auto_toggle') and self.auto_toggle else 'OFF'}")
                
                elif event.key == pygame.K_UP:
                    # Increase blur strength
                    self.motion_blur.blur_strength = min(5.0, self.motion_blur.blur_strength + 0.5)
                    print(f"Blur strength increased to {self.motion_blur.blur_strength}")
                
                elif event.key == pygame.K_DOWN:
                    # Decrease blur strength
                    self.motion_blur.blur_strength = max(0.5, self.motion_blur.blur_strength - 0.5)
                    print(f"Blur strength decreased to {self.motion_blur.blur_strength}")
                
                elif event.key == pygame.K_1:
                    # Speed up headsets
                    for headset in self.flying_headsets:
                        headset.velocity.x *= 1.25
                    print("Headset speeds increased by 25%")
                
                elif event.key == pygame.K_2:
                    # Slow down headsets
                    for headset in self.flying_headsets:
                        headset.velocity.x *= 0.8
                    print("Headset speeds decreased by 20%")
        
        return True
    
    def update_headsets(self, dt):
        """Update all flying headsets' positions and orientations"""
        # Update each headset
        for i, headset in enumerate(self.flying_headsets):
            # Update position based on velocity
            headset.update(dt)
            
            # Add vertical oscillation (different for each headset)
            amplitude = 3.0 - i * 0.5  # Less amplitude for slower headsets
            frequency = 0.02 + i * 0.005  # Slower oscillation (reduced from 0.03 to 0.02)
            headset.position.y = (15 - i * 5) + amplitude * math.sin(self.frame_count * frequency + i * math.pi/3)
            
            # Add some Z oscillation too
            z_amplitude = 1.5 - i * 0.3
            z_frequency = 0.025 + i * 0.005  # Slower oscillation (reduced from 0.05 to 0.025)
            base_z = -35 + i * 5  # Starting z position
            headset.position.z = base_z + z_amplitude * math.sin(self.frame_count * z_frequency + i * math.pi/3)
            
            # Update model position
            headset.model.model.setPosition(
                headset.position.x, 
                headset.position.y, 
                headset.position.z
            )
            
            # Add rotation for visual interest (slower rotation)
            rotation_speed = 0.03  # Reduced from 0.05 to 0.03
            headset.model.model.setRotation(
                self.frame_count * rotation_speed,
                self.frame_count * rotation_speed * 0.7,
                self.frame_count * rotation_speed * 0.5
            )
            
            # If headset reaches right side, reset to left
            if headset.position.x > 40:  # Increased from 30 to 40
                headset.position.x = -40  # Increased from -30 to -40
                headset.model.model.setPosition(
                    headset.position.x, 
                    headset.position.y, 
                    headset.position.z
                )
    
    def run_demo(self, frames=360, toggle_interval=90):
        """
        Run the motion blur demo for the specified number of frames.
        
        Args:
            frames: Total number of frames to render
            toggle_interval: Number of frames between blur toggles
        """
        print(f"Starting Enhanced Motion Blur Headset Demo - {frames} frames")
        print("Controls:")
        print("  B: Toggle motion blur on/off")
        print("  S: Toggle split view (left: with blur, right: without blur)")
        print("  SPACE: Toggle automatic blur switching")
        print("  UP/DOWN: Increase/decrease blur strength")
        print("  1/2: Increase/decrease headset speeds")
        print("  R: Reset demo")
        print("  ESC: Quit")
        print(f"Motion blur will toggle every {toggle_interval} frames if auto-toggle is enabled")
        
        # Start video recording - even slower FPS for better visibility
        self.video_recorder = VideoRecorder(self.width, self.height, fps=10)  # Reduced from 15 to 10 fps
        self.video_recorder.start_recording()
        
        # Main demo loop
        running = True
        clock = pygame.time.Clock()
        self.auto_toggle = True
        
        # Save all frames option - can be expensive on disk space
        save_all_frames = False
        save_interval = 5  # Save every 5th frame to show more gradual progression
        
        while running and self.frame_count < frames:
            # Limit to 10 FPS instead of 15 for even slower motion
            target_fps = 10
            dt = clock.tick(target_fps) / 1000.0
            
            # Further slow down the physics for better visibility (reduce effective dt)
            physics_slowdown = 0.3  # 30% speed physics (reduced from 0.5)
            effective_dt = dt * physics_slowdown
            
            # Handle events
            running = self.handle_events()
            if not running:
                break
            
            # Update flying headsets with slower physics
            self.update_headsets(effective_dt)
            
            # Auto-toggle blur if enabled (only in non-split view mode)
            if hasattr(self, 'auto_toggle') and self.auto_toggle and not self.split_view_mode and self.frame_count % toggle_interval == 0:
                self.blur_enabled = not self.blur_enabled
                print(f"Frame {self.frame_count}: Motion blur {'enabled' if self.blur_enabled else 'disabled'}")
            
            # Render scene
            self.render_scene()
            
            # Capture frame for video
            self.video_recorder.capture_frame(self.screen)
            
            # Save individual frames (either all or at intervals)
            if save_all_frames or self.frame_count % save_interval == 0:
                frame_path = os.path.join(self.output_dir, f"frame_{self.frame_count:04d}.png")
                pygame.image.save(self.screen, frame_path)
            
            # Increment frame counter
            self.frame_count += 1
        
        # Stop recording and save video
        video_path = self.video_recorder.save_video(os.path.join(self.output_dir, "motion_blur_demo.mp4"))
        print(f"Demo completed. Video saved to: {video_path}")
        if save_all_frames:
            print(f"All frames saved to: {self.output_dir}")
        else:
            print(f"Selected frames saved to: {self.output_dir} (every {save_interval}th frame)")
        
        # Try to generate higher quality video with ffmpeg - with even slower playback
        try:
            ffmpeg_path = self.video_recorder.generate_ffmpeg_video(
                output_path=os.path.join(self.output_dir, "motion_blur_demo_slow.mp4"),
                quality="high",
                extra_args=["-filter:v", "setpts=2.5*PTS"]  # 2.5x slower playback (increased from 2.0)
            )
            if ffmpeg_path:
                print(f"Slow-motion high quality video saved to: {ffmpeg_path}")
                
            # Create an extra-slow version for frame-by-frame analysis
            ffmpeg_path = self.video_recorder.generate_ffmpeg_video(
                output_path=os.path.join(self.output_dir, "motion_blur_demo_very_slow.mp4"),
                quality="high",
                extra_args=["-filter:v", "setpts=5.0*PTS"]  # 5x slower playback
            )
            if ffmpeg_path:
                print(f"Very slow-motion video saved to: {ffmpeg_path}")
        except Exception as e:
            print(f"Failed to generate slow-motion video: {e}")
        
        # Clean up
        pygame.quit()


if __name__ == "__main__":
    demo = MotionBlurHeadsetDemo(width=1280, height=720, output_dir="enhanced_motion_blur_demo")
    demo.run_demo(frames=360, toggle_interval=90)
    # pygame.quit() is already called in run_demo

# if __name__ == "__main__":
#     demo = MotionBlurHeadsetDemo(width=1280, height=720, output_dir="enhanced_motion_blur_demo")
#     demo.run_demo(frames=360, toggle_interval=90)
#     # pygame.quit() is already called in run_demo

# if __name__ == "__main__":
#     demo = MotionBlurHeadsetDemo(width=1024, height=768, output_dir="enhanced_motion_blur_demo")
#     demo.run_demo(frames=240, toggle_interval=60)
#     pygame.quit()

# if __name__ == "__main__":
#     simulation = HeadsetSimulation()
    # simulation.run()
