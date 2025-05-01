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
class DepthOfFieldEffect:
    def __init__(self, focal_distance=15.0, focal_range=5.0, blur_strength=1.0):
        """
        focal_distance: Distance at which objects are perfectly in focus
        focal_range: Range around focal distance where objects remain relatively sharp
        blur_strength: Overall intensity of the blur effect
        """
        self.focal_distance = focal_distance
        self.focal_range = focal_range
        self.blur_strength = blur_strength
        self.enabled = True
    
    def process(self, image, z_buffer, width, height):
        """Apply depth of field effect to left half of screen only"""
        result = Image(width, height, Color(0, 0, 0, 255))
        
        # Create blur map based on depth
        blur_map = self._create_blur_map(z_buffer, width, height)
        
        # Process entire image
        for y in range(height):
            for x in range(width):
                # Left half: apply DoF
                if x < width // 2:
                    blur_radius = blur_map[y * width + x]
                    if blur_radius <= 0.5:
                        self._copy_pixel(image, result, x, y)
                    else:
                        self._apply_blur(image, result, x, y, int(blur_radius))
                # Right half: direct copy (no DoF)
                else:
                    self._copy_pixel(image, result, x, y)
        
        # Draw dividing line
        for y in range(height):
            idx = self._get_pixel_index(result, width // 2, y)
            if idx + 2 < len(result.buffer):
                result.buffer[idx] = 255    # White
                result.buffer[idx + 1] = 255
                result.buffer[idx + 2] = 255
        
        return result
    
    def _create_blur_map(self, z_buffer, width, height):
        """Calculate blur amount for each pixel based on Z-depth"""
        blur_map = [0] * (width * height)
        
        for i in range(len(z_buffer)):
            depth = z_buffer[i]
            if depth == -float('inf'):
                blur_map[i] = self.blur_strength  # Background blur
            else:
                # Calculate blur based on distance from focal plane
                distance_from_focal = abs(depth - self.focal_distance)
                if distance_from_focal <= self.focal_range:
                    # In focus region
                    blur_factor = distance_from_focal / self.focal_range
                    blur_map[i] = blur_factor * self.blur_strength * 0.5
                else:
                    # Out of focus region
                    blur_factor = min(1.0, (distance_from_focal - self.focal_range) / self.focal_range)
                    blur_map[i] = self.blur_strength * (0.5 + blur_factor)
        
        return blur_map
    
    def _copy_pixel(self, source, dest, x, y):
        """Copy a pixel from source to destination image"""
        idx = self._get_pixel_index(source, x, y)
        if idx + 3 < len(source.buffer):
            r = source.buffer[idx]
            g = source.buffer[idx + 1]
            b = source.buffer[idx + 2]
            a = source.buffer[idx + 3]
            dest.setPixel(x, y, Color(r, g, b, a))
    
    def _apply_blur(self, source, dest, x, y, blur_radius):
        """Apply a simple box blur with the specified radius"""
        width, height = source.width, source.height
        r_sum, g_sum, b_sum = 0, 0, 0
        count = 0
        
        # Define blur range
        radius = max(1, min(5, int(blur_radius)))
        
        # Sample pixels in a square around the center
        for by in range(max(0, y - radius), min(height, y + radius + 1)):
            for bx in range(max(0, x - radius), min(width, x + radius + 1)):
                idx = self._get_pixel_index(source, bx, by)
                if idx + 2 < len(source.buffer):
                    r_sum += source.buffer[idx]
                    g_sum += source.buffer[idx + 1]
                    b_sum += source.buffer[idx + 2]
                    count += 1
        
        # Calculate average
        if count > 0:
            r = int(r_sum / count)
            g = int(g_sum / count)
            b = int(b_sum / count)
            
            # Get alpha from original pixel
            idx = self._get_pixel_index(source, x, y)
            a = source.buffer[idx + 3] if idx + 3 < len(source.buffer) else 255
            
            # Set blurred pixel
            dest.setPixel(x, y, Color(r, g, b, a))
    
    def _get_pixel_index(self, image, x, y):
        """Calculate index in the image buffer for a pixel"""
        flipY = (image.height - y - 1)
        index = (flipY * image.width + x) * 4 + flipY + 1
        return index
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
        self.camera_target = Vector(0, 5, -20)
        self.light_dir = Vector(0.5, -1, -0.5).normalize()
        
        # Motion blur
        self.motion_blur = MotionBlurEffect(blur_strength=0.8)
        self.blur_enabled = True
        
        # Video recorder
        self.video_recorder = VideoRecorder(width, height, fps=30)
        self.is_recording = False

        self.depth_of_field = DepthOfFieldEffect(focal_distance=5.0, focal_range=1.5, blur_strength=3.0)
        self.dof_enabled = True
        
        # Load IMU data
        self.load_imu_data()
        
        # Setup scene objects
        self.main_headset = None
        self.floor_headsets = []
        self.setup_scene()
        
        # Physics settings - Reduced friction significantly for longer movement
        self.friction_coefficient = 0.95  # Changed from 0.98 to 0.995 (much less friction)
        self.accumulator = 0

        # Target frames for 27 seconds at 30fps
        self.target_frames = 810
        
        # Display
        self.font = pygame.font.SysFont('Arial', 18)
        self.frame_count = 0
        self.fps_history = []
        self.paused = False
        self.imu_progress = 0

    def draw_focal_plane(self):
        """Draw a visual indicator of the current focal plane"""
        # Calculate z-depth in world space
        focal_z = -self.depth_of_field.focal_distance
        
        # Define corners of a rectangle at the focal plane
        rect_size = 20
        corners = [
            Vector(-rect_size, 0, focal_z),
            Vector(rect_size, 0, focal_z),
            Vector(rect_size, rect_size*2, focal_z),
            Vector(-rect_size, rect_size*2, focal_z)
        ]
        
        # Project corners to screen space
        screen_points = []
        for point in corners:
            screen_x, screen_y = self.perspective_projection(point.x, point.y, point.z)
            if screen_x >= 0 and screen_y >= 0:
                screen_points.append((screen_x, screen_y))
        
        # Draw outline if we have enough points
        if len(screen_points) >= 4:
            pygame.draw.lines(
                self.screen,
                (0, 255, 255),  # Cyan color
                True,  # Closed shape
                screen_points,
                2  # Line width
            )

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

        if self.dof_enabled:
            self.image = self.depth_of_field.process(self.image, self.zBuffer, self.width, self.height)
    
        
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
        self.draw_focal_plane()
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
        

        dof_text = self.font.render(
            f"Depth of Field: {'ON' if self.dof_enabled else 'OFF'} (Focal: {self.depth_of_field.focal_distance:.1f}, Range: {self.depth_of_field.focal_range:.1f})",
            True, (255, 255, 255)
        )
        self.screen.blit(dof_text, (10, 35))
        
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

if __name__ == "__main__":
    simulation = HeadsetSimulation()
    simulation.run()