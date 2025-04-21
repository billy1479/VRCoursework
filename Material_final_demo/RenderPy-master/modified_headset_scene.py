import pygame
from image import Image, Color
from model import Model, DeadReckoningFilter, SensorDataParser, CollisionObject
from vector import Vector
from shape import Triangle, Point
import math
import time
import os

# Import the VideoRecorder from the separate file
from video_recorder import VideoRecorder
class FixedCameraHeadsetScene:
    """
    A 3D scene with multiple VR headsets:
    - One rotating in the center based on sensor data
    - Multiple headsets sliding on the floor with friction and collisions
    - Camera maintains a fixed position to view all the action
    - Recording capability using VideoRecorder
    """
    def __init__(self, width=800, height=600, csv_path="IMUData.csv", auto_record=True):
        # Initialize pygame
        pygame.init()
        self.width = width
        self.height = height
        self.screen = pygame.display.set_mode((width, height))
        pygame.display.set_caption("VR Headset Physics Scene - Fixed Camera")
        
        # Create buffer surface for rendering
        self.buffer_surface = pygame.Surface((width, height))
        
        # Image and Z-buffer for rendering
        self.image = Image(width, height, Color(20, 20, 40, 255))  # Dark blue background
        self.zBuffer = [-float('inf')] * width * height
        
        # Fixed camera position with good view of entire scene
        self.camera_pos = Vector(0, 20, -50)  # Higher and further back for better view
        self.camera_target = Vector(0, 5, -15)  # Look at central area
        self.light_dir = Vector(0.5, -1, -0.5).normalize()

        # IMU playback speed multiplier
        self.imu_playback_speed = 5  # Process 5 samples per frame
        
        # Frame counter
        self.frame_count = 0
        self.fps_history = []
        
        # Motion blur effect
        from extreme_motion_blur import ExtremeMotionBlur
        self.motion_blur = ExtremeMotionBlur(blur_strength=2.0)
        self.blur_enabled = True
        
        # Load sensor data for rotating headset
        self.csv_path = csv_path
        self.load_sensor_data()
        
        # Create scene objects
        self.central_headset = None
        self.floor_headsets = []
        self.setup_scene()
        
        # Physics settings
        self.friction_coefficient = 0.96  # Higher = less friction
        self.accumulator = 0  # For fixed timestep physics
        
        # Debug and control flags
        self.show_debug = True
        self.paused = False
        
        # Font for info display
        self.font = pygame.font.SysFont('Arial', 18)
        
        # Set up video recording
        self.video_recorder = VideoRecorder(width, height, fps=30)
        self.is_recording = False
        self.auto_record = auto_record  # Flag to control automatic recording
        self.recording_finished = False  # Flag to track if recording has been stopped

    def load_sensor_data(self):
        """Load and preprocess sensor data from CSV file"""
        try:
            parser = SensorDataParser(self.csv_path)
            self.sensor_data = parser.parse()
            print(f"Loaded {len(self.sensor_data)} sensor data points")
            
            # Create dead reckoning filter
            self.dr_filter = DeadReckoningFilter(alpha=0.98)
            # Calibrate using first 100 samples
            self.dr_filter.calibrate(self.sensor_data[:min(100, len(self.sensor_data))])
            
            self.current_data_index = 0
        except Exception as e:
            print(f"Error loading sensor data: {e}")
            print("Using fallback rotation pattern instead")
            self.sensor_data = None
            self.dr_filter = None

    def create_floor_headsets(self):
        """Create multiple headsets that slide on the floor with different colors"""
        import random
        headsets = []
        
        # Define boundary limits
        boundary = {
            'min_x': -28.0,
            'max_x': 28.0,
            'min_z': -38.0,
            'max_z': -2.0
        }
        
        # Colors for headsets
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
        
        # Create headsets in a circle formation
        num_circle = 8
        circle_radius = 15
        center_x = 0
        center_z = -15  # Position circle in front of the camera
        
        for i in range(num_circle):
            angle = (i / num_circle) * 2 * math.pi
            
            # Position in circle
            pos = Vector(
                center_x + circle_radius * math.cos(angle),
                1,  # Slightly above floor
                center_z + circle_radius * math.sin(angle)
            )
            
            # Ensure position is within boundaries
            pos.x = max(boundary['min_x'] + 2, min(boundary['max_x'] - 2, pos.x))
            pos.z = max(boundary['min_z'] + 2, min(boundary['max_z'] - 2, pos.z))
            
            # Velocity toward center
            speed = 3 + (i % 3)  # Different speeds
            vel = Vector(
                -math.cos(angle) * speed,
                0,
                -math.sin(angle) * speed
            )
            
            model = Model('data/headset.obj')
            model.normalizeGeometry()
            model.setPosition(pos.x, pos.y, pos.z)
            
            # Assign color
            model.diffuse_color = colors[i % len(colors)]
            
            # Create collision object
            headset = CollisionObject(model, pos, vel, radius=1.0)
            headsets.append(headset)
        
        # "Billiards break" pattern
        triangle_size = 3
        start_z = -5
        color_index = 0
        for row in range(triangle_size):
            for col in range(row + 1):
                pos = Vector(
                    (col - row/2) * 2.5,  # Wider spacing
                    1,
                    start_z + row * 2.5
                )
                
                # Small random initial velocity
                vel = Vector(
                    (random.random() - 0.5) * 0.2,
                    0,
                    (random.random() - 0.5) * 0.2
                )
                
                model = Model('data/headset.obj')
                model.normalizeGeometry()
                model.setPosition(pos.x, pos.y, pos.z)
                
                # Assign color
                model.diffuse_color = colors[color_index % len(colors)]
                color_index += 1
                
                headset = CollisionObject(model, pos, vel, radius=1.0)
                headsets.append(headset)
        
        # Add "cue ball" headsets from different directions
        cue_positions = [
            # From behind
            (0, 1, -25),
            # From sides
            (-20, 1, -15),
            (20, 1, -15)
        ]
        
        for i, (x, y, z) in enumerate(cue_positions):
            pos = Vector(x, y, z)
            
            # Calculate velocity toward center
            direction = Vector(0 - x, 0, -10 - z)
            length = math.sqrt(direction.x**2 + direction.z**2)
            if length > 0:
                direction.x /= length
                direction.z /= length
            
            # Set speed
            speed = 6  # Fast enough to create interesting collisions
            vel = Vector(direction.x * speed, 0, direction.z * speed)
            
            model = Model('data/headset.obj')
            model.normalizeGeometry()
            model.setPosition(pos.x, pos.y, pos.z)
            
            # White color for "cue ball"
            model.diffuse_color = (255, 255, 255)
            
            headsets.append(CollisionObject(model, pos, vel, radius=1.0))
        
        return headsets

    def update_floor_physics(self, dt):
        """Update physics for floor headsets with collisions and friction"""
        # Use a fixed time step for consistent physics
        fixed_dt = 1/60.0
        
        # Accumulate leftover time
        self.accumulator += dt
        
        # Define boundary walls
        boundary = {
            'min_x': -30.0,
            'max_x': 30.0,
            'min_z': -40.0,
            'max_z': 0.0,
            'bounce_factor': 0.8  # Energy retained after bouncing
        }
        
        # Run physics updates with fixed timestep
        while self.accumulator >= fixed_dt:
            # Clear collision records
            for headset in self.floor_headsets:
                headset.clear_collision_history()
            
            # Apply gravity
            for headset in self.floor_headsets:
                headset.velocity.y -= 9.81 * fixed_dt
            
            # Check collisions between headsets
            for i in range(len(self.floor_headsets)):
                for j in range(i + 1, len(self.floor_headsets)):
                    # Quick distance check
                    dx = self.floor_headsets[i].position.x - self.floor_headsets[j].position.x
                    dy = self.floor_headsets[i].position.y - self.floor_headsets[j].position.y
                    dz = self.floor_headsets[i].position.z - self.floor_headsets[j].position.z
                    dist_sq = dx*dx + dy*dy + dz*dz
                    
                    # Only check collision if objects are close enough
                    max_dist = self.floor_headsets[i].radius + self.floor_headsets[j].radius
                    if dist_sq < max_dist * max_dist * 1.5:
                        if self.floor_headsets[i].check_collision(self.floor_headsets[j]):
                            self.floor_headsets[i].resolve_collision(self.floor_headsets[j])
            
            # Apply floor constraints and friction
            for headset in self.floor_headsets:
                # Check if headset is on or near the floor
                is_on_floor = headset.position.y - headset.radius <= 0.01
                
                if is_on_floor:
                    # Ensure the headset doesn't go below the floor
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
                        
                        # Stop completely if very slow
                        if horizontal_speed_squared < 0.025:
                            headset.velocity.x = 0
                            headset.velocity.z = 0
                
                # Apply boundary constraints (invisible walls)
                # X-axis boundaries
                if headset.position.x - headset.radius < boundary['min_x']:
                    headset.position.x = boundary['min_x'] + headset.radius
                    headset.velocity.x = -headset.velocity.x * boundary['bounce_factor']
                elif headset.position.x + headset.radius > boundary['max_x']:
                    headset.position.x = boundary['max_x'] - headset.radius
                    headset.velocity.x = -headset.velocity.x * boundary['bounce_factor']
                
                # Z-axis boundaries
                if headset.position.z - headset.radius < boundary['min_z']:
                    headset.position.z = boundary['min_z'] + headset.radius
                    headset.velocity.z = -headset.velocity.z * boundary['bounce_factor']
                elif headset.position.z + headset.radius > boundary['max_z']:
                    headset.position.z = boundary['max_z'] - headset.radius
                    headset.velocity.z = -headset.velocity.z * boundary['bounce_factor']
            
            # Update positions
            for headset in self.floor_headsets:
                headset.update(fixed_dt)
            
            self.accumulator -= fixed_dt

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
        """Set up the scene with a central rotating headset and floor headsets"""
        # Create central headset - bigger and elevated for better visibility
        model = Model('data/headset.obj')
        model.normalizeGeometry()
        
        # Position it above the floor for better visibility
        position = Vector(0, 15, -25)  # Higher up and in front
        model.setPosition(position.x, position.y, position.z)
        
        # Make it larger
        model.scale = [0.5, 0.5, 0.5]
        model.updateTransform()
        
        # Add a distinctive color
        model.diffuse_color = (255, 215, 0)  # Gold color
        
        # Store the model
        self.central_headset = {
            "model": model,
            "rotation": [0, 0, 0],
            "position": position
        }
        
        # Create floor headsets
        self.floor_headsets = self.create_floor_headsets()
        
        # Load the floor object
        # floor_model = Model('data/floor.obj')
        try:
            floor_model = Model('data/floor.obj')
            print("Floor model loaded successfully")
        except Exception as e:
            print(f"Error loading floor model: {e}")
        floor_model.normalizeGeometry()
        floor_model.setPosition(0, 0, -20)  # Adjust position as needed
        floor_model.scale = [60.0, 1.0, 40.0]  # Adjust scale as needed
        floor_model.updateTransform()
        floor_model.diffuse_color = (120, 120, 160)  # Gray color for the floor
        
        # Store the floor model
        self.floor_model = floor_model

    def update_central_headset(self, dt):
        """Update the rotating central headset with IMU data"""
        if self.sensor_data and self.dr_filter:
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
                    
                    # Update filter and get orientation
                    _, orientation = self.dr_filter.update(sensor_data)
                    
                    # Convert quaternion to Euler angles for model rotation
                    roll, pitch, yaw = self.dr_filter.get_euler_angles()
                    
                    # Apply rotation to model
                    self.central_headset["model"].setRotation(roll, pitch, yaw)
                    self.central_headset["rotation"] = [roll, pitch, yaw]
                    
                    # Ensure position is updated from model's transform
                    model = self.central_headset["model"]
                    self.central_headset["position"] = Vector(
                        model.trans[0], 
                        model.trans[1], 
                        model.trans[2]
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
            # Fallback: simple rotation pattern
            self.central_headset["rotation"][0] += dt * 1.0  # Roll
            self.central_headset["rotation"][1] += dt * 1.5  # Pitch
            self.central_headset["rotation"][2] += dt * 0.8  # Yaw
            
            # Apply rotation to model
            model = self.central_headset["model"]
            model.setRotation(
                self.central_headset["rotation"][0],
                self.central_headset["rotation"][1],
                self.central_headset["rotation"][2]
            )
            
            # Update position from model's transform
            self.central_headset["position"] = Vector(
                model.trans[0], 
                model.trans[1], 
                model.trans[2]
            )
            
    def render_scene(self):
        """Render the scene with all headsets and the floor"""
        # Store previous positions for motion blur
        if self.blur_enabled:
            self.motion_blur.update_object_positions(self.floor_headsets)
        
        # Clear image and z-buffer for new frame
        self.image = Image(self.width, self.height, Color(20, 20, 40, 255))
        self.zBuffer = [-float('inf')] * self.width * self.height
        
        if self.floor_model:
            self.render_model(self.floor_model)
        
        # Render central rotating headset
        if self.central_headset:
            self.render_model(self.central_headset["model"])
        
        # Render floor headsets
        for headset in self.floor_headsets:
            self.render_model(headset.model)
        
        # Apply motion blur if enabled
        if self.blur_enabled:
            final_image = self.motion_blur.apply_blur(
                self.image, 
                self.floor_headsets, 
                self.width, 
                self.height, 
                self.perspective_projection
            )
        else:
            final_image = self.image
        
        # Update display
        self.update_display(final_image)
        
        # Render boundary walls only (not floor grid)
        self.render_floor_grid()
        
        # Capture frame for video if recording
        if self.is_recording:
            self.video_recorder.capture_frame(self.screen)
        
        # Draw debug info
        if self.show_debug:
            self.draw_debug_info()        
    
    def render_floor_model(self, floor_model):
        """
        Special rendering method for the floor to ensure it appears as a solid surface.
        """
        if floor_model is None:
            print("No floor model to render")
            return
            
        # Precalculate transformed vertices
        transformed_vertices = []
        for i in range(len(floor_model.vertices)):
            transformed_vertex = floor_model.getTransformedVertex(i)
            transformed_vertices.append(transformed_vertex)
        
        # Get model color (with ambient lighting for better visibility)
        floor_color = getattr(floor_model, 'diffuse_color', (120, 120, 160))
        ambient_factor = 0.7  # Higher ambient factor for better visibility
        base_color = Color(
            int(floor_color[0] * ambient_factor),
            int(floor_color[1] * ambient_factor),
            int(floor_color[2] * ambient_factor),
            255
        )
        
        # Render all faces of the floor
        for face in floor_model.faces:
            p0 = transformed_vertices[face[0]]
            p1 = transformed_vertices[face[1]]
            p2 = transformed_vertices[face[2]]
            
            # Skip backface culling for floor - we want to see it from all angles
            
            # Project vertices to screen coordinates
            screen_points = []
            for p in [p0, p1, p2]:
                screenX, screenY = self.perspective_projection(p.x, p.y, p.z)
                
                # Skip if offscreen
                if screenX < 0 or screenY < 0 or screenX >= self.width or screenY >= self.height:
                    continue
                
                # Add slight shading based on distance from center for visual interest
                dist_factor = 1.0 - min(1.0, (p.x**2 + p.z**2) / 1000.0) * 0.3
                color = Color(
                    int(base_color.r() * dist_factor),
                    int(base_color.g() * dist_factor),
                    int(base_color.b() * dist_factor),
                    255
                )
                
                # Create point with floor color
                from shape import Point
                point = Point(screenX, screenY, p.z, color)
                screen_points.append(point)
            
            # Draw the triangle if all points are valid
            if len(screen_points) == 3:
                from shape import Triangle
                Triangle(
                    screen_points[0],
                    screen_points[1],
                    screen_points[2]
                ).draw_faster(self.image, self.zBuffer)
    
    def render_model(self, model_obj):
        """Render a 3D model with lighting"""
        # Handle models with or without a 'model' attribute
        model = getattr(model_obj, 'model', model_obj)
        
        if hasattr(self, 'floor_model') and model == self.floor_model:
            self.render_floor_model(model)
            return

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

    def render_floor_grid(self):
        """Render boundary walls and debug info for the floor"""
        # Define boundary walls
        boundary = {
            'min_x': -30.0,
            'max_x': 30.0,
            'min_z': -40.0,
            'max_z': 0.0,
            'height': 5.0  # Height of walls
        }
        
        # Draw boundary walls (simple outlines)
        wall_color = (100, 100, 220)
        wall_points = [
            # Floor corners
            (boundary['min_x'], 0, boundary['min_z']),
            (boundary['max_x'], 0, boundary['min_z']),
            (boundary['max_x'], 0, boundary['max_z']),
            (boundary['min_x'], 0, boundary['max_z']),
            # Ceiling corners
            (boundary['min_x'], boundary['height'], boundary['min_z']),
            (boundary['max_x'], boundary['height'], boundary['min_z']),
            (boundary['max_x'], boundary['height'], boundary['max_z']),
            (boundary['min_x'], boundary['height'], boundary['max_z'])
        ]
        
        # Project all points
        screen_points = []
        for point in wall_points:
            screen_point = self.perspective_projection(point[0], point[1], point[2])
            if screen_point[0] >= 0 and screen_point[1] >= 0:
                screen_points.append(screen_point)
            else:
                screen_points.append(None)
        
        # Draw vertical edges
        for i in range(4):
            if screen_points[i] and screen_points[i+4]:
                pygame.draw.line(
                    self.screen,
                    wall_color,
                    screen_points[i],
                    screen_points[i+4],
                    1
                )
        
        # Draw floor edges
        for i in range(4):
            next_i = (i + 1) % 4
            if screen_points[i] and screen_points[next_i]:
                pygame.draw.line(
                    self.screen,
                    wall_color,
                    screen_points[i],
                    screen_points[next_i],
                    1
                )
        
        # Draw ceiling edges
        for i in range(4, 8):
            next_i = 4 + ((i + 1) % 4)
            if screen_points[i] and screen_points[next_i]:
                pygame.draw.line(
                    self.screen,
                    wall_color,
                    screen_points[i],
                    screen_points[next_i],
                    1
                )

    def update_display(self, image):
        """Update the display with current image buffer"""
        # Convert image buffer to pygame surface
        for y in range(self.height):
            for x in range(self.width):
                # Calculate buffer index
                flipY = (self.height - y - 1)
                index = (flipY * self.width + x) * 4 + flipY + 1  # +1 for null byte
                
                # Extract RGB values
                if index + 2 < len(image.buffer):
                    r = image.buffer[index]
                    g = image.buffer[index + 1]
                    b = image.buffer[index + 2]
                    
                    # Set pixel on screen
                    self.screen.set_at((x, y), (r, g, b))

    
    def draw_debug_info(self):
        """
        Draw debug information on screen including IMU dataset progress.
        Replace your existing draw_debug_info method with this enhanced version.
        """
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
            f"Motion Blur: {'ON' if self.blur_enabled else 'OFF'} (Strength: {self.motion_blur.blur_strength:.1f})",
            True, (255, 255, 255)
        )
        self.screen.blit(blur_text, (10, 35))
        
        # Display central headset rotation
        if self.central_headset:
            rot = self.central_headset["rotation"]
            rot_text = self.font.render(
                f"Rotation: Roll={math.degrees(rot[0]):.1f}°, Pitch={math.degrees(rot[1]):.1f}°, Yaw={math.degrees(rot[2]):.1f}°",
                True, (255, 255, 255)
            )
            self.screen.blit(rot_text, (10, 60))
        
        # Display IMU dataset progress
        if hasattr(self, 'sensor_data') and self.sensor_data:
            # Calculate progress as percentage
            progress_percent = (self.current_data_index / len(self.sensor_data)) * 100
            
            # Display textual progress info
            imu_text = self.font.render(
                f"IMU Data: {self.current_data_index}/{len(self.sensor_data)} ({progress_percent:.1f}%)",
                True, (255, 255, 255)
            )
            self.screen.blit(imu_text, (10, 85))
            
            # Draw progress bar
            progress_bar_width = 200
            progress_bar_height = 10
            x_pos = 10
            y_pos = 110
            
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
        
        # Display recording status
        if self.is_recording:
            rec_text = self.font.render(
                f"RECORDING [{len(self.video_recorder.frames)} frames]",
                True, (255, 0, 0)
            )
            self.screen.blit(rec_text, (self.width - 300, 10))
        
        # Display controls
        controls_text = self.font.render(
            "B: Toggle blur | +/-: Adjust blur | R: Reset | P: Pause/Play | V: Start/Stop Recording | ESC: Quit",
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
                
                elif event.key == pygame.K_b:
                    # Toggle motion blur
                    self.blur_enabled = not self.blur_enabled
                    print(f"Motion Blur: {'Enabled' if self.blur_enabled else 'Disabled'}")
                
                elif event.key == pygame.K_EQUALS or event.key == pygame.K_PLUS:
                    # Increase blur strength
                    self.motion_blur.blur_strength = min(5.0, self.motion_blur.blur_strength + 0.5)
                    print(f"Blur Strength: {self.motion_blur.blur_strength:.1f}")
                
                elif event.key == pygame.K_MINUS:
                    # Decrease blur strength
                    self.motion_blur.blur_strength = max(0.5, self.motion_blur.blur_strength - 0.5)
                    print(f"Blur Strength: {self.motion_blur.blur_strength:.1f}")
                
                elif event.key == pygame.K_r:
                    # Reset scene
                    self.floor_headsets = self.create_floor_headsets()
                    if self.sensor_data:
                        self.current_data_index = 0
                    print("Scene Reset")
                
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
                
                elif event.key == pygame.K_UP:
                    # Move camera up
                    self.camera_pos.y += 1
                    self.camera_target.y += 1
                    print(f"Camera Position: {self.camera_pos.x}, {self.camera_pos.y}, {self.camera_pos.z}")
                
                elif event.key == pygame.K_DOWN:
                    # Move camera down
                    self.camera_pos.y -= 1
                    self.camera_target.y -= 1
                    print(f"Camera Position: {self.camera_pos.x}, {self.camera_pos.y}, {self.camera_pos.z}")
                
                elif event.key == pygame.K_LEFT:
                    # Move camera left
                    self.camera_pos.x -= 1
                    self.camera_target.x -= 1
                    print(f"Camera Position: {self.camera_pos.x}, {self.camera_pos.y}, {self.camera_pos.z}")
                
                elif event.key == pygame.K_RIGHT:
                    # Move camera right
                    self.camera_pos.x += 1
                    self.camera_target.x += 1
                    print(f"Camera Position: {self.camera_pos.x}, {self.camera_pos.y}, {self.camera_pos.z}")
                
                elif event.key == pygame.K_w:
                    # Move camera forward
                    direction = (self.camera_target - self.camera_pos).normalize()
                    self.camera_pos = self.camera_pos + direction * 2
                    self.camera_target = self.camera_target + direction * 2
                    print(f"Camera Position: {self.camera_pos.x}, {self.camera_pos.y}, {self.camera_pos.z}")
                
                elif event.key == pygame.K_s:
                    # Move camera backward
                    direction = (self.camera_target - self.camera_pos).normalize()
                    self.camera_pos = self.camera_pos - direction * 2
                    self.camera_target = self.camera_target - direction * 2
                    print(f"Camera Position: {self.camera_pos.x}, {self.camera_pos.y}, {self.camera_pos.z}")
        
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
            file_path = self.video_recorder.save_video(filename="headset_simulation.mp4")
            if file_path:
                print(f"Video saved to {file_path}")
            else:
                print("Failed to save video with OpenCV, saving frames as images...")
                # Fall back to saving frames as images
                frames_dir = self.video_recorder.save_frames_as_images()
                print(f"Frames saved to {frames_dir}")
                print("You can convert these to a video using an external tool like FFmpeg")
        except Exception as e:
            print(f"Error saving video: {e}")
    
    def run(self):
        """Main loop to run the simulation"""
        clock = pygame.time.Clock()
        running = True
        
        print("VR Headset Physics Scene - Fixed Camera View with Recording")
        print("--------------------------------------------------------")
        print("Controls:")
        print("  B: Toggle motion blur")
        print("  +/-: Adjust blur strength")
        print("  R: Reset scene")
        print("  P: Pause/resume simulation")
        print("  D: Toggle debug info")
        print("  V: Start/stop video recording")
        print("  Arrow keys: Move camera position")
        print("  W/S: Move camera forward/backward")
        print("  ESC: Quit")
        
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
                # Update central headset rotation
                self.update_central_headset(dt)
                
                # Update floor headsets physics
                self.update_floor_physics(dt)
            
            # Render the scene
            self.render_scene()
            
            # Update display
            pygame.display.flip()
            
            # Increment frame counter
            self.frame_count += 1
        
        # Clean up
        pygame.quit()
        print(f"Simulation ended after {self.frame_count} frames")

def run_fixed_camera_scene_with_recording():
    """
    Start a headset scene with a fixed camera that shows both the rotating headset
    and the floor of colliding headsets. Includes video recording capability.
    """
    try:
        # Try to use a local path first
        scene = FixedCameraHeadsetScene(csv_path="IMUData.csv")
    except Exception as e:
        # If that fails, try using a relative path with parent directory
        try:
            scene = FixedCameraHeadsetScene(csv_path="../IMUData.csv")
        except Exception as e2:
            print(f"Error loading IMU data: {e2}")
            print("Continuing with fallback rotation pattern")
            scene = FixedCameraHeadsetScene(csv_path="")
    
    scene.run()

# Run this function to start the simulation
if __name__ == "__main__":
    run_fixed_camera_scene_with_recording()
    
