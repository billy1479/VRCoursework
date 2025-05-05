import random
import pygame
import os
import time
import math
import numpy as np
from image import Image, Color
from vector import Vector
from model import Model, SensorDataParser
from dead_reckoning_filter import DeadReckoningFilter
from collision import CollisionObject
from shape import Triangle, Point
from color_support import ColoredModel
from motion_blur import MotionBlurEffect
from video_recorder import VideoRecorder
from depth_of_field import DepthOfFieldEffect

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
        self.blur_enabled = False
        
        # Video recorder
        self.video_recorder = VideoRecorder(width, height, fps=30)
        self.is_recording = True

        self.depth_of_field = DepthOfFieldEffect(focal_distance=-25, focal_range=10, blur_strength=2.0)
        self.dof_enabled = True
        
        # Load IMU data
        self.load_imu_data()
        
        # Setup scene objects
        self.main_headset = None
        self.floor_headsets = []
        self.setup_scene()
        
        # Physics settings - Reduced friction significantly for longer movement
        self.friction_coefficient = 0.97
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
        
        # Create the floor object
        try:
            self.floor_object = self.create_floor_model()
            print("Created floor model successfully")
            
            # Debug output to check floor properties
            floor_model = self.floor_object.model
            print(f"Floor position: {floor_model.trans}")
            print(f"Floor scale: {floor_model.scale}")
            
            # Debug - test perspective projection of floor corners
            vertices = []
            for i in range(len(floor_model.vertices)):
                vertex = floor_model.getTransformedVertex(i)
                vertices.append(vertex)
                
            for i, v in enumerate(vertices):
                screen_x, screen_y = self.perspective_projection(v.x, v.y, v.z)
                print(f"Floor vertex {i}: ({v.x}, {v.y}, {v.z}) -> Screen: ({screen_x}, {screen_y})")
            
        except Exception as e:
            print(f"Error creating floor model: {e}")
            import traceback
            traceback.print_exc()
            self.floor_object = None
        
        # For testing, let's temporarily skip creating floor headsets
        self.floor_headsets = self.create_floor_headsets() # enable this when the simulation is ready

    def create_floor_model(self):
        """Create a floor model programmatically to ensure it matches the scene boundaries"""
        floor_model = Model('./data/floor.obj')  # Try to load the floor model

        # Define scene boundaries (match render_boundaries)
        scene_boundary = {
            'min_x': -30.0,
            'max_x': 30.0,
            'min_z': -40.0,
            'max_z': 0.0,
            'min_y': -1.0,  # Assuming floor height
            'max_y': 1.0    # Small margin above the floor
        }

        # Calculate floor dimensions based on scene boundaries
        floor_width = scene_boundary['max_x'] - scene_boundary['min_x']
        floor_depth = scene_boundary['max_z'] - scene_boundary['min_z']
        floor_height = scene_boundary['min_y']

        # Position and scale the floor model to match the scene boundaries
        floor_model.setPosition(
            (scene_boundary['min_x'] + scene_boundary['max_x']) / 2,  # Center X
            floor_height,  # Y position
            (scene_boundary['min_z'] + scene_boundary['max_z']) / 2   # Center Z
        )
        floor_model.scale = [floor_width / 10, 1, floor_depth / 10]  # Scale to match dimensions
        floor_model.updateTransform()

        # Create a colored version of the floor
        floor_object = ColoredModel(
            floor_model,
            diffuse_color=(125, 125, 125)  # Light gray
        )

        return floor_object
    
    def create_floor_headsets(self):
        """Create sliding headsets that will collide on the floor using actual floor vertices"""
        headsets = []
        
        # Get actual floor vertices from the transformed floor model
        if not hasattr(self, 'floor_object') or not self.floor_object:
            print("ERROR: Floor object not available for headset placement")
            return []
        
        floor_model = self.floor_object.model
        floor_vertices = []
        
        # Get all transformed vertices from the floor model
        for i in range(len(floor_model.vertices)):
            vertex = floor_model.getTransformedVertex(i)
            floor_vertices.append(vertex)
        
        # Find the min and max boundaries from actual vertices
        min_x = float('inf')
        max_x = float('-inf')
        min_z = float('inf')
        max_z = float('-inf')
        floor_y = 0
        
        for vertex in floor_vertices:
            min_x = min(min_x, vertex.x)
            max_x = max(max_x, vertex.x)
            min_z = min(min_z, vertex.z)
            max_z = max(max_z, vertex.z)
            floor_y = vertex.y  # All floor vertices should have same Y
        
        # Debug output
        print(f"Actual floor boundaries from vertices: X={min_x} to {max_x}, Z={min_z} to {max_z}, Y={floor_y}")
        
        # Define boundary using actual vertex positions
        boundary = {
            'min_x': min_x,
            'max_x': max_x,
            'min_z': min_z,
            'max_z': max_z,
            'floor_y': floor_y
        }
        
        # Headset properties
        headset_radius = 1.0
        safety_margin = 1.0
        
        # Define safe area with margin for headset radius
        safe_area = {
            'min_x': boundary['min_x'] + headset_radius + safety_margin,
            'max_x': boundary['max_x'] - headset_radius - safety_margin,
            'min_z': boundary['min_z'] + headset_radius + safety_margin,
            'max_z': boundary['max_z'] - headset_radius - safety_margin,
            'height': boundary['floor_y'] + headset_radius + 0.5  # Place above floor
        }
        
        # Check if safe area is valid
        if safe_area['min_x'] >= safe_area['max_x'] or safe_area['min_z'] >= safe_area['max_z']:
            print("WARNING: Safe area too small, adjusting to minimal values")
            # Use minimal margins if needed
            safe_area['min_x'] = boundary['min_x'] + headset_radius
            safe_area['max_x'] = boundary['max_x'] - headset_radius
            safe_area['min_z'] = boundary['min_z'] + headset_radius
            safe_area['max_z'] = boundary['max_z'] - headset_radius
        
        # Calculate center and dimensions
        floor_center_x = (boundary['min_x'] + boundary['max_x']) / 2
        floor_center_z = (boundary['min_z'] + boundary['max_z']) / 2
        usable_width = safe_area['max_x'] - safe_area['min_x']
        usable_depth = safe_area['max_z'] - safe_area['min_z']
        
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
        
        # Circle of headsets - reduced number and speed for smaller safe area
        num_circle = 6
        # Use smaller radius to ensure they stay in bounds
        circle_radius = min(usable_width, usable_depth) * 0.3
        
        for i in range(num_circle):
            angle = (i / num_circle) * 2 * math.pi
            
            # Calculate position within safe area
            pos_x = floor_center_x + circle_radius * math.cos(angle)
            pos_z = floor_center_z + circle_radius * math.sin(angle)
            
            # Clamp to safe boundaries
            pos_x = max(safe_area['min_x'], min(safe_area['max_x'], pos_x))
            pos_z = max(safe_area['min_z'], min(safe_area['max_z'], pos_z))
            
            pos = Vector(pos_x, safe_area['height'], pos_z)
            
            # Set velocity toward center with reduced speed
            speed = 10.0  # Reduced from previous values
            vel = Vector(
                -math.cos(angle) * speed,
                0,
                -math.sin(angle) * speed
            )
            
            model = Model('./data/headset.obj')
            model.normalizeGeometry()
            model.setPosition(pos.x, pos.y, pos.z)
            
            colored_model = ColoredModel(model, diffuse_color=colors[i % len(colors)])
            headsets.append(CollisionObject(colored_model, pos, vel, radius=headset_radius, elasticity=0.9))
        
        # Add a few simple headsets in random positions for variety
        num_random = 6
        for i in range(num_random):
            # Random position within safe area
            pos_x = safe_area['min_x'] + random.random() * (safe_area['max_x'] - safe_area['min_x'])
            pos_z = safe_area['min_z'] + random.random() * (safe_area['max_z'] - safe_area['min_z'])
            
            pos = Vector(pos_x, safe_area['height'], pos_z)
            
            # Random velocity
            vel_x = (random.random() * 2 - 1) * 5.0
            vel_z = (random.random() * 2 - 1) * 5.0
            vel = Vector(vel_x, 0, vel_z)
            
            model = Model('./data/headset.obj')
            model.normalizeGeometry()
            model.setPosition(pos.x, pos.y, pos.z)
            
            colored_model = ColoredModel(model, diffuse_color=colors[i % len(colors)])
            headsets.append(CollisionObject(colored_model, pos, vel, radius=headset_radius, elasticity=0.9))
        
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
        """Update physics for floor headsets with guaranteed containment"""
        # Fixed timestep for consistent physics
        fixed_dt = 1/ self.video_recorder.fps
        self.accumulator += dt
        
        # Get actual floor boundaries from floor vertices
        if not hasattr(self, 'floor_object') or not self.floor_object:
            print("ERROR: Floor object not available for physics boundaries")
            return
        
        floor_model = self.floor_object.model
        floor_vertices = []
        
        # Get all transformed vertices from the floor model
        for i in range(len(floor_model.vertices)):
            vertex = floor_model.getTransformedVertex(i)
            floor_vertices.append(vertex)
        
        # Find the min and max boundaries from actual vertices
        min_x = float('inf')
        max_x = float('-inf')
        min_z = float('inf')
        max_z = float('-inf')
        floor_y = 0
        
        for vertex in floor_vertices:
            min_x = min(min_x, vertex.x)
            max_x = max(max_x, vertex.x)
            min_z = min(min_z, vertex.z)
            max_z = max(max_z, vertex.z)
            floor_y = vertex.y  # All floor vertices should have same Y
        
        # Define boundary using actual vertex positions with safety margin
        boundary = {
            'min_x': min_x,
            'max_x': max_x,
            'min_z': min_z,
            'max_z': max_z,
            'floor_y': floor_y,
            'elasticity': 0.95  # High elasticity for boundaries
        }
        
        # Headset properties
        headset_radius = 2.0
        
        # Minimal friction
        self.friction_coefficient = 0.995
        
        while self.accumulator >= fixed_dt:
            # Clear collision records
            for headset in self.floor_headsets:
                headset.clear_collision_history()
            
            # Check collisions between headsets
            for i in range(len(self.floor_headsets)):
                for j in range(i + 1, len(self.floor_headsets)):
                    if self.floor_headsets[i].check_collision(self.floor_headsets[j]):
                        self.floor_headsets[i].resolve_collision(self.floor_headsets[j])
            
            # Update positions and ensure containment within boundaries
            for headset in self.floor_headsets:
                # Apply velocity to get next position
                next_x = headset.position.x + headset.velocity.x * fixed_dt
                next_z = headset.position.z + headset.velocity.z * fixed_dt
                next_y = headset.position.y + headset.velocity.y * fixed_dt
                
                # Check and handle boundary containment
                if next_x - headset_radius < boundary['min_x']:
                    # Left boundary collision
                    excess = boundary['min_x'] - (next_x - headset_radius)
                    next_x = boundary['min_x'] + headset_radius
                    # Reflection with elasticity (like bouncing off another headset)
                    headset.velocity.x = abs(headset.velocity.x) * boundary['elasticity']
                
                if next_x + headset_radius > boundary['max_x']:
                    # Right boundary collision
                    excess = (next_x + headset_radius) - boundary['max_x']
                    next_x = boundary['max_x'] - headset_radius
                    headset.velocity.x = -abs(headset.velocity.x) * boundary['elasticity']
                
                if next_z - headset_radius < boundary['min_z']:
                    # Back boundary collision
                    excess = boundary['min_z'] - (next_z - headset_radius)
                    next_z = boundary['min_z'] + headset_radius
                    headset.velocity.z = abs(headset.velocity.z) * boundary['elasticity']
                
                if next_z + headset_radius > boundary['max_z']:
                    # Front boundary collision
                    excess = (next_z + headset_radius) - boundary['max_z']
                    next_z = boundary['max_z'] - headset_radius
                    headset.velocity.z = -abs(headset.velocity.z) * boundary['elasticity']
                
                # Floor collision handling
                min_height = boundary['floor_y'] + headset_radius + 0.5
                if next_y < min_height:
                    # Floor collision
                    next_y = min_height
                    headset.velocity.y = abs(headset.velocity.y) * 0.5
                    
                    # Apply friction on floor
                    horizontal_speed_squared = headset.velocity.x**2 + headset.velocity.z**2
                    if horizontal_speed_squared > 0.001:
                        headset.velocity.x *= self.friction_coefficient
                        headset.velocity.z *= self.friction_coefficient
                else:
                    # Apply gravity
                    headset.velocity.y -= 9.8 * fixed_dt
                
                # Update position
                headset.position.x = next_x
                headset.position.z = next_z
                headset.position.y = next_y
                
                # CRITICAL: Double-check bounds again after update to ensure no escape
                # This is a fail-safe mechanism
                if headset.position.x - headset_radius < boundary['min_x']:
                    headset.position.x = boundary['min_x'] + headset_radius
                    headset.velocity.x = abs(headset.velocity.x) * boundary['elasticity']
                
                if headset.position.x + headset_radius > boundary['max_x']:
                    headset.position.x = boundary['max_x'] - headset_radius
                    headset.velocity.x = -abs(headset.velocity.x) * boundary['elasticity']
                
                if headset.position.z - headset_radius < boundary['min_z']:
                    headset.position.z = boundary['min_z'] + headset_radius
                    headset.velocity.z = abs(headset.velocity.z) * boundary['elasticity']
                
                if headset.position.z + headset_radius > boundary['max_z']:
                    headset.position.z = boundary['max_z'] - headset_radius
                    headset.velocity.z = -abs(headset.velocity.z) * boundary['elasticity']
                
                # Update model position
                headset.model.model.setPosition(headset.position.x, headset.position.y, headset.position.z)
            
            self.accumulator -= fixed_dt
        
        # Third safety check after all updates to absolutely guarantee containment
        for headset in self.floor_headsets:
            contained = True
            
            if headset.position.x - headset_radius < boundary['min_x']:
                headset.position.x = boundary['min_x'] + headset_radius
                headset.velocity.x = abs(headset.velocity.x) * boundary['elasticity']
                contained = False
            
            if headset.position.x + headset_radius > boundary['max_x']:
                headset.position.x = boundary['max_x'] - headset_radius
                headset.velocity.x = -abs(headset.velocity.x) * boundary['elasticity']
                contained = False
            
            if headset.position.z - headset_radius < boundary['min_z']:
                headset.position.z = boundary['min_z'] + headset_radius
                headset.velocity.z = abs(headset.velocity.z) * boundary['elasticity']
                contained = False
            
            if headset.position.z + headset_radius > boundary['max_z']:
                headset.position.z = boundary['max_z'] - headset_radius
                headset.velocity.z = -abs(headset.velocity.z) * boundary['elasticity']
                contained = False
            
            if not contained:
                # Update model position if corrections were made
                headset.model.model.setPosition(headset.position.x, headset.position.y, headset.position.z)
                if self.frame_count % 60 == 0:
                    print("Applied tertiary boundary enforcement")
    
    def render_model(self, model_obj, floor=False):
        """Render a 3D model with lighting - modified to fix Z-buffer issues"""
        # Get the actual model
        model = getattr(model_obj, 'model', model_obj)
        
        # Precalculate transformed vertices
        transformed_vertices = []
        for i in range(len(model.vertices)):
            vertex = model.getTransformedVertex(i)
            transformed_vertices.append(vertex)
        
        # Calculate face normals and vertex normals
        face_normals = {}
        for face_idx, face in enumerate(model.faces):
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
                vertex_normals.append(Vector(0, 1, 0))  # Default up normal
        
        # Get model color
        if floor:
            model_color = (180, 180, 180)  # Light grey for floor
        else:
            if hasattr(model_obj, 'diffuse_color'):
                model_color = model_obj.diffuse_color
            elif hasattr(model, 'diffuse_color'):
                model_color = model.diffuse_color
            else:
                model_color = (200, 200, 200)
        
        # CRITICAL: Skip backface culling for floor
        # Render faces
        for face in model.faces:
            v0 = transformed_vertices[face[0]]
            v1 = transformed_vertices[face[1]]
            v2 = transformed_vertices[face[2]]
            
            n0 = vertex_normals[face[0]]
            n1 = vertex_normals[face[1]]
            n2 = vertex_normals[face[2]]
            
            # Backface culling - SKIP for floor objects
            if not floor:
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
                if floor:
                    # For floor, use a fixed light grey color
                    r, g, b = model_color
                    color = Color(r, g, b, 255)
                else:
                    # For other objects, apply regular lighting
                    intensity = max(0.2, n * self.light_dir)
                    r, g, b = model_color
                    color = Color(
                        int(r * intensity),
                        int(g * intensity),
                        int(b * intensity),
                        255
                    )
                
                # Create point with correct z-value
                # CRUCIAL FIX: For floor, use a significantly lower z-value
                if floor:
                    # Use a very low z-value to ensure floor is drawn behind everything
                    point = Point(int(screen_x), int(screen_y), v.z - 1000.0, color)
                else:
                    point = Point(int(screen_x), int(screen_y), v.z, color)
                
                point.normal = n
                triangle_points.append(point)
            
            # Render triangle if all points are valid
            if len(triangle_points) == 3:
                self.draw_triangle(triangle_points[0], triangle_points[1], triangle_points[2])

    def draw_triangle(self, p0, p1, p2):
        """Draw a triangle with properly converted integer coordinates with Z-buffer handling"""
        # Create a Triangle instance
        tri = Triangle(p0, p1, p2)
        
        # Special Z-buffer handling for floor
        is_floor_triangle = False
        avg_y = (p0.y + p1.y + p2.y) / 3.0
        if avg_y < 0.1:  # If it's near y=0, it's probably the floor
            is_floor_triangle = True
        
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
                        # For floor triangles, slightly decrease z-value to ensure objects appear on top
                        # This is the key fix - it creates a small Z-buffer bias for the floor
                        if is_floor_triangle:
                            z_value -= 0.01
                        
                        # Perform z-buffer check
                        buffer_index = y * self.image.width + x
                        if buffer_index < len(self.zBuffer) and self.zBuffer[buffer_index] < z_value:
                            self.zBuffer[buffer_index] = z_value
                            self.image.setPixel(x, y, color)

    def draw_triangle_old(self, p0, p1, p2):
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
        """Render a white floor with grid lines"""
        size = 30
        height = 0
        
        # Create a simpler approach using pygame
        # Map 3D corners to screen space
        floor_corners = [
            Vector(-size, height, -size - 10),
            Vector(size, height, -size - 10),
            Vector(size, height, size - 10),
            Vector(-size, height, size - 10)
        ]
        
        # Project to screen space
        screen_corners = []
        for corner in floor_corners:
            screen_x, screen_y = self.perspective_projection(corner.x, corner.y, corner.z)
            if screen_x >= 0 and screen_y >= 0:
                screen_corners.append((screen_x, screen_y))
        
        # Draw floor as a filled polygon if we have all corners
        if len(screen_corners) == 4:
            # Draw filled white floor
            pygame.draw.polygon(
                self.screen,
                (230, 230, 230),  # Light gray/white
                screen_corners
            )
            
            # Draw grid lines on top
            grid_size = 5
            for i in range(-size, size + 1, grid_size):
                # X grid lines
                start_x, start_y = self.perspective_projection(i, height, -size - 10)
                end_x, end_y = self.perspective_projection(i, height, size - 10)
                if start_x >= 0 and start_y >= 0 and end_x >= 0 and end_y >= 0:
                    pygame.draw.line(
                        self.screen,
                        (180, 180, 180),  # Grid line color
                        (start_x, start_y),
                        (end_x, end_y),
                        1  # Line width
                    )
                
                # Z grid lines
                for j in range(-size - 10, size - 9, grid_size):
                    start_x, start_y = self.perspective_projection(-size, height, j)
                    end_x, end_y = self.perspective_projection(size, height, j)
                    if start_x >= 0 and start_y >= 0 and end_x >= 0 and end_y >= 0:
                        pygame.draw.line(
                            self.screen,
                            (180, 180, 180),  # Grid line color
                            (start_x, start_y),
                            (end_x, end_y),
                            1  # Line width
                        )
        
        # Also add the floor to the z-buffer and main image for proper rendering
        # This ensures objects disappearing behind the floor are handled correctly
        corners = [
            Vector(-size, height, -size - 10),
            Vector(size, height, -size - 10),
            Vector(size, height, size - 10),
            Vector(-size, height, size - 10)
        ]
        
        # Create floor points with proper z-values for the z-buffer
        p0 = Point(corners[0].x, corners[0].y, corners[0].z)
        p1 = Point(corners[1].x, corners[1].y, corners[1].z)
        p2 = Point(corners[2].x, corners[2].y, corners[2].z)
        p3 = Point(corners[3].x, corners[3].y, corners[3].z)
        
        # Set normal and color - white with proper opacity
        normal = Vector(0, 1, 0)
        for p in [p0, p1, p2, p3]:
            p.normal = normal
            p.color = Color(255, 255, 255, 255)  # White floor
        
        # Render floor triangles to the z-buffer and image
        self.draw_triangle(p0, p1, p2)
        self.draw_triangle(p0, p2, p3)

    def render_scene(self):
        """Render all scene elements"""
        # Clear image and z-buffer
        self.image = Image(self.width, self.height, Color(20, 20, 40, 255))
        self.zBuffer = [-float('inf')] * self.width * self.height
        
        # ----- Debug visualization for floor boundaries -----
        # Draw a wireframe boundary to show floor position
        boundary = {
            'min_x': -30.0,
            'max_x': 30.0,
            'min_z': -40.0,
            'max_z': 0.0,
            'height': 5.0
        }
        
        # Project floor corners to screen space for debugging
        floor_corners = [
            Vector(boundary['min_x'], boundary['height'], boundary['min_z']),
            Vector(boundary['max_x'], boundary['height'], boundary['min_z']),
            Vector(boundary['max_x'], boundary['height'], boundary['max_z']),
            Vector(boundary['min_x'], boundary['height'], boundary['max_z'])
        ]
                
        # Render floor object first (so it's behind everything else)
        if hasattr(self, 'floor_object') and self.floor_object:
            self.render_model(self.floor_object, True)
        else:
            # Render simple floor if floor model failed to load
            print("Falling back to simple floor rendering") # Debug output
            self.render_floor()
        
        # Render main headset
        self.render_model(self.main_headset["model"])
        
        # Render floor headsets
        for headset in self.floor_headsets:
            self.render_model(headset.model)

        # Apply post-processing effects
        if self.dof_enabled:
            self.image = self.depth_of_field.process(self.image, self.zBuffer, self.width, self.height)
        
        if self.blur_enabled:
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
        self.draw_focal_plane()
        self.draw_debug_info()
        
        # Update display
        pygame.display.flip()
        
        # Capture frame for video if recording
        if self.is_recording:
            self.video_recorder.capture_frame(self.screen)

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
        path = "../IMUdata.csv"
        self.sensor_data = None
        self.dr_filter = None  # Initialize to None explicitly
    
        try:
            if os.path.exists(path):
                print(f"Found IMU data at {path}")
                parser = SensorDataParser(path)
                self.sensor_data = parser.parse()
                print(f"Loaded {len(self.sensor_data)} IMU data points")
                
                # Target for ~27 seconds at 30fps
                self.target_frames = 810
                
                # Calculate speedup factor needed to fit all IMU data in target frames
                self.imu_speedup_factor = max(1, len(self.sensor_data) / self.target_frames)
                print(f"Using speedup factor of {self.imu_speedup_factor:.2f}x to fit all IMU data in 27 seconds")
                
                # Create and calibrate filter
                self.dr_filter = DeadReckoningFilter(alpha=0.98)  # Higher alpha to reduce shakiness
                if self.sensor_data and len(self.sensor_data) > 0:
                    self.dr_filter.calibrate(self.sensor_data[:min(100, len(self.sensor_data))])
                    self.current_data_index = 0
                return
        except Exception as e:
            print(f"Error loading IMU data from {path}: {e}")

    def update_main_headset(self, dt):
        """Update main headset orientation based on IMU data - with speedup"""
        if self.sensor_data and hasattr(self, 'dr_filter') and self.dr_filter:
            if self.current_data_index < len(self.sensor_data):
                # Calculate how many samples to process this frame based on speedup factor
                samples_this_frame = min(
                    math.ceil(self.imu_speedup_factor),
                    len(self.sensor_data) - self.current_data_index
                )
                
                # Process calculated number of samples, but only keep last orientation
                for _ in range(samples_this_frame):
                    if self.current_data_index < len(self.sensor_data):
                        sensor_data = self.sensor_data[self.current_data_index]
                        self.current_data_index += 1
                        orientation = self.dr_filter.update(sensor_data)
                
                # Apply the final orientation to the model
                self.main_headset["model"].model.setQuaternionRotation(orientation)
                
                # Store rotation angles
                roll, pitch, yaw = self.dr_filter.get_euler_angles()
                self.main_headset["rotation"] = [roll, pitch, yaw]
            
            # Calculate progress percentage for display
            self.imu_progress = min(100, (self.current_data_index / len(self.sensor_data)) * 100)
        else:
            # Simple rotation if no IMU data or filter
            self.main_headset["rotation"][0] += dt * 1.0
            self.main_headset["rotation"][1] += dt * 1.5
            self.main_headset["rotation"][2] += dt * 0.8
            
            self.main_headset["model"].model.setRotation(
                self.main_headset["rotation"][0],
                self.main_headset["rotation"][1],
                self.main_headset["rotation"][2]
            )
            
    def run_old(self):
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
                self.stop_recording()
                print("Simulation complete")
                break
        
        # Clean up
        pygame.quit()
        print(f"Simulation ended after {self.frame_count} frames")

    def run(self):
        """Main simulation loop"""
        clock = pygame.time.Clock()
        running = True
        
        print("VR Headset Simulation")
        print("Controls: B (blur), R (reset), P (pause), V (record), ESC (quit)")
        
        # Start recording automatically
        self.start_recording()
        
        while running:
            dt = min(clock.tick(30) / 1000.0, 0.1)
            
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
            
            # Exit when we reach target frames or used all IMU data
            # if self.frame_count >= self.target_frames or \
            # (self.sensor_data and self.current_data_index >= len(self.sensor_data)):
            if self.frame_count >= self.target_frames:
                self.stop_recording()
                print("Simulation complete")
                break
        
        # Clean up
        pygame.quit()
        print(f"Simulation ended after {self.frame_count} frames")

    
if __name__ == "__main__":
    simulation = HeadsetSimulation()
    simulation.run()