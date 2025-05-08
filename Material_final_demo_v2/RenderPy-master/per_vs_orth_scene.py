import pygame
import os
import math
import numpy as np
from image import Image, Color
from vector import Vector
from model import Model
from shape import Triangle, Point

class ProjectionComparisonDemo:
    def __init__(self, width=1000, height=600):
        # Initialize pygame
        pygame.init()
        self.width = width
        self.height = height
        self.screen = pygame.display.set_mode((width, height))
        pygame.display.set_caption("Projection Comparison: Perspective vs Orthographic")
        
        # Split the screen into two halves
        self.half_width = width // 2
        
        # Images and Z-buffers for both projections
        self.perspective_image = Image(self.half_width, height, Color(20, 20, 40, 255))
        self.perspective_zBuffer = [-float('inf')] * self.half_width * height
        
        self.orthographic_image = Image(self.half_width, height, Color(20, 20, 40, 255))
        self.orthographic_zBuffer = [-float('inf')] * self.half_width * height
        
        # Camera and lighting
        self.camera_pos = Vector(0, 15, -40)
        self.camera_target = Vector(0, 0, 0)
        self.light_dir = Vector(0.5, -1, -0.5).normalize()
        
        # Set up scene
        self.models = []
        self.setup_scene()
        
        # Font for labels
        self.font = pygame.font.SysFont('Arial', 24)
        
        # Animation
        self.rotation = 0
        self.demo_mode = "rotate"  # "rotate" or "zoom"
        self.zoom_factor = 1.0
        self.zoom_direction = 1
    
    def setup_scene(self):
        """Create a scene that emphasizes projection differences"""
        # Create a row of cubes at different distances
        self.create_row_of_cubes()
        
        # Create a grid for reference
        self.create_grid()
    
    def create_row_of_cubes(self):
        """Create a row of identical cubes at increasing distances"""
        colors = [
            (255, 0, 0),    # Red
            (0, 255, 0),    # Green
            (0, 0, 255),    # Blue
            (255, 255, 0),  # Yellow
            (255, 0, 255),  # Magenta
        ]
        
        # Create 5 cubes in a row, each farther away
        for i in range(5):
            model = Model('./data/headset.obj')  # Using headset model from provided code
            model.normalizeGeometry()
            
            # Position cubes in a row along Z axis
            z_pos = -5 * (i + 1)  # Each cube 5 units farther away
            x_pos = -8 + (i * 4)  # Spread them out a bit on X axis
            model.setPosition(x_pos, 0, z_pos)
            
            # Scale to make them a bit smaller
            model.scale = [0.5, 0.5, 0.5]
            model.updateTransform()
            
            # Store with color data
            self.models.append({
                "model": model,
                "color": colors[i % len(colors)],
                "position": Vector(x_pos, 0, z_pos)
            })
    
    def create_grid(self):
        """Create a reference grid to help show the projection effects"""
        # Creating a simple grid model is omitted for brevity
        # We'll draw grid lines directly in our render method
        pass
    
    def perspective_projection(self, x, y, z, width=None, height=None):
        """Project 3D coordinates to 2D screen space using perspective projection"""
        if width is None:
            width = self.half_width
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
        
        # This is the key difference in perspective projection:
        # Divide by Z distance (forward_comp) to make distant objects smaller
        x_ndc = right_comp / (forward_comp * math.tan(fov/2) * aspect)
        y_ndc = up_comp / (forward_comp * math.tan(fov/2))
        
        # Convert to screen coordinates
        screen_x = int((x_ndc + 1.0) * width / 2.0)
        screen_y = int((-y_ndc + 1.0) * height / 2.0)
        
        return screen_x, screen_y
    
    def orthographic_projection(self, x, y, z, width=None, height=None):
        """Project 3D coordinates to 2D screen space using orthographic projection"""
        if width is None:
            width = self.half_width
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
        
        # Apply orthographic projection
        # Key difference: No division by Z distance!
        # This keeps objects the same size regardless of distance
        scale = 0.05  # Scale factor to control zoom level
        x_ndc = right_comp * scale
        y_ndc = up_comp * scale
        
        # Convert to screen coordinates
        screen_x = int((x_ndc + 1.0) * width / 2.0)
        screen_y = int((-y_ndc + 1.0) * height / 2.0)
        
        return screen_x, screen_y
    
    def render_model(self, model_obj, image, zBuffer, projection_func, color=None):
        """Render a 3D model with the specified projection function"""
        # Get the actual model
        model = model_obj["model"]
        model_color = model_obj["color"] if color is None else color
        
        # Extract position for easier access
        position = model_obj["position"]
        
        # Calculate model rotation
        rotation_matrix = None
        if self.demo_mode == "rotate":
            # Apply rotation around Y axis
            model.setRotation(0, self.rotation, 0)
        
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
            
            # Create screen points
            triangle_points = []
            for v, n in zip([v0, v1, v2], [n0, n1, n2]):
                screen_x, screen_y = projection_func(v.x, v.y, v.z)
                
                if screen_x < 0 or screen_y < 0 or screen_x >= image.width or screen_y >= image.height:
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
                
                point = Point(screen_x, screen_y, v.z, color)
                point.normal = n
                triangle_points.append(point)
            
            # Render triangle if all points are valid
            if len(triangle_points) == 3:
                self.draw_triangle(triangle_points[0], triangle_points[1], triangle_points[2], image, zBuffer)
    
    def draw_triangle(self, p0, p1, p2, image, zBuffer):
        """Draw a triangle with Z-buffer depth testing"""
        tri = Triangle(p0, p1, p2)
        
        # Calculate bounding box with integer conversion
        ymin = max(min(p0.y, p1.y, p2.y), 0)
        ymax = min(max(p0.y, p1.y, p2.y), image.height - 1)
        
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
                x_end_int = min(int(x_end), image.width - 1)
                
                # Draw horizontal span
                for x in range(x_start_int, x_end_int + 1):
                    point = Point(x, y, color=None)
                    in_triangle, color, z_value = tri.contains_point(point)
                    
                    if in_triangle:
                        # Perform z-buffer check
                        buffer_index = y * image.width + x
                        if buffer_index < len(zBuffer) and zBuffer[buffer_index] < z_value:
                            zBuffer[buffer_index] = z_value
                            image.setPixel(x, y, color)
    
    def render_grid(self, image, zBuffer, projection_func):
        """Render a reference grid to show perspective effects"""
        grid_size = 50
        grid_step = 5
        
        # Draw lines along Z axis (showing perspective)
        for x in range(-grid_size, grid_size + 1, grid_step):
            points = []
            for z in range(-grid_size, grid_size + 1, grid_step):
                screen_x, screen_y = projection_func(x, 0, z)
                if screen_x >= 0 and screen_y >= 0 and screen_x < image.width and screen_y < image.height:
                    points.append((screen_x, screen_y))
            
            # Draw lines connecting points
            if len(points) > 1:
                for i in range(len(points) - 1):
                    p1, p2 = points[i], points[i + 1]
                    self.draw_line(image, p1[0], p1[1], p2[0], p2[1], Color(100, 100, 100, 255))
        
        # Draw lines along X axis
        for z in range(-grid_size, grid_size + 1, grid_step):
            points = []
            for x in range(-grid_size, grid_size + 1, grid_step):
                screen_x, screen_y = projection_func(x, 0, z)
                if screen_x >= 0 and screen_y >= 0 and screen_x < image.width and screen_y < image.height:
                    points.append((screen_x, screen_y))
            
            # Draw lines connecting points
            if len(points) > 1:
                for i in range(len(points) - 1):
                    p1, p2 = points[i], points[i + 1]
                    self.draw_line(image, p1[0], p1[1], p2[0], p2[1], Color(100, 100, 100, 255))
    
    def draw_line(self, image, x1, y1, x2, y2, color):
        """Draw a simple line using Bresenham's algorithm"""
        dx = abs(x2 - x1)
        dy = abs(y2 - y1)
        sx = 1 if x1 < x2 else -1
        sy = 1 if y1 < y2 else -1
        err = dx - dy
        
        while True:
            image.setPixel(x1, y1, color)
            if x1 == x2 and y1 == y2:
                break
            e2 = 2 * err
            if e2 > -dy:
                err -= dy
                x1 += sx
            if e2 < dx:
                err += dx
                y1 += sy
    
    def render_scene(self):
        """Render the scene with both projection methods"""
        # Clear images and z-buffers
        self.perspective_image = Image(self.half_width, self.height, Color(20, 20, 40, 255))
        self.perspective_zBuffer = [-float('inf')] * self.half_width * self.height
        
        self.orthographic_image = Image(self.half_width, self.height, Color(20, 20, 40, 255))
        self.orthographic_zBuffer = [-float('inf')] * self.half_width * self.height
        
        # Zoom the camera in/out for zoom demonstration
        if self.demo_mode == "zoom":
            # Calculate camera position with zoom
            direction = Vector(
                self.camera_target.x - self.camera_pos.x,
                self.camera_target.y - self.camera_pos.y,
                self.camera_target.z - self.camera_pos.z
            ).normalize()
            
            # Move camera along view direction
            distance = 40 - (self.zoom_factor * 30)  # Vary distance between 10 and 40
            self.camera_pos = Vector(
                self.camera_target.x - direction.x * distance,
                self.camera_target.y - direction.y * distance,
                self.camera_target.z - direction.z * distance
            )
        
        # Render reference grid
        self.render_grid(self.perspective_image, self.perspective_zBuffer, self.perspective_projection)
        self.render_grid(self.orthographic_image, self.orthographic_zBuffer, self.orthographic_projection)
        
        # Render models on both projections
        for model in self.models:
            self.render_model(model, self.perspective_image, self.perspective_zBuffer, self.perspective_projection)
            self.render_model(model, self.orthographic_image, self.orthographic_zBuffer, self.orthographic_projection)
        
        # Convert images to pygame surfaces
        # First clear the screen
        self.screen.fill((0, 0, 0))
        
        # Transfer perspective image to left half
        for y in range(self.height):
            for x in range(self.half_width):
                idx = (self.perspective_image.height - y - 1) * self.perspective_image.width * 4 + x * 4 + (self.perspective_image.height - y - 1) + 1
                if idx + 2 < len(self.perspective_image.buffer):
                    r = self.perspective_image.buffer[idx]
                    g = self.perspective_image.buffer[idx + 1]
                    b = self.perspective_image.buffer[idx + 2]
                    self.screen.set_at((x, y), (r, g, b))
        
        # Transfer orthographic image to right half
        for y in range(self.height):
            for x in range(self.half_width):
                idx = (self.orthographic_image.height - y - 1) * self.orthographic_image.width * 4 + x * 4 + (self.orthographic_image.height - y - 1) + 1
                if idx + 2 < len(self.orthographic_image.buffer):
                    r = self.orthographic_image.buffer[idx]
                    g = self.orthographic_image.buffer[idx + 1]
                    b = self.orthographic_image.buffer[idx + 2]
                    self.screen.set_at((x + self.half_width, y), (r, g, b))
        
        # Draw dividing line
        pygame.draw.line(self.screen, (255, 255, 255), (self.half_width, 0), (self.half_width, self.height), 2)
        
        # Add labels
        perspective_label = self.font.render("Perspective Projection", True, (255, 255, 255))
        orthographic_label = self.font.render("Orthographic Projection", True, (255, 255, 255))
        
        self.screen.blit(perspective_label, (20, 20))
        self.screen.blit(orthographic_label, (self.half_width + 20, 20))
        
        # Draw current demo mode
        if self.demo_mode == "rotate":
            mode_text = "Mode: Rotation (Press Z to switch to Zoom)"
        else:
            mode_text = "Mode: Zoom (Press R to switch to Rotation)"
        
        mode_label = self.font.render(mode_text, True, (255, 255, 255))
        self.screen.blit(mode_label, (20, self.height - 30))
        
        # Update display
        pygame.display.flip()
    
    def update_animation(self):
        """Update animation parameters based on current demo mode"""
        if self.demo_mode == "rotate":
            # Update rotation
            self.rotation += 0.01
            if self.rotation > 2 * math.pi:
                self.rotation = 0
        else:
            # Update zoom
            self.zoom_factor += 0.005 * self.zoom_direction
            if self.zoom_factor > 1.0:
                self.zoom_factor = 1.0
                self.zoom_direction = -1
            elif self.zoom_factor < 0.0:
                self.zoom_factor = 0.0
                self.zoom_direction = 1
    
    def handle_events(self):
        """Handle user input events"""
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return False
            
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    return False
                elif event.key == pygame.K_r:
                    self.demo_mode = "rotate"
                elif event.key == pygame.K_z:
                    self.demo_mode = "zoom"
        
        return True
    
    def run(self):
        """Main demo loop"""
        clock = pygame.time.Clock()
        running = True
        
        print("Projection Comparison Demo")
        print("Controls: R (rotation mode), Z (zoom mode), ESC (quit)")
        
        while running:
            # Cap the frame rate
            clock.tick(60)
            
            # Handle events
            running = self.handle_events()
            
            # Update animation
            self.update_animation()
            
            # Render the scene
            self.render_scene()
        
        # Clean up
        pygame.quit()
        print("Demo ended")

if __name__ == "__main__":
    demo = ProjectionComparisonDemo()
    demo.run()