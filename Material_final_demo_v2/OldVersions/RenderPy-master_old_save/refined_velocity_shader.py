from image import Image, Color
from vector import Vector
import math
from model import Matrix4, Vec4

class ViewProjectionState:
    """Tracks the current and previous view-projection matrices for motion blur"""
    
    def __init__(self, width, height):
        # Create initial matrices
        self.current_view_proj = self._create_view_projection_matrix(width, height)
        self.previous_view_proj = self._create_view_projection_matrix(width, height)
    
    def _create_view_projection_matrix(self, width, height):
        """Create a typical view-projection matrix for the camera"""
        # Create view matrix (camera transformation)
        # Assuming camera at (0, 5, -25) looking at origin
        view_matrix = Matrix4()
        # ... setup view matrix based on camera position and orientation
        # For simplicity, we'll use identity matrix for now
        
        # Create projection matrix 
        fov = math.pi / 3.0  # 60-degree field of view
        aspect = width / height
        near = 0.1
        far = 100.0
        proj_matrix = Matrix4.perspective(fov, aspect, near, far)
        
        # Combine view and projection
        return proj_matrix  # Simplified - normally would be proj_matrix * view_matrix
    
    def update(self, camera_position, camera_target, width, height):
        """Update matrices for the current frame"""
        # Store the current matrix as the previous one
        self.previous_view_proj = self.current_view_proj
        
        # Create a new current view-projection matrix
        self.current_view_proj = self._create_view_projection_matrix(width, height)
        
        # In a real engine, you would update based on actual camera movement

def compute_velocity_buffer(vertices, current_view_proj, previous_view_proj, width, height):
    """
    Compute velocity buffer using current and previous view-projection matrices.
    This closely follows the HLSL shader approach from the code snippet.
    
    Args:
        vertices: List of vertices in world space
        current_view_proj: Current view-projection matrix
        previous_view_proj: Previous view-projection matrix
        width, height: Screen dimensions
        
    Returns:
        List of velocity vectors (x,y) for each vertex
    """
    velocities = []
    
    for vertex in vertices:
        # Current viewport position (HLSL: float4 currentPos = H)
        world_pos = Vec4(vertex.x, vertex.y, vertex.z, 1.0)
        current_pos = current_view_proj.multiply(world_pos)
        
        # Previous position using previous view-projection matrix
        # (HLSL: float4 previousPos = mul(worldPos, g_previousViewProjectionMatrix))
        previous_pos = previous_view_proj.multiply(world_pos)
        
        # Convert to nonhomogeneous points [-1,1] by dividing by w
        # (HLSL: previousPos /= previousPos.w)
        if abs(current_pos.w) > 0.001:
            current_ndc_x = current_pos.x / current_pos.w
            current_ndc_y = current_pos.y / current_pos.w
        else:
            current_ndc_x, current_ndc_y = 0, 0
            
        if abs(previous_pos.w) > 0.001:
            previous_ndc_x = previous_pos.x / previous_pos.w
            previous_ndc_y = previous_pos.y / previous_pos.w
        else:
            previous_ndc_x, previous_ndc_y = 0, 0
        
        # Calculate velocity in NDC space
        # (HLSL: float2 velocity = (currentPos - previousPos) / 2.f)
        velocity_ndc_x = (current_ndc_x - previous_ndc_x) / 2.0
        velocity_ndc_y = (current_ndc_y - previous_ndc_y) / 2.0
        
        # Convert from NDC space [-1,1] to screen space [0,width/height]
        velocity_screen_x = velocity_ndc_x * width / 2.0
        velocity_screen_y = velocity_ndc_y * height / 2.0
        
        velocities.append((velocity_screen_x, velocity_screen_y))
    
    return velocities

def create_full_velocity_buffer(objects, view_projection_state, width, height):
    """
    Create a full screen velocity buffer based on object movement and camera changes.
    This is the full implementation of the shader approach.
    
    Args:
        objects: List of renderable objects
        view_projection_state: ViewProjectionState object tracking matrices
        width, height: Screen dimensions
        
    Returns:
        2D array of velocity vectors for each pixel
    """
    # Initialize empty velocity buffer
    velocity_buffer = [[None for x in range(width)] for y in range(height)]
    
    # Process each object
    for obj in objects:
        model = obj.model
        
        # Get world-space vertices
        world_vertices = []
        for vertex_idx in range(len(model.vertices)):
            world_vertices.append(model.getTransformedVertex(vertex_idx))
        
        # Compute per-vertex velocities based on view-projection changes
        vertex_velocities = compute_velocity_buffer(
            world_vertices,
            view_projection_state.current_view_proj,
            view_projection_state.previous_view_proj,
            width, height
        )
        
        # For each face in the model
        for face_idx, face in enumerate(model.faces):
            # Get vertices for this face
            v0, v1, v2 = face
            
            # Get world positions and velocities
            pos0 = world_vertices[v0]
            pos1 = world_vertices[v1]
            pos2 = world_vertices[v2]
            
            vel0 = vertex_velocities[v0]
            vel1 = vertex_velocities[v1]
            vel2 = vertex_velocities[v2]
            
            # Project to screen space
            screen_pos0 = ndc_to_screen(view_projection_state.current_view_proj.multiply_vector(pos0), width, height)
            screen_pos1 = ndc_to_screen(view_projection_state.current_view_proj.multiply_vector(pos1), width, height)
            screen_pos2 = ndc_to_screen(view_projection_state.current_view_proj.multiply_vector(pos2), width, height)
            
            # Determine bounding box of triangle in screen space
            min_x = max(0, min(screen_pos0[0], screen_pos1[0], screen_pos2[0], width))
            max_x = min(width-1, max(screen_pos0[0], screen_pos1[0], screen_pos2[0], 0))
            min_y = max(0, min(screen_pos0[1], screen_pos1[1], screen_pos2[1], height))
            max_y = min(height-1, max(screen_pos0[1], screen_pos1[1], screen_pos2[1], 0))
            
            # For each pixel in the bounding box
            for y in range(int(min_y), int(max_y)+1):
                for x in range(int(min_x), int(max_x)+1):
                    # Check if pixel is inside the triangle
                    pixel_pos = (x, y)
                    if point_in_triangle(pixel_pos, screen_pos0, screen_pos1, screen_pos2):
                        # Calculate barycentric coordinates
                        b0, b1, b2 = barycentric_coords(pixel_pos, screen_pos0, screen_pos1, screen_pos2)
                        
                        # Interpolate velocity based on barycentric coordinates
                        vel_x = b0 * vel0[0] + b1 * vel1[0] + b2 * vel2[0]
                        vel_y = b0 * vel0[1] + b1 * vel1[1] + b2 * vel2[1]
                        
                        # Scale velocity for more visible effect
                        # In real-time rendering, velocity scaling depends on factors like frame time
                        velocity_scale = 1.5
                        vel_x *= velocity_scale
                        vel_y *= velocity_scale
                        
                        # Store in velocity buffer (overwrite existing values)
                        velocity_buffer[y][x] = (vel_x, vel_y)
    
    return velocity_buffer

def ndc_to_screen(pos, width, height):
    """Convert normalized device coordinates to screen space"""
    # Perform perspective division
    if abs(pos.w) > 0.001:
        ndc_x = pos.x / pos.w
        ndc_y = pos.y / pos.w
    else:
        ndc_x, ndc_y = 0, 0
    
    # Convert from [-1,1] to screen coordinates
    screen_x = (ndc_x + 1.0) * width / 2.0
    screen_y = (ndc_y + 1.0) * height / 2.0
    
    return (screen_x, screen_y)

def point_in_triangle(p, v0, v1, v2):
    """Check if point p is inside triangle (v0, v1, v2)"""
    # Compute vectors
    v0p = (p[0] - v0[0], p[1] - v0[1])
    v1p = (p[0] - v1[0], p[1] - v1[1])
    v2p = (p[0] - v2[0], p[1] - v2[1])
    
    # Compute cross products
    c1 = v0p[0] * (v1[1] - v0[1]) - v0p[1] * (v1[0] - v0[0])
    c2 = v1p[0] * (v2[1] - v1[1]) - v1p[1] * (v2[0] - v1[0])
    c3 = v2p[0] * (v0[1] - v2[1]) - v2p[1] * (v0[0] - v2[0])
    
    # Check if all have the same sign
    return (c1 >= 0 and c2 >= 0 and c3 >= 0) or (c1 <= 0 and c2 <= 0 and c3 <= 0)

def barycentric_coords(p, v0, v1, v2):
    """Calculate barycentric coordinates of point p in triangle (v0, v1, v2)"""
    # Calculate area of the triangle using cross product
    area = 0.5 * abs((v1[0] - v0[0]) * (v2[1] - v0[1]) - (v2[0] - v0[0]) * (v1[1] - v0[1]))
    
    # Calculate areas of sub-triangles
    area0 = 0.5 * abs((v1[0] - p[0]) * (v2[1] - p[1]) - (v2[0] - p[0]) * (v1[1] - p[1]))
    area1 = 0.5 * abs((v0[0] - p[0]) * (v2[1] - p[1]) - (v2[0] - p[0]) * (v0[1] - p[1]))
    area2 = 0.5 * abs((v0[0] - p[0]) * (v1[1] - p[1]) - (v1[0] - p[0]) * (v0[1] - p[1]))
    
    # Calculate barycentric coordinates
    if area > 0:
        b0 = area0 / area
        b1 = area1 / area
        b2 = area2 / area
    else:
        # Degenerate triangle
        b0, b1, b2 = 1/3, 1/3, 1/3
    
    return b0, b1, b2

def apply_velocity_blur(image, velocity_buffer, blur_strength=1.0, num_samples=8):
    """
    Apply motion blur using velocity buffer.
    This is the rasterization equivalent of a motion blur shader.
    
    Args:
        image: Original rendered image
        velocity_buffer: 2D array of velocity vectors
        blur_strength: Strength of the blur effect
        num_samples: Number of samples along the velocity vector
        
    Returns:
        Image with motion blur applied
    """
    width, height = image.width, image.height
    result = Image(width, height, Color(0, 0, 0, 255))
    
    # Copy original image to result
    for y in range(height):
        for x in range(width):
            idx = get_pixel_index(image, x, y)
            r = image.buffer[idx]
            g = image.buffer[idx + 1]
            b = image.buffer[idx + 2]
            a = image.buffer[idx + 3]
            result.setPixel(x, y, Color(r, g, b, a))
    
    # For each pixel in the image
    for y in range(height):
        for x in range(width):
            # Get velocity for this pixel
            velocity = velocity_buffer[y][x]
            
            # Skip if no velocity
            if velocity is None:
                continue
            
            vel_x, vel_y = velocity
            
            # Skip if velocity is too small
            velocity_magnitude = math.sqrt(vel_x**2 + vel_y**2)
            if velocity_magnitude < 0.5:
                continue
            
            # Scale velocity by blur strength
            vel_x *= blur_strength
            vel_y *= blur_strength
            
            # Recalculate magnitude after scaling
            velocity_magnitude = math.sqrt(vel_x**2 + vel_y**2)
            
            # Limit maximum blur length
            max_blur_length = 20
            if velocity_magnitude > max_blur_length:
                scale_factor = max_blur_length / velocity_magnitude
                vel_x *= scale_factor
                vel_y *= scale_factor
                velocity_magnitude = max_blur_length
            
            # Skip if still minimal blur
            if velocity_magnitude < 1.0:
                continue
            
            # Sample along velocity vector
            r_sum, g_sum, b_sum = 0, 0, 0
            
            for i in range(num_samples):
                # Calculate sample position
                t = (i / (num_samples - 1) - 0.5)  # Range from -0.5 to 0.5
                sample_x = int(x + vel_x * t)
                sample_y = int(y + vel_y * t)
                
                # Clamp to image bounds
                sample_x = max(0, min(width - 1, sample_x))
                sample_y = max(0, min(height - 1, sample_y))
                
                # Get color at sample position
                idx = get_pixel_index(image, sample_x, sample_y)
                r_sum += image.buffer[idx]
                g_sum += image.buffer[idx + 1]
                b_sum += image.buffer[idx + 2]
            
            # Average the samples
            r = int(r_sum / num_samples)
            g = int(g_sum / num_samples)
            b = int(b_sum / num_samples)
            
            # Set color in result
            idx = get_pixel_index(image, x, y)
            a = image.buffer[idx + 3]
            result.setPixel(x, y, Color(r, g, b, a))
    
    return result

def get_pixel_index(image, x, y):
    """Calculate index in the image buffer for a pixel"""
    flipY = (image.height - y - 1)
    index = (flipY * image.width + x) * 4 + flipY + 1
    return index