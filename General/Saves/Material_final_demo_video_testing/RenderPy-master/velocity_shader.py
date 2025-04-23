from image import Image, Color
from vector import Vector
import math
from model import Matrix4

def create_velocity_buffer(objects, prev_transforms, width, height, perspective_function):
    """
    Create a velocity buffer based on current and previous object transformations.
    This is similar to the shader approach used in game engines.
    
    Args:
        objects: List of objects with position/transformation data
        prev_transforms: Dictionary mapping object IDs to previous transformation matrices
        width, height: Screen dimensions
        perspective_function: Function to convert 3D to screen coordinates
        
    Returns:
        List of velocity vectors for each pixel
    """
    # Initialize velocity buffer with None values
    velocity_buffer = [None] * (width * height)
    
    # Process each object
    for obj_id, obj in enumerate(objects):
        # Skip objects with no previous transform
        if obj_id not in prev_transforms:
            continue
            
        # Get current and previous transformation matrices
        current_transform = obj.model.transform
        previous_transform = prev_transforms[obj_id]
        
        # Get object vertices
        vertices = obj.model.vertices
        
        # Process each vertex
        for vertex_idx, vertex in enumerate(vertices):
            # Transform vertex with current matrix
            current_transformed = current_transform.multiply_vector(vertex)
            
            # Transform vertex with previous matrix
            previous_transformed = previous_transform.multiply_vector(vertex)
            
            # Project both positions to screen space
            current_x, current_y = perspective_function(
                current_transformed.x, 
                current_transformed.y, 
                current_transformed.z,
                width, height
            )
            
            previous_x, previous_y = perspective_function(
                previous_transformed.x, 
                previous_transformed.y, 
                previous_transformed.z,
                width, height
            )
            
            # Calculate screen-space velocity
            velocity_x = current_x - previous_x
            velocity_y = current_y - previous_y
            
            # Store velocity in buffer (for pixels around this vertex)
            for y in range(max(0, current_y - 5), min(height, current_y + 6)):
                for x in range(max(0, current_x - 5), min(width, current_x + 6)):
                    # Calculate distance from vertex to this pixel
                    dist_sq = (x - current_x)**2 + (y - current_y)**2
                    
                    # Skip if too far away
                    if dist_sq > 25:  # 5 pixel radius
                        continue
                        
                    # Weight based on distance (closer = stronger influence)
                    weight = 1.0 - math.sqrt(dist_sq) / 5.0
                    
                    # Calculate buffer index
                    buffer_idx = y * width + x
                    
                    # Store velocity
                    if velocity_buffer[buffer_idx] is None:
                        velocity_buffer[buffer_idx] = (velocity_x * weight, velocity_y * weight)
                    else:
                        # Blend with existing velocity
                        existing_vx, existing_vy = velocity_buffer[buffer_idx]
                        velocity_buffer[buffer_idx] = (
                            existing_vx + velocity_x * weight, 
                            existing_vy + velocity_y * weight
                        )
    
    return velocity_buffer

def store_current_transforms(objects):
    """
    Store current transformation matrices for all objects.
    
    Args:
        objects: List of objects with transform data
        
    Returns:
        Dictionary mapping object IDs to transformation matrices
    """
    transforms = {}
    for obj_id, obj in enumerate(objects):
        # Make a deep copy of the transformation matrix
        transform_copy = Matrix4()
        for i in range(4):
            for j in range(4):
                transform_copy.data[i][j] = obj.model.transform.data[i][j]
        transforms[obj_id] = transform_copy
    return transforms

def apply_velocity_blur(image, velocity_buffer, blur_strength=0.7, max_samples=4):
    """
    Apply motion blur based on a velocity buffer.
    Similar to how shader-based motion blur works.
    
    Args:
        image: Input image
        velocity_buffer: Buffer containing velocity vectors
        blur_strength: Overall blur intensity
        max_samples: Maximum samples to use
        
    Returns:
        Blurred image
    """
    width, height = image.width, image.height
    result = Image(width, height, Color(0, 0, 0, 255))
    
    # Copy all pixels first
    for y in range(height):
        for x in range(width):
            idx = _get_pixel_index(image, x, y)
            r = image.buffer[idx]
            g = image.buffer[idx + 1]
            b = image.buffer[idx + 2]
            a = image.buffer[idx + 3]
            result.setPixel(x, y, Color(r, g, b, a))
    
    # Apply blur only where we have velocity information
    for y in range(height):
        for x in range(width):
            buffer_idx = y * width + x
            
            # Skip if no velocity data
            if buffer_idx >= len(velocity_buffer) or velocity_buffer[buffer_idx] is None:
                continue
                
            # Get velocity vector
            velocity_x, velocity_y = velocity_buffer[buffer_idx]
            
            # Skip if negligible velocity
            velocity_magnitude = math.sqrt(velocity_x**2 + velocity_y**2)
            if velocity_magnitude < 0.5:
                continue
                
            # Scale velocity by blur strength
            blur_length = min(20, velocity_magnitude * blur_strength)
            
            # Skip if minimal blur
            if blur_length < 1.0:
                continue
                
            # Calculate number of samples based on velocity
            num_samples = min(max_samples, max(2, int(blur_length / 3) + 1))
            
            # Accumulate samples
            r_sum, g_sum, b_sum = 0, 0, 0
            
            for i in range(num_samples):
                # Calculate sample position along velocity vector
                t = (i / (num_samples - 1) - 0.5) * blur_length
                sample_x = int(x + velocity_x * t / velocity_magnitude)
                sample_y = int(y + velocity_y * t / velocity_magnitude)
                
                # Clamp to image bounds
                sample_x = max(0, min(width - 1, sample_x))
                sample_y = max(0, min(height - 1, sample_y))
                
                # Get color at sample position
                idx = _get_pixel_index(image, sample_x, sample_y)
                r_sum += image.buffer[idx]
                g_sum += image.buffer[idx + 1]
                b_sum += image.buffer[idx + 2]
            
            # Average the samples
            r = int(r_sum / num_samples)
            g = int(g_sum / num_samples)
            b = int(b_sum / num_samples)
            
            # Get alpha from original pixel
            idx = _get_pixel_index(image, x, y)
            a = image.buffer[idx + 3]
            
            # Set the pixel
            result.setPixel(x, y, Color(r, g, b, a))
    
    return result

def _get_pixel_index(image, x, y):
    """Calculate the index of a pixel in the image buffer."""
    flipY = (image.height - y - 1)
    index = (flipY * image.width + x) * 4 + flipY + 1
    return index