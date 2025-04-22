from image import Image, Color
from vector import Vector
import math

class SimpleMotionBlur:
    """
    Extremely simplified motion blur for better performance.
    Uses basic velocity-based blur with minimal computation.
    """
    def __init__(self, blur_strength=1.0):
        self.blur_strength = blur_strength
        self.previous_positions = {}  # Store previous positions by object ID
        
    def update_object_positions(self, objects):
        """Update the stored positions for next frame"""
        for obj_id, obj in enumerate(objects):
            # Store current position for next frame
            self.previous_positions[obj_id] = Vector(obj.position.x, obj.position.y, obj.position.z)
    
    def apply_blur(self, image, objects, width, height, projection_func):
        """
        Apply a very simple and fast motion blur effect.
        Only processes a few of the fastest moving objects.
        
        Args:
            image: The rendered image
            objects: List of scene objects
            width, height: Screen dimensions
            projection_func: Function to project 3D coordinates to screen
            
        Returns:
            Image with blur applied
        """
        # Simple copy to start with
        result = Image(width, height, Color(0, 0, 0, 255))
        for y in range(height):
            for x in range(width):
                idx = self._get_pixel_index(image, x, y)
                result.buffer[idx] = image.buffer[idx]
                result.buffer[idx+1] = image.buffer[idx+1]
                result.buffer[idx+2] = image.buffer[idx+2]
                result.buffer[idx+3] = image.buffer[idx+3]
        
        # Find the fastest 3 objects
        moving_objects = []
        for obj_id, obj in enumerate(objects):
            if obj_id in self.previous_positions:
                prev_pos = self.previous_positions[obj_id]
                # Calculate screen-space velocity
                curr_screen_x, curr_screen_y = projection_func(obj.position.x, obj.position.y, obj.position.z, width, height)
                prev_screen_x, prev_screen_y = projection_func(prev_pos.x, prev_pos.y, prev_pos.z, width, height)
                
                # Calculate velocity
                vel_x = curr_screen_x - prev_screen_x
                vel_y = curr_screen_y - prev_screen_y
                vel_mag = math.sqrt(vel_x*vel_x + vel_y*vel_y)
                
                if vel_mag > 1.0:  # Only consider objects moving fast enough
                    moving_objects.append((obj_id, obj, vel_x, vel_y, vel_mag))
        
        # Sort by velocity magnitude and take top 3
        moving_objects.sort(key=lambda x: x[4], reverse=True)
        moving_objects = moving_objects[:3]
        
        # Process each fast-moving object
        for obj_id, obj, vel_x, vel_y, vel_mag in moving_objects:
            # Project center to screen
            center_x, center_y = projection_func(obj.position.x, obj.position.y, obj.position.z, width, height)
            
            # Calculate radius in screen space (simplified)
            radius = int(obj.radius * 30)  # Just a rough approximation
            
            # Calculate blur parameters
            blur_length = min(15, vel_mag * self.blur_strength)
            samples = 3  # Very few samples for performance
            
            # Process a small area around the object
            min_x = max(0, int(center_x - radius))
            max_x = min(width-1, int(center_x + radius))
            min_y = max(0, int(center_y - radius))
            max_y = min(height-1, int(center_y + radius))
            
            # Sample a very sparse grid (every 3 pixels) for performance
            for y in range(min_y, max_y+1, 3):
                for x in range(min_x, max_x+1, 3):
                    # Simple circle test
                    dx = x - center_x
                    dy = y - center_y
                    if dx*dx + dy*dy > radius*radius:
                        continue
                    
                    # Very simple blur - just average a few samples
                    r_sum = g_sum = b_sum = 0
                    for i in range(samples):
                        # Sample along velocity vector
                        t = (i / (samples-1) - 0.5) * blur_length
                        sample_x = int(x + vel_x * t / vel_mag)
                        sample_y = int(y + vel_y * t / vel_mag)
                        
                        # Clamp to bounds
                        sample_x = max(0, min(width-1, sample_x))
                        sample_y = max(0, min(height-1, sample_y))
                        
                        # Get color
                        idx = self._get_pixel_index(image, sample_x, sample_y)
                        r_sum += image.buffer[idx]
                        g_sum += image.buffer[idx+1]
                        b_sum += image.buffer[idx+2]
                    
                    # Set blurred color
                    idx = self._get_pixel_index(result, x, y)
                    result.buffer[idx] = r_sum // samples
                    result.buffer[idx+1] = g_sum // samples
                    result.buffer[idx+2] = b_sum // samples
        
        return result
    
    def _get_pixel_index(self, image, x, y):
        """Get buffer index for a pixel"""
        flipY = (image.height - y - 1)
        return (flipY * image.width + x) * 4 + flipY + 1