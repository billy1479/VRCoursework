from image import Image, Color
from vector import Vector
import math

class ExtremeMotionBlur:
    """
    Extreme motion blur effect with maximum visibility.
    Designed to make the effect very obvious.
    """
    def __init__(self, blur_strength=3.0):
        self.blur_strength = blur_strength
        self.previous_positions = {}  # Store previous positions by object ID
        self.frame_history = []       # Store previous frames for blending
        self.max_history = 2          # Only keep 2 frames
        
    def update_object_positions(self, objects):
        """Update the stored positions for next frame"""
        for obj_id, obj in enumerate(objects):
            # Store current position for next frame
            self.previous_positions[obj_id] = Vector(obj.position.x, obj.position.y, obj.position.z)
    
    def apply_blur(self, image, objects, width, height, projection_func):
        """
        Apply an extreme motion blur effect that's very visible.
        Uses a combination of velocity blur and frame accumulation.
        
        Args:
            image: The rendered image
            objects: List of scene objects
            width, height: Screen dimensions
            projection_func: Function to project 3D coordinates to screen
            
        Returns:
            Image with extreme blur applied
        """
        # Store current frame in history
        self.frame_history.append(image)
        if len(self.frame_history) > self.max_history:
            self.frame_history.pop(0)
        
        # Create result image
        result = Image(width, height, Color(0, 0, 0, 255))
        
        # PART 1: Apply frame accumulation blur (more visible)
        if len(self.frame_history) > 1:
            current = self.frame_history[-1]
            previous = self.frame_history[-2]
            
            # Frame blending weights
            current_weight = 0.7
            previous_weight = 0.3
            
            # Blend frames
            for y in range(height):
                for x in range(width):
                    idx_curr = self._get_pixel_index(current, x, y)
                    idx_prev = self._get_pixel_index(previous, x, y)
                    
                    # Get current and previous colors
                    r_curr = current.buffer[idx_curr]
                    g_curr = current.buffer[idx_curr + 1]
                    b_curr = current.buffer[idx_curr + 2]
                    a_curr = current.buffer[idx_curr + 3]
                    
                    r_prev = previous.buffer[idx_prev]
                    g_prev = previous.buffer[idx_prev + 1]
                    b_prev = previous.buffer[idx_prev + 2]
                    
                    # Blend with weights
                    r = int(r_curr * current_weight + r_prev * previous_weight)
                    g = int(g_curr * current_weight + g_prev * previous_weight)
                    b = int(b_curr * current_weight + b_prev * previous_weight)
                    
                    # Set in result
                    idx_result = self._get_pixel_index(result, x, y)
                    result.buffer[idx_result] = r
                    result.buffer[idx_result + 1] = g
                    result.buffer[idx_result + 2] = b
                    result.buffer[idx_result + 3] = a_curr
        else:
            # If no previous frame, just copy current
            for y in range(height):
                for x in range(width):
                    idx = self._get_pixel_index(image, x, y)
                    idx_result = self._get_pixel_index(result, x, y)
                    result.buffer[idx_result] = image.buffer[idx]
                    result.buffer[idx_result + 1] = image.buffer[idx + 1]
                    result.buffer[idx_result + 2] = image.buffer[idx + 2]
                    result.buffer[idx_result + 3] = image.buffer[idx + 3]
        
        # PART 2: Add extreme directional blur for fast objects
        # Find all moving objects
        moving_objects = []
        for obj_id, obj in enumerate(objects):
            if obj_id in self.previous_positions:
                prev_pos = self.previous_positions[obj_id]
                
                # Calculate velocity vector
                vel = Vector(
                    obj.position.x - prev_pos.x,
                    obj.position.y - prev_pos.y,
                    obj.position.z - prev_pos.z
                )
                vel_mag = vel.length()
                
                # Calculate screen velocity
                curr_screen_x, curr_screen_y = projection_func(obj.position.x, obj.position.y, obj.position.z, width, height)
                prev_screen_x, prev_screen_y = projection_func(prev_pos.x, prev_pos.y, prev_pos.z, width, height)
                
                # Screen velocity components
                vel_screen_x = curr_screen_x - prev_screen_x
                vel_screen_y = curr_screen_y - prev_screen_y
                vel_screen_mag = math.sqrt(vel_screen_x*vel_screen_x + vel_screen_y*vel_screen_y)
                
                if vel_screen_mag > 1.0:  # Only consider objects moving on screen
                    moving_objects.append((obj_id, obj, vel_screen_x, vel_screen_y, vel_screen_mag))
        
        # Process all moving objects
        for obj_id, obj, vel_x, vel_y, vel_mag in moving_objects:
            # Project center to screen
            center_x, center_y = projection_func(obj.position.x, obj.position.y, obj.position.z, width, height)
            
            # Calculate object radius in screen space
            radius = int(obj.radius * 40)  # Make larger for visibility
            
            # Calculate exaggerated blur parameters
            blur_length = min(40, vel_mag * self.blur_strength)  # Limit maximum blur length
            samples = 6  # More samples for smoother effect
            
            # Process area around the object
            min_x = max(0, int(center_x - radius))
            max_x = min(width-1, int(center_x + radius))
            min_y = max(0, int(center_y - radius))
            max_y = min(height-1, int(center_y + radius))
            
            # Visual debugging - mark the object bounds
            """
            # Draw a circle to show the object boundary
            for angle in range(0, 360, 5):
                rad_x = int(center_x + radius * math.cos(math.radians(angle)))
                rad_y = int(center_y + radius * math.sin(math.radians(angle)))
                
                if 0 <= rad_x < width and 0 <= rad_y < height:
                    idx = self._get_pixel_index(result, rad_x, rad_y)
                    result.buffer[idx] = 255  # Red outline
                    result.buffer[idx+1] = 0
                    result.buffer[idx+2] = 0
            """
            
            # Process every pixel - need to process all for visibility
            for y in range(min_y, max_y+1):
                for x in range(min_x, max_x+1):
                    # Check if within radius
                    dx = x - center_x
                    dy = y - center_y
                    dist_sq = dx*dx + dy*dy
                    
                    if dist_sq > radius*radius:
                        continue
                    
                    # Calculate object influence (stronger near center)
                    influence = 1.0 - math.sqrt(dist_sq) / radius
                    
                    # Only apply blur if significant influence
                    if influence < 0.1:
                        continue
                    
                    # Apply directional blur
                    r_sum = g_sum = b_sum = 0
                    total_weight = 0
                    
                    for i in range(samples):
                        # Calculate sample along velocity vector
                        # Range is skewed to -0.8 to 0.2 to create trailing effect
                        t = (i / (samples-1) * 1.0 - 0.8) * influence
                        
                        # Scale by blur length
                        sample_x = int(x + vel_x * t * blur_length / vel_mag)
                        sample_y = int(y + vel_y * t * blur_length / vel_mag)
                        
                        # Clamp to image bounds
                        sample_x = max(0, min(width-1, sample_x))
                        sample_y = max(0, min(height-1, sample_y))
                        
                        # Weight trails more than leading edge (makes streak effect)
                        weight = 1.0 - abs(t + 0.3)  # Weight centered around -0.3 (trailing)
                        weight = max(0.1, weight) * influence
                        
                        # Get sample color from original image (not result)
                        idx = self._get_pixel_index(image, sample_x, sample_y)
                        r_sum += image.buffer[idx] * weight
                        g_sum += image.buffer[idx+1] * weight
                        b_sum += image.buffer[idx+2] * weight
                        total_weight += weight
                    
                    if total_weight > 0:
                        # Calculate final color
                        r = int(r_sum / total_weight)
                        g = int(g_sum / total_weight)
                        b = int(b_sum / total_weight)
                        
                        # Set in result
                        idx = self._get_pixel_index(result, x, y)
                        result.buffer[idx] = r
                        result.buffer[idx+1] = g
                        result.buffer[idx+2] = b
        
        return result
    
    def _get_pixel_index(self, image, x, y):
        """Get buffer index for a pixel"""
        flipY = (image.height - y - 1)
        return (flipY * image.width + x) * 4 + flipY + 1