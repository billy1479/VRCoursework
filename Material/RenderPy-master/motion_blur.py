from image import Image, Color
from copy import deepcopy
from vector import Vector
import math
import numpy as np

class MotionBlurEffect:
    """
    Implements a motion blur post-processing effect for the renderer.
    
    Motion blur simulates the effect of movement during camera exposure time,
    creating a more realistic and dynamic appearance for moving objects.
    """
    def __init__(self, blur_strength=0.7, velocity_scale=1.0, max_samples=5):
        """
        Initialize the motion blur effect with customizable parameters.
        
        Args:
            blur_strength: Overall intensity of the blur effect (0.0-1.0)
            velocity_scale: Scaling factor for velocity vectors
            max_samples: Maximum number of samples for the blur effect
        """
        self.blur_strength = blur_strength
        self.velocity_scale = velocity_scale
        self.max_samples = max_samples
        self.previous_frame = None
        self.frame_history = []
        self.max_history = max_samples
        
        # Log initial state
        self._log_status()

    def _log_status(self):
        """Log the current status of motion blur settings to console"""
        status = "ON" if self.blur_strength > 0 else "OFF"
        print(f"[Motion Blur] Status: {status}, Strength: {self.blur_strength:.2f}, Velocity Scale: {self.velocity_scale:.2f}")

    # Add helper methods to change settings with logging:

    def set_blur_strength(self, value):
        """Set blur strength with range checking and logging"""
        old_value = self.blur_strength
        self.blur_strength = max(0.0, min(1.0, value))
        
        # Only log if the value actually changed
        if old_value != self.blur_strength:
            # Check if we're turning it on or off
            if old_value == 0 and self.blur_strength > 0:
                print(f"[Motion Blur] Turned ON with strength {self.blur_strength:.2f}")
            elif old_value > 0 and self.blur_strength == 0:
                print(f"[Motion Blur] Turned OFF")
            else:
                print(f"[Motion Blur] Strength changed: {old_value:.2f} → {self.blur_strength:.2f}")

    def toggle(self):
        """Toggle motion blur on/off"""
        if self.blur_strength > 0:
            # Store the current value to restore when toggled back on
            self._last_strength = self.blur_strength
            self.blur_strength = 0
            print(f"[Motion Blur] Turned OFF")
        else:
            # Restore previous strength or default to 0.7
            self.blur_strength = getattr(self, '_last_strength', 0.7)
            print(f"[Motion Blur] Turned ON with strength {self.blur_strength:.2f}")

    def increase_strength(self, amount=0.1):
        """Increase blur strength by the specified amount"""
        old_value = self.blur_strength
        self.blur_strength = min(1.0, self.blur_strength + amount)
        if old_value != self.blur_strength:
            print(f"[Motion Blur] Strength increased: {old_value:.2f} → {self.blur_strength:.2f}")

    def decrease_strength(self, amount=0.1):
        """Decrease blur strength by the specified amount"""
        old_value = self.blur_strength
        self.blur_strength = max(0.0, self.blur_strength - amount)
        if old_value != self.blur_strength:
            if self.blur_strength == 0:
                print(f"[Motion Blur] Turned OFF")
            else:
                print(f"[Motion Blur] Strength decreased: {old_value:.2f} → {self.blur_strength:.2f}")
        
    def process(self, current_image, velocity_buffer=None):
        """
        Apply motion blur effect to the current frame.
        
        This can work in two modes:
        1. Accumulation blur (frame history)
        2. Velocity-based blur (if velocity_buffer is provided)
        
        Args:
            current_image: Current frame as an Image object
            velocity_buffer: Optional buffer containing velocity vectors for each pixel
            
        Returns:
            Image: Processed image with motion blur applied
        """
        # Make a copy of the current image
        result = deepcopy(current_image)
        
        # If we have a velocity buffer, use velocity-based blur
        if velocity_buffer is not None:
            return self._apply_velocity_blur(current_image, velocity_buffer)
        
        # Otherwise, use temporal accumulation blur
        return self._apply_accumulation_blur(current_image)
    
    def _apply_accumulation_blur(self, current_image):
        """
        Apply motion blur by accumulating and blending previous frames.
        
        Args:
            current_image: Current frame as an Image object
            
        Returns:
            Image: Blurred image
        """
        # Add current frame to history
        self.frame_history.append(deepcopy(current_image))
        
        # Limit history size
        if len(self.frame_history) > self.max_history:
            self.frame_history.pop(0)
        
        # If we don't have enough frames yet, return the current frame
        if len(self.frame_history) < 2:
            return current_image
        
        # Create a new image for the result
        result = Image(current_image.width, current_image.height)
        
        # Calculate weights for each frame (newer frames have more weight)
        weights = []
        total_weight = 0.0
        
        for i in range(len(self.frame_history)):
            # Weight increases with recency (newest frame has most weight)
            weight = (i + 1) / len(self.frame_history) * self.blur_strength
            weights.append(weight)
            total_weight += weight
        
        # Normalize weights
        weights = [w / total_weight for w in weights]
        
        # Blend frames with weighted average
        for y in range(current_image.height):
            for x in range(current_image.width):
                r_sum, g_sum, b_sum, a_sum = 0, 0, 0, 0
                
                # Process each historical frame
                for i, frame in enumerate(self.frame_history):
                    pixel_idx = self._get_pixel_index(frame, x, y)
                    
                    # Get color values
                    r = frame.buffer[pixel_idx]
                    g = frame.buffer[pixel_idx + 1]
                    b = frame.buffer[pixel_idx + 2]
                    a = frame.buffer[pixel_idx + 3]
                    
                    # Add weighted contribution
                    r_sum += int(r * weights[i])
                    g_sum += int(g * weights[i])
                    b_sum += int(b * weights[i])
                    a_sum += int(a * weights[i])
                
                # Set pixel in result image
                result.setPixel(x, y, Color(r_sum, g_sum, b_sum, a_sum))
        
        return result
    
    def _apply_velocity_blur(self, image, velocity_buffer):
        """
        Apply motion blur based on per-pixel velocity vectors.
        
        Args:
            image: Current frame as an Image object
            velocity_buffer: Buffer containing velocity vectors for each pixel
            
        Returns:
            Image: Blurred image
        """
        # Create a new image for the result
        result = Image(image.width, image.height)
        
        # For each pixel
        for y in range(image.height):
            for x in range(image.width):
                buffer_idx = y * image.width + x
                
                # Skip if no velocity data
                if buffer_idx >= len(velocity_buffer) or velocity_buffer[buffer_idx] is None:
                    # Copy original pixel
                    pixel_idx = self._get_pixel_index(image, x, y)
                    r = image.buffer[pixel_idx]
                    g = image.buffer[pixel_idx + 1]
                    b = image.buffer[pixel_idx + 2]
                    a = image.buffer[pixel_idx + 3]
                    result.setPixel(x, y, Color(r, g, b, a))
                    continue
                
                # Get velocity vector
                velocity = velocity_buffer[buffer_idx]
                vel_x, vel_y = velocity[0], velocity[1]
                
                # Scale velocity by blur strength
                scaled_vel_x = vel_x * self.velocity_scale * self.blur_strength
                scaled_vel_y = vel_y * self.velocity_scale * self.blur_strength
                
                # Determine number of samples based on velocity magnitude
                velocity_magnitude = math.sqrt(scaled_vel_x**2 + scaled_vel_y**2)
                num_samples = min(self.max_samples, max(2, int(velocity_magnitude)))
                
                # Accumulate color along velocity vector
                r_sum, g_sum, b_sum, a_sum = 0, 0, 0, 0
                sample_weight = 1.0 / num_samples
                
                for i in range(num_samples):
                    # Calculate sample position (from -0.5 to 0.5 of the velocity)
                    t = (i / (num_samples - 1) - 0.5)
                    sample_x = x + int(scaled_vel_x * t)
                    sample_y = y + int(scaled_vel_y * t)
                    
                    # Clamp coordinates to image bounds
                    sample_x = max(0, min(image.width - 1, sample_x))
                    sample_y = max(0, min(image.height - 1, sample_y))
                    
                    # Get color at sample position
                    pixel_idx = self._get_pixel_index(image, sample_x, sample_y)
                    r = image.buffer[pixel_idx]
                    g = image.buffer[pixel_idx + 1]
                    b = image.buffer[pixel_idx + 2]
                    a = image.buffer[pixel_idx + 3]
                    
                    # Add weighted contribution
                    r_sum += int(r * sample_weight)
                    g_sum += int(g * sample_weight)
                    b_sum += int(b * sample_weight)
                    a_sum += int(a * sample_weight)
                
                # Set pixel in result image
                result.setPixel(x, y, Color(r_sum, g_sum, b_sum, a_sum))
        
        return result
    
    def generate_velocity_buffer(self, positions, prev_positions, width, height, camera_matrix=None):
        """
        Generate a velocity buffer based on current and previous object positions.
        
        Args:
            positions: List of current object positions (world space)
            prev_positions: List of previous object positions (world space)
            width, height: Screen dimensions
            camera_matrix: Optional camera transformation matrix
            
        Returns:
            List: Buffer with velocity vectors for each pixel
        """
        # Create empty velocity buffer
        velocity_buffer = [None] * (width * height)
        
        # For each object
        for i, (curr_pos, prev_pos) in enumerate(zip(positions, prev_positions)):
            # Skip if no previous position
            if prev_pos is None:
                continue
            
            # Calculate velocity vector (world space)
            velocity = Vector(
                curr_pos.x - prev_pos.x,
                curr_pos.y - prev_pos.y,
                curr_pos.z - prev_pos.z
            )
            
            # Project positions to screen space
            curr_screen_x, curr_screen_y = self._project_to_screen(curr_pos, width, height, camera_matrix)
            prev_screen_x, prev_screen_y = self._project_to_screen(prev_pos, width, height, camera_matrix)
            
            # Calculate screen-space velocity
            screen_velocity_x = curr_screen_x - prev_screen_x
            screen_velocity_y = curr_screen_y - prev_screen_y
            
            # Store in velocity buffer (for a small region around the current position)
            radius = 10  # Pixel radius affected by this object's velocity
            for y in range(curr_screen_y - radius, curr_screen_y + radius + 1):
                for x in range(curr_screen_x - radius, curr_screen_x + radius + 1):
                    # Skip if out of bounds
                    if x < 0 or x >= width or y < 0 or y >= height:
                        continue
                    
                    # Calculate distance from object center
                    dx = x - curr_screen_x
                    dy = y - curr_screen_y
                    distance_sq = dx**2 + dy**2
                    
                    # Skip if too far from object center
                    if distance_sq > radius**2:
                        continue
                    
                    # Calculate falloff based on distance (closer = stronger effect)
                    falloff = 1.0 - math.sqrt(distance_sq) / radius
                    
                    # Store velocity vector in buffer
                    buffer_idx = y * width + x
                    velocity_buffer[buffer_idx] = (
                        screen_velocity_x * falloff,
                        screen_velocity_y * falloff
                    )
        
        return velocity_buffer
    
    def _project_to_screen(self, position, width, height, camera_matrix=None):
        """
        Project a 3D position to screen space.
        
        Args:
            position: 3D position vector
            width, height: Screen dimensions
            camera_matrix: Optional camera transformation matrix
            
        Returns:
            tuple: (screen_x, screen_y)
        """
        # Use a simplified projection if no camera matrix provided
        if camera_matrix is None:
            # Simple perspective projection
            z = position.z
            if z >= 0:  # Avoid division by zero or negative z
                z = -0.1
                
            # Project x, y based on z (perspective division)
            screen_x = int((position.x / -z) * width/2 + width/2)
            screen_y = int((position.y / -z) * height/2 + height/2)
        else:
            # Use camera matrix for proper projection
            # (This would need to be implemented based on your camera matrix format)
            # For example:
            # transformed = camera_matrix.multiply_vector(position)
            # screen_x = int((transformed.x + 1.0) * width / 2.0)
            # screen_y = int((transformed.y + 1.0) * height / 2.0)
            
            # Simplified version:
            screen_x = int((position.x + 1.0) * width / 2.0)
            screen_y = int((position.y + 1.0) * height / 2.0)
        
        return screen_x, screen_y
    
    def _get_pixel_index(self, image, x, y):
        """
        Calculate the index of a pixel in the image buffer.
        
        Args:
            image: Image containing the buffer
            x, y: Pixel coordinates
            
        Returns:
            int: Index in the buffer
        """
        # Flip Y coordinate so that up is positive
        flipY = (image.height - y - 1)
        
        # Each row consists of a null byte followed by colors for each pixel
        # Each pixel takes 4 bytes (RGBA)
        index = (flipY * image.width + x) * 4 + flipY + 1
        
        return index
    
    def per_object_velocity_blur(self, image, objects, width, height, perspective_function):
        """
        Apply motion blur individually to objects based on their velocity.
        
        Args:
            image: Current frame as an Image object
            objects: List of objects with position and velocity attributes
            width, height: Screen dimensions
            perspective_function: Function to project 3D points to screen
            
        Returns:
            Image: Image with motion blur applied to moving objects
        """
        # Create a copy of the input image for the result
        result = deepcopy(image)
        
        # For each object
        for obj in objects:
            # Skip if velocity is very low
            velocity_magnitude = obj.velocity.length()
            if velocity_magnitude < 0.1:
                continue
            
            # Calculate blur direction from velocity
            normalized_velocity = Vector(
                obj.velocity.x / velocity_magnitude,
                obj.velocity.y / velocity_magnitude,
                obj.velocity.z / velocity_magnitude
            )
            
            # Project object position to screen space
            center_x, center_y = perspective_function(
                obj.position.x, obj.position.y, obj.position.z, width, height
            )
            
            # Calculate a point along the velocity vector
            vel_point = Vector(
                obj.position.x + normalized_velocity.x,
                obj.position.y + normalized_velocity.y,
                obj.position.z + normalized_velocity.z
            )
            
            # Project the velocity point to screen space
            vel_x, vel_y = perspective_function(
                vel_point.x, vel_point.y, vel_point.z, width, height
            )
            
            # Calculate screen-space velocity direction
            screen_vel_x = vel_x - center_x
            screen_vel_y = vel_y - center_y
            
            # Normalize screen velocity
            screen_vel_length = math.sqrt(screen_vel_x**2 + screen_vel_y**2)
            if screen_vel_length < 0.001:
                continue
                
            screen_vel_x /= screen_vel_length
            screen_vel_y /= screen_vel_length
            
            # Scale by velocity magnitude and blur strength
            blur_length = velocity_magnitude * self.velocity_scale * self.blur_strength
            
            # Convert to pixel units (clamped to reasonable range)
            pixel_blur_length = min(40, max(2, int(blur_length * 10)))
            
            # Calculate bounding box for the object (with some margin)
            margin = pixel_blur_length + 20
            radius = max(obj.radius * 40, 20)  # Convert object radius to pixel radius
            
            min_x = max(0, int(center_x - radius - margin))
            max_x = min(width - 1, int(center_x + radius + margin))
            min_y = max(0, int(center_y - radius - margin))
            max_y = min(height - 1, int(center_y + radius + margin))
            
            # For each pixel in the bounding box
            for y in range(min_y, max_y + 1):
                for x in range(min_x, max_x + 1):
                    # Calculate distance from object center
                    dx = x - center_x
                    dy = y - center_y
                    distance_sq = dx**2 + dy**2
                    
                    # Skip if too far from object
                    if distance_sq > (radius + margin)**2:
                        continue
                        
                    # Calculate how much this pixel belongs to the object (simple mask)
                    # 1.0 = fully belongs to object, 0.0 = not part of object
                    object_influence = max(0.0, 1.0 - math.sqrt(distance_sq) / radius)
                    
                    # Skip if pixel not part of object
                    if object_influence < 0.1:
                        continue
                    
                    # Calculate number of samples based on distance from center and velocity
                    samples = max(2, min(self.max_samples, int(object_influence * pixel_blur_length / 2)))
                    
                    # Accumulate color along blur direction
                    r_sum, g_sum, b_sum, a_sum = 0, 0, 0, 0
                    total_weight = 0.0
                    
                    for s in range(samples):
                        # Calculate sample position along blur direction
                        # Samples are biased toward the motion direction
                        t = (s / (samples - 1) - 0.2) * object_influence
                        sample_x = int(x + screen_vel_x * t * pixel_blur_length)
                        sample_y = int(y + screen_vel_y * t * pixel_blur_length)
                        
                        # Clamp to image bounds
                        sample_x = max(0, min(width - 1, sample_x))
                        sample_y = max(0, min(height - 1, sample_y))
                        
                        # Sample weight decreases with distance from original position
                        weight = 1.0 - abs(t)
                        
                        # Get color at sample position
                        sample_index = self._get_pixel_index(image, sample_x, sample_y)
                        r = image.buffer[sample_index]
                        g = image.buffer[sample_index + 1]
                        b = image.buffer[sample_index + 2]
                        a = image.buffer[sample_index + 3]
                        
                        # Add weighted contribution
                        r_sum += int(r * weight)
                        g_sum += int(g * weight)
                        b_sum += int(b * weight)
                        a_sum += int(a * weight)
                        total_weight += weight
                    
                    # Normalize and set pixel
                    if total_weight > 0:
                        r_final = int(r_sum / total_weight)
                        g_final = int(g_sum / total_weight)
                        b_final = int(b_sum / total_weight)
                        a_final = int(a_sum / total_weight)
                        
                        # Blend with original based on object influence
                        orig_index = self._get_pixel_index(image, x, y)
                        orig_r = image.buffer[orig_index]
                        orig_g = image.buffer[orig_index + 1]
                        orig_b = image.buffer[orig_index + 2]
                        orig_a = image.buffer[orig_index + 3]
                        
                        blend = object_influence * self.blur_strength
                        r_final = int(orig_r * (1 - blend) + r_final * blend)
                        g_final = int(orig_g * (1 - blend) + g_final * blend)
                        b_final = int(orig_b * (1 - blend) + b_final * blend)
                        
                        result.setPixel(x, y, Color(r_final, g_final, b_final, orig_a))
        
        return result