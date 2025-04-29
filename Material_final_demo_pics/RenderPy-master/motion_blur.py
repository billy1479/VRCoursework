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
    def __init__(self, blur_strength=0.7, velocity_scale=1.0, max_samples=4):
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
        self.enabled = True
        self.frame_history = []
        self.max_history = 2  # Reduced history size for better performance
        
        # Log initial state
        self._log_status()

    def _log_status(self):
        """Log the current status of motion blur settings to console"""
        status = "ON" if self.enabled else "OFF"
        print(f"[Motion Blur] Status: {status}, Strength: {self.blur_strength:.2f}, Velocity Scale: {self.velocity_scale:.2f}")

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
        if self.enabled:
            # Store the current value to restore when toggled back on
            self._last_strength = self.blur_strength
            self.enabled = False
            print(f"[Motion Blur] Turned OFF")
        else:
            # Restore previous strength or default to 0.7
            self.enabled = True
            print(f"[Motion Blur] Turned ON with strength {self.blur_strength:.2f}")
        return self.enabled

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
        
    def process(self, current_image):
        """
        Apply simple frame accumulation blur - fastest method
        
        Args:
            current_image: Current frame as an Image object
            
        Returns:
            Image: Processed image with motion blur
        """
        if not self.enabled or self.blur_strength < 0.01:
            return current_image
            
        # Add current frame to history
        self.frame_history.append(deepcopy(current_image))
        
        # Limit history size
        if len(self.frame_history) > self.max_history:
            self.frame_history.pop(0)
        
        # If we don't have enough frames yet, return the current frame
        if len(self.frame_history) < 2:
            return current_image
            
        # Create a new image for the result (copy of current frame)
        result = Image(current_image.width, current_image.height, Color(0, 0, 0, 255))
        
        # Calculate weights - newest frame has most weight
        weight_current = 1.0 - (self.blur_strength * 0.5)
        weight_prev = self.blur_strength
        
        # Only blend two frames - current and previous
        current = self.frame_history[-1]
        previous = self.frame_history[-2]
        
        # Simple and fast blend of just two frames
        for y in range(current_image.height):
            for x in range(current_image.width):
                idx_current = self._get_pixel_index(current, x, y)
                idx_prev = self._get_pixel_index(previous, x, y)
                
                # Get colors from both frames
                r_curr = current.buffer[idx_current]
                g_curr = current.buffer[idx_current + 1]
                b_curr = current.buffer[idx_current + 2]
                a_curr = current.buffer[idx_current + 3]
                
                r_prev = previous.buffer[idx_prev]
                g_prev = previous.buffer[idx_prev + 1]
                b_prev = previous.buffer[idx_prev + 2]
                
                # Blend with weighted average
                r = int(r_curr * weight_current + r_prev * weight_prev)
                g = int(g_curr * weight_current + g_prev * weight_prev)
                b = int(b_curr * weight_current + b_prev * weight_prev)
                
                # Clamp values
                r = min(255, max(0, r))
                g = min(255, max(0, g))
                b = min(255, max(0, b))
                
                result.setPixel(x, y, Color(r, g, b, a_curr))
                
        return result

    def per_object_velocity_blur(self, image, objects, width, height, perspective_function):
        """
        Apply optimized per-object velocity blur with performance in mind.
        
        Args:
            image: Current frame as an Image object
            objects: List of objects with position and velocity attributes
            width, height: Screen dimensions
            perspective_function: Function to project 3D points to screen
            
        Returns:
            Image: Blurred image
        """
        if not self.enabled or self.blur_strength < 0.01:
            return image
            
        # Create a copy of the input image for the result
        result = deepcopy(image)
        
        # Process only fast-moving objects
        velocity_threshold = 0.5
        processed_regions = set()
        
        for obj in objects:
            # Skip if velocity is very low
            velocity_magnitude = obj.velocity.length()
            if velocity_magnitude < velocity_threshold:
                continue
                
            # Calculate blur direction from velocity
            blur_scale = min(velocity_magnitude * self.velocity_scale * self.blur_strength, 15.0)
            
            # Project object position to screen space
            center_x, center_y = perspective_function(
                obj.position.x, obj.position.y, obj.position.z, width, height
            )
            
            # Skip if off-screen
            if center_x < 0 or center_x >= width or center_y < 0 or center_y >= height:
                continue
                
            # Calculate a point along the velocity vector
            vel_point_x = obj.position.x + obj.velocity.x * 0.1
            vel_point_y = obj.position.y + obj.velocity.y * 0.1
            vel_point_z = obj.position.z + obj.velocity.z * 0.1
            
            # Project the velocity point to screen space
            vel_x, vel_y = perspective_function(vel_point_x, vel_point_y, vel_point_z, width, height)
            
            # Calculate screen-space velocity direction
            screen_vel_x = vel_x - center_x
            screen_vel_y = vel_y - center_y
            
            # Normalize screen velocity
            screen_vel_length = math.sqrt(screen_vel_x**2 + screen_vel_y**2)
            if screen_vel_length < 0.1:
                continue
                
            screen_vel_x /= screen_vel_length
            screen_vel_y /= screen_vel_length
            
            # Calculate bounding box for the object
            radius = int(max(obj.radius * 30, 10))  # Convert object radius to pixel radius
            blur_length = int(blur_scale)
            
            min_x = max(0, int(center_x - radius - blur_length))
            max_x = min(width - 1, int(center_x + radius + blur_length))
            min_y = max(0, int(center_y - radius - blur_length))
            max_y = min(height - 1, int(center_y + radius + blur_length))
            
            # Reduced sample count for performance
            samples = min(self.max_samples, max(2, int(blur_scale / 2)))
            
            # For each pixel in the bounding box
            for y in range(min_y, max_y + 1):
                for x in range(min_x, max_x + 1):
                    # Calculate distance from object center
                    dx = x - center_x
                    dy = y - center_y
                    distance_sq = dx**2 + dy**2
                    
                    # Skip if too far from object
                    if distance_sq > (radius + blur_length)**2:
                        continue
                        
                    # Calculate object influence (simple radial falloff)
                    object_influence = max(0.0, 1.0 - math.sqrt(distance_sq) / radius)
                    
                    # Skip if pixel not clearly part of object
                    if object_influence < 0.2:
                        continue
                        
                    # Skip if we've already processed this pixel
                    pixel_key = (x, y)
                    if pixel_key in processed_regions:
                        continue
                        
                    processed_regions.add(pixel_key)
                    
                    # Accumulate color along blur direction
                    r_sum, g_sum, b_sum = 0, 0, 0
                    total_weight = 0.0
                    
                    # Sample along velocity vector
                    for s in range(samples):
                        # Calculate position along blur direction
                        t = (s / (samples - 1) - 0.5) * object_influence
                        sample_x = int(x + screen_vel_x * t * blur_length)
                        sample_y = int(y + screen_vel_y * t * blur_length)
                        
                        # Clamp to image bounds
                        sample_x = max(0, min(width - 1, sample_x))
                        sample_y = max(0, min(height - 1, sample_y))
                        
                        # Sample weight (center samples have more weight)
                        weight = 1.0 - abs(t) * 2
                        
                        # Get color at sample position
                        sample_index = self._get_pixel_index(image, sample_x, sample_y)
                        r = image.buffer[sample_index]
                        g = image.buffer[sample_index + 1]
                        b = image.buffer[sample_index + 2]
                        
                        # Add weighted contribution
                        r_sum += int(r * weight)
                        g_sum += int(g * weight)
                        b_sum += int(b * weight)
                        total_weight += weight
                    
                    # Normalize and set pixel
                    if total_weight > 0:
                        r_final = min(255, max(0, int(r_sum / total_weight)))
                        g_final = min(255, max(0, int(g_sum / total_weight)))
                        b_final = min(255, max(0, int(b_sum / total_weight)))
                        
                        # Get alpha from original pixel
                        orig_index = self._get_pixel_index(image, x, y)
                        orig_a = image.buffer[orig_index + 3]
                        
                        result.setPixel(x, y, Color(r_final, g_final, b_final, orig_a))
        
        return result
        
    def fastest_blur(self, image, objects, width, height, perspective_function):
        """
        Extremely optimized motion blur - sacrifices quality for performance.
        Only applies directional blur to the fastest moving objects.
        
        Args:
            image: Current frame as an Image object
            objects: List of objects with position and velocity attributes
            width, height: Screen dimensions
            perspective_function: Function to project 3D points to screen
            
        Returns:
            Image: Blurred image
        """
        if not self.enabled or self.blur_strength < 0.01:
            return image
            
        # Create a copy of the input image for the result
        result = deepcopy(image)
        
        # Sort objects by velocity and only process the fastest 3
        velocity_threshold = 1.0
        fast_objects = [(obj, obj.velocity.length()) for obj in objects if obj.velocity.length() > velocity_threshold]
        fast_objects.sort(key=lambda x: x[1], reverse=True)
        fast_objects = fast_objects[:3]  # Only process top 3 fastest objects
        
        for obj, vel_magnitude in fast_objects:
            # Calculate blur parameters
            blur_scale = min(vel_magnitude * self.velocity_scale * self.blur_strength, 10.0)
            
            # Project object position to screen space
            center_x, center_y = perspective_function(
                obj.position.x, obj.position.y, obj.position.z, width, height
            )
            
            # Skip if off-screen
            if center_x < 0 or center_x >= width or center_y < 0 or center_y >= height:
                continue
                
            # Calculate a point along the velocity vector
            vel_point_x = obj.position.x + obj.velocity.x * 0.1
            vel_point_y = obj.position.y + obj.velocity.y * 0.1
            vel_point_z = obj.position.z + obj.velocity.z * 0.1
            
            # Project the velocity point to screen space
            vel_x, vel_y = perspective_function(vel_point_x, vel_point_y, vel_point_z, width, height)
            
            # Calculate screen-space velocity direction
            screen_vel_x = vel_x - center_x
            screen_vel_y = vel_y - center_y
            
            # Normalize and scale velocity
            screen_vel_length = math.sqrt(screen_vel_x**2 + screen_vel_y**2)
            if screen_vel_length < 0.1:
                continue
                
            screen_vel_x /= screen_vel_length
            screen_vel_y /= screen_vel_length
            
            # Calculate blur region
            radius = int(max(obj.radius * 25, 8))
            blur_length = int(blur_scale)
            
            # Define reduced bounding box
            min_x = max(0, int(center_x - radius))
            max_x = min(width - 1, int(center_x + radius))
            min_y = max(0, int(center_y - radius))
            max_y = min(height - 1, int(center_y + radius))
            
            # Fixed small number of samples for speed
            samples = 3
            
            # For each pixel in the bounding box
            for y in range(min_y, max_y + 1, 2):  # Process only every other pixel for speed
                for x in range(min_x, max_x + 1, 2):
                    # Simple distance check from center
                    dx = x - center_x
                    dy = y - center_y
                    distance_sq = dx**2 + dy**2
                    
                    # Skip if too far from object center
                    if distance_sq > radius * radius:
                        continue
                    
                    # Simplified blur calculation - sample in velocity direction
                    r_sum, g_sum, b_sum = 0, 0, 0
                    
                    for s in range(samples):
                        # Calculate sample position along velocity vector
                        offset = (s / (samples - 1) - 0.5) * blur_length
                        sample_x = int(x + screen_vel_x * offset)
                        sample_y = int(y + screen_vel_y * offset)
                        
                        # Clamp to image bounds
                        sample_x = max(0, min(width - 1, sample_x))
                        sample_y = max(0, min(height - 1, sample_y))
                        
                        # Get color at sample position
                        sample_index = self._get_pixel_index(image, sample_x, sample_y)
                        r_sum += image.buffer[sample_index]
                        g_sum += image.buffer[sample_index + 1]
                        b_sum += image.buffer[sample_index + 2]
                    
                    # Average the samples
                    r = int(r_sum / samples)
                    g = int(g_sum / samples)
                    b = int(b_sum / samples)
                    
                    # Get alpha from original pixel
                    orig_index = self._get_pixel_index(image, x, y)
                    a = image.buffer[orig_index + 3]
                    
                    # Set the pixel
                    result.setPixel(x, y, Color(r, g, b, a))
        
        return result

    def _get_pixel_index(self, image, x, y):
        """Calculate the index of a pixel in the image buffer."""
        # Flip Y coordinate 
        flipY = (image.height - y - 1)
        
        # Each row has a null byte at the start followed by 4 bytes per pixel
        index = (flipY * image.width + x) * 4 + flipY + 1
        
        return index