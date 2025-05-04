import numpy as np
from image import Image, Color

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
        # In the provided code snippet, the line `result = Image(width, height, Color(0, 0, 0, 255))`
        # is creating a new image object named `result`.
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