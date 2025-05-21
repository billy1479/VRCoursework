import numpy as np
from image import Image, Color

class DepthOfFieldEffect:
    def __init__(self, focal_distance=15.0, focal_range=5.0, blur_strength=1.0):
        self.focal_distance = focal_distance
        self.focal_range = focal_range
        self.blur_strength = blur_strength
        self.enabled = True
    
    def process(self, image, z_buffer, width, height):
        result = Image(width, height, Color(0, 0, 0, 255))
        
        blur_map = self._create_blur_map(z_buffer, width, height)
        
        for y in range(height):
            for x in range(width):
                # Left half: apply DoF
                if x < width // 2:
                    blur_radius = blur_map[y * width + x]
                    if blur_radius <= 0.5:
                        self._copy_pixel(image, result, x, y)
                    else:
                        self._apply_gaussian_blur(image, result, x, y, int(blur_radius))
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
    
    def _create_gaussian_kernel(self, radius):
        size = 2 * radius + 1
        kernel = np.zeros((size, size))
        
        # Standard deviation based on radius
        sigma = radius / 2.0
        
        # Compute Gaussian kernel
        for y in range(size):
            for x in range(size):
                # Calculate distance from center
                dx = x - radius
                dy = y - radius
                distance_squared = dx * dx + dy * dy
                
                # Gaussian function
                kernel[y, x] = np.exp(-distance_squared / (2 * sigma * sigma))
        
        # Normalize kernel so weights sum to 1
        kernel = kernel / np.sum(kernel)
        
        return kernel
    
    def _copy_pixel(self, source, dest, x, y):
        idx = self._get_pixel_index(source, x, y)
        if idx + 3 < len(source.buffer):
            r = source.buffer[idx]
            g = source.buffer[idx + 1]
            b = source.buffer[idx + 2]
            a = source.buffer[idx + 3]
            dest.setPixel(x, y, Color(r, g, b, a))
    
    def _apply_gaussian_blur(self, source, dest, x, y, blur_radius):
        width, height = source.width, source.height
        
        # Limit blur radius to reasonable range
        radius = max(1, min(5, int(blur_radius)))
        
        # Create Gaussian kernel
        kernel = self._create_gaussian_kernel(radius)
        
        # Initialize color accumulators
        r_sum, g_sum, b_sum = 0.0, 0.0, 0.0
        weight_sum = 0.0
        
        # Apply kernel to pixels in a square around the center
        for ky in range(kernel.shape[0]):
            by = y - radius + ky  # source image y-coordinate
            
            if by < 0 or by >= height:
                continue
                
            for kx in range(kernel.shape[1]):
                bx = x - radius + kx  # source image x-coordinate
                
                if bx < 0 or bx >= width:
                    continue
                    
                # Get weight from kernel
                weight = kernel[ky, kx]
                
                # Get pixel color
                idx = self._get_pixel_index(source, bx, by)
                if idx + 2 < len(source.buffer):
                    r_sum += source.buffer[idx] * weight
                    g_sum += source.buffer[idx + 1] * weight
                    b_sum += source.buffer[idx + 2] * weight
                    weight_sum += weight
        
        # Calculate weighted average
        if weight_sum > 0:
            r = int(r_sum / weight_sum)
            g = int(g_sum / weight_sum)
            b = int(b_sum / weight_sum)
            
            # Get alpha from original pixel
            idx = self._get_pixel_index(source, x, y)
            a = source.buffer[idx + 3] if idx + 3 < len(source.buffer) else 255
            
            # Set blurred pixel
            dest.setPixel(x, y, Color(r, g, b, a))
    
    def _get_pixel_index(self, image, x, y):
        flipY = (image.height - y - 1)
        index = (flipY * image.width + x) * 4 + flipY + 1
        return index