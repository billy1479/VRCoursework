from image import Color
from vector import Vector

class ColoredModel:
    def __init__(self, model, diffuse_color=None, specular_color=None, ambient_color=None, shininess=10):
        self.model = model
        
        if diffuse_color is None:
            diffuse_color = (200, 200, 200)  # Light gray default
        if specular_color is None:
            specular_color = (255, 255, 255)  # White highlights
        if ambient_color is None:
            ambient_color = (
                int(diffuse_color[0] * 0.2),
                int(diffuse_color[1] * 0.2),
                int(diffuse_color[2] * 0.2)
            )
            
        self.diffuse_color = diffuse_color
        self.specular_color = specular_color
        self.ambient_color = ambient_color
        self.shininess = shininess
    
    def calculate_vertex_color(self, normal, light_dir, view_dir=None):
        diffuse_intensity = max(0, normal * light_dir)
        
        r = int(self.ambient_color[0] + self.diffuse_color[0] * diffuse_intensity)
        g = int(self.ambient_color[1] + self.diffuse_color[1] * diffuse_intensity)
        b = int(self.ambient_color[2] + self.diffuse_color[2] * diffuse_intensity)
        
        if view_dir is not None:
            reflect_dir = light_dir - normal * (2 * (light_dir * normal))
            reflect_dir = reflect_dir.normalize()
            
            specular_intensity = max(0, view_dir * reflect_dir) ** self.shininess
            
            r += int(self.specular_color[0] * specular_intensity)
            g += int(self.specular_color[1] * specular_intensity)
            b += int(self.specular_color[2] * specular_intensity)
        
        r = min(255, max(0, r))
        g = min(255, max(0, g))
        b = min(255, max(0, b))
        
        return Color(r, g, b, 255)
    
    def __getattr__(self, name):
        return getattr(self.model, name)