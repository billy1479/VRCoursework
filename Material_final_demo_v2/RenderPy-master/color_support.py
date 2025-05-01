from image import Color
from vector import Vector

class ColoredModel:
    """
    Wrapper class for Model that adds color properties.
    This enables different objects to have different colors in the rendered output.
    """
    def __init__(self, model, diffuse_color=None, specular_color=None, ambient_color=None, shininess=10):
        """
        Initialize a colored model with various material properties.
        
        Args:
            model: The base Model object
            diffuse_color: Main color of the object (RGB tuple)
            specular_color: Color of highlights (RGB tuple)
            ambient_color: Color in shadows (RGB tuple)
            shininess: Specular highlight sharpness (higher = sharper)
        """
        self.model = model
        
        # Default colors if none provided
        if diffuse_color is None:
            diffuse_color = (200, 200, 200)  # Light gray default
        if specular_color is None:
            specular_color = (255, 255, 255)  # White highlights
        if ambient_color is None:
            # Default ambient is darker version of diffuse
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
        """
        Calculate the color of a vertex based on lighting.
        
        Args:
            normal: Surface normal vector at the vertex
            light_dir: Direction vector to the light source
            view_dir: Direction vector to the viewer (for specular)
            
        Returns:
            Color: RGBA color for the vertex
        """
        # Diffuse lighting (Lambert's cosine law)
        diffuse_intensity = max(0, normal * light_dir)
        
        # Calculate base color from diffuse lighting
        r = int(self.ambient_color[0] + self.diffuse_color[0] * diffuse_intensity)
        g = int(self.ambient_color[1] + self.diffuse_color[1] * diffuse_intensity)
        b = int(self.ambient_color[2] + self.diffuse_color[2] * diffuse_intensity)
        
        # Add specular highlight if view direction is provided
        if view_dir is not None:
            # Calculate reflection vector
            reflect_dir = light_dir - normal * (2 * (light_dir * normal))
            reflect_dir = reflect_dir.normalize()
            
            # Calculate specular intensity (Phong model)
            specular_intensity = max(0, view_dir * reflect_dir) ** self.shininess
            
            # Add specular component
            r += int(self.specular_color[0] * specular_intensity)
            g += int(self.specular_color[1] * specular_intensity)
            b += int(self.specular_color[2] * specular_intensity)
        
        # Clamp RGB values to valid range
        r = min(255, max(0, r))
        g = min(255, max(0, g))
        b = min(255, max(0, b))
        
        return Color(r, g, b, 255)
    
    def __getattr__(self, name):
        """
        Delegate attribute access to the wrapped model.
        This allows ColoredModel to be used as a drop-in replacement for Model.
        """
        return getattr(self.model, name)


def setup_colored_scene():
    """
    Create a scene with multiple colored headsets.
    
    Returns:
        list: List of ColoredCollisionObject instances
    """
    from model import Model, CollisionObject
    
    headsets = []
    
    # Define some interesting colors (RGB tuples)
    colors = [
        (255, 0, 0),    # Red
        (0, 255, 0),    # Green
        (0, 0, 255),    # Blue
        (255, 255, 0),  # Yellow
        (255, 0, 255),  # Magenta
        (0, 255, 255),  # Cyan
        (255, 165, 0),  # Orange
        (128, 0, 128)   # Purple
    ]
    
    # Create several patterns of headsets with different colors
    
    # Pattern 1: Circle of headsets moving inward
    num_circle = 8  # Number of headsets in the circle
    circle_radius = 20  # Distance from center
    for i in range(num_circle):
        angle = (i / num_circle) * 2 * 3.14159
        # Position headsets in a circle
        pos = Vector(
            circle_radius * 3.14159.cos(angle),
            1,  # Slightly elevated
            circle_radius * 3.14159.sin(angle) - 10  # Centered at z=-10
        )
        # Velocity pointing toward center
        vel = Vector(
            -3.14159.cos(angle) * 2,  # Scale factor of 2 controls speed
            0,
            -3.14159.sin(angle) * 2
        )
        
        model = Model('data/headset.obj')
        model.normalizeGeometry()
        model.setPosition(pos.x, pos.y, pos.z)
        
        # Assign a color from our list
        color = colors[i % len(colors)]
        
        headsets.append(ColoredCollisionObject(model, pos, vel, radius=1.0, diffuse_color=color))

    # Add a "billiards break" pattern of different colored headsets
    triangle_size = 3  # Number of rows in triangle
    start_z = -5
    color_index = 0
    for row in range(triangle_size):
        for col in range(row + 1):
            pos = Vector(
                (col - row/2) * 2,  # Center the triangle
                1,
                start_z + row * 2
            )
            # These headsets start stationary
            vel = Vector(0, 0, 0)
            
            model = Model('data/headset.obj')
            model.normalizeGeometry()
            model.setPosition(pos.x, pos.y, pos.z)
            
            # Get next color
            color = colors[color_index % len(colors)]
            color_index += 1
            
            headsets.append(ColoredCollisionObject(model, pos, vel, radius=1.0, diffuse_color=color))
    
    # Add a "cue ball" white headset
    pos = Vector(0, 1, -15)  # Position behind the triangle
    vel = Vector(0, 0, 4)    # Moving forward to hit the triangle
    
    model = Model('data/headset.obj')
    model.normalizeGeometry()
    model.setPosition(pos.x, pos.y, pos.z)
    
    # White color for the "cue ball"
    headsets.append(ColoredCollisionObject(model, pos, vel, radius=1.0, diffuse_color=(255, 255, 255)))

    return headsets


def render_triangle_with_color(triangle_points, image, zBuffer, colored_model, light_dir, camera_pos=None):
    """
    Render a triangle with proper coloring based on the model's material.
    
    Args:
        triangle_points: List of 3 Points with position and normal data
        image: Image to render to
        zBuffer: Z-buffer for depth testing
        colored_model: ColoredModel instance with material properties
        light_dir: Direction vector to the light source
        camera_pos: Optional camera position for specular highlights
    """
    from shape import Triangle, Point
    
    # Unpack points and normals
    p0, p1, p2 = triangle_points
    n0, n1, n2 = [p.normal for p in triangle_points] if hasattr(p0, 'normal') else [None, None, None]
    
    # Skip if no normals (shouldn't happen in a proper setup)
    if n0 is None or n1 is None or n2 is None:
        return
    
    # Calculate view directions if camera position is provided
    view_dirs = None
    if camera_pos is not None:
        view_dirs = [
            (camera_pos - Vector(p.x, p.y, p.z)).normalize() for p in triangle_points
        ]
    
    # Calculate colors for each vertex based on lighting
    colors = []
    for i, (p, n) in enumerate(zip(triangle_points, [n0, n1, n2])):
        view_dir = view_dirs[i] if view_dirs else None
        color = colored_model.calculate_vertex_color(n, light_dir, view_dir)
        colors.append(color)
    
    # Create new points with calculated colors
    colored_points = [
        Point(p.x, p.y, p.z, color) for p, color in zip(triangle_points, colors)
    ]
    
    # Render the triangle with the new colored points
    Triangle(colored_points[0], colored_points[1], colored_points[2]).draw_faster(image, zBuffer)