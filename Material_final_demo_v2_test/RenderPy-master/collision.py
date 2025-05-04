from vector import Vector
from color_support import ColoredModel

class CollisionObject:
    """
    A physics object that can collide with other objects.
    Uses a spherical bounding volume for collision detection.
    """
    def __init__(self, model, position, velocity, radius, elasticity=0.0):
        """
        Initialize a collidable object.
        
        Args:
            model: The 3D model for rendering
            position: Vector representing position in 3D space
            velocity: Vector representing velocity
            radius: Radius of bounding sphere
            elasticity: Coefficient of restitution (0.0 = inelastic, 1.0 = perfectly elastic)
        """
        self.model = model
        self.position = position
        self.velocity = velocity
        self.radius = radius
        self.elasticity = elasticity
        
        # Keep track of previous collisions to prevent "sticking"
        self.recent_collisions = set()
        
    def update(self, dt):
        """Update position based on velocity"""
        self.position = self.position + (self.velocity * dt)
        
        # Apply a simple bounce if object hits the floor
        if self.position.y - self.radius < 0:
            self.position.y = self.radius  # Place on floor
            self.velocity.y = abs(self.velocity.y) * 0.8  # Bounce with damping
        
        # Update model position
        self.model.setPosition(self.position.x, self.position.y, self.position.z)
        
    def check_collision(self, other):
        """
        Check for collision with another object using sphere intersection.
        
        Args:
            other: Another CollisionObject to check against
            
        Returns:
            bool: True if collision detected
        """
        # Calculate distance between centers
        distance = (self.position - other.position).length()
        
        # Collision occurs if distance is less than sum of radii
        return distance < (self.radius + other.radius)
    
    def resolve_collision(self, other):
        """
        Handle collision response between two objects.
        Uses elastic collision formulas for spheres.
        
        Args:
            other: The CollisionObject we're colliding with
        """
        # Skip if we recently collided with this object (prevents sticking)
        if other in self.recent_collisions:
            return
            
        # Calculate collision normal
        normal = (other.position - self.position).normalize()
        
        # Relative velocity
        relative_velocity = other.velocity - self.velocity
        
        # Calculate the velocity along the normal
        velocity_along_normal = relative_velocity * normal
        
        # If objects are separating, no need to resolve
        if velocity_along_normal > 0:
            return
        
        # Use coefficient of restitution (elasticity)
        cor = min(self.elasticity, other.elasticity)
        
        # Calculate impulse scalar
        impulse = -(1 + cor) * velocity_along_normal
        impulse /= 2  # Assuming equal masses for simplicity
        
        # Apply impulse to velocities
        impulse_vector = normal * impulse
        self.velocity = self.velocity - impulse_vector
        other.velocity = other.velocity + impulse_vector
        
        # Add to recent collisions
        self.recent_collisions.add(other)
        other.recent_collisions.add(self)
        
    def clear_collision_history(self):
        """Clear the set of recent collisions"""
        self.recent_collisions.clear()

def setup_collision_scene(models_path='./data/headset.obj', elasticity=0.8):
    """
    Create a scene with multiple colliding objects.
    
    Args:
        models_path: Path to the 3D model file
        elasticity: Coefficient of restitution for collisions
        
    Returns:
        list: List of ColoredCollisionObject instances
    """
    from model import Model
    import math
    import random
    
    objects = []
    
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
    
    # Create several patterns of objects
    
    # Pattern 1: Circle of objects moving inward
    num_circle = 8  # Number of objects in the circle
    circle_radius = 20  # Distance from center
    for i in range(num_circle):
        angle = (i / num_circle) * 2 * math.pi
        # Position objects in a circle
        pos = Vector(
            circle_radius * math.cos(angle),
            1,  # Slightly elevated
            circle_radius * math.sin(angle) - 10  # Centered at z=-10
        )
        # Velocity pointing toward center
        vel = Vector(
            -math.cos(angle) * 2,  # Scale factor of 2 controls speed
            0,
            -math.sin(angle) * 2
        )
        
        model = Model(models_path)
        model.normalizeGeometry()
        model.setPosition(pos.x, pos.y, pos.z)
        
        # Assign a color from our list
        color = colors[i % len(colors)]
        
        objects.append(ColoredCollisionObject(model, pos, vel, radius=1.0, diffuse_color=color, elasticity=elasticity))

    # Add a "billiards break" pattern of different colored objects
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
            # These objects start stationary
            vel = Vector(0, 0, 0)
            
            model = Model(models_path)
            model.normalizeGeometry()
            model.setPosition(pos.x, pos.y, pos.z)
            
            # Get next color
            color = colors[color_index % len(colors)]
            color_index += 1
            
            objects.append(ColoredCollisionObject(model, pos, vel, radius=1.0, diffuse_color=color, elasticity=elasticity))
    
    # Add a "cue ball" white object
    pos = Vector(0, 1, -15)  # Position behind the triangle
    vel = Vector(0, 0, 4)    # Moving forward to hit the triangle
    
    model = Model(models_path)
    model.normalizeGeometry()
    model.setPosition(pos.x, pos.y, pos.z)
    
    # White color for the "cue ball"
    objects.append(ColoredCollisionObject(model, pos, vel, radius=1.0, diffuse_color=(255, 255, 255), elasticity=elasticity))

    return objects

def update_physics(objects, dt, boundaries=None, friction_coefficient=0.95):
    """
    Update physics for all collision objects.
    
    Args:
        objects: List of CollisionObject instances
        dt: Time step
        boundaries: Dictionary with min/max values for x/z and bounce factor
        friction_coefficient: Friction applied to objects on the floor
    """
    # Default boundary settings if none provided
    if boundaries is None:
        boundaries = {
            'min_x': -30.0,
            'max_x': 30.0,
            'min_z': -40.0,
            'max_z': 0.0,
            'bounce_factor': 0.9
        }
    
    # Clear collision records
    for obj in objects:
        obj.clear_collision_history()
    
    # Check collisions between objects
    for i in range(len(objects)):
        for j in range(i + 1, len(objects)):
            if objects[i].check_collision(objects[j]):
                objects[i].resolve_collision(objects[j])
    
    # Apply updates, floor constraints, and friction
    for obj in objects:
        # Update position based on velocity
        obj.update(dt)
        
        # Apply friction when on floor
        if obj.position.y - obj.radius <= 0.01:
            obj.position.y = obj.radius
            
            # Apply friction to horizontal velocity components
            horizontal_speed_squared = (
                obj.velocity.x**2 + 
                obj.velocity.z**2
            )
            
            # Only apply friction if moving horizontally
            if horizontal_speed_squared > 0.001:
                # Apply friction by reducing horizontal velocity
                obj.velocity.x *= friction_coefficient
                obj.velocity.z *= friction_coefficient
                
                # Stop completely if very slow after friction is applied
                if horizontal_speed_squared * friction_coefficient**2 < 0.025:
                    obj.velocity.x = 0
                    obj.velocity.z = 0
        
        # Apply boundary constraints
        if obj.position.x - obj.radius < boundaries['min_x']:
            obj.position.x = boundaries['min_x'] + obj.radius
            obj.velocity.x = -obj.velocity.x * boundaries['bounce_factor']
            
        elif obj.position.x + obj.radius > boundaries['max_x']:
            obj.position.x = boundaries['max_x'] - obj.radius
            obj.velocity.x = -obj.velocity.x * boundaries['bounce_factor']
        
        if obj.position.z - obj.radius < boundaries['min_z']:
            obj.position.z = boundaries['min_z'] + obj.radius
            obj.velocity.z = -obj.velocity.z * boundaries['bounce_factor']
            
        elif obj.position.z + obj.radius > boundaries['max_z']:
            obj.position.z = boundaries['max_z'] - obj.radius
            obj.velocity.z = -obj.velocity.z * boundaries['bounce_factor']
        
        # Update model position
        obj.model.model.setPosition(obj.position.x, obj.position.y, obj.position.z)