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
        
        self.recent_collisions = set()
        
    def update(self, dt):
        self.position = self.position + (self.velocity * dt)
        
        if self.position.y - self.radius < 0:
            self.position.y = self.radius  # Place on floor
            self.velocity.y = abs(self.velocity.y) * 0.8  # Bounce with damping
        
        self.model.setPosition(self.position.x, self.position.y, self.position.z)
        
    def check_collision(self, other):
        """
        Check for collision with another object using sphere intersection.
        
        Args:
            other: Another CollisionObject to check against
            
        Returns:
            bool: True if collision detected
        """
        distance = (self.position - other.position).length()
        
        return distance < (self.radius + other.radius)
    
    def resolve_collision(self, other):
        """
        Handle collision response between two objects.
        Uses elastic collision formulas for spheres.
        
        Args:
            other: The CollisionObject we're colliding with
        """
        if other in self.recent_collisions:
            return
            
        normal = (other.position - self.position).normalize()
        
        relative_velocity = other.velocity - self.velocity
        
        velocity_along_normal = relative_velocity * normal
        
        if velocity_along_normal > 0:
            return
        
        cor = min(self.elasticity, other.elasticity)
        
        impulse = -(1 + cor) * velocity_along_normal
        impulse /= 2  # Equal mass assumption
        
        impulse_vector = normal * impulse
        self.velocity = self.velocity - impulse_vector
        other.velocity = other.velocity + impulse_vector
        
        self.recent_collisions.add(other)
        other.recent_collisions.add(self)
        
    def clear_collision_history(self):
        self.recent_collisions.clear()