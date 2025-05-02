import math
import numpy as np
from vector import Vector

class Quaternion:
    """
    Quaternion class for representing 3D rotations.
    Quaternions provide a more efficient and numerically stable way to represent rotations
    compared to matrices or Euler angles, and avoid gimbal lock issues.
    """
    def __init__(self, w, x, y, z):
        """
        Initialize a quaternion with components (w, x, y, z).
        
        Args:
            w: Scalar (real) part
            x, y, z: Vector (imaginary) parts
        """
        self.w, self.x, self.y, self.z = w, x, y, z

    def normalize(self):
        """Normalize the quaternion to unit length."""
        norm = math.sqrt(self.w**2 + self.x**2 + self.y**2 + self.z**2)
        if norm < 1e-10:  # Avoid division by near-zero
            return self
            
        self.w /= norm
        self.x /= norm
        self.y /= norm
        self.z /= norm
        return self

    def conjugate(self):
        """Return the conjugate of this quaternion (inverses the imaginary parts)."""
        return Quaternion(self.w, -self.x, -self.y, -self.z)
    
    def inverse(self):
        """
        Return the inverse of this quaternion.
        For unit quaternions, the inverse is the same as the conjugate.
        """
        return self.conjugate()  # Assuming we're working with unit quaternions

    def __mul__(self, other):
        """
        Multiply this quaternion with another quaternion or scalar.
        Quaternion multiplication corresponds to composition of rotations.
        
        Args:
            other: Another quaternion or a scalar
            
        Returns:
            Quaternion: Result of multiplication
        """
        if isinstance(other, Quaternion):
            return Quaternion(
                self.w * other.w - self.x * other.x - self.y * other.y - self.z * other.z,
                self.w * other.x + self.x * other.w + self.y * other.z - self.z * other.y,
                self.w * other.y - self.x * other.z + self.y * other.w + self.z * other.x,
                self.w * other.z + self.x * other.y - self.y * other.x + self.z * other.w
            )
        else:  # Scalar multiplication
            return Quaternion(
                self.w * other,
                self.x * other,
                self.y * other,
                self.z * other
            )
    
    def rotate_vector(self, vec):
        """
        Rotate a vector by this quaternion.
        
        Args:
            vec: Vector to rotate
            
        Returns:
            Vector: Rotated vector
        """
        # Convert vector to pure quaternion with w=0
        v_quat = Quaternion(0, vec.x, vec.y, vec.z)
        
        # Apply rotation: q * v * q^-1
        rotated = self * v_quat * self.inverse()
        
        # Extract vector part
        return Vector(rotated.x, rotated.y, rotated.z)

    def to_euler_angles(self):
        """
        Convert quaternion to Euler angles (roll, pitch, yaw).
        Uses the convention: roll is rotation around X, pitch around Y, yaw around Z.
        
        Returns:
            tuple: (roll, pitch, yaw) in radians
        """
        # Roll (x-axis rotation)
        sinr_cosp = 2.0 * (self.w * self.x + self.y * self.z)
        cosr_cosp = 1.0 - 2.0 * (self.x * self.x + self.y * self.y)
        roll = math.atan2(sinr_cosp, cosr_cosp)
        
        # Pitch (y-axis rotation)
        sinp = 2.0 * (self.w * self.y - self.z * self.x)
        if abs(sinp) >= 1:
            pitch = math.copysign(math.pi / 2, sinp)  # Use 90 degrees if out of range
        else:
            pitch = math.asin(sinp)
            
        # Yaw (z-axis rotation)
        siny_cosp = 2.0 * (self.w * self.z + self.x * self.y)
        cosy_cosp = 1.0 - 2.0 * (self.y * self.y + self.z * self.z)
        yaw = math.atan2(siny_cosp, cosy_cosp)
        
        return (roll, pitch, yaw)
    
    def to_matrix(self):
        """
        Convert quaternion to rotation matrix.
        
        Returns:
            list: 3x3 rotation matrix as nested lists
        """
        # Normalize quaternion to ensure proper rotation
        self.normalize()
        
        # Calculate matrix elements
        m = [
            [0, 0, 0],
            [0, 0, 0],
            [0, 0, 0]
        ]
        
        m[0][0] = 1.0 - 2.0 * (self.y**2 + self.z**2)
        m[0][1] = 2.0 * (self.x * self.y - self.w * self.z)
        m[0][2] = 2.0 * (self.x * self.z + self.w * self.y)
        
        m[1][0] = 2.0 * (self.x * self.y + self.w * self.z)
        m[1][1] = 1.0 - 2.0 * (self.x**2 + self.z**2)
        m[1][2] = 2.0 * (self.y * self.z - self.w * self.x)
        
        m[2][0] = 2.0 * (self.x * self.z - self.w * self.y)
        m[2][1] = 2.0 * (self.y * self.z + self.w * self.x)
        m[2][2] = 1.0 - 2.0 * (self.x**2 + self.y**2)
        
        return m
    
    def to_axis_angle(self):
        """
        Convert quaternion to axis-angle representation.
        
        Returns:
            tuple: (axis, angle) where axis is a Vector and angle is in radians
        """
        self.normalize()
        
        # Calculate angle
        angle = 2.0 * math.acos(self.w)
        
        # Calculate axis
        s = math.sqrt(1.0 - self.w * self.w)
        if s < 1e-10:  # If s is close to zero, direction doesn't matter
            return Vector(1, 0, 0), 0.0
        else:
            axis = Vector(self.x / s, self.y / s, self.z / s)
            return axis, angle
    
    def slerp(self, other, t):
        """
        Spherical linear interpolation between quaternions.
        
        Args:
            other: Target quaternion
            t: Interpolation parameter (0 to 1)
            
        Returns:
            Quaternion: Interpolated quaternion
        """
        # Calculate the dot product
        dot = self.w * other.w + self.x * other.x + self.y * other.y + self.z * other.z
        
        # Ensure shortest path by flipping one quaternion if necessary
        if dot < 0:
            dot = -dot
            other_w, other_x, other_y, other_z = -other.w, -other.x, -other.y, -other.z
        else:
            other_w, other_x, other_y, other_z = other.w, other.x, other.y, other.z
        
        # If quaternions are very close, use linear interpolation
        if dot > 0.9995:
            result = Quaternion(
                self.w * (1-t) + other_w * t,
                self.x * (1-t) + other_x * t,
                self.y * (1-t) + other_y * t,
                self.z * (1-t) + other_z * t
            )
            result.normalize()
            return result
        
        # Clamp dot product to valid domain of acos
        dot = max(min(dot, 1.0), -1.0)
        
        # Calculate the angle between quaternions
        theta_0 = math.acos(dot)
        sin_theta_0 = math.sin(theta_0)
        
        # Calculate interpolation factors
        theta = theta_0 * t
        sin_theta = math.sin(theta)
        
        # Calculate coefficients
        s1 = math.sin(theta_0 - theta) / sin_theta_0
        s2 = math.sin(theta) / sin_theta_0
        
        # Perform the interpolation
        result = Quaternion(
            s1 * self.w + s2 * other_w,
            s1 * self.x + s2 * other_x,
            s1 * self.y + s2 * other_y,
            s1 * self.z + s2 * other_z
        )
        
        return result


# Static methods for quaternion creation

def from_euler_angles(roll, pitch, yaw):
    """
    Create a quaternion from Euler angles (roll, pitch, yaw).
    Uses the convention: roll is rotation around X, pitch around Y, yaw around Z.
    
    Args:
        roll: Rotation around X-axis in radians
        pitch: Rotation around Y-axis in radians
        yaw: Rotation around Z-axis in radians
        
    Returns:
        Quaternion: Representing the specified rotation
    """
    # Calculate half angles
    cr = math.cos(roll * 0.5)
    sr = math.sin(roll * 0.5)
    cp = math.cos(pitch * 0.5)
    sp = math.sin(pitch * 0.5)
    cy = math.cos(yaw * 0.5)
    sy = math.sin(yaw * 0.5)
    
    # Calculate quaternion components
    w = cr * cp * cy + sr * sp * sy
    x = sr * cp * cy - cr * sp * sy
    y = cr * sp * cy + sr * cp * sy
    z = cr * cp * sy - sr * sp * cy
    
    return Quaternion(w, x, y, z)

def from_axis_angle(axis, angle):
    """
    Create a quaternion from axis-angle representation.
    
    Args:
        axis: Rotation axis as a Vector (will be normalized)
        angle: Rotation angle in radians
        
    Returns:
        Quaternion: Representing the specified rotation
    """
    # Normalize axis
    axis_norm = axis.normalize()
    
    # Calculate half angle
    half_angle = angle * 0.5
    sin_half = math.sin(half_angle)
    
    # Create quaternion
    return Quaternion(
        math.cos(half_angle),
        axis_norm.x * sin_half,
        axis_norm.y * sin_half,
        axis_norm.z * sin_half
    )

def from_rotation_matrix(matrix):
    """
    Create a quaternion from a 3x3 rotation matrix.
    
    Args:
        matrix: 3x3 rotation matrix (list of lists)
        
    Returns:
        Quaternion: Representing the specified rotation
    """
    trace = matrix[0][0] + matrix[1][1] + matrix[2][2]
    
    if trace > 0:
        s = 0.5 / math.sqrt(trace + 1.0)
        w = 0.25 / s
        x = (matrix[2][1] - matrix[1][2]) * s
        y = (matrix[0][2] - matrix[2][0]) * s
        z = (matrix[1][0] - matrix[0][1]) * s
    elif matrix[0][0] > matrix[1][1] and matrix[0][0] > matrix[2][2]:
        s = 2.0 * math.sqrt(1.0 + matrix[0][0] - matrix[1][1] - matrix[2][2])
        w = (matrix[2][1] - matrix[1][2]) / s
        x = 0.25 * s
        y = (matrix[0][1] + matrix[1][0]) / s
        z = (matrix[0][2] + matrix[2][0]) / s
    elif matrix[1][1] > matrix[2][2]:
        s = 2.0 * math.sqrt(1.0 + matrix[1][1] - matrix[0][0] - matrix[2][2])
        w = (matrix[0][2] - matrix[2][0]) / s
        x = (matrix[0][1] + matrix[1][0]) / s
        y = 0.25 * s
        z = (matrix[1][2] + matrix[2][1]) / s
    else:
        s = 2.0 * math.sqrt(1.0 + matrix[2][2] - matrix[0][0] - matrix[1][1])
        w = (matrix[1][0] - matrix[0][1]) / s
        x = (matrix[0][2] + matrix[2][0]) / s
        y = (matrix[1][2] + matrix[2][1]) / s
        z = 0.25 * s
    
    return Quaternion(w, x, y, z)

def identity():
    """
    Create an identity quaternion (no rotation).
    
    Returns:
        Quaternion: Identity quaternion
    """
    return Quaternion(1.0, 0.0, 0.0, 0.0)


class QuaternionIntegrator:
    """
    Utilities for integrating angular velocities using quaternions.
    Useful for IMU-based orientation tracking.
    """
    
    @staticmethod
    def integrate_angular_velocity(quaternion, angular_velocity, dt):
        """
        Integrate angular velocity to update orientation quaternion.
        
        Args:
            quaternion: Current orientation quaternion
            angular_velocity: Angular velocity vector (wx, wy, wz) in radians/sec
            dt: Time step in seconds
            
        Returns:
            Quaternion: Updated orientation
        """
        # Convert angular velocity to quaternion rate of change
        wx, wy, wz = angular_velocity
        
        # Calculate quaternion derivative
        q_dot = Quaternion(
            0.5 * (-quaternion.x * wx - quaternion.y * wy - quaternion.z * wz),
            0.5 * (quaternion.w * wx + quaternion.y * wz - quaternion.z * wy),
            0.5 * (quaternion.w * wy - quaternion.x * wz + quaternion.z * wx),
            0.5 * (quaternion.w * wz + quaternion.x * wy - quaternion.y * wx)
        )
        
        # Apply integration step
        result = Quaternion(
            quaternion.w + q_dot.w * dt,
            quaternion.x + q_dot.x * dt,
            quaternion.y + q_dot.y * dt,
            quaternion.z + q_dot.z * dt
        )
        
        # Normalize to prevent drift
        return result.normalize()
    
    @staticmethod
    def complementary_filter(gyro_quat, accel_quat, alpha=0.98):
        """
        Apply complementary filter to fuse gyroscope and accelerometer orientations.
        
        Args:
            gyro_quat: Orientation from gyroscope integration
            accel_quat: Orientation from accelerometer
            alpha: Weight for gyroscope data (0-1)
            
        Returns:
            Quaternion: Fused orientation
        """
        return gyro_quat.slerp(accel_quat, 1.0 - alpha)


def acceleration_to_orientation(acceleration):
    """
    Convert acceleration vector to orientation quaternion.
    This estimates the tilt of an object based on gravity direction.
    
    Args:
        acceleration: Acceleration vector (includes gravity)
        
    Returns:
        Quaternion: Orientation quaternion
    """
    # Normalize acceleration vector
    acc_norm = acceleration.normalize()
    
    # Reference up vector (gravity points down)
    ref_up = Vector(0, 0, 1)
    
    # Negate acceleration since gravity points down
    acc_up = acc_norm * -1
    
    # Get rotation axis (perpendicular to both vectors)
    axis = acc_up.cross(ref_up)
    
    # Special case: vectors are parallel or anti-parallel
    if axis.norm() < 1e-6:
        if acc_up.z > 0:  # Roughly aligned with up
            return identity()
        else:  # Roughly anti-aligned with up
            return from_axis_angle(Vector(1, 0, 0), math.pi)  # 180° around X
    
    # Find angle between vectors
    dot = acc_up * ref_up
    dot = max(min(dot, 1.0), -1.0)  # Clamp to [-1, 1]
    angle = math.acos(dot)
    
    # Create quaternion from axis-angle
    return from_axis_angle(axis.normalize(), angle)