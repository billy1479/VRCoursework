
import math
import numpy as np
from vector import Vector
from model import Quaternion

class DeadReckoningFilter:
    def __init__(self, alpha=0.98):
        self.orientation = Quaternion(1, 0, 0, 0)  # Identity quaternion
        self.position = Vector(0, 0, 0)  # Initial position
        self.last_time = None
        self.gyro_bias = (0, 0, 0)  # Calibrated gyroscope bias
        self.alpha = alpha  # Weight for complementary filter (higher = more gyroscope)
        self.velocity_x = 0
        self.velocity_y = 0
        self.velocity_z = 0

    def update(self, sensor_data):
        """
        Update position and orientation based on sensor readings with
        gravity-based tilt correction.
        """
        # Initialize time on first update
        if self.last_time is None:
            self.last_time = sensor_data.time
            return self.orientation

        # Calculate time delta
        dt = sensor_data.time - self.last_time
        self.last_time = sensor_data.time
        
        # Skip if dt is too small (prevent division by zero)
        if dt <= 0:
            return self.orientation
            
        # Bias-corrected angular velocity
        gyro_x = sensor_data.gyroscope[0] - self.gyro_bias[0]
        gyro_y = sensor_data.gyroscope[1] - self.gyro_bias[1]
        gyro_z = sensor_data.gyroscope[2] - self.gyro_bias[2]
        
        # -------------------
        # GYROSCOPE INTEGRATION
        # -------------------
        
        # Create a quaternion representing rotation rate (w=0 for pure rotation)
        rotation_quat = Quaternion(0, gyro_x, gyro_y, gyro_z)
        
        # Calculate quaternion derivative (q' = 0.5 * q * omega)
        q_dot = self._quaternion_multiply(self.orientation, rotation_quat)
        q_dot.w *= 0.5
        q_dot.x *= 0.5
        q_dot.y *= 0.5
        q_dot.z *= 0.5
        
        # Integrate to get new orientation from gyroscope
        gyro_orientation = Quaternion(
            self.orientation.w + q_dot.w * dt,
            self.orientation.x + q_dot.x * dt,
            self.orientation.y + q_dot.y * dt,
            self.orientation.z + q_dot.z * dt
        )
        
        # Normalize to ensure it remains a unit quaternion
        gyro_orientation.normalize()
        
        # -------------------
        # ACCELEROMETER TILT CORRECTION
        # -------------------
        
        # Get accelerometer data
        accel_x, accel_y, accel_z = sensor_data.accelerometer
        
        # Normalize accelerometer data to get unit vector
        accel_magnitude = math.sqrt(accel_x**2 + accel_y**2 + accel_z**2)
        if accel_magnitude > 0.1:  # Only apply correction if acceleration is significant
            # Create normalized acceleration vector in body frame
            accel_normalized_x = accel_x / accel_magnitude
            accel_normalized_y = accel_y / accel_magnitude
            accel_normalized_z = accel_z / accel_magnitude
            
            # Transform acceleration to global frame using current orientation
            accel_body = Quaternion(0, accel_normalized_x, accel_normalized_y, accel_normalized_z)
            orientation_conjugate = self._quaternion_conjugate(self.orientation)
            accel_world = self._quaternion_multiply(
                self._quaternion_multiply(self.orientation, accel_body),
                orientation_conjugate
            )
            
            # Global reference up vector (gravity points down so negate)
            reference_up = Vector(0, 0, 1)  # Z-up coordinate system
            
            # Calculate tilt axis - cross product of measured up and reference up
            # (cross product gives vector perpendicular to both)
            measured_up = Vector(accel_world.x, accel_world.y, accel_world.z).normalize()
            
            # Negate measured_up since gravity points down but we want up vector
            measured_up = measured_up * -1
            
            # Calculate tilt axis (perpendicular to both vectors)
            tilt_axis = measured_up.cross(reference_up)
            
            # Calculate the angle between the two up vectors using dot product
            dot_product = measured_up * reference_up
            # Clamp dot product to [-1, 1] to avoid floating-point errors
            dot_product = max(min(dot_product, 1.0), -1.0)
            tilt_angle = math.acos(dot_product)
            
            # Create a quaternion representing the tilt correction
            if tilt_axis.norm() > 0.001:  # Avoid issues with zero-length axis
                tilt_axis = tilt_axis.normalize()
                
                # Create a rotation quaternion to align measured up with reference up
                half_angle = tilt_angle * 0.5
                sin_half = math.sin(half_angle)
                
                tilt_quat = Quaternion(
                    math.cos(half_angle),
                    tilt_axis.x * sin_half,
                    tilt_axis.y * sin_half,
                    tilt_axis.z * sin_half
                )
                
                # Compute new orientation from accelerometer
                accel_orientation = self._quaternion_multiply(tilt_quat, gyro_orientation)
                accel_orientation.normalize()
            else:
                # No significant tilt axis - use gyro orientation
                accel_orientation = gyro_orientation
                
            # -------------------
            # COMPLEMENTARY FILTER
            # -------------------
            
            # Apply complementary filter to fuse gyro and accelerometer orientations
            # alpha controls the weight: higher alpha means more gyroscope influence
            self.orientation = self._quaternion_slerp(accel_orientation, gyro_orientation, self.alpha)
        else:
            # If acceleration is too low, just use gyroscope data
            self.orientation = gyro_orientation
        
        # Normalize final orientation
        self.orientation.normalize()
        
        return self.orientation
    
    def calibrate(self, sensor_data_list, num_samples=100):
        """
        Calibrate the gyroscope bias using a series of readings at rest.
        
        Args:
            sensor_data_list (List[SensorData]): List of sensor readings
            num_samples (int): Number of samples to use for calibration
        """
        if len(sensor_data_list) < num_samples:
            num_samples = len(sensor_data_list)
            
        gyro_sum = [0.0, 0.0, 0.0]
        
        for i in range(num_samples):
            gyro_sum[0] += sensor_data_list[i].gyroscope[0]
            gyro_sum[1] += sensor_data_list[i].gyroscope[1]
            gyro_sum[2] += sensor_data_list[i].gyroscope[2]
            
        self.gyro_bias = (
            gyro_sum[0] / num_samples,
            gyro_sum[1] / num_samples,
            gyro_sum[2] / num_samples
        )
        
        print(f"Calibrated gyro bias: {self.gyro_bias}")
    
    def _quaternion_conjugate(self, q):
        """Return the conjugate of a quaternion (inverses the imaginary parts)"""
        return Quaternion(q.w, -q.x, -q.y, -q.z)
    
    def _quaternion_multiply(self, q1, q2):
        """Multiply two quaternions (q1 * q2)"""
        return Quaternion(
            q1.w * q2.w - q1.x * q2.x - q1.y * q2.y - q1.z * q2.z,
            q1.w * q2.x + q1.x * q2.w + q1.y * q2.z - q1.z * q2.y,
            q1.w * q2.y - q1.x * q2.z + q1.y * q2.w + q1.z * q2.x,
            q1.w * q2.z + q1.x * q2.y - q1.y * q2.x + q1.z * q2.w
        )
    
    def _quaternion_slerp(self, q1, q2, t):
        """
        Spherical linear interpolation between quaternions.
        
        Args:
            q1: Starting quaternion
            q2: Ending quaternion
            t: Interpolation parameter (0-1), higher means more q2
        
        Returns:
            Interpolated quaternion
        """
        # Calculate the dot product
        dot = q1.w * q2.w + q1.x * q2.x + q1.y * q2.y + q1.z * q2.z
        
        # If quaternions are very close, just use linear interpolation
        if abs(dot) > 0.9995:
            result = Quaternion(
                q1.w * (1-t) + q2.w * t,
                q1.x * (1-t) + q2.x * t,
                q1.y * (1-t) + q2.y * t,
                q1.z * (1-t) + q2.z * t
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
            s1 * q1.w + s2 * q2.w,
            s1 * q1.x + s2 * q2.x,
            s1 * q1.y + s2 * q2.y,
            s1 * q1.z + s2 * q2.z
        )
        
        return result
    
    def get_euler_angles(self):
        """
        Convert current orientation quaternion to Euler angles.
        
        Returns:
            Tuple[float, float, float]: Roll, pitch, yaw in radians
        """
        # Roll (x-axis rotation)
        sinr_cosp = 2.0 * (self.orientation.w * self.orientation.x + 
                          self.orientation.y * self.orientation.z)
        cosr_cosp = 1.0 - 2.0 * (self.orientation.x * self.orientation.x + 
                               self.orientation.y * self.orientation.y)
        roll = math.atan2(sinr_cosp, cosr_cosp)
        
        # Pitch (y-axis rotation)
        sinp = 2.0 * (self.orientation.w * self.orientation.y - 
                     self.orientation.z * self.orientation.x)
        if abs(sinp) >= 1:
            pitch = math.copysign(math.pi / 2, sinp)  # Use 90 degrees if out of range
        else:
            pitch = math.asin(sinp)
            
        # Yaw (z-axis rotation)
        siny_cosp = 2.0 * (self.orientation.w * self.orientation.z + 
                          self.orientation.x * self.orientation.y)
        cosy_cosp = 1.0 - 2.0 * (self.orientation.y * self.orientation.y + 
                               self.orientation.z * self.orientation.z)
        yaw = math.atan2(siny_cosp, cosy_cosp)
        
        return roll, pitch, yaw