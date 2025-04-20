from model import SensorDataParser, Quaternion
from vector import Vector
import math

class CameraTracker:
    """
    Tracks camera movement based on IMU sensor data.
    Provides camera orientation and position updates for the renderer.
    """
    def __init__(self, imu_data_path):
        """
        Initialize the camera tracker with IMU data.
        
        Args:
            imu_data_path: Path to the IMU dataset CSV file
        """
        self.parser = SensorDataParser(imu_data_path)
        self.sensor_data = self.parser.parse()
        self.current_index = 0
        self.total_frames = len(self.sensor_data)
        
        # Camera state
        self.position = Vector(0, 1.7, -5)  # Default eye-level height
        self.orientation = Quaternion(1, 0, 0, 0)  # Identity quaternion (no rotation)
        
        # Calibrate the gyroscope to remove bias
        self._calibrate_gyro()
        
        print(f"Loaded {self.total_frames} frames of IMU data")
        
    def _calibrate_gyro(self, num_samples=100):
        """Calibrate gyroscope by averaging the first few readings"""
        samples = min(num_samples, len(self.sensor_data))
        
        gyro_sum = [0.0, 0.0, 0.0]
        for i in range(samples):
            gyro_sum[0] += self.sensor_data[i].gyroscope[0]
            gyro_sum[1] += self.sensor_data[i].gyroscope[1]
            gyro_sum[2] += self.sensor_data[i].gyroscope[2]
            
        self.gyro_bias = (
            gyro_sum[0] / samples,
            gyro_sum[1] / samples,
            gyro_sum[2] / samples
        )
        
        print(f"Gyroscope bias calibrated: {self.gyro_bias}")
    
    def update(self, delta_time=1/60):
        """
        Update camera position and orientation based on current IMU data.
        Advances to the next data frame.
        
        Args:
            delta_time: Time elapsed since last update (for interpolation)
            
        Returns:
            Tuple containing (position, forward, up) vectors for the camera
        """
        if self.current_index >= self.total_frames:
            # Loop back to beginning when we reach the end
            self.current_index = 0
            
        # Get current sensor reading
        sensor_data = self.sensor_data[self.current_index]
        
        # Extract gyroscope data (apply bias correction)
        gyro_x = sensor_data.gyroscope[0] - self.gyro_bias[0]
        gyro_y = sensor_data.gyroscope[1] - self.gyro_bias[1]
        gyro_z = sensor_data.gyroscope[2] - self.gyro_bias[2]
        
        # Integrate gyroscope data to update orientation
        # First convert to quaternion rate of change
        q_dot = Quaternion(
            0.5 * (-self.orientation.x * gyro_x - self.orientation.y * gyro_y - self.orientation.z * gyro_z),
            0.5 * (self.orientation.w * gyro_x + self.orientation.y * gyro_z - self.orientation.z * gyro_y),
            0.5 * (self.orientation.w * gyro_y - self.orientation.x * gyro_z + self.orientation.z * gyro_x),
            0.5 * (self.orientation.w * gyro_z + self.orientation.x * gyro_y - self.orientation.y * gyro_x)
        )
        
        # Apply integration step
        self.orientation = Quaternion(
            self.orientation.w + q_dot.w * delta_time,
            self.orientation.x + q_dot.x * delta_time,
            self.orientation.y + q_dot.y * delta_time,
            self.orientation.z + q_dot.z * delta_time
        )
        
        # Normalize quaternion to prevent drift
        magnitude = math.sqrt(
            self.orientation.w**2 + 
            self.orientation.x**2 + 
            self.orientation.y**2 + 
            self.orientation.z**2
        )
        
        self.orientation.w /= magnitude
        self.orientation.x /= magnitude
        self.orientation.y /= magnitude
        self.orientation.z /= magnitude
        
        # Compute forward and up vectors from quaternion
        forward, up = self._get_camera_vectors()
        
        # Advance to next frame
        self.current_index += 1
        
        return self.position, forward, up
    
    def _get_camera_vectors(self):
        """
        Calculate forward and up vectors from quaternion orientation.
        
        Returns:
            Tuple containing (forward, up) vectors
        """
        # Default vectors (before rotation)
        default_forward = Vector(0, 0, -1)  # Looking down the negative Z axis
        default_up = Vector(0, 1, 0)       # Y is up
        
        # Apply quaternion rotation to get current forward vector
        forward_x = (-2 * (self.orientation.x * self.orientation.z - self.orientation.w * self.orientation.y))
        forward_y = (-2 * (self.orientation.y * self.orientation.z + self.orientation.w * self.orientation.x))
        forward_z = (-1 + 2 * (self.orientation.x * self.orientation.x + self.orientation.y * self.orientation.y))
        
        # Apply quaternion rotation to get current up vector
        up_x = (2 * (self.orientation.x * self.orientation.y - self.orientation.w * self.orientation.z))
        up_y = (1 - 2 * (self.orientation.x * self.orientation.x + self.orientation.z * self.orientation.z))
        up_z = (2 * (self.orientation.y * self.orientation.z + self.orientation.w * self.orientation.x))
        
        # Normalize vectors
        forward = Vector(forward_x, forward_y, forward_z).normalize()
        up = Vector(up_x, up_y, up_z).normalize()
        
        return forward, up
    
    def get_euler_angles(self):
        """
        Convert current orientation quaternion to Euler angles.
        Useful for debugging.
        
        Returns:
            Tuple of (roll, pitch, yaw) in radians
        """
        # Roll (x-axis rotation)
        sinr_cosp = 2 * (self.orientation.w * self.orientation.x + 
                        self.orientation.y * self.orientation.z)
        cosr_cosp = 1 - 2 * (self.orientation.x * self.orientation.x + 
                            self.orientation.y * self.orientation.y)
        roll = math.atan2(sinr_cosp, cosr_cosp)
        
        # Pitch (y-axis rotation)
        sinp = 2 * (self.orientation.w * self.orientation.y - 
                    self.orientation.z * self.orientation.x)
        if abs(sinp) >= 1:
            pitch = math.copysign(math.pi / 2, sinp)  # Use 90 degrees if out of range
        else:
            pitch = math.asin(sinp)
            
        # Yaw (z-axis rotation)
        siny_cosp = 2 * (self.orientation.w * self.orientation.z + 
                        self.orientation.x * self.orientation.y)
        cosy_cosp = 1 - 2 * (self.orientation.y * self.orientation.y + 
                            self.orientation.z * self.orientation.z)
        yaw = math.atan2(siny_cosp, cosy_cosp)
        
        return roll, pitch, yaw
    
    def look_at_matrix(self):
        """
        Generate a look-at matrix for the camera.
        
        Returns:
            Look-at matrix as a 4x4 transformation matrix
        """
        from model import Matrix4
        
        forward, up = self._get_camera_vectors()
        
        # Calculate right vector using cross product
        right = up.cross(forward).normalize()
        
        # Recalculate up vector to ensure orthogonality
        up = forward.cross(right).normalize()
        
        # Create look-at matrix
        matrix = Matrix4()
        
        # First three rows contain the right, up, and forward vectors
        matrix.data[0][0] = right.x
        matrix.data[0][1] = right.y
        matrix.data[0][2] = right.z
        
        matrix.data[1][0] = up.x
        matrix.data[1][1] = up.y
        matrix.data[1][2] = up.z
        
        matrix.data[2][0] = forward.x
        matrix.data[2][1] = forward.y
        matrix.data[2][2] = forward.z
        
        # Translation component
        matrix.data[0][3] = -right * self.position
        matrix.data[1][3] = -up * self.position
        matrix.data[2][3] = -forward * self.position
        
        return matrix