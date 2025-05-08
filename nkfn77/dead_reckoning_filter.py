import math
import numpy as np
from vector import Vector
from quaternion import Quaternion
from collections import deque

class DeadReckoningFilter:
    def __init__(self, alpha=0.95, beta=0.05, mag_weight=0.02):
        self.orientation = Quaternion(1, 0, 0, 0)  # Identity quaternion
        self.position = Vector(0, 0, 0)  # Fixed position (pivot point)
        self.last_time = None
        
        self.gyro_bias = (0, 0, 0)
        self.accel_bias = (0, 0, 0)
        self.mag_bias = (0, 0, 0)
        self.accel_scale = 1.0
        self.mag_scale = 1.0
        
        self.alpha = alpha  # Gyro weight
        self.beta = beta    # Accelerometer correction weight
        self.mag_weight = mag_weight  # Magnetometer correction weight
        
        self.rotation_rate = Vector(0, 0, 0)  # Angular velocity
        self.is_rotating = False
        self.reference_gravity = (0, 0, 1)  # Expected gravity direction (Z-up)
        
        self.gyro_buffer = deque(maxlen=10)
        self.accel_buffer = deque(maxlen=10)
        
        self.reference_mag_field = None
        
        self.rotation_threshold = 0.1  # rad/s
        self.stationary_threshold = 0.05  # rad/s

    def update(self, sensor_data):
        """
        Update orientation using gyroscope, accelerometer and magnetometer.
        Optimized for a headset rotating around a fixed pivot point.
        """
        if self.last_time is None:
            self.last_time = sensor_data.time
            self._init_reference_magnetic_field(sensor_data.magnetometer)
            return self.orientation

        dt = sensor_data.time - self.last_time
        self.last_time = sensor_data.time
        
        if dt <= 0 or dt > 0.1:
            dt = 0.01  # Use reasonable default
        
        # ------------------- DETECT ROTATION STATE -------------------
        self._update_rotation_state(sensor_data)
        
        # ------------------- GYROSCOPE INTEGRATION -------------------
        gyro_x = sensor_data.gyroscope[0] - self.gyro_bias[0]
        gyro_y = sensor_data.gyroscope[1] - self.gyro_bias[1]
        gyro_z = sensor_data.gyroscope[2] - self.gyro_bias[2]
        
        self.rotation_rate = Vector(gyro_x, gyro_y, gyro_z)
        
        rotation_quat = Quaternion(0, gyro_x, gyro_y, gyro_z)
        
        q_dot = self._quaternion_multiply(self.orientation, rotation_quat)
        q_dot.w *= 0.5
        q_dot.x *= 0.5
        q_dot.y *= 0.5
        q_dot.z *= 0.5
        
        gyro_orientation = Quaternion(
            self.orientation.w + q_dot.w * dt,
            self.orientation.x + q_dot.x * dt,
            self.orientation.y + q_dot.y * dt,
            self.orientation.z + q_dot.z * dt
        )
        gyro_orientation.normalize()
        
        # ------------------- ACCELEROMETER TILT CORRECTION -------------------
        accel_x, accel_y, accel_z = sensor_data.accelerometer
        
        # Apply bias and scale corrections
        accel_x = (accel_x - self.accel_bias[0]) * self.accel_scale
        accel_y = (accel_y - self.accel_bias[1]) * self.accel_scale
        accel_z = (accel_z - self.accel_bias[2]) * self.accel_scale
        
        accel_magnitude = math.sqrt(accel_x**2 + accel_y**2 + accel_z**2)
        
        adaptive_alpha = self._get_adaptive_alpha(accel_magnitude)
        
        accel_orientation = gyro_orientation  # Default to gyro if acceleration is too low
        
        if accel_magnitude > 0.1:  # Only apply correction if acceleration is significant
            accel_normalized_x = accel_x / accel_magnitude
            accel_normalized_y = accel_y / accel_magnitude
            accel_normalized_z = accel_z / accel_magnitude
            
            accel_body = Quaternion(0, accel_normalized_x, accel_normalized_y, accel_normalized_z)
            orientation_conjugate = self._quaternion_conjugate(gyro_orientation)
            accel_world = self._quaternion_multiply(
                self._quaternion_multiply(gyro_orientation, accel_body),
                orientation_conjugate
            )
            
            reference_up = Vector(0, 0, 1)  # Z-up coordinate system
            
            measured_up = Vector(accel_world.x, accel_world.y, accel_world.z).normalize() * -1
            
            tilt_axis = measured_up.cross(reference_up)
            
            dot_product = measured_up * reference_up
            dot_product = max(min(dot_product, 1.0), -1.0)  # Clamp to prevent numerical errors
            tilt_angle = math.acos(dot_product)
            
            if tilt_axis.norm() > 0.001:
                tilt_axis = tilt_axis.normalize()
                
                correction_rate = self.beta
                if self.is_rotating:
                    correction_rate *= 0.5 
                
                half_angle = tilt_angle * correction_rate
                sin_half = math.sin(half_angle)
                
                tilt_quat = Quaternion(
                    math.cos(half_angle),
                    tilt_axis.x * sin_half,
                    tilt_axis.y * sin_half,
                    tilt_axis.z * sin_half
                )
                
                # Apply tilt correction
                accel_orientation = self._quaternion_multiply(tilt_quat, gyro_orientation)
                accel_orientation.normalize()
        
        # ------------------- MAGNETOMETER HEADING CORRECTION -------------------
        mag_orientation = accel_orientation  
        
        mag_x, mag_y, mag_z = sensor_data.magnetometer
        
        mag_x = (mag_x - self.mag_bias[0]) * self.mag_scale
        mag_y = (mag_y - self.mag_bias[1]) * self.mag_scale
        mag_z = (mag_z - self.mag_bias[2]) * self.mag_scale
        
        mag_magnitude = math.sqrt(mag_x**2 + mag_y**2 + mag_z**2)
        
        if mag_magnitude > 0.1 and self.reference_mag_field:
            mag_normalized_x = mag_x / mag_magnitude
            mag_normalized_y = mag_y / mag_magnitude
            mag_normalized_z = mag_z / mag_magnitude
            
            mag_body = Quaternion(0, mag_normalized_x, mag_normalized_y, mag_normalized_z)
            mag_world = self._quaternion_multiply(
                self._quaternion_multiply(accel_orientation, mag_body),
                self._quaternion_conjugate(accel_orientation)
            )
            
            mag_horizontal = Vector(mag_world.x, mag_world.y, 0).normalize()
            ref_horizontal = Vector(self.reference_mag_field[0], self.reference_mag_field[1], 0).normalize()
            
            heading_correction_axis = mag_horizontal.cross(ref_horizontal)
            
            heading_dot = mag_horizontal * ref_horizontal
            heading_dot = max(min(heading_dot, 1.0), -1.0)  # Clamp
            heading_angle = math.acos(heading_dot)
            
            if heading_correction_axis.norm() > 0.001:
                correction_rate = self.mag_weight
                if self.is_rotating:
                    correction_rate *= 0.3  # Reduce mag correction when rotating quickly
                
                heading_half_angle = heading_angle * correction_rate
                heading_sin_half = math.sin(heading_half_angle)
                
                heading_quat = Quaternion(
                    math.cos(heading_half_angle),
                    0,  # Only rotate around Z axis (yaw)
                    0,
                    heading_sin_half if heading_correction_axis.z >= 0 else -heading_sin_half
                )
                
                # Apply heading correction
                mag_orientation = self._quaternion_multiply(heading_quat, accel_orientation)
                mag_orientation.normalize()
        
        # ------------------- COMBINE ORIENTATION ESTIMATES -------------------
        self.orientation = self._quaternion_slerp(mag_orientation, gyro_orientation, adaptive_alpha)
        self.orientation.normalize()
        
        return self.orientation

    def _init_reference_magnetic_field(self, mag_data):
        mag_x, mag_y, mag_z = mag_data
        mag_magnitude = math.sqrt(mag_x**2 + mag_y**2 + mag_z**2)
        
        if mag_magnitude > 0.1:
            self.reference_mag_field = (
                mag_x / mag_magnitude,
                mag_y / mag_magnitude,
                mag_z / mag_magnitude
            )
        else:
            # Default to magnetic north if readings are too weak
            self.reference_mag_field = (1, 0, 0)

    def _get_adaptive_alpha(self, accel_magnitude):
        accel_deviation = abs(accel_magnitude - 9.81)
        
        if self.is_rotating:
            # During rotation, trust gyro more
            if accel_deviation > 1.0:
                return min(0.98, self.alpha + 0.1)  # High acceleration during rotation
            else:
                return self.alpha  # Normal rotation
        else:
            # When stationary, trust accelerometer more for drift correction
            if accel_deviation < 0.5:
                return max(0.9, self.alpha - 0.15)  # Clean acceleration when stationary
            else:
                return self.alpha  # Keep default alpha
    
    def _update_rotation_state(self, sensor_data):
        self.gyro_buffer.append(sensor_data.gyroscope)
        
        gyro_x, gyro_y, gyro_z = sensor_data.gyroscope
        current_rotation = math.sqrt(gyro_x**2 + gyro_y**2 + gyro_z**2)
        
        if len(self.gyro_buffer) >= 5:
            avg_rotation = 0
            for gyro in self.gyro_buffer:
                avg_rotation += math.sqrt(gyro[0]**2 + gyro[1]**2 + gyro[2]**2)
            avg_rotation /= len(self.gyro_buffer)
            
            if self.is_rotating:
                self.is_rotating = avg_rotation > self.stationary_threshold
            else:
                self.is_rotating = avg_rotation > self.rotation_threshold
        else:
            self.is_rotating = current_rotation > self.rotation_threshold

    def calibrate(self, sensor_data_list, num_samples=100):
        """Calibrate sensor biases during a stationary period"""
        if len(sensor_data_list) < num_samples:
            num_samples = len(sensor_data_list)
            print(f"Warning: Only {num_samples} samples available for calibration")
        
        gyro_sum = [0.0, 0.0, 0.0]
        accel_sum = [0.0, 0.0, 0.0]
        mag_sum = [0.0, 0.0, 0.0]
        
        # Calculate mean values during stationary period
        for i in range(num_samples):
            # Gyroscope
            gyro_sum[0] += sensor_data_list[i].gyroscope[0]
            gyro_sum[1] += sensor_data_list[i].gyroscope[1]
            gyro_sum[2] += sensor_data_list[i].gyroscope[2]
            
            # Accelerometer
            accel_sum[0] += sensor_data_list[i].accelerometer[0]
            accel_sum[1] += sensor_data_list[i].accelerometer[1]
            accel_sum[2] += sensor_data_list[i].accelerometer[2]
            
            # Magnetometer
            mag_sum[0] += sensor_data_list[i].magnetometer[0]
            mag_sum[1] += sensor_data_list[i].magnetometer[1]
            mag_sum[2] += sensor_data_list[i].magnetometer[2]
        
        # Calculate gyroscope bias
        self.gyro_bias = (
            gyro_sum[0] / num_samples,
            gyro_sum[1] / num_samples,
            gyro_sum[2] / num_samples
        )
        
        # Calculate average accelerometer reading
        avg_accel = (
            accel_sum[0] / num_samples,
            accel_sum[1] / num_samples,
            accel_sum[2] / num_samples
        )
        
        # Calculate magnitude of average acceleration
        avg_accel_magnitude = math.sqrt(
            avg_accel[0]**2 + avg_accel[1]**2 + avg_accel[2]**2
        )
        
        # Calculate accelerometer scale factor
        self.accel_scale = 9.81 / avg_accel_magnitude if avg_accel_magnitude > 0.1 else 1.0
        
        # Calculate normalized gravity direction from average readings
        normalized_gravity = (
            avg_accel[0] / avg_accel_magnitude if avg_accel_magnitude > 0.1 else 0.0,
            avg_accel[1] / avg_accel_magnitude if avg_accel_magnitude > 0.1 else 0.0,
            avg_accel[2] / avg_accel_magnitude if avg_accel_magnitude > 0.1 else 1.0
        )
        
        # Store gravity direction for reference
        self.reference_gravity = normalized_gravity
        
        # Calculate accelerometer bias
        expected_gravity = (
            normalized_gravity[0] * 9.81,
            normalized_gravity[1] * 9.81,
            normalized_gravity[2] * 9.81
        )
        
        self.accel_bias = (
            (avg_accel[0] * self.accel_scale) - expected_gravity[0],
            (avg_accel[1] * self.accel_scale) - expected_gravity[1],
            (avg_accel[2] * self.accel_scale) - expected_gravity[2]
        )
        
        # Calculate average magnetometer reading and set reference
        avg_mag = (
            mag_sum[0] / num_samples,
            mag_sum[1] / num_samples,
            mag_sum[2] / num_samples
        )
        
        avg_mag_magnitude = math.sqrt(
            avg_mag[0]**2 + avg_mag[1]**2 + avg_mag[2]**2
        )
        
        if avg_mag_magnitude > 0.1:
            self.reference_mag_field = (
                avg_mag[0] / avg_mag_magnitude,
                avg_mag[1] / avg_mag_magnitude,
                avg_mag[2] / avg_mag_magnitude
            )
        
        print(f"Calibrated gyro bias: {self.gyro_bias}")
        print(f"Calibrated accel bias: {self.accel_bias}")
        print(f"Reference gravity direction: {self.reference_gravity}")
        print(f"Reference magnetic field direction: {self.reference_mag_field}")
        
    def _quaternion_conjugate(self, q):
        return Quaternion(q.w, -q.x, -q.y, -q.z)
    
    def _quaternion_multiply(self, q1, q2):
        return Quaternion(
            q1.w * q2.w - q1.x * q2.x - q1.y * q2.y - q1.z * q2.z,
            q1.w * q2.x + q1.x * q2.w + q1.y * q2.z - q1.z * q2.y,
            q1.w * q2.y - q1.x * q2.z + q1.y * q2.w + q1.z * q2.x,
            q1.w * q2.z + q1.x * q2.y - q1.y * q2.x + q1.z * q2.w
        )
    
    def _quaternion_slerp(self, q1, q2, t):
        dot = q1.w * q2.w + q1.x * q2.x + q1.y * q2.y + q1.z * q2.z
        
        if abs(dot) > 0.9995:
            result = Quaternion(
                q1.w * (1-t) + q2.w * t,
                q1.x * (1-t) + q2.x * t,
                q1.y * (1-t) + q2.y * t,
                q1.z * (1-t) + q2.z * t
            )
            result.normalize()
            return result
        
        dot = max(min(dot, 1.0), -1.0)
        theta_0 = math.acos(dot)
        sin_theta_0 = math.sin(theta_0)
        
        theta = theta_0 * t
        sin_theta = math.sin(theta)
        
        s1 = math.sin(theta_0 - theta) / sin_theta_0
        s2 = math.sin(theta) / sin_theta_0
        
        result = Quaternion(
            s1 * q1.w + s2 * q2.w,
            s1 * q1.x + s2 * q2.x,
            s1 * q1.y + s2 * q2.y,
            s1 * q1.z + s2 * q2.z
        )
        
        return result
    
    def get_euler_angles(self):
        sinr_cosp = 2.0 * (self.orientation.w * self.orientation.x + 
                          self.orientation.y * self.orientation.z)
        cosr_cosp = 1.0 - 2.0 * (self.orientation.x * self.orientation.x + 
                               self.orientation.y * self.orientation.y)
        roll = math.atan2(sinr_cosp, cosr_cosp)
        
        sinp = 2.0 * (self.orientation.w * self.orientation.y - 
                     self.orientation.z * self.orientation.x)
        if abs(sinp) >= 1:
            pitch = math.copysign(math.pi / 2, sinp)
        else:
            pitch = math.asin(sinp)
            
        siny_cosp = 2.0 * (self.orientation.w * self.orientation.z + 
                          self.orientation.x * self.orientation.y)
        cosy_cosp = 1.0 - 2.0 * (self.orientation.y * self.orientation.y + 
                               self.orientation.z * self.orientation.z)
        yaw = math.atan2(siny_cosp, cosy_cosp)
        
        return roll, pitch, yaw