from vector import Vector
from math import sin, cos
import math
import pandas as pd
import numpy as np
from dataclasses import dataclass
from typing import List, Tuple

@dataclass
class SensorData:
    time: float
    gyroscope: Tuple[float, float, float]
    accelerometer: Tuple[float, float, float]
    magnetometer: Tuple[float, float, float]

@dataclass
class Quaternion:
    def __init__(self, w, x, y, z):
        self.w, self.x, self.y, self.z = w, x, y, z

    def normalize(self):
        norm = math.sqrt(self.w**2 + self.x**2 + self.y**2 + self.z**2)
        self.w /= norm
        self.x /= norm
        self.y /= norm
        self.z /= norm

    def __mul__(self, other):
        return Quaternion(
            self.w * other.w - self.x * other.x - self.y * other.y - self.z * other.z,
            self.w * other.x + self.x * other.w + self.y * other.z - self.z * other.y,
            self.w * other.y - self.x * other.z + self.y * other.w + self.z * other.x,
            self.w * other.z + self.x * other.y - self.y * other.x + self.z * other.w
        )

class SensorDataParser:
    def __init__(self, csv_path: str):
        """
        Initialize the parser with the path to the CSV file.
        
        Args:
            csv_path (str): Path to the sensor data CSV file
        """
        self.csv_path = csv_path
        self.data = None

    def convert_rotational_rate_to_radians_sec(self, degrees: float) -> float:
        return degrees * (np.pi / 180.0)
    
    def _euler_to_quaternion(self, roll: float, pitch: float, yaw: float) -> Quaternion:
        """
        Convert Euler angles to quaternion.
        
        Args:
            roll (float): Rotation around x-axis in radians
            pitch (float): Rotation around y-axis in radians
            yaw (float): Rotation around z-axis in radians
            
        Returns:
            Quaternion: Quaternion representation of the rotation
        """
        # Calculate half angles
        cr = np.cos(roll * 0.5)
        sr = np.sin(roll * 0.5)
        cp = np.cos(pitch * 0.5)
        sp = np.sin(pitch * 0.5)
        cy = np.cos(yaw * 0.5)
        sy = np.sin(yaw * 0.5)
        
        # Calculate quaternion components
        w = cr * cp * cy + sr * sp * sy
        x = sr * cp * cy - cr * sp * sy
        y = cr * sp * cy + sr * cp * sy
        z = cr * cp * sy - sr * sp * cy
        
        return Quaternion(w, x, y, z)
        
    def _quaternion_to_euler(self, q: Quaternion) -> Tuple[float, float, float]:
        sinr_cosp = 2.0 * (q.w * q.x + q.y * q.z)
        cosr_cosp = 1.0 - 2.0 * (q.x * q.x + q.y * q.y)
        roll = np.arctan2(sinr_cosp, cosr_cosp)
        
        # Pitch (rotation around Y axis)
        sinp = 2.0 * (q.w * q.y - q.z * q.x)
        if abs(sinp) >= 1:
            # Handle edge case when sinp = ±1 (gimbal lock)
            pitch = np.copysign(np.pi / 2, sinp)
        else:
            pitch = np.arcsin(sinp)
        
        # Yaw (rotation around Z axis)
        siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
        cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
        yaw = np.arctan2(siny_cosp, cosy_cosp)
        
        return (roll, pitch, yaw)
    
    def _quaternion_conjugate(self, q: Quaternion) -> Quaternion:
        return Quaternion(q.w, -q.x, -q.y, -q.z)
    
    def _quaternion_multiply(self, q1: Quaternion, q2: Quaternion) -> Quaternion:
        w = q1.w * q2.w - q1.x * q2.x - q1.y * q2.y - q1.z * q2.z
        x = q1.x * q2.w + q1.w * q2.x + q1.y * q2.z - q1.z * q2.y
        y = q1.y * q2.w + q1.w * q2.y + q1.z * q2.x - q1.x * q2.z
        z = q1.z * q2.w + q1.w * q2.z + q1.x * q2.y - q1.y * q2.x
        return Quaternion(w, x, y, z)

    def parse(self) -> List[SensorData]:
        """
        Parse the CSV file and return a list of SensorData objects.
        
        Returns:
            List[SensorData]: List of parsed sensor data entries
        """
        try:
            df = pd.read_csv(self.csv_path)
            df.columns = df.columns.str.strip()
            sensor_data_list = []
            
            for _, row in df.iterrows():
                sensor_data = SensorData(
                    time=row['time'],
                    gyroscope=(
                        self.convert_rotational_rate_to_radians_sec(row['gyroscope.X']),
                        self.convert_rotational_rate_to_radians_sec(row['gyroscope.Y']),
                        self.convert_rotational_rate_to_radians_sec(row['gyroscope.Z'])
                    ),
                    accelerometer=(
                        row['accelerometer.X'],
                        row['accelerometer.Y'],
                        row['accelerometer.Z']
                    ),
                    magnetometer=(
                        row['magnetometer.X'],
                        row['magnetometer.Y'],
                        row['magnetometer.Z']
                    )
                )
                sensor_data_list.append(sensor_data)
            
            self.data = sensor_data_list
            return sensor_data_list
        
        except Exception as e:
            raise Exception(f"Error parsing CSV file: {str(e)}")
    
    def get_sensor_stats(self) -> dict:
        """
        Calculate basic statistics for each sensor.
        
        Returns:
            dict: Dictionary containing mean, min, max values for each sensor
        """
        if self.data is None:
            raise Exception("Data not parsed yet. Call parse() first.")
            
        stats = {
            'gyroscope': {
                'mean': np.mean([d.gyroscope for d in self.data], axis=0),
                'min': np.min([d.gyroscope for d in self.data], axis=0),
                'max': np.max([d.gyroscope for d in self.data], axis=0)
            },
            'accelerometer': {
                'mean': np.mean([d.accelerometer for d in self.data], axis=0),
                'min': np.min([d.accelerometer for d in self.data], axis=0),
                'max': np.max([d.accelerometer for d in self.data], axis=0)
            },
            'magnetometer': {
                'mean': np.mean([d.magnetometer for d in self.data], axis=0),
                'min': np.min([d.magnetometer for d in self.data], axis=0),
                'max': np.max([d.magnetometer for d in self.data], axis=0)
            }
        }
        return stats
    
class Matrix4:
    """
    4x4 transformation matrix for 3D graphics operations.
    This allows us to properly handle model transformations in homogeneous coordinates.
    """
    def __init__(self):
        # Initialize as identity matrix
        self.data = [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0]
        ]
    
    @staticmethod
    def translation(x, y, z):
        """Creates a translation matrix"""
        m = Matrix4()
        m.data[0][3] = x
        m.data[1][3] = y
        m.data[2][3] = z
        return m
    
    @staticmethod
    def rotation_x(angle):
        m = Matrix4()
        c = cos(angle)
        s = sin(angle)
        m.data[1][1] = c
        m.data[1][2] = -s
        m.data[2][1] = s
        m.data[2][2] = c
        return m
    
    def rotation_y(angle):
        matrix = Matrix4()
        cos_a = math.cos(angle)
        sin_a = math.sin(angle)
        matrix.data[0][0] = cos_a
        matrix.data[0][2] = sin_a
        matrix.data[2][0] = -sin_a
        matrix.data[2][2] = cos_a
        return matrix
    
    @staticmethod
    def rotation_z(angle):
        matrix = Matrix4()
        cos_a = math.cos(angle)
        sin_a = math.sin(angle)
        matrix.data[0][0] = cos_a
        matrix.data[0][1] = -sin_a
        matrix.data[1][0] = sin_a
        matrix.data[1][1] = cos_a
        return matrix
    
    @staticmethod
    def scaling(s):
        """Creates a uniform scaling matrix"""
        m = Matrix4()
        m.data[0][0] = s
        m.data[1][1] = s
        m.data[2][2] = s
        return m
    
    def multiply(self, vec4):
        """
        Multiplies this matrix with a Vec4 to transform it.
        This is where the actual perspective transformation happens.
        """
        result = Vec4(0, 0, 0, 0)
        result.x = (self.data[0][0] * vec4.x + self.data[0][1] * vec4.y + 
                   self.data[0][2] * vec4.z + self.data[0][3] * vec4.w)
        result.y = (self.data[1][0] * vec4.x + self.data[1][1] * vec4.y + 
                   self.data[1][2] * vec4.z + self.data[1][3] * vec4.w)
        result.z = (self.data[2][0] * vec4.x + self.data[2][1] * vec4.y + 
                   self.data[2][2] * vec4.z + self.data[2][3] * vec4.w)
        result.w = (self.data[3][0] * vec4.x + self.data[3][1] * vec4.y + 
                   self.data[3][2] * vec4.z + self.data[3][3] * vec4.w)
        return result
    
    def multiply_matrix(self, other):
        """
        Multiplies this matrix with another matrix.
        This is used for combining transformations, like rotation and translation.
        """
        result = Matrix4()
        for i in range(4):
            for j in range(4):
                result.data[i][j] = sum(
                    self.data[i][k] * other.data[k][j] 
                    for k in range(4)
                )
        return result
    
    def multiply_vector(self, vector):
        """
        Multiplies this matrix with a vector.
        This is used for transforming individual points in 3D space.
        """
        x = (self.data[0][0] * vector.x + 
             self.data[0][1] * vector.y + 
             self.data[0][2] * vector.z + 
             self.data[0][3])
        
        y = (self.data[1][0] * vector.x + 
             self.data[1][1] * vector.y + 
             self.data[1][2] * vector.z + 
             self.data[1][3])
        
        z = (self.data[2][0] * vector.x + 
             self.data[2][1] * vector.y + 
             self.data[2][2] * vector.z + 
             self.data[2][3])
        
        w = (self.data[3][0] * vector.x + 
             self.data[3][1] * vector.y + 
             self.data[3][2] * vector.z + 
             self.data[3][3])
        
        return Vec4(x, y, z, w)
    
    @staticmethod
    def perspective(fov, aspect, near, far):
        """
        Creates a perspective projection matrix.
        
        Parameters:
            fov: Field of view in radians - controls how "wide" the camera sees
            aspect: Width/height ratio of the screen - prevents stretching
            near: Distance to near clipping plane - closest visible point
            far: Distance to far clipping plane - farthest visible point
            
        Returns:
            A Matrix4 configured for perspective projection
        """
        matrix = Matrix4()
        
        # Calculate scale based on field of view
        # This determines how much things shrink with distance
        f = 1.0 / math.tan(fov / 2)
        
        # Set up the perspective transformation
        matrix.data[0][0] = f / aspect  # Scale X by FOV and aspect ratio
        matrix.data[1][1] = f           # Scale Y by FOV
        
        # Handle depth (Z coordinate) transformation
        matrix.data[2][2] = (far + near) / (near - far)  # Scale Z
        matrix.data[2][3] = -1.0        # Enable perspective division
        matrix.data[3][2] = (2 * far * near) / (near - far)  # More Z scaling
        matrix.data[3][3] = 0.0         # Required for perspective division
        
        return matrix
    
# First, let's create a class for homogeneous coordinates and matrices
class Vec4:
    """
    A vector class for homogeneous coordinates (x, y, z, w).
    The w component is what enables perspective effects.
    """
    def __init__(self, x, y, z, w=1.0):
        self.x = x
        self.y = y
        self.z = z
        self.w = w
    
    def perspectiveDivide(self):
        """
        Performs perspective division to create the perspective effect.
        This is what makes distant objects appear smaller.
        """
        if self.w != 0:
            return Vector(
                self.x / self.w,
                self.y / self.w,
                self.z / self.w
            )
        return Vector(self.x, self.y, self.z)

class Model(object):
    def __init__(self, file):
        self.vertices = []
        self.faces = []
        self.scale = [1, 1, 1]
        self.rot = [0, 0, 0]
        self.trans = [0,0, 0]
        self.transform = Matrix4()

        # Read in the file
        f = open(file, 'r')
        for line in f:
            if line.startswith('#'): continue
            segments = line.split()
            if not segments: continue

            # Vertices
            if segments[0] == 'v':
                vertex = Vector(*[float(i) for i in segments[1:4]])
                self.vertices.append(vertex)

            # Faces
            elif segments[0] == 'f':
                # Support models that have faces with more than 3 points
                # Parse the face as a triangle fan
                for i in range(2, len(segments)-1):
                    corner1 = int(segments[1].split('/')[0])-1
                    corner2 = int(segments[i].split('/')[0])-1
                    corner3 = int(segments[i+1].split('/')[0])-1
                    self.faces.append([corner1, corner2, corner3])

    def normalizeGeometry(self):
        maxCoords = [0, 0, 0]

        for vertex in self.vertices:
            maxCoords[0] = max(abs(vertex.x), maxCoords[0])
            maxCoords[1] = max(abs(vertex.y), maxCoords[1])
            maxCoords[2] = max(abs(vertex.z), maxCoords[2])

        s = 1/max(maxCoords)
        # s=1
        for vertex in self.vertices:
            vertex.x = vertex.x * s
            vertex.y = vertex.y * s
            vertex.z = vertex.z * s

    def updateTransform(self):
        """
        Updates the model's transformation matrix by combining scale, rotation, and translation.
        The order of operations is important: first scale, then rotate, then translate.
        """
        # Start with scaling
        scale_matrix = Matrix4.scaling(self.scale[0])
        
        # Apply rotations
        rot_x = Matrix4.rotation_x(self.rot[0])
        rot_y = Matrix4.rotation_y(self.rot[1])
        rot_z = Matrix4.rotation_z(self.rot[2])
        
        # Combine rotations using matrix multiplication
        rotation = rot_z.multiply_matrix(rot_y.multiply_matrix(rot_x))
        
        # Apply translation
        trans = Matrix4.translation(*self.trans)
        
        # Combine all transformations
        # Order: first scale, then rotate, then translate
        self.transform = trans.multiply_matrix(rotation.multiply_matrix(scale_matrix))

    def getTransformedVertex(self, index):
        """
        Returns a vertex transformed by the model's current transformation matrix.
        This converts the vertex from model space to world space.
        """
        vertex = self.vertices[index]
        # Transform the vertex using our transformation matrix
        transformed = self.transform.multiply_vector(vertex)
        # Return the transformed vertex's x, y, z components
        return Vector(transformed.x, transformed.y, transformed.z)

    def normalizeGeometry(self):
        """
        Normalizes the model's geometry to fit in a unit cube while preserving proportions.
        Now updates the scale factor instead of modifying vertices directly.
        """
        maxCoords = [0, 0, 0]

        for vertex in self.vertices:
            maxCoords[0] = max(abs(vertex.x), maxCoords[0])
            maxCoords[1] = max(abs(vertex.y), maxCoords[1])
            maxCoords[2] = max(abs(vertex.z), maxCoords[2])

        # Calculate scaling factor
        s = 1/max(maxCoords)
        
        # Store scale instead of modifying vertices
        self.scale = [s, s, s]
        self.updateTransform()

    def setPosition(self, x, y, z):
        """Sets the model's position in world space"""
        self.trans = [x, y, z]
        self.updateTransform()

    def setRotation(self, x, y, z):
        """Sets the model's rotation in radians"""
        self.rot = [x, y, z]
        self.updateTransform()

    def setQuaternionRotation(self, quaternion):
        """
        Sets the model's rotation using a quaternion.
        
        Args:
            quaternion (Quaternion): Rotation quaternion
        """
        # Convert quaternion to Euler angles
        # Roll (x-axis rotation)
        sinr_cosp = 2.0 * (quaternion.w * quaternion.x + quaternion.y * quaternion.z)
        cosr_cosp = 1.0 - 2.0 * (quaternion.x * quaternion.x + quaternion.y * quaternion.y)
        roll = math.atan2(sinr_cosp, cosr_cosp)
        
        # Pitch (y-axis rotation)
        sinp = 2.0 * (quaternion.w * quaternion.y - quaternion.z * quaternion.x)
        if abs(sinp) >= 1:
            pitch = math.copysign(math.pi / 2, sinp)
        else:
            pitch = math.asin(sinp)
            
        # Yaw (z-axis rotation)
        siny_cosp = 2.0 * (quaternion.w * quaternion.z + quaternion.x * quaternion.y)
        cosy_cosp = 1.0 - 2.0 * (quaternion.y * quaternion.y + quaternion.z * quaternion.z)
        yaw = math.atan2(siny_cosp, cosy_cosp)
        
        # Set rotation using Euler angles
        self.rot = [roll, pitch, yaw]
        self.updateTransform()