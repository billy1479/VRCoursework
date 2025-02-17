""" Module for reading a .obj file into a stored model,
	retrieving vertices, faces, properties of that model.
	Written using only the Python standard library.
"""

from vector import Vector
from math import sin, cos
import math

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
        rot_y = Matrix4.rotation_x(self.rot[1])
        rot_z = Matrix4.rotation_x(self.rot[2])
        
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
