# """ Module for reading a .obj file into a stored model,
# 	retrieving vertices, faces, properties of that model.
# 	Written using only the Python standard library.
# """

# from vector import Vector
# from math import sin, cos

# class Matrix4:
#     """
#     4x4 transformation matrix for 3D graphics operations.
#     This allows us to properly handle model transformations in homogeneous coordinates.
#     """
#     def __init__(self):
#         # Initialize as identity matrix
#         self.data = [
#             [1.0, 0.0, 0.0, 0.0],
#             [0.0, 1.0, 0.0, 0.0],
#             [0.0, 0.0, 1.0, 0.0],
#             [0.0, 0.0, 0.0, 1.0]
#         ]
    
#     @staticmethod
#     def translation(x, y, z):
#         """Creates a translation matrix"""
#         m = Matrix4()
#         m.data[0][3] = x
#         m.data[1][3] = y
#         m.data[2][3] = z
#         return m
    
#     @staticmethod
#     def rotation_x(angle):
#         """Creates a rotation matrix around X axis"""
#         m = Matrix4()
#         c = cos(angle)
#         s = sin(angle)
#         m.data[1][1] = c
#         m.data[1][2] = -s
#         m.data[2][1] = s
#         m.data[2][2] = c
#         return m
    
#     @staticmethod
#     def scaling(s):
#         """Creates a uniform scaling matrix"""
#         m = Matrix4()
#         m.data[0][0] = s
#         m.data[1][1] = s
#         m.data[2][2] = s
#         return m
    
#     def multiply(self, other):
#         """Multiplies this matrix with another matrix or a vector"""
#         result = Matrix4()
#         for i in range(4):
#             for j in range(4):
#                 result.data[i][j] = sum(
#                     self.data[i][k] * other.data[k][j] 
#                     for k in range(4)
#                 )
#         return result

# class Model(object):
#     def __init__(self, file):
#         self.vertices = []
#         self.faces = []
#         self.scale = [1, 1, 1]
#         self.rot = [0, 0, 0]
#         self.trans = [0,0, 0]
#         self.transform = Matrix4()

#         # Read in the file
#         f = open(file, 'r')
#         for line in f:
#             if line.startswith('#'): continue
#             segments = line.split()
#             if not segments: continue

#             # Vertices
#             if segments[0] == 'v':
#                 vertex = Vector(*[float(i) for i in segments[1:4]])
#                 self.vertices.append(vertex)

#             # Faces
#             elif segments[0] == 'f':
#                 # Support models that have faces with more than 3 points
#                 # Parse the face as a triangle fan
#                 for i in range(2, len(segments)-1):
#                     corner1 = int(segments[1].split('/')[0])-1
#                     corner2 = int(segments[i].split('/')[0])-1
#                     corner3 = int(segments[i+1].split('/')[0])-1
#                     self.faces.append([corner1, corner2, corner3])

#     def normalizeGeometry(self):
#         maxCoords = [0, 0, 0]

#         for vertex in self.vertices:
#             maxCoords[0] = max(abs(vertex.x), maxCoords[0])
#             maxCoords[1] = max(abs(vertex.y), maxCoords[1])
#             maxCoords[2] = max(abs(vertex.z), maxCoords[2])

#         s = 1/max(maxCoords)
#         # s=1
#         for vertex in self.vertices:
#             vertex.x = vertex.x * s
#             vertex.y = vertex.y * s
#             vertex.z = vertex.z * s

#     def updateTransform(self):
#         """
#         Updates the model's transformation matrix based on scale, rotation, and translation.
#         This combines all transformations into a single matrix for efficiency.
#         """
#         # Start with scaling
#         scale_matrix = Matrix4.scaling(self.scale[0])
        
#         # Apply rotations
#         rot_x = Matrix4.rotation_x(self.rot[0])
#         rot_y = Matrix4.rotation_x(self.rot[1])
#         rot_z = Matrix4.rotation_x(self.rot[2])
#         rotation = rot_z.multiply(rot_y.multiply(rot_x))
        
#         # Apply translation
#         trans = Matrix4.translation(*self.trans)
        
#         # Combine all transformations
#         self.transform = trans.multiply(rotation.multiply(scale_matrix))

#     def getTransformedVertex(self, index):
#         """
#         Returns a vertex transformed by the model's current transformation matrix.
#         This is used during rendering to get the final position of each vertex.
#         """
#         vertex = self.vertices[index]
#         # Convert to homogeneous coordinates (w=1)
#         v4 = [vertex.x, vertex.y, vertex.z, 1.0]
        
#         # Apply transformation
#         result = [0.0] * 4
#         for i in range(4):
#             result[i] = sum(self.transform.data[i][j] * v4[j] for j in range(4))
            
#         # Convert back to Vector (perspective division happens later in the pipeline)
#         return Vector(result[0], result[1], result[2])

#     def normalizeGeometry(self):
#         """
#         Normalizes the model's geometry to fit in a unit cube while preserving proportions.
#         Now updates the scale factor instead of modifying vertices directly.
#         """
#         maxCoords = [0, 0, 0]

#         for vertex in self.vertices:
#             maxCoords[0] = max(abs(vertex.x), maxCoords[0])
#             maxCoords[1] = max(abs(vertex.y), maxCoords[1])
#             maxCoords[2] = max(abs(vertex.z), maxCoords[2])

#         # Calculate scaling factor
#         s = 1/max(maxCoords)
        
#         # Store scale instead of modifying vertices
#         self.scale = [s, s, s]
#         self.updateTransform()

#     def setPosition(self, x, y, z):
#         """Sets the model's position in world space"""
#         self.trans = [x, y, z]
#         self.updateTransform()

#     def setRotation(self, x, y, z):
#         """Sets the model's rotation in radians"""
#         self.rot = [x, y, z]
#         self.updateTransform()


""" Module for reading a .obj file into a stored model,
	retrieving vertices, faces, properties of that model.
	Written using only the Python standard library.
"""

from vector import Vector

class Model(object):
	def __init__(self, file):
		self.vertices = []
		self.faces = []
		self.scale = [0, 0, 0]
		self.rot = [0, 0, 0]
		self.trans = [0,0, 0]

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
