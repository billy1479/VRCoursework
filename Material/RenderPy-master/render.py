from image import Image, Color
from model import Model
from model import Matrix4
from model import Vec4
import math
from shape import Point, Line, Triangle
from vector import Vector
import pygame
import numpy as np

pygame.init()

width = 512
height = 512
image = Image(width, height, Color(255, 255, 255, 255))

screen = pygame.display.set_mode((width, height))
pygame.display.set_caption("3D Renderer")

buffer_surface = pygame.Surface((width, height))

# Init z-buffer
zBuffer = [-float('inf')] * width * height

# Load the model
model = Model('data/headset.obj')
model.normalizeGeometry()
model.setPosition(0, 0, -12)
# model.setRotation(90, 0, 90)

def getOrthographicProjection(x, y, z):
	# Convert vertex from world space to screen space
	# by dropping the z-coordinate (Orthographic projection)
	screenX = int((x+1.0)*width/2.0)
	screenY = int((y+1.0)*height/2.0)

	return screenX, screenY

def getPerspectiveProjection(x, y, z, width, height):
    # Set up perspective parameters
    fov = math.pi / 3.0  # 60-degree field of view
    aspect = width / height
    near = 0.1     # Near clipping plane
    far = 100.0    # Far clipping plane
    
    # Create the perspective matrix
    perspective_matrix = Matrix4.perspective(fov, aspect, near, far)
    
    # Create a vector in homogeneous coordinates
    point = Vec4(x, y, z, 1.0)
    
    # Apply perspective transformation
    projected = perspective_matrix.multiply(point)
    
    # Perform perspective division
    normalized = projected.perspectiveDivide()
    
    # Convert to screen coordinates
    screenX = int((normalized.x + 1.0) * width / 2.0)
    screenY = int((normalized.y + 1.0) * height / 2.0)
    
    return screenX, screenY

def getVertexNormal(vertIndex, faceNormalsByVertex):
	# Compute vertex normals by averaging the normals of adjacent faces
	normal = Vector(0, 0, 0)
	for adjNormal in faceNormalsByVertex[vertIndex]:
		normal = normal + adjNormal

	return normal / len(faceNormalsByVertex[vertIndex])

def update_display(image):
    """
    Updates the display with the current state of the frame buffer.
    This function accounts for the Image class's specific buffer format
    where each row starts with a null byte followed by RGBA values.
    """
    # Convert the image buffer to a Pygame surface
    pixel_array = pygame.surfarray.pixels3d(buffer_surface)
    
    for y in range(height):
        for x in range(width):
            # Calculate the index in the buffer, accounting for the row format
            # Each row has a null byte at the start, so we need to add y to account for these bytes
            flipY = (height - y - 1)  # Flip Y coordinate to match Image class convention
            index = (flipY * width + x) * 4 + flipY + 1  # +1 for the null byte at start of row
            
            # Extract RGB values from the buffer (ignore alpha)
            r = image.buffer[index]
            g = image.buffer[index + 1]
            b = image.buffer[index + 2]
            
            # Update the pixel array
            pixel_array[x, y] = (r, g, b)
    
    del pixel_array  # Release the surface lock
    
    # Blit the buffer to the screen
    screen.blit(buffer_surface, (0, 0))
    pygame.display.flip()

    # Handle events
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
            return False
    return True


# Calculate face normals
faceNormals = {}
for face in model.faces:
	# p0, p1, p2 = [model.vertices[i] for i in face]
	p0 = model.getTransformedVertex(face[0])
	p1 = model.getTransformedVertex(face[1])
	p2 = model.getTransformedVertex(face[2])
	faceNormal = (p2-p0).cross(p1-p0).normalize()

	for i in face:
		if not i in faceNormals:
			faceNormals[i] = []

		faceNormals[i].append(faceNormal)

# Calculate vertex normals
vertexNormals = []
for vertIndex in range(len(model.vertices)):
	vertNorm = getVertexNormal(vertIndex, faceNormals)
	vertexNormals.append(vertNorm)

# Render the image iterating through faces
running = True
face_count = 0

for face in model.faces:
	if not running:
		break

	# p0, p1, p2 = [model.vertices[i] for i in face]
	p0 = model.getTransformedVertex(face[0])
	p1 = model.getTransformedVertex(face[1])
	p2 = model.getTransformedVertex(face[2])
	n0, n1, n2 = [vertexNormals[i] for i in face]

	# Define the light direction
	lightDir = Vector(0, 0, -1)

	# Set to true if face should be culled
	cull = False

	# Transform vertices and calculate lighting intensity per vertex
	transformedPoints = []
	for p, n in zip([p0, p1, p2], [n0, n1, n2]):
		intensity = n * lightDir

		# Intensity < 0 means light is shining through the back of the face
		# In this case, don't draw the face at all ("back-face culling")
		if intensity < 0:
			cull = True # Back face culling is disabled in this version
			
		screenX, screenY = getPerspectiveProjection(p.x, p.y, p.z, width, height)
		transformedPoints.append(Point(screenX, screenY, p.z, Color(intensity*255, intensity*255, intensity*255, 255)))

	if not cull:
		Triangle(transformedPoints[0], transformedPoints[1], transformedPoints[2]).draw_faster(image, zBuffer)

	face_count += 1
	if face[0] % 10 == 0:  # Adjust this number to control update frequency
		running = update_display(image)

while running:
	running = update_display(image)

pygame.quit()
image.saveAsPNG("image.png")
