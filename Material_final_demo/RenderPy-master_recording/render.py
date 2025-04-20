from matplotlib import pyplot as plt
from image import Image, Color
from model import DeadReckoningFilter, Model
from model import Matrix4
from model import Vec4
from model import SensorData
from model import SensorDataParser
from model import CollisionObject
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
model = Model('./data/headset.obj')
model.normalizeGeometry()
model.setPosition(0, 0, -12)
# model.setRotation(90, 0, 90)

def load_csv_data(file_path):
	parser = SensorDataParser(file_path)
	sensor_data = parser.parse()

	print("Loaded sensor data from CSV file")

	print("Number of sensor data entries: ", len(sensor_data))
	print("First sensor data entry: ", sensor_data[0])

	return sensor_data

def getOrthographicProjection(x, y, z):
	# Convert vertex from world space to screen space
	# by dropping the z-coordinate (Orthographic projection)
	screenX = int((x+1.0)*width/2.0)
	screenY = int((y+1.0)*height/2.0)

	return screenX, screenY

def getPerspectiveProjection_old(x, y, z, width, height):
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

# Dead reckoning filter test
def test_alpha_values(csv_contents, model, image_width, image_height):
    """
    Test different alpha values for the complementary filter and compare results.
    
    Args:
        csv_contents: List of sensor data readings
        model: 3D model to visualize
        image_width, image_height: Dimensions for rendering
    """
    # Define a range of alpha values to test
    alpha_values = [0.5, 0.7, 0.9, 0.95, 0.98, 0.99]
    
    # For each alpha value, we'll record orientation data over time
    results = {alpha: [] for alpha in alpha_values}
    
    # We'll use a subset of the data for quicker testing
    test_duration = min(len(csv_contents), 1000)  # Limit to 1000 samples
    
    print("Testing alpha values...")
    
    # Test each alpha value
    for alpha in alpha_values:
        print(f"\nTesting alpha = {alpha}")
        
        # Create a new filter with this alpha value
        dr_filter = DeadReckoningFilter(alpha=alpha)
        
        # Calibrate using first 100 samples (assuming the device is at rest)
        dr_filter.calibrate(csv_contents[:100])
        
        # Process all sensor data
        for i in range(test_duration):
            sensor_data = csv_contents[i]
            
            # Update the filter and record orientation
            position, orientation = dr_filter.update(sensor_data)
            
            # Convert quaternion to Euler angles (easier to interpret)
            roll, pitch, yaw = dr_filter.get_euler_angles()
            
            # Record results
            results[alpha].append({
                'time': sensor_data.time,
                'roll': math.degrees(roll),
                'pitch': math.degrees(pitch),
                'yaw': math.degrees(yaw)
            })
            
            # Print progress
            if i % 100 == 0:
                print(f"  Processed {i}/{test_duration} samples")
    
    # Analyze the results
    analyze_alpha_results(results)
    
    # Visualize results (save charts for each alpha)
    for alpha in alpha_values:
        visualize_filter_results(results[alpha], f"alpha_{alpha:.2f}")
    
    return results

def analyze_alpha_results(results):
    """Analyze and print statistics about different alpha values"""
    for alpha, data in results.items():
        # Calculate standard deviation of orientation (higher means more variation)
        roll_values = [entry['roll'] for entry in data]
        pitch_values = [entry['pitch'] for entry in data]
        yaw_values = [entry['yaw'] for entry in data]
        
        roll_std = np.std(roll_values)
        pitch_std = np.std(pitch_values)
        yaw_std = np.std(yaw_values)
        
        # Calculate average rate of change (smoothness measure)
        roll_changes = [abs(roll_values[i] - roll_values[i-1]) for i in range(1, len(roll_values))]
        pitch_changes = [abs(pitch_values[i] - pitch_values[i-1]) for i in range(1, len(pitch_values))]
        yaw_changes = [abs(yaw_values[i] - yaw_values[i-1]) for i in range(1, len(yaw_values))]
        
        avg_roll_change = sum(roll_changes) / len(roll_changes)
        avg_pitch_change = sum(pitch_changes) / len(pitch_changes)
        avg_yaw_change = sum(yaw_changes) / len(yaw_changes)
        
        print(f"\nResults for alpha = {alpha}:")
        print(f"  Standard deviations: Roll={roll_std:.2f}°, Pitch={pitch_std:.2f}°, Yaw={yaw_std:.2f}°")
        print(f"  Average changes per step: Roll={avg_roll_change:.4f}°, Pitch={avg_pitch_change:.4f}°, Yaw={avg_yaw_change:.4f}°")

def visualize_filter_results(data, filename_prefix):
    """Create visualization of filter results and save to file"""
    # Extract data
    times = [entry['time'] for entry in data]
    rolls = [entry['roll'] for entry in data]
    pitches = [entry['pitch'] for entry in data]
    yaws = [entry['yaw'] for entry in data]
    
    # Create figure with subplots
    plt.figure(figsize=(12, 8))
    
    # Plot roll
    plt.subplot(3, 1, 1)
    plt.plot(times, rolls, 'r-')
    plt.title('Roll over Time')
    plt.ylabel('Degrees')
    plt.grid(True)
    
    # Plot pitch
    plt.subplot(3, 1, 2)
    plt.plot(times, pitches, 'g-')
    plt.title('Pitch over Time')
    plt.ylabel('Degrees')
    plt.grid(True)
    
    # Plot yaw
    plt.subplot(3, 1, 3)
    plt.plot(times, yaws, 'b-')
    plt.title('Yaw over Time')
    plt.xlabel('Time (seconds)')
    plt.ylabel('Degrees')
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(f"{filename_prefix}_orientation.png")
    plt.close()

def problem_3():
    csv_contents = load_csv_data("../IMUData.csv")

    test_result = test_alpha_values(csv_contents, model, width, height)

    print(test_result)

    best_alpha = 0.99

    dr_filter = DeadReckoningFilter(alpha=best_alpha)
    # Calibrate using first 100 samples (assuming the device is at rest)
    dr_filter.calibrate(csv_contents[:100])

    # Create pygame clock for timing
    clock = pygame.time.Clock()
    running = True
    current_data_index = 0

    # Main rendering loop
    while running and current_data_index < len(csv_contents):
        # Handle timing
        delta_time = clock.tick(60) / 1000.0
        
        # Get current sensor data and update filter
        current_sensor_data = csv_contents[current_data_index]
        position, orientation = dr_filter.update(current_sensor_data)
        
        # Convert quaternion to Euler angles
        roll, pitch, yaw = dr_filter.get_euler_angles()
        
        # Update model rotation
        model.setRotation(roll, pitch, yaw)
        
        # Reset image and z-buffer for new frame
        image = Image(width, height, Color(255, 255, 255, 255))
        zBuffer = [-float('inf')] * width * height
        
        # Calculate face normals
        faceNormals = {}
        for face in model.faces:
            p0 = model.getTransformedVertex(face[0])
            p1 = model.getTransformedVertex(face[1])
            p2 = model.getTransformedVertex(face[2])
            faceNormal = (p2-p0).cross(p1-p0).normalize()

            for i in face:
                if i not in faceNormals:
                    faceNormals[i] = []

                faceNormals[i].append(faceNormal)

        # Calculate vertex normals
        vertexNormals = []
        for vertIndex in range(len(model.vertices)):
            vertNorm = getVertexNormal(vertIndex, faceNormals)
            vertexNormals.append(vertNorm)

        # Render all faces for this frame
        for face in model.faces:
            p0 = model.getTransformedVertex(face[0])
            p1 = model.getTransformedVertex(face[1])
            p2 = model.getTransformedVertex(face[2])
            n0, n1, n2 = [vertexNormals[i] for i in face]

            # Define the light direction
            lightDir = Vector(0, 0, -1)

            # Set to true if face should be culled
            cull = False

            # Transform vertices and calculate lighting
            transformedPoints = []
            for p, n in zip([p0, p1, p2], [n0, n1, n2]):
                intensity = n * lightDir

                if intensity < 0:
                    cull = True
                    
                screenX, screenY = getPerspectiveProjection(p.x, p.y, p.z, width, height)
                
                transformedPoints.append(Point(screenX, screenY, p.z, Color(intensity*255, intensity*255, intensity*255, 255)))

            if not cull:
                Triangle(transformedPoints[0], transformedPoints[1], transformedPoints[2]).draw_faster(image, zBuffer)

        # Update display
        running = update_display(image)
        
        # Increment data index
        current_data_index += 1

    # Cleanup
    pygame.quit()
    image.saveAsPNG("image.png")

def problem_4_1():
    def setup_scene_old():
        """Create initial scene with multiple headsets"""
        headsets = []
        
        # Create several headsets with different positions and velocities
        positions = [
            Vector(-5, 1, -10),  # Left
            Vector(5, 1, -10),   # Right
            Vector(0, 1, -8),    # Center front
            Vector(0, 1, -12),   # Center back
        ]
        
        velocities = [
            Vector(2, 0, 0),    # Moving right
            Vector(-2, 0, 0),   # Moving left
            Vector(0, 0, 2),    # Moving forward
            Vector(0, 0, -2),   # Moving backward
        ]
        
        # Create collision objects for each headset
        for pos, vel in zip(positions, velocities):
            # Load a new model instance for each headset
            model = Model('data/headset.obj')
            model.normalizeGeometry()
            model.setPosition(pos.x, pos.y, pos.z)
            
            # Create collision object
            # Radius is chosen based on model size - adjust as needed
            headset = CollisionObject(model, pos, vel, radius=1.0)
            headsets.append(headset)
        
        return headsets

    def setup_scene():
        """
        Create initial scene with multiple headsets arranged in patterns that will result 
        in interesting collisions.
        """
        headsets = []
        
        # Create several patterns of headsets that will interact in interesting ways
        
        # Pattern 1: Circle of headsets moving inward
        num_circle = 8  # Number of headsets in the circle
        circle_radius = 20  # Distance from center
        for i in range(num_circle):
            angle = (i / num_circle) * 2 * math.pi
            # Position headsets in a circle
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
            
            model = Model('data/headset.obj')
            model.normalizeGeometry()
            model.setPosition(pos.x, pos.y, pos.z)
            headsets.append(CollisionObject(model, pos, vel, radius=1.0))

        # Pattern 2: Stack of headsets dropping from above
        stack_height = 4
        for i in range(stack_height):
            pos = Vector(0, 5 + (i * 2), -10)  # Stack them vertically
            vel = Vector(0, -1, 0)  # Falling down
            
            model = Model('data/headset.obj')
            model.normalizeGeometry()
            model.setPosition(pos.x, pos.y, pos.z)
            headsets.append(CollisionObject(model, pos, vel, radius=1.0))

        # Pattern 3: Line of headsets moving sideways
        num_line = 6
        for i in range(num_line):
            # Alternate between left and right sides
            x_pos = -10 if i % 2 == 0 else 10
            z_offset = -15 + (i * 2)  # Space them along z-axis
            pos = Vector(x_pos, 1, z_offset)
            # Move towards opposite side
            vel = Vector(1 if x_pos < 0 else -1, 0, 0)
            
            model = Model('data/headset.obj')
            model.normalizeGeometry()
            model.setPosition(pos.x, pos.y, pos.z)
            headsets.append(CollisionObject(model, pos, vel, radius=1.0))

        # Pattern 4: "Billiards break" pattern
        triangle_size = 3  # Number of rows in triangle
        start_z = -5
        for row in range(triangle_size):
            for col in range(row + 1):
                pos = Vector(
                    (col - row/2) * 2,  # Center the triangle
                    1,
                    start_z + row * 2
                )
                # These headsets start stationary
                vel = Vector(0, 0, 0)
                
                model = Model('data/headset.obj')
                model.normalizeGeometry()
                model.setPosition(pos.x, pos.y, pos.z)
                headsets.append(CollisionObject(model, pos, vel, radius=1.0))
        
        # Add a "cue ball" headset to hit the triangle
        pos = Vector(0, 1, -15)  # Position behind the triangle
        vel = Vector(0, 0, 4)    # Moving forward to hit the triangle
        
        model = Model('data/headset.obj')
        model.normalizeGeometry()
        model.setPosition(pos.x, pos.y, pos.z)
        headsets.append(CollisionObject(model, pos, vel, radius=1.0))

        return headsets

    def update_physics_old(headsets, dt):
        """Update physics and handle collisions for all objects"""
        # Clear previous collision records
        for headset in headsets:
            headset.clear_collision_history()
        
        # Check all pairs of headsets for collisions
        for i in range(len(headsets)):
            for j in range(i + 1, len(headsets)):
                if headsets[i].check_collision(headsets[j]):
                    headsets[i].resolve_collision(headsets[j])
        
        # Update positions
        for headset in headsets:
            headset.update(dt)

    def update_physics(headsets, dt):
        """
        Optimized physics update with frame rate independence.
        The simulation will now run at a consistent speed regardless of frame rate.
        """
        # Use a fixed time step for physics
        fixed_dt = 1/60  # Target 60 physics updates per second
        
        # Accumulate any leftover time
        update_physics.accumulator = getattr(update_physics, 'accumulator', 0) + dt
        
        # Run physics updates with fixed timestep
        while update_physics.accumulator >= fixed_dt:
            # Clear collision records only when needed
            for headset in headsets:
                headset.clear_collision_history()
            
            # Use numpy for vectorized gravity calculation
            gravity_vector = np.array([0, -9.81 * fixed_dt, 0])
            
            # Batch update velocities
            for headset in headsets:
                headset.velocity = Vector(
                    headset.velocity.x + gravity_vector[0],
                    headset.velocity.y + gravity_vector[1],
                    headset.velocity.z + gravity_vector[2]
                )
            
            # Optimize collision detection with spatial partitioning
            # Only check collisions between objects that are close enough
            for i in range(len(headsets)):
                # Create a bounding box for quick rejection
                for j in range(i + 1, len(headsets)):
                    # Quick distance check before detailed collision
                    dx = headsets[i].position.x - headsets[j].position.x
                    dy = headsets[i].position.y - headsets[j].position.y
                    dz = headsets[i].position.z - headsets[j].position.z
                    
                    # Square of the distance - faster than square root
                    dist_sq = dx*dx + dy*dy + dz*dz
                    
                    # Only do detailed collision if objects are close enough
                    max_dist = (headsets[i].radius + headsets[j].radius) * 1.5  # Add some margin
                    if dist_sq < max_dist * max_dist:
                        if headsets[i].check_collision(headsets[j]):
                            headsets[i].resolve_collision(headsets[j])
            
            # Update positions
            for headset in headsets:
                headset.update(fixed_dt)
            
            update_physics.accumulator -= fixed_dt

    # Modify the main rendering loop
    def main_old():
        # Initialize pygame and create window
        pygame.init()
        width = 512
        height = 512
        screen = pygame.display.set_mode((width, height))
        pygame.display.set_caption("3D Renderer with Collisions")
        
        # Create scene with multiple headsets
        headsets = setup_scene()

        debug_visualization = True
        
        # Main rendering loop
        running = True
        clock = pygame.time.Clock()

        while running:
            # Handle timing
            dt = clock.tick(60) / 1000.0  # Convert to seconds
            
            # Update physics
            update_physics(headsets, dt)
            
            # Clear screen
            image = Image(width, height, Color(255, 255, 255, 255))
            zBuffer = [-float('inf')] * width * height
            
            # Render all headsets
            for headset in headsets:
                model = headset.model
                
                # Calculate face normals
                faceNormals = {}
                for face in model.faces:
                    p0 = model.getTransformedVertex(face[0])
                    p1 = model.getTransformedVertex(face[1])
                    p2 = model.getTransformedVertex(face[2])
                    faceNormal = (p2-p0).cross(p1-p0).normalize()
                    
                    for i in face:
                        if i not in faceNormals:
                            faceNormals[i] = []
                        faceNormals[i].append(faceNormal)
                
                # Calculate vertex normals
                vertexNormals = []
                for vertIndex in range(len(model.vertices)):
                    vertNorm = getVertexNormal(vertIndex, faceNormals)
                    vertexNormals.append(vertNorm)
                
                # Render faces
                for face in model.faces:
                    p0 = model.getTransformedVertex(face[0])
                    p1 = model.getTransformedVertex(face[1])
                    p2 = model.getTransformedVertex(face[2])
                    n0, n1, n2 = [vertexNormals[i] for i in face]
                    
                    lightDir = Vector(0, 0, -1)
                    cull = False
                    
                    transformedPoints = []
                    for p, n in zip([p0, p1, p2], [n0, n1, n2]):
                        intensity = n * lightDir
                        
                        if intensity < 0:
                            cull = True
                        
                        screenX, screenY = getPerspectiveProjection(p.x, p.y, p.z, width, height)
                        transformedPoints.append(Point(screenX, screenY, p.z, 
                                                    Color(intensity*255, intensity*255, intensity*255, 255)))
                    
                    if not cull:
                        Triangle(transformedPoints[0], transformedPoints[1], 
                                transformedPoints[2]).draw_faster(image, zBuffer)
            
            # Optional: Draw debug spheres to visualize collision boundaries
            if debug_visualization:
                for headset in headsets:
                    # Calculate screen position of sphere center
                    center_screen_x, center_screen_y = getPerspectiveProjection(
                        headset.position.x, 
                        headset.position.y, 
                        headset.position.z, 
                        width, 
                        height
                    )
                    
                    # Draw a circle representing the collision sphere
                    pygame.draw.circle(
                        screen,
                        (255, 0, 0),  # Red color
                        (int(center_screen_x), int(center_screen_y)),  # Convert to integers
                        int(headset.radius * 20),  # Scale radius for visibility
                        1  # Line width
                    )
            def draw_debug_sphere(obj):
                # Project sphere boundaries to screen space
                center = obj.position
                screenX, screenY = getPerspectiveProjection(center.x, center.y, center.z, width, height)
                
                # Calculate projected radius (this is a simple approximation)
                radius_point = center + Vector(obj.radius, 0, 0)
                radiusX, radiusY = getPerspectiveProjection(radius_point.x, radius_point.y, radius_point.z, width, height)
                radius_pixels = abs(radiusX - screenX)
                
                # Draw circle
                pygame.draw.circle(screen, (255, 0, 0), (screenX, screenY), radius_pixels, 1)
            
            # Update display
            running = update_display(image)
            
            # Handle events
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
        pygame.quit()

    def main():
        # Initialize pygame and create window
        pygame.init()
        width = 512
        height = 512
        screen = pygame.display.set_mode((width, height))
        pygame.display.set_caption("3D Renderer with Collisions")
        
        # Create a background surface that we'll keep persistent
        background = pygame.Surface(screen.get_size())
        background = background.convert()
        background.fill((255, 255, 255))  # White background
        
        # Create scene with multiple headsets
        headsets = setup_scene()
        
        # Add a debug visualization flag
        debug_visualization = True
        
        # Create a separate surface for the 3D rendering
        render_surface = pygame.Surface((width, height))
        
        running = True
        clock = pygame.time.Clock()
        
        while running:
            # Handle timing with a maximum time step to prevent large jumps
            dt = min(clock.tick(60) / 1000.0, 0.1)  # Cap at 0.1 seconds
            
            try:
                # Handle events
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        running = False
                    elif event.type == pygame.KEYDOWN:
                        if event.key == pygame.K_ESCAPE:
                            running = False
                        elif event.key == pygame.K_d:
                            debug_visualization = not debug_visualization
                
                # Update physics
                update_physics(headsets, dt)
                
                # Start with a fresh render surface each frame
                render_surface.fill((255, 255, 255))
                
                # Create new image for 3D rendering
                image = Image(width, height, Color(255, 255, 255, 255))
                zBuffer = [-float('inf')] * width * height
                
                # Render all headsets
                for headset in headsets:
                    model = headset.model
                    
                    # Calculate face normals
                    faceNormals = {}
                    for face in model.faces:
                        p0 = model.getTransformedVertex(face[0])
                        p1 = model.getTransformedVertex(face[1])
                        p2 = model.getTransformedVertex(face[2])
                        faceNormal = (p2-p0).cross(p1-p0).normalize()
                        
                        for i in face:
                            if i not in faceNormals:
                                faceNormals[i] = []
                            faceNormals[i].append(faceNormal)
                    
                    # Calculate vertex normals
                    vertexNormals = []
                    for vertIndex in range(len(model.vertices)):
                        vertNorm = getVertexNormal(vertIndex, faceNormals)
                        vertexNormals.append(vertNorm)
                    
                    # Render each face
                    for face in model.faces:
                        p0 = model.getTransformedVertex(face[0])
                        p1 = model.getTransformedVertex(face[1])
                        p2 = model.getTransformedVertex(face[2])
                        n0, n1, n2 = [vertexNormals[i] for i in face]
                        
                        lightDir = Vector(0, 0, -1)
                        cull = False
                        
                        transformedPoints = []
                        for p, n in zip([p0, p1, p2], [n0, n1, n2]):
                            intensity = n * lightDir
                            if intensity < 0:
                                cull = True
                            screenX, screenY = getPerspectiveProjection(p.x, p.y, p.z, width, height)
                            transformedPoints.append(Point(screenX, screenY, p.z, 
                                                        Color(intensity*255, intensity*255, intensity*255, 255)))
                        
                        if not cull:
                            Triangle(transformedPoints[0], transformedPoints[1], 
                                    transformedPoints[2]).draw_faster(image, zBuffer)
                
                # Draw the rendered image to the screen
                if update_display(image) == False:
                    running = False
                    continue
                
                # Add debug visualization on top
                if debug_visualization:
                    for headset in headsets:
                        # Calculate screen position of sphere center
                        center_screen_x, center_screen_y = getPerspectiveProjection(
                            headset.position.x, 
                            headset.position.y, 
                            headset.position.z, 
                            width, 
                            height
                        )
                        
                        # Ensure coordinates are valid integers
                        center_screen_x = int(max(0, min(width-1, center_screen_x)))
                        center_screen_y = int(max(0, min(height-1, center_screen_y)))
                        
                        # Draw collision sphere
                        try:
                            pygame.draw.circle(
                                screen,
                                (255, 0, 0),  # Red color
                                (center_screen_x, center_screen_y),
                                int(headset.radius * 40),  # Scale radius for visibility
                                1  # Line width
                            )
                        except Exception as e:
                            print(f"Error drawing debug circle: {e}")
                
                # Update the display
                pygame.display.flip()
                
            except Exception as e:
                print(f"Error in main loop: {e}")
                continue
        
        # Clean up
        pygame.quit()

    main()

## Problem 4.2

import pygame
from image import Image, Color
from model import Model, CollisionObject
from vector import Vector
from shape import Triangle, Point
import math

class SingleObjectSimulation:
    """
    Class for setting up and running a simulation of a single object
    with friction in 3D space.
    """
    def __init__(self, width=512, height=512):
        """Initialize the simulation environment"""
        self.width = width
        self.height = height
        self.image = Image(width, height, Color(255, 255, 255, 255))
        
        # Initialize pygame 
        pygame.init()
        self.screen = pygame.display.set_mode((width, height))
        pygame.display.set_caption("Single Object Friction Simulation")
        self.buffer_surface = pygame.Surface((width, height))
        
        # Initialize z-buffer for rendering
        self.zBuffer = [-float('inf')] * width * height
        
        # Create the object
        self.object = self.create_single_object()
        
        # Physics settings
        self.friction_coefficient = 0.98  # Higher value = less friction
        self.gravity = -9.81  # Gravity acceleration (m/s²)
        self.floor_y = 0  # Y position of the floor
        
        # Debug settings
        self.show_debug = True
        self.debug_trail = []  # Store positions for trail rendering
        self.max_trail_length = 100
        
    def create_single_object(self):
        """Create a single object with initial position and velocity"""
        # Load the model
        model = Model('data/headset.obj')  # Adjust path as needed
        model.normalizeGeometry()
        
        # Initial position and velocity
        position = Vector(0, 1, -10)  # Centered, slightly above floor
        
        # Give it an initial velocity along the x-z plane
        velocity = Vector(3, 0, 1)  # Moving diagonally
        
        # Set the model position
        model.setPosition(position.x, position.y, position.z)
        
        # Create and return a collision object
        return CollisionObject(model, position, velocity, radius=1.0)
    
    def update_physics(self, dt):
        """
        Update physics for the object, applying friction when in contact with the floor.
        
        Args:
            dt: Time step in seconds
        """
        # Store the current position for the debug trail
        if self.show_debug:
            self.debug_trail.append((
                self.object.position.x, 
                self.object.position.y, 
                self.object.position.z
            ))
            # Limit trail length
            if len(self.debug_trail) > self.max_trail_length:
                self.debug_trail.pop(0)
        
        # Apply gravity to vertical velocity
        self.object.velocity.y += self.gravity * dt
        
        # Store original position before movement
        original_position = Vector(
            self.object.position.x, 
            self.object.position.y, 
            self.object.position.z
        )
        
        # Apply velocity to position
        self.object.position = self.object.position + (self.object.velocity * dt)
        
        # Check if object is on or near the floor
        is_on_floor = self.object.position.y - self.object.radius <= self.floor_y + 0.01
        
        if is_on_floor:
            # Ensure the object doesn't go below the floor
            self.object.position.y = self.object.radius + self.floor_y
            
            # Bounce with energy loss if hitting the floor
            if self.object.velocity.y < 0:
                self.object.velocity.y = abs(self.object.velocity.y) * 0.5  # 50% bounce
            
            # Apply friction to horizontal velocity components
            horizontal_speed_squared = (
                self.object.velocity.x**2 + 
                self.object.velocity.z**2
            )
            
            # Only apply friction if moving horizontally
            if horizontal_speed_squared > 0.001:
                # Calculate friction based on speed (optional dynamic friction)
                speed = math.sqrt(horizontal_speed_squared)
                friction = self.calculate_friction(speed)
                
                # Apply friction by reducing horizontal velocity
                self.object.velocity.x *= friction
                self.object.velocity.z *= friction
                
                # Stop completely if very slow (avoid endless tiny sliding)
                if horizontal_speed_squared < 0.05:
                    self.object.velocity.x = 0
                    self.object.velocity.z = 0
        
        # Update the model's position to match the physics object
        self.object.model.setPosition(
            self.object.position.x,
            self.object.position.y,
            self.object.position.z
        )
        
        # Print debug info occasionally
        if pygame.time.get_ticks() % 500 < 20:  # About every 500ms
            print(f"Position: ({self.object.position.x:.2f}, {self.object.position.y:.2f}, {self.object.position.z:.2f})")
            print(f"Velocity: ({self.object.velocity.x:.2f}, {self.object.velocity.y:.2f}, {self.object.velocity.z:.2f})")
            print(f"On floor: {is_on_floor}")
            print("-" * 30)
    
    def calculate_friction(self, speed):
        """
        Calculate friction coefficient based on speed.
        
        Args:
            speed: Current horizontal speed
            
        Returns:
            Friction coefficient (higher = less friction)
        """
        # Basic friction - constant coefficient
        basic_friction = self.friction_coefficient
        
        # Optional: Dynamic friction that varies with speed
        # Uncomment and adjust for dynamic friction behavior
        """
        if speed > 5.0:
            return 0.99  # Less friction at high speeds
        elif speed > 2.0:
            return 0.97  # Medium friction
        else:
            return 0.95  # More friction at low speeds
        """
        
        return basic_friction
    
    def perspective_projection(self, x, y, z):
        """
        Convert 3D world coordinates to 2D screen coordinates using perspective projection.
        
        Args:
            x, y, z: World coordinates
            
        Returns:
            (screenX, screenY): Screen coordinates
        """
        # Simple perspective projection
        fov = math.pi / 3.0  # 60-degree field of view
        aspect = self.width / self.height
        near = 0.1
        far = 100.0
        
        # Convert to normalized device coordinates
        f = 1.0 / math.tan(fov / 2)
        nf = 1.0 / (near - far)
        
        # Perspective division
        depth = z
        if abs(depth) < 0.1:  # Avoid division by very small numbers
            depth = -0.1
            
        x_normalized = x / -depth
        y_normalized = y / -depth
        
        # Scale to screen coordinates
        screenX = int((x_normalized + 1.0) * self.width / 2.0)
        screenY = int((y_normalized + 1.0) * self.height / 2.0)
        
        return screenX, screenY
    
    def render_scene(self):
        """Render the current state of the scene"""
        # Clear image and z-buffer
        self.image = Image(self.width, self.height, Color(255, 255, 255, 255))
        self.zBuffer = [-float('inf')] * self.width * self.height
        
        # Render the model
        model = self.object.model
        
        # Calculate face normals for lighting
        faceNormals = {}
        for face in model.faces:
            p0 = model.getTransformedVertex(face[0])
            p1 = model.getTransformedVertex(face[1])
            p2 = model.getTransformedVertex(face[2])
            faceNormal = (p2-p0).cross(p1-p0).normalize()
            
            for i in face:
                if i not in faceNormals:
                    faceNormals[i] = []
                faceNormals[i].append(faceNormal)
        
        # Calculate vertex normals by averaging adjacent face normals
        vertexNormals = []
        for vertIndex in range(len(model.vertices)):
            if vertIndex in faceNormals:
                normal = Vector(0, 0, 0)
                for adjNormal in faceNormals[vertIndex]:
                    normal = normal + adjNormal
                vertexNormals.append(normal / len(faceNormals[vertIndex]))
            else:
                vertexNormals.append(Vector(0, 1, 0))  # Default normal
        
        # Render all faces
        lightDir = Vector(0, 0, -1)  # Light from camera position
        
        for face in model.faces:
            p0 = model.getTransformedVertex(face[0])
            p1 = model.getTransformedVertex(face[1])
            p2 = model.getTransformedVertex(face[2])
            n0, n1, n2 = [vertexNormals[i] for i in face]
            
            # Back-face culling
            cull = False
            
            # Transform vertices and calculate lighting
            transformedPoints = []
            for p, n in zip([p0, p1, p2], [n0, n1, n2]):
                # Calculate lighting intensity
                intensity = n * lightDir
                
                if intensity < 0:
                    cull = True
                    
                # Project to screen coordinates
                screenX, screenY = self.perspective_projection(p.x, p.y, p.z)
                
                # Create point with lighting color
                transformedPoints.append(Point(
                    screenX, screenY, p.z, 
                    Color(intensity*255, intensity*255, intensity*255, 255)
                ))
            
            # Draw the triangle if not culled
            if not cull:
                Triangle(
                    transformedPoints[0], 
                    transformedPoints[1], 
                    transformedPoints[2]
                ).draw_faster(self.image, self.zBuffer)
        
        # Draw floor grid (optional)
        self.render_floor_grid()
        
        # Draw debug visualization if enabled
        if self.show_debug:
            self.render_debug_info()
        
        # Update display
        self.update_display()
    
    def render_floor_grid(self):
        """Render a simple grid on the floor for better visualization"""
        grid_size = 20
        grid_step = 2
        
        # Draw grid lines
        for x in range(-grid_size, grid_size + 1, grid_step):
            for z in range(-grid_size, grid_size + 1, grid_step):
                # Draw horizontal lines (constant z)
                if z != grid_size:
                    p1x, p1y = self.perspective_projection(x, 0, -z)
                    p2x, p2y = self.perspective_projection(x, 0, -(z + grid_step))
                    
                    # Only draw if within screen bounds
                    if (0 <= p1x < self.width and 0 <= p1y < self.height and
                        0 <= p2x < self.width and 0 <= p2y < self.height):
                        pygame.draw.line(
                            self.screen, 
                            (200, 200, 200),  # Light gray
                            (p1x, p1y), 
                            (p2x, p2y), 
                            1
                        )
                
                # Draw vertical lines (constant x)
                if x != grid_size:
                    p1x, p1y = self.perspective_projection(x, 0, -z)
                    p2x, p2y = self.perspective_projection(x + grid_step, 0, -z)
                    
                    # Only draw if within screen bounds
                    if (0 <= p1x < self.width and 0 <= p1y < self.height and
                        0 <= p2x < self.width and 0 <= p2y < self.height):
                        pygame.draw.line(
                            self.screen, 
                            (200, 200, 200),  # Light gray
                            (p1x, p1y), 
                            (p2x, p2y), 
                            1
                        )
    
    def render_debug_info(self):
        """Render debug information including velocity and trail"""
        # Draw collision sphere
        center_x, center_y = self.perspective_projection(
            self.object.position.x,
            self.object.position.y,
            self.object.position.z
        )
        
        # Draw collision boundary
        radius_point = Vector(
            self.object.position.x + self.object.radius,
            self.object.position.y,
            self.object.position.z
        )
        radius_x, radius_y = self.perspective_projection(
            radius_point.x,
            radius_point.y,
            radius_point.z
        )
        
        # Calculate radius in screen space
        radius_pixels = abs(radius_x - center_x)
        
        # Only draw if the center is within bounds
        if (0 <= center_x < self.width and 0 <= center_y < self.height):
            pygame.draw.circle(
                self.screen,
                (255, 0, 0),  # Red
                (center_x, center_y),
                radius_pixels,
                1  # Line width
            )
        
        # Draw velocity vector
        vel_end_pos = Vector(
            self.object.position.x + self.object.velocity.x,
            self.object.position.y + self.object.velocity.y,
            self.object.position.z + self.object.velocity.z
        )
        
        vel_end_x, vel_end_y = self.perspective_projection(
            vel_end_pos.x,
            vel_end_pos.y,
            vel_end_pos.z
        )
        
        # Only draw if both points are on screen
        if (0 <= center_x < self.width and 0 <= center_y < self.height and
            0 <= vel_end_x < self.width and 0 <= vel_end_y < self.height):
            pygame.draw.line(
                self.screen,
                (0, 255, 0),  # Green
                (center_x, center_y),
                (vel_end_x, vel_end_y),
                2  # Line width
            )
        
        # Draw trail
        last_valid_point = None
        
        for i, position in enumerate(self.debug_trail):
            # Decrease opacity for older points
            alpha = int((i / len(self.debug_trail)) * 255)
            
            screen_x, screen_y = self.perspective_projection(
                position[0], position[1], position[2]
            )
            
            # Check if point is valid
            if (0 <= screen_x < self.width and 
                0 <= screen_y < self.height):
                
                # Draw point
                pygame.draw.circle(
                    self.screen,
                    (0, 0, 255, alpha),  # Blue with alpha
                    (screen_x, screen_y),
                    2  # Radius
                )
                
                # Connect with line if we have a previous point
                if last_valid_point:
                    pygame.draw.line(
                        self.screen,
                        (0, 0, 255, alpha),  # Blue with alpha
                        last_valid_point,
                        (screen_x, screen_y),
                        1  # Line width
                    )
                
                last_valid_point = (screen_x, screen_y)
    
    def update_display(self):
        """Update the display with the current render buffer"""
        # Convert the image buffer to pygame surface
        pixel_array = pygame.surfarray.pixels3d(self.buffer_surface)
        
        for y in range(self.height):
            for x in range(self.width):
                # Calculate the index in the buffer
                # Each row has a null byte at the start, plus 4 bytes per pixel
                flipY = (self.height - y - 1)  # Flip Y to match Image class convention
                index = (flipY * self.width + x) * 4 + flipY + 1  # +1 for null byte
                
                # Extract RGB values from the buffer (ignore alpha)
                r = self.image.buffer[index]
                g = self.image.buffer[index + 1]
                b = self.image.buffer[index + 2]
                
                # Update the pixel array
                pixel_array[x, y] = (r, g, b)
        
        del pixel_array  # Release the surface lock
        
        # Blit the buffer to the screen
        self.screen.blit(self.buffer_surface, (0, 0))
        pygame.display.flip()
    
    def run_simulation(self):
        """Run the main simulation loop"""
        clock = pygame.time.Clock()
        running = True
        
        # Print instructions
        print("Single Object Friction Simulation")
        print("--------------------------------")
        print("Press D to toggle debug visualization")
        print("Press R to reset the object position")
        print("Press +/- to increase/decrease friction")
        print("Press ESC or close the window to exit")
        print("\nInitial friction coefficient:", self.friction_coefficient)
        
        while running:
            # Handle timing
            dt = min(clock.tick(60) / 1000.0, 0.1)  # Cap at 0.1s to prevent jumps
            
            # Handle events
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        running = False
                    
                    elif event.key == pygame.K_d:
                        # Toggle debug visualization
                        self.show_debug = not self.show_debug
                        print("Debug visualization:", "ON" if self.show_debug else "OFF")
                    
                    elif event.key == pygame.K_r:
                        # Reset object position and velocity
                        self.object = self.create_single_object()
                        self.debug_trail = []
                        print("Object reset")
                    
                    elif event.key == pygame.K_EQUALS or event.key == pygame.K_PLUS:
                        # Increase friction coefficient (less friction)
                        self.friction_coefficient = min(0.999, self.friction_coefficient + 0.01)
                        print(f"Friction coefficient: {self.friction_coefficient:.3f}")
                    
                    elif event.key == pygame.K_MINUS:
                        # Decrease friction coefficient (more friction)
                        self.friction_coefficient = max(0.5, self.friction_coefficient - 0.01)
                        print(f"Friction coefficient: {self.friction_coefficient:.3f}")
            
            # Update physics
            self.update_physics(dt)
            
            # Render the current frame
            self.render_scene()
        
        # Clean up
        pygame.quit()
        print("Simulation ended")

import pygame
from image import Image, Color
from model import Model, CollisionObject, Matrix4
from vector import Vector
from shape import Triangle, Point
import math

# Import the motion blur effect
from motion_blur import MotionBlurEffect

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

def setup_colored_scene():
    """Create a scene with multiple colored headsets"""
    from model import Model
    
    headsets = []
    
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
    
    # Create headsets in a circle formation
    num_circle = 8  # Number of headsets in the circle
    circle_radius = 15  # Distance from center
    for i in range(num_circle):
        angle = (i / num_circle) * 2 * math.pi
        
        # Position headsets in a circle
        pos = Vector(
            circle_radius * math.cos(angle),
            1,  # Slightly elevated
            circle_radius * math.sin(angle) - 10  # Centered at z=-10
        )
        
        # Velocity points toward center with varying speeds
        speed = 2 + (i % 3)  # Different speeds (2, 3, or 4 units/sec)
        vel = Vector(
            -math.cos(angle) * speed,
            0,
            -math.sin(angle) * speed
        )
        
        model = Model('data/headset.obj')
        model.normalizeGeometry()
        model.setPosition(pos.x, pos.y, pos.z)
        
        # Create collision object with color
        from color_support import ColoredCollisionObject
        headset = ColoredCollisionObject(
            model, pos, vel, radius=1.0, 
            diffuse_color=colors[i % len(colors)]
        )
        headsets.append(headset)
    
    return headsets

def update_physics(headsets, dt):
    """Update physics for all objects"""
    # Use a fixed time step for physics
    fixed_dt = 1/60  # Target 60 physics updates per second
    
    # Accumulate any leftover time
    update_physics.accumulator = getattr(update_physics, 'accumulator', 0) + dt
    
    # Run physics updates with fixed timestep
    while update_physics.accumulator >= fixed_dt:
        # Clear collision records
        for headset in headsets:
            headset.clear_collision_history()
        
        # Apply gravity
        for headset in headsets:
            headset.velocity.y -= 9.81 * fixed_dt
        
        # Check collisions
        for i in range(len(headsets)):
            for j in range(i + 1, len(headsets)):
                if headsets[i].check_collision(headsets[j]):
                    headsets[i].resolve_collision(headsets[j])
        
        # Update positions
        for headset in headsets:
            headset.update(fixed_dt)
        
        update_physics.accumulator -= fixed_dt

def store_previous_positions(headsets):
    """Store current positions of all objects for velocity calculation"""
    positions = []
    for headset in headsets:
        positions.append(Vector(
            headset.position.x,
            headset.position.y,
            headset.position.z
        ))
    return positions

def getVertexNormal(vertIndex, faceNormalsByVertex):
    """Compute vertex normals by averaging the normals of adjacent faces"""
    normal = Vector(0, 0, 0)
    for adjNormal in faceNormalsByVertex[vertIndex]:
        normal = normal + adjNormal
    return normal / len(faceNormalsByVertex[vertIndex])

def render_scene_with_motion_blur_old():
    """Main function to render a scene with motion blur effect"""
    # Initialize pygame
    pygame.init()
    width, height = 800, 600
    screen = pygame.display.set_mode((width, height))
    pygame.display.set_caption("3D Renderer with Motion Blur")
    
    # Create scene
    headsets = setup_colored_scene()
    
    # Create motion blur processor
    motion_blur = MotionBlurEffect(blur_strength=0.7, velocity_scale=2.5, max_samples=7)
    
    # Track previous positions for each object
    prev_positions = [None] * len(headsets)
    
    # Main rendering loop
    running = True
    clock = pygame.time.Clock()
    
    # Camera position for lighting calculations
    camera_pos = Vector(0, 5, -25)
    light_dir = Vector(0.5, -1, -0.5).normalize()
    
    while running:
        # Handle timing
        dt = min(clock.tick(60) / 1000.0, 0.1)  # Cap at 0.1s to prevent jumps
        
        # Store current positions before updating
        current_positions = store_previous_positions(headsets)
        
        # Update physics
        update_physics(headsets, dt)
        
        # Clear image for new frame
        image = Image(width, height, Color(10, 10, 40, 255))  # Dark blue background
        zBuffer = [-float('inf')] * width * height
        
        # Render each headset
        for headset in headsets:
            model = headset.model
            
            # Calculate face normals
            faceNormals = {}
            for face in model.faces:
                p0 = model.getTransformedVertex(face[0])
                p1 = model.getTransformedVertex(face[1])
                p2 = model.getTransformedVertex(face[2])
                faceNormal = (p2-p0).cross(p1-p0).normalize()
                
                for i in face:
                    if i not in faceNormals:
                        faceNormals[i] = []
                    faceNormals[i].append(faceNormal)
            
            # Calculate vertex normals
            vertexNormals = []
            for vertIndex in range(len(model.vertices)):
                if vertIndex in faceNormals:
                    # from shape import getVertexNormal
                    vertNorm = getVertexNormal(vertIndex, faceNormals)
                    vertexNormals.append(vertNorm)
                else:
                    vertexNormals.append(Vector(0, 1, 0))  # Default normal
            
            # Render faces
            for face in model.faces:
                p0 = model.getTransformedVertex(face[0])
                p1 = model.getTransformedVertex(face[1])
                p2 = model.getTransformedVertex(face[2])
                n0, n1, n2 = [vertexNormals[i] for i in face]
                
                # Skip back-facing triangles
                cull = False
                avg_normal = (n0 + n1 + n2) / 3
                if avg_normal * light_dir < 0:
                    cull = True
                
                if not cull:
                    # Create points with positions and normals
                    triangle_points = []
                    for p, n in zip([p0, p1, p2], [n0, n1, n2]):
                        screenX, screenY = getPerspectiveProjection(p.x, p.y, p.z, width, height)
                        
                        # Create a point with position, normal, and placeholder color
                        point = Point(screenX, screenY, p.z, Color(255, 255, 255, 255))
                        point.normal = n  # Add normal as an attribute for shading
                        
                        triangle_points.append(point)
                    
                    # Render the triangle with appropriate color from the colored model
                    from color_support import render_triangle_with_color
                    render_triangle_with_color(
                        triangle_points,
                        image,
                        zBuffer,
                        model,  # ColoredModel instance
                        light_dir,
                        camera_pos
                    )
        
        # Apply motion blur
        # Option 1: Per-object blur based on velocity
        final_image = motion_blur.per_object_velocity_blur(
            image, headsets, width, height, getPerspectiveProjection
        )
        
        # Option 2: Velocity buffer approach (requires more setup)
        # velocity_buffer = motion_blur.generate_velocity_buffer(
        #     [h.position for h in headsets],
        #     prev_positions,
        #     width, height
        # )
        # final_image = motion_blur.process(image, velocity_buffer)
        
        # Option 3: Simple frame accumulation (easier but less accurate)
        # final_image = motion_blur.process(image)
        
        # Update previous positions for next frame
        prev_positions = current_positions
        
        # Display the final image
        # Convert the image buffer to a Pygame surface
        buffer_surface = pygame.Surface((width, height))
        pixel_array = pygame.surfarray.pixels3d(buffer_surface)
        
        for y in range(height):
            for x in range(width):
                # Calculate the index in the buffer
                # Each row has a null byte at the start
                flipY = (height - y - 1)  # Flip Y coordinate
                index = (flipY * width + x) * 4 + flipY + 1  # +1 for null byte
                
                # Extract RGB values from the buffer (ignore alpha)
                r = final_image.buffer[index]
                g = final_image.buffer[index + 1]
                b = final_image.buffer[index + 2]
                
                # Update the pixel array
                pixel_array[x, y] = (r, g, b)
        
        del pixel_array  # Release the surface lock
        
        # Blit the buffer to the screen
        screen.blit(buffer_surface, (0, 0))
        
        # Draw FPS counter
        font = pygame.font.SysFont('Arial', 18)
        fps_text = font.render(f'FPS: {int(clock.get_fps())}', True, (255, 255, 255))
        screen.blit(fps_text, (10, 10))
        
        # Draw motion blur controls info
        controls_text = font.render('M: Toggle motion blur | +/-: Adjust blur strength', True, (255, 255, 255))
        screen.blit(controls_text, (10, height - 30))
        
        # Update display
        pygame.display.flip()
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                elif event.key == pygame.K_m:
                    # Use the toggle method for better logging
                    motion_blur.toggle()
                elif event.key == pygame.K_EQUALS or event.key == pygame.K_PLUS:
                    # Use the increase method for logging
                    motion_blur.increase_strength(0.1)
                elif event.key == pygame.K_MINUS:
                    # Use the decrease method for logging
                    motion_blur.decrease_strength(0.1)
                elif event.key == pygame.K_r:
                    # Reset scene
                    headsets = setup_colored_scene()
                    prev_positions = [None] * len(headsets)
                    print("[Scene] Reset with new headsets")
                # Optional: Add key to reset blur to default
                elif event.key == pygame.K_0:
                    old_value = motion_blur.blur_strength
                    motion_blur.set_blur_strength(0.7)
                    print(f"[Motion Blur] Reset to default strength: {old_value:.2f} → {motion_blur.blur_strength:.2f}")

    
    # Clean up
    pygame.quit()
    final_image.saveAsPNG("motion_blur_render.png")
    print("Render saved as motion_blur_render.png")
    
# New attempt for Motion Blur

def render_scene_with_motion_blur():
    """Main function to render a scene with motion blur effect"""
    # Initialize pygame
    pygame.init()
    width, height = 800, 600
    screen = pygame.display.set_mode((width, height))
    pygame.display.set_caption("3D Renderer with Motion Blur")
    
    # Create scene
    headsets = setup_colored_scene()
    
    # Create motion blur processor with optimized settings
    motion_blur = MotionBlurEffect(blur_strength=0.5, velocity_scale=1.5, max_samples=3)
    
    # Main rendering loop
    running = True
    clock = pygame.time.Clock()
    
    # Camera position for lighting calculations
    camera_pos = Vector(0, 5, -25)
    light_dir = Vector(0.5, -1, -0.5).normalize()
    
    # Create buffer surface for faster rendering
    buffer_surface = pygame.Surface((width, height))
    
    # Create font for FPS display
    font = pygame.font.SysFont('Arial', 18)
    
    # Current blur mode (0=disabled, 1=frame accumulation, 2=velocity blur, 3=fastest)
    blur_mode = 2
    blur_mode_names = ["Disabled", "Frame Accumulation", "Velocity Blur", "Ultra Fast"]
    
    # FPS tracking
    fps_history = []
    
    while running:
        # Handle timing
        dt = min(clock.tick(60) / 1000.0, 0.1)  # Cap at 0.1s to prevent jumps
        
        # Update physics
        update_physics(headsets, dt)
        
        # Clear image for new frame
        image = Image(width, height, Color(20, 20, 40, 255))  # Dark blue background
        zBuffer = [-float('inf')] * width * height
        
        # Render scene with optimized approach
        render_scene_optimized(image, headsets, zBuffer, camera_pos, light_dir, width, height)
        
        # Apply motion blur based on current mode
        if blur_mode == 0:
            final_image = image  # No blur
        elif blur_mode == 1:
            final_image = motion_blur.process(image)  # Frame accumulation
        elif blur_mode == 2:
            final_image = motion_blur.per_object_velocity_blur(image, headsets, width, height, perspective_projection)
        elif blur_mode == 3:
            final_image = motion_blur.fastest_blur(image, headsets, width, height, perspective_projection)
        
        # Update display with optimized approach
        update_display_optimized(screen, final_image, buffer_surface)
        
        # Track FPS
        current_fps = clock.get_fps()
        fps_history.append(current_fps)
        if len(fps_history) > 10:
            fps_history.pop(0)
        avg_fps = sum(fps_history) / len(fps_history) if fps_history else 0
        
        # Draw FPS counter and info
        fps_text = font.render(f'FPS: {int(avg_fps)}', True, (255, 255, 255))
        screen.blit(fps_text, (10, 10))
        
        # Show blur mode
        mode_text = font.render(f'Blur: {blur_mode_names[blur_mode]} (Strength: {motion_blur.blur_strength:.2f})', 
                                True, (255, 255, 255))
        screen.blit(mode_text, (10, 30))
        
        # Draw controls info
        controls_text = font.render('M: Toggle blur mode | +/-: Adjust strength | R: Reset | ESC: Quit', 
                                   True, (200, 200, 200))
        screen.blit(controls_text, (10, height - 30))
        
        # Update the display
        pygame.display.flip()
        
        # Event handling
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                elif event.key == pygame.K_m:
                    # Cycle through blur modes
                    blur_mode = (blur_mode + 1) % len(blur_mode_names)
                    print(f"[Blur] Mode changed to: {blur_mode_names[blur_mode]}")
                elif event.key == pygame.K_EQUALS or event.key == pygame.K_PLUS:
                    motion_blur.increase_strength(0.1)
                elif event.key == pygame.K_MINUS:
                    motion_blur.decrease_strength(0.1)
                elif event.key == pygame.K_r:
                    # Reset scene
                    headsets = setup_colored_scene()
                    print("[Scene] Reset with new objects")
                elif event.key == pygame.K_0:
                    motion_blur.set_blur_strength(0.5)
    
    # Clean up
    pygame.quit()
    final_image.saveAsPNG("motion_blur_render.png")
    print("Render saved as motion_blur_render.png")

def render_scene_with_shader_motion_blur():
    """Main function to render a scene with motion blur using a shader-like approach"""
    # Initialize pygame
    pygame.init()
    width, height = 800, 600
    screen = pygame.display.set_mode((width, height))
    pygame.display.set_caption("3D Renderer with Shader-like Motion Blur")
    
    # Create scene
    headsets = setup_colored_scene()
    
    # Create motion blur processor
    from velocity_shader import create_velocity_buffer, store_current_transforms, apply_velocity_blur
    
    # Main rendering loop
    running = True
    clock = pygame.time.Clock()
    
    # Camera position for lighting
    camera_pos = Vector(0, 5, -25)
    light_dir = Vector(0.5, -1, -0.5).normalize()
    
    # Create buffer surface
    buffer_surface = pygame.Surface((width, height))
    
    # Create font for info display
    font = pygame.font.SysFont('Arial', 18)
    
    # Track previous transforms
    previous_transforms = {}
    
    # Blur settings
    blur_strength = 0.7
    blur_enabled = True
    
    # FPS tracking
    fps_history = []
    
    while running:
        # Handle timing
        dt = min(clock.tick(60) / 1000.0, 0.1)
        
        # Store current transforms before updating
        previous_transforms = store_current_transforms(headsets)
        
        # Update physics
        update_physics(headsets, dt)
        
        # Create image for rendering
        image = Image(width, height, Color(20, 20, 40, 255))
        zBuffer = [-float('inf')] * width * height
        
        # Render scene
        render_scene_optimized(image, headsets, zBuffer, camera_pos, light_dir, width, height)
        
        # Create velocity buffer from current and previous transforms
        velocity_buffer = create_velocity_buffer(
            headsets, 
            previous_transforms, 
            width, height, 
            perspective_projection
        )
        
        # Apply motion blur if enabled
        if blur_enabled:
            final_image = apply_velocity_blur(
                image, 
                velocity_buffer, 
                blur_strength=blur_strength,
                max_samples=4
            )
        else:
            final_image = image
        
        # Update display
        update_display_optimized(screen, final_image, buffer_surface)
        
        # Track FPS
        current_fps = clock.get_fps()
        fps_history.append(current_fps)
        if len(fps_history) > 10:
            fps_history.pop(0)
        avg_fps = sum(fps_history) / len(fps_history) if fps_history else 0
        
        # Draw info
        fps_text = font.render(f'FPS: {int(avg_fps)}', True, (255, 255, 255))
        screen.blit(fps_text, (10, 10))
        
        blur_text = font.render(
            f'Motion Blur: {"ON" if blur_enabled else "OFF"} (Strength: {blur_strength:.2f})', 
            True, (255, 255, 255)
        )
        screen.blit(blur_text, (10, 30))
        
        # Controls info
        controls_text = font.render(
            'B: Toggle blur | +/-: Adjust strength | R: Reset | ESC: Quit', 
            True, (200, 200, 200)
        )
        screen.blit(controls_text, (10, height - 30))
        
        # Update display
        pygame.display.flip()
        
        # Handle events
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                elif event.key == pygame.K_b:
                    blur_enabled = not blur_enabled
                    print(f"[Blur] {'Enabled' if blur_enabled else 'Disabled'}")
                elif event.key == pygame.K_EQUALS or event.key == pygame.K_PLUS:
                    blur_strength = min(1.5, blur_strength + 0.1)
                    print(f"[Blur] Strength: {blur_strength:.2f}")
                elif event.key == pygame.K_MINUS:
                    blur_strength = max(0.1, blur_strength - 0.1)
                    print(f"[Blur] Strength: {blur_strength:.2f}")
                elif event.key == pygame.K_r:
                    headsets = setup_colored_scene()
                    previous_transforms = {}
                    print("[Scene] Reset")
    
    # Cleanup
    pygame.quit()
    final_image.saveAsPNG("shader_motion_blur.png")
    print("Render saved as shader_motion_blur.png")

def render_scene_with_hlsl_motion_blur():
    """
    Render a scene with motion blur using an approach that directly follows HLSL shaders.
    This implements the velocity calculation:
    
    float4 currentPos = H;
    float4 previousPos = mul(worldPos, g_previousViewProjectionMatrix);
    previousPos /= previousPos.w;
    float2 velocity = (currentPos - previousPos) / 2.f;
    """
    # Initialize pygame
    pygame.init()
    width, height = 800, 600
    screen = pygame.display.set_mode((width, height))
    pygame.display.set_caption("3D Renderer with HLSL-style Motion Blur")
    
    # Create scene
    headsets = setup_colored_scene()
    
    # Import shader-like motion blur implementation
    from refined_velocity_shader import ViewProjectionState, create_full_velocity_buffer, apply_velocity_blur
    
    # Create view-projection state to track matrices between frames
    view_proj_state = ViewProjectionState(width, height)
    
    # Main rendering loop
    running = True
    clock = pygame.time.Clock()
    
    # Camera position
    camera_pos = Vector(0, 5, -25)
    camera_target = Vector(0, 0, 0)
    light_dir = Vector(0.5, -1, -0.5).normalize()
    
    # Create buffer surface
    buffer_surface = pygame.Surface((width, height))
    
    # Create font for info display
    font = pygame.font.SysFont('Arial', 18)
    
    # Blur settings
    blur_strength = 1.0
    blur_samples = 8
    blur_enabled = True
    
    # FPS tracking
    fps_history = []
    frame_count = 0
    
    while running:
        # Handle timing
        dt = min(clock.tick(60) / 1000.0, 0.1)
        frame_count += 1
        
        # Update physics
        update_physics(headsets, dt)
        
        # Update view-projection matrices
        view_proj_state.update(camera_pos, camera_target, width, height)
        
        # Create image for rendering
        image = Image(width, height, Color(20, 20, 40, 255))
        zBuffer = [-float('inf')] * width * height
        
        # Render scene
        render_scene_optimized(image, headsets, zBuffer, camera_pos, light_dir, width, height)
        
        # Create velocity buffer (only every few frames to improve performance)
        if frame_count % 2 == 0:
            # Generate full velocity buffer based on current and previous view-projection matrices
            velocity_buffer = create_full_velocity_buffer(
                headsets,
                view_proj_state,
                width, height
            )
            
            # Apply motion blur if enabled
            if blur_enabled:
                final_image = apply_velocity_blur(
                    image, 
                    velocity_buffer, 
                    blur_strength=blur_strength,
                    num_samples=blur_samples
                )
            else:
                final_image = image
        else:
            final_image = image
        
        # Update display
        update_display_optimized(screen, final_image, buffer_surface)
        
        # Track FPS
        current_fps = clock.get_fps()
        fps_history.append(current_fps)
        if len(fps_history) > 10:
            fps_history.pop(0)
        avg_fps = sum(fps_history) / len(fps_history) if fps_history else 0
        
        # Draw info
        fps_text = font.render(f'FPS: {int(avg_fps)}', True, (255, 255, 255))
        screen.blit(fps_text, (10, 10))
        
        blur_text = font.render(
            f'HLSL-style Motion Blur: {"ON" if blur_enabled else "OFF"} ' +
            f'(Strength: {blur_strength:.1f}, Samples: {blur_samples})', 
            True, (255, 255, 255)
        )
        screen.blit(blur_text, (10, 30))
        
        # Controls info
        controls_text = font.render(
            'B: Toggle blur | +/-: Adjust strength | [/]: Adjust samples | R: Reset', 
            True, (200, 200, 200)
        )
        screen.blit(controls_text, (10, height - 30))
        
        # Update display
        pygame.display.flip()
        
        # Handle events
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                elif event.key == pygame.K_b:
                    blur_enabled = not blur_enabled
                    print(f"[Blur] {'Enabled' if blur_enabled else 'Disabled'}")
                elif event.key == pygame.K_EQUALS or event.key == pygame.K_PLUS:
                    blur_strength = min(2.0, blur_strength + 0.2)
                    print(f"[Blur] Strength: {blur_strength:.1f}")
                elif event.key == pygame.K_MINUS:
                    blur_strength = max(0.2, blur_strength - 0.2)
                    print(f"[Blur] Strength: {blur_strength:.1f}")
                elif event.key == pygame.K_RIGHTBRACKET:
                    blur_samples = min(16, blur_samples + 2)
                    print(f"[Blur] Samples: {blur_samples}")
                elif event.key == pygame.K_LEFTBRACKET:
                    blur_samples = max(4, blur_samples - 2)
                    print(f"[Blur] Samples: {blur_samples}")
                elif event.key == pygame.K_r:
                    headsets = setup_colored_scene()
                    print("[Scene] Reset")
    
    # Cleanup
    pygame.quit()
    final_image.saveAsPNG("hlsl_motion_blur.png")
    print("Render saved as hlsl_motion_blur.png")

def render_scene_with_simple_motion_blur():
    """Render a scene with extremely simplified motion blur for better performance"""
    # Initialize pygame
    pygame.init()
    width, height = 800, 600
    screen = pygame.display.set_mode((width, height))
    pygame.display.set_caption("Simple Motion Blur")
    
    # Create scene
    headsets = setup_colored_scene()
    
    # Create simplified motion blur processor
    from simplified_motion_blur import SimpleMotionBlur
    motion_blur = SimpleMotionBlur(blur_strength=1.5)
    
    # Main rendering loop
    running = True
    clock = pygame.time.Clock()
    
    # Camera position
    camera_pos = Vector(0, 5, -25)
    light_dir = Vector(0.5, -1, -0.5).normalize()
    
    # Blur settings
    blur_enabled = True
    
    # Performance tracking
    fps_history = []
    render_time_history = []
    
    while running:
        frame_start_time = pygame.time.get_ticks()
        
        # Handle timing
        dt = min(clock.tick(60) / 1000.0, 0.1)
        
        # Update blur with object positions from previous frame
        motion_blur.update_object_positions(headsets)
        
        # Update physics
        update_physics(headsets, dt)
        
        # Create image for rendering
        image = Image(width, height, Color(20, 20, 40, 255))
        zBuffer = [-float('inf')] * width * height
        
        # Render scene
        for obj in headsets:
            # Very simplified rendering - just get transformed vertices once
            model = obj.model
            transformed_vertices = []
            for i in range(len(model.vertices)):
                transformed_vertices.append(model.getTransformedVertex(i))
            
            # Render visible faces
            for face in model.faces:
                p0 = transformed_vertices[face[0]]
                p1 = transformed_vertices[face[1]]
                p2 = transformed_vertices[face[2]]
                
                # Simple lighting
                normal = (p2-p0).cross(p1-p0).normalize()
                intensity = max(0.2, normal * light_dir)
                
                # Project to screen
                points = []
                for p in [p0, p1, p2]:
                    screenX, screenY = perspective_projection(p.x, p.y, p.z, width, height)
                    points.append(Point(screenX, screenY, p.z, Color(
                        intensity*200, intensity*200, intensity*200, 255
                    )))
                
                # Draw triangle
                Triangle(points[0], points[1], points[2]).draw_faster(image, zBuffer)
        
        # Apply motion blur if enabled
        if blur_enabled:
            final_image = motion_blur.apply_blur(image, headsets, width, height, perspective_projection)
        else:
            final_image = image
        
        # Direct pixel copy to screen (faster)
        pixel_array = pygame.PixelArray(screen)
        for y in range(height):
            for x in range(width):
                idx = (height - y - 1) * width * 4 + x * 4 + (height - y - 1) + 1
                if idx < len(final_image.buffer) - 3:
                    r = final_image.buffer[idx]
                    g = final_image.buffer[idx+1]
                    b = final_image.buffer[idx+2]
                    pixel_array[x, y] = (r, g, b)
        del pixel_array
        
        # Calculate frame time
        frame_time = (pygame.time.get_ticks() - frame_start_time) / 1000.0
        render_time_history.append(frame_time)
        if len(render_time_history) > 30:
            render_time_history.pop(0)
        avg_frame_time = sum(render_time_history) / len(render_time_history)
        
        # Track FPS
        current_fps = clock.get_fps()
        fps_history.append(current_fps)
        if len(fps_history) > 10:
            fps_history.pop(0)
        avg_fps = sum(fps_history) / len(fps_history) if fps_history else 0
        
        # Draw FPS info
        font = pygame.font.SysFont('Arial', 18)
        fps_text = font.render(f'FPS: {int(avg_fps)} | Frame Time: {avg_frame_time*1000:.1f}ms', 
                              True, (255, 255, 255))
        screen.blit(fps_text, (10, 10))
        
        blur_text = font.render(f'Blur: {"ON" if blur_enabled else "OFF"} (Strength: {motion_blur.blur_strength:.1f})', 
                              True, (255, 255, 255))
        screen.blit(blur_text, (10, 30))
        
        controls_text = font.render('B: Toggle blur | +/-: Adjust strength | R: Reset', 
                                  True, (200, 200, 200))
        screen.blit(controls_text, (10, height - 30))
        
        # Update display
        pygame.display.flip()
        
        # Handle events
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                elif event.key == pygame.K_b:
                    blur_enabled = not blur_enabled
                    print(f"[Blur] {'Enabled' if blur_enabled else 'Disabled'}")
                elif event.key == pygame.K_EQUALS or event.key == pygame.K_PLUS:
                    motion_blur.blur_strength = min(3.0, motion_blur.blur_strength + 0.5)
                    print(f"[Blur] Strength: {motion_blur.blur_strength:.1f}")
                elif event.key == pygame.K_MINUS:
                    motion_blur.blur_strength = max(0.5, motion_blur.blur_strength - 0.5)
                    print(f"[Blur] Strength: {motion_blur.blur_strength:.1f}")
                elif event.key == pygame.K_r:
                    headsets = setup_colored_scene()
                    motion_blur.previous_positions = {}  # Reset positions
                    print("[Scene] Reset")
    
    pygame.quit()

def render_scene_with_extreme_motion_blur():
    """Render a scene with extremely exaggerated motion blur effect for maximum visibility"""
    # Initialize pygame
    pygame.init()
    width, height = 800, 600
    screen = pygame.display.set_mode((width, height))
    pygame.display.set_caption("Extreme Motion Blur")
    
    # Create scene with fast-moving objects
    headsets = setup_colored_scene_fast()  # Special setup with faster objects
    
    # Create extreme motion blur processor
    from extreme_motion_blur import ExtremeMotionBlur
    motion_blur = ExtremeMotionBlur(blur_strength=4.0)  # Start with very high strength
    
    # Main rendering loop
    running = True
    clock = pygame.time.Clock()
    
    # Camera position
    camera_pos = Vector(0, 5, -25)
    light_dir = Vector(0.5, -1, -0.5).normalize()
    
    # Blur settings
    blur_enabled = True
    
    # Performance tracking
    fps_history = []
    
    # Font for info display
    font = pygame.font.SysFont('Arial', 18)
    
    while running:
        # Handle timing
        dt = min(clock.tick(60) / 1000.0, 0.1)
        
        # Update blur with object positions from previous frame
        motion_blur.update_object_positions(headsets)
        
        # Update physics
        update_physics(headsets, dt)
        
        # Create image for rendering
        image = Image(width, height, Color(20, 20, 40, 255))
        zBuffer = [-float('inf')] * width * height
        
        # Render scene
        for obj in headsets:
            # Very simplified rendering - just get transformed vertices once
            model = obj.model
            transformed_vertices = []
            for i in range(len(model.vertices)):
                transformed_vertices.append(model.getTransformedVertex(i))
            
            # Render visible faces
            for face in model.faces:
                p0 = transformed_vertices[face[0]]
                p1 = transformed_vertices[face[1]]
                p2 = transformed_vertices[face[2]]
                
                # Simple lighting
                normal = (p2-p0).cross(p1-p0).normalize()
                intensity = max(0.2, normal * light_dir)
                
                # Project to screen
                points = []
                for p in [p0, p1, p2]:
                    screenX, screenY = perspective_projection(p.x, p.y, p.z, width, height)
                    
                    # Use object's color from color_support if available
                    color = None
                    if hasattr(obj, 'model') and hasattr(obj.model, 'diffuse_color'):
                        # Get color from the model
                        r, g, b = obj.model.diffuse_color
                        color = Color(int(r * intensity), int(g * intensity), int(b * intensity), 255)
                    else:
                        # Default color
                        color = Color(intensity*200, intensity*200, intensity*200, 255)
                    
                    points.append(Point(screenX, screenY, p.z, color))
                
                # Draw triangle
                Triangle(points[0], points[1], points[2]).draw_faster(image, zBuffer)
        
        # Apply motion blur if enabled
        if blur_enabled:
            final_image = motion_blur.apply_blur(image, headsets, width, height, perspective_projection)
        else:
            final_image = image
        
        # Draw to screen
        update_display_optimized(screen, final_image)
        
        # Track FPS
        current_fps = clock.get_fps()
        fps_history.append(current_fps)
        if len(fps_history) > 10:
            fps_history.pop(0)
        avg_fps = sum(fps_history) / len(fps_history) if fps_history else 0
        
        # Draw info
        fps_text = font.render(f'FPS: {int(avg_fps)}', True, (255, 255, 255))
        screen.blit(fps_text, (10, 10))
        
        blur_text = font.render(f'EXTREME Blur: {"ON" if blur_enabled else "OFF"} (Strength: {motion_blur.blur_strength:.1f})', 
                              True, (255, 255, 255))
        screen.blit(blur_text, (10, 30))
        
        controls_text = font.render('B: Toggle blur | +/-: Adjust strength | R: Reset', 
                                  True, (200, 200, 200))
        screen.blit(controls_text, (10, height - 30))
        
        # Update display
        pygame.display.flip()
        
        # Handle events
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                elif event.key == pygame.K_b:
                    blur_enabled = not blur_enabled
                    print(f"[Blur] {'Enabled' if blur_enabled else 'Disabled'}")
                elif event.key == pygame.K_EQUALS or event.key == pygame.K_PLUS:
                    motion_blur.blur_strength = min(6.0, motion_blur.blur_strength + 0.5)
                    print(f"[Blur] Strength: {motion_blur.blur_strength:.1f}")
                elif event.key == pygame.K_MINUS:
                    motion_blur.blur_strength = max(1.0, motion_blur.blur_strength - 0.5)
                    print(f"[Blur] Strength: {motion_blur.blur_strength:.1f}")
                elif event.key == pygame.K_r:
                    headsets = setup_colored_scene_fast()
                    motion_blur.previous_positions = {}
                    motion_blur.frame_history = []
                    print("[Scene] Reset")
    
    pygame.quit()

def setup_colored_scene_fast():
    """Create a scene with faster-moving objects for more visible motion blur"""
    from color_support import ColoredCollisionObject
    
    headsets = []
    
    # Define some colors (RGB tuples)
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
    
    # Create a circle of headsets with higher speeds
    num_circle = 8
    circle_radius = 15
    for i in range(num_circle):
        angle = (i / num_circle) * 2 * math.pi
        
        # Position in circle
        pos = Vector(
            circle_radius * math.cos(angle),
            1,  # Slightly elevated
            circle_radius * math.sin(angle) - 10  # Centered at z=-10
        )
        
        # Higher velocity toward center (4-8 range instead of 2-4)
        speed = 5 + (i % 4)  # Much faster speeds
        vel = Vector(
            -math.cos(angle) * speed,
            0,
            -math.sin(angle) * speed
        )
        
        model = Model('data/headset.obj')
        model.normalizeGeometry()
        model.setPosition(pos.x, pos.y, pos.z)
        
        # Create collision object with color
        headset = ColoredCollisionObject(
            model, pos, vel, radius=1.0, 
            diffuse_color=colors[i % len(colors)]
        )
        headsets.append(headset)
    
    return headsets

def update_display_optimized(screen, image):
    """Optimized display update function"""
    width, height = image.width, image.height
    
    # Use PixelArray for faster updates
    try:
        pixel_array = pygame.PixelArray(screen)
        
        for y in range(height):
            flipY = (height - y - 1)
            row_start = (flipY * width) * 4 + flipY + 1
            
            for x in range(width):
                idx = row_start + x * 4
                
                # Check bounds to avoid errors
                if idx+2 < len(image.buffer):
                    r = image.buffer[idx]
                    g = image.buffer[idx+1]
                    b = image.buffer[idx+2]
                    pixel_array[x, y] = (r, g, b)
        
        del pixel_array  # Release the surface lock
        
    except:
        # Fall back to slower update if pixel_array fails
        for y in range(height):
            for x in range(width):
                idx = (height - y - 1) * width * 4 + x * 4 + (height - y - 1) + 1
                if idx+2 < len(image.buffer):
                    r = image.buffer[idx]
                    g = image.buffer[idx+1]
                    b = image.buffer[idx+2]
                    screen.set_at((x, y), (r, g, b))

def perspective_projection(x, y, z, width, height):
    """
    Optimized perspective projection function.
    Simpler version of getPerspectiveProjection for better performance.
    """
    # Simple perspective projection
    fov = math.pi / 3.0  # 60-degree field of view
    near = 0.1
    aspect = width / height
    
    # Avoid division by very small z values
    depth = min(z, -0.1) if z < 0 else -0.1
    
    # Perspective division
    x_normalized = x / -depth
    y_normalized = y / -depth
    
    # Scale to screen coordinates
    screenX = int((x_normalized + 1.0) * width / 2.0)
    screenY = int((y_normalized + 1.0) * height / 2.0)
    
    return screenX, screenY

def update_physics(headsets, dt):
    """
    Optimized physics update for headsets.
    Uses fixed timestep for consistency.
    """
    # Fixed physics time step
    fixed_dt = 1/60.0
    
    # Accumulate leftover time
    update_physics.accumulator = getattr(update_physics, 'accumulator', 0) + dt
    
    # Run physics updates with fixed timestep
    while update_physics.accumulator >= fixed_dt:
        # Clear collisions for all objects
        for headset in headsets:
            headset.clear_collision_history()
        
        # Update velocities (apply gravity)
        for headset in headsets:
            headset.velocity.y -= 9.81 * fixed_dt
        
        # Check collisions more efficiently
        for i in range(len(headsets)):
            for j in range(i + 1, len(headsets)):
                # Quick distance check before detailed collision test
                dx = headsets[i].position.x - headsets[j].position.x
                dy = headsets[i].position.y - headsets[j].position.y
                dz = headsets[i].position.z - headsets[j].position.z
                dist_sq = dx*dx + dy*dy + dz*dz
                
                # Only check collision if objects are close enough
                if dist_sq < (headsets[i].radius + headsets[j].radius) * 2:
                    if headsets[i].check_collision(headsets[j]):
                        headsets[i].resolve_collision(headsets[j])
        
        # Update positions
        for headset in headsets:
            headset.update(fixed_dt)
        
        update_physics.accumulator -= fixed_dt

def render_scene_optimized(image, headsets, zBuffer, camera_pos, light_dir, width, height):
    """
    Optimized version of scene rendering with better performance.
    """
    from color_support import render_triangle_with_color
    
    # Render each headset
    for headset in headsets:
        model = headset.model
        
        # Pre-compute transformed vertices to avoid redundant calculations
        transformed_vertices = []
        for i in range(len(model.vertices)):
            transformed_vertices.append(model.getTransformedVertex(i))
        
        # Calculate face normals
        faceNormals = {}
        for face in model.faces:
            p0 = transformed_vertices[face[0]]
            p1 = transformed_vertices[face[1]]
            p2 = transformed_vertices[face[2]]
            faceNormal = (p2-p0).cross(p1-p0).normalize()
            
            for i in face:
                if i not in faceNormals:
                    faceNormals[i] = []
                faceNormals[i].append(faceNormal)
        
        # Calculate vertex normals
        vertexNormals = []
        for vertIndex in range(len(model.vertices)):
            if vertIndex in faceNormals:
                # Average the normals from adjacent faces
                normal = Vector(0, 0, 0)
                for adjNormal in faceNormals[vertIndex]:
                    normal = normal + adjNormal
                vertexNormals.append(normal / len(faceNormals[vertIndex]))
            else:
                vertexNormals.append(Vector(0, 1, 0))  # Default normal
        
        # Filter visible faces before rendering
        visible_faces = []
        for face_idx, face in enumerate(model.faces):
            # Average the normals for this face for quicker culling
            avg_normal = (vertexNormals[face[0]] + vertexNormals[face[1]] + vertexNormals[face[2]]) / 3
            
            # Skip back-facing triangles
            if avg_normal * light_dir >= 0:
                visible_faces.append(face)
        
        # Render only visible faces
        for face in visible_faces:
            p0 = transformed_vertices[face[0]]
            p1 = transformed_vertices[face[1]]
            p2 = transformed_vertices[face[2]]
            n0, n1, n2 = [vertexNormals[i] for i in face]
            
            # Create points with positions and normals
            triangle_points = []
            for p, n in zip([p0, p1, p2], [n0, n1, n2]):
                screenX, screenY = perspective_projection(p.x, p.y, p.z, width, height)
                
                # Create a point with position, normal, and placeholder color
                point = Point(screenX, screenY, p.z, Color(255, 255, 255, 255))
                point.normal = n  # Add normal for shading
                
                triangle_points.append(point)
            
            # Render the triangle with proper coloring
            render_triangle_with_color(
                triangle_points,
                image,
                zBuffer,
                model,  # ColoredModel instance
                light_dir,
                camera_pos
            )

def update_display_optimized_old(screen, image, buffer_surface=None):
    """Optimized version of updating the display from an image"""
    width, height = image.width, image.height
    
    # Create buffer surface if not provided
    if buffer_surface is None:
        buffer_surface = pygame.Surface((width, height))
    
    try:
        # Use a faster method with PixelArray
        pixel_array = pygame.PixelArray(buffer_surface)
        
        for y in range(height):
            flipY = (height - y - 1)
            row_start = (flipY * width) * 4 + flipY + 1
            
            for x in range(width):
                index = row_start + x * 4
                r = image.buffer[index]
                g = image.buffer[index + 1]
                b = image.buffer[index + 2]
                
                pixel_array[x, y] = (r, g, b)
                
        del pixel_array  # Release the surface lock
        
    except pygame.error:
        # Fall back to direct pixel setting if PixelArray fails
        for y in range(height):
            for x in range(width):
                flipY = (height - y - 1)
                index = (flipY * width + x) * 4 + flipY + 1
                
                r = image.buffer[index]
                g = image.buffer[index + 1]
                b = image.buffer[index + 2]
                
                buffer_surface.set_at((x, y), (r, g, b))
    
    # Blit the buffer to the screen
    screen.blit(buffer_surface, (0, 0))
    return True


def setup_colored_scene():
    """Create a scene with multiple colored headsets for testing motion blur"""
    from color_support import ColoredCollisionObject
    
    headsets = []
    
    # Define some colors (RGB tuples)
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
    
    # Create a circle of headsets
    num_circle = 8
    circle_radius = 15
    for i in range(num_circle):
        angle = (i / num_circle) * 2 * math.pi
        
        # Position in circle
        pos = Vector(
            circle_radius * math.cos(angle),
            1,  # Slightly elevated
            circle_radius * math.sin(angle) - 10  # Centered at z=-10
        )
        
        # Velocity toward center
        speed = 2 + (i % 3)  # Different speeds
        vel = Vector(
            -math.cos(angle) * speed,
            0,
            -math.sin(angle) * speed
        )
        
        model = Model('data/headset.obj')
        model.normalizeGeometry()
        model.setPosition(pos.x, pos.y, pos.z)
        
        # Create collision object with color
        headset = ColoredCollisionObject(
            model, pos, vel, radius=1.0, 
            diffuse_color=colors[i % len(colors)]
        )
        headsets.append(headset)
    
    return headsets

def problem_4_2():
    simulation = SingleObjectSimulation()
    simulation.run_simulation()

def problem_5_V1(shader=True):
    if shader:
        render_scene_with_shader_motion_blur()
    else:
        render_scene_with_motion_blur()

def problem_5_V2(): # uses HLSL-style motion blur but has performance issues due to objects
    """Entry point for motion blur renderer - using HLSL-style implementation"""
    render_scene_with_hlsl_motion_blur()

def problem_5_V3(): # Better performance with simplified motion blur
    """Entry point for the simplified motion blur renderer"""
    render_scene_with_simple_motion_blur()

def problem_5_V4():
    """Entry point for the extreme motion blur renderer"""
    render_scene_with_extreme_motion_blur()

# problem_4_1() # multiple collisions 4.1

# problem_4_2() # multiple collisions 4.2

# problem_5_1_V1(False) # motion blur 5.1 # Set to true to use the shader approach from this link: https://developer.nvidia.com/gpugems/gpugems3/part-iv-image-effects/chapter-27-motion-blur-post-processing-effect

# problem_5_V2()

# problem_5_V3() # Simplified motion blur but not as visible (if at all?)

# problem_5_V4() # Extreme motion blur for maximum visibility and uses the colour system

# ^^^^^ THIS METHOD WORKS THE BEST AT THE MOMENT FOR VISIBILITY
# -> WILL NEED TO RESEARCH WHAT IT IS ACTUALLY DOING FOR THE REPORT?

# FINAL DEMO METHODS

from extreme_motion_blur import ExtremeMotionBlur
from color_support import ColoredCollisionObject, ColoredModel

class HeadsetScene:
    """
    A 3D scene with multiple VR headsets:
    - One rotating in the center based on sensor data
    - Multiple headsets sliding on the floor with friction and collisions
    """
    def __init__(self, width=800, height=600, csv_path="IMUData.csv"):
        # Initialize pygame
        pygame.init()
        self.width = width
        self.height = height
        self.screen = pygame.display.set_mode((width, height))
        pygame.display.set_caption("VR Headset Physics Scene")
        
        # Create buffer surface for rendering
        self.buffer_surface = pygame.Surface((width, height))
        
        # Image and Z-buffer for rendering
        self.image = Image(width, height, Color(20, 20, 40, 255))  # Dark blue background
        self.zBuffer = [-float('inf')] * width * height
        
        # Camera position and lighting
        self.camera_pos = Vector(0, 10, -30)
        self.light_dir = Vector(0.5, -1, -0.5).normalize()

        # Add a playback speed multiplier
        self.imu_playback_speed = 10.0  # Process 10 samples per frame initially
        
        # Tracking for update rate
        self.frame_count = 0
        self.time_accumulator = 0
        self.fps = 0
        
        # Motion blur effect
        self.motion_blur = ExtremeMotionBlur(blur_strength=2.5)
        self.blur_enabled = True
        
        # Load sensor data for rotating headset
        self.csv_path = csv_path
        self.load_sensor_data()
        
        # Create scene objects
        self.central_headset = None
        self.floor_headsets = []
        self.setup_scene()
        
        # Physics settings
        self.friction_coefficient = 0.95  # Higher = less friction
        self.accumulator = 0  # For fixed timestep physics
        
        # Debug and control flags
        self.show_debug = True
        self.paused = False
        
        # Font for info display
        self.font = pygame.font.SysFont('Arial', 18)

    def load_sensor_data(self):
        """Load sensor data from CSV file"""
        try:
            parser = SensorDataParser(self.csv_path)
            self.sensor_data = parser.parse()
            print(f"Loaded {len(self.sensor_data)} sensor data points")
            
            # Create dead reckoning filter
            self.dr_filter = DeadReckoningFilter(alpha=0.98)
            # Calibrate using first 100 samples
            self.dr_filter.calibrate(self.sensor_data[:min(100, len(self.sensor_data))])
            
            self.current_data_index = 0
        except Exception as e:
            print(f"Error loading sensor data: {e}")
            print("Using fallback rotation pattern instead")
            self.sensor_data = None
            self.dr_filter = None

    def create_floor_headsets(self):
        """Create multiple headsets that slide on the floor"""
        headsets = []
        
        # Simple RGB color tuples - no complex structures
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
        
        # Create headsets in circle formation
        num_circle = 8
        circle_radius = 15
        for i in range(num_circle):
            angle = (i / num_circle) * 2 * math.pi
            
            # Position in circle
            pos = Vector(
                circle_radius * math.cos(angle),
                1,  # Slightly above floor
                circle_radius * math.sin(angle) - 15  # Centered at z=-15
            )
            
            # Velocity toward center
            speed = 2 + (i % 3)  # Different speeds for variety
            vel = Vector(
                -math.cos(angle) * speed,
                0,
                -math.sin(angle) * speed
            )
            
            model = Model('data/headset.obj')
            model.normalizeGeometry()
            model.setPosition(pos.x, pos.y, pos.z)
            
            # Just assign the color directly to the model
            model.diffuse_color = colors[i % len(colors)]
            
            # Create regular collision object
            headset = CollisionObject(model, pos, vel, radius=1.0)
            headsets.append(headset)
        
        # "Billiards break" pattern
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
                # Stationary initially
                vel = Vector(0, 0, 0)
                
                model = Model('data/headset.obj')
                model.normalizeGeometry()
                model.setPosition(pos.x, pos.y, pos.z)
                
                # Assign color directly
                model.diffuse_color = colors[color_index % len(colors)]
                color_index += 1
                
                headset = CollisionObject(model, pos, vel, radius=1.0)
                headsets.append(headset)
        
        # "Cue ball" headset - plain white
        pos = Vector(0, 1, -20)  # Behind the triangle
        vel = Vector(0, 0, 8)    # Moving forward to hit the triangle
        
        model = Model('data/headset.obj')
        model.normalizeGeometry()
        model.setPosition(pos.x, pos.y, pos.z)
        model.diffuse_color = (255, 255, 255)  # White
        
        headsets.append(CollisionObject(model, pos, vel, radius=1.0))
        
        return headsets

    def update_floor_physics(self, dt):
        """Update physics for floor headsets with collisions and friction"""
        # Use a fixed time step for consistent physics
        fixed_dt = 1/60.0  # Target 60 physics updates per second
        
        # Accumulate leftover time
        self.accumulator += dt
        
        # Run physics updates with fixed timestep
        while self.accumulator >= fixed_dt:
            # Clear collision records
            for headset in self.floor_headsets:
                headset.clear_collision_history()
            
            # Apply gravity
            for headset in self.floor_headsets:
                headset.velocity.y -= 9.81 * fixed_dt
            
            # Check collisions between headsets
            for i in range(len(self.floor_headsets)):
                for j in range(i + 1, len(self.floor_headsets)):
                    # Quick distance check before detailed collision test
                    dx = self.floor_headsets[i].position.x - self.floor_headsets[j].position.x
                    dy = self.floor_headsets[i].position.y - self.floor_headsets[j].position.y
                    dz = self.floor_headsets[i].position.z - self.floor_headsets[j].position.z
                    dist_sq = dx*dx + dy*dy + dz*dz
                    
                    # Only check collision if objects are close enough
                    max_dist = self.floor_headsets[i].radius + self.floor_headsets[j].radius
                    if dist_sq < max_dist * max_dist * 1.5:
                        if self.floor_headsets[i].check_collision(self.floor_headsets[j]):
                            self.floor_headsets[i].resolve_collision(self.floor_headsets[j])
            
            # Apply friction to headsets on the floor
            for headset in self.floor_headsets:
                # Check if headset is on or near the floor
                is_on_floor = headset.position.y - headset.radius <= 0.01
                
                if is_on_floor:
                    # Ensure the headset doesn't go below the floor
                    headset.position.y = headset.radius
                    
                    # Apply friction to horizontal velocity components
                    horizontal_speed_squared = (
                        headset.velocity.x**2 + 
                        headset.velocity.z**2
                    )
                    
                    # Only apply friction if moving horizontally
                    if horizontal_speed_squared > 0.001:
                        # Apply friction by reducing horizontal velocity
                        headset.velocity.x *= self.friction_coefficient
                        headset.velocity.z *= self.friction_coefficient
                        
                        # Stop completely if very slow (avoid endless tiny sliding)
                        if horizontal_speed_squared < 0.05:
                            headset.velocity.x = 0
                            headset.velocity.z = 0
            
            # Update positions
            for headset in self.floor_headsets:
                headset.update(fixed_dt)
            
            self.accumulator -= fixed_dt

    def perspective_projection(self, x, y, z, width=None, height=None):
        """Convert 3D world coordinates to 2D screen coordinates with perspective"""
        # Use class width/height if not provided
        if width is None:
            width = self.width
        if height is None:
            height = self.height
            
        # Perspective projection parameters
        fov = math.pi / 3.0  # 60-degree field of view
        aspect = width / height
        near = 0.1
        
        # Avoid division by very small z values
        depth = z
        if depth > -0.1:  # Ensure z is negative (behind camera)
            depth = -0.1
            
        # Perspective division
        x_normalized = x / -depth
        y_normalized = y / -depth
        
        # Scale to screen coordinates
        screenX = int((x_normalized + 1.0) * width / 2.0)
        screenY = int((y_normalized + 1.0) * height / 2.0)
        
        return screenX, screenY

    def render_scene(self):
        """Render the entire scene with all headsets"""
        if self.central_headset and "model" in self.central_headset:
            model = self.central_headset["model"]
            pos = Vector(model.trans[0], model.trans[1], model.trans[2])
            print(f"Rendering central headset at position {pos.x}, {pos.y}, {pos.z}")
            # Store previous positions for motion blur
            self.motion_blur.update_object_positions(self.floor_headsets)
            
            # Clear image and z-buffer for new frame
            self.image = Image(self.width, self.height, Color(20, 20, 40, 255))
            self.zBuffer = [-float('inf')] * self.width * self.height
            
            # Render floor grid
            self.render_floor_grid()
            
            # Render central rotating headset
            self.render_model(self.central_headset["model"])
            
            # Render floor headsets
            for headset in self.floor_headsets:
                self.render_model(headset.model)
            
            # Apply motion blur if enabled
            if self.blur_enabled:
                final_image = self.motion_blur.apply_blur(
                    self.image, 
                    self.floor_headsets, 
                    self.width, 
                    self.height, 
                    self.perspective_projection
                )
            else:
                final_image = self.image
            
            # Update display
            self.update_display(final_image)
            
            # Draw debug info
            if self.show_debug:
                self.draw_debug_info()

    def setup_scene(self):
        """Set up the scene with a clearly visible central rotating headset"""
        # Create a much larger central headset
        model = Model('data/headset.obj')
        model.normalizeGeometry()
        
        # Position it higher up and more forward for better visibility
        position = Vector(0, 8, -10)  # Higher up (y=8) and centered in view
        model.setPosition(position.x, position.y, position.z)
        
        # Make it larger (3x normal size)
        model.scale = [3.0, 3.0, 3.0]
        model.updateTransform()
        
        # Add a bright, distinctive color
        model.diffuse_color = (255, 215, 0)  # Gold color
        
        # Store the model directly
        self.central_headset = {
            "model": model,
            "rotation": [0, 0, 0],
            "position": position
        }
        
        # Print confirmation that central headset is set up
        print(f"Central headset created at position {position.x}, {position.y}, {position.z}")
        
        # Create floor headsets with different colors
        self.floor_headsets = self.create_floor_headsets()

    def update_central_headset(self, dt):
        """Update the rotating central headset with detailed debugging"""
        # Always print current rotation for debugging
        current_rot = self.central_headset["rotation"]
        print(f"Current rotation: Roll={math.degrees(current_rot[0]):.1f}°, Pitch={math.degrees(current_rot[1]):.1f}°, Yaw={math.degrees(current_rot[2]):.1f}°")
        
        if self.sensor_data and self.dr_filter:
            print(f"Using IMU data (point {self.current_data_index}/{len(self.sensor_data)})")
            # Use sensor data for rotation if available
            if self.current_data_index < len(self.sensor_data):
                sensor_data = self.sensor_data[self.current_data_index]
                self.current_data_index += 1
                
                # Update filter and get orientation
                _, orientation = self.dr_filter.update(sensor_data)
                
                # Convert quaternion to Euler angles for model rotation
                roll, pitch, yaw = self.dr_filter.get_euler_angles()
                
                # Print the calculated rotation from IMU data
                print(f"IMU rotation: Roll={math.degrees(roll):.1f}°, Pitch={math.degrees(pitch):.1f}°, Yaw={math.degrees(yaw):.1f}°")
                
                # Apply rotation directly to the model
                self.central_headset["model"].setRotation(roll, pitch, yaw)
                self.central_headset["rotation"] = [roll, pitch, yaw]
                print("Applied rotation to central headset model")
            else:
                # Reset to beginning of data when we reach the end
                print("Reached end of IMU data, resetting to beginning")
                self.current_data_index = 0
        else:
            # Fallback: simple rotation pattern with more noticeable movement
            print("Using fallback rotation pattern")
            self.central_headset["rotation"][0] += dt * 1.0  # Roll - faster
            self.central_headset["rotation"][1] += dt * 1.5  # Pitch - faster
            self.central_headset["rotation"][2] += dt * 0.8  # Yaw - faster
            
            # Apply rotation directly to the model
            model = self.central_headset["model"]
            model.setRotation(
                self.central_headset["rotation"][0],
                self.central_headset["rotation"][1],
                self.central_headset["rotation"][2]
            )
            model.updateTransform()  # Make sure transform is updated
            print("Applied fallback rotation to central headset model")

    def render_model(self, model_obj):
        """Render a 3D model with lighting using the approach from extreme_motion_blur"""
        # For CollisionObject, get the model inside
        if hasattr(model_obj, 'model'):
            model = model_obj.model
        else:
            model = model_obj
        
        # Calculate face normals
        faceNormals = {}
        for face in model.faces:
            p0 = model.getTransformedVertex(face[0])
            p1 = model.getTransformedVertex(face[1])
            p2 = model.getTransformedVertex(face[2])
            faceNormal = (p2-p0).cross(p1-p0).normalize()
            
            for i in face:
                if i not in faceNormals:
                    faceNormals[i] = []
                faceNormals[i].append(faceNormal)
        
        # Calculate vertex normals
        vertexNormals = []
        for vertIndex in range(len(model.vertices)):
            if vertIndex in faceNormals:
                normal = Vector(0, 0, 0)
                for adjNormal in faceNormals[vertIndex]:
                    normal = normal + adjNormal
                vertexNormals.append(normal / len(faceNormals[vertIndex]))
            else:
                vertexNormals.append(Vector(0, 1, 0))  # Default normal
        
        # Render all faces
        for face in model.faces:
            p0 = model.getTransformedVertex(face[0])
            p1 = model.getTransformedVertex(face[1])
            p2 = model.getTransformedVertex(face[2])
            n0, n1, n2 = [vertexNormals[i] for i in face]
            
            # Skip back-facing triangles
            cull = False
            avg_normal = (n0 + n1 + n2) / 3
            if avg_normal * self.light_dir < 0:
                cull = True
            
            if not cull:
                # Create points with positions, normals, and colors
                triangle_points = []
                for p, n in zip([p0, p1, p2], [n0, n1, n2]):
                    screenX, screenY = self.perspective_projection(p.x, p.y, p.z)
                    
                    # Calculate lighting intensity
                    intensity = max(0.2, n * self.light_dir)
                    
                    # Determine color - check if the model has a diffuse_color attribute
                    if hasattr(model, 'diffuse_color'):
                        # Model has a color - use it with lighting
                        r, g, b = model.diffuse_color
                        color = Color(
                            int(r * intensity), 
                            int(g * intensity), 
                            int(b * intensity), 
                            255
                        )
                    else:
                        # No color - use default white with lighting
                        color = Color(
                            int(255 * intensity),
                            int(255 * intensity),
                            int(255 * intensity),
                            255
                        )
                    
                    # Create point with position and calculated color
                    point = Point(screenX, screenY, p.z, color)
                    triangle_points.append(point)
                
                # Draw the triangle
                Triangle(
                    triangle_points[0],
                    triangle_points[1],
                    triangle_points[2]
                ).draw_faster(self.image, self.zBuffer)

    def render_floor_grid(self):
        """Render a grid on the floor for better visual reference"""
        grid_size = 20
        grid_step = 2
        grid_color = Color(80, 80, 100, 255)
        
        # Draw grid lines on the floor
        for x in range(-grid_size, grid_size + 1, grid_step):
            for z in range(-grid_size, grid_size + 1, grid_step):
                # Only draw lines at the edges of grid cells
                if x % grid_step == 0 or z % grid_step == 0:
                    # Horizontal and vertical lines on the floor
                    points = []
                    
                    # X-axis lines
                    if x % grid_step == 0:
                        p1 = Vector(x, 0, -grid_size)
                        p2 = Vector(x, 0, grid_size)
                        screen_p1 = self.perspective_projection(p1.x, p1.y, p1.z)
                        screen_p2 = self.perspective_projection(p2.x, p2.y, p2.z)
                        
                        if (0 <= screen_p1[0] < self.width and 0 <= screen_p1[1] < self.height and
                            0 <= screen_p2[0] < self.width and 0 <= screen_p2[1] < self.height):
                            points.append((screen_p1, screen_p2))
                    
                    # Z-axis lines
                    if z % grid_step == 0:
                        p1 = Vector(-grid_size, 0, z)
                        p2 = Vector(grid_size, 0, z)
                        screen_p1 = self.perspective_projection(p1.x, p1.y, p1.z)
                        screen_p2 = self.perspective_projection(p2.x, p2.y, p2.z)
                        
                        if (0 <= screen_p1[0] < self.width and 0 <= screen_p1[1] < self.height and
                            0 <= screen_p2[0] < self.width and 0 <= screen_p2[1] < self.height):
                            points.append((screen_p1, screen_p2))
                    
                    # Draw the lines
                    for p1, p2 in points:
                        pygame.draw.line(
                            self.screen,
                            (grid_color.r(), grid_color.g(), grid_color.b()),
                            p1, p2, 1
                        )

    def update_display(self, image):
        """Draw the image buffer to the screen"""
        # Convert the image buffer to a Pygame surface
        for y in range(self.height):
            for x in range(self.width):
                # Calculate the index in the buffer (accounting for image format)
                flipY = (self.height - y - 1)
                index = (flipY * self.width + x) * 4 + flipY + 1  # +1 for null byte
                
                # Extract RGB values
                if index + 2 < len(image.buffer):
                    r = image.buffer[index]
                    g = image.buffer[index + 1]
                    b = image.buffer[index + 2]
                    
                    # Set pixel on screen
                    self.screen.set_at((x, y), (r, g, b))

    def draw_debug_info(self):
        """Draw debug information on screen"""
        # Display FPS
        fps = pygame.time.Clock().get_fps()
        fps_text = self.font.render(f"FPS: {int(fps)}", True, (255, 255, 255))
        self.screen.blit(fps_text, (10, 10))
        
        # Display motion blur status
        blur_text = self.font.render(
            f"Motion Blur: {'ON' if self.blur_enabled else 'OFF'} (Strength: {self.motion_blur.blur_strength:.1f})",
            True, (255, 255, 255)
        )
        self.screen.blit(blur_text, (10, 35))
        
        # Display central headset rotation
        if self.central_headset:
            rot = self.central_headset["rotation"]
            rot_text = self.font.render(
                f"Rotation: Roll={math.degrees(rot[0]):.1f}°, Pitch={math.degrees(rot[1]):.1f}°, Yaw={math.degrees(rot[2]):.1f}°",
                True, (255, 255, 255)
            )
            self.screen.blit(rot_text, (10, 60))
        
        # Display controls
        controls_text = self.font.render(
            "B: Toggle blur | +/-: Adjust blur | R: Reset scene | P: Pause/Play | ESC: Quit",
            True, (200, 200, 200)
        )
        self.screen.blit(controls_text, (10, self.height - 30))

    def handle_events(self):
        """Handle user input events"""
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return False
            
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    return False
                
                elif event.key == pygame.K_b:
                    # Toggle motion blur
                    self.blur_enabled = not self.blur_enabled
                    print(f"Motion Blur: {'Enabled' if self.blur_enabled else 'Disabled'}")
                
                elif event.key == pygame.K_EQUALS or event.key == pygame.K_PLUS:
                    # Increase blur strength
                    self.motion_blur.blur_strength = min(5.0, self.motion_blur.blur_strength + 0.5)
                    print(f"Blur Strength: {self.motion_blur.blur_strength:.1f}")
                
                elif event.key == pygame.K_MINUS:
                    # Decrease blur strength
                    self.motion_blur.blur_strength = max(0.5, self.motion_blur.blur_strength - 0.5)
                    print(f"Blur Strength: {self.motion_blur.blur_strength:.1f}")
                
                elif event.key == pygame.K_r:
                    # Reset scene
                    self.floor_headsets = self.create_floor_headsets()
                    if self.sensor_data:
                        self.current_data_index = 0
                    print("Scene Reset")
                
                elif event.key == pygame.K_p:
                    # Pause/resume simulation
                    self.paused = not self.paused
                    print(f"Simulation {'Paused' if self.paused else 'Resumed'}")
                
                elif event.key == pygame.K_d:
                    # Toggle debug visualization
                    self.show_debug = not self.show_debug
                    print(f"Debug Info: {'Enabled' if self.show_debug else 'Disabled'}")
                
                elif event.key == pygame.K_f:
                    # Adjust friction
                    self.friction_coefficient = max(0.8, min(0.99, self.friction_coefficient + 0.01))
                    print(f"Friction Coefficient: {self.friction_coefficient:.2f}")
                
                elif event.key == pygame.K_v:
                    # Adjust friction
                    self.friction_coefficient = max(0.8, min(0.99, self.friction_coefficient - 0.01))
                    print(f"Friction Coefficient: {self.friction_coefficient:.2f}")
        
        return True

    def run(self):
        """Main loop to run the simulation"""
        clock = pygame.time.Clock()
        running = True
        
        print("VR Headset Physics Scene")
        print("------------------------")
        print("Controls:")
        print("  B: Toggle motion blur")
        print("  +/-: Adjust blur strength")
        print("  R: Reset scene")
        print("  P: Pause/resume simulation")
        print("  D: Toggle debug info")
        print("  F/V: Increase/decrease friction")
        print("  ESC: Quit")
        
        while running:
            # Handle timing
            dt = min(clock.tick(60) / 1000.0, 0.1)  # Cap at 0.1s to prevent physics jumps
            
            # Handle events
            running = self.handle_events()
            
            # Skip updates if paused
            if not self.paused:
                # Update central headset rotation
                self.update_central_headset(dt)
                
                # Update floor headsets physics
                self.update_floor_physics(dt)
            
            # Render the scene
            self.render_scene()
            
            # Update display
            pygame.display.flip()
        
        # Clean up
        pygame.quit()
        print("Simulation ended")

    # This function will be added to render.py to run the headset scene

def problem_6():
    """
    Renders a scene with a rotating VR headset in mid-air (using sensor data)
    and multiple headsets sliding on the floor with friction and collisions.
    """
    print("Running VR Headset Physics Scene")
    # Instantiate and run the headset scene
    scene = HeadsetScene(csv_path="../IMUData.csv")  # Adjust path as needed
    scene.run()

import numpy as np
import os
import pygame
import time
import subprocess

class VideoRecorder:
    """
    Helper class for recording PyGame simulations to video files.
    Supports both OpenCV and FFmpeg-based video saving.
    """
    def __init__(self, width, height, fps=30, output_dir="output"):
        self.width = width
        self.height = height
        self.fps = fps
        self.output_dir = output_dir
        self.frames = []
        self.is_recording = False
        self.frame_count = 0
        
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(os.path.join(output_dir, "frames"), exist_ok=True)
    
    def start_recording(self):
        """Start recording frames"""
        self.is_recording = True
        self.frames = []
        self.frame_count = 0
        print("Recording started")
    
    def stop_recording(self):
        """Stop recording frames"""
        self.is_recording = False
        print(f"Recording stopped. Captured {len(self.frames)} frames")
    
    def capture_frame(self, surface):
        """Capture current PyGame surface as a video frame"""
        if not self.is_recording:
            return
            
        # Capture the pygame surface as a numpy array
        frame = pygame.surfarray.array3d(surface)
        # Convert from HWC to CHW format
        frame = np.transpose(frame, (1, 0, 2))
        self.frames.append(frame)
        self.frame_count += 1
        
        # Print progress every 30 frames
        if self.frame_count % 30 == 0:
            print(f"Captured {self.frame_count} frames")
    
    def save_opencv_video(self, filename="simulation.mp4"):
        """Save recorded frames as a video file using OpenCV"""
        try:
            import cv2
            
            if not self.frames:
                print("No frames to save")
                return None
                
            print(f"Saving video with {len(self.frames)} frames...")
            
            # Create output video file
            output_path = os.path.join(self.output_dir, filename)
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_path, fourcc, self.fps, (self.width, self.height))
            
            # Write each frame
            for i, frame in enumerate(self.frames):
                # Convert from RGB to BGR for OpenCV
                frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                out.write(frame_bgr)
                
                # Print progress
                if i % 30 == 0:
                    print(f"Processing frame {i}/{len(self.frames)}")
            
            # Release the video writer
            out.release()
            print(f"Video saved to {output_path}")
            return output_path
            
        except ImportError:
            print("OpenCV not available. Using alternative method.")
            return self.save_frames_as_images()
    
    def save_frames_as_images(self, prefix="frame_"):
        """Save frames as individual PNG images"""
        if not self.frames:
            print("No frames to save")
            return None
            
        frames_dir = os.path.join(self.output_dir, "frames")
        os.makedirs(frames_dir, exist_ok=True)
        
        print(f"Saving {len(self.frames)} frames as images...")
        
        for i, frame in enumerate(self.frames):
            # Convert back to HWC format for PyGame
            frame = np.transpose(frame, (1, 0, 2))
            temp_surface = pygame.surfarray.make_surface(frame)
            
            # Save the surface as an image
            image_path = os.path.join(frames_dir, f"{prefix}{i:04d}.png")
            pygame.image.save(temp_surface, image_path)
            
            # Print progress
            if i % 30 == 0:
                print(f"Saving frame {i}/{len(self.frames)}")
        
        print(f"Frames saved to {frames_dir}/")
        return frames_dir
    
    def generate_ffmpeg_video(self, fps=None, quality="medium"):
        """
        Generate a video from saved frames using FFmpeg (if available)
        
        Args:
            fps: Frames per second (defaults to self.fps)
            quality: Video quality setting ('low', 'medium', 'high')
        
        Returns:
            Path to the created video file or None if failed
        """
        if fps is None:
            fps = self.fps
            
        # Save frames as images first
        frames_dir = self.save_frames_as_images()
        if not frames_dir:
            return None
            
        # Define quality presets for FFmpeg
        quality_settings = {
            "low": "-crf 28 -preset veryfast",
            "medium": "-crf 23 -preset medium",
            "high": "-crf 18 -preset slow"
        }
        
        # Use the specified quality or default to medium
        quality_args = quality_settings.get(quality, quality_settings["medium"])
        
        # Output video path
        output_video = os.path.join(self.output_dir, "simulation_ffmpeg.mp4")
        
        # FFmpeg command
        cmd = f"ffmpeg -y -framerate {fps} -i {frames_dir}/frame_%04d.png {quality_args} -pix_fmt yuv420p {output_video}"
        
        try:
            print("Generating video with FFmpeg...")
            print(f"Running command: {cmd}")
            
            # Run FFmpeg
            result = subprocess.run(cmd, shell=True, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            
            print("FFmpeg video generation completed")
            print(f"Video saved to {output_video}")
            return output_video
            
        except (subprocess.SubprocessError, FileNotFoundError) as e:
            print(f"Error generating video with FFmpeg: {e}")
            print("FFmpeg may not be installed or available in PATH")
            print("You can manually create video from the saved frames using:")
            print(f"ffmpeg -framerate {fps} -i {frames_dir}/frame_%04d.png -c:v libx264 -pix_fmt yuv420p {output_video}")
            return None
    
    def save_video(self, filename="simulation.mp4"):
        """
        Save recorded frames as a video file using OpenCV.
        """
        import cv2
        if not self.frames:
            print("No frames to save")
            return None

        try:
            print(f"Saving video with {len(self.frames)} frames using OpenCV...")

            # Create output video file
            output_path = os.path.join(self.output_dir, filename)
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Use 'mp4v' codec for MP4
            out = cv2.VideoWriter(output_path, fourcc, self.fps, (self.width, self.height))

            # Write each frame
            for i, frame in enumerate(self.frames):
                # Convert from RGB to BGR for OpenCV
                frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                out.write(frame_bgr)

                # Print progress every 30 frames
                if i % 30 == 0:
                    print(f"Processing frame {i}/{len(self.frames)}")

            # Release the video writer
            out.release()
            print(f"Video saved to {output_path}")
            return output_path

        except Exception as e:
            print(f"Error saving video with OpenCV: {e}")
            return None

    def clear_frames(self):
        """Clear recorded frames to free memory"""
        self.frames = []
        self.frame_count = 0
        print("Cleared all recorded frames")

def record_simulation_to_video(simulation_function, duration_seconds=10, fps=30, width=800, height=600):
    """
    Record any pygame-based simulation to a video file
    
    Args:
        simulation_function: Function that runs a single frame of the simulation
                            Should take dt (time delta) as input and return False when done
        duration_seconds: How long to record
        fps: Frames per second for recording
        width, height: Dimensions of the recording
    
    Returns:
        Path to the saved video file
    """
    # Initialize pygame
    pygame.init()
    screen = pygame.display.set_mode((width, height))
    pygame.display.set_caption("Recording Simulation")
    
    # Create video recorder
    recorder = VideoRecorder(width, height, fps=fps)
    recorder.start_recording()
    
    # Setup clock
    clock = pygame.time.Clock()
    running = True
    start_time = time.time()
    
    # Main loop
    print(f"Recording simulation for {duration_seconds} seconds at {fps} FPS...")
    
    try:
        while running:
            # Calculate time delta
            dt = clock.tick(fps) / 1000.0
            
            # Process events
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        running = False
            
            # Run simulation frame
            if simulation_function(dt) is False:
                # Simulation function returned False, so we're done
                running = False
            
            # Capture frame
            recorder.capture_frame(screen)
            
            # Update display
            pygame.display.flip()
            
            # Check if we've reached the duration
            elapsed = time.time() - start_time
            if elapsed >= duration_seconds:
                print(f"Reached recording duration of {duration_seconds} seconds")
                running = False
    
    finally:
        # Stop recording and save video
        recorder.stop_recording()
        video_path = recorder.save_video()
        
        # Clean up
        pygame.quit()
        print("Recording complete")
        
        return video_path

import pygame
import math
import numpy as np
from image import Image, Color
from model import Model, DeadReckoningFilter, CollisionObject, Matrix4, SensorDataParser, Vec4
from vector import Vector
from shape import Triangle, Point
import time
import os

class ImprovedHeadsetScene:
    """
    An optimized 3D scene with multiple VR headsets and video recording capability
    """
    def __init__(self, width=800, height=600, csv_path="IMUData.csv", record_video=False):
        # Initialize pygame
        pygame.init()
        self.width = width
        self.height = height
        self.screen = pygame.display.set_mode((width, height))
        pygame.display.set_caption("VR Headset Physics Scene")
        
        # Camera settings
        self.camera_follows_headset = True  # Set to True to make camera follow headset
        self.camera_offset = Vector(0, 2, -5)
        self.camera_pos = Vector(0, 10, -30)  # Initial camera position
        self.camera_target = Vector(0, 0, -10)  # Initial target position
        
        # Create buffer surface for faster rendering
        self.buffer_surface = pygame.Surface((width, height))
        
        # Image and Z-buffer for rendering
        self.image = Image(width, height, Color(20, 20, 40, 255))  # Dark blue background
        self.zBuffer = [-float('inf')] * width * height
        
        # Lighting
        self.light_dir = Vector(0.5, -1, -0.5).normalize()

        # Add a playback speed multiplier - process multiple IMU samples per frame
        self.imu_playback_speed = 5  # Process 5 samples per frame
        
        # Tracking for update rate
        self.frame_count = 0
        self.fps_history = []
        
        # Motion blur settings - reduced for better performance
        from extreme_motion_blur import ExtremeMotionBlur
        self.motion_blur = ExtremeMotionBlur(blur_strength=1.5)
        self.blur_enabled = True
        
        # Load sensor data for rotating headset
        self.csv_path = csv_path
        self.load_sensor_data()
        
        # Create scene objects
        self.central_headset = None
        self.floor_headsets = []
        self.setup_scene()
        
        # Physics settings
        self.friction_coefficient = 0.95  # Higher = less friction
        self.accumulator = 0  # For fixed timestep physics
        
        # Debug and control flags
        self.show_debug = True
        self.paused = False
        
        # Font for info display
        self.font = pygame.font.SysFont('Arial', 18)
        
        # Video recording setup
        self.record_video = record_video
        self.video_frames = []
        if record_video:
            # Create output directory if it doesn't exist
            os.makedirs("output", exist_ok=True)
            
    # In the update_camera() method, modify it to:

    def update_camera(self):
        """Update camera to stay fixed across the axis of the central rotating object"""
        if not self.central_headset:
            return
            
        # Get current headset position and orientation
        position = self.central_headset["position"]
        roll, pitch, yaw = self.central_headset["rotation"]
        
        # Set camera position slightly offset from the headset
        # We're fixing it on an axis, so we maintain a constant distance and position
        # relative to the object's center, regardless of its rotation
        distance = 15.0  # Fixed distance from headset
        height_offset = 5.0  # Fixed height above headset
        
        # Fixed camera position - doesn't change with rotation
        self.camera_pos = Vector(
            position.x,  # Same x coordinate as headset
            position.y + height_offset,  # Fixed height above headset
            position.z - distance  # Fixed distance behind headset
        )
        
        # Always target the headset's position
        self.camera_target = position
        
        # Keep light direction fixed relative to camera
        self.light_dir = Vector(0.5, -1, -0.5).normalize()

    def load_sensor_data(self):
        """Load and preprocess sensor data from CSV file"""
        try:
            parser = SensorDataParser(self.csv_path)
            self.sensor_data = parser.parse()
            print(f"Loaded {len(self.sensor_data)} sensor data points")
            
            # Create dead reckoning filter
            self.dr_filter = DeadReckoningFilter(alpha=0.98)
            # Calibrate using first 100 samples
            self.dr_filter.calibrate(self.sensor_data[:min(100, len(self.sensor_data))])
            
            # Precompute orientations for all sensor data
            # This will significantly speed up processing during runtime
            print("Precomputing orientations...")
            self.precomputed_orientations = []
            
            for i, data_point in enumerate(self.sensor_data):
                _, orientation = self.dr_filter.update(data_point)
                roll, pitch, yaw = self.dr_filter.get_euler_angles()
                self.precomputed_orientations.append((roll, pitch, yaw))
                
                # Print progress every 1000 samples
                if i % 1000 == 0:
                    print(f"Processed {i}/{len(self.sensor_data)} samples")
            
            print("Orientation precomputation complete")
            self.current_data_index = 0
        except Exception as e:
            print(f"Error loading sensor data: {e}")
            print("Using fallback rotation pattern instead")
            self.sensor_data = None
            self.dr_filter = None
            self.precomputed_orientations = None

    def create_floor_headsets(self):
        """Create multiple headsets that slide on the floor within boundaries"""
        import random  # Import at the top of the function
        headsets = []
        
        # Define boundary limits - must match physics boundaries
        boundary = {
            'min_x': -28.0,  # Slightly inside the physical boundaries
            'max_x': 28.0,   # to prevent starting too close to walls
            'min_z': -38.0,
            'max_z': -2.0    # Keep away from closest edge
        }
        
        # Simple RGB color tuples with brighter colors for better visibility
        colors = [
            (255, 0, 0),     # Red
            (0, 255, 0),     # Green
            (0, 0, 255),     # Blue
            (255, 255, 0),   # Yellow
            (255, 0, 255),   # Magenta
            (0, 255, 255),   # Cyan
            (255, 165, 0),   # Orange
            (128, 0, 128),   # Purple
            (220, 20, 60),   # Crimson
            (50, 205, 50),   # Lime Green
            (70, 130, 180),  # Steel Blue
            (255, 215, 0)    # Gold
        ]
        
        # Create headsets in a modified circle formation that stays within boundaries
        num_circle = 8
        # Use smaller circle radius to ensure headsets stay within bounds
        circle_radius = min(abs(boundary['min_x']), abs(boundary['max_x']), 
                           abs(boundary['min_z']), abs(boundary['max_z'])) * 0.6
        
        # Central point for the circle (offset from origin to fit in boundaries)
        center_x = 0
        center_z = (boundary['min_z'] + boundary['max_z']) / 2  # Middle of z range
        
        for i in range(num_circle):
            angle = (i / num_circle) * 2 * math.pi
            
            # Position in circle around the center point
            pos = Vector(
                center_x + circle_radius * math.cos(angle),
                1,  # Slightly above floor
                center_z + circle_radius * math.sin(angle)
            )
            
            # Ensure position is within boundaries
            pos.x = max(boundary['min_x'] + 2, min(boundary['max_x'] - 2, pos.x))
            pos.z = max(boundary['min_z'] + 2, min(boundary['max_z'] - 2, pos.z))
            
            # Velocity - more random for interesting collisions
            angle_offset = (i * 0.7) % (2 * math.pi)  # Varied angles
            speed = 3 + (i % 4)  # Different speeds (3-6)
            vel = Vector(
                math.cos(angle_offset) * speed,
                0,
                math.sin(angle_offset) * speed
            )
            
            model = Model('data/headset.obj')
            model.normalizeGeometry()
            model.setPosition(pos.x, pos.y, pos.z)
            
            # Assign color directly to the model
            model.diffuse_color = colors[i % len(colors)]
            
            # Create regular collision object
            headset = CollisionObject(model, pos, vel, radius=1.0)
            headsets.append(headset)
        
        # "Billiards break" pattern in the center
        triangle_size = 3  # Number of rows in triangle
        start_z = center_z - 5  # Position relative to center
        color_index = len(headsets) % len(colors)  # Continue color rotation
        
        for row in range(triangle_size):
            for col in range(row + 1):
                pos = Vector(
                    center_x + (col - row/2) * 2.5,  # Center the triangle, wider spacing
                    1,
                    start_z + row * 2.5  # More space between rows
                )
                
                # Small random initial velocity to make it less static
                vel = Vector(
                    (random.random() - 0.5) * 0.2,  # Small random x velocity
                    0,
                    (random.random() - 0.5) * 0.2   # Small random z velocity
                )
                
                model = Model('data/headset.obj')
                model.normalizeGeometry()
                model.setPosition(pos.x, pos.y, pos.z)
                
                # Assign color directly
                model.diffuse_color = colors[color_index % len(colors)]
                color_index += 1
                
                headset = CollisionObject(model, pos, vel, radius=1.0)
                headsets.append(headset)
        
        # Multiple "cue ball" headsets from different directions
        cue_positions = [
            # From back
            (center_x, 1, boundary['min_z'] + 5),
            # From left
            (boundary['min_x'] + 5, 1, center_z),
            # From right
            (boundary['max_x'] - 5, 1, center_z),
            # From corners
            (boundary['min_x'] + 5, 1, boundary['min_z'] + 5),
            (boundary['max_x'] - 5, 1, boundary['min_z'] + 5)
        ]
        
        for i, (x, y, z) in enumerate(cue_positions):
            pos = Vector(x, y, z)
            
            # Calculate velocity toward center
            direction = Vector(center_x - x, 0, center_z - z)
            length = math.sqrt(direction.x**2 + direction.z**2)
            if length > 0:
                direction.x /= length
                direction.z /= length
            
            # Set speed
            speed = 5 + (i % 3)  # Varied speeds
            vel = Vector(direction.x * speed, 0, direction.z * speed)
            
            model = Model('data/headset.obj')
            model.normalizeGeometry()
            model.setPosition(pos.x, pos.y, pos.z)
            
            # Alternate between white and a bright color
            if i % 2 == 0:
                model.diffuse_color = (255, 255, 255)  # White
            else:
                model.diffuse_color = colors[i % len(colors)]
            
            headsets.append(CollisionObject(model, pos, vel, radius=1.0))
        
        return headsets

    def update_floor_physics(self, dt):
        """Update physics for floor headsets with optimized approach, including boundary walls"""
        # Use a fixed time step for consistent physics
        fixed_dt = 1/60.0
        
        # Accumulate leftover time
        self.accumulator += dt
        
        # Define boundary walls - creates a contained area for the simulation
        # These values should create a visible box on the floor
        boundary = {
            'min_x': -30.0,
            'max_x': 30.0,
            'min_z': -40.0,  # Further away from camera
            'max_z': 0.0,    # Closer to camera
            'bounce_factor': 0.8  # Energy retained after bouncing (0.8 = 80%)
        }
        
        # Run physics updates with fixed timestep
        while self.accumulator >= fixed_dt:
            # Clear collision records
            for headset in self.floor_headsets:
                headset.clear_collision_history()
            
            # Apply gravity
            for headset in self.floor_headsets:
                headset.velocity.y -= 9.81 * fixed_dt
            
            # Optimized collision detection using spatial binning
            # Group objects by position to reduce collision checks
            spatial_grid = {}
            grid_size = 5.0  # Size of each grid cell
            
            # Place objects in grid cells
            for i, headset in enumerate(self.floor_headsets):
                grid_x = int(headset.position.x / grid_size)
                grid_y = int(headset.position.y / grid_size)
                grid_z = int(headset.position.z / grid_size)
                grid_key = (grid_x, grid_y, grid_z)
                
                if grid_key not in spatial_grid:
                    spatial_grid[grid_key] = []
                spatial_grid[grid_key].append(i)
            
            # Check collisions only between objects in same or adjacent cells
            for grid_key, indices in spatial_grid.items():
                grid_x, grid_y, grid_z = grid_key
                
                # Get all objects in current and neighboring cells
                nearby_indices = set(indices)
                for nx in range(grid_x-1, grid_x+2):
                    for ny in range(grid_y-1, grid_y+2):
                        for nz in range(grid_z-1, grid_z+2):
                            neighbor_key = (nx, ny, nz)
                            if neighbor_key in spatial_grid and neighbor_key != grid_key:
                                nearby_indices.update(spatial_grid[neighbor_key])
                
                # Check collisions between objects in this cell
                for i in indices:
                    for j in nearby_indices:
                        if i >= j:  # Skip duplicates and self-checks
                            continue
                            
                        # Quick distance check before detailed collision test
                        dx = self.floor_headsets[i].position.x - self.floor_headsets[j].position.x
                        dy = self.floor_headsets[i].position.y - self.floor_headsets[j].position.y
                        dz = self.floor_headsets[i].position.z - self.floor_headsets[j].position.z
                        dist_sq = dx*dx + dy*dy + dz*dz
                        
                        max_dist = self.floor_headsets[i].radius + self.floor_headsets[j].radius
                        if dist_sq < max_dist * max_dist:
                            if self.floor_headsets[i].check_collision(self.floor_headsets[j]):
                                self.floor_headsets[i].resolve_collision(self.floor_headsets[j])
            
            # Apply floor constraints, boundary walls, and friction
            for headset in self.floor_headsets:
                # Check if headset is on the floor
                is_on_floor = headset.position.y - headset.radius <= 0.01
                
                if is_on_floor:
                    # Ensure headset doesn't go below floor
                    headset.position.y = headset.radius
                    
                    # Apply friction to horizontal velocity
                    if headset.velocity.x**2 + headset.velocity.z**2 > 0.001:
                        headset.velocity.x *= self.friction_coefficient
                        headset.velocity.z *= self.friction_coefficient
                        
                        # Stop if very slow
                        if headset.velocity.x**2 + headset.velocity.z**2 < 0.05:
                            headset.velocity.x = 0
                            headset.velocity.z = 0
                
                # Apply boundary constraints (invisible walls)
                # X-axis boundaries (left/right walls)
                if headset.position.x - headset.radius < boundary['min_x']:
                    # Bounce off left wall
                    headset.position.x = boundary['min_x'] + headset.radius
                    headset.velocity.x = -headset.velocity.x * boundary['bounce_factor']
                elif headset.position.x + headset.radius > boundary['max_x']:
                    # Bounce off right wall
                    headset.position.x = boundary['max_x'] - headset.radius
                    headset.velocity.x = -headset.velocity.x * boundary['bounce_factor']
                
                # Z-axis boundaries (front/back walls)
                if headset.position.z - headset.radius < boundary['min_z']:
                    # Bounce off back wall (far from camera)
                    headset.position.z = boundary['min_z'] + headset.radius
                    headset.velocity.z = -headset.velocity.z * boundary['bounce_factor']
                elif headset.position.z + headset.radius > boundary['max_z']:
                    # Bounce off front wall (close to camera)
                    headset.position.z = boundary['max_z'] - headset.radius
                    headset.velocity.z = -headset.velocity.z * boundary['bounce_factor']
            
            # Update positions
            for headset in self.floor_headsets:
                headset.update(fixed_dt)
            
            self.accumulator -= fixed_dt

    def perspective_projection(self, x, y, z, width=None, height=None):
        """Calculate screen coordinates using the camera's view matrix"""
        # Use class width/height if not provided
        if width is None:
            width = self.width
        if height is None:
            height = self.height
        
        # Vector from camera to point
        rel_x = x - self.camera_pos.x
        rel_y = y - self.camera_pos.y
        rel_z = z - self.camera_pos.z
        
        # Calculate camera orientation vectors
        if hasattr(self, 'camera_target'):
            # Calculate view direction
            view_dir = Vector(
                self.camera_target.x - self.camera_pos.x,
                self.camera_target.y - self.camera_pos.y,
                self.camera_target.z - self.camera_pos.z
            ).normalize()
            
            # Calculate right vector
            world_up = Vector(0, 1, 0)
            right = view_dir.cross(world_up).normalize()
            
            # Calculate up vector
            up = right.cross(view_dir).normalize()
            
            # Project point into view space
            forward_dist = rel_x * view_dir.x + rel_y * view_dir.y + rel_z * view_dir.z
            right_dist = rel_x * right.x + rel_y * right.y + rel_z * right.z
            up_dist = rel_x * up.x + rel_y * up.y + rel_z * up.z
            
            # Skip if behind camera
            if forward_dist < 0.1:
                return -1, -1
                
            # Calculate perspective projection
            fov = math.pi / 3.0  # 60 degrees
            aspect = width / height
            
            # Convert to NDC coordinates
            right_ndc = right_dist / (forward_dist * math.tan(fov/2) * aspect)
            up_ndc = up_dist / (forward_dist * math.tan(fov/2))
            
            # Convert to screen coordinates
            screen_x = int((right_ndc + 1.0) * width / 2.0)
            screen_y = int((-up_ndc + 1.0) * height / 2.0)  # Invert Y
            
            return screen_x, screen_y
        else:
            # Fallback to basic projection
            if rel_z > -0.1:  # Skip if behind camera
                return -1, -1
                
            # Basic perspective division
            x_norm = rel_x / -rel_z
            y_norm = rel_y / -rel_z
            
            # Convert to screen coordinates
            screen_x = int((x_norm + 1.0) * width / 2.0)
            screen_y = int((y_norm + 1.0) * height / 2.0)
            
            return screen_x, screen_y

    def perspective_projection_old(self, x, y, z, width=None, height=None):
        """
        Convert 3D world coordinates to 2D screen coordinates with perspective,
        taking into account current camera position and orientation.
        """
        # Use class width/height if not provided
        if width is None:
            width = self.width
        if height is None:
            height = self.height
            
        # Vector from camera to point
        view_vector = Vector(
            x - self.camera_pos.x,
            y - self.camera_pos.y,
            z - self.camera_pos.z
        )
        
        # Camera parameters
        if hasattr(self, 'camera_target'):
            # Calculate camera direction vector (normalized)
            camera_dir = Vector(
                self.camera_target.x - self.camera_pos.x,
                self.camera_target.y - self.camera_pos.y,
                self.camera_target.z - self.camera_pos.z
            ).normalize()
            
            # Camera up vector (world up)
            up_vector = Vector(0, 1, 0)
            
            # Camera right vector (cross product of dir and up)
            right_vector = camera_dir.cross(up_vector).normalize()
            
            # Recalculate true up vector (to ensure orthogonality)
            true_up = right_vector.cross(camera_dir).normalize()
            
            # Project point onto camera vectors
            # These are the view-space coordinates
            x_view = right_vector * view_vector
            y_view = true_up * view_vector
            z_view = camera_dir * view_vector
            
            # Perspective parameters
            fov = math.pi / 3.0  # 60 degrees
            aspect = width / height
            
            # Check if point is behind camera
            if z_view < 0.1:
                return -1, -1  # Invalid screen coordinates
            
            # Apply perspective transformation
            x_ndc = x_view / (z_view * math.tan(fov/2) * aspect)
            y_ndc = y_view / (z_view * math.tan(fov/2))
            
            # Convert to screen coordinates
            screen_x = int((x_ndc + 1.0) * width / 2.0)
            screen_y = int((y_ndc + 1.0) * height / 2.0)
            
            return screen_x, screen_y
        else:
            # Fallback to basic perspective projection
            fov = math.pi / 3.0
            aspect = width / height
            
            # Transform to camera space (simple translation)
            rel_x = x - self.camera_pos.x
            rel_y = y - self.camera_pos.y
            rel_z = z - self.camera_pos.z
            
            # Skip if behind camera
            if rel_z > -0.1:
                return -1, -1
                
            # Simple perspective division
            x_norm = rel_x / -rel_z
            y_norm = rel_y / -rel_z
            
            # Convert to screen space
            screen_x = int((x_norm + 1.0) * width / 2.0)
            screen_y = int((y_norm + 1.0) * height / 2.0)
            
            return screen_x, screen_y
        
    def setup_scene(self):
        """Set up the scene with a clearly visible central rotating headset"""
        # Create the central headset
        model = Model('data/headset.obj')
        model.normalizeGeometry()
        
        # Position it at camera height, centered in view
        position = Vector(0, 5, -15)
        model.setPosition(position.x, position.y, position.z)
        
        # Make it larger for better visibility
        model.scale = [2.0, 2.0, 2.0]
        model.updateTransform()
        
        # Add a distinctive color
        model.diffuse_color = (255, 215, 0)  # Gold color
        
        # Store the model and position
        self.central_headset = {
            "model": model,
            "rotation": [0, 0, 0],
            "position": position
        }
        
        # Set camera position to fixed orientation relative to headset
        self.camera_pos = Vector(0, 10, -30)
        self.camera_target = position
        
        # Create floor headsets
        self.floor_headsets = self.create_floor_headsets()
    
    def setup_scene_old(self):
        """Set up the scene with a central rotating headset"""
        # Create a larger central headset
        # Create the central headset
        model = Model('data/headset.obj')
        model.normalizeGeometry()
        
        # Position it further from camera but still visible
        position = Vector(0, 5, -15)  # Centered, at camera height, further from camera
        model.setPosition(position.x, position.y, position.z)
        
        # Make it smaller
        model.scale = [0.3, 0.3, 0.3]  # Reduced scale
        model.updateTransform()
        
        # Add gold color
        model.diffuse_color = (255, 215, 0)
        
        # Store the model
        self.central_headset = {
            "model": model,
            "rotation": [0, 0, 0],
            "position": position
        }
        
        # Create floor headsets (unchanged)
        self.floor_headsets = self.create_floor_headsets()

    def update_central_headset_old(self):
        """Update the rotating central headset and ensure position is tracked correctly"""
        if self.precomputed_orientations:
            # Use multiple orientations per frame for smoother animation
            for _ in range(self.imu_playback_speed):
                if self.current_data_index < len(self.precomputed_orientations):
                    # Get precomputed orientation
                    roll, pitch, yaw = self.precomputed_orientations[self.current_data_index]
                    self.current_data_index += 1
                    
                    # Apply rotation to model
                    self.central_headset["model"].setRotation(roll, pitch, yaw)
                    self.central_headset["rotation"] = [roll, pitch, yaw]
                    
                    # IMPORTANT: Make sure to update the position Vector from the model's position
                    model = self.central_headset["model"]
                    self.central_headset["position"] = Vector(
                        model.trans[0], 
                        model.trans[1], 
                        model.trans[2]
                    )
                else:
                    # Reset to beginning of data when we reach the end
                    self.current_data_index = 0
        else:
            # Fallback: simple rotation pattern
            dt = 1/60.0  # Assume 60fps for consistent animation
            self.central_headset["rotation"][0] += dt * 1.0  # Roll
            self.central_headset["rotation"][1] += dt * 1.5  # Pitch
            self.central_headset["rotation"][2] += dt * 0.8  # Yaw
            
            # Apply rotation to model
            model = self.central_headset["model"]
            model.setRotation(
                self.central_headset["rotation"][0],
                self.central_headset["rotation"][1],
                self.central_headset["rotation"][2]
            )
            
            # IMPORTANT: Make sure to update the position Vector from the model's position
            self.central_headset["position"] = Vector(
                model.trans[0], 
                model.trans[1], 
                model.trans[2]
            )

    def update_central_headset(self, dt):
        """Update the rotating central headset with IMU data"""
        if self.sensor_data and self.dr_filter:
            # Use sensor data for rotation if available
            if self.current_data_index < len(self.sensor_data):
                # Process multiple samples per frame for smoother animation
                for _ in range(self.imu_playback_speed):
                    if self.current_data_index >= len(self.sensor_data):
                        break
                        
                    sensor_data = self.sensor_data[self.current_data_index]
                    self.current_data_index += 1
                    
                    # Update filter and get orientation
                    _, orientation = self.dr_filter.update(sensor_data)
                    
                    # Convert quaternion to Euler angles for model rotation
                    roll, pitch, yaw = self.dr_filter.get_euler_angles()
                    
                    # Apply rotation directly to the model
                    self.central_headset["model"].setRotation(roll, pitch, yaw)
                    self.central_headset["rotation"] = [roll, pitch, yaw]
                    
                    # Ensure position is updated from model's transform
                    model = self.central_headset["model"]
                    self.central_headset["position"] = Vector(
                        model.trans[0], 
                        model.trans[1], 
                        model.trans[2]
                    )
            else:
                # Reset to beginning of data when we reach the end
                self.current_data_index = 0
        else:
            # Fallback: simple rotation pattern with more noticeable movement
            self.central_headset["rotation"][0] += dt * 1.0  # Roll
            self.central_headset["rotation"][1] += dt * 1.5  # Pitch
            self.central_headset["rotation"][2] += dt * 0.8  # Yaw
            
            # Apply rotation directly to the model
            model = self.central_headset["model"]
            model.setRotation(
                self.central_headset["rotation"][0],
                self.central_headset["rotation"][1],
                self.central_headset["rotation"][2]
            )
            model.updateTransform()
            
            # Update position from model's transform
            self.central_headset["position"] = Vector(
                model.trans[0], 
                model.trans[1], 
                model.trans[2]
            )

    def render_scene(self):
        """Render the scene with camera fixed on central object's axis"""
        # Update camera to follow rotation of central headset
        self.update_camera_fixed_axis()
        
        # Store previous positions for motion blur
        if self.blur_enabled:
            self.motion_blur.update_object_positions(self.floor_headsets)
        
        # Clear image and z-buffer for new frame
        self.image = Image(self.width, self.height, Color(20, 20, 40, 255))
        self.zBuffer = [-float('inf')] * self.width * self.height
        
        # Render floor grid
        self.render_floor_grid()
        
        # Render central rotating headset
        if self.central_headset:
            self.render_model(self.central_headset["model"])
        
        # Render floor headsets
        for headset in self.floor_headsets:
            self.render_model(headset.model)
        
        # Apply motion blur if enabled
        if self.blur_enabled:
            final_image = self.motion_blur.apply_blur(
                self.image, 
                self.floor_headsets, 
                self.width, 
                self.height, 
                self.perspective_projection
            )
        else:
            final_image = self.image
        
        # Update display
        self.update_display(final_image)
        
        # Draw debug info
        if self.show_debug:
            self.draw_debug_info()

    def update_camera_fixed_axis(self):
        """Fix camera on the central object's axis as it rotates"""
        if not self.central_headset:
            return
        
        # Get current headset position and orientation
        position = self.central_headset["position"]
        roll, pitch, yaw = self.central_headset["rotation"]
        
        # Calculate direction vector from headset orientation
        # This converts Euler angles to a forward direction vector
        forward_x = -math.sin(yaw) * math.cos(pitch)
        forward_y = math.sin(pitch)
        forward_z = -math.cos(yaw) * math.cos(pitch)
        forward = Vector(forward_x, forward_y, forward_z).normalize()
        
        # Calculate up vector based on roll
        sin_roll = math.sin(roll)
        cos_roll = math.cos(roll)
        
        # Start with world up vector
        world_up = Vector(0, 1, 0)
        
        # Calculate right vector (perpendicular to forward and world up)
        right = forward.cross(world_up).normalize()
        
        # Calculate actual up vector (rolled)
        up = Vector(
            right.x * sin_roll + world_up.x * cos_roll,
            right.y * sin_roll + world_up.y * cos_roll,
            right.z * sin_roll + world_up.z * cos_roll
        ).normalize()
        
        # Distance from object to camera
        distance = 8.0
        
        # Position camera directly along the object's axis
        # We move it back along the forward vector to view from behind
        self.camera_pos = Vector(
            position.x - forward.x * distance,
            position.y - forward.y * distance,
            position.z - forward.z * distance
        )
        
        # Target the object's position
        self.camera_target = position
        
        # Update light direction relative to camera
        self.light_dir = forward  # 

    def render_scene_old(self):
        """Optimized scene rendering"""
        # Store previous positions for motion blur
        if self.blur_enabled:
            self.motion_blur.update_object_positions(self.floor_headsets)
        
        # Clear image and z-buffer for new frame
        self.image = Image(self.width, self.height, Color(20, 20, 40, 255))
        self.zBuffer = [-float('inf')] * self.width * self.height
        
        # Render floor grid
        self.render_floor_grid()
        
        # Render central rotating headset
        self.render_model(self.central_headset["model"])
        
        # Render floor headsets
        for headset in self.floor_headsets:
            self.render_model(headset.model)
        
        # Apply motion blur if enabled
        if self.blur_enabled:
            final_image = self.motion_blur.apply_blur(
                self.image, 
                self.floor_headsets, 
                self.width, 
                self.height, 
                self.perspective_projection
            )
        else:
            final_image = self.image
        
        # Update display
        self.update_display(final_image)
        
        # Capture frame for video if recording
        if self.record_video:
            self.capture_frame()
        
        # Draw debug info
        if self.show_debug:
            self.draw_debug_info()

    def render_model(self, model_obj):
        """Optimized 3D model rendering with frustum culling"""
        # Extract model
        if hasattr(model_obj, 'model'):
            model = model_obj.model
        else:
            model = model_obj
        
        # Quick frustum culling check
        model_pos = Vector(model.trans[0], model.trans[1], model.trans[2])
        
        # Skip if model is behind camera
        if model_pos.z > -1.0:
            return
            
        # Skip if model is too far away
        dist_sq = model_pos.x**2 + model_pos.y**2 + (model_pos.z + 30)**2
        if dist_sq > 2000:  # Arbitrary distance threshold
            return
        
        # Pre-compute all transformed vertices (eliminates redundant calculations)
        transformed_vertices = []
        for i in range(len(model.vertices)):
            transformed_vertices.append(model.getTransformedVertex(i))
        
        # Calculate face normals
        faceNormals = {}
        for face_idx, face in enumerate(model.faces):
            p0 = transformed_vertices[face[0]]
            p1 = transformed_vertices[face[1]]
            p2 = transformed_vertices[face[2]]
            faceNormal = (p2-p0).cross(p1-p0).normalize()
            
            for i in face:
                if i not in faceNormals:
                    faceNormals[i] = []
                faceNormals[i].append(faceNormal)
        
        # Calculate vertex normals
        vertexNormals = []
        for vertIndex in range(len(model.vertices)):
            if vertIndex in faceNormals:
                normal = Vector(0, 0, 0)
                for adjNormal in faceNormals[vertIndex]:
                    normal = normal + adjNormal
                vertexNormals.append(normal / len(faceNormals[vertIndex]))
            else:
                vertexNormals.append(Vector(0, 1, 0))  # Default normal
        
        # First filter visible faces before rendering
        visible_faces = []
        for face_idx, face in enumerate(model.faces):
            # Average face normal for back-face culling
            p0 = transformed_vertices[face[0]]
            p1 = transformed_vertices[face[1]]
            p2 = transformed_vertices[face[2]]
            face_normal = (p2-p0).cross(p1-p0).normalize()
            
            # Skip back-facing triangles (dot product with camera direction)
            view_vec = Vector(0, 0, 1)  # Camera looks along negative z-axis
            if face_normal * view_vec <= 0:
                visible_faces.append(face)
        
        # Get model color
        model_color = getattr(model, 'diffuse_color', (255, 255, 255))
        
        # Render only visible faces
        for face in visible_faces:
            p0 = transformed_vertices[face[0]]
            p1 = transformed_vertices[face[1]]
            p2 = transformed_vertices[face[2]]
            n0, n1, n2 = [vertexNormals[i] for i in face]
            
            # Create points with lighting
            triangle_points = []
            for p, n in zip([p0, p1, p2], [n0, n1, n2]):
                screenX, screenY = self.perspective_projection(p.x, p.y, p.z)
                
                # Calculate lighting
                intensity = max(0.2, n * self.light_dir)
                
                # Apply lighting to model color
                r, g, b = model_color
                color = Color(
                    int(r * intensity), 
                    int(g * intensity), 
                    int(b * intensity), 
                    255
                )
                
                # Create point
                point = Point(screenX, screenY, p.z, color)
                triangle_points.append(point)
            
            # Draw the triangle
            Triangle(
                triangle_points[0],
                triangle_points[1],
                triangle_points[2]
            ).draw_faster(self.image, self.zBuffer)

    def render_floor_grid(self):
        """Render a simplified grid on the floor and boundary walls"""
        grid_size = 20
        grid_step = 4  # Larger steps for better performance
        
        # Define boundary walls - must match values in update_floor_physics
        boundary = {
            'min_x': -30.0,
            'max_x': 30.0,
            'min_z': -40.0,
            'max_z': 0.0,
            'height': 5.0  # Height of the boundary walls
        }
        
        # Draw grid lines directly on the screen for better performance
        for x in range(-grid_size, grid_size + 1, grid_step):
            # X-axis lines
            p1 = Vector(x, 0, -grid_size)
            p2 = Vector(x, 0, grid_size)
            screen_p1 = self.perspective_projection(p1.x, p1.y, p1.z)
            screen_p2 = self.perspective_projection(p2.x, p2.y, p2.z)
            
            # Check if on screen
            if (0 <= screen_p1[0] < self.width and 0 <= screen_p1[1] < self.height and
                0 <= screen_p2[0] < self.width and 0 <= screen_p2[1] < self.height):
                pygame.draw.line(
                    self.screen,
                    (80, 80, 100),
                    screen_p1, screen_p2, 1
                )
        
        for z in range(-grid_size, grid_size + 1, grid_step):
            # Z-axis lines
            p1 = Vector(-grid_size, 0, z)
            p2 = Vector(grid_size, 0, z)
            screen_p1 = self.perspective_projection(p1.x, p1.y, p1.z)
            screen_p2 = self.perspective_projection(p2.x, p2.y, p2.z)
            
            # Check if on screen
            if (0 <= screen_p1[0] < self.width and 0 <= screen_p1[1] < self.height and
                0 <= screen_p2[0] < self.width and 0 <= screen_p2[1] < self.height):
                pygame.draw.line(
                    self.screen,
                    (80, 80, 100),
                    screen_p1, screen_p2, 1
                )
        
        # Render boundary walls with translucent effect
        wall_color = (100, 100, 180, 128)  # Light blue with transparency
        
        # Draw the four walls of the boundary box
        walls = [
            # Left wall (vertices in clockwise order)
            [
                Vector(boundary['min_x'], 0, boundary['min_z']),
                Vector(boundary['min_x'], boundary['height'], boundary['min_z']),
                Vector(boundary['min_x'], boundary['height'], boundary['max_z']),
                Vector(boundary['min_x'], 0, boundary['max_z'])
            ],
            # Right wall
            [
                Vector(boundary['max_x'], 0, boundary['min_z']),
                Vector(boundary['max_x'], 0, boundary['max_z']),
                Vector(boundary['max_x'], boundary['height'], boundary['max_z']),
                Vector(boundary['max_x'], boundary['height'], boundary['min_z'])
            ],
            # Back wall (far from camera)
            [
                Vector(boundary['min_x'], 0, boundary['min_z']),
                Vector(boundary['max_x'], 0, boundary['min_z']),
                Vector(boundary['max_x'], boundary['height'], boundary['min_z']),
                Vector(boundary['min_x'], boundary['height'], boundary['min_z'])
            ],
            # Front wall (close to camera)
            [
                Vector(boundary['min_x'], 0, boundary['max_z']),
                Vector(boundary['min_x'], boundary['height'], boundary['max_z']),
                Vector(boundary['max_x'], boundary['height'], boundary['max_z']),
                Vector(boundary['max_x'], 0, boundary['max_z'])
            ]
        ]
        
        # Draw each wall as a filled polygon
        for wall in walls:
            # Project 3D vertices to 2D screen coordinates
            screen_points = []
            for vertex in wall:
                screen_x, screen_y = self.perspective_projection(vertex.x, vertex.y, vertex.z)
                if 0 <= screen_x < self.width and 0 <= screen_y < self.height:
                    screen_points.append((screen_x, screen_y))
            
            # Only draw if we have at least 3 points and all are on screen
            if len(screen_points) >= 3:
                # Draw filled polygon with translucent effect
                s = pygame.Surface((self.width, self.height), pygame.SRCALPHA)
                pygame.draw.polygon(s, wall_color, screen_points)
                self.screen.blit(s, (0, 0))
                
                # Draw outline for better visibility
                pygame.draw.polygon(self.screen, (100, 100, 220), screen_points, 1)

    def update_display(self, image):
        """Optimized display update using Pygame's PixelArray"""
        try:
            # More efficient pixel array approach
            pixel_array = pygame.PixelArray(self.buffer_surface)
            
            # Batch process rows for efficiency
            for y in range(self.height):
                # Calculate flipped y-coordinate
                flipY = (self.height - y - 1)
                base_idx = (flipY * self.width * 4) + flipY + 1  # +1 for null byte
                
                for x in range(self.width):
                    idx = base_idx + (x * 4)
                    
                    # Ensure index is valid
                    if idx+2 < len(image.buffer):
                        r = image.buffer[idx]
                        g = image.buffer[idx+1]
                        b = image.buffer[idx+2]
                        pixel_array[x, y] = (r, g, b)
            
            del pixel_array  # Release lock
            
            # Blit buffer to screen
            self.screen.blit(self.buffer_surface, (0, 0))
        except:
            # Fallback to safer but slower method
            for y in range(self.height):
                for x in range(self.width):
                    flipY = (self.height - y - 1)
                    idx = (flipY * self.width + x) * 4 + flipY + 1
                    
                    if idx+2 < len(image.buffer):
                        r = image.buffer[idx]
                        g = image.buffer[idx+1]
                        b = image.buffer[idx+2]
                        self.screen.set_at((x, y), (r, g, b))

    def capture_frame(self):
        """Capture current screen as a frame for video recording"""
        frame = pygame.surfarray.array3d(self.screen)
        # Convert from HWC to CHW format (width, height, channels) to (channels, height, width)
        frame = np.transpose(frame, (1, 0, 2))
        self.video_frames.append(frame)

    def save_video(self, fps=30):
        """Save all captured frames as a video file"""
        if not self.video_frames:
            print("No frames to save")
            return
            
        try:
            import cv2
            
            print(f"Saving video with {len(self.video_frames)} frames...")
            
            # Create output video file
            output_path = "output/headset_physics.mp4"
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_path, fourcc, fps, (self.width, self.height))
            
            # Write each frame
            for i, frame in enumerate(self.video_frames):
                # Convert from RGB to BGR for OpenCV
                frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                out.write(frame_bgr)
                
                # Print progress
                if i % 10 == 0:
                    print(f"Processing frame {i}/{len(self.video_frames)}")
            
            # Release the video writer
            out.release()
            print(f"Video saved to {output_path}")
            
        except ImportError:
            print("OpenCV not available. Saving frames as images instead...")
            
            # Save frames as individual images
            os.makedirs("output/frames", exist_ok=True)
            
            for i, frame in enumerate(self.video_frames):
                # Create a surface from the array
                frame = np.transpose(frame, (1, 0, 2))  # Convert back to HWC
                temp_surface = pygame.surfarray.make_surface(frame)
                
                # Save the surface as an image
                pygame.image.save(temp_surface, f"output/frames/frame_{i:04d}.png")
                
                # Print progress
                if i % 10 == 0:
                    print(f"Saving frame {i}/{len(self.video_frames)}")
            
            print(f"Frames saved to output/frames/ directory")
            print("To create a video, use a tool like FFmpeg:")
            print("ffmpeg -framerate 30 -i output/frames/frame_%04d.png -c:v libx264 -pix_fmt yuv420p output/headset_physics.mp4")

    def draw_debug_info(self):
        """Draw performance and debug info on screen"""
        # Calculate average FPS
        current_fps = 0
        if self.fps_history:
            current_fps = len(self.fps_history) / sum(self.fps_history) if sum(self.fps_history) > 0 else 0
        
        # Display FPS
        fps_text = self.font.render(f"FPS: {int(current_fps)}", True, (255, 255, 255))
        self.screen.blit(fps_text, (10, 10))
        
        # Display motion blur status
        blur_text = self.font.render(
            f"Motion Blur: {'ON' if self.blur_enabled else 'OFF'} (Strength: {self.motion_blur.blur_strength:.1f})",
            True, (255, 255, 255)
        )
        self.screen.blit(blur_text, (10, 35))
        
        # Display object counts and IMU speed
        count_text = self.font.render(
            f"Objects: {len(self.floor_headsets) + 1} | Frame: {self.frame_count} | IMU Speed: {self.imu_playback_speed}x",
            True, (255, 255, 255)
        )
        self.screen.blit(count_text, (10, 60))
        
        # Display video recording status if active
        if self.record_video:
            rec_text = self.font.render(
                f"RECORDING: {len(self.video_frames)} frames",
                True, (255, 0, 0)
            )
            self.screen.blit(rec_text, (self.width - 250, 10))
        
        # Display controls
        controls_text = self.font.render(
            "B: Blur | +/-: Strength | R: Reset | V: Record | U/J: Speed | ESC: Quit",
            True, (200, 200, 200)
        )
        self.screen.blit(controls_text, (10, self.height - 30))

    def handle_events(self):
        """Handle user input events"""
        events = pygame.event.get()
        for event in events:
            if event.type == pygame.QUIT:
                return False
            
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    return False
                
                elif event.key == pygame.K_b:
                    # Toggle motion blur
                    self.blur_enabled = not self.blur_enabled
                    print(f"Motion Blur: {'Enabled' if self.blur_enabled else 'Disabled'}")
                
                elif event.key == pygame.K_EQUALS or event.key == pygame.K_PLUS:
                    # Increase blur strength
                    self.motion_blur.blur_strength = min(5.0, self.motion_blur.blur_strength + 0.5)
                    print(f"Blur Strength: {self.motion_blur.blur_strength:.1f}")
                
                elif event.key == pygame.K_MINUS:
                    # Decrease blur strength
                    self.motion_blur.blur_strength = max(0.5, self.motion_blur.blur_strength - 0.5)
                    print(f"Blur Strength: {self.motion_blur.blur_strength:.1f}")
                
                elif event.key == pygame.K_r:
                    # Reset scene
                    self.floor_headsets = self.create_floor_headsets()
                    if self.precomputed_orientations:
                        self.current_data_index = 0
                    print("Scene Reset")
                
                elif event.key == pygame.K_p:
                    # Pause/resume simulation
                    self.paused = not self.paused
                    print(f"Simulation {'Paused' if self.paused else 'Resumed'}")
                
                elif event.key == pygame.K_v:
                    # Toggle video recording
                    if not self.record_video:
                        self.record_video = True
                        print("Started video recording")
                    else:
                        self.record_video = False
                        print("Stopped video recording - saving video...")
                        self.save_video()
                
                elif event.key == pygame.K_u:
                    # Increase IMU playback speed
                    self.imu_playback_speed = min(20, self.imu_playback_speed + 1)
                    print(f"IMU Playback Speed: {self.imu_playback_speed}")
        
        return True
        
    def run(self):
        """Main loop to run the simulation with performance monitoring"""
        clock = pygame.time.Clock()
        running = True
        
        print("Optimized VR Headset Physics Scene")
        print("----------------------------------")
        print("Controls:")
        print("  B: Toggle motion blur")
        print("  +/-: Adjust blur strength")
        print("  R: Reset scene")
        print("  P: Pause/resume simulation")
        print("  V: Toggle video recording")
        print("  U/J: Increase/decrease IMU playback speed")
        print("  ESC: Quit")
        
        # For tracking performance
        frame_times = []
        
        try:
            while running:
                # Start frame timing
                frame_start = time.time()
                
                # Handle timing - Cap at 60fps
                dt = min(clock.tick(60) / 1000.0, 0.1)
                
                # Track frame time for FPS calculation
                self.fps_history.append(dt)
                if len(self.fps_history) > 20:
                    self.fps_history.pop(0)
                
                # Handle events
                running = self.handle_events()
                
                # Skip updates if paused
                if not self.paused:
                    # Update central headset rotation - process multiple IMU samples per frame
                    self.update_central_headset()
                    
                    # self.update_camera()
                    
                    # Update floor headsets physics
                    self.update_floor_physics(dt)
                
                self.update_camera()
                
                # Render the scene
                self.render_scene()
                
                # Update display
                pygame.display.flip()
                
                # Increment frame counter
                self.frame_count += 1
                
                # Calculate frame time for performance monitoring
                frame_end = time.time()
                frame_time = frame_end - frame_start
                frame_times.append(frame_time)
                
                # Periodically show performance stats
                if self.frame_count % 60 == 0:
                    avg_frame_time = sum(frame_times) / len(frame_times) if frame_times else 0
                    avg_fps = 1.0 / avg_frame_time if avg_frame_time > 0 else 0
                    print(f"Frame {self.frame_count}: {avg_fps:.1f} FPS (avg {avg_frame_time*1000:.1f}ms per frame)")
                    frame_times = []
        
        except KeyboardInterrupt:
            print("Interrupted by user")
        
        finally:
            # Always save video if we were recording
            if self.record_video and self.video_frames:
                print("Saving video before exit...")
                self.save_video()
            
            # Clean up
            pygame.quit()
            print(f"Simulation ended after {self.frame_count} frames")

def run_headset_simulation(record_video=False, auto_save=False, duration_seconds=None, playback_speed=5):
    """
    Run the headset simulation with improved performance.
    
    Args:
        record_video: Whether to record a video of the simulation
        auto_save: If True, automatically save the video after duration_seconds
        duration_seconds: How long to run before auto-saving (if auto_save is True)
        playback_speed: Initial IMU data playback speed (samples per frame)
    """
    print("Starting optimized headset simulation...")
    
    # Create the scene
    scene = ImprovedHeadsetScene(csv_path="../IMUData.csv", record_video=record_video)
    
    # Set initial playback speed
    scene.imu_playback_speed = playback_speed
    print(f"IMU playback speed set to {playback_speed}x")
    
    if auto_save and duration_seconds is not None:
        # Run for a specific duration then auto-save
        print(f"Will auto-save video after {duration_seconds} seconds")
        
        # Start the simulation in a way that allows us to exit after the duration
        clock = pygame.time.Clock()
        running = True
        start_time = time.time()
        
        try:
            while running:
                # Handle timing
                dt = min(clock.tick(60) / 1000.0, 0.1)
                
                # Handle events (allow manual exit)
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        running = False
                    elif event.type == pygame.KEYDOWN:
                        if event.key == pygame.K_ESCAPE:
                            running = False
                
                # Update central headset
                scene.update_central_headset(dt)
                
                # Update physics
                scene.update_floor_physics(dt)
                
                # Render scene
                scene.render_scene()
                
                # Update display
                pygame.display.flip()
                
                # Increment frame counter
                scene.frame_count += 1
                
                # Check if we've reached the duration
                elapsed = time.time() - start_time
                if elapsed >= duration_seconds:
                    print(f"Reached duration of {duration_seconds} seconds")
                    running = False
            
            # Save the video
            print("Saving video...")
            scene.save_video()
            
        finally:
            pygame.quit()
            print(f"Auto-save simulation ended after {scene.frame_count} frames")
    else:
        # Run the normal interactive simulation
        scene.run()

def problem_6_improved(auto_record=True, duration=20, playback_speed=5):
    if auto_record:
        # Run in auto-record mode for specified seconds then save video
        run_headset_simulation(record_video=True, auto_save=True, duration_seconds=duration)
    else:
        # Run in interactive mode
        run_headset_simulation(record_video=False)

# problem_6_improved(auto_record=True, duration=2000, playback_speed=5)

import pygame
from image import Image, Color
from model import Model, DeadReckoningFilter, SensorDataParser, CollisionObject
from vector import Vector
from shape import Triangle, Point
import math
import random

class FixedAxisCameraScene:
    def __init__(self, width=800, height=600, csv_path="../IMUData.csv"):
        # Initialize pygame
        pygame.init()
        self.width = width
        self.height = height
        self.screen = pygame.display.set_mode((width, height))
        pygame.display.set_caption("Fixed-Axis Camera Demo")
        
        # Image and Z-buffer
        self.image = Image(width, height, Color(20, 20, 40, 255))
        self.zBuffer = [-float('inf')] * width * height
        
        # Camera settings
        self.camera_pos = Vector(0, 5, -25)
        self.camera_target = Vector(0, 5, -15)
        
        # Lighting
        self.light_dir = Vector(0.5, -0.7, -0.5).normalize()
        
        # Load sensor data
        self.csv_path = csv_path
        self.load_sensor_data()
        
        # Setup scene objects
        self.central_headset = None
        self.floor_headsets = []
        self.setup_scene()
        
        # Physics settings
        self.friction_coefficient = 0.95  # Higher = less friction
        self.accumulator = 0  # For fixed timestep physics
        
        # Control flags
        self.paused = False
        self.show_debug = True
        
        # Font for info display
        self.font = pygame.font.SysFont('Arial', 18)
        
        # Frame counter
        self.frame_count = 0
    
    def load_sensor_data(self):
        """Load and preprocess sensor data from CSV file"""
        try:
            parser = SensorDataParser(self.csv_path)
            self.sensor_data = parser.parse()
            print(f"Loaded {len(self.sensor_data)} sensor data points")
            
            # Create dead reckoning filter
            self.dr_filter = DeadReckoningFilter(alpha=0.98)
            # Calibrate using first 100 samples
            self.dr_filter.calibrate(self.sensor_data[:min(100, len(self.sensor_data))])
            
            self.current_data_index = 0
        except Exception as e:
            print(f"Error loading sensor data: {e}")
            print("Using fallback rotation pattern instead")
            self.sensor_data = None
            self.dr_filter = None
    
    def create_floor_headsets(self):
        """Create multiple headsets that slide on the floor"""
        headsets = []
        
        # Colors for headsets
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
        
        # Create headsets in a circle formation
        num_circle = 8
        circle_radius = 15
        for i in range(num_circle):
            angle = (i / num_circle) * 2 * math.pi
            
            # Position in circle
            pos = Vector(
                circle_radius * math.cos(angle),
                1,  # Slightly above floor
                circle_radius * math.sin(angle) - 15  # Centered at z=-15
            )
            
            # Velocity toward center
            speed = 2 + (i % 3)  # Different speeds
            vel = Vector(
                -math.cos(angle) * speed,
                0,
                -math.sin(angle) * speed
            )
            
            model = Model('data/headset.obj')
            model.normalizeGeometry()
            model.setPosition(pos.x, pos.y, pos.z)
            
            # Assign color
            model.diffuse_color = colors[i % len(colors)]
            
            # Create collision object
            headset = CollisionObject(model, pos, vel, radius=1.0)
            headsets.append(headset)
        
        # Add a "billiards break" pattern
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
                # Stationary initially
                vel = Vector(
                    (random.random() - 0.5) * 0.2,  # Small random velocity
                    0,
                    (random.random() - 0.5) * 0.2
                )
                
                model = Model('data/headset.obj')
                model.normalizeGeometry()
                model.setPosition(pos.x, pos.y, pos.z)
                
                # Assign color
                model.diffuse_color = colors[color_index % len(colors)]
                color_index += 1
                
                headset = CollisionObject(model, pos, vel, radius=1.0)
                headsets.append(headset)
        
        # Add a "cue ball" headset
        pos = Vector(0, 1, -25)  # Behind the triangle
        vel = Vector(0, 0, 8)    # Moving forward
        
        model = Model('data/headset.obj')
        model.normalizeGeometry()
        model.setPosition(pos.x, pos.y, pos.z)
        model.diffuse_color = (255, 255, 255)  # White
        
        headsets.append(CollisionObject(model, pos, vel, radius=1.0))
        
        return headsets
    
    def setup_scene(self):
        """Set up the scene with a central rotating headset and floor headsets"""
        # Create central headset
        model = Model('data/headset.obj')
        model.normalizeGeometry()
        
        # Position it for good visibility
        position = Vector(0, 5, -15)
        model.setPosition(position.x, position.y, position.z)
        
        # Distinctive color
        model.diffuse_color = (255, 215, 0)  # Gold
        
        # Store model details
        self.central_headset = {
            "model": model,
            "rotation": [0, 0, 0],
            "position": position
        }
        
        # Create floor headsets
        self.floor_headsets = self.create_floor_headsets()
    
    def update_central_headset(self, dt):
        """Update the rotating central headset with IMU data"""
        if self.sensor_data and self.dr_filter:
            # Use sensor data for rotation
            if self.current_data_index < len(self.sensor_data):
                sensor_data = self.sensor_data[self.current_data_index]
                self.current_data_index += 1
                
                # Update filter and get orientation
                _, orientation = self.dr_filter.update(sensor_data)
                
                # Convert quaternion to Euler angles
                roll, pitch, yaw = self.dr_filter.get_euler_angles()
                
                # Apply rotation to model
                self.central_headset["model"].setRotation(roll, pitch, yaw)
                self.central_headset["rotation"] = [roll, pitch, yaw]
            else:
                # Reset to beginning when data ends
                self.current_data_index = 0
        else:
            # Fallback rotation pattern
            self.central_headset["rotation"][0] += dt * 0.5  # Roll
            self.central_headset["rotation"][1] += dt * 0.7  # Pitch
            self.central_headset["rotation"][2] += dt * 0.3  # Yaw
            
            # Apply rotation to model
            model = self.central_headset["model"]
            model.setRotation(
                self.central_headset["rotation"][0],
                self.central_headset["rotation"][1],
                self.central_headset["rotation"][2]
            )
    
    def update_camera_fixed_axis(self):
        """Fix camera on the central object's axis as it rotates"""
        if not self.central_headset:
            return
        
        # Get object's rotation
        roll, pitch, yaw = self.central_headset["rotation"]
        position = self.central_headset["position"]
        
        # Create rotation matrices for each axis
        # Roll (X-axis rotation)
        cos_roll, sin_roll = math.cos(roll), math.sin(roll)
        roll_matrix = [
            [1, 0, 0],
            [0, cos_roll, -sin_roll],
            [0, sin_roll, cos_roll]
        ]
        
        # Pitch (Y-axis rotation)
        cos_pitch, sin_pitch = math.cos(pitch), math.sin(pitch)
        pitch_matrix = [
            [cos_pitch, 0, sin_pitch],
            [0, 1, 0],
            [-sin_pitch, 0, cos_pitch]
        ]
        
        # Yaw (Z-axis rotation)
        cos_yaw, sin_yaw = math.cos(yaw), math.sin(yaw)
        yaw_matrix = [
            [cos_yaw, -sin_yaw, 0],
            [sin_yaw, cos_yaw, 0],
            [0, 0, 1]
        ]
        
        # Calculate view direction vector by applying rotation matrices
        # Start with initial view vector pointing along negative z-axis
        view_vec = [0, 0, -1]
        
        # Apply yaw (z-axis rotation)
        view_vec = [
            yaw_matrix[0][0] * view_vec[0] + yaw_matrix[0][1] * view_vec[1] + yaw_matrix[0][2] * view_vec[2],
            yaw_matrix[1][0] * view_vec[0] + yaw_matrix[1][1] * view_vec[1] + yaw_matrix[1][2] * view_vec[2],
            yaw_matrix[2][0] * view_vec[0] + yaw_matrix[2][1] * view_vec[1] + yaw_matrix[2][2] * view_vec[2]
        ]
        
        # Apply pitch (y-axis rotation)
        view_vec = [
            pitch_matrix[0][0] * view_vec[0] + pitch_matrix[0][1] * view_vec[1] + pitch_matrix[0][2] * view_vec[2],
            pitch_matrix[1][0] * view_vec[0] + pitch_matrix[1][1] * view_vec[1] + pitch_matrix[1][2] * view_vec[2],
            pitch_matrix[2][0] * view_vec[0] + pitch_matrix[2][1] * view_vec[1] + pitch_matrix[2][2] * view_vec[2]
        ]
        
        # Apply roll (x-axis rotation)
        view_vec = [
            roll_matrix[0][0] * view_vec[0] + roll_matrix[0][1] * view_vec[1] + roll_matrix[0][2] * view_vec[2],
            roll_matrix[1][0] * view_vec[0] + roll_matrix[1][1] * view_vec[1] + roll_matrix[1][2] * view_vec[2],
            roll_matrix[2][0] * view_vec[0] + roll_matrix[2][1] * view_vec[1] + roll_matrix[2][2] * view_vec[2]
        ]
        
        # Convert to Vector
        view_direction = Vector(view_vec[0], view_vec[1], view_vec[2]).normalize()
        
        # Position camera at a fixed distance along the view direction
        # This is the key part - we place the camera so it's looking directly
        # along the object's axis
        distance = 10.0
        self.camera_pos = Vector(
            position.x - view_direction.x * distance,
            position.y - view_direction.y * distance,
            position.z - view_direction.z * distance
        )
        
        # The camera always targets the object position
        self.camera_target = position
    
    def update_floor_physics(self, dt):
        """Update physics for floor headsets with collisions and friction"""
        # Use a fixed time step for physics
        fixed_dt = 1/60.0
        
        # Accumulate leftover time
        self.accumulator += dt
        
        # Define boundary limits
        boundary = {
            'min_x': -30.0,
            'max_x': 30.0,
            'min_z': -40.0,
            'max_z': 0.0,
            'bounce_factor': 0.8  # Energy retained after bounce
        }
        
        # Run physics updates with fixed timestep
        while self.accumulator >= fixed_dt:
            # Clear collision records
            for headset in self.floor_headsets:
                headset.clear_collision_history()
            
            # Apply gravity
            for headset in self.floor_headsets:
                headset.velocity.y -= 9.81 * fixed_dt
            
            # Check collisions between headsets
            for i in range(len(self.floor_headsets)):
                for j in range(i + 1, len(self.floor_headsets)):
                    # Quick distance check
                    dx = self.floor_headsets[i].position.x - self.floor_headsets[j].position.x
                    dy = self.floor_headsets[i].position.y - self.floor_headsets[j].position.y
                    dz = self.floor_headsets[i].position.z - self.floor_headsets[j].position.z
                    dist_sq = dx*dx + dy*dy + dz*dz
                    
                    # Only check collision if objects are close enough
                    max_dist = self.floor_headsets[i].radius + self.floor_headsets[j].radius
                    if dist_sq < max_dist * max_dist * 1.5:
                        if self.floor_headsets[i].check_collision(self.floor_headsets[j]):
                            self.floor_headsets[i].resolve_collision(self.floor_headsets[j])
            
            # Apply floor constraints and friction
            for headset in self.floor_headsets:
                # Check if headset is on the floor
                is_on_floor = headset.position.y - headset.radius <= 0.01
                
                if is_on_floor:
                    # Ensure headset doesn't go below floor
                    headset.position.y = headset.radius
                    
                    # Apply friction to horizontal velocity
                    horizontal_speed_squared = (
                        headset.velocity.x**2 + 
                        headset.velocity.z**2
                    )
                    
                    if horizontal_speed_squared > 0.001:
                        # Apply friction
                        headset.velocity.x *= self.friction_coefficient
                        headset.velocity.z *= self.friction_coefficient
                        
                        # Stop if very slow
                        if horizontal_speed_squared < 0.05:
                            headset.velocity.x = 0
                            headset.velocity.z = 0
                
                # Apply boundary constraints
                # X-axis boundaries
                if headset.position.x - headset.radius < boundary['min_x']:
                    headset.position.x = boundary['min_x'] + headset.radius
                    headset.velocity.x = -headset.velocity.x * boundary['bounce_factor']
                elif headset.position.x + headset.radius > boundary['max_x']:
                    headset.position.x = boundary['max_x'] - headset.radius
                    headset.velocity.x = -headset.velocity.x * boundary['bounce_factor']
                
                # Z-axis boundaries
                if headset.position.z - headset.radius < boundary['min_z']:
                    headset.position.z = boundary['min_z'] + headset.radius
                    headset.velocity.z = -headset.velocity.z * boundary['bounce_factor']
                elif headset.position.z + headset.radius > boundary['max_z']:
                    headset.position.z = boundary['max_z'] - headset.radius
                    headset.velocity.z = -headset.velocity.z * boundary['bounce_factor']
            
            # Update positions
            for headset in self.floor_headsets:
                headset.update(fixed_dt)
            
            self.accumulator -= fixed_dt
    
    def perspective_projection(self, x, y, z):
        """Advanced perspective projection with proper view matrix"""
        # Vector from camera to point
        to_point = Vector(
            x - self.camera_pos.x,
            y - self.camera_pos.y,
            z - self.camera_pos.z
        )
        
        # Camera orientation vectors
        forward = Vector(
            self.camera_target.x - self.camera_pos.x,
            self.camera_target.y - self.camera_pos.y,
            self.camera_target.z - self.camera_pos.z
        ).normalize()
        
        # Define camera's up vector (world up)
        world_up = Vector(0, 1, 0)
        
        # Calculate camera's right vector (perpendicular to forward and up)
        right = forward.cross(world_up).normalize()
        
        # Calculate true up vector (perpendicular to forward and right)
        up = right.cross(forward).normalize()
        
        # Project point onto camera orientation vectors
        right_component = to_point.x * right.x + to_point.y * right.y + to_point.z * right.z
        up_component = to_point.x * up.x + to_point.y * up.y + to_point.z * up.z
        forward_component = to_point.x * forward.x + to_point.y * forward.y + to_point.z * forward.z
        
        # Skip if point is behind camera
        if forward_component < 0.1:
            return -1, -1
        
        # Apply perspective projection
        fov = math.pi / 3.0  # 60 degrees
        aspect = self.width / self.height
        
        # Convert to NDC coordinates (-1 to 1)
        x_ndc = right_component / (forward_component * math.tan(fov/2) * aspect)
        y_ndc = up_component / (forward_component * math.tan(fov/2))
        
        # Convert to screen coordinates
        screen_x = int((x_ndc + 1.0) * self.width / 2.0)
        screen_y = int((-y_ndc + 1.0) * self.height / 2.0)  # Flip Y
        
        return screen_x, screen_y
    
    def render_model(self, model):
        """Render a 3D model with lighting"""
        # Get the model object
        if hasattr(model, 'model'):
            model = model.model
            
        # Precalculate transformed vertices
        transformed_vertices = []
        for i in range(len(model.vertices)):
            transformed_vertices.append(model.getTransformedVertex(i))
        
        # Calculate face normals
        faceNormals = {}
        for face in model.faces:
            p0 = transformed_vertices[face[0]]
            p1 = transformed_vertices[face[1]]
            p2 = transformed_vertices[face[2]]
            faceNormal = (p2-p0).cross(p1-p0).normalize()
            
            for i in face:
                if i not in faceNormals:
                    faceNormals[i] = []
                faceNormals[i].append(faceNormal)
        
        # Calculate vertex normals
        vertexNormals = []
        for vertIndex in range(len(model.vertices)):
            if vertIndex in faceNormals:
                normal = Vector(0, 0, 0)
                for adjNormal in faceNormals[vertIndex]:
                    normal = normal + adjNormal
                vertexNormals.append(normal / len(faceNormals[vertIndex]))
            else:
                vertexNormals.append(Vector(0, 1, 0))  # Default normal
        
        # Get model color
        model_color = getattr(model, 'diffuse_color', (255, 255, 255))
        
        # Render all faces
        for face in model.faces:
            p0 = transformed_vertices[face[0]]
            p1 = transformed_vertices[face[1]]
            p2 = transformed_vertices[face[2]]
            n0, n1, n2 = [vertexNormals[i] for i in face]
            
            # Skip back-facing triangles
            avg_normal = (n0 + n1 + n2) / 3
            view_dir = Vector(
                self.camera_target.x - self.camera_pos.x,
                self.camera_target.y - self.camera_pos.y,
                self.camera_target.z - self.camera_pos.z
            ).normalize()
            if avg_normal * view_dir <= 0:
                continue
            
            # Create points with lighting
            triangle_points = []
            for p, n in zip([p0, p1, p2], [n0, n1, n2]):
                screenX, screenY = self.perspective_projection(p.x, p.y, p.z)
                
                # Skip if offscreen
                if screenX < 0 or screenY < 0 or screenX >= self.width or screenY >= self.height:
                    continue
                
                # Calculate lighting intensity
                intensity = max(0.2, n * self.light_dir)
                
                # Apply lighting to model color
                r, g, b = model_color
                color = Color(
                    int(r * intensity),
                    int(g * intensity),
                    int(b * intensity),
                    255
                )
                
                # Create point
                point = Point(screenX, screenY, p.z, color)
                triangle_points.append(point)
            
            # Draw the triangle if all points are valid
            if len(triangle_points) == 3:
                Triangle(
                    triangle_points[0],
                    triangle_points[1],
                    triangle_points[2]
                ).draw_faster(self.image, self.zBuffer)
    
    def render_floor_grid(self):
        """Render a grid on the floor"""
        grid_size = 20
        grid_step = 4
        grid_color = Color(80, 80, 100, 255)
        
        # Draw grid on the floor
        for x in range(-grid_size, grid_size + 1, grid_step):
            for z in range(-grid_size, grid_size + 1, grid_step):
                # X-axis lines
                if x % grid_step == 0:
                    p1 = Vector(x, 0, -grid_size)
                    p2 = Vector(x, 0, grid_size)
                    screen_p1 = self.perspective_projection(p1.x, p1.y, p1.z)
                    screen_p2 = self.perspective_projection(p2.x, p2.y, p2.z)
                    
                    # Draw line if on screen
                    if (screen_p1[0] > 0 and screen_p1[1] > 0 and
                        screen_p2[0] > 0 and screen_p2[1] > 0):
                        pygame.draw.line(
                            self.screen,
                            (grid_color.r(), grid_color.g(), grid_color.b()),
                            screen_p1, screen_p2, 1
                        )
                
                # Z-axis lines
                if z % grid_step == 0:
                    p1 = Vector(-grid_size, 0, z)
                    p2 = Vector(grid_size, 0, z)
                    screen_p1 = self.perspective_projection(p1.x, p1.y, p1.z)
                    screen_p2 = self.perspective_projection(p2.x, p2.y, p2.z)
                    
                    # Draw line if on screen
                    if (screen_p1[0] > 0 and screen_p1[1] > 0 and
                        screen_p2[0] > 0 and screen_p2[1] > 0):
                        pygame.draw.line(
                            self.screen,
                            (grid_color.r(), grid_color.g(), grid_color.b()),
                            screen_p1, screen_p2, 1
                        )
    
    def render_scene(self):
        """Render the current scene state"""
        # Update camera to fixed position on object's axis
        self.update_camera_fixed_axis()
        
        # Clear image and z-buffer
        self.image = Image(self.width, self.height, Color(20, 20, 40, 255))
        self.zBuffer = [-float('inf')] * self.width * self.height
        
        # Render floor grid
        self.render_floor_grid()
        
        # Render floor headsets
        for headset in self.floor_headsets:
            self.render_model(headset)
        
        # Render the central headset
        if self.central_headset:
            self.render_model(self.central_headset["model"])
        
        # Convert image to pygame surface
        self.update_display()
        
        # Draw debug info
        if self.show_debug:
            self.draw_debug_info()
    
    def update_display(self):
        """Update the display with current image buffer"""
        for y in range(self.height):
            for x in range(self.width):
                # Calculate buffer index
                flipY = (self.height - y - 1)
                index = (flipY * self.width + x) * 4 + flipY + 1  # +1 for null byte
                
                # Extract RGB values
                if index + 2 < len(self.image.buffer):
                    r = self.image.buffer[index]
                    g = self.image.buffer[index + 1]
                    b = self.image.buffer[index + 2]
                    
                    # Set pixel on screen
                    self.screen.set_at((x, y), (r, g, b))
    
    def draw_debug_info(self):
        """Draw debug information on screen"""
        # Display rotation values
        if self.central_headset:
            rot = self.central_headset["rotation"]
            rot_text = self.font.render(
                f"Rotation: Roll={math.degrees(rot[0]):.1f}°, Pitch={math.degrees(rot[1]):.1f}°, Yaw={math.degrees(rot[2]):.1f}°",
                True, (255, 255, 255)
            )
            self.screen.blit(rot_text, (10, 10))
            
            # Display current data index if using IMU data
            if self.sensor_data:
                data_text = self.font.render(
                    f"IMU Data: {self.current_data_index}/{len(self.sensor_data)}",
                    True, (255, 255, 255)
                )
                self.screen.blit(data_text, (10, 35))
        
        # Display object count
        count_text = self.font.render(
            f"Objects: {len(self.floor_headsets) + 1}",
            True, (255, 255, 255)
        )
        self.screen.blit(count_text, (10, 60))
        
        # Display controls
        controls_text = self.font.render(
            "R: Reset | P: Pause/Play | ESC: Quit",
            True, (200, 200, 200)
        )
        self.screen.blit(controls_text, (10, self.height - 30))
    
    def handle_events(self):
        """Handle user input events"""
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return False
            
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    return False
                
                elif event.key == pygame.K_r:
                    # Reset to beginning of data
                    if self.sensor_data:
                        self.current_data_index = 0
                    # Reset rotation
                    if self.central_headset:
                        self.central_headset["rotation"] = [0, 0, 0]
                    # Reset floor headsets
                    self.floor_headsets = self.create_floor_headsets()
                    print("Scene reset")
                
                elif event.key == pygame.K_p:
                    # Pause/resume simulation
                    self.paused = not self.paused
                    print(f"Simulation {'Paused' if self.paused else 'Resumed'}")
        
        return True
    
    def run(self):
        """Main loop to run the simulation"""
        clock = pygame.time.Clock()
        running = True
        
        print("Fixed-Axis Camera Demo with Floor Headsets")
        print("------------------------------------------")
        print("Controls:")
        print("  R: Reset scene")
        print("  P: Pause/resume simulation")
        print("  ESC: Quit")
        
        while running:
            # Handle timing
            dt = min(clock.tick(60) / 1000.0, 0.1)  # Cap at 0.1s
            
            # Handle events
            running = self.handle_events()
            
            # Skip updates if paused
            if not self.paused:
                # Update central headset rotation
                self.update_central_headset(dt)
                
                # Update floor headsets physics
                self.update_floor_physics(dt)
            
            # Render the scene
            self.render_scene()
            
            # Update display
            pygame.display.flip()
            
            # Update frame counter
            self.frame_count += 1
        
        # Clean up
        pygame.quit()
        print("Simulation ended")

# Entry point function
def fixed_axis_camera_demo():
    """Run the fixed-axis camera demo with floor headsets"""
    scene = FixedAxisCameraScene()
    scene.run()

# fixed_axis_camera_demo()

def problem_6_fixed_camera():
    """
    Modified version of problem_6 that renders a scene with a rotating VR headset
    and multiple headsets sliding on the floor with friction and collisions.
    Uses a fixed camera position rather than a camera that follows the headset's axis.
    Includes video recording capability.
    """
    from modified_headset_scene import FixedCameraHeadsetScene
    
    print("Running VR Headset Physics Scene with Fixed Camera and Recording")
    
    # Instantiate and run the headset scene with a fixed camera
    try:
        # Try to use a local path first
        scene = FixedCameraHeadsetScene(csv_path="IMUData.csv")
    except Exception as e:
        # If that fails, try using a relative path with parent directory
        try:
            scene = FixedCameraHeadsetScene(csv_path="../IMUData.csv")
        except Exception as e2:
            print(f"Error loading IMU data: {e2}")
            print("Continuing with fallback rotation pattern")
            scene = FixedCameraHeadsetScene(csv_path="")
    
    scene.run()
    
problem_6_fixed_camera()