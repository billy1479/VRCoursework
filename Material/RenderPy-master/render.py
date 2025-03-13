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
model = Model('data/headset.obj')
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
    """Convert 3D coordinates to screen coordinates using perspective projection"""
    # Set up perspective parameters
    fov = math.pi / 3.0  # 60-degree field of view
    aspect = width / height
    near = 0.1     # Near clipping plane
    far = 100.0    # Far clipping plane
    
    # Create the perspective matrix
    perspective_matrix = Matrix4.perspective(fov, aspect, near, far)
    
    # Create a vector in homogeneous coordinates
    from model import Vec4
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

def render_scene_with_motion_blur():
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
                    from shape import getVertexNormal
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
        
        # Handle events
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                elif event.key == pygame.K_m:
                    # Toggle motion blur
                    motion_blur.blur_strength = 0 if motion_blur.blur_strength > 0 else 0.7
                    print(f"Motion blur: {'ON' if motion_blur.blur_strength > 0 else 'OFF'}")
                elif event.key == pygame.K_EQUALS or event.key == pygame.K_PLUS:
                    # Increase blur strength
                    motion_blur.blur_strength = min(1.0, motion_blur.blur_strength + 0.1)
                    print(f"Motion blur strength: {motion_blur.blur_strength:.1f}")
                elif event.key == pygame.K_MINUS:
                    # Decrease blur strength
                    motion_blur.blur_strength = max(0.0, motion_blur.blur_strength - 0.1)
                    print(f"Motion blur strength: {motion_blur.blur_strength:.1f}")
                elif event.key == pygame.K_r:
                    # Reset scene
                    headsets = setup_colored_scene()
                    prev_positions = [None] * len(headsets)
                    print("Scene reset")
    
    # Clean up
    pygame.quit()
    final_image.saveAsPNG("motion_blur_render.png")
    print("Render saved as motion_blur_render.png")
    
def problem_4_2():
    simulation = SingleObjectSimulation()
    simulation.run_simulation()

def problem_5_1():
    render_scene_with_motion_blur()

# problem_4_1() # multiple collisions 4.1

# problem_4_2() # multiple collisions 4.2

problem_5_1() # motion blur 5.1