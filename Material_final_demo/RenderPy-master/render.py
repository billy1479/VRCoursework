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
    
    def setup_scene_original(self):
        """Set up the scene with central rotating headset and floor headsets"""
        # Create central rotating headset
        model = Model('data/headset.obj')
        model.normalizeGeometry()
        model.setPosition(0, 5, -10)  # Position in the middle, elevated
        
        # Wrap with color
        colored_model = ColoredModel(model, diffuse_color=(200, 200, 255))
        self.central_headset = {"model": colored_model, "rotation": [0, 0, 0]}
        
        # Create floor headsets with different colors
        self.floor_headsets = self.create_floor_headsets()

    def setup_scene_old(self):
        """Set up the scene with central rotating headset and floor headsets"""
        # Create central rotating headset
        model = Model('data/headset.obj')
        model.normalizeGeometry()
        model.setPosition(0, 5, -10)  # Position in the middle, elevated
        
        # Apply a gold color directly to the model
        model.diffuse_color = (255, 215, 0)  # Gold color
        
        self.central_headset = {"model": model, "rotation": [0, 0, 0]}
        
        # Create floor headsets with different colors
        self.floor_headsets = self.create_floor_headsets() 

    def setup_scene_old2(self):
        """Set up the scene with central rotating headset and floor headsets"""
        # Create central rotating headset
        model = Model('data/headset.obj')
        model.normalizeGeometry()
        model.setPosition(0, 5, -10)  # Position in the middle, elevated
        
        # Simply attach the color directly to the model
        model.diffuse_color = (255, 215, 0)  # Gold color
        
        self.central_headset = {"model": model, "rotation": [0, 0, 0]}
        
        # Create floor headsets with different colors
        self.floor_headsets = self.create_floor_headsets()

    def setup_scene_old(self):
        """Set up the scene with central rotating headset and floor headsets"""
        # Create central rotating headset
        model = Model('data/headset.obj')
        model.normalizeGeometry()
        position = Vector(0, 5, -10)  # Position in the middle, elevated
        model.setPosition(position.x, position.y, position.z)
        
        # Simply attach the color directly to the model
        model.diffuse_color = (255, 215, 0)  # Gold color
        
        # Create a collision object for the central headset
        # Note: We use zero velocity since it's not moving, just rotating
        central_obj = CollisionObject(model, position, Vector(0, 0, 0), radius=1.0)
        
        self.central_headset = {"model": central_obj, "rotation": [0, 0, 0]}
        
        # Create floor headsets with different colors
        self.floor_headsets = self.create_floor_headsets()

    def setup_scene(self):
        """Set up the scene with central rotating headset and floor headsets"""
        # Create central rotating headset
        model = Model('data/headset.obj')
        model.normalizeGeometry()
        position = Vector(0, 5, -10)  # Position in the middle, elevated
        model.setPosition(position.x, position.y, position.z)
        
        # Add the color directly to the model
        model.diffuse_color = (255, 215, 0)  # Gold color
        
        # Store the model directly but in a structure that matches our update expectations
        self.central_headset = {
            "model": model,  # This is the Model object directly
            "rotation": [0, 0, 0],
            "position": position
        }
        
        # Create floor headsets with different colors
        self.floor_headsets = self.create_floor_headsets()

    def update_central_headset(self, dt):
        """Update the rotating central headset based on sensor data"""
        if self.sensor_data and self.dr_filter:
            # Use sensor data for rotation if available
            if self.current_data_index < len(self.sensor_data):
                sensor_data = self.sensor_data[self.current_data_index]
                self.current_data_index += 1
                
                # Update filter and get orientation
                _, orientation = self.dr_filter.update(sensor_data)
                
                # Convert quaternion to Euler angles for model rotation
                roll, pitch, yaw = self.dr_filter.get_euler_angles()
                
                # Update model rotation - direct access to the Model object
                self.central_headset["model"].setRotation(roll, pitch, yaw)
                self.central_headset["rotation"] = [roll, pitch, yaw]
            else:
                # Reset to beginning of data when we reach the end
                self.current_data_index = 0
        else:
            # Fallback: simple rotation pattern
            self.central_headset["rotation"][0] += dt * 0.5  # Roll
            self.central_headset["rotation"][1] += dt * 0.7  # Pitch
            self.central_headset["rotation"][2] += dt * 0.3  # Yaw
            
            # Update model with new rotation - direct access to the Model
            self.central_headset["model"].setRotation(
                self.central_headset["rotation"][0],
                self.central_headset["rotation"][1],
                self.central_headset["rotation"][2]
            )

    def create_floor_headsets_originial(self):
        """Create multiple headsets that slide on the floor"""
        headsets = []
        
        # Define interesting colors for the headsets
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
        
        # Create headsets in various starting positions
        # 1. Circle formation with inward velocities
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
            
            # Create colored collision object
            headset = ColoredCollisionObject(
                model, pos, vel, radius=1.0, 
                diffuse_color=colors[i % len(colors)]
            )
            headsets.append(headset)
        
        # 2. "Billiards break" pattern
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
                
                # Get next color
                color = colors[color_index % len(colors)]
                color_index += 1
                
                headset = ColoredCollisionObject(
                    model, pos, vel, radius=1.0, 
                    diffuse_color=color
                )
                headsets.append(headset)
        
        # 3. "Cue ball" headset
        pos = Vector(0, 1, -20)  # Behind the triangle
        vel = Vector(0, 0, 8)    # Moving forward to hit the triangle
        
        model = Model('data/headset.obj')
        model.normalizeGeometry()
        model.setPosition(pos.x, pos.y, pos.z)
        
        # White color for the "cue ball"
        headsets.append(ColoredCollisionObject(
            model, pos, vel, radius=1.0, 
            diffuse_color=(255, 255, 255)
        ))
        
        return headsets

    def create_floor_headsets_old(self): # For coloured headsets
        """Create multiple headsets that slide on the floor"""
        headsets = []
        
        # Define vibrant colors as RGB tuples
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
        
        # Create circle formation with inward velocities
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
            
            # Directly apply color to the model
            model.diffuse_color = colors[i % len(colors)]
            
            # Create collision object
            headset = CollisionObject(model, pos, vel, radius=1.0)
            headsets.append(headset)
        
        # "Billiards break" pattern
        triangle_size = 3  # Number of rows in triangle
        start_z = -5
        color_index = 8  # Start from a different point in the color list
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
                
                # Get a color set for this headset
                color_idx = color_index % len(colors)
                color_index += 1
                diffuse = colors[color_idx][0]
                specular = colors[color_idx][1]
                ambient = colors[color_idx][2]
                shininess = colors[color_idx][3]
                
                # Create a fully defined ColoredModel
                colored_model = ColoredModel(
                    model, 
                    diffuse_color=diffuse,
                    specular_color=specular,
                    ambient_color=ambient,
                    shininess=shininess
                )
                
                # Create collision object with the colored model
                headset = ColoredCollisionObject(colored_model, pos, vel, radius=1.0)
                headsets.append(headset)
        
        # "Cue ball" headset - white with high shininess
        pos = Vector(0, 1, -20)  # Behind the triangle
        vel = Vector(0, 0, 8)    # Moving forward to hit the triangle
        
        model = Model('data/headset.obj')
        model.normalizeGeometry()
        model.setPosition(pos.x, pos.y, pos.z)
        
        # Create a special shiny white "cue ball" headset
        white_model = ColoredModel(
            model,
            diffuse_color=(255, 255, 255),
            specular_color=(255, 255, 255),
            ambient_color=(80, 80, 80),
            shininess=30  # Extra shiny
        )
        
        headsets.append(ColoredCollisionObject(
            white_model, pos, vel, radius=1.0
        ))
        
        return headsets

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

    def update_central_headset_old(self, dt):
        """Update the rotating central headset based on sensor data"""
        if self.sensor_data and self.dr_filter:
            # Use sensor data for rotation if available
            if self.current_data_index < len(self.sensor_data):
                sensor_data = self.sensor_data[self.current_data_index]
                self.current_data_index += 1
                
                # Update filter and get orientation
                _, orientation = self.dr_filter.update(sensor_data)
                
                # Convert quaternion to Euler angles for model rotation
                roll, pitch, yaw = self.dr_filter.get_euler_angles()
                
                # Update model rotation
                self.central_headset["model"].model.setRotation(roll, pitch, yaw)
                self.central_headset["rotation"] = [roll, pitch, yaw]
            else:
                # Reset to beginning of data when we reach the end
                self.current_data_index = 0
        else:
            # Fallback: simple rotation pattern
            self.central_headset["rotation"][0] += dt * 0.5  # Roll
            self.central_headset["rotation"][1] += dt * 0.7  # Pitch
            self.central_headset["rotation"][2] += dt * 0.3  # Yaw
            
            self.central_headset["model"].setRotation(
                self.central_headset["rotation"][0],
                self.central_headset["rotation"][1],
                self.central_headset["rotation"][2]
            )

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

    def perspective_projection_old(self, x, y, z):
        """Convert 3D world coordinates to 2D screen coordinates with perspective"""
        # Perspective projection parameters
        fov = math.pi / 3.0  # 60-degree field of view
        aspect = self.width / self.height
        near = 0.1
        
        # Avoid division by very small z values
        depth = z
        if depth > -0.1:  # Ensure z is negative (behind camera)
            depth = -0.1
            
        # Perspective division
        x_normalized = x / -depth
        y_normalized = y / -depth
        
        # Scale to screen coordinates
        screenX = int((x_normalized + 1.0) * self.width / 2.0)
        screenY = int((y_normalized + 1.0) * self.height / 2.0)
        
        return screenX, screenY

    def perspective_projection_old2(self, x, y, z, width=None, height=None):
        """Convert 3D world coordinates to 2D screen coordinates with perspective"""
        # Use class properties if parameters are not provided
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

    def perspective_projection_old(self, x, y, z, width=None, height=None):
        """Convert 3D world coordinates to 2D screen coordinates with perspective"""
        # Use class width/height if not provided
        if width is None:
            width = self.width
        if height is None:
            height = self.height
            
        # Rest of the perspective projection code...
        fov = math.pi / 3.0
        aspect = width / height
        near = 0.1
        
        depth = z
        if depth > -0.1:
            depth = -0.1
            
        x_normalized = x / -depth
        y_normalized = y / -depth
        
        screenX = int((x_normalized + 1.0) * width / 2.0)
        screenY = int((y_normalized + 1.0) * height / 2.0)
        
        return screenX, screenY

    def render_scene(self):
        """Render the entire scene with all headsets"""
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

    def render_model_original(self, model):
        """Render a 3D model with lighting"""
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
                # Create points with positions and normals
                triangle_points = []
                for p, n in zip([p0, p1, p2], [n0, n1, n2]):
                    screenX, screenY = self.perspective_projection(p.x, p.y, p.z)
                    
                    # Create a point with position and normal
                    point = Point(screenX, screenY, p.z, Color(255, 255, 255, 255))
                    point.normal = n  # Add normal as attribute for shading
                    
                    triangle_points.append(point)
                
                # Render the triangle with colored material if available
                from color_support import render_triangle_with_color
                if hasattr(model, 'diffuse_color'):
                    render_triangle_with_color(
                        triangle_points,
                        self.image,
                        self.zBuffer,
                        model,  # Colored model
                        self.light_dir,
                        self.camera_pos
                    )
                else:
                    # Basic lighting for non-colored models
                    for i in range(3):
                        intensity = max(0.2, triangle_points[i].normal * self.light_dir)
                        triangle_points[i].color = Color(
                            int(255 * intensity),
                            int(255 * intensity),
                            int(255 * intensity),
                            255
                        )
                    
                    Triangle(
                        triangle_points[0],
                        triangle_points[1],
                        triangle_points[2]
                    ).draw_faster(self.image, self.zBuffer)

    def render_model_old(self, model_obj):
        """Render a 3D model with lighting using the simplified coloring approach"""
        # For ColoredCollisionObject or CollisionObject, get the model inside
        if hasattr(model_obj, 'model'):
            model = model_obj.model
        else:
            model = model_obj
        
        # Pre-compute transformed vertices to avoid redundant calculations
        transformed_vertices = []
        for i in range(len(model.vertices)):
            transformed_vertices.append(model.getTransformedVertex(i))
        
        # Render all faces
        for face in model.faces:
            # Get vertices for this face
            p0 = transformed_vertices[face[0]]
            p1 = transformed_vertices[face[1]]
            p2 = transformed_vertices[face[2]]
            
            # Calculate face normal and simple lighting
            normal = (p2-p0).cross(p1-p0).normalize()
            
            # Skip back-facing triangles (simple culling)
            if normal * self.light_dir < 0:
                continue
            
            # Calculate lighting intensity
            intensity = max(0.2, normal * self.light_dir)
            
            # Create points for this triangle
            points = []
            for p in [p0, p1, p2]:
                # Project to screen
                screenX, screenY = self.perspective_projection(p.x, p.y, p.z)
                
                # Determine the color to use
                if hasattr(model, 'diffuse_color'):
                    # The model has color information - use it with lighting
                    r, g, b = model.diffuse_color
                    color = Color(int(r * intensity), int(g * intensity), int(b * intensity), 255)
                else:
                    # No color info - use default grey with lighting
                    color = Color(int(200 * intensity), int(200 * intensity), int(200 * intensity), 255)
                
                # Create point with position and color
                points.append(Point(screenX, screenY, p.z, color))
            
            # Draw the triangle using the standard Triangle class
            Triangle(points[0], points[1], points[2]).draw_faster(self.image, self.zBuffer)

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

problem_6()