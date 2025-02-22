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

def problem_4():
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

problem_4()