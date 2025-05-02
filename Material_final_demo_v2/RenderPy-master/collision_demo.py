import pygame
import os
import math
import numpy as np
import subprocess
from image import Image, Color
from vector import Vector
from model import Model
from collision import CollisionObject
from color_support import ColoredModel
from shape import Triangle, Point

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
    
    def save_video(self, filename="simulation.mp4"):
        """
        Save recorded frames as a video file using OpenCV.
        """
        try:
            import cv2
            
            if not self.frames:
                print("No frames to save")
                return None

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

        except ImportError:
            print("OpenCV not available. Saving frames as images instead...")
            return self.save_frames_as_images()
        except Exception as e:
            print(f"Error saving video with OpenCV: {e}")
            print("Falling back to saving frames as images...")
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

    def clear_frames(self):
        """Clear recorded frames to free memory"""
        self.frames = []
        self.frame_count = 0
        print("Cleared all recorded frames")

class CollisionDemo:
    def __init__(self, width=800, height=600):
        # Initialize pygame
        pygame.init()
        self.width = width
        self.height = height
        self.screen = pygame.display.set_mode((width, height))
        pygame.display.set_caption("Headset Collision Demonstration")
        
        # Image and Z-buffer
        self.image = Image(width, height, Color(20, 20, 40, 255))
        self.zBuffer = [-float('inf')] * width * height
        
        # Camera and lighting
        self.camera_pos = Vector(0, 15, -40)
        self.camera_target = Vector(0, 1, -15)
        self.light_dir = Vector(0.5, -1, -0.5).normalize()
        
        # Setup headsets
        self.headsets = []
        self.setup_demo()
        
        # Physics settings - higher value = less friction
        self.friction_coefficient = 0.98  # Reduced friction (was 0.95)
        
        # Display
        self.font = pygame.font.SysFont('Arial', 18)
        self.frame_count = 0
        self.fps_history = []
        self.paused = False
        self.show_spheres = True
        
        # Floor surface
        self.floor_y = 0
        
        # Video recorder
        self.video_recorder = VideoRecorder(width, height, fps=30)
        self.is_recording = False
        
    def setup_demo(self):
        """Create two headsets moving toward each other"""
        print("Setting up collision demonstration...")
        
        # Define separation distance
        headset_distance = 14
        
        # Define colors
        red = (255, 0, 0)    # Red
        blue = (0, 0, 255)   # Blue
        green = (0, 255, 0)  # Green
        yellow = (255, 255, 0)  # Yellow
        
        # Create first headset (left side, moving right)
        pos1 = Vector(-headset_distance, 1, -15)
        vel1 = Vector(5, 0, 0)  # Moving right (faster)
        
        model1 = Model('./data/headset.obj')
        model1.normalizeGeometry()
        model1.setPosition(pos1.x, pos1.y, pos1.z)
        
        colored_model1 = ColoredModel(model1, diffuse_color=red)
        headset1 = CollisionObject(colored_model1, pos1, vel1, radius=2.0, elasticity=1)
        self.headsets.append(headset1)
        
        # Create second headset (right side, moving left)
        pos2 = Vector(headset_distance, 1, -15)
        vel2 = Vector(-5, 0, 0)  # Moving left (faster)
        
        model2 = Model('./data/headset.obj')
        model2.normalizeGeometry()
        model2.setPosition(pos2.x, pos2.y, pos2.z)
        
        colored_model2 = ColoredModel(model2, diffuse_color=blue)
        headset2 = CollisionObject(colored_model2, pos2, vel2, radius=2.0, elasticity=1)
        self.headsets.append(headset2)
        
        # Add a third headset (bottom, moving up)
        pos3 = Vector(0, 1, -20)
        vel3 = Vector(0, 0, 4)  # Moving up
        
        model3 = Model('./data/headset.obj')
        model3.normalizeGeometry()
        model3.setPosition(pos3.x, pos3.y, pos3.z)
        
        colored_model3 = ColoredModel(model3, diffuse_color=green)
        headset3 = CollisionObject(colored_model3, pos3, vel3, radius=2.0, elasticity=1)
        self.headsets.append(headset3)
        
        # Add a fourth headset (top, moving down with an angle)
        pos4 = Vector(4, 1, -10)
        vel4 = Vector(-2, 0, -4)  # Moving down and left
        
        model4 = Model('./data/headset.obj')
        model4.normalizeGeometry()
        model4.setPosition(pos4.x, pos4.y, pos4.z)
        
        colored_model4 = ColoredModel(model4, diffuse_color=yellow)
        headset4 = CollisionObject(colored_model4, pos4, vel4, radius=2.0, elasticity=1)
        self.headsets.append(headset4)
        
        print("Collision demonstration created. Headsets will move and collide.")

    def perspective_projection(self, x, y, z, width=None, height=None):
        """Project 3D coordinates to 2D screen space"""
        if width is None:
            width = self.width
        if height is None:
            height = self.height
        
        # Calculate vector from camera to point
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
        
        world_up = Vector(0, 1, 0)
        right = forward.cross(world_up).normalize()
        up = right.cross(forward).normalize()
        
        # Project point onto camera vectors
        right_comp = to_point * right
        up_comp = to_point * up
        forward_comp = to_point * forward
        
        if forward_comp < 0.1:
            return -1, -1
        
        # Apply perspective projection
        fov = math.pi / 3.0
        aspect = width / height
        
        x_ndc = right_comp / (forward_comp * math.tan(fov/2) * aspect)
        y_ndc = up_comp / (forward_comp * math.tan(fov/2))
        
        # Convert to screen coordinates
        screen_x = int((x_ndc + 1.0) * width / 2.0)
        screen_y = int((-y_ndc + 1.0) * height / 2.0)
        
        return screen_x, screen_y

    def render_collision_spheres(self):
        """Render wireframe spheres showing collision boundaries"""
        if not self.show_spheres:
            return
            
        for headset in self.headsets:
            # Get the center of the sphere in world space
            center = headset.position
            radius = headset.radius
            
            # Number of segments for the wireframe sphere
            segments = 16
            
            # Draw circles in three planes (xy, xz, yz)
            for plane in range(3):
                points = []
                
                # Generate points for a circle
                for i in range(segments + 1):
                    angle = (i / segments) * 2 * math.pi
                    
                    if plane == 0:  # XY plane
                        x = center.x + radius * math.cos(angle)
                        y = center.y + radius * math.sin(angle)
                        z = center.z
                    elif plane == 1:  # XZ plane
                        x = center.x + radius * math.cos(angle)
                        y = center.y
                        z = center.z + radius * math.sin(angle)
                    else:  # YZ plane
                        x = center.x
                        y = center.y + radius * math.cos(angle)
                        z = center.z + radius * math.sin(angle)
                    
                    # Project to screen space
                    screen_x, screen_y = self.perspective_projection(x, y, z)
                    if screen_x >= 0 and screen_y >= 0:
                        points.append((screen_x, screen_y))
                
                # Draw the circle if we have enough points
                if len(points) > 2:
                    # Get color from the headset, but with transparency
                    color = headset.model.diffuse_color
                    sphere_color = (color[0], color[1], color[2])
                    
                    # Draw the wireframe circle
                    pygame.draw.lines(
                        self.screen,
                        sphere_color,
                        True,  # Closed shape
                        points,
                        1  # Line width
                    )

    def render_floor(self):
        """Render a simple floor grid"""
        # Draw floor rectangle
        size = 30
        floor_corners = [
            (-size, self.floor_y, -30),
            (size, self.floor_y, -30),
            (size, self.floor_y, 0),
            (-size, self.floor_y, 0)
        ]
        
        # Project corners to screen space
        screen_corners = []
        for corner in floor_corners:
            screen_x, screen_y = self.perspective_projection(corner[0], corner[1], corner[2])
            if screen_x >= 0 and screen_y >= 0:
                screen_corners.append((screen_x, screen_y))
        
        # Draw floor rectangle
        if len(screen_corners) == 4:
            pygame.draw.polygon(
                self.screen,
                (50, 50, 70),
                screen_corners
            )
            
            # Draw grid lines
            grid_size = 5
            for i in range(-size, size + 1, grid_size):
                # X lines
                start_x, start_y = self.perspective_projection(i, self.floor_y, -30)
                end_x, end_y = self.perspective_projection(i, self.floor_y, 0)
                if start_x >= 0 and start_y >= 0 and end_x >= 0 and end_y >= 0:
                    pygame.draw.line(
                        self.screen,
                        (100, 100, 120),
                        (start_x, start_y),
                        (end_x, end_y),
                        1
                    )
                
                # Z lines
                for j in range(-30, 1, grid_size):
                    start_x, start_y = self.perspective_projection(-size, self.floor_y, j)
                    end_x, end_y = self.perspective_projection(size, self.floor_y, j)
                    if start_x >= 0 and start_y >= 0 and end_x >= 0 and end_y >= 0:
                        pygame.draw.line(
                            self.screen,
                            (100, 100, 120),
                            (start_x, start_y),
                            (end_x, end_y),
                            1
                        )

    def render_model(self, model_obj):
        """Render a 3D model with lighting"""
        # Get the actual model (handle both direct models and ColoredModel objects)
        model = getattr(model_obj, 'model', model_obj)
        
        # Precalculate transformed vertices
        transformed_vertices = []
        for i in range(len(model.vertices)):
            vertex = model.getTransformedVertex(i)
            transformed_vertices.append(vertex)
        
        # Calculate face normals and vertex normals
        face_normals = {}
        for face in model.faces:
            v0 = transformed_vertices[face[0]]
            v1 = transformed_vertices[face[1]]
            v2 = transformed_vertices[face[2]]
            
            edge1 = Vector(v1.x - v0.x, v1.y - v0.y, v1.z - v0.z)
            edge2 = Vector(v2.x - v0.x, v2.y - v0.y, v2.z - v0.z)
            normal = edge1.cross(edge2).normalize()
            
            for i in face:
                if i not in face_normals:
                    face_normals[i] = []
                face_normals[i].append(normal)
        
        vertex_normals = []
        for vert_idx in range(len(model.vertices)):
            if vert_idx in face_normals:
                normal = Vector(0, 0, 0)
                for face_normal in face_normals[vert_idx]:
                    normal = normal + face_normal
                vertex_normals.append(normal.normalize())
            else:
                vertex_normals.append(Vector(0, 1, 0))
        
        # Get model color
        if hasattr(model_obj, 'diffuse_color'):
            model_color = model_obj.diffuse_color
        elif hasattr(model, 'diffuse_color'):
            model_color = model.diffuse_color
        else:
            model_color = (200, 200, 200)
        
        # Render faces
        for face in model.faces:
            v0 = transformed_vertices[face[0]]
            v1 = transformed_vertices[face[1]]
            v2 = transformed_vertices[face[2]]
            
            n0 = vertex_normals[face[0]]
            n1 = vertex_normals[face[1]]
            n2 = vertex_normals[face[2]]
            
            # Backface culling
            avg_normal = (n0 + n1 + n2).normalize()
            view_dir = Vector(
                self.camera_pos.x - (v0.x + v1.x + v2.x) / 3,
                self.camera_pos.y - (v0.y + v1.y + v2.y) / 3,
                self.camera_pos.z - (v0.z + v1.z + v2.z) / 3
            ).normalize()
            
            if avg_normal * view_dir <= 0:
                continue
            
            # Create screen points
            screen_points = []
            colors = []
            for v, n in zip([v0, v1, v2], [n0, n1, n2]):
                screen_x, screen_y = self.perspective_projection(v.x, v.y, v.z)
                
                if screen_x < 0 or screen_y < 0 or screen_x >= self.width or screen_y >= self.height:
                    continue
                
                # Calculate lighting
                intensity = max(0.2, n * self.light_dir)
                
                r, g, b = model_color
                color = (
                    int(r * intensity),
                    int(g * intensity),
                    int(b * intensity)
                )
                
                screen_points.append((screen_x, screen_y))
                colors.append(color)
            
            # Render triangle if all points are valid
            if len(screen_points) == 3:
                # Draw filled triangle with pygame
                pygame.draw.polygon(
                    self.screen,
                    colors[0],  # Use color of first vertex
                    screen_points
                )

    def update_physics(self, dt):
        """Update physics for headsets"""
        # Clear collision records
        for headset in self.headsets:
            headset.clear_collision_history()
        
        # Check collisions between headsets
        for i in range(len(self.headsets)):
            for j in range(i + 1, len(self.headsets)):
                if self.headsets[i].check_collision(self.headsets[j]):
                    self.headsets[i].resolve_collision(self.headsets[j])
                    
                    # Add a small random boost on collision to keep things moving
                    random_boost = 0.2
                    self.headsets[i].velocity.x += (np.random.random() * 2 - 1) * random_boost
                    self.headsets[i].velocity.z += (np.random.random() * 2 - 1) * random_boost
                    self.headsets[j].velocity.x += (np.random.random() * 2 - 1) * random_boost
                    self.headsets[j].velocity.z += (np.random.random() * 2 - 1) * random_boost
                    
                    print(f"Collision detected between headset {i} and {j}")
        
        # Apply updates and constraints
        for headset in self.headsets:
            # Update position based on velocity
            headset.update(dt)
            
            # Apply friction when on floor
            if headset.position.y - headset.radius <= self.floor_y:
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
                    
                    # Lower stopping threshold (was 0.025)
                    if horizontal_speed_squared * self.friction_coefficient**2 < 0.01:
                        # Instead of stopping completely, apply tiny random movement
                        # to keep interesting motion longer
                        headset.velocity.x = (np.random.random() * 2 - 1) * 0.05
                        headset.velocity.z = (np.random.random() * 2 - 1) * 0.05
            
            # Apply model position update
            headset.model.model.setPosition(headset.position.x, headset.position.y, headset.position.z)

    def render_scene(self):
        """Render all scene elements"""
        # Clear screen
        self.screen.fill((20, 20, 40))
        
        # Render floor
        self.render_floor()
        
        # Render headsets
        for headset in self.headsets:
            self.render_model(headset.model)
        
        # Render collision spheres
        self.render_collision_spheres()
        
        # Draw debug info
        self.draw_debug_info()
        
        # Update display
        pygame.display.flip()
        
        # Capture frame if recording
        if self.is_recording:
            self.video_recorder.capture_frame(self.screen)

    def draw_debug_info(self):
        """Draw debug information on screen"""
        # Calculate FPS
        if len(self.fps_history) > 0:
            fps = len(self.fps_history) / sum(self.fps_history)
        else:
            fps = 0
        
        # Display FPS
        fps_text = self.font.render(f"FPS: {int(fps)}", True, (255, 255, 255))
        self.screen.blit(fps_text, (10, 10))
        
        # Display headset velocities
        for i, headset in enumerate(self.headsets):
            vel_magnitude = math.sqrt(headset.velocity.x**2 + headset.velocity.y**2 + headset.velocity.z**2)
            vel_text = self.font.render(
                f"Headset {i+1} velocity: {vel_magnitude:.2f}",
                True, (255, 255, 255)
            )
            self.screen.blit(vel_text, (10, 35 + i * 25))
        
        # Display recording status
        if self.is_recording:
            rec_text = self.font.render(
                f"RECORDING [{self.video_recorder.frame_count} frames]",
                True, (255, 0, 0)
            )
            self.screen.blit(rec_text, (self.width - 300, 10))
        
        # Display controls
        controls_text = self.font.render(
            "SPACE: Toggle spheres | R: Reset | P: Pause | V: Record | ESC: Quit",
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
                    # Reset demo
                    self.headsets = []
                    self.setup_demo()
                
                elif event.key == pygame.K_p:
                    # Pause/resume
                    self.paused = not self.paused
                
                elif event.key == pygame.K_SPACE:
                    # Toggle collision spheres
                    self.show_spheres = not self.show_spheres
                
                elif event.key == pygame.K_v:
                    # Toggle recording
                    if not self.is_recording:
                        self.start_recording()
                    else:
                        self.stop_recording()
        
        return True
        
    def start_recording(self):
        """Start video recording"""
        self.is_recording = True
        self.video_recorder.start_recording()
        print("Started recording")

    def stop_recording(self):
        """Stop and save video recording"""
        if not self.is_recording:
            return
        
        self.is_recording = False
        self.video_recorder.stop_recording()
        
        # Save video
        video_path = self.video_recorder.save_video("collision_demo.mp4")
        print(f"Video saved to: {video_path}")
        
        # Try to generate higher quality video with ffmpeg
        ffmpeg_path = self.video_recorder.generate_ffmpeg_video(quality="high")
        if ffmpeg_path:
            print(f"High quality video saved to: {ffmpeg_path}")

    def run(self):
        """Main demo loop"""
        clock = pygame.time.Clock()
        running = True
        
        print("Headset Collision Demonstration")
        print("Controls: SPACE (toggle spheres), R (reset), P (pause), V (record), ESC (quit)")
        
        # Start recording automatically
        self.start_recording()
        
        # Set a longer run time before auto-reset (15 seconds at 60fps)
        auto_reset_frames = 900
        min_record_frames = 300  # Minimum frames before considering reset
        
        while running:
            dt = min(clock.tick(60) / 1000.0, 0.1)
            
            # Track frame time for FPS
            self.fps_history.append(dt)
            if len(self.fps_history) > 20:
                self.fps_history.pop(0)
            
            # Handle events
            running = self.handle_events()
            
            # Skip updates if paused
            if not self.paused:
                # Update physics
                self.update_physics(dt)
                
                # Add occasional random impulses to keep things interesting
                if self.frame_count % 120 == 0 and self.frame_count < auto_reset_frames * 0.7:
                    slow_headsets = 0
                    for headset in self.headsets:
                        vel_magnitude = math.sqrt(headset.velocity.x**2 + headset.velocity.z**2)
                        if vel_magnitude < 1.0:
                            # Add small random impulse to slow headsets
                            headset.velocity.x += (np.random.random() * 2 - 1) * 0.8
                            headset.velocity.z += (np.random.random() * 2 - 1) * 0.8
                            slow_headsets += 1
                    
                    if slow_headsets > 0:
                        print(f"Added impulses to {slow_headsets} slow headsets to keep movement interesting")
            
            # Render the scene
            self.render_scene()
            
            # Increment frame counter
            self.frame_count += 1
            
            # Auto-reset if enough time has passed and headsets have mostly stopped
            if self.frame_count % 60 == 0 and self.frame_count >= min_record_frames:  # Check every second
                # Count how many headsets are still moving
                moving_headsets = 0
                for headset in self.headsets:
                    vel_magnitude = math.sqrt(headset.velocity.x**2 + headset.velocity.z**2)
                    if vel_magnitude > 0.2:  # Higher threshold
                        moving_headsets += 1
                
                # Reset if most headsets have stopped or max frames reached
                if (moving_headsets <= 1 and self.frame_count >= min_record_frames) or self.frame_count >= auto_reset_frames:
                    # If recording, let's stop and save the video
                    if self.is_recording:
                        self.stop_recording()
                        
                    print(f"Demo complete after {self.frame_count} frames, resetting...")
                    self.headsets = []
                    self.setup_demo()
                    self.frame_count = 0
                    
                    # Start recording for the new iteration
                    self.start_recording()
        
        # Stop recording if still active
        if self.is_recording:
            self.stop_recording()
            
        # Clean up
        pygame.quit()
        print("Demonstration ended")

if __name__ == "__main__":
    demo = CollisionDemo()
    demo.run()