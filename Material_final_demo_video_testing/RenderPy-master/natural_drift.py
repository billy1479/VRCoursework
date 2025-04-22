#!/usr/bin/env python3
"""
Script to show natural drift between gyroscope-only and tilt-corrected filters
without artificial bias.
"""

import os
import sys
import math
import pygame
from image import Image, Color
from model import DeadReckoningFilter, Model, SensorDataParser
from vector import Vector
from shape import Point, Triangle
from video_recorder import VideoRecorder

def load_sensor_data(file_path):
    """Load sensor data from CSV file"""
    parser = SensorDataParser(file_path)
    sensor_data = parser.parse()
    print(f"Loaded {len(sensor_data)} sensor data points")
    return sensor_data

def render_natural_comparison(csv_path="../IMUData.csv"):
    """Compare dead reckoning with and without tilt correction showing natural drift"""
    # Initialize pygame
    pygame.init()
    width, height = 1024, 512
    screen = pygame.display.set_mode((width, height))
    pygame.display.set_caption("Dead Reckoning Comparison (Natural Drift)")
    
    # Initialize recorder
    recorder = VideoRecorder(width, height, fps=30)
    recorder.start_recording()
    
    # Load CSV data
    try:
        csv_contents = load_sensor_data(csv_path)
    except Exception as e:
        print(f"Error loading CSV: {e}")
        try:
            csv_contents = load_sensor_data("../IMUData.csv")
        except Exception as e2:
            print(f"Error loading CSV from parent directory: {e2}")
            print("Exiting due to missing IMU data")
            return

    # Load models
    model_gyro_only = Model('./data/headset.obj')
    model_gyro_only.normalizeGeometry()
    model_gyro_only.setPosition(0, 0, -12)
    
    model_with_correction = Model('./data/headset.obj')
    model_with_correction.normalizeGeometry()
    model_with_correction.setPosition(0, 0, -12)
    
    # Create two filters with different settings
    # 1. Pure gyroscope (alpha=1.0 means 100% gyroscope, no accelerometer)
    dr_filter_gyro_only = DeadReckoningFilter(alpha=1.0)
    dr_filter_gyro_only.calibrate(csv_contents[:100])
    
    # 2. Gyroscope + accelerometer tilt correction
    dr_filter_with_correction = DeadReckoningFilter(alpha=0.9)
    dr_filter_with_correction.calibrate(csv_contents[:100])
    
    # Prepare rendering
    left_image = Image(width//2, height, Color(255, 255, 255, 255))
    right_image = Image(width//2, height, Color(255, 255, 255, 255))
    
    left_zBuffer = [-float('inf')] * (width//2 * height)
    right_zBuffer = [-float('inf')] * (width//2 * height)
    
    # Tracking variables
    clock = pygame.time.Clock()
    running = True
    current_data_index = 0
    
    # Setup fonts
    font = pygame.font.SysFont('Arial', 24)
    left_label = font.render("WITHOUT Tilt Correction (Gyro Only)", True, (0, 0, 0))
    right_label = font.render("WITH Tilt Correction (Gyro + Accel)", True, (0, 0, 0))
    
    # Process multiple IMU samples per frame to speed up simulation
    samples_per_frame = 3
    
    # Track maximum observed difference
    max_observed_diff = 0
    start_time = pygame.time.get_ticks()
    
    # Main rendering loop
    print("Rendering comparison showing natural drift...")
    max_frames = 3600  # 2 minutes at 30fps
    
    while running and current_data_index < len(csv_contents) and recorder.frame_count < max_frames:
        delta_time = clock.tick(60) / 1000.0
        
        # Process multiple IMU samples per visual frame
        for _ in range(samples_per_frame):
            if current_data_index >= len(csv_contents):
                break
                
            # Get current sensor data
            current_sensor_data = csv_contents[current_data_index]
            current_data_index += 1
            
            # Update both filters
            gyro_position, gyro_orientation = dr_filter_gyro_only.update(current_sensor_data)
            corrected_position, corrected_orientation = dr_filter_with_correction.update(current_sensor_data)
        
        # Get Euler angles after multiple updates
        gyro_roll, gyro_pitch, gyro_yaw = dr_filter_gyro_only.get_euler_angles()
        corrected_roll, corrected_pitch, corrected_yaw = dr_filter_with_correction.get_euler_angles()
        
        # Update model rotations
        model_gyro_only.setRotation(gyro_roll, gyro_pitch, gyro_yaw)
        model_with_correction.setRotation(corrected_roll, corrected_pitch, corrected_yaw)
        
        # Reset images for new frame
        left_image = Image(width//2, height, Color(255, 255, 255, 255))
        right_image = Image(width//2, height, Color(255, 255, 255, 255))
        left_zBuffer = [-float('inf')] * (width//2 * height)
        right_zBuffer = [-float('inf')] * (width//2 * height)
        
        # Render models
        render_model(model_gyro_only, left_image, left_zBuffer)
        render_model(model_with_correction, right_image, right_zBuffer)
        
        # Clear screen and update display
        screen.fill((255, 255, 255))
        update_display(left_image, right_image, screen)
        
        # Add visual separation between sides
        pygame.draw.line(screen, (0, 0, 0), (width//2, 0), (width//2, height), 2)
        
        # Add labels
        screen.blit(left_label, (10, 10))
        screen.blit(right_label, (width//2 + 10, 10))
        
        # Calculate differences
        gyro_roll_deg = math.degrees(gyro_roll)
        gyro_pitch_deg = math.degrees(gyro_pitch)
        gyro_yaw_deg = math.degrees(gyro_yaw)
        
        corr_roll_deg = math.degrees(corrected_roll)
        corr_pitch_deg = math.degrees(corrected_pitch)
        corr_yaw_deg = math.degrees(corrected_yaw)
        
        roll_diff = abs(gyro_roll_deg - corr_roll_deg)
        pitch_diff = abs(gyro_pitch_deg - corr_pitch_deg)
        yaw_diff = abs(gyro_yaw_deg - corr_yaw_deg)
        
        current_max_diff = max(roll_diff, pitch_diff, yaw_diff)
        max_observed_diff = max(max_observed_diff, current_max_diff)
        
        # Show values with difference coloring
        diff_threshold = 2.0  # Highlight differences greater than 2 degrees
        
        roll_color = (min(255, int(roll_diff * 25)), 0, 0) if roll_diff > diff_threshold else (0, 0, 0)
        pitch_color = (min(255, int(pitch_diff * 25)), 0, 0) if pitch_diff > diff_threshold else (0, 0, 0)
        yaw_color = (min(255, int(yaw_diff * 25)), 0, 0) if yaw_diff > diff_threshold else (0, 0, 0)
        
        # Show more precise angle values (2 decimal places) to see small drift
        roll_text_left = font.render(f"Roll: {gyro_roll_deg:.2f}°", True, roll_color)
        pitch_text_left = font.render(f"Pitch: {gyro_pitch_deg:.2f}°", True, pitch_color)
        yaw_text_left = font.render(f"Yaw: {gyro_yaw_deg:.2f}°", True, yaw_color)
        
        roll_text_right = font.render(f"Roll: {corr_roll_deg:.2f}°", True, (0, 0, 0))
        pitch_text_right = font.render(f"Pitch: {corr_pitch_deg:.2f}°", True, (0, 0, 0))
        yaw_text_right = font.render(f"Yaw: {corr_yaw_deg:.2f}°", True, (0, 0, 0))
        
        # Position text
        screen.blit(roll_text_left, (10, height - 90))
        screen.blit(pitch_text_left, (10, height - 60))
        screen.blit(yaw_text_left, (10, height - 30))
        
        screen.blit(roll_text_right, (width//2 + 10, height - 90))
        screen.blit(pitch_text_right, (width//2 + 10, height - 60))
        screen.blit(yaw_text_right, (width//2 + 10, height - 30))
        
        # Time and progress info
        elapsed_time = (pygame.time.get_ticks() - start_time) / 1000
        time_text = font.render(f"Real time: {elapsed_time:.1f}s | Sim speed: {samples_per_frame}x", True, (0, 0, 0))
        screen.blit(time_text, (width//2 - 200, 40))
        
        # Show current and maximum difference
        diff_text = font.render(f"Current diff: {current_max_diff:.2f}° | Max observed: {max_observed_diff:.2f}°", True, 
                               (200, 0, 0) if current_max_diff > diff_threshold else (0, 0, 0))
        screen.blit(diff_text, (width//2 - 250, 70))
        
        # Progress bar showing drift accumulation
        bar_width = int(min(400, max_observed_diff * 20))  # Scale bar based on drift
        pygame.draw.rect(screen, (200, 200, 200), (width//2 - 200, 100, 400, 20))
        pygame.draw.rect(screen, (255, 0, 0), (width//2 - 200, 100, bar_width, 20))
        pygame.draw.rect(screen, (0, 0, 0), (width//2 - 200, 100, 400, 20), 1)
        
        # Update display
        pygame.display.flip()
        
        # Capture frame
        recorder.capture_frame(screen)
        
        # Handle events
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                elif event.key == pygame.K_s:
                    pygame.image.save(screen, "output/natural_drift_screenshot.png")
                    print("Screenshot saved")
                elif event.key == pygame.K_UP:
                    # Increase simulation speed
                    samples_per_frame = min(10, samples_per_frame + 1)
                    print(f"Simulation speed: {samples_per_frame}x")
                elif event.key == pygame.K_DOWN:
                    # Decrease simulation speed
                    samples_per_frame = max(1, samples_per_frame - 1)
                    print(f"Simulation speed: {samples_per_frame}x")
        
        # Print progress
        if recorder.frame_count % 100 == 0 and recorder.frame_count > 0:
            print(f"Captured {recorder.frame_count} frames, max diff: {max_observed_diff:.2f}°")
    
    # Save video
    recorder.stop_recording()
    try:
        video_path = recorder.save_video("natural_drift_comparison.mp4")
        if not video_path:
            video_path = recorder.generate_ffmpeg_video(quality="high")
    except Exception as e:
        print(f"Error saving video: {e}")
        frames_dir = recorder.save_frames_as_images()
        print(f"Frames saved to {frames_dir}")
    
    pygame.quit()
    print(f"Rendering complete! Maximum observed difference: {max_observed_diff:.2f}°")

def render_model(model, image, zBuffer):
    """Render a 3D model to the image"""
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
            normal = normal / len(faceNormals[vertIndex])
            vertexNormals.append(normal)
        else:
            vertexNormals.append(Vector(0, 1, 0))  # Default normal

    # Define the light direction
    lightDir = Vector(0, 0, -1)

    # Render all faces
    for face in model.faces:
        p0 = model.getTransformedVertex(face[0])
        p1 = model.getTransformedVertex(face[1])
        p2 = model.getTransformedVertex(face[2])
        n0, n1, n2 = [vertexNormals[i] for i in face]

        # Set to true if face should be culled
        cull = False

        # Transform vertices and calculate lighting
        transformedPoints = []
        for p, n in zip([p0, p1, p2], [n0, n1, n2]):
            intensity = n * lightDir

            if intensity < 0:
                cull = True
                
            screenX, screenY = getPerspectiveProjection(p.x, p.y, p.z, image.width, image.height)
            
            transformedPoints.append(Point(screenX, screenY, p.z, Color(intensity*255, intensity*255, intensity*255, 255)))

        if not cull:
            Triangle(transformedPoints[0], transformedPoints[1], transformedPoints[2]).draw_faster(image, zBuffer)

def getPerspectiveProjection(x, y, z, width, height):
    """Calculate perspective projection of a 3D point"""
    # Set up perspective parameters
    fov = math.pi / 3.0  # 60-degree field of view
    aspect = width / height
    near = 0.1     # Near clipping plane
    far = 100.0    # Far clipping plane
    
    # Simple perspective division
    f = 1.0 / math.tan(fov / 2)
    
    # Calculate projection
    screenX = int((x / (-z * aspect) * f + 1.0) * width / 2.0)
    screenY = int((y / -z * f + 1.0) * height / 2.0)
    
    return screenX, screenY

def update_display(left_image, right_image, screen):
    """Update the display with two images side by side"""
    width = left_image.width + right_image.width
    height = left_image.height
    
    # Convert the image buffers to pygame surfaces
    for y in range(height):
        for x in range(left_image.width):
            # Left image
            flipY = (height - y - 1)
            idx_left = (flipY * left_image.width + x) * 4 + flipY + 1
            
            if idx_left + 2 < len(left_image.buffer):
                r = left_image.buffer[idx_left]
                g = left_image.buffer[idx_left + 1]
                b = left_image.buffer[idx_left + 2]
                screen.set_at((x, y), (r, g, b))
            
            # Right image
            idx_right = (flipY * right_image.width + x) * 4 + flipY + 1
            
            if idx_right + 2 < len(right_image.buffer):
                r = right_image.buffer[idx_right]
                g = right_image.buffer[idx_right + 1]
                b = right_image.buffer[idx_right + 2]
                screen.set_at((x + left_image.width, y), (r, g, b))

def main():
    """Main function"""
    print("Natural Drift Comparison Generator")
    print("---------------------------------")
    
    # Search for IMU data file
    if os.path.exists("IMUData.csv"):
        csv_path = "IMUData.csv"
    elif os.path.exists("../IMUData.csv"):
        csv_path = "../IMUData.csv"
    else:
        print("Error: IMUData.csv not found in current or parent directory.")
        return
    
    # Generate the comparison video
    render_natural_comparison(csv_path)

if __name__ == "__main__":
    main()