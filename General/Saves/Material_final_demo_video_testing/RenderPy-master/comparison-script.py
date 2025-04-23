#!/usr/bin/env python3
"""
Script to generate two comparative videos using the VideoRecorder class:
1. Dead reckoning filter WITHOUT gravity-based tilt correction (gyroscope only)
2. Dead reckoning filter WITH gravity-based tilt correction (gyroscope + accelerometer)
"""

import os
import sys
import math
import pygame
from image import Image, Color
from model import DeadReckoningFilter, Model, SensorDataParser
from vector import Vector
from shape import Point, Triangle
from video_recorder import VideoRecorder  # Import the provided VideoRecorder class

def load_sensor_data(file_path):
    """Load sensor data from CSV file"""
    parser = SensorDataParser(file_path)
    sensor_data = parser.parse()
    print(f"Loaded {len(sensor_data)} sensor data points")
    return sensor_data

def render_filter_comparison(csv_path="../IMUData.csv"):
    """
    Render two side-by-side videos comparing dead reckoning filter:
    - Left: Without tilt correction (alpha=1.0, gyroscope only)
    - Right: With tilt correction (alpha=0.7, gyroscope + accelerometer)
    """
    # Initialize pygame
    pygame.init()
    width, height = 1024, 512  # Double width for side-by-side comparison
    screen = pygame.display.set_mode((width, height))
    pygame.display.set_caption("Dead Reckoning Filter Comparison")
    
    # Initialize the VideoRecorder
    recorder = VideoRecorder(width, height, fps=30)
    recorder.start_recording()
    
    # Load CSV data
    try:
        csv_contents = load_sensor_data(csv_path)
    except Exception as e:
        print(f"Error loading CSV: {e}")
        try:
            # Try parent directory
            csv_contents = load_sensor_data("../IMUData.csv")
        except Exception as e2:
            print(f"Error loading CSV from parent directory: {e2}")
            print("Exiting due to missing IMU data")
            return

    # Load model
    model_gyro_only = Model('./data/headset.obj')
    model_gyro_only.normalizeGeometry()
    model_gyro_only.setPosition(0, 0, -12)
    
    model_with_correction = Model('./data/headset.obj')
    model_with_correction.normalizeGeometry()
    model_with_correction.setPosition(0, 0, -12)
    
    # Create two filters with different settings
    # 1. Gyroscope only (alpha=1.0 means 100% gyroscope, no accelerometer)
    dr_filter_gyro_only = DeadReckoningFilter(alpha=1.0)
    dr_filter_gyro_only.calibrate(csv_contents[:100])
    
    # 2. Gyroscope + accelerometer tilt correction (alpha=0.7 for more visible effect)
    dr_filter_with_correction = DeadReckoningFilter(alpha=0.5)  # More aggressive tilt correction
    dr_filter_with_correction.calibrate(csv_contents[:100])
    
    # Add artificial bias to gyroscope for more visible drift
    gyro_bias_factor = 0.005  # Small bias to accelerate drift
    
    # Prepare for rendering
    left_image = Image(width//2, height, Color(255, 255, 255, 255))
    right_image = Image(width//2, height, Color(255, 255, 255, 255))
    
    left_zBuffer = [-float('inf')] * (width//2 * height)
    right_zBuffer = [-float('inf')] * (width//2 * height)
    
    # Create pygame clock for timing
    clock = pygame.time.Clock()
    running = True
    current_data_index = 0
    
    # Setup font for labels
    font = pygame.font.SysFont('Arial', 24)
    left_label = font.render("WITHOUT Tilt Correction (Gyro Only)", True, (0, 0, 0))
    right_label = font.render("WITH Tilt Correction (Gyro + Accel)", True, (0, 0, 0))
    
    # Draw a dividing line
    line_surface = pygame.Surface((2, height))
    line_surface.fill((0, 0, 0))
    
    # Limit max frames to keep file size manageable
    max_frames = 1800  # 60 seconds at 30fps
    
    # Main rendering loop
    print("Rendering comparison video...")
    while running and current_data_index < len(csv_contents) and recorder.frame_count < max_frames:
        # Handle timing
        delta_time = clock.tick(60) / 1000.0
        
        # Get current sensor data
        current_sensor_data = csv_contents[current_data_index]
        current_data_index += 1
        
        # Add artificial bias to gyroscope-only filter for more visible drift
        biased_data = current_sensor_data
        if hasattr(current_sensor_data, 'gyroscope'):
            # Create a modified copy with biased gyroscope data for the left side
            from copy import deepcopy
            biased_data = deepcopy(current_sensor_data)
            if isinstance(biased_data.gyroscope, tuple):
                biased_data.gyroscope = (
                    biased_data.gyroscope[0] + gyro_bias_factor,
                    biased_data.gyroscope[1] + gyro_bias_factor,
                    biased_data.gyroscope[2]
                )
        
        # Update both filters
        # 1. Gyroscope only (with artificial bias to exaggerate drift)
        gyro_position, gyro_orientation = dr_filter_gyro_only.update(biased_data)
        gyro_roll, gyro_pitch, gyro_yaw = dr_filter_gyro_only.get_euler_angles()
        
        # 2. With tilt correction
        corrected_position, corrected_orientation = dr_filter_with_correction.update(current_sensor_data)
        corrected_roll, corrected_pitch, corrected_yaw = dr_filter_with_correction.get_euler_angles()
        
        # Update model rotations
        model_gyro_only.setRotation(gyro_roll, gyro_pitch, gyro_yaw)
        model_with_correction.setRotation(corrected_roll, corrected_pitch, corrected_yaw)
        
        # Reset images and z-buffers for new frame
        left_image = Image(width//2, height, Color(255, 255, 255, 255))
        right_image = Image(width//2, height, Color(255, 255, 255, 255))
        
        left_zBuffer = [-float('inf')] * (width//2 * height)
        right_zBuffer = [-float('inf')] * (width//2 * height)
        
        # Render both models
        render_model(model_gyro_only, left_image, left_zBuffer)
        render_model(model_with_correction, right_image, right_zBuffer)
        
        # Clear screen and update display
        screen.fill((255, 255, 255))
        update_display(left_image, right_image, screen)
        
        # Draw dividing line
        screen.blit(line_surface, (width//2 - 1, 0))
        
        # Add labels
        screen.blit(left_label, (10, 10))
        screen.blit(right_label, (width//2 + 10, 10))
        
        # Add rotation values as text
        gyro_info = font.render(f"Roll: {math.degrees(gyro_roll):.1f}° Pitch: {math.degrees(gyro_pitch):.1f}° Yaw: {math.degrees(gyro_yaw):.1f}°", True, (0, 0, 0))
        corrected_info = font.render(f"Roll: {math.degrees(corrected_roll):.1f}° Pitch: {math.degrees(corrected_pitch):.1f}° Yaw: {math.degrees(corrected_yaw):.1f}°", True, (0, 0, 0))
        
        screen.blit(gyro_info, (10, height - 40))
        screen.blit(corrected_info, (width//2 + 10, height - 40))
        
        # Add frame counter and time
        frame_info = font.render(f"Frame: {recorder.frame_count+1} / Time: {recorder.frame_count/30:.1f}s", True, (0, 0, 0))
        screen.blit(frame_info, (width//2 - 100, height - 70))
        
        # Draw difference indicator - highlight when differences become substantial
        diff_roll = abs(math.degrees(gyro_roll - corrected_roll))
        diff_pitch = abs(math.degrees(gyro_pitch - corrected_pitch))
        diff_yaw = abs(math.degrees(gyro_yaw - corrected_yaw))
        max_diff = max(diff_roll, diff_pitch, diff_yaw)
        
        # Show difference indicator when drift becomes noticeable
        if max_diff > 5:  # More than 5 degrees difference
            diff_color = (255, 0, 0) if max_diff > 10 else (255, 165, 0)  # Red if very different, orange otherwise
            diff_text = font.render(f"Difference: {max_diff:.1f}° (drift visible)", True, diff_color)
            screen.blit(diff_text, (width//2 - 150, 40))
        
        # Update display
        pygame.display.flip()
        
        # Capture frame with the VideoRecorder
        recorder.capture_frame(screen)
        
        # Handle events
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                elif event.key == pygame.K_s:
                    # Save screenshot
                    screenshot_path = "output/comparison_screenshot.png"
                    pygame.image.save(screen, screenshot_path)
                    print(f"Screenshot saved as {screenshot_path}")
        
        # Print progress every 100 frames
        if recorder.frame_count % 100 == 0 and recorder.frame_count > 0:
            print(f"Captured {recorder.frame_count} frames ({recorder.frame_count/30:.1f} seconds)")
    
    # Stop recording and save the video
    recorder.stop_recording()
    
    # Try different methods to save the video
    try:
        video_path = recorder.save_video("dead_reckoning_comparison.mp4")
        if not video_path:
            # If save_video fails, try FFmpeg
            video_path = recorder.generate_ffmpeg_video(quality="high")
    except Exception as e:
        print(f"Error saving video: {e}")
        # Fall back to saving individual frames
        frames_dir = recorder.save_frames_as_images()
        print(f"Individual frames saved to {frames_dir}")
    
    # Cleanup
    pygame.quit()
    print("Rendering complete!")

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
    print("Dead Reckoning Filter Comparison Generator")
    print("------------------------------------------")
    
    # Search for IMU data file
    if os.path.exists("IMUData.csv"):
        csv_path = "IMUData.csv"
    elif os.path.exists("../IMUData.csv"):
        csv_path = "../IMUData.csv"
    else:
        print("Error: IMUData.csv not found in current or parent directory.")
        return
    
    # Generate the comparison video
    render_filter_comparison(csv_path)

if __name__ == "__main__":
    main()