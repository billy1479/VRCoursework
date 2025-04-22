#!/usr/bin/env python3
"""
Script with EXTREME differences between gyro-only and corrected filters
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

def render_extreme_comparison(csv_path="../IMUData.csv"):
    """Create a visualization with very extreme differences"""
    # Initialize pygame
    pygame.init()
    width, height = 1024, 512
    screen = pygame.display.set_mode((width, height))
    pygame.display.set_caption("Dead Reckoning Comparison (EXTREME)")
    
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
    
    # Create filters with EXTREME differences
    # 1. Pure gyroscope with NO correction and HEAVY bias
    dr_filter_gyro_only = DeadReckoningFilter(alpha=1.0)  # No accelerometer correction
    dr_filter_gyro_only.calibrate(csv_contents[:100])
    
    # 2. Properly fused filter
    dr_filter_with_correction = DeadReckoningFilter(alpha=0.5)  # Much higher accelerometer weight
    dr_filter_with_correction.calibrate(csv_contents[:100])
    
    # LARGE bias to guarantee visible drift
    gyro_bias_factor = 0.05  # 10x stronger than before
    
    # Prepare rendering
    left_image = Image(width//2, height, Color(255, 255, 255, 255))
    right_image = Image(width//2, height, Color(255, 255, 255, 255))
    
    left_zBuffer = [-float('inf')] * (width//2 * height)
    right_zBuffer = [-float('inf')] * (width//2 * height)
    
    clock = pygame.time.Clock()
    running = True
    current_data_index = 0
    
    # Font and labels
    font = pygame.font.SysFont('Arial', 24)
    title_font = pygame.font.SysFont('Arial', 30)
    left_label = title_font.render("GYROSCOPE ONLY", True, (200, 0, 0))
    right_label = title_font.render("WITH TILT CORRECTION", True, (0, 120, 0))
    subtitle = font.render("See how the left side drifts while the right side remains stable", True, (0, 0, 0))
    
    # Main rendering loop
    print("Rendering comparison with extreme differences...")
    while running and current_data_index < len(csv_contents) and recorder.frame_count < 900:  # Limit to 30 seconds
        delta_time = clock.tick(60) / 1000.0
        
        # Get sensor data
        current_sensor_data = csv_contents[current_data_index]
        current_data_index += 1
        
        # Create biased data for left side
        from copy import deepcopy
        biased_data = deepcopy(current_sensor_data)
        if hasattr(biased_data, 'gyroscope'):
            # Apply strong bias to make drift very obvious
            biased_data.gyroscope = (
                biased_data.gyroscope[0] + gyro_bias_factor,
                biased_data.gyroscope[1] + gyro_bias_factor/2,
                biased_data.gyroscope[2] + gyro_bias_factor/3
            )
        
        # Update filters
        gyro_position, gyro_orientation = dr_filter_gyro_only.update(biased_data)
        gyro_roll, gyro_pitch, gyro_yaw = dr_filter_gyro_only.get_euler_angles()
        
        corrected_position, corrected_orientation = dr_filter_with_correction.update(current_sensor_data)
        corrected_roll, corrected_pitch, corrected_yaw = dr_filter_with_correction.get_euler_angles()
        
        # Update models
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
        screen.blit(subtitle, (width//2 - 300, height - 100))
        
        # Show rotation values with highlighted differences
        gyro_roll_deg = math.degrees(gyro_roll)
        gyro_pitch_deg = math.degrees(gyro_pitch)
        gyro_yaw_deg = math.degrees(gyro_yaw)
        
        corr_roll_deg = math.degrees(corrected_roll)
        corr_pitch_deg = math.degrees(corrected_pitch)
        corr_yaw_deg = math.degrees(corrected_yaw)
        
        # Calculate differences for color coding
        roll_diff = abs(gyro_roll_deg - corr_roll_deg)
        pitch_diff = abs(gyro_pitch_deg - corr_pitch_deg)
        yaw_diff = abs(gyro_yaw_deg - corr_yaw_deg)
        
        # Color based on difference magnitude (red = big difference)
        roll_color = (min(255, int(roll_diff * 5)), 0, 0) if roll_diff > 5 else (0, 0, 0)
        pitch_color = (min(255, int(pitch_diff * 5)), 0, 0) if pitch_diff > 5 else (0, 0, 0)
        yaw_color = (min(255, int(yaw_diff * 5)), 0, 0) if yaw_diff > 5 else (0, 0, 0)
        
        # Show values
        roll_text_left = font.render(f"Roll: {gyro_roll_deg:.1f}°", True, roll_color)
        pitch_text_left = font.render(f"Pitch: {gyro_pitch_deg:.1f}°", True, pitch_color)
        yaw_text_left = font.render(f"Yaw: {gyro_yaw_deg:.1f}°", True, yaw_color)
        
        roll_text_right = font.render(f"Roll: {corr_roll_deg:.1f}°", True, (0, 0, 0))
        pitch_text_right = font.render(f"Pitch: {corr_pitch_deg:.1f}°", True, (0, 0, 0))
        yaw_text_right = font.render(f"Yaw: {corr_yaw_deg:.1f}°", True, (0, 0, 0))
        
        # Position text
        screen.blit(roll_text_left, (10, height - 90))
        screen.blit(pitch_text_left, (10, height - 60))
        screen.blit(yaw_text_left, (10, height - 30))
        
        screen.blit(roll_text_right, (width//2 + 10, height - 90))
        screen.blit(pitch_text_right, (width//2 + 10, height - 60))
        screen.blit(yaw_text_right, (width//2 + 10, height - 30))
        
        # Add frame counter
        frame_text = font.render(f"Frame: {recorder.frame_count}", True, (100, 100, 100))
        screen.blit(frame_text, (width - 150, 10))
        
        # Show difference indicator
        max_diff = max(roll_diff, pitch_diff, yaw_diff)
        if max_diff > 10:
            diff_text = title_font.render(f"DIFFERENCE: {max_diff:.1f}°", True, (255, 0, 0))
            screen.blit(diff_text, (width//2 - 150, 50))
        
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
                    pygame.image.save(screen, "output/extreme_comparison.png")
                    print("Screenshot saved")
        
        # Print progress
        if recorder.frame_count % 100 == 0 and recorder.frame_count > 0:
            print(f"Captured {recorder.frame_count} frames")
    
    # Save video
    recorder.stop_recording()
    try:
        video_path = recorder.save_video("extreme_comparison.mp4")
        if not video_path:
            # Try FFmpeg
            video_path = recorder.generate_ffmpeg_video(quality="high")
    except Exception as e:
        print(f"Error saving video: {e}")
        frames_dir = recorder.save_frames_as_images()
        print(f"Frames saved to {frames_dir}")
    
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
    print("EXTREME Difference Comparison Generator")
    print("--------------------------------------")
    
    # Search for IMU data file
    if os.path.exists("IMUData.csv"):
        csv_path = "IMUData.csv"
    elif os.path.exists("../IMUData.csv"):
        csv_path = "../IMUData.csv"
    else:
        print("Error: IMUData.csv not found in current or parent directory.")
        return
    
    # Generate the comparison video
    render_extreme_comparison(csv_path)

if __name__ == "__main__":
    main()