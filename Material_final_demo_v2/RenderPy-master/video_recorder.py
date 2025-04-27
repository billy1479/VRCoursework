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