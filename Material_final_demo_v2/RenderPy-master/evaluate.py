import os
import time
import math
import numpy as np
import matplotlib.pyplot as plt
from model import SensorDataParser, SensorData
from dead_reckoning_filter import DeadReckoningFilter
from quaternion import Quaternion
from vector import Vector

class DeadReckoningTester:
    def __init__(self, csv_path="../imudata.csv"):
        self.csv_path = csv_path
        self.sensor_data = None
        self.load_imu_data()
        self.orientations_with_correction = []
        self.orientations_without_correction = []
        self.timestamps = []
        
    def load_imu_data(self):
        if not os.path.exists(self.csv_path):
            raise FileNotFoundError(f"IMU data file not found at {self.csv_path}")
            
        print(f"Loading IMU data from {self.csv_path}...")
        parser = SensorDataParser(self.csv_path)
        self.sensor_data = parser.parse()
        print(f"Loaded {len(self.sensor_data)} IMU data points")
        
    def extend_data_duration(self, multiplier=20):
        """Extend data duration by repeating with progressive drift."""
        if not self.sensor_data:
            return
            
        original_data = self.sensor_data.copy()
        extended_data = []
        
        # Track a reference orientation to ensure continuous motion
        reference_roll = 0
        reference_pitch = 0
        reference_yaw = 0
        
        for i in range(multiplier):
            time_offset = 0 if i == 0 else extended_data[-1].time - original_data[0].time + 0.01
            
            for j, data_point in enumerate(original_data):
                # Apply progressive scaling to make movements larger over time
                scaling = 1.0 + (i * 0.1)
                
                # Gyroscope data with scaling and drift
                gyro_x = data_point.gyroscope[0] * scaling
                gyro_y = data_point.gyroscope[1] * scaling 
                gyro_z = data_point.gyroscope[2] * scaling
                
                # Keep accelerometer and magnetometer data at proper magnitudes
                accel_data = data_point.accelerometer
                mag_data = data_point.magnetometer
                
                new_point = SensorData(
                    time=data_point.time + time_offset,
                    gyroscope=(gyro_x, gyro_y, gyro_z),
                    accelerometer=accel_data,
                    magnetometer=mag_data
                )
                extended_data.append(new_point)
        
        self.sensor_data = extended_data
        print(f"Extended data to {len(extended_data)} samples (multiplier: {multiplier}x)")
    
    def add_synthetic_drift(self, drift_pattern="realistic"):
        """Add synthetic drift with different patterns."""
        if not self.sensor_data:
            return
        
        total_samples = len(self.sensor_data)
        
        if drift_pattern == "realistic":
            # Realistic drift with temperature effects, random walk, and bias instability
            bias_drift_x = 0
            bias_drift_y = 0 
            bias_drift_z = 0
            
            random_walk_x = 0
            random_walk_y = 0
            random_walk_z = 0
            
            # Thermal drift parameters (slow sinusoidal changes)
            thermal_period = total_samples / 3  # Full temperature cycle
            thermal_amplitude = 0.005  # rad/s
            
            for i in range(total_samples):
                # Calculate progress as percentage
                progress = i / total_samples
                
                # Random walk component (accumulates over time)
                random_walk_x += np.random.normal(0, 0.0001)
                random_walk_y += np.random.normal(0, 0.0001)
                random_walk_z += np.random.normal(0, 0.0002)  # Stronger on yaw
                
                # Bias instability (slowly changing bias)
                if i % 1000 == 0:
                    bias_drift_x += np.random.normal(0, 0.0002)
                    bias_drift_y += np.random.normal(0, 0.0002)
                    bias_drift_z += np.random.normal(0, 0.0005)  # Stronger on yaw
                
                # Thermal drift (sinusoidal)
                thermal_x = thermal_amplitude * math.sin(2 * math.pi * i / thermal_period)
                thermal_y = thermal_amplitude * math.sin(2 * math.pi * i / thermal_period + 2.1)
                thermal_z = thermal_amplitude * math.sin(2 * math.pi * i / thermal_period + 4.2)
                
                # Combine all drift components
                total_drift_x = bias_drift_x + random_walk_x + thermal_x
                total_drift_y = bias_drift_y + random_walk_y + thermal_y
                total_drift_z = bias_drift_z + random_walk_z + thermal_z + progress * 0.005  # Progressive yaw drift
                
                # Add drift to original gyroscope data
                original = self.sensor_data[i].gyroscope
                self.sensor_data[i] = SensorData(
                    time=self.sensor_data[i].time,
                    gyroscope=(
                        original[0] + total_drift_x,
                        original[1] + total_drift_y,
                        original[2] + total_drift_z
                    ),
                    accelerometer=self.sensor_data[i].accelerometer,
                    magnetometer=self.sensor_data[i].magnetometer
                )
        
        elif drift_pattern == "catastrophic":
            # Extreme drift scenario
            for i in range(total_samples):
                progress = i / total_samples
                drift_factor = progress * progress * 0.1  # Quadratic growth
                
                original = self.sensor_data[i].gyroscope
                self.sensor_data[i] = SensorData(
                    time=self.sensor_data[i].time,
                    gyroscope=(
                        original[0] + drift_factor * math.sin(i * 0.01),
                        original[1] + drift_factor * math.cos(i * 0.015),
                        original[2] + drift_factor * 2 * math.sin(i * 0.007)  # Stronger yaw drift
                    ),
                    accelerometer=self.sensor_data[i].accelerometer,
                    magnetometer=self.sensor_data[i].magnetometer
                )
        
        print(f"Added synthetic {drift_pattern} drift pattern")
        
    def add_magnetic_disturbances(self, disturbance_intervals=5):
        """Add periods of magnetic disturbance to test magnetometer robustness."""
        if not self.sensor_data:
            return
            
        total_samples = len(self.sensor_data)
        interval_size = total_samples // disturbance_intervals
        
        for i in range(total_samples):
            # Create periodic magnetic disturbances
            interval_position = (i % interval_size) / interval_size
            
            # Add disturbance only in the middle part of each interval
            if 0.3 < interval_position < 0.7:
                strength = 0.6 * math.sin(interval_position * math.pi)  # Peak in the middle
                
                # Magnetic disturbance in local space
                magx, magy, magz = self.sensor_data[i].magnetometer
                mag_magnitude = math.sqrt(magx**2 + magy**2 + magz**2)
                
                # Create disturbance vector perpendicular to magnetic field
                disturbance_x = magy * strength * mag_magnitude
                disturbance_y = -magx * strength * mag_magnitude
                disturbance_z = 0.2 * strength * mag_magnitude
                
                # Apply disturbance
                self.sensor_data[i] = SensorData(
                    time=self.sensor_data[i].time,
                    gyroscope=self.sensor_data[i].gyroscope,
                    accelerometer=self.sensor_data[i].accelerometer,
                    magnetometer=(
                        magx + disturbance_x,
                        magy + disturbance_y,
                        magz + disturbance_z
                    )
                )
        
        print(f"Added {disturbance_intervals} magnetic disturbance intervals")
        
    def run_filters(self, calibration_samples=100, with_correction_alpha=0.95):
        if not self.sensor_data:
            print("No IMU data available.")
            return
            
        # Create filters with different configurations
        dr_with_correction = DeadReckoningFilter(alpha=with_correction_alpha, beta=0.05, mag_weight=0.02)
        dr_without_correction = DeadReckoningFilter(alpha=1.0, beta=0.0, mag_weight=0.0)
        
        # Calibrate both filters with the same initial data
        if len(self.sensor_data) >= calibration_samples:
            print(f"Calibrating filters using first {calibration_samples} samples...")
            dr_with_correction.calibrate(self.sensor_data[:calibration_samples])
            dr_without_correction.calibrate(self.sensor_data[:calibration_samples])
        else:
            print(f"Warning: Not enough samples for calibration.")
            dr_with_correction.calibrate(self.sensor_data)
            dr_without_correction.calibrate(self.sensor_data)
        
        # Reset lists
        self.orientations_with_correction = []
        self.orientations_without_correction = []
        self.timestamps = []
        
        # Process all IMU data
        start_time = time.time()
        print(f"Processing {len(self.sensor_data)} IMU data points...")
        
        for i, data_point in enumerate(self.sensor_data):
            # Update filters
            orientation_with_correction = dr_with_correction.update(data_point)
            orientation_without_correction = dr_without_correction.update(data_point)
            
            # Store results
            self.orientations_with_correction.append(self._quaternion_to_euler(orientation_with_correction))
            self.orientations_without_correction.append(self._quaternion_to_euler(orientation_without_correction))
            self.timestamps.append(data_point.time)
            
            # Print progress
            if i % 2000 == 0:
                print(f"Processed {i}/{len(self.sensor_data)} samples ({i/len(self.sensor_data)*100:.1f}%)")
        
        elapsed = time.time() - start_time
        print(f"Processing complete. Took {elapsed:.2f} seconds.")
    
    def _quaternion_to_euler(self, q):
        # Roll (x-axis rotation)
        sinr_cosp = 2.0 * (q.w * q.x + q.y * q.z)
        cosr_cosp = 1.0 - 2.0 * (q.x * q.x + q.y * q.y)
        roll = math.atan2(sinr_cosp, cosr_cosp)
        
        # Pitch (y-axis rotation)
        sinp = 2.0 * (q.w * q.y - q.z * q.x)
        if abs(sinp) >= 1:
            pitch = math.copysign(math.pi / 2, sinp)
        else:
            pitch = math.asin(sinp)
            
        # Yaw (z-axis rotation)
        siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
        cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
        yaw = math.atan2(siny_cosp, cosy_cosp)
        
        return (math.degrees(roll), math.degrees(pitch), math.degrees(yaw))
    
    def calculate_drift_metrics(self):
        """Calculate comprehensive drift metrics."""
        if not self.orientations_with_correction or not self.orientations_without_correction:
            print("No orientation data available.")
            return
        
        # Convert to numpy arrays
        with_correction = np.array(self.orientations_with_correction)
        without_correction = np.array(self.orientations_without_correction)
        
        # Calculate drift (difference)
        drift = without_correction - with_correction
        
        # Basic metrics
        axes = ['Roll', 'Pitch', 'Yaw']
        print("\nDrift Statistics:")
        print("-" * 40)
        
        for i, axis in enumerate(axes):
            drift_mean = np.mean(drift[:, i])
            drift_std = np.std(drift[:, i])
            drift_max = np.max(np.abs(drift[:, i]))
            drift_final = drift[-1, i]
            
            print(f"{axis} Axis:")
            print(f"  Mean Drift: {drift_mean:.2f}°")
            print(f"  Std Dev: {drift_std:.2f}°")
            print(f"  Max Abs Drift: {drift_max:.2f}°")
            print(f"  Final Drift: {drift_final:.2f}°")
            print("-" * 40)
        
        # Total 3D orientation error (Euclidean norm across axes)
        total_drift = np.sqrt(np.sum(drift**2, axis=1))
        mean_total = np.mean(total_drift)
        max_total = np.max(total_drift)
        final_total = total_drift[-1]
        
        print("Total Orientation Error:")
        print(f"  Mean Total Error: {mean_total:.2f}°")
        print(f"  Max Total Error: {max_total:.2f}°")
        print(f"  Final Total Error: {final_total:.2f}°")
        print("-" * 40)
        
        # Calculate drift rate per minute
        if len(self.timestamps) > 1:
            total_time = (self.timestamps[-1] - self.timestamps[0]) / 60.0  # minutes
            if total_time > 0:
                drift_rates = np.abs(drift[-1, :]) / total_time
                print("Drift Rates (degrees per minute):")
                for i, axis in enumerate(axes):
                    print(f"  {axis}: {drift_rates[i]:.4f}°/min")
                print(f"  Total: {final_total/total_time:.4f}°/min")
        
        return drift
        
    def plot_orientations(self, downsample_factor=10):
        """Plot the orientation data with downsampling for clarity."""
        if not self.orientations_with_correction or not self.orientations_without_correction:
            print("No orientation data available.")
            return
        
        # Downsample for clearer visualization
        indices = range(0, len(self.timestamps), downsample_factor)
        
        # Convert to numpy arrays
        with_correction = np.array(self.orientations_with_correction)[indices]
        without_correction = np.array(self.orientations_without_correction)[indices]
        timestamps = np.array(self.timestamps)[indices]
        
        # Normalize timestamps to minutes
        if len(timestamps) > 0:
            timestamps = (timestamps - timestamps[0]) / 60.0  # Convert to minutes
        
        # Create subplot figure
        fig, axes = plt.subplots(3, 1, figsize=(15, 12), sharex=True)
        fig.suptitle('Dead Reckoning Filter Orientation Comparison (Extended Duration)', fontsize=16)
        
        # Labels for each subplot
        labels = ['Roll (X-axis)', 'Pitch (Y-axis)', 'Yaw (Z-axis)']
        
        # Plot each orientation component
        for i in range(3):
            ax = axes[i]
            ax.plot(timestamps, without_correction[:, i], 'b-', label='Gyroscope Only', linewidth=1)
            ax.plot(timestamps, with_correction[:, i], 'r-', label='Full Sensor Fusion', linewidth=1)
            ax.set_ylabel(f'{labels[i]} (degrees)')
            ax.grid(True)
            ax.legend(loc='upper right')
        
        # X label 
        axes[2].set_xlabel('Time (minutes)')
        
        plt.tight_layout()
        plt.savefig('orientation_comparison_extended.png', dpi=300)
        print("Saved extended orientation comparison plot to 'orientation_comparison_extended.png'")
        plt.close()
    
    def plot_drift(self, downsample_factor=10):
        """Plot drift with downsampling for clarity."""
        if not self.orientations_with_correction or not self.orientations_without_correction:
            print("No orientation data available.")
            return
        
        # Downsample for clearer visualization
        indices = range(0, len(self.timestamps), downsample_factor)
        
        # Convert to numpy arrays
        with_correction = np.array(self.orientations_with_correction)[indices]
        without_correction = np.array(self.orientations_without_correction)[indices]
        drift = without_correction - with_correction
        timestamps = np.array(self.timestamps)[indices]
        
        # Normalize timestamps to minutes
        if len(timestamps) > 0:
            timestamps = (timestamps - timestamps[0]) / 60.0  # Convert to minutes
        
        # Create plot
        fig, axes = plt.subplots(3, 1, figsize=(15, 12), sharex=True)
        fig.suptitle('Sensor Fusion Contribution Analysis (Extended Duration)', fontsize=16)
        
        # Labels for each subplot
        labels = ['Roll Drift (X-axis)', 'Pitch Drift (Y-axis)', 'Yaw Drift (Z-axis)']
        
        # Plot each drift component
        for i in range(3):
            ax = axes[i]
            ax.plot(timestamps, drift[:, i], 'g-', linewidth=1)
            ax.set_ylabel(f'{labels[i]} (degrees)')
            ax.grid(True)
            
            # Calculate and display accumulated drift
            final_drift = drift[-1, i]
            ax.text(0.02, 0.92, f'Final Drift: {final_drift:.2f}°',
                   transform=ax.transAxes, bbox=dict(facecolor='white', alpha=0.7))
        
        # X label
        axes[2].set_xlabel('Time (minutes)')
        
        plt.tight_layout()
        plt.savefig('fusion_contribution_extended.png', dpi=300)
        print("Saved extended sensor fusion contribution plot to 'fusion_contribution_extended.png'")
        plt.close()
        
        # Plot total error magnitude
        plt.figure(figsize=(15, 6))
        total_error = np.sqrt(np.sum(drift**2, axis=1))
        plt.plot(timestamps, total_error, 'r-', linewidth=1.5)
        plt.title('Total Orientation Error Over Time', fontsize=16)
        plt.xlabel('Time (minutes)')
        plt.ylabel('Total Error (degrees)')
        plt.grid(True)
        plt.tight_layout()
        plt.savefig('total_error_extended.png', dpi=300)
        print("Saved total error plot to 'total_error_extended.png'")
        plt.close()

        # Plot cumulative drift
        plt.figure(figsize=(15, 6))
        cumulative_drift = np.zeros_like(timestamps)
        
        for i in range(1, len(timestamps)):
            # Calculate absolute change in each axis
            delta_roll = abs(drift[i, 0] - drift[i-1, 0])
            delta_pitch = abs(drift[i, 1] - drift[i-1, 1])
            delta_yaw = abs(drift[i, 2] - drift[i-1, 2])
            
            # Sum to get total drift magnitude
            cumulative_drift[i] = cumulative_drift[i-1] + delta_roll + delta_pitch + delta_yaw
        
        plt.plot(timestamps, cumulative_drift, 'r-', linewidth=1.5)
        plt.title('Cumulative Drift Without Sensor Fusion', fontsize=16)
        plt.xlabel('Time (minutes)')
        plt.ylabel('Cumulative Drift (degrees)')
        plt.grid(True)
        plt.tight_layout()
        plt.savefig('cumulative_drift_extended.png', dpi=300)
        print("Saved cumulative drift plot to 'cumulative_drift_extended.png'")
        plt.close()
    
    def run_realistic_long_test(self, duration_multiplier=20):
        """Run a realistic long-duration test with various drift patterns."""
        original_data = self.sensor_data.copy() if self.sensor_data else None
        
        # Extend data duration significantly
        self.extend_data_duration(multiplier=duration_multiplier)
        
        # Add realistic drift patterns
        self.add_synthetic_drift(drift_pattern="realistic")
        
        # Add magnetic disturbances
        self.add_magnetic_disturbances(disturbance_intervals=10)
        
        # Run with appropriate configurations
        self.run_filters(calibration_samples=100, with_correction_alpha=0.95)
        
        # Calculate comprehensive drift metrics
        self.calculate_drift_metrics()
        
        # Generate visualizations
        self.plot_orientations(downsample_factor=max(1, len(self.timestamps) // 10000))
        self.plot_drift(downsample_factor=max(1, len(self.timestamps) // 10000))
        
        # Restore original data
        if original_data:
            self.sensor_data = original_data

    def run_catastrophic_test(self, duration_multiplier=10):
        """Run a test with catastrophic drift to show extreme correction."""
        original_data = self.sensor_data.copy() if self.sensor_data else None
        
        # Extend data duration
        self.extend_data_duration(multiplier=duration_multiplier)
        
        # Add extreme drift patterns
        self.add_synthetic_drift(drift_pattern="catastrophic")
        
        # Run with appropriate configurations
        self.run_filters(calibration_samples=100, with_correction_alpha=0.95)
        
        # Calculate comprehensive drift metrics
        self.calculate_drift_metrics()
        
        # Generate visualizations
        self.plot_orientations(downsample_factor=max(1, len(self.timestamps) // 5000))
        self.plot_drift(downsample_factor=max(1, len(self.timestamps) // 5000))
        
        # Restore original data
        if original_data:
            self.sensor_data = original_data

def main():
    print("Starting Enhanced Dead Reckoning Filter Test Suite")
    print("=================================================")
    
    # Create tester instance
    tester = DeadReckoningTester(csv_path="../imudata.csv")
    
    # Run realistic long-duration test (primary test)
    print("\nRunning Realistic Long-Duration Test")
    print("=================================================")
    tester.run_realistic_long_test(duration_multiplier=20)
    
    # Run catastrophic drift test
    print("\nRunning Catastrophic Drift Test")
    print("=================================================")
    tester.run_catastrophic_test(duration_multiplier=10)
    
    print("\nTest suite complete.")

if __name__ == "__main__":
    main()