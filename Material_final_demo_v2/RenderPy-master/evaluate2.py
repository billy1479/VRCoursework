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
        
    def run_filters(self, calibration_samples=100, with_correction_alpha=0.95):
        if not self.sensor_data:
            print("No IMU data available.")
            return
            
        # Create filter configurations
        dr_with_correction = DeadReckoningFilter(alpha=with_correction_alpha, beta=0.05, mag_weight=0.02)
        dr_without_correction = DeadReckoningFilter(alpha=1.0, beta=0.0, mag_weight=0.0)
        
        # Calibrate filters with the same initial data
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
            if i % 1000 == 0:
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
        
    def plot_orientations(self):
        """Plot the orientation data."""
        if not self.orientations_with_correction or not self.orientations_without_correction:
            print("No orientation data available.")
            return
        
        # Convert to numpy arrays
        with_correction = np.array(self.orientations_with_correction)
        without_correction = np.array(self.orientations_without_correction)
        timestamps = np.array(self.timestamps)
        
        # Normalize timestamps to start at 0
        if len(timestamps) > 0:
            timestamps = timestamps - timestamps[0]
        
        # Create subplot figure
        fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)
        fig.suptitle('Dead Reckoning Filter Orientation Comparison', fontsize=16)
        
        # Labels for each subplot
        labels = ['Roll (X-axis)', 'Pitch (Y-axis)', 'Yaw (Z-axis)']
        
        # Plot each orientation component
        for i in range(3):
            ax = axes[i]
            ax.plot(timestamps, without_correction[:, i], 'b-', label='Gyroscope Only')
            ax.plot(timestamps, with_correction[:, i], 'r-', label='Full Sensor Fusion')
            ax.set_ylabel(f'{labels[i]} (degrees)')
            ax.grid(True)
            ax.legend()
        
        # X label only on bottom subplot
        axes[2].set_xlabel('Time (s)')
        
        plt.tight_layout()
        plt.savefig('orientation_comparison_real.png')
        print("Saved orientation comparison plot to 'orientation_comparison_real.png'")
        plt.close()
    
    def plot_drift(self):
        """Plot the calculated drift between filters."""
        if not self.orientations_with_correction or not self.orientations_without_correction:
            print("No orientation data available.")
            return
        
        # Calculate drift
        with_correction = np.array(self.orientations_with_correction)
        without_correction = np.array(self.orientations_without_correction)
        drift = without_correction - with_correction
        timestamps = np.array(self.timestamps)
        
        # Normalize timestamps to start at 0
        if len(timestamps) > 0:
            timestamps = timestamps - timestamps[0]
        
        # Create plot
        fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)
        fig.suptitle('Sensor Fusion Contribution Analysis', fontsize=16)
        
        # Labels for each subplot
        labels = ['Roll Drift (X-axis)', 'Pitch Drift (Y-axis)', 'Yaw Drift (Z-axis)']
        
        # Plot each drift component
        for i in range(3):
            ax = axes[i]
            ax.plot(timestamps, drift[:, i], 'g-')
            ax.set_ylabel(f'{labels[i]} (degrees)')
            ax.grid(True)
            
            # Calculate and display accumulated drift
            final_drift = drift[-1, i]
            ax.text(0.02, 0.92, f'Final Drift: {final_drift:.2f}°',
                   transform=ax.transAxes, bbox=dict(facecolor='white', alpha=0.7))
        
        # X label only on bottom subplot
        axes[2].set_xlabel('Time (s)')
        
        plt.tight_layout()
        plt.savefig('fusion_contribution_real.png')
        print("Saved fusion contribution plot to 'fusion_contribution_real.png'")
        plt.close()
        
        # Plot cumulative drift magnitude
        plt.figure(figsize=(12, 6))
        cumulative_drift = np.zeros_like(timestamps)
        for i in range(1, len(timestamps)):
            # Calculate absolute change in each axis
            delta_roll = abs(drift[i, 0] - drift[i-1, 0])
            delta_pitch = abs(drift[i, 1] - drift[i-1, 1])
            delta_yaw = abs(drift[i, 2] - drift[i-1, 2])
            
            # Sum to get total drift magnitude
            cumulative_drift[i] = cumulative_drift[i-1] + delta_roll + delta_pitch + delta_yaw
        
        plt.plot(timestamps, cumulative_drift, 'r-')
        plt.title('Cumulative Drift Without Sensor Fusion', fontsize=16)
        plt.xlabel('Time (s)')
        plt.ylabel('Cumulative Drift (degrees)')
        plt.grid(True)
        plt.tight_layout()
        plt.savefig('cumulative_drift_real.png')
        print("Saved cumulative drift plot to 'cumulative_drift_real.png'")
        plt.close()
        
        # Plot total instantaneous error magnitude
        plt.figure(figsize=(12, 6))
        total_error = np.sqrt(np.sum(drift**2, axis=1))
        plt.plot(timestamps, total_error, 'r-')
        plt.title('Total Orientation Error Over Time', fontsize=16)
        plt.xlabel('Time (s)')
        plt.ylabel('Total Error (degrees)')
        plt.grid(True)
        plt.tight_layout()
        plt.savefig('total_error_real.png')
        print("Saved total error plot to 'total_error_real.png'")
        plt.close()
    
    def run_individual_sensor_tests(self):
        """Run tests to compare contributions from individual sensors."""
        if not self.sensor_data:
            print("No IMU data available.")
            return
            
        # Create different filter configurations
        dr_full = DeadReckoningFilter(alpha=0.95, beta=0.05, mag_weight=0.02)  # Full fusion
        dr_gyro = DeadReckoningFilter(alpha=1.0, beta=0.0, mag_weight=0.0)     # Gyro only
        dr_gyro_accel = DeadReckoningFilter(alpha=0.95, beta=0.05, mag_weight=0.0)  # Gyro + Accel
        dr_gyro_mag = DeadReckoningFilter(alpha=0.95, beta=0.0, mag_weight=0.02)    # Gyro + Mag
        
        # Calibrate all filters with the same data
        calib_samples = min(100, len(self.sensor_data))
        dr_full.calibrate(self.sensor_data[:calib_samples])
        dr_gyro.calibrate(self.sensor_data[:calib_samples])
        dr_gyro_accel.calibrate(self.sensor_data[:calib_samples])
        dr_gyro_mag.calibrate(self.sensor_data[:calib_samples])
        
        # Storage for results
        orientations_full = []
        orientations_gyro = []
        orientations_gyro_accel = []
        orientations_gyro_mag = []
        timestamps = []
        
        # Process all IMU data
        print("Processing data with different sensor combinations...")
        
        for i, data_point in enumerate(self.sensor_data):
            # Update all filters
            orient_full = dr_full.update(data_point)
            orient_gyro = dr_gyro.update(data_point)
            orient_gyro_accel = dr_gyro_accel.update(data_point)
            orient_gyro_mag = dr_gyro_mag.update(data_point)
            
            # Store results
            orientations_full.append(self._quaternion_to_euler(orient_full))
            orientations_gyro.append(self._quaternion_to_euler(orient_gyro))
            orientations_gyro_accel.append(self._quaternion_to_euler(orient_gyro_accel))
            orientations_gyro_mag.append(self._quaternion_to_euler(orient_gyro_mag))
            timestamps.append(data_point.time)
            
            # Print progress
            if i % 1000 == 0:
                print(f"Processed {i}/{len(self.sensor_data)} samples ({i/len(self.sensor_data)*100:.1f}%)")
        
        # Convert to numpy arrays
        full = np.array(orientations_full)
        gyro = np.array(orientations_gyro)
        gyro_accel = np.array(orientations_gyro_accel)
        gyro_mag = np.array(orientations_gyro_mag)
        timestamps = np.array(timestamps)
        
        # Normalize timestamps
        timestamps = timestamps - timestamps[0]
        
        # Plot comparison
        fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)
        fig.suptitle('Sensor Contribution Analysis', fontsize=16)
        
        labels = ['Roll (X-axis)', 'Pitch (Y-axis)', 'Yaw (Z-axis)']
        
        for i in range(3):
            ax = axes[i]
            ax.plot(timestamps, gyro[:, i], 'b-', label='Gyro Only')
            ax.plot(timestamps, gyro_accel[:, i], 'g-', label='Gyro + Accel')
            ax.plot(timestamps, gyro_mag[:, i], 'm-', label='Gyro + Mag')
            ax.plot(timestamps, full[:, i], 'r-', label='Full Fusion')
            ax.set_ylabel(f'{labels[i]} (degrees)')
            ax.grid(True)
            ax.legend()
        
        axes[2].set_xlabel('Time (s)')
        
        plt.tight_layout()
        plt.savefig('sensor_contribution_analysis.png')
        print("Saved sensor contribution analysis to 'sensor_contribution_analysis.png'")
        plt.close()
        
        # Calculate and print sensor-specific metrics
        # How much does each sensor contribute to final accuracy?
        print("\nSensor Contribution Metrics:")
        print("-" * 40)
        
        # Calculate gyro vs full error as baseline
        gyro_vs_full = gyro - full
        gyro_total_error = np.sqrt(np.sum(gyro_vs_full**2, axis=1))
        gyro_final_error = gyro_total_error[-1]
        
        # Calculate error reduction from accelerometer
        accel_vs_gyro = gyro - gyro_accel
        accel_error_reduction = np.sqrt(np.sum(accel_vs_gyro**2, axis=1))
        accel_final_reduction = accel_error_reduction[-1]
        
        # Calculate error reduction from magnetometer
        mag_vs_gyro = gyro - gyro_mag
        mag_error_reduction = np.sqrt(np.sum(mag_vs_gyro**2, axis=1))
        mag_final_reduction = mag_error_reduction[-1]
        
        # Combined reduction
        combined_reduction = accel_final_reduction + mag_final_reduction
        
        # Print contribution percentage
        if gyro_final_error > 0:
            accel_contribution = (accel_final_reduction / gyro_final_error) * 100
            mag_contribution = (mag_final_reduction / gyro_final_error) * 100
            print(f"Accelerometer Contribution: {accel_contribution:.2f}%")
            print(f"Magnetometer Contribution: {mag_contribution:.2f}%")
            print(f"Combined Contribution: {(combined_reduction/gyro_final_error)*100:.2f}%")
        
        print("-" * 40)
        
        # Plot specific magnetometer contribution for yaw axis
        plt.figure(figsize=(12, 6))
        plt.plot(timestamps, gyro[:, 2], 'b-', label='Gyro Only')
        plt.plot(timestamps, gyro_mag[:, 2], 'm-', label='Gyro + Mag')
        plt.plot(timestamps, full[:, 2], 'r-', label='Full Fusion')
        plt.title('Magnetometer Contribution to Yaw Stability', fontsize=16)
        plt.xlabel('Time (s)')
        plt.ylabel('Yaw Angle (degrees)')
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.savefig('magnetometer_yaw_contribution.png')
        print("Saved magnetometer yaw contribution plot to 'magnetometer_yaw_contribution.png'")
        plt.close()

def main():
    print("Starting Dead Reckoning Filter Test Suite - Real Data Only")
    print("=================================================")
    
    # Create tester instance
    tester = DeadReckoningTester(csv_path="../imudata.csv")
    
    # Standard test with real data
    tester.run_filters(calibration_samples=100, with_correction_alpha=0.95)
    tester.plot_orientations()
    tester.plot_drift()
    tester.calculate_drift_metrics()
    
    # Run individual sensor contribution analysis
    tester.run_individual_sensor_tests()
    
    print("\nTest suite complete.")

if __name__ == "__main__":
    main()