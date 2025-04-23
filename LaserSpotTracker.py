import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import time
import datetime
import os
import sys
from collections import deque
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, 
                           QComboBox, QLabel, QPushButton, QSlider, QCheckBox, 
                           QFileDialog, QGroupBox, QGridLayout, QSplitter)
from PyQt5.QtCore import Qt, QTimer, pyqtSignal, QThread
from PyQt5.QtGui import QPixmap, QImage


def list_available_cameras(max_index=10):
    """Identify available camera devices"""
    available = []
    for i in range(max_index):
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            available.append(i)
            cap.release()
    return available


class MplCanvas(FigureCanvas):
    """Matplotlib canvas for displaying graphs"""
    def __init__(self, parent=None, width=5, height=6, dpi=100):
        self.fig = Figure(figsize=(width, height), dpi=dpi)
        self.axes = self.fig.add_subplot(111)
        super(MplCanvas, self).__init__(self.fig)
        self.fig.tight_layout()


class VideoThread(QThread):
    """Thread for handling video capture and processing"""
    update_frame = pyqtSignal(np.ndarray, np.ndarray, tuple, float)
    
    def __init__(self, camera_index):
        super().__init__()
        self.camera_index = camera_index
        self.running = False
        self.cap = None
        
        # Detection parameters
        self.brightness_threshold = 220
        self.green_red_ratio = 1.3
        self.green_blue_ratio = 1.3
        self.blur_kernel_size = 5
        self.lower_green = np.array([40, 30, 200])
        self.upper_green = np.array([90, 255, 255])
        
        # FPS calculation
        self.frame_times = deque(maxlen=30)
        self.last_frame_time = time.time()

    def run(self):
        self.running = True
        self.cap = cv2.VideoCapture(self.camera_index)
        
        if not self.cap.isOpened():
            print(f"Error: Cannot open camera {self.camera_index}")
            return
            
        while self.running:
            ret, frame = self.cap.read()
            if not ret:
                break
                
            # Flip frame horizontally for mirror effect
            frame = cv2.flip(frame, 1)
            
            # Process frame to detect laser
            frame_start_time = time.time()
            laser_position, debug_mask = self.detect_laser(frame)
            processing_time = (time.time() - frame_start_time) * 1000  # in ms
            
            # Calculate FPS
            current_time = time.time()
            frame_time = current_time - self.last_frame_time
            self.last_frame_time = current_time
            self.frame_times.append(frame_time)
            fps = 1.0 / (sum(self.frame_times) / len(self.frame_times)) if self.frame_times else 0
            
            # Emit the processed frame and detection results
            self.update_frame.emit(frame, debug_mask, laser_position if laser_position else (None, None), fps)
            
            # Small delay to reduce CPU usage
            time.sleep(0.001)
            
    def detect_laser(self, frame):
        """Detect laser spot in the given frame"""
        # Apply blur
        blur_val = self.blur_kernel_size
        if blur_val % 2 == 0:
            blur_val += 1  # Ensure kernel size is odd
        blur_val = max(1, blur_val)
        blurred = cv2.GaussianBlur(frame, (blur_val, blur_val), 0)
        
        # Convert to HSV and split channels
        hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)
        b, g, r = cv2.split(blurred)
        
        # Green mask for HSV range
        green_mask = cv2.inRange(hsv, self.lower_green, self.upper_green)
        
        # Create mask based on brightness and color ratios
        core_mask = np.zeros_like(g)
        core_mask[(g > self.brightness_threshold) &
                  (g > r * self.green_red_ratio) &
                  (g > b * self.green_blue_ratio)] = 255
        
        # Find contours
        contours, _ = cv2.findContours(core_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        laser_position = None
        
        if contours:
            largest = max(contours, key=cv2.contourArea)
            M = cv2.moments(largest)
            if M['m00'] > 0:
                cx = int(M['m10'] / M['m00'])
                cy = int(M['m01'] / M['m00'])
                laser_position = (cx, cy)
        
        # Create debug visualization
        debug_mask = cv2.cvtColor(core_mask, cv2.COLOR_GRAY2BGR)
        cv2.drawContours(debug_mask, contours, -1, (0, 255, 255), 1)
        
        return laser_position, debug_mask
        
    def stop(self):
        """Stop the video thread"""
        self.running = False
        if self.cap:
            self.cap.release()
        self.wait()
        
    def update_parameters(self, brightness, gr_ratio, gb_ratio, blur_size):
        """Update detection parameters"""
        self.brightness_threshold = brightness
        self.green_red_ratio = gr_ratio
        self.green_blue_ratio = gb_ratio
        self.blur_kernel_size = blur_size
        
class LaserSpotTrackerApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Laser Spot Tracker")
        self.resize(1200, 800)
        
        # State variables
        self.camera_index = 0
        self.video_thread = None
        self.positions = deque(maxlen=500)  # Position history
        self.timestamps = deque(maxlen=500)  # Timestamp history
        self.start_time = None
        self.recording = False
        self.save_file = None
        self.log_file = None
        
        # Setup UI
        self.setup_ui()
        
        # Populate camera dropdown
        self.refresh_cameras()
        
    def setup_ui(self):
        """Create the main user interface"""
        # Main widget and layout
        main_widget = QWidget()
        main_layout = QVBoxLayout()
        
        # Top controls section
        top_controls = QWidget()
        top_layout = QHBoxLayout()
        
        # Camera selection
        camera_group = QGroupBox("Camera")
        camera_layout = QHBoxLayout()
        self.camera_combo = QComboBox()
        self.refresh_button = QPushButton("Refresh")
        self.refresh_button.clicked.connect(self.refresh_cameras)
        camera_layout.addWidget(QLabel("Camera:"))
        camera_layout.addWidget(self.camera_combo)
        camera_layout.addWidget(self.refresh_button)
        camera_group.setLayout(camera_layout)
        
        # Start/Stop controls
        control_group = QGroupBox("Controls")
        control_layout = QHBoxLayout()
        self.start_button = QPushButton("Start")
        self.start_button.clicked.connect(self.start_tracking)
        self.stop_button = QPushButton("Stop")
        self.stop_button.clicked.connect(self.stop_tracking)
        self.stop_button.setEnabled(False)
        control_layout.addWidget(self.start_button)
        control_layout.addWidget(self.stop_button)
        control_group.setLayout(control_layout)
        
        # Recording controls
        recording_group = QGroupBox("Data Recording")
        recording_layout = QHBoxLayout()
        self.record_button = QPushButton("Start Recording")
        self.record_button.clicked.connect(self.toggle_recording)
        self.record_button.setEnabled(False)
        self.save_location_button = QPushButton("Set Save Location")
        self.save_location_button.clicked.connect(self.set_save_location)
        self.save_location_label = QLabel("Not set")
        recording_layout.addWidget(self.record_button)
        recording_layout.addWidget(self.save_location_button)
        recording_layout.addWidget(self.save_location_label)
        recording_group.setLayout(recording_layout)
        
        # Add groups to top layout
        top_layout.addWidget(camera_group)
        top_layout.addWidget(control_group)
        top_layout.addWidget(recording_group)
        top_controls.setLayout(top_layout)
        
        # Detection parameter controls
        param_group = QGroupBox("Detection Parameters")
        param_layout = QGridLayout()
        
        # Brightness threshold
        param_layout.addWidget(QLabel("Brightness:"), 0, 0)
        self.brightness_slider = QSlider(Qt.Horizontal)
        self.brightness_slider.setRange(100, 255)
        self.brightness_slider.setValue(220)
        self.brightness_value = QLabel("220")
        self.brightness_slider.valueChanged.connect(lambda v: self.brightness_value.setText(str(v)))
        self.brightness_slider.valueChanged.connect(self.update_detection_params)
        self.brightness_slider.sliderReleased.connect(lambda: self.log_parameter_change("Brightness"))
        param_layout.addWidget(self.brightness_slider, 0, 1)
        param_layout.addWidget(self.brightness_value, 0, 2)
        
        # Green/Red ratio
        param_layout.addWidget(QLabel("G/R Ratio:"), 1, 0)
        self.gr_ratio_slider = QSlider(Qt.Horizontal)
        self.gr_ratio_slider.setRange(100, 300)
        self.gr_ratio_slider.setValue(130)
        self.gr_ratio_value = QLabel("1.30")
        self.gr_ratio_slider.valueChanged.connect(lambda v: self.gr_ratio_value.setText(f"{v/100:.2f}"))
        self.gr_ratio_slider.valueChanged.connect(self.update_detection_params)
        self.gr_ratio_slider.sliderReleased.connect(lambda: self.log_parameter_change("G/R Ratio"))
        param_layout.addWidget(self.gr_ratio_slider, 1, 1)
        param_layout.addWidget(self.gr_ratio_value, 1, 2)
        
        # Green/Blue ratio
        param_layout.addWidget(QLabel("G/B Ratio:"), 2, 0)
        self.gb_ratio_slider = QSlider(Qt.Horizontal)
        self.gb_ratio_slider.setRange(100, 300)
        self.gb_ratio_slider.setValue(130)
        self.gb_ratio_value = QLabel("1.30")
        self.gb_ratio_slider.valueChanged.connect(lambda v: self.gb_ratio_value.setText(f"{v/100:.2f}"))
        self.gb_ratio_slider.valueChanged.connect(self.update_detection_params)
        self.gb_ratio_slider.sliderReleased.connect(lambda: self.log_parameter_change("G/B Ratio"))
        param_layout.addWidget(self.gb_ratio_slider, 2, 1)
        param_layout.addWidget(self.gb_ratio_value, 2, 2)
        
        # Blur kernel size
        param_layout.addWidget(QLabel("Blur Size:"), 3, 0)
        self.blur_slider = QSlider(Qt.Horizontal)
        self.blur_slider.setRange(1, 31)
        self.blur_slider.setValue(5)
        self.blur_value = QLabel("5")
        self.blur_slider.valueChanged.connect(lambda v: self.blur_value.setText(str(v)))
        self.blur_slider.valueChanged.connect(self.update_detection_params)
        self.blur_slider.sliderReleased.connect(lambda: self.log_parameter_change("Blur Size"))
        param_layout.addWidget(self.blur_slider, 3, 1)
        param_layout.addWidget(self.blur_value, 3, 2)
        
        param_group.setLayout(param_layout)
        
        # Display options
        display_group = QGroupBox("Display Options")
        display_layout = QGridLayout()
        
        self.show_video_check = QCheckBox("Show Video")
        self.show_video_check.setChecked(True)
        display_layout.addWidget(self.show_video_check, 0, 0)
        
        self.show_mask_check = QCheckBox("Show Detection Mask")
        self.show_mask_check.setChecked(True)
        display_layout.addWidget(self.show_mask_check, 0, 1)
        
        self.show_x_plot_check = QCheckBox("Show X Position Plot")
        self.show_x_plot_check.setChecked(True)
        display_layout.addWidget(self.show_x_plot_check, 1, 0)
        
        self.show_y_plot_check = QCheckBox("Show Y Position Plot")
        self.show_y_plot_check.setChecked(True)
        display_layout.addWidget(self.show_y_plot_check, 1, 1)
        
        # Connect checkboxes to update display function
        self.show_video_check.stateChanged.connect(self.update_display_options)
        self.show_mask_check.stateChanged.connect(self.update_display_options)
        self.show_x_plot_check.stateChanged.connect(self.update_display_options)
        self.show_y_plot_check.stateChanged.connect(self.update_display_options)
        
        display_group.setLayout(display_layout)
        # Main content area with splitter
        self.content_splitter = QSplitter(Qt.Vertical)
        
        # Video display area
        self.video_widget = QWidget()
        self.video_layout = QHBoxLayout()
        
        # Video feed display
        self.video_label = QLabel("No video feed")
        self.video_label.setAlignment(Qt.AlignCenter)
        self.video_label.setMinimumSize(320, 240)
        self.video_layout.addWidget(self.video_label)
        
        # Detection mask display
        self.mask_label = QLabel("No detection mask")
        self.mask_label.setAlignment(Qt.AlignCenter)
        self.mask_label.setMinimumSize(320, 240)
        self.video_layout.addWidget(self.mask_label)
        
        self.video_widget.setLayout(self.video_layout)
        self.content_splitter.addWidget(self.video_widget)
        
        # Plots display area
        self.plots_widget = QWidget()
        self.plots_layout = QHBoxLayout()
        
        # X position plot
        self.x_plot = MplCanvas(self, width=5, height=3, dpi=100)
        self.x_plot.axes.set_title('X Position vs Time')
        self.x_plot.axes.set_xlabel('Time (s)')
        self.x_plot.axes.set_ylabel('X Position')
        self.plots_layout.addWidget(self.x_plot)
        
        # Y position plot
        self.y_plot = MplCanvas(self, width=5, height=3, dpi=100)
        self.y_plot.axes.set_title('Y Position vs Time')
        self.y_plot.axes.set_xlabel('Time (s)')
        self.y_plot.axes.set_ylabel('Y Position')
        self.plots_layout.addWidget(self.y_plot)
        
        self.plots_widget.setLayout(self.plots_layout)
        self.content_splitter.addWidget(self.plots_widget)
        
        # Status bar for information
        self.status_label = QLabel("Ready")
        
        # Add all components to main layout
        main_layout.addWidget(top_controls)
        main_layout.addWidget(param_group)
        main_layout.addWidget(display_group)
        main_layout.addWidget(self.content_splitter, 1)  # Give plots/video more space
        main_layout.addWidget(self.status_label)
        
        main_widget.setLayout(main_layout)
        self.setCentralWidget(main_widget)
        
        # Timer for plot updates
        self.plot_timer = QTimer()
        self.plot_timer.timeout.connect(self.update_plots)
        self.plot_timer.setInterval(200)  # Update plots every 200ms
    
    def refresh_cameras(self):
        """Refresh the available cameras list"""
        self.camera_combo.clear()
        cameras = list_available_cameras()
        if not cameras:
            self.camera_combo.addItem("No cameras found")
            self.start_button.setEnabled(False)
        else:
            for i in cameras:
                self.camera_combo.addItem(f"Camera {i}", i)
            self.start_button.setEnabled(True)
    
    def start_tracking(self):
        """Start the laser tracking"""
        self.camera_index = self.camera_combo.currentData()
        if self.camera_index is None:
            return
            
        # Clear previous data
        self.positions.clear()
        self.timestamps.clear()
        self.start_time = time.time()
        
        # Create and start video thread
        self.video_thread = VideoThread(self.camera_index)
        self.video_thread.update_frame.connect(self.update_frame)
        self.video_thread.start()
        
        # Start plot timer
        self.plot_timer.start()
        
        # Update UI state
        self.start_button.setEnabled(False)
        self.stop_button.setEnabled(True)
        self.record_button.setEnabled(True)
        self.status_label.setText(f"Tracking started on Camera {self.camera_index}")
        
        # Initialize with current parameter values
        self.update_detection_params()
    
    def stop_tracking(self):
        """Stop the laser tracking"""
        if self.video_thread:
            self.video_thread.stop()
            self.video_thread = None
        
        # Stop plot timer
        self.plot_timer.stop()
        
        # Stop recording if active
        if self.recording:
            self.toggle_recording()
        
        # Update UI state
        self.start_button.setEnabled(True)
        self.stop_button.setEnabled(False)
        self.record_button.setEnabled(False)
        self.status_label.setText("Tracking stopped")
        
        
        
    def update_frame(self, frame, debug_mask, laser_position, fps):
            """Update the UI with new frame and detection results"""
            # Add marker to the frame if laser detected
            marked_frame = frame.copy()
            
            # Add position data if laser detected
            if laser_position[0] is not None:
                # Add to data
                current_time = time.time() - self.start_time
                self.positions.append(laser_position)
                self.timestamps.append(current_time)
                
                # Draw marker on frame
                x, y = laser_position
                cv2.circle(marked_frame, (x, y), 10, (255, 255, 255), 1)
                cv2.circle(marked_frame, (x, y), 2, (0, 0, 255), -1)
                cv2.line(marked_frame, (x - 15, y), (x + 15, y), (0, 0, 255), 1)
                cv2.line(marked_frame, (x, y - 15), (x, y + 15), (0, 0, 255), 1)
                
                # Add text with position
                text = f"X: {x}, Y: {y}"
                (text_width, text_height), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
                cv2.rectangle(marked_frame, (10, 10), (10 + text_width, 10 + text_height + 10), (0, 0, 0), -1)
                cv2.putText(marked_frame, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                
                # Save data if recording
                if self.recording and self.save_file:
                    with open(self.save_file, 'a') as f:
                        f.write(f"{current_time},{x},{y}\n")
            
            # Add FPS info
            cv2.putText(marked_frame, f"FPS: {fps:.1f}", (10, frame.shape[0] - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            # Convert frames to Qt format for display
            if self.show_video_check.isChecked():
                h, w, c = marked_frame.shape
                q_img = QImage(marked_frame.data, w, h, w*c, QImage.Format_RGB888).rgbSwapped()
                self.video_label.setPixmap(QPixmap.fromImage(q_img).scaled(
                    self.video_label.width(), self.video_label.height(), 
                    Qt.KeepAspectRatio, Qt.SmoothTransformation))
            
            if self.show_mask_check.isChecked():
                h, w, c = debug_mask.shape
                q_mask = QImage(debug_mask.data, w, h, w*c, QImage.Format_RGB888).rgbSwapped()
                self.mask_label.setPixmap(QPixmap.fromImage(q_mask).scaled(
                    self.mask_label.width(), self.mask_label.height(), 
                    Qt.KeepAspectRatio, Qt.SmoothTransformation))
    
    def update_plots(self):
        """Update the position plots"""
        if len(self.timestamps) > 1:
            # Extract position data
            times = list(self.timestamps)
            x_positions = [pos[0] for pos in self.positions]
            y_positions = [pos[1] for pos in self.positions]
            
            # Update X plot if enabled
            if self.show_x_plot_check.isChecked():
                self.x_plot.axes.clear()
                self.x_plot.axes.plot(times, x_positions, 'r-')
                self.x_plot.axes.set_title('X Position vs Time')
                self.x_plot.axes.set_xlabel('Time (s)')
                self.x_plot.axes.set_ylabel('X Position')
                self.x_plot.draw()
            
            # Update Y plot if enabled
            if self.show_y_plot_check.isChecked():
                self.y_plot.axes.clear()
                self.y_plot.axes.plot(times, y_positions, 'b-')
                self.y_plot.axes.set_title('Y Position vs Time')
                self.y_plot.axes.set_xlabel('Time (s)')
                self.y_plot.axes.set_ylabel('Y Position')
                self.y_plot.draw()
    
    def update_detection_params(self):
        """Update the detection parameters in the video thread"""
        if self.video_thread:
            brightness = self.brightness_slider.value()
            gr_ratio = self.gr_ratio_slider.value() / 100.0
            gb_ratio = self.gb_ratio_slider.value() / 100.0
            blur_size = self.blur_slider.value()
            
            self.video_thread.update_parameters(brightness, gr_ratio, gb_ratio, blur_size)
            
    def log_parameter_change(self, param_name):
        """Log parameter changes to the log file if recording is active"""
        if self.recording and self.log_file:
            timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            elapsed_time = time.time() - self.start_time
            
            # Get current parameter values
            brightness = self.brightness_slider.value()
            gr_ratio = self.gr_ratio_slider.value() / 100.0
            gb_ratio = self.gb_ratio_slider.value() / 100.0
            blur_size = self.blur_slider.value()
            
            # Write to log file
            with open(self.log_file, 'a') as f:
                f.write(f"[{timestamp}] (Elapsed: {elapsed_time:.2f}s) Parameter changed: {param_name}\n")
                f.write(f"  Current parameters: Brightness={brightness}, G/R Ratio={gr_ratio:.2f}, "
                        f"G/B Ratio={gb_ratio:.2f}, Blur Size={blur_size}\n")
    
    def update_display_options(self):
        """Update which display elements are visible"""
        # Update video displays
        self.video_label.setVisible(self.show_video_check.isChecked())
        self.mask_label.setVisible(self.show_mask_check.isChecked())
        
        # Update plot visibility
        if hasattr(self, 'x_plot'):
            self.x_plot.setVisible(self.show_x_plot_check.isChecked())
        if hasattr(self, 'y_plot'):
            self.y_plot.setVisible(self.show_y_plot_check.isChecked())
        
        # Log display option changes if recording
        if self.recording and self.log_file:
            timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            elapsed_time = time.time() - self.start_time
            with open(self.log_file, 'a') as f:
                f.write(f"[{timestamp}] (Elapsed: {elapsed_time:.2f}s) Display options changed:\n")
                f.write(f"  Video: {self.show_video_check.isChecked()}, "
                        f"Mask: {self.show_mask_check.isChecked()}, "
                        f"X Plot: {self.show_x_plot_check.isChecked()}, "
                        f"Y Plot: {self.show_y_plot_check.isChecked()}\n")
    
    def set_save_location(self):
        """Open dialog to set save file location"""
        now = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        default_filename = f"laser_tracking_data_{now}.csv"
        file_path, _ = QFileDialog.getSaveFileName(
            self, "Save Data File", default_filename, "CSV Files (*.csv)")
        
        if file_path:
            self.save_location_label.setText(os.path.basename(file_path))
            self.status_label.setText(f"Save location set to: {file_path}")
            
            # Store full path
            self.save_file_path = file_path
            
            # Create file with header
            with open(file_path, 'w') as f:
                f.write("timestamp,x_position,y_position\n")
    
    def create_log_file(self):
        """Create a log file alongside the data file"""
        if not hasattr(self, 'save_file_path') or not self.save_file_path:
            return False
            
        # Create log file with same base name as data file
        base_path = os.path.splitext(self.save_file_path)[0]
        log_path = f"{base_path}_log.txt"
        
        try:
            with open(log_path, 'a') as f:
                timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                f.write(f"Laser Spot Tracker Log - Started at {timestamp}\n")
                f.write("=" * 60 + "\n\n")
                
                # Log initial setup
                f.write("Initial Parameters:\n")
                f.write(f"  Camera Index: {self.camera_index}\n")
                f.write(f"  Brightness Threshold: {self.brightness_slider.value()}\n")
                f.write(f"  Green/Red Ratio: {self.gr_ratio_slider.value()/100:.2f}\n")
                f.write(f"  Green/Blue Ratio: {self.gb_ratio_slider.value()/100:.2f}\n")
                f.write(f"  Blur Kernel Size: {self.blur_slider.value()}\n")
                f.write(f"  Data File: {os.path.basename(self.save_file_path)}\n\n")
                f.write("Parameter Changes:\n")
            
            self.log_file = log_path
            return True
            
        except Exception as e:
            self.status_label.setText(f"Error creating log file: {e}")
            return False
    
                
    def toggle_recording(self):
        """Start or stop recording data"""
        if not self.recording:
            # Check if save location is set
            if self.save_location_label.text() == "Not set":
                self.set_save_location()
                if self.save_location_label.text() == "Not set":
                    return  # User canceled save dialog
            
            # Set data file path
            self.save_file = self.save_file_path
            
            # Create log file
            if self.create_log_file():
                self.recording = True
                self.record_button.setText("Stop Recording")
                self.status_label.setText(f"Recording to {self.save_file}")
                
                # Log recording start
                with open(self.log_file, 'a') as f:
                    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    f.write(f"[{timestamp}] Recording started\n")
            else:
                self.status_label.setText("Failed to create log file, recording not started")
        else:
            # This is where the issue is - we need to actually stop recording
            if self.log_file:
                # Log recording stop
                with open(self.log_file, 'a') as f:
                    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    elapsed_time = time.time() - self.start_time
                    f.write(f"[{timestamp}] Recording stopped after {elapsed_time:.2f} seconds\n")
                    f.write("\nRecording Summary:\n")
                    f.write(f"  Total data points: {len(self.positions)}\n")
            
            # Reset recording state
            self.recording = False
            self.save_file = None
            self.record_button.setText("Start Recording")
            self.status_label.setText("Recording stopped")
            """Create a log file alongside the data file"""
            if not hasattr(self, 'save_file_path') or not self.save_file_path:
                return False
                
            # Create log file with same base name as data file
            base_path = os.path.splitext(self.save_file_path)[0]
            log_path = f"{base_path}_log.txt"
            
            try:
                with open(log_path, 'a') as f:
                    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    f.write(f"Laser Spot Tracker Log - Started at {timestamp}\n")
                    f.write("=" * 60 + "\n\n")
                    
                    # Log initial setup
                    f.write("Initial Parameters:\n")
                    f.write(f"  Camera Index: {self.camera_index}\n")
                    f.write(f"  Brightness Threshold: {self.brightness_slider.value()}\n")
                    f.write(f"  Green/Red Ratio: {self.gr_ratio_slider.value()/100:.2f}\n")
                    f.write(f"  Green/Blue Ratio: {self.gb_ratio_slider.value()/100:.2f}\n")
                    f.write(f"  Blur Kernel Size: {self.blur_slider.value()}\n")
                    f.write(f"  Data File: {os.path.basename(self.save_file_path)}\n\n")
                    f.write("Parameter Changes:\n")
                
                self.log_file = log_path
                return True
                
            except Exception as e:
                self.status_label.setText(f"Error creating log file: {e}")
                return False
            
                    
if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = LaserSpotTrackerApp()
    window.show()
    sys.exit(app.exec_())