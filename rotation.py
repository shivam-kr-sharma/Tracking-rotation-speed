###### Final Code. Saves all the data to a CSV file and a plot. And measures the angle in the range of -180 to 180.



import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import csv

# Try to use Tkinter file dialog to select a save directory.
try:
    from tkinter import Tk, filedialog
    root = Tk()
    root.withdraw()  # Hide the root window
    save_directory = filedialog.askdirectory(title="Select Directory to Save Data")
    root.destroy()
    if not save_directory:
        print("No directory selected. Exiting.")
        exit()
except Exception as e:
    print("Tkinter filedialog error:", e)
    save_directory = input("Enter the directory path to save data: ")
    if not os.path.isdir(save_directory):
        print("Invalid directory. Exiting.")
        exit()

# Open the external webcam
external_cam_index = 1  # Adjust if needed
cap = cv2.VideoCapture(external_cam_index)

# Get user input for cropping dimensions
crop_width = int(input("Enter the desired crop width: "))
crop_height = int(input("Enter the desired crop height: "))
x_start = 100
y_start = 100

# Check if the camera opened successfully
if not cap.isOpened():
    print("Error: Could not open external webcam.")
    exit()

# Prepare CSV file for saving angle data
angles_file_path = os.path.join(save_directory, "angles.csv")
csv_file = open(angles_file_path, mode="w", newline="")
csv_writer = csv.writer(csv_file)
csv_writer.writerow(["Frame", "Angle"])

# Initialize variables
angle_history = []
frame_count = 0

# Set up Matplotlib for real-time plotting
plt.ion()
fig, ax = plt.subplots(figsize=(12, 8))
ax.set_xlabel("Frame Number")
ax.set_ylabel("Angle (degrees)")
ax.set_title("Rotation Angle Over Time (Close this plot window to exit)")
line, = ax.plot([], [], 'o-', label="Angle (°)")
ax.legend()

def get_centroid(contour):
    M = cv2.moments(contour)
    if M["m00"] == 0:
        return None
    cx = int(M["m10"] / M["m00"])
    cy = int(M["m01"] / M["m00"])
    return (cx, cy)

print("Press 'q' in any OpenCV window OR close the live plot window to exit the loop.")

# Main loop
while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame")
        break
    
    # Crop the frame
    cropped_frame = frame[y_start:y_start + crop_height, x_start:x_start + crop_width]
    
    # Convert to grayscale and apply Gaussian blur
    gray_frame = cv2.cvtColor(cropped_frame, cv2.COLOR_BGR2GRAY)
    blurred_frame = cv2.GaussianBlur(gray_frame, (21, 21), 0)
    
    # Apply thresholding
    _, binary = cv2.threshold(blurred_frame, 175, 255, cv2.THRESH_BINARY)
    
    # Find contours
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    centroids = []
    for contour in contours:
        if cv2.contourArea(contour) > 10:
            centroid = get_centroid(contour)
            if centroid:
                centroids.append((centroid, contour))
    
    if len(centroids) == 2:
        # Sort by area to identify center and rotating contour
        centroids.sort(key=lambda x: cv2.contourArea(x[1]))
        center_contour, rotating_contour = centroids[0][0], centroids[1][0]
        
        # Draw centroids on the cropped frame
        cv2.circle(cropped_frame, center_contour, 5, (255, 0, 0), -1)
        cv2.circle(cropped_frame, rotating_contour, 5, (0, 0, 255), -1)
        
        # Draw a line between centroids
        cv2.line(cropped_frame, center_contour, rotating_contour, (0, 255, 0), 2)
        
        # Calculate the angle between the line and the horizontal axis
        dx = rotating_contour[0] - center_contour[0]
        dy = rotating_contour[1] - center_contour[1]
        angle = np.degrees(np.arctan2(dy, dx))
        
        # Store the angle for plotting and logging
        angle_history.append(angle)
        frame_count += 1
        
        # Write the angle to the CSV file
        csv_writer.writerow([frame_count, angle])
        csv_file.flush()
        
        # Display the angle on the frame
        cv2.putText(cropped_frame, f'Angle: {angle:.2f} deg', (20, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    
    # Display the frames
    cv2.imshow('External Webcam', cropped_frame)
    cv2.imshow('Binary Image', binary)
    
    # Update the live plot
    line.set_xdata(range(len(angle_history)))
    line.set_ydata(angle_history)
    ax.relim()
    ax.autoscale_view()
    plt.draw()
    plt.pause(0.01)  # Short pause to update the plot
    
    # If the live plot window was closed by the user, break the loop
    if not plt.fignum_exists(fig.number):
        print("Live plot closed by user.")
        break
    
    # Also allow exiting by pressing 'q' in any OpenCV window
    if cv2.waitKey(1) & 0xFF == ord('q'):
        print("Exiting loop as 'q' was pressed.")
        break

# Release resources and close windows
cap.release()
cv2.destroyAllWindows()

# Save the final plot image
plot_file_path = os.path.join(save_directory, "angle_plot.png")
plt.savefig(plot_file_path)
plt.close('all')
csv_file.close()

print(f"\nAngle data saved to: {angles_file_path}")
print(f"Plot saved to: {plot_file_path}")
print("Exiting script.")
