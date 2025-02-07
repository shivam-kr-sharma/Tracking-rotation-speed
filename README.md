This code works by taking continuous video and then processing it with opencv.
First a video stream is started and then, the user is asked to select the region of interest in the video which is done by providing cropping parameters.
Then it is converted into binary format based on a threshold pixel value.
In this code, in the region of interest, there are two contours detected which differ from each other in terms of area. One contour with smaller area is at the center and the other contour with larger area is away from the center and is revolving around the center. 
The centroids of these two contours is detected and then used in tracking the angle formed by the line joining the centroids in each frame of the video.
The plotting of the video is also done live. After the task is done, the plot and the data file will be saved in the selected directory.

# Rotation Angle Tracking using OpenCV and Matplotlib

## Overview
This script captures video from an external webcam, processes the frames to detect two key points, and calculates the rotation angle in the range of **-180 to 180 degrees**. The angle data is saved in a CSV file, and a real-time plot is generated to visualize the angle variations. Once the user exits, the final plot is saved as an image.

## Features
- Uses **OpenCV** for real-time video capture and processing.
- Detects two key points (centroids) and calculates the rotation angle.
- Saves angle data to a **CSV file** in a user-selected directory.
- Displays a **real-time Matplotlib plot** of the rotation angle.
- Allows the user to exit by:
  - Pressing **'q'** in any OpenCV window.
  - Closing the live Matplotlib plot window.
- Saves the final plot as a **PNG image** in the selected directory.

## Requirements
Ensure you have the following dependencies installed:
```bash
pip install opencv-python numpy matplotlib
```

## How to Run
1. **Connect an external webcam** to your computer.
2. **Run the script**:
   ```bash
   python script.py
   ```
3. A **file dialog** will prompt you to **select a directory** to save the data.
4. Enter the **desired crop width** and **height** for the frame.
5. The script will **start processing frames** in real-time:
   - The external webcam feed will be displayed.
   - A binary (thresholded) image will be shown.
   - The **rotation angle** will be calculated and updated in the live plot.
6. **Exit options:**
   - Press **'q'** in any OpenCV window.
   - Close the **Matplotlib live plot window**.
7. The script will automatically **save the angle data and plot**, then exit.

## Output Files
Upon exiting, the following files are saved in the selected directory:
- `angles.csv` - Contains the frame number and the corresponding angle.
- `angle_plot.png` - A visualization of the rotation angle over time.

## Explanation of Key Functions
- **get_centroid(contour):** Calculates the centroid of a given contour.
- **Main loop:**
  - Captures and crops frames.
  - Converts to grayscale and applies Gaussian blur.
  - Applies thresholding and finds contours.
  - Identifies two key points and calculates the angle.
  - Updates the real-time Matplotlib plot.
  - Saves the data in real-time to the CSV file.

## Notes
- Ensure your **external webcam** is properly connected and recognized by OpenCV.
- The script measures angles between **-180 to 180 degrees**.
- If the **Tkinter file dialog fails**, the user will be prompted to enter the save directory manually.

## Example Output (CSV File)
```
Frame,Angle
1,-45.23
2,-44.78
3,-46.10
...
```

## Example Output (Plot)
A line plot showing angle variation over time.

![Example Plot](example_angle_plot.png) *(Replace with actual generated plot image)*

## License
This script is free to use and modify. Attribution is appreciated.

