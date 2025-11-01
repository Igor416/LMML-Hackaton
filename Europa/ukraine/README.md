# Sigmoid City Autonomous Vehicle Tracking

## Mission
Track an erratically moving 2D toy car across black screen videos. The car draws letters as it moves, and we need to detect those letters to solve the mystery!

## Overview
The VehicleTracker system processes video files containing a black screen with a 2D toy car that moves chaotically for ~2 hours (1:56:40) in a 400x400 resolution. The car's movement path traces out letters that need to be detected.

## Features
- **Vehicle Tracking**: Detects the 2D car on black background using simple thresholding
- **Path Reconstruction**: Draws the car's trajectory across all frames to reveal letters
- **Letter Detection**: Uses OCR (pytesseract) to recognize letters from the drawn path
- **Multi-Video Processing**: Processes all videos and combines results

## Usage
```bash
pip install opencv-python numpy pytesseract pillow

# On Windows, you may need to install Tesseract OCR separately
# Download from: https://github.com/UB-Mannheim/tesseract/wiki

python main.py
```

## Requirements
- OpenCV (`cv2`)
- NumPy
- PIL (Pillow)
- pytesseract (OCR engine)
- Tesseract OCR binary (install separately)

## How It Works
1. Loads all video files from the `videos/` directory
2. For each video:
   - Reads each frame and detects the car's position
   - Draws a path by connecting consecutive positions
   - Applies morphological operations to clean up the path
   - Uses OCR to detect letters from the drawn path
3. Combines all detected letters from all videos
4. Displays the final answer

## Expected Output
The system will output:
- Progress indicators for each video being processed
- Detected letters from each video
- Combined result showing all letters in sequence
- Saved path images (path_*.png) for manual inspection if OCR fails

## Video Format
- Resolution: 400x400 pixels
- Duration: 1:56:40 (~7000 seconds)
- Content: Black background with moving 2D toy car
- Multiple videos: Each video draws different letters

## Troubleshooting
If letters aren't detected:
1. Check the saved `path_*.png` images to see if the path is clear
2. Adjust morphological operation parameters if path is too fragmented
3. Manually inspect the path images if needed
4. Consider increasing dilation iterations if the path is too thin
