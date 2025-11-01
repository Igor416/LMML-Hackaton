"""
Autonomous Vehicle Tracking System for Sigmoid City
Analyze video recordings to track the mysterious erratically moving vehicle
The car draws letters while moving - we need to detect those letters!
"""

import cv2
import numpy as np
import os
from pathlib import Path
from PIL import Image
import pytesseract

VIDEO_DIR = Path("D:/Coding/Python/Practice/main/Europa/ukraine/videos")
OUTPUT_DIR = Path("D:/Coding/Python/Practice/main/Europa/ukraine/output")

class VehicleTracker:
    """Track and analyze vehicle movement to detect drawn letters"""
    
    def __init__(self, video_dir=VIDEO_DIR):
        self.video_dir = video_dir
        self.detected_letters = []
        
    def load_videos(self):
        """Load all video files from the videos directory"""
        if not self.video_dir.exists():
            print(f"Video directory '{self.video_dir}' not found. Please add video files.")
            return []
        video_files = list(self.video_dir.glob(f'*.mp4'))
        return sorted(video_files)
    
    def detect_vehicle(self, frame):
        """
        Detect the 2D toy car in a frame
        Since the background is black and the car is the only non-black object,
        we can use thresholding
        """
        # Convert to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Threshold to find non-black pixels (the car)
        _, binary = cv2.threshold(gray, 10, 255, cv2.THRESH_BINARY)
        
        # Find contours
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Find the largest contour (should be the car)
        if contours:
            largest_contour = max(contours, key=cv2.contourArea)
            area = cv2.contourArea(largest_contour)
            
            # The car should be small (not filling the entire frame)
            if 10 < area < 10000:
                x, y, w, h = cv2.boundingRect(largest_contour)
                return (x + w // 2, y + h // 2)  # Return center
        
        return None
    
    def track_and_draw_path(self, video_path):
        """
        Track the vehicle through the video and draw its path
        The path will reveal the letters being drawn
        """
        print(f"\nAnalyzing video: {video_path.name}")
        
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            print(f"Error: Could not open video {video_path}")
            return None
        
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        print(f"Video: {frame_count} frames @ {fps} fps, resolution: {width}x{height}")
        
        # Create a blank canvas to draw the path
        path_canvas = np.zeros((height, width), dtype=np.uint8)
        
        frame_idx = 0
        previous_position = None
        positions_tracked = 0
        last_update = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Detect car position
            position = self.detect_vehicle(frame)
            
            if position is not None:
                # Draw a line from previous position to current
                if previous_position is not None:
                    cv2.line(path_canvas, previous_position, position, 255, 2)
                previous_position = position
                positions_tracked += 1
            
            frame_idx += 1
            
            # Progress indicator
            if frame_idx - last_update >= 1000:
                progress = (frame_idx / frame_count) * 100
                print(f"Progress: {progress:.1f}%", end='\r')
                last_update = frame_idx
        
        cap.release()
        print(f"\nTracked vehicle through {positions_tracked} frames ({frame_idx} total frames)")
        print(f"Path canvas contains {np.sum(path_canvas > 0)} path pixels")
        
        return path_canvas
    
    def detect_letters_from_path(self, path_image, video_name):
        """
        Detect letters from the drawn path
        """
        print(f"\nDetecting letters from path for {video_name}...")
        
        # Apply morphological operations to clean up the path
        # Dilate to make lines thicker and more connected
        kernel = np.ones((3, 3), np.uint8)
        dilated = cv2.dilate(path_image, kernel, iterations=2)
        
        # Apply Gaussian blur to smooth the path
        blurred = cv2.GaussianBlur(dilated, (5, 5), 0)
        
        # Find contours (should be the letters)
        contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        print(f"Found {len(contours)} potential letter contours")
        
        # For better OCR, let's use pytesseract on the whole image
        # Convert to 3-channel for PIL
        path_bgr = cv2.cvtColor(dilated, cv2.COLOR_GRAY2BGR)
        
        # Use pytesseract to detect text
        try:
            # Configure tesseract for single character recognition
            custom_config = r'--oem 3 --psm 8 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ'
            
            detected_text = pytesseract.image_to_string(path_bgr, config=custom_config)
            detected_text = ''.join(c for c in detected_text if c.isalpha())
            
            if detected_text:
                print(f"Detected letters: {detected_text}")
                return detected_text
        except Exception as e:
            print(f"Error during OCR: {e}")
            print("Attempting alternative detection method...")
        
        # Alternative method: Extract regions and try OCR on each
        if len(contours) > 0:
            letters = []
            
            # Sort contours by position (left to right, top to bottom)
            contours = sorted(contours, key=lambda c: (cv2.boundingRect(c)[1], cv2.boundingRect(c)[0]))
            
            for i, contour in enumerate(contours):
                x, y, w, h = cv2.boundingRect(contour)
                
                # Filter by size (letters should be reasonably sized)
                if w > 20 and h > 30 and w < 200 and h < 200:
                    # Extract ROI
                    roi = dilated[y:y+h, x:x+w]
                    
                    # Create a padded version for better OCR
                    padded = cv2.copyMakeBorder(roi, 10, 10, 10, 10, cv2.BORDER_CONSTANT, value=0)
                    roi_bgr = cv2.cvtColor(padded, cv2.COLOR_GRAY2BGR)
                    
                    try:
                        custom_config = r'--oem 3 --psm 10 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ'
                        text = pytesseract.image_to_string(roi_bgr, config=custom_config)
                        text = text.strip().upper()
                        
                        if text and len(text) == 1 and text.isalpha():
                            letters.append(text)
                            print(f"  Detected: {text} (contour {i+1})")
                    except:
                        pass
            
            if letters:
                detected_word = ''.join(letters)
                print(f"Detected word: {detected_word}")
                return detected_word
        
        # If OCR fails, save the path image for manual inspection
        output_path = OUTPUT_DIR / f"path_{video_name}.png"
        cv2.imwrite(str(output_path), dilated)
        print(f"Saved path image to {output_path} for manual inspection")
        
        return None
    
    def process_video(self, video_path):
        """Process a single video and detect the drawn letters"""
        print(f"\n{'='*60}")
        print(f"PROCESSING: {video_path.name}")
        print('='*60)
        
        # Track vehicle and get path
        path_image = self.track_and_draw_path(video_path)
        
        if path_image is None or np.sum(path_image) == 0:
            print("No path detected!")
            self.detected_letters.append(None)
            return None
        
        # Detect letters from path
        detected_word = self.detect_letters_from_path(path_image, video_path.stem)
        
        # Store the result
        self.detected_letters.append({
            'video': video_path.name,
            'letters': detected_word
        })
        
        return detected_word
    
    def process_all_videos(self):
        """Process all videos in the directory"""
        video_files = self.load_videos()
        
        if not video_files:
            print("No video files found. Please add video files to the 'videos' directory.")
            return
        
        print(f"\n{'='*60}")
        print(f"Found {len(video_files)} video file(s)")
        print('='*60)
        
        # Process each video
        for video_file in video_files:
            self.process_video(video_file)
        
        # Display summary
        self.display_results()
    
    def display_results(self):
        """Display all detected letters"""
        print(f"\n{'='*60}")
        print("SUMMARY OF DETECTED LETTERS")
        print('='*60)
        
        all_letters = []
        
        for result in self.detected_letters:
            if result and result['letters']:
                print(f"{result['video']}: {result['letters']}")
                all_letters.append(result['letters'])
            else:
                print(f"{result['video']}: NO LETTERS DETECTED")
        
        print(f"\n{'='*60}")
        print("COMBINED RESULT")
        print('='*60)
        
        if all_letters:
            combined = ' '.join(all_letters)
            print(f"\n{combined}\n")
            
            # Try to form the answer
            print("="*60)
            print("THE SECRET ANSWER:")
            print("="*60)
            print(combined.upper().replace(' ', ''))
        else:
            print("\nNo letters detected in any video.")
            print("Please check the saved path images (path_*.png) for manual inspection.")


def main():
    """Main entry point"""
    print("="*60)
    print("SIGMOID CITY AUTONOMOUS VEHICLE TRACKING SYSTEM")
    print("="*60)
    print("\nMission: Track the mysterious vehicle and detect drawn letters")
    print("Goal: Decode the hidden secret from the movement patterns\n")
    
    tracker = VehicleTracker()
    tracker.process_all_videos()
    
    print("\n" + "="*60)
    print("MISSION COMPLETE")
    print("="*60)


if __name__ == "__main__":
    main()
