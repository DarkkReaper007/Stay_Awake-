"""
Data Collection Script for Face Detection Training
Collects training images from webcam for two classes:
- Class 0: No Face (empty frame or background)
- Class 1: Face Present
"""

import cv2
import os
import time
import numpy as np
from pathlib import Path


class DataCollector:
    """Collect training data from webcam"""
    
    def __init__(self, data_dir='training_data', img_size=(128, 128)):
        """
        Initialize data collector
        
        Args:
            data_dir: Directory to save collected images
            img_size: Size to resize images (width, height)
        """
        self.data_dir = Path(data_dir)
        self.img_size = img_size
        
        # Create directories
        self.face_dir = self.data_dir / 'face'
        self.no_face_dir = self.data_dir / 'no_face'
        
        self.face_dir.mkdir(parents=True, exist_ok=True)
        self.no_face_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize webcam
        self.cap = None
        self._init_camera()
        
        # Stats
        self.face_count = len(list(self.face_dir.glob('*.jpg')))
        self.no_face_count = len(list(self.no_face_dir.glob('*.jpg')))
        
    def _init_camera(self):
        """Initialize webcam with retry logic"""
        for camera_index in [0, 1, 2]:
            print(f"Trying camera index {camera_index}...")
            self.cap = cv2.VideoCapture(camera_index, cv2.CAP_DSHOW)
            if self.cap.isOpened():
                ret, test_frame = self.cap.read()
                if ret:
                    print(f"✅ Camera {camera_index} opened successfully")
                    self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                    self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
                    return
                else:
                    self.cap.release()
        
        raise Exception("Cannot open webcam")
    
    def save_image(self, frame, class_name):
        """Save a frame as training image"""
        # Resize frame
        resized = cv2.resize(frame, self.img_size)
        
        # Save to appropriate directory
        if class_name == 'face':
            filename = self.face_dir / f'face_{self.face_count:05d}.jpg'
            self.face_count += 1
        else:
            filename = self.no_face_dir / f'no_face_{self.no_face_count:05d}.jpg'
            self.no_face_count += 1
        
        cv2.imwrite(str(filename), resized)
        return filename
    
    def display_instructions(self, frame):
        """Display instructions on frame"""
        height, width = frame.shape[:2]
        
        # Create semi-transparent overlay
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (width, 150), (0, 0, 0), -1)
        frame = cv2.addWeighted(overlay, 0.4, frame, 0.6, 0)
        
        # Instructions
        instructions = [
            "Press '1' to capture FACE image",
            "Press '0' to capture NO FACE image",
            "Press 'q' to quit",
            f"Face images: {self.face_count} | No Face: {self.no_face_count}"
        ]
        
        y_offset = 25
        for i, text in enumerate(instructions):
            color = (0, 255, 0) if i < 2 else (255, 255, 255)
            cv2.putText(frame, text, (10, y_offset + i * 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        
        return frame
    
    def run(self):
        """Main data collection loop"""
        print("\n" + "="*60)
        print("Data Collection Mode")
        print("="*60)
        print("\nInstructions:")
        print("  1. Press '1' when your FACE is visible in frame")
        print("  2. Press '0' when NO FACE is visible (step away)")
        print("  3. Collect at least 200-300 images per class")
        print("  4. Vary your position, lighting, and expressions")
        print("  5. Press 'q' to quit\n")
        
        input("Press Enter to start collecting data...")
        
        try:
            while True:
                ret, frame = self.cap.read()
                if not ret:
                    print("Failed to grab frame")
                    continue
                
                # Display instructions
                display_frame = self.display_instructions(frame)
                
                # Show frame
                cv2.imshow('Data Collection', display_frame)
                
                # Handle key presses
                key = cv2.waitKey(1) & 0xFF
                
                if key == ord('q'):
                    break
                elif key == ord('1'):
                    # Save face image
                    filename = self.save_image(frame, 'face')
                    print(f"✅ Saved FACE image: {filename.name} (Total: {self.face_count})")
                    # Visual feedback
                    height, width = frame.shape[:2]
                    cv2.circle(display_frame, (width // 2, height // 2), 50, (0, 255, 0), 3)
                    cv2.imshow('Data Collection', display_frame)
                    cv2.waitKey(100)
                    
                elif key == ord('0'):
                    # Save no-face image
                    filename = self.save_image(frame, 'no_face')
                    print(f"✅ Saved NO FACE image: {filename.name} (Total: {self.no_face_count})")
                    # Visual feedback
                    height, width = frame.shape[:2]
                    cv2.circle(display_frame, (width // 2, height // 2), 50, (0, 0, 255), 3)
                    cv2.imshow('Data Collection', display_frame)
                    cv2.waitKey(100)
        
        finally:
            self.cap.release()
            cv2.destroyAllWindows()
            
            print("\n" + "="*60)
            print("Data Collection Complete!")
            print("="*60)
            print(f"Total Face images: {self.face_count}")
            print(f"Total No Face images: {self.no_face_count}")
            print(f"Images saved to: {self.data_dir}")
            
            # Recommendations
            if self.face_count < 200 or self.no_face_count < 200:
                print("\n⚠️  Recommendation: Collect at least 200 images per class")
                print("   More data = better model performance!")


def auto_collect_with_countdown(collector, class_name, num_images, delay=1.0):
    """
    Auto-collect images with countdown
    
    Args:
        collector: DataCollector instance
        class_name: 'face' or 'no_face'
        num_images: Number of images to collect
        delay: Delay between captures in seconds
    """
    print(f"\nAuto-collecting {num_images} {class_name.upper()} images...")
    print("Get ready!")
    time.sleep(2)
    
    for i in range(num_images):
        ret, frame = collector.cap.read()
        if ret:
            collector.save_image(frame, class_name)
            print(f"  Captured {i+1}/{num_images}", end='\r')
            time.sleep(delay)
    
    print(f"\n✅ Auto-collection complete!")


def main():
    """Main entry point"""
    print("="*60)
    print("Training Data Collection for Face Detection")
    print("="*60)
    
    try:
        collector = DataCollector()
        collector.run()
        
    except KeyboardInterrupt:
        print("\n\n⚠️  Interrupted by user")
    except Exception as e:
        print(f"\n❌ Error: {e}")


if __name__ == "__main__":
    main()
