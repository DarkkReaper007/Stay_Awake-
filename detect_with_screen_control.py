"""
Face Presence Detection with Custom Trained CNN
Uses your custom-trained neural network to control screen power
"""

import torch
import torch.nn.functional as F
import cv2
import time
import ctypes
import numpy as np
from model import FaceDetectionCNN, SimpleFaceDetectionCNN


class ScreenController:
    """Controls Windows screen power state"""
    
    # Windows constants for screen control
    SC_MONITORPOWER = 0xF170
    WM_SYSCOMMAND = 0x0112
    HWND_BROADCAST = 0xFFFF
    
    SCREEN_ON = -1
    SCREEN_OFF = 2
    
    # Windows constants for preventing sleep
    ES_CONTINUOUS = 0x80000000
    ES_SYSTEM_REQUIRED = 0x00000001
    ES_DISPLAY_REQUIRED = 0x00000002
    
    @staticmethod
    def turn_off_screen():
        """Turn off the display (NOT sleep mode)"""
        ctypes.windll.user32.SendMessageW(
            ScreenController.HWND_BROADCAST,
            ScreenController.WM_SYSCOMMAND,
            ScreenController.SC_MONITORPOWER,
            ScreenController.SCREEN_OFF
        )
        print("🌙 Screen turned OFF (laptop still running)")
    
    @staticmethod
    def turn_on_screen():
        """Turn on the display"""
        ctypes.windll.user32.SendMessageW(
            ScreenController.HWND_BROADCAST,
            ScreenController.WM_SYSCOMMAND,
            ScreenController.SC_MONITORPOWER,
            ScreenController.SCREEN_ON
        )
        print("☀️ Screen turned ON")
    
    @staticmethod
    def prevent_sleep():
        """Prevent system from going to sleep while keeping screen control available"""
        ctypes.windll.kernel32.SetThreadExecutionState(
            ScreenController.ES_CONTINUOUS | ScreenController.ES_SYSTEM_REQUIRED
        )
        print("🔒 System sleep prevented (screen can still turn off)")
    
    @staticmethod
    def allow_sleep():
        """Re-enable system sleep"""
        ctypes.windll.kernel32.SetThreadExecutionState(ScreenController.ES_CONTINUOUS)
        print("🔓 System sleep re-enabled")


class CustomFaceDetector:
    """Face presence detector using custom trained CNN"""
    
    def __init__(self, model_path='best_model.pth', model_type='simple', 
                 absence_timeout=10, confidence_threshold=0.7):
        """
        Initialize the custom face detector
        
        Args:
            model_path: Path to trained model checkpoint
            model_type: 'simple' or 'full'
            absence_timeout: Seconds to wait before turning off screen
            confidence_threshold: Minimum confidence for face detection (0.0-1.0)
        """
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        
        # Load model
        print(f"Loading model from {model_path}...")
        if model_type == 'simple':
            self.model = SimpleFaceDetectionCNN()
        else:
            self.model = FaceDetectionCNN()
        
        checkpoint = torch.load(model_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model = self.model.to(self.device)
        self.model.eval()
        print("✅ Model loaded successfully")
        
        # Configuration
        self.absence_timeout = absence_timeout
        self.confidence_threshold = confidence_threshold
        self.img_size = (128, 128)
        
        # State tracking
        self.last_face_detected_time = time.time()
        self.screen_is_on = True
        self.face_detected = False
        self.confidence = 0.0
        
        # Initialize webcam
        self.cap = None
        self._init_camera()
        
        print(f"✅ Initialized - Absence timeout: {absence_timeout}s")
        print(f"   Confidence threshold: {confidence_threshold}")
    
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
    
    def preprocess_frame(self, frame):
        """Preprocess frame for model input"""
        # Resize
        resized = cv2.resize(frame, self.img_size)
        
        # Convert to RGB and normalize
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        normalized = rgb.astype(np.float32) / 255.0
        
        # Convert to tensor (CHW format)
        tensor = np.transpose(normalized, (2, 0, 1))
        tensor = torch.from_numpy(tensor).unsqueeze(0)  # Add batch dimension
        
        return tensor.to(self.device)
    
    def detect_face(self, frame):
        """
        Detect face in frame using custom CNN
        
        Returns:
            bool: True if face detected with sufficient confidence
        """
        with torch.no_grad():
            # Preprocess
            input_tensor = self.preprocess_frame(frame)
            
            # Forward pass
            output = self.model(input_tensor)
            probabilities = F.softmax(output, dim=1)
            
            # Get prediction
            confidence, predicted = torch.max(probabilities, 1)
            self.confidence = confidence.item()
            face_present = (predicted.item() == 1)  # Class 1 is face
            
            # Check confidence threshold
            if face_present and self.confidence >= self.confidence_threshold:
                return True
            
            return False
    
    def update_screen_state(self):
        """Update screen power state based on face presence"""
        current_time = time.time()
        time_since_last_face = current_time - self.last_face_detected_time
        
        # If face detected and screen is off, turn it on IMMEDIATELY
        if self.face_detected and not self.screen_is_on:
            print("\n👤 Face detected! Turning screen back ON...")
            ScreenController.turn_on_screen()
            self.screen_is_on = True
            # Wake up the screen by moving mouse slightly
            try:
                ctypes.windll.user32.SetCursorPos(100, 100)
                ctypes.windll.user32.SetCursorPos(101, 101)
            except:
                pass
        
        # If no face detected for timeout period and screen is on, turn it off
        elif not self.face_detected and self.screen_is_on:
            if time_since_last_face >= self.absence_timeout:
                print(f"\n⏰ No face detected for {self.absence_timeout}s.")
                print("   Turning screen OFF (laptop continues running)...")
                ScreenController.turn_off_screen()
                self.screen_is_on = False
    
    def display_status(self, frame):
        """Display status information on frame"""
        height, width = frame.shape[:2]
        
        # Create status overlay
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (width, 100), (0, 0, 0), -1)
        frame = cv2.addWeighted(overlay, 0.3, frame, 0.7, 0)
        
        # Display detection status
        if self.face_detected:
            status_text = f"✓ Face Detected (Conf: {self.confidence:.2f})"
            color = (0, 255, 0)
        else:
            status_text = f"✗ No Face (Conf: {self.confidence:.2f})"
            color = (0, 0, 255)
        
        cv2.putText(frame, status_text, (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        
        # Display screen state
        screen_text = f"Screen: {'ON' if self.screen_is_on else 'OFF'}"
        cv2.putText(frame, screen_text, (10, 65), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Display time since last face
        if not self.face_detected:
            time_away = time.time() - self.last_face_detected_time
            timer_text = f"Away: {time_away:.1f}s / {self.absence_timeout}s"
            cv2.putText(frame, timer_text, (width - 250, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 165, 255), 2)
        
        # Add "Custom CNN" badge
        cv2.putText(frame, "Custom CNN", (width - 150, height - 10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
        
        return frame
    
    def run(self):
        """Main detection loop"""
        print("\n🎥 Starting custom face presence detection...")
        print("Press 'q' to quit\n")
        
        # Prevent system sleep
        ScreenController.prevent_sleep()
        
        frame_skip_count = 0
        max_skip = 30
        last_status_print = time.time()
        
        try:
            while True:
                ret, frame = self.cap.read()
                if not ret:
                    frame_skip_count += 1
                    if frame_skip_count >= max_skip:
                        print("❌ Failed to grab frames continuously.")
                        break
                    time.sleep(0.1)
                    continue
                
                frame_skip_count = 0
                
                # Detect face using custom CNN
                self.face_detected = self.detect_face(frame)
                
                # Update last detected time if face is present
                if self.face_detected:
                    self.last_face_detected_time = time.time()
                
                # Update screen state based on presence
                self.update_screen_state()
                
                # Print status every 5 seconds when screen is off
                if not self.screen_is_on and (time.time() - last_status_print) > 5:
                    print("⏸️  Screen OFF - Waiting for face... (detection still running)")
                    last_status_print = time.time()
                
                # Display status on frame
                frame = self.display_status(frame)
                
                # Show the frame (with error handling for when screen is off)
                try:
                    cv2.imshow('Custom Face Detector', frame)
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord('q'):
                        print("\n👋 Exiting...")
                        break
                except:
                    time.sleep(0.1)
        
        finally:
            # Cleanup
            if self.cap:
                self.cap.release()
            cv2.destroyAllWindows()
            
            # Re-enable system sleep
            ScreenController.allow_sleep()
            
            # Ensure screen is turned back on before exiting
            if not self.screen_is_on:
                print("\n🔆 Ensuring screen is ON before exit...")
                ScreenController.turn_on_screen()
                try:
                    ctypes.windll.user32.SetCursorPos(100, 100)
                except:
                    pass


def main():
    """Main entry point"""
    print("="*60)
    print("Custom CNN Face Presence Detector")
    print("Turns OFF screen only (laptop stays awake)")
    print("="*60)
    
    # Configuration
    MODEL_PATH = 'best_model.pth'
    MODEL_TYPE = 'simple'
    ABSENCE_TIMEOUT = 10
    CONFIDENCE_THRESHOLD = 0.7
    
    try:
        detector = CustomFaceDetector(
            model_path=MODEL_PATH,
            model_type=MODEL_TYPE,
            absence_timeout=ABSENCE_TIMEOUT,
            confidence_threshold=CONFIDENCE_THRESHOLD
        )
        detector.run()
        
    except FileNotFoundError:
        print(f"\n❌ Error: Model file '{MODEL_PATH}' not found!")
        print("Please train the model first:")
        print("  1. Run 'python collect_data.py' to collect training data")
        print("  2. Run 'python train.py' to train the model")
        print("  3. Then run this script again")
    except KeyboardInterrupt:
        print("\n\n⚠️ Interrupted by user")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        print("\n✅ Cleanup complete")


if __name__ == "__main__":
    main()
