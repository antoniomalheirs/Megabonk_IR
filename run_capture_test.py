"""
Test script for DXCam Screen Capture setup.
Ensures we can grab frames at 60 FPS and displays them.
"""

import cv2
import time
from Capture.screen_capture import ScreenCapture

def main():
    print("Starting screen capture test...")
    # Optionally configure a region e.g. region=(0, 0, 1920, 1080)
    cap = ScreenCapture(target_fps=60, output_color="BGR")
    cap.start()
    
    frames_captured = 0
    start_time = time.time()
    
    try:
        print("Press 'q' in the window to exit.")
        while True:
            frame = cap.get_latest_frame()
            if frame is not None:
                frames_captured += 1
                
                # Display resized version to not fill screen
                display = cv2.resize(frame, (1280, 720))
                
                # Add FPS counter overlay
                elapsed = time.time() - start_time
                fps = frames_captured / elapsed if elapsed > 0 else 0
                cv2.putText(display, f"Capture FPS: {fps:.1f}", (10, 30), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                
                cv2.imshow("DXCam Test", display)
                
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    except KeyboardInterrupt:
        pass
    finally:
        cap.stop()
        cv2.destroyAllWindows()
        print("Capture stopped.")

if __name__ == "__main__":
    main()
