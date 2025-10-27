"""
ClearVision Off-Axis Projection Demo
Uses MediaPipe face tracking and PyOpenGL to create head-coupled perspective projection.
"""

import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks.python import vision
import pygame
from pygame.locals import *
from OpenGL.GL import *
from OpenGL.GLU import *
import os
import time
import threading

# Configuration - Screen physical dimensions and resolution
SCREEN_WIDTH_MM = 595.0
SCREEN_HEIGHT_MM = 395.0
SCREEN_WIDTH_PX = 1920
SCREEN_HEIGHT_PX = 1080
DEFAULT_VIEWING_DISTANCE_MM = 540.0

# MediaPipe setup
BaseOptions = mp.tasks.BaseOptions
FaceLandmarker = vision.FaceLandmarker
FaceLandmarkerOptions = vision.FaceLandmarkerOptions
VisionRunningMode = vision.RunningMode

# Global variables for thread-safe face tracking
latest_result = None
result_lock = threading.Lock()


class CalibrationData:
    """Store and manage calibration data"""
    def __init__(self):
        self.reference_distance_mm = DEFAULT_VIEWING_DISTANCE_MM
        self.reference_nose_x = 0.5  # Normalized coordinates
        self.reference_nose_y = 0.5
        self.is_calibrated = False


class StarfieldRenderer:
    """Handle OpenGL rendering of starfield with off-axis projection"""
    
    def __init__(self, image_path="image.png"):
        self.image_path = image_path
        self.texture_id = None
        self.image_width = 0
        self.image_height = 0
        
    def load_texture(self):
        """Load starfield image as OpenGL texture"""
        # Try to load the starfield image
        image = None
        if os.path.exists(self.image_path):
            image = cv2.imread(self.image_path)
            if image is None:
                raise FileNotFoundError(f"Starfield image not found. Please provide '{self.image_path}'")
            else:
                print(f"Loaded starfield image: {self.image_path}")
            
        # Convert BGR to RGB
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        self.image_height, self.image_width = image.shape[:2]
        
        # Create OpenGL texture
        self.texture_id = glGenTextures(1)
        glBindTexture(GL_TEXTURE_2D, self.texture_id)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, self.image_width, self.image_height, 
                     0, GL_RGB, GL_UNSIGNED_BYTE, image)
        
        print(f"Texture loaded: {self.image_width}x{self.image_height}")
    
    def setup_asymmetric_frustum(self, head_x_mm, head_y_mm, head_z_mm):
        """
        Calculate and apply asymmetric frustum for off-axis projection.
        
        Args:
            head_x_mm: Head X position in mm (0 is screen center, + is right)
            head_y_mm: Head Y position in mm (0 is screen center, + is up)
            head_z_mm: Head distance from screen in mm (+ is away from screen)
        """
        # Near and far clipping planes
        near = 1.0  # mm - very close to screen plane
        far = 10000.0  # mm - far clipping plane
        
        # Screen dimensions in mm (half widths for calculations)
        screen_half_w = SCREEN_WIDTH_MM / 2.0
        screen_half_h = SCREEN_HEIGHT_MM / 2.0
        
        # Calculate frustum bounds based on head position
        # The screen plane is at z=0, viewer is at z=head_z_mm
        # We need to calculate the view frustum from the viewer's perspective
        
        # Left/Right/Bottom/Top are defined at the near plane
        # Scale factors based on viewing distance
        scale = near / head_z_mm if head_z_mm > 0 else 0.01
        
        # Calculate frustum edges accounting for head offset
        left = (-screen_half_w - head_x_mm) * scale
        right = (screen_half_w - head_x_mm) * scale
        bottom = (-screen_half_h - head_y_mm) * scale
        top = (screen_half_h - head_y_mm) * scale
        
        # Apply the frustum
        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        glFrustum(left, right, bottom, top, near, far)
        
        # Set up modelview matrix
        glMatrixMode(GL_MODELVIEW)
        glLoadIdentity()
        
        # Position camera at head location looking at screen
        gluLookAt(head_x_mm, head_y_mm, head_z_mm,  # Eye position
                  head_x_mm, head_y_mm, 0.0,          # Look at point (screen plane)
                  0.0, 1.0, 0.0)                       # Up vector
    
    def render_starfield_quad(self):
        """Render the starfield as a textured quad at the screen plane"""
        if self.texture_id is None:
            return
        
        glEnable(GL_TEXTURE_2D)
        glBindTexture(GL_TEXTURE_2D, self.texture_id)
        
        # Draw a large quad at z=0 (screen plane)
        quad_size = max(SCREEN_WIDTH_MM, SCREEN_HEIGHT_MM) * 2
        
        glBegin(GL_QUADS)
        glTexCoord2f(0, 0)
        glVertex3f(-quad_size, -quad_size, 0)
        
        glTexCoord2f(1, 0)
        glVertex3f(quad_size, -quad_size, 0)
        
        glTexCoord2f(1, 1)
        glVertex3f(quad_size, quad_size, 0)
        
        glTexCoord2f(0, 1)
        glVertex3f(-quad_size, quad_size, 0)
        glEnd()
        
        glDisable(GL_TEXTURE_2D)


def face_landmarker_callback(result, output_image: mp.Image, timestamp_ms: int):
    """Callback for MediaPipe face landmarker results"""
    global latest_result
    with result_lock:
        latest_result = result


def get_nose_position(landmarks, image_width, image_height):
    """
    Extract nose tip position from face landmarks.
    Returns normalized coordinates (0-1 range).
    """
    NOSE_TIP = 1
    nose = landmarks[NOSE_TIP]
    return nose.x, nose.y


def run_calibration(camera, landmarker):
    """
    Run calibration routine.
    Shows center dot, user positions head at reference distance and confirms.
    """
    print("\n=== CALIBRATION MODE ===")
    print("Instructions:")
    print("1. Position yourself at a comfortable viewing distance")
    print("2. Look directly at the GREEN DOT in the center of the screen")
    print("3. Press SPACE to confirm and lock in calibration")
    print("4. Press ESC to skip calibration")
    print("========================\n")
    
    calibration = CalibrationData()
    calibrated = False
    timestamp = 0
    
    while not calibrated:
        ret, frame = camera.read()
        if not ret:
            break
        
        frame = cv2.flip(frame, 1)  # Mirror effect
        h, w, _ = frame.shape
        
        # Process with MediaPipe
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)
        landmarker.detect_async(mp_image, timestamp)
        timestamp += 1
        
        # Draw calibration UI
        center_x, center_y = w // 2, h // 2
        
        # Draw large green dot at center
        cv2.circle(frame, (center_x, center_y), 20, (0, 255, 0), -1)
        cv2.circle(frame, (center_x, center_y), 22, (255, 255, 255), 2)
        
        # Draw instruction text
        cv2.putText(frame, "Look at the GREEN DOT", (center_x - 200, center_y - 50),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        cv2.putText(frame, "Press SPACE to calibrate", (center_x - 220, center_y + 80),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        cv2.putText(frame, "Press ESC to skip", (center_x - 150, center_y + 110),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
        
        # Show current nose position if detected
        with result_lock:
            current_result = latest_result
        
        if current_result and current_result.face_landmarks:
            face_landmarks = current_result.face_landmarks[0]
            nose_x, nose_y = get_nose_position(face_landmarks, w, h)
            
            # Draw nose position indicator
            nose_px_x = int(nose_x * w)
            nose_px_y = int(nose_y * h)
            cv2.circle(frame, (nose_px_x, nose_px_y), 8, (0, 0, 255), -1)
            cv2.line(frame, (nose_px_x, nose_px_y), (center_x, center_y), (255, 255, 0), 2)
            
            # Show distance from center
            dist = np.sqrt((nose_px_x - center_x)**2 + (nose_px_y - center_y)**2)
            cv2.putText(frame, f"Distance from center: {dist:.1f}px", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        cv2.imshow("Calibration", frame)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord(' '):  # Space - confirm calibration
            with result_lock:
                current_result = latest_result
            
            if current_result and current_result.face_landmarks:
                face_landmarks = current_result.face_landmarks[0]
                nose_x, nose_y = get_nose_position(face_landmarks, w, h)
                
                calibration.reference_nose_x = nose_x
                calibration.reference_nose_y = nose_y
                calibration.reference_distance_mm = DEFAULT_VIEWING_DISTANCE_MM
                calibration.is_calibrated = True
                
                print("\n✓ Calibration complete!")
                print(f"  Reference position: ({nose_x:.3f}, {nose_y:.3f})")
                print(f"  Reference distance: {calibration.reference_distance_mm}mm\n")
                
                calibrated = True
            else:
                print("⚠ No face detected! Please ensure your face is visible and try again.")
        
        elif key == 27:  # ESC - skip calibration
            print("⚠ Calibration skipped. Using default values.")
            calibration.is_calibrated = False
            calibrated = True
    
    cv2.destroyWindow("Calibration")
    return calibration


def main():
    """Main demo loop"""
    global latest_result
    
    # Initialize camera
    camera = cv2.VideoCapture(0)
    if not camera.isOpened():
        print("Error: Camera not found!")
        return
    
    # Set camera resolution
    camera.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    
    # Initialize MediaPipe Face Landmarker
    options = FaceLandmarkerOptions(
        base_options=BaseOptions(model_asset_path="face_landmarker.task"),
        running_mode=VisionRunningMode.LIVE_STREAM,
        result_callback=face_landmarker_callback,
        num_faces=1
    )
    
    # CALIBRATION PHASE
    with FaceLandmarker.create_from_options(options) as landmarker:
        calibration = run_calibration(camera, landmarker)
    
    # END CALIBRATION - Clean everything up
    cv2.destroyAllWindows()
    cv2.waitKey(1)
    camera.release()
    
    # Reset state
    with result_lock:
        latest_result = None
    
    print("\nCalibration complete. Starting demo...")
    time.sleep(1)
    
    # DEMO PHASE - Fresh start
    # Reopen camera
    camera = cv2.VideoCapture(0)
    camera.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    
    # Recreate face landmarker
    with FaceLandmarker.create_from_options(options) as landmarker:
        # Initialize Pygame and OpenGL
        pygame.init()
        display = (SCREEN_WIDTH_PX, SCREEN_HEIGHT_PX)
        pygame.display.set_mode(display, DOUBLEBUF | OPENGL)
        pygame.display.set_caption("ClearVision Off-Axis Projection Demo")
        
        # Initialize OpenGL settings
        glEnable(GL_DEPTH_TEST)
        glClearColor(0.0, 0.0, 0.0, 1.0)
        
        # Create starfield renderer
        renderer = StarfieldRenderer()
        renderer.load_texture()
        
        # Main loop variables
        timestamp = 0
        clock = pygame.time.Clock()
        running = True
        frame_count = 0
        
        print("\n=== DEMO RUNNING ===")
        print("Move your head to see the off-axis projection effect!")
        print("Press ESC or close window to exit")
        print("====================\n")
        
        while running:
            # Handle pygame events
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        running = False
            
            # Capture and process camera frame
            ret, frame = camera.read()
            if not ret:
                # Skip this frame if camera read fails
                print("Warning: Camera frame read failed, skipping...")
                time.sleep(0.01)
                continue
            
            frame = cv2.flip(frame, 1)
            cam_h, cam_w, _ = frame.shape
            
            # Process with MediaPipe - use monotonic timestamp
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)
            timestamp = int(time.time() * 1000)  # Current time in milliseconds
            landmarker.detect_async(mp_image, timestamp)
            frame_count += 1
            
            # Get head position
            head_x_mm = 0.0
            head_y_mm = 0.0
            head_z_mm = calibration.reference_distance_mm
            
            with result_lock:
                current_result = latest_result
            
            if current_result and current_result.face_landmarks and calibration.is_calibrated:
                face_landmarks = current_result.face_landmarks[0]
                nose_x, nose_y = get_nose_position(face_landmarks, cam_w, cam_h)
                
                # Calculate offset from calibration reference
                # Map normalized coordinates to screen mm coordinates
                delta_x = nose_x - calibration.reference_nose_x
                delta_y = nose_y - calibration.reference_nose_y
                
                # Convert to mm (mapping camera FOV to screen size)
                # Assuming camera FOV maps roughly to screen size at viewing distance
                head_x_mm = delta_x * SCREEN_WIDTH_MM * 2.0  # Amplify for better effect
                head_y_mm = -delta_y * SCREEN_HEIGHT_MM * 2.0  # Negative because screen Y is inverted
            
            # Render OpenGL scene
            glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
            
            # Set up asymmetric frustum based on head position
            renderer.setup_asymmetric_frustum(head_x_mm, head_y_mm, head_z_mm)
            
            # Render the starfield
            renderer.render_starfield_quad()
            
            # Update display
            pygame.display.flip()
            clock.tick(60)  # Target 60 FPS for smooth rendering
        
        # Cleanup
        camera.release()
        pygame.quit()
        print("\nDemo ended.")


if __name__ == "__main__":
    main()
