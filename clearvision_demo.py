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
from PIL import Image
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
        """Load starfield image as OpenGL texture using Pygame"""
        # Load image using PIL
        if not os.path.exists(self.image_path):
            raise FileNotFoundError(f"Starfield image not found. Please provide '{self.image_path}'")
        
        print(f"Loading starfield image: {self.image_path}")
        
        # Load image using pygame
        texture_surface = pygame.image.load(self.image_path)
        
        # Resize to smaller dimensions if too large (helps with memory issues)
        orig_width = texture_surface.get_width()
        orig_height = texture_surface.get_height()
        
        # Limit to max 2048x2048 for compatibility
        max_size = 2048
        if orig_width > max_size or orig_height > max_size:
            scale = min(max_size / orig_width, max_size / orig_height)
            new_width = int(orig_width * scale)
            new_height = int(orig_height * scale)
            print(f"Resizing from {orig_width}x{orig_height} to {new_width}x{new_height}")
            texture_surface = pygame.transform.smoothscale(texture_surface, (new_width, new_height))
        
        self.image_width = texture_surface.get_width()
        self.image_height = texture_surface.get_height()
        print(f"Image size: {self.image_width}x{self.image_height}")
        
        # Convert to string data for OpenGL
        texture_data = pygame.image.tostring(texture_surface, "RGB", 1)
        
        # Create OpenGL texture
        self.texture_id = glGenTextures(1)
        print(f"Generated texture ID: {self.texture_id}")
        
        glBindTexture(GL_TEXTURE_2D, self.texture_id)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT)
        
        print(f"Uploading texture data...")
        # Use gluBuild2DMipmaps which is more robust
        gluBuild2DMipmaps(GL_TEXTURE_2D, GL_RGB, self.image_width, self.image_height,
                          GL_RGB, GL_UNSIGNED_BYTE, texture_data)
        
        print(f"✓ Texture loaded successfully: {self.image_width}x{self.image_height}")
    
    def setup_asymmetric_frustum(self, head_x_mm, head_y_mm, head_z_mm):
        """
        Calculate and apply asymmetric frustum for off-axis projection.
        
        Args:
            head_x_mm: Head X position in mm (0 is screen center, + is right)
            head_y_mm: Head Y position in mm (0 is screen center, + is up)
            head_z_mm: Head distance from screen in mm (+ is away from screen)
        """
        # Near and far clipping planes
        # Near plane should be at the screen distance for proper perspective
        near = head_z_mm * 0.95  # Slightly in front of viewer
        far = head_z_mm + 2000.0  # Far clipping plane
        
        # Screen dimensions in mm (half widths for calculations)
        screen_half_w = SCREEN_WIDTH_MM / 2.0
        screen_half_h = SCREEN_HEIGHT_MM / 2.0
        
        # Calculate frustum bounds at the near plane
        # Scale screen dimensions to near plane distance
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
        """Render the starfield as a textured quad behind the screen plane"""
        if self.texture_id is None:
            return
        
        glEnable(GL_TEXTURE_2D)
        glBindTexture(GL_TEXTURE_2D, self.texture_id)
        
        # Make quad MUCH larger than screen and place it behind the screen plane
        # This way edges will move in/out of view as head moves
        scale = 3.0  # Make it 3x larger than screen
        half_w = (SCREEN_WIDTH_MM / 2.0) * scale
        half_h = (SCREEN_HEIGHT_MM / 2.0) * scale
        depth = -300.0  # Place it 300mm behind the screen plane
        
        glBegin(GL_QUADS)
        glTexCoord2f(0, 0)
        glVertex3f(-half_w, -half_h, depth)
        
        glTexCoord2f(1, 0)
        glVertex3f(half_w, -half_h, depth)
        
        glTexCoord2f(1, 1)
        glVertex3f(half_w, half_h, depth)
        
        glTexCoord2f(0, 1)
        glVertex3f(-half_w, half_h, depth)
        glEnd()
        
        glDisable(GL_TEXTURE_2D)
    
    def render_depth_cues(self):
        """Render 3D reference objects to make the parallax effect visible"""
        glDisable(GL_TEXTURE_2D)
        
        # Draw some colored cubes at different depths
        cube_positions = [
            (0, 0, -100, (1.0, 0.0, 0.0)),      # Red cube close to screen
            (150, 100, -200, (0.0, 1.0, 0.0)),  # Green cube
            (-150, -80, -250, (0.0, 0.0, 1.0)), # Blue cube
            (100, -120, -150, (1.0, 1.0, 0.0)), # Yellow cube
            (-120, 90, -180, (1.0, 0.0, 1.0)),  # Magenta cube
        ]
        
        for x, y, z, color in cube_positions:
            glPushMatrix()
            glTranslatef(x, y, z)
            self.draw_cube(30, color)
            glPopMatrix()
    
    def draw_cube(self, size, color):
        """Draw a simple colored cube"""
        s = size / 2.0
        glColor3f(*color)
        
        glBegin(GL_QUADS)
        # Front face
        glVertex3f(-s, -s, s)
        glVertex3f(s, -s, s)
        glVertex3f(s, s, s)
        glVertex3f(-s, s, s)
        
        # Back face
        glVertex3f(-s, -s, -s)
        glVertex3f(-s, s, -s)
        glVertex3f(s, s, -s)
        glVertex3f(s, -s, -s)
        
        # Top face
        glVertex3f(-s, s, -s)
        glVertex3f(-s, s, s)
        glVertex3f(s, s, s)
        glVertex3f(s, s, -s)
        
        # Bottom face
        glVertex3f(-s, -s, -s)
        glVertex3f(s, -s, -s)
        glVertex3f(s, -s, s)
        glVertex3f(-s, -s, s)
        
        # Right face
        glVertex3f(s, -s, -s)
        glVertex3f(s, s, -s)
        glVertex3f(s, s, s)
        glVertex3f(s, -s, s)
        
        # Left face
        glVertex3f(-s, -s, -s)
        glVertex3f(-s, -s, s)
        glVertex3f(-s, s, s)
        glVertex3f(-s, s, -s)
        glEnd()
        
        glColor3f(1.0, 1.0, 1.0)


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
        screen = pygame.display.set_mode(display, DOUBLEBUF | OPENGL)
        pygame.display.set_caption("ClearVision Off-Axis Projection Demo")
        
        # Initialize OpenGL settings
        glEnable(GL_DEPTH_TEST)
        glClearColor(0.0, 0.0, 0.0, 1.0)
        
        # Clear the screen once to ensure context is ready
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        pygame.display.flip()
        
        # Create starfield renderer and load texture AFTER OpenGL context is ready
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
                delta_x = nose_x - calibration.reference_nose_x
                delta_y = nose_y - calibration.reference_nose_y
                
                # Convert to mm - use moderate amplification
                head_x_mm = delta_x * SCREEN_WIDTH_MM * 3.0
                head_y_mm = -delta_y * SCREEN_HEIGHT_MM * 3.0
                
                # Debug output every 30 frames
                if frame_count % 30 == 0:
                    print(f"Head: ({head_x_mm:.1f}, {head_y_mm:.1f}, {head_z_mm:.1f}) mm | "
                          f"Delta: ({delta_x:.3f}, {delta_y:.3f})")
            
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
