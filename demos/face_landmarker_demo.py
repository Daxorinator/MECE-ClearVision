"""
    Face Landmarker Head Pose Vector Plotting Demo
    Uses MediaPipe Face Landmarker to track facial landmarks and plot head pose direction vector
"""

import mediapipe as mp
from mediapipe.tasks.python import vision
import numpy as np
import cv2 as cv
import time
import threading

BaseOptions = mp.tasks.BaseOptions
FaceLandmarker = vision.FaceLandmarker
FaceLandmarkerOptions = vision.FaceLandmarkerOptions
VisionRunningMode = vision.RunningMode

# Global variables for thread-safe frame handling
# To-Do: Rewrite this so it doesn't rely on the GIL for thread safety,
# relies on 3.14 though so waiting for Mediapipe to update
latest_result = None
lock = threading.Lock()

def get_head_pose_vector(landmarks, image_width, image_height):
    """
    Calculate head pose direction using key facial landmarks.
    Returns the nose tip position and the direction vector.
    TO-DO: See if there's a better position for the face normal calculation, might be able to remove downward bias.
    """
    
    # Key landmark indices (MediaPipe 468-point model)
    NOSE_TIP = 1
    CHIN = 152
    LEFT_EYE = 33
    RIGHT_EYE = 263
    
    # Extract 2D coordinates
    def get_2d(idx):
        return np.array([landmarks[idx].x * image_width, 
                        landmarks[idx].y * image_height])
    
    # Extract 3D coordinates (x, y are normalized, z is relative depth)
    def get_3d(idx):
        return np.array([landmarks[idx].x, 
                        landmarks[idx].y, 
                        landmarks[idx].z])
    
    # Get 3D points for pose estimation
    nose_3d = get_3d(NOSE_TIP)
    chin_3d = get_3d(CHIN)
    left_eye_3d = get_3d(LEFT_EYE)
    right_eye_3d = get_3d(RIGHT_EYE)
    
    # Calculate face center
    face_center_3d = (nose_3d + chin_3d + left_eye_3d + right_eye_3d) / 4
    
    # Create vectors from face center to key points
    to_nose = nose_3d - face_center_3d
    to_chin = chin_3d - face_center_3d
    eye_line = right_eye_3d - left_eye_3d
    
    # Calculate face normal using cross product
    # This gives us the direction the face is pointing
    vertical_vec = to_nose - to_chin  # Approximate up vector of face
    normal = np.cross(eye_line, vertical_vec)

    # Normalize the direction vector
    normal = normal / (np.linalg.norm(normal) + 1e-6)

    # Shift the y-component upward slightly to correct downward bias
    normal[1] -= 0.3
    
    # Get 2D nose position for visualization
    nose_2d = get_2d(NOSE_TIP)
    
    return nose_2d, normal


def draw_pose_vector(image, nose_2d, direction_3d, scale=200):
    """
    Draw the head pose direction vector on the image.
    """
    # Convert 3D direction to 2D screen space
    # We ignore the z component for 2D projection
    end_point = nose_2d + np.array([direction_3d[0], direction_3d[1]]) * scale
    
    start = tuple(nose_2d.astype(int))
    end = tuple(end_point.astype(int))
    
    # Draw the direction arrow
    cv.arrowedLine(image, start, end, (0, 255, 0), 3, tipLength=0.3)
    cv.circle(image, start, 6, (0, 0, 255), -1)
    
    # Add depth indicator (color based on z direction)
    # Red = facing away, Blue = facing toward camera
    depth_color = (0, int(max(0, -direction_3d[2]) * 255), 
                   int(max(0, direction_3d[2]) * 255))
    cv.circle(image, start, 10, depth_color, 2)
    
    # Display direction info
    text = f"Dir: ({direction_3d[0]:.2f}, {direction_3d[1]:.2f}, {direction_3d[2]:.2f})"
    cv.putText(image, text, (10, 30), cv.FONT_HERSHEY_SIMPLEX, 
              0.6, (255, 255, 255), 2)


def result_callback(result, output_image: mp.Image, timestamp_ms: int):
    """Callback that stores the result for the main thread to process."""
    global latest_result
    with lock:
        latest_result = result


def main():
    global latest_result
    
    camera = cv.VideoCapture(0)
    if not camera.isOpened():
        print("Error: Camera not found")
        return
    
    options = FaceLandmarkerOptions(
        base_options=BaseOptions(model_asset_path="face_landmarker.task"),
        running_mode=VisionRunningMode.LIVE_STREAM,
        result_callback=result_callback,
        num_faces=1
    )
    
    with FaceLandmarker.create_from_options(options) as landmarker:
        timestamp = 0
        fps_start_time = time.time()
        fps_frame_count = 0
        fps = 0
        
        while True:
            ret, frame = camera.read()
            if not ret:
                break
            
            # Flip for mirror effect
            frame = cv.flip(frame, 1)
            h, w, _ = frame.shape
            
            # Convert to MediaPipe format and send for async processing
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)
            landmarker.detect_async(mp_image, timestamp)
            timestamp += 1
            
            # Get the latest result from callback and draw on current frame
            with lock:
                current_result = latest_result
            
            if current_result and current_result.face_landmarks:
                for face_landmarks in current_result.face_landmarks:
                    nose_pos, direction = get_head_pose_vector(face_landmarks, w, h)
                    draw_pose_vector(frame, nose_pos, direction, scale=200)
            
            # Calculate and display FPS
            fps_frame_count += 1
            elapsed_time = time.time() - fps_start_time
            if elapsed_time >= 1.0:
                fps = fps_frame_count / elapsed_time
                fps_frame_count = 0
                fps_start_time = time.time()
            
            cv.putText(frame, f"FPS: {fps:.1f}", (10, 60), cv.FONT_HERSHEY_SIMPLEX, 
                      0.6, (255, 255, 0), 2)
            
            # Show the frame
            cv.imshow("Head Pose Vector", frame)
            
            # Exit on 'q'
            if cv.waitKey(1) & 0xFF == ord('q'):
                break
    
    camera.release()
    cv.destroyAllWindows()


if __name__ == "__main__":
    main()