import cv2
import numpy as np
import math

# Fixed window size - defined early so mouse_callback can use it
WINDOW_WIDTH = 800
WINDOW_HEIGHT = 600

def update_perspective(img, camera_x, camera_y, camera_z, yaw, pitch):
    h, w = img.shape[:2]
    
    # Define 3D coordinates for image corners (z=0 plane)
    points_3d = np.float32([
        [-w/2, -h/2, 0],  # top-left
        [w/2, -h/2, 0],   # top-right
        [w/2, h/2, 0],    # bottom-right
        [-w/2, h/2, 0]    # bottom-left
    ])
    
    # Create rotation matrices
    # Yaw rotation (around Y axis)
    cos_y = math.cos(yaw)
    sin_y = math.sin(yaw)
    R_yaw = np.array([
        [cos_y, 0, sin_y],
        [0, 1, 0],
        [-sin_y, 0, cos_y]
    ])
    
    # Pitch rotation (around X axis)
    cos_p = math.cos(pitch)
    sin_p = math.sin(pitch)
    R_pitch = np.array([
        [1, 0, 0],
        [0, cos_p, -sin_p],
        [0, sin_p, cos_p]
    ])
    
    # Combined rotation
    R = R_pitch @ R_yaw
    
    # Apply rotation to 3D points
    points_rotated = points_3d @ R.T
    
    # Add camera translation
    points_translated = points_rotated + [camera_x, camera_y, camera_z + 1000]  # Z offset for perspective
    
    # Project to 2D
    # Simple perspective projection
    f = 1000  # focal length
    points_2d = np.zeros((4, 2), dtype=np.float32)
    for i in range(4):
        z = points_translated[i, 2]
        # Allow points behind camera - OpenCV will handle clipping
        if z <= 0:
            z = 0.1  # Use a small positive value to avoid division by zero
        points_2d[i, 0] = points_translated[i, 0] * f / z + w/2
        points_2d[i, 1] = points_translated[i, 1] * f / z + h/2
    
    # Source points are the corners of the image
    pts1 = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
    pts2 = points_2d
    
    try:
        # Get and apply the perspective transform
        M = cv2.getPerspectiveTransform(pts1, pts2)
        result = cv2.warpPerspective(img, M, (w, h))
        return result
    except cv2.error:
        # If transform fails, return a black image instead of the original
        return np.zeros_like(img)

def mouse_callback(event, x, y, flags, param):
    global yaw, pitch
    if event == cv2.EVENT_MOUSEMOVE:
        # Use fixed window dimensions for mouse calculations
        h, w = WINDOW_HEIGHT, WINDOW_WIDTH
        # Convert mouse position to rotation angles
        yaw = (x - w/2) / (w/2) * math.pi / 3    # +/- 60 degrees rotation
        pitch = (y - h/2) / (h/2) * math.pi / 4   # +/- 45 degrees tilt
        print(f"Mouse: x={x}, y={y}, yaw={yaw:.2f}, pitch={pitch:.2f}")

# Load image
img = cv2.imread("image.png")  # Replace with your image path
if img is None:
    print("Error: Could not load image.png")
    # Create a test image if no image is found
    img = np.zeros((1080, 1920, 3), dtype=np.uint8)
    cv2.putText(img, "Test Image", (800, 540), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 2)
    cv2.rectangle(img, (100, 100), (1820, 980), (0, 255, 0), 2)
    # Add diagonal lines to help visualize perspective
    cv2.line(img, (100, 100), (1820, 980), (0, 0, 255), 2)
    cv2.line(img, (100, 980), (1820, 100), (0, 0, 255), 2)
    # Add grid to make panning more visible
    for i in range(0, 1920, 200):
        cv2.line(img, (i, 0), (i, 1080), (128, 128, 128), 1)
    for i in range(0, 1080, 200):
        cv2.line(img, (0, i), (1920, i), (128, 128, 128), 1)

# Initialize camera parameters
camera_x = 0      # Left/Right position
camera_y = 0      # Up/Down position
camera_z = 0      # Distance from image
yaw = 0          # Left/Right rotation
pitch = 0        # Up/Down tilt

# Create window and set mouse callback
cv2.namedWindow("Off-Axis Projection")
cv2.setWindowProperty("Off-Axis Projection", cv2.WND_PROP_TOPMOST, 1)
param = {'img': img, 'need_update': True}
cv2.setMouseCallback("Off-Axis Projection", mouse_callback, param)

# Movement speed
move_speed = 20

while True:
    # Always update every frame
    warped = update_perspective(img, camera_x, camera_y, camera_z, yaw, pitch)
    
    # Crop the center of the warped image to the fixed window size
    h, w = warped.shape[:2]
    start_y = max(0, (h - WINDOW_HEIGHT) // 2)
    start_x = max(0, (w - WINDOW_WIDTH) // 2)
    end_y = start_y + WINDOW_HEIGHT
    end_x = start_x + WINDOW_WIDTH
    
    # Handle case where image might be smaller than window
    if h >= WINDOW_HEIGHT and w >= WINDOW_WIDTH:
        display_img = warped[start_y:end_y, start_x:end_x].copy()
    else:
        # Image too small, just use the whole thing
        display_img = cv2.resize(warped, (WINDOW_WIDTH, WINDOW_HEIGHT))
    
    # Draw debug info ONLY on display image
    cv2.putText(display_img, f"Yaw: {yaw:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    cv2.putText(display_img, f"Pitch: {pitch:.2f}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    cv2.putText(display_img, f"Pos: ({camera_x:.0f}, {camera_y:.0f})", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    
    # Show the image
    cv2.imshow("Off-Axis Projection", display_img)
    
    # Handle keyboard input
    key = cv2.waitKey(1) & 0xFF
    
    # Movement controls
    if key == ord('a'):  # Move left
        camera_x -= move_speed
        print(f"Camera: x={camera_x}, y={camera_y}, z={camera_z}")
    elif key == ord('d'):  # Move right
        camera_x += move_speed
        print(f"Camera: x={camera_x}, y={camera_y}, z={camera_z}")
    elif key == ord('w'):  # Move up
        camera_y -= move_speed
        print(f"Camera: x={camera_x}, y={camera_y}, z={camera_z}")
    elif key == ord('s'):  # Move down
        camera_y += move_speed
        print(f"Camera: x={camera_x}, y={camera_y}, z={camera_z}")
    elif key == 27:  # ESC key to exit
        break

cv2.destroyAllWindows()