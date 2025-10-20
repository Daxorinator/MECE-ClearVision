import cv2
import numpy as np
import math

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
        if z <= 0:  # Point is behind camera
            return img
        points_2d[i, 0] = points_translated[i, 0] * f / z + w/2
        points_2d[i, 1] = points_translated[i, 1] * f / z + h/2
    
    # Source points are the corners of the image
    pts1 = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
    pts2 = points_2d
    
    try:
        # Get and apply the perspective transform
        M = cv2.getPerspectiveTransform(pts1, pts2)
        result = cv2.warpPerspective(img, M, (w, h))
        print("Transform applied successfully")  # Debug print
        return result
    except cv2.error as e:
        print(f"Transform failed: {e}")
        return img

def mouse_callback(event, x, y, flags, param):
    global yaw, pitch
    if event == cv2.EVENT_MOUSEMOVE:
        h, w = param['img'].shape[:2]
        # Convert mouse position to rotation angles
        yaw = (x - w/2) / (w/2) * math.pi / 3    # +/- 60 degrees rotation
        pitch = (y - h/2) / (h/2) * math.pi / 4   # +/- 45 degrees tilt
        print(f"Mouse: x={x}, y={y}, yaw={yaw:.2f}, pitch={pitch:.2f}")

# Load image
img = cv2.imread("image.jpg")  # Replace with your image path
if img is None:
    print("Error: Could not load image.jpg")
    # Create a test image if no image is found
    img = np.zeros((480, 640, 3), dtype=np.uint8)
    cv2.putText(img, "Test Image", (200, 240), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 2)
    cv2.rectangle(img, (100, 100), (540, 380), (0, 255, 0), 2)
    # Add diagonal lines to help visualize perspective
    cv2.line(img, (100, 100), (540, 380), (0, 0, 255), 2)
    cv2.line(img, (100, 380), (540, 100), (0, 0, 255), 2)

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
    
    # Draw some debug info on the image
    debug_img = warped.copy()
    cv2.putText(debug_img, f"Yaw: {yaw:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.putText(debug_img, f"Pitch: {pitch:.2f}", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.putText(debug_img, f"Pos: ({camera_x:.0f}, {camera_y:.0f}, {camera_z:.0f})", (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    
    # Show the image
    cv2.imshow("Off-Axis Projection", debug_img)
    
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