"""
Stereo Camera Calibration Script

Usage:
    python calibrate.py

Requirements:
    - Two cameras connected (or two video files)
    - Checkerboard pattern visible to both cameras
    - Press 'c' to capture calibration image pair
    - Press 'q' to quit capture and run calibration
    - Press ESC to exit anytime
"""

import cv2
import numpy as np
import json
from pathlib import Path
from datetime import datetime
from picamera2 import Picamera2

# ============================================================================
# CONFIGURATION
# ============================================================================

# Checkerboard pattern (internal corners, not squares!)
CHECKERBOARD_SIZE = (9, 6)  # width x height of internal corners
SQUARE_SIZE = 25.0  # millimeters - MEASURE YOUR ACTUAL PRINTED PATTERN!

# Pi Camera numbers (0 for CAM0, 1 for CAM1 on Pi 5)
LEFT_CAMERA_ID = 0
RIGHT_CAMERA_ID = 1

# Camera resolution (Pi Camera 3 native is 4608x2592, but use lower for calibration)
CAMERA_WIDTH = 1920
CAMERA_HEIGHT = 1080

# Minimum number of calibration image pairs needed
MIN_CALIBRATION_IMAGES = 15

# Output directory for calibration data
OUTPUT_DIR = Path("calibration")
OUTPUT_DIR.mkdir(exist_ok=True)

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def prepare_object_points(checkerboard_size, square_size):
    """
    Generate 3D coordinates of checkerboard corners in real world
    
    Args:
        checkerboard_size: (width, height) of internal corners
        square_size: size of one square in mm
        
    Returns:
        objp: Nx3 array of 3D points
    """
    objp = np.zeros((checkerboard_size[0] * checkerboard_size[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:checkerboard_size[0], 
                            0:checkerboard_size[1]].T.reshape(-1, 2)
    objp *= square_size
    return objp


def find_and_refine_corners(gray_image, checkerboard_size):
    """
    Detect checkerboard corners and refine to sub-pixel accuracy
    
    Args:
        gray_image: Grayscale image
        checkerboard_size: (width, height) of internal corners
        
    Returns:
        success: Boolean indicating if corners were found
        corners: Refined corner locations (or None if not found)
    """
    # Find corners
    ret, corners = cv2.findChessboardCorners(
        gray_image,
        checkerboard_size,
        flags=cv2.CALIB_CB_ADAPTIVE_THRESH + 
              cv2.CALIB_CB_NORMALIZE_IMAGE +
              cv2.CALIB_CB_FAST_CHECK
    )
    
    if not ret:
        return False, None
    
    # Refine corners to sub-pixel accuracy
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    corners_refined = cv2.cornerSubPix(
        gray_image,
        corners,
        winSize=(11, 11),
        zeroZone=(-1, -1),
        criteria=criteria
    )
    
    return True, corners_refined


def init_pi_cameras(left_cam_id, right_cam_id, width, height):
    """
    Initialize both Pi Cameras
    
    Args:
        left_cam_id: Camera number for left camera (0 or 1)
        right_cam_id: Camera number for right camera (0 or 1)
        width: Image width
        height: Image height
        
    Returns:
        picam_left, picam_right: Initialized Picamera2 objects
    """
    # Initialize left camera
    picam_left = Picamera2(camera_num=left_cam_id)
    config_left = picam_left.create_preview_configuration(
        main={"size": (width, height), "format": "RGB888"} # This is wrong, I think YUV420 or BGR888 is needed, check "The Picamera2 Library"
    )
    picam_left.configure(config_left)
    picam_left.start()
    
    # Initialize right camera
    picam_right = Picamera2(camera_num=right_cam_id)
    config_right = picam_right.create_preview_configuration(
        main={"size": (width, height), "format": "RGB888"} # In fact, this whole line might be wrong - I think it should be create_video_configuration with a lower res
    )
    
    picam_right.configure(config_right)
    # Start preview with Preview.QTGL here and configure width and height explicitly
    picam_right.start()
    
    # Let cameras warm up
    import time
    time.sleep(2)
    
    print(f"✓ Left camera (CAM{left_cam_id}): {width}x{height}")
    print(f"✓ Right camera (CAM{right_cam_id}): {width}x{height}")
    
    return picam_left, picam_right


def capture_calibration_images(picam_left, picam_right, checkerboard_size):
    """
    Capture synchronized image pairs from both Pi Cameras for calibration
    
    Args:
        picam_left: Initialized left Picamera2 instance
        picam_right: Initialized right Picamera2 instance
        checkerboard_size: (width, height) of internal corners
        
    Returns:
        objpoints: List of 3D object points
        imgpoints_left: List of 2D image points from left camera
        imgpoints_right: List of 2D image points from right camera
        image_size: (width, height) of images
    """
    # Storage for calibration data
    objpoints = []  # 3D points in real world
    imgpoints_left = []  # 2D points in left camera
    imgpoints_right = []  # 2D points in right camera
    
    objp = prepare_object_points(checkerboard_size, SQUARE_SIZE)
    image_size = (CAMERA_WIDTH, CAMERA_HEIGHT)
    
    print("\n" + "="*60)
    print("CALIBRATION IMAGE CAPTURE")
    print("="*60)
    print(f"Target: {MIN_CALIBRATION_IMAGES} image pairs")
    print("\nControls:")
    print("  'c' - Capture image pair (when checkerboard detected in both)")
    print("  'q' - Quit capture and start calibration")
    print("  ESC - Exit program")
    print("="*60 + "\n")
    
    captured_count = 0
    
    try:
        while True:
            # Capture frames from both cameras
            # Picamera2 returns RGB, so we need to convert to BGR for OpenCV
            frame_left = picam_left.capture_array()
            frame_right = picam_right.capture_array()
            
            # Convert RGB to BGR for OpenCV
            frame_left = cv2.cvtColor(frame_left, cv2.COLOR_RGB2BGR)
            frame_right = cv2.cvtColor(frame_right, cv2.COLOR_RGB2BGR)
            
            # Convert to grayscale
            gray_left = cv2.cvtColor(frame_left, cv2.COLOR_BGR2GRAY)
            gray_right = cv2.cvtColor(frame_right, cv2.COLOR_BGR2GRAY)
            
            # Find corners in both images
            found_left, corners_left = find_and_refine_corners(gray_left, checkerboard_size)
            found_right, corners_right = find_and_refine_corners(gray_right, checkerboard_size)
            
            # Draw corners for visualization
            display_left = frame_left.copy()
            display_right = frame_right.copy()
            
            cv2.drawChessboardCorners(display_left, checkerboard_size, corners_left, found_left)
            cv2.drawChessboardCorners(display_right, checkerboard_size, corners_right, found_right)
            
            # Add status text
            status_left = "READY" if found_left else "NOT FOUND"
            status_right = "READY" if found_right else "NOT FOUND"
            color_left = (0, 255, 0) if found_left else (0, 0, 255)
            color_right = (0, 255, 0) if found_right else (0, 0, 255)
            
            cv2.putText(display_left, f"Left: {status_left}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, color_left, 2)
            cv2.putText(display_right, f"Right: {status_right}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, color_right, 2)
            cv2.putText(display_left, f"Captured: {captured_count}/{MIN_CALIBRATION_IMAGES}", 
                        (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            
            # Show frames (resize for display if needed)
            display_scale = 0.5  # Scale down for display
            display_left_scaled = cv2.resize(display_left, None, fx=display_scale, fy=display_scale)
            display_right_scaled = cv2.resize(display_right, None, fx=display_scale, fy=display_scale)
            combined = np.hstack([display_left_scaled, display_right_scaled])
            
            cv2.imshow('Stereo Calibration - Press C to capture, Q when done', combined)
            
            # Handle key presses
            key = cv2.waitKey(1) & 0xFF
            
            if key == 27:  # ESC
                print("\nCalibration cancelled")
                cv2.destroyAllWindows()
                return None, None, None, None
            
            elif key == ord('c'):
                if found_left and found_right:
                    # Store calibration data
                    objpoints.append(objp)
                    imgpoints_left.append(corners_left)
                    imgpoints_right.append(corners_right)
                    captured_count += 1
                    print(f"✓ Captured pair {captured_count}/{MIN_CALIBRATION_IMAGES}")
                    
                    # Visual feedback
                    cv2.imshow('Stereo Calibration - Press C to capture, Q when done', combined)
                    cv2.waitKey(200)  # Brief flash
                else:
                    print("✗ Checkerboard not detected in both cameras!")
            
            elif key == ord('q'):
                if captured_count >= MIN_CALIBRATION_IMAGES:
                    print(f"\n✓ Captured {captured_count} pairs. Starting calibration...")
                    break
                else:
                    print(f"\n✗ Need at least {MIN_CALIBRATION_IMAGES} pairs. Currently have {captured_count}")
    
    finally:
        cv2.destroyAllWindows()
    
    return objpoints, imgpoints_left, imgpoints_right, image_size


def calibrate_stereo(objpoints, imgpoints_left, imgpoints_right, image_size):
    """
    Perform stereo camera calibration
    
    Args:
        objpoints: List of 3D object points
        imgpoints_left: List of 2D points from left camera
        imgpoints_right: List of 2D points from right camera
        image_size: (width, height) of images
        
    Returns:
        calibration_data: Dictionary containing all calibration parameters
    """
    print("\n" + "="*60)
    print("RUNNING CALIBRATION")
    print("="*60)
    
    # Step 1: Calibrate left camera
    print("\n1. Calibrating left camera...")
    ret_left, K_left, dist_left, rvecs_left, tvecs_left = cv2.calibrateCamera(
        objpoints, imgpoints_left, image_size, None, None
    )
    print(f"   Left camera RMS reprojection error: {ret_left:.4f} pixels")
    
    # Step 2: Calibrate right camera
    print("\n2. Calibrating right camera...")
    ret_right, K_right, dist_right, rvecs_right, tvecs_right = cv2.calibrateCamera(
        objpoints, imgpoints_right, image_size, None, None
    )
    print(f"   Right camera RMS reprojection error: {ret_right:.4f} pixels")
    
    # Step 3: Stereo calibration
    print("\n3. Calibrating stereo pair...")
    stereocalib_criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 1e-5)
    
    ret_stereo, K_left, dist_left, K_right, dist_right, R, T, E, F = cv2.stereoCalibrate(
        objpoints,
        imgpoints_left,
        imgpoints_right,
        K_left, dist_left,
        K_right, dist_right,
        image_size,
        criteria=stereocalib_criteria,
        flags=cv2.CALIB_FIX_INTRINSIC
    )
    print(f"   Stereo calibration RMS error: {ret_stereo:.4f} pixels")
    
    # Step 4: Stereo rectification
    print("\n4. Computing rectification transforms...")
    R1, R2, P1, P2, Q, roi_left, roi_right = cv2.stereoRectify(
        K_left, dist_left,
        K_right, dist_right,
        image_size,
        R, T,
        alpha=0  # 0 = crop to valid pixels
    )
    
    # Calculate baseline
    baseline = np.linalg.norm(T)
    
    print("\n" + "="*60)
    print("CALIBRATION RESULTS")
    print("="*60)
    print(f"Baseline: {baseline:.2f} mm")
    print(f"Left camera focal length: {K_left[0,0]:.1f} px")
    print(f"Right camera focal length: {K_right[0,0]:.1f} px")
    print(f"Left camera center: ({K_left[0,2]:.1f}, {K_left[1,2]:.1f})")
    print(f"Right camera center: ({K_right[0,2]:.1f}, {K_right[1,2]:.1f})")
    print("="*60)
    
    # Package all calibration data
    calibration_data = {
        'K_left': K_left,
        'K_right': K_right,
        'dist_left': dist_left,
        'dist_right': dist_right,
        'R': R,  # Rotation from left to right camera
        'T': T,  # Translation from left to right camera
        'E': E,  # Essential matrix
        'F': F,  # Fundamental matrix
        'R1': R1,  # Rectification rotation for left
        'R2': R2,  # Rectification rotation for right
        'P1': P1,  # Projection matrix for left after rectification
        'P2': P2,  # Projection matrix for right after rectification
        'Q': Q,   # Disparity-to-depth mapping matrix
        'roi_left': roi_left,
        'roi_right': roi_right,
        'image_size': image_size,
        'baseline': float(baseline),
        'rms_error': float(ret_stereo)
    }
    
    return calibration_data


def save_calibration(calibration_data, filename):
    """
    Save calibration data to file
    
    Args:
        calibration_data: Dictionary of calibration parameters
        filename: Path to save file
    """
    # Convert numpy arrays to lists for JSON serialization
    data_to_save = {}
    for key, value in calibration_data.items():
        if isinstance(value, np.ndarray):
            data_to_save[key] = value.tolist()
        else:
            data_to_save[key] = value
    
    with open(filename, 'w') as f:
        json.dump(data_to_save, f, indent=2)
    
    print(f"\n✓ Calibration data saved to: {filename}")


def test_rectification(calibration_data, picam_left, picam_right):
    """
    Test rectification by displaying rectified images with epipolar lines
    
    Args:
        calibration_data: Dictionary of calibration parameters
        picam_left: Initialized left Picamera2 instance
        picam_right: Initialized right Picamera2 instance
    """
    print("\n" + "="*60)
    print("TESTING RECTIFICATION")
    print("="*60)
    print("Displaying rectified images with epipolar lines")
    print("Corresponding points should lie on the same horizontal line")
    print("Press any key to exit")
    print("="*60 + "\n")
    
    # Extract calibration parameters
    K_left = np.array(calibration_data['K_left'])
    K_right = np.array(calibration_data['K_right'])
    dist_left = np.array(calibration_data['dist_left'])
    dist_right = np.array(calibration_data['dist_right'])
    R1 = np.array(calibration_data['R1'])
    R2 = np.array(calibration_data['R2'])
    P1 = np.array(calibration_data['P1'])
    P2 = np.array(calibration_data['P2'])
    image_size = tuple(calibration_data['image_size'])
    
    # Compute rectification maps
    map_left_x, map_left_y = cv2.initUndistortRectifyMap(
        K_left, dist_left, R1, P1, image_size, cv2.CV_32FC1
    )
    map_right_x, map_right_y = cv2.initUndistortRectifyMap(
        K_right, dist_right, R2, P2, image_size, cv2.CV_32FC1
    )
    
    try:
        while True:
            # Capture frames
            frame_left = picam_left.capture_array()
            frame_right = picam_right.capture_array()
            
            # Convert RGB to BGR
            frame_left = cv2.cvtColor(frame_left, cv2.COLOR_RGB2BGR)
            frame_right = cv2.cvtColor(frame_right, cv2.COLOR_RGB2BGR)
            
            # Apply rectification
            rect_left = cv2.remap(frame_left, map_left_x, map_left_y, cv2.INTER_LINEAR)
            rect_right = cv2.remap(frame_right, map_right_x, map_right_y, cv2.INTER_LINEAR)
            
            # Draw epipolar lines (horizontal lines every 50 pixels)
            for y in range(0, rect_left.shape[0], 50):
                cv2.line(rect_left, (0, y), (rect_left.shape[1], y), (0, 255, 0), 1)
                cv2.line(rect_right, (0, y), (rect_right.shape[1], y), (0, 255, 0), 1)
            
            # Combine and resize for display
            combined = np.hstack([rect_left, rect_right])
            combined_scaled = cv2.resize(combined, None, fx=0.5, fy=0.5)
            
            # Display
            cv2.imshow('Rectified Stereo (Green lines = epipolar lines)', combined_scaled)
            
            if cv2.waitKey(1) != -1:
                break
    
    finally:
        cv2.destroyAllWindows()

# ============================================================================
# MAIN
# ============================================================================

def main():
    print("\n" + "="*60)
    print("STEREO CAMERA CALIBRATION")
    print("="*60)
    print(f"Checkerboard: {CHECKERBOARD_SIZE[0]}x{CHECKERBOARD_SIZE[1]} internal corners")
    print(f"Square size: {SQUARE_SIZE} mm")
    print(f"Left camera: {LEFT_CAMERA_ID}")
    print(f"Right camera: {RIGHT_CAMERA_ID}")
    print("="*60)
    
    # Initialize cameras once and reuse for rectification
    picam_left, picam_right = init_pi_cameras(
        LEFT_CAMERA_ID, RIGHT_CAMERA_ID, CAMERA_WIDTH, CAMERA_HEIGHT
    )
    
    try:
        # Step 1: Capture calibration images
        objpoints, imgpoints_left, imgpoints_right, image_size = capture_calibration_images(
            picam_left, picam_right, CHECKERBOARD_SIZE
        )
        
        if objpoints is None:
            return
        
        # Step 2: Run calibration
        calibration_data = calibrate_stereo(
            objpoints, imgpoints_left, imgpoints_right, image_size
        )
        
        # Step 3: Save calibration
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = OUTPUT_DIR / f"stereo_calibration_{timestamp}.json"
        save_calibration(calibration_data, output_file)
        
        # Step 4: Test rectification
        print("\nWould you like to test the rectification? (y/n): ", end='')
        response = input().strip().lower()
        
        if response == 'y':
            test_rectification(calibration_data, picam_left, picam_right)
        
        print("\n✓ Calibration complete!")
    finally:
        picam_left.stop()
        picam_right.stop()
        cv2.destroyAllWindows()


if __name__ == '__main__':
    main()