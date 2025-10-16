import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import numpy as np
import time
import cv2 as cv

BaseOptions = mp.tasks.BaseOptions
FaceDetector = mp.tasks.vision.FaceDetector
FaceDetectorOptions = mp.tasks.vision.FaceDetectorOptions
VisionRunningMode = mp.tasks.vision.RunningMode

def draw_detections(image, detections):
    for detection in detections:
        bbox = detection.bounding_box
        start_point = (int(bbox.origin_x), int(bbox.origin_y))
        end_point = (int(bbox.origin_x + bbox.width), int(bbox.origin_y + bbox.height))
        cv.rectangle(image, start_point, end_point, (0, 255, 0), 2)
    cv.imshow('Face Detection', image)

camera = cv.VideoCapture(0)

# Create a face detector instance in video mode
options = FaceDetectorOptions(
    base_options=BaseOptions(model_asset_path="blaze_face_short_range.tflite"),
    running_mode=VisionRunningMode.LIVE_STREAM,
    result_callback=draw_detections
)

def main():
    with FaceDetector.create_from_options(options) as detector:
        while True:
            ret, frame = camera.read()
            if not ret:
                break
                
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)
            detector.detect_async(mp_image, frame_timestamp_ms=int(time.time() * 1000))
            
            detector.detect_async_result()

            if cv.waitKey(1) & 0xFF == ord('q'):
                break

        camera.release()
        cv.destroyAllWindows()

if __name__ == "__main__":
    main()
