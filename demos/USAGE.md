# clearvision_demo.py - Usage Guide

## Quick Start

1. **Install dependencies:**
   ```bash
   uv add pygame pyopengl
   ```
   **OR**
   ```bash
   uv sync
   ```

2. **Run the demo:**
   ```bash
   python clearvision_demo.py
   ```

## Calibration Instructions

When you first run the demo, you'll see a calibration screen:

1. **Position yourself comfortably** at your typical viewing distance (set to 540mm by default)
2. **Look directly at the GREEN DOT** in the center of the screen
3. You'll see a red dot showing your current nose position tracked by the camera
4. When you're centered and looking at the green dot, **press SPACE**

## Using the Demo

Once calibrated, the main demo window will open:

- **Move your head LEFT/RIGHT** - The background will shift as if looking through a window
- **Move your head UP/DOWN** - The perspective will adjust vertically
- **Press ESC** - Exit the demo

## Troubleshooting

### "Error: Camera not found!"
- Make sure your webcam is connected and not in use by another application
- Try unplugging and reconnecting the webcam
- Check Windows privacy settings to ensure apps can access the camera

### "No face detected!" during calibration
- Ensure good lighting on your face
- Move closer to the camera
- Remove glasses or reflective objects that might interfere
- Make sure the camera has a clear view of your face

## Configuration

You can modify these settings in `clearvision_demo.py`:

```python
# Screen physical size (in millimeters)
SCREEN_WIDTH_MM = 595.0  # Change to match your screen
SCREEN_HEIGHT_MM = 395.0

# Screen resolution
SCREEN_WIDTH_PX = 1920
SCREEN_HEIGHT_PX = 1080

# Default viewing distance
DEFAULT_VIEWING_DISTANCE_MM = 540.0  # Change to your preferred distance
```

## Technical Notes
- Head rotation is NOT used, only translation (X, Y position) - thus the projection is orthographic
- The calibration assumes you maintain roughly the same viewing distance
- The asymmetric frustum is recalculated every frame based on head position
