# ClearVision Demo - Usage Guide

## Quick Start

1. **Install dependencies:**
   ```bash
   pip install pygame pyopengl
   ```

2. **Prepare a starfield image (optional):**
   - Place a file named `starfield.png` or `starfield.jpg` in the project directory
   - If no image is provided, a procedural starfield will be generated automatically

3. **Run the demo:**
   ```bash
   python clearvision_demo.py
   ```

## Calibration Instructions

When you first run the demo, you'll see a calibration screen:

1. **Position yourself comfortably** at your typical viewing distance (about 540mm / 21 inches)
2. **Look directly at the GREEN DOT** in the center of the screen
3. You'll see a red dot showing your current nose position tracked by the camera
4. When you're centered and looking at the green dot, **press SPACE**
5. Calibration data will be saved to `calibration.json`

### Calibration Tips:
- Make sure your face is well-lit and clearly visible to the webcam
- Try to maintain the same viewing distance during the demo
- The red dot should be close to the green dot when you're centered
- If calibration doesn't work, press ESC to skip and use default values

## Using the Demo

Once calibrated, the main demo window will open showing the starfield:

- **Move your head LEFT/RIGHT** - The starfield will shift as if looking through a window
- **Move your head UP/DOWN** - The perspective will adjust vertically
- **Move FORWARD/BACKWARD** - (limited) The scale may adjust slightly
- **Press ESC** - Exit the demo

### Expected Effect:

When working correctly, you should see the starfield appear to have depth and move as if you're looking through a window into space. The parallax effect creates an illusion that the stars are at different depths behind the screen.

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

### The effect doesn't look 3D
- Make sure you completed calibration while looking at the center dot
- Try re-running and calibrating again
- Ensure you're at roughly the same distance as during calibration
- The effect is more pronounced with larger movements

### Poor framerate
- Close other applications
- Try a smaller starfield image
- The target is 30 FPS minimum - check terminal output for performance info

### Starfield not loading
- Check that `starfield.png` or `starfield.jpg` exists in the project directory
- If the file is named differently, update `StarfieldRenderer()` initialization in the code
- The demo will generate a procedural starfield if no image is found

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

### Finding Your Screen Dimensions:
1. Measure your screen width and height with a ruler (in mm or inches)
2. Convert inches to mm: multiply by 25.4
3. Update the constants in the code

## Technical Notes

- The demo uses **monocular vision** (single eye) - true stereoscopic 3D will be added later
- Head rotation is NOT used, only translation (X, Y position)
- The calibration assumes you maintain roughly the same viewing distance
- The asymmetric frustum is recalculated every frame based on head position

## Creating a Starfield Image

For best results, create a starfield image with:
- High resolution (2048x2048 or larger recommended)
- Black background
- White or colored dots representing stars
- Various star sizes for depth variety
- PNG or JPG format

You can use tools like:
- Photoshop with custom brushes
- GIMP with particle/star plugins
- Space Engine (screenshot mode)
- Online starfield generators
