# MECE-ClearVision

Head-tracked off-axis projection demo using MediaPipe face tracking and OpenGL.

## Overview

This project demonstrates an **off-axis projection** technique that creates a 3D illusion by tracking the viewer's head position and adjusting the camera perspective in real-time. By using anamorphic projection techniques and viewer tracking, the rendered scene appears to have depth relative to the screen itself.

## How It Works

1. **Face Tracking**: MediaPipe Face Landmarker tracks the user's nose position via webcam
2. **Calibration**: User calibrates by looking at a center point from their normal viewing distance
3. **Asymmetric Frustum**: OpenGL frustum is dynamically adjusted based on head position
4. **Off-Axis Projection**: The starfield image is rendered with perspective that matches the viewer's actual position

## Files

- `clearvision_demo.py` - Main demo application with integrated face tracking and rendering
- `main.py` - Standalone face tracking demo (reference)
- `face_landmarker.task` - MediaPipe face landmark model (required)
- `starfield.png` or `starfield.jpg` - Starfield image for demo (optional - generates procedural if missing)

## Requirements

- Python 3.12
- Webcam
- Dependencies (see `pyproject.toml`):
  - mediapipe
  - numpy
  - opencv-python
  - pygame
  - pyopengl

## Installation

```bash
# Install dependencies (if using uv)
uv sync

# Or with pip
pip install -e .
```

## Usage

```bash
python clearvision_demo.py
```

### Calibration Process

1. The demo will start in calibration mode
2. Position yourself at a comfortable viewing distance (default: 540mm)
3. Look directly at the **green dot** in the center of the screen
4. Press **SPACE** to confirm and lock in the calibration
5. Press **ESC** to skip calibration (uses default values)

Calibration data is saved to `calibration.json` for future use.

### Demo Controls

- Move your head left/right/up/down to see the perspective shift
- **ESC** - Exit the demo
- Close window - Exit the demo

## Configuration

You can adjust the screen physical dimensions and viewing distance in `clearvision_demo.py`:

```python
# Screen physical dimensions (in mm)
SCREEN_WIDTH_MM = 595.0
SCREEN_HEIGHT_MM = 395.0

# Screen resolution
SCREEN_WIDTH_PX = 1920
SCREEN_HEIGHT_PX = 1080

# Default viewing distance (in mm)
DEFAULT_VIEWING_DISTANCE_MM = 540.0
```

## Technical Details

### Off-Axis Projection Math

The demo uses an **asymmetric viewing frustum** that adjusts based on viewer position:

1. The screen plane is at z=0
2. The viewer's head position is tracked in 3D space (x, y, z in mm)
3. OpenGL frustum bounds are calculated relative to head position:
   - `left = (-screen_half_w - head_x) * (near / head_z)`
   - `right = (screen_half_w - head_x) * (near / head_z)`
   - `bottom = (-screen_half_h - head_y) * (near / head_z)`
   - `top = (screen_half_h - head_y) * (near / head_z)`

This creates a perspective projection that appears correct from the viewer's actual position.

### Why It Works

Traditional rendering assumes the viewer is centered in front of the screen. Off-axis projection calculates what the viewer *should* see from their actual position, creating an anamorphic image that only looks "correct" from that specific viewpoint. When the viewer moves, the projection updates to match, creating the illusion that the scene has physical depth behind the screen.

## Future Enhancements

- Stereoscopic rendering for true binocular depth perception
- Multiple viewer support
- 3D scene rendering (currently uses flat starfield)
- Port to C using OpenCV and OpenGL for Raspberry Pi 5 deployment

## License

See project license.
