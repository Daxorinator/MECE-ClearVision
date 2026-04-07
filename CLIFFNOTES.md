# MECE-ClearVision — `oakd-lite` Technical CliffNotes

---

## System Overview

Real-time stereoscopic DIBR on a Jetson Nano (Maxwell GPU, OpenGL ES 3.1). The OAK-D Lite
offloads stereo rectification and disparity to its Myriad X VPU; the Jetson only renders.
A viewer-facing USB webcam feeds a TensorRT FaceMesh tracker. Both streams are consumed
each frame by a compute-shader pipeline that synthesises a novel viewpoint from the viewer's
measured 3D head position.

---

## OAK-D Receiver Thread (`oak_receiver.cpp`)

Configures a DepthAI pipeline on the device:
- **Cameras:** both mono sensors at 480p (`THE_480_P`), `LEFT` and `RIGHT` board sockets.
- **StereoDepth node:** left-right consistency check ON, subpixel mode ON (values ÷ 32 to
  get whole-pixel disparity), 7×7 median filter, configurable confidence threshold.
- **Runtime config:** a `XLinkIn` stream (`"stereoConfig"`) lets the host push a new
  `StereoDepthConfig` mid-stream without restarting the pipeline (toggle median, confidence).
- **Outputs:** disparity as `CV_16U`, rectified left/right as `CV_8U` grayscale, all 640×480.
- **Intrinsics + baseline:** read from EEPROM via `device->readCalibration()`. Baseline
  returned in cm, converted to metres on the host.
- **Queue discipline:** disparity queue blocks (drives frame rate); colour/rect queues use
  `tryGet` (non-blocking, last-frame fallback). Queue size = 1.

---

## Host-side Disparity Processing (`view_synthesis.cpp` — `paintGL`)

Per frame, before GPU upload:

1. **Speckle filter** — `cv::filterSpeckles(disp_s16, 0, maxSpeckle=150, maxDiff=16)`:
   reinterprets `CV_16U` as `CV_16SC1` (safe; OAK-D subpixel values are always positive).
2. **Subpixel → float** — `convertTo(CV_32F, 1.0/32.0)`.
3. **Resize** — `INTER_AREA` to `proc_w × proc_h` (640 × 480 at 1×, 320 × 240 at 0.5×, etc.).
4. **IIR temporal filter** (α = 0.3):
   `blended = 0.3·new + 0.7·prev`, but where `new == 0` (invalid), the previous value is
   preserved unchanged. This stops valid depth decaying into holes over time.
5. **Right disparity construction** (CPU, per scanline):
   - Forward pass: for each left pixel `(xl, y)` with disparity `d`, write `d` at right
     pixel `xr = round(xl − d)`, keeping the max when multiple left pixels map to the same
     right pixel (nearest object wins).
   - Backward fill: scan right-to-left; fill zero (disoccluded) pixels with the last
     nonzero value to the right (background depth propagates leftward into holes).
6. **Intrinsic scaling:** `vfx = fx · (proc_w / 640)`, same for `fy, cx, cy`.

All four textures (`tex_left_color`, `tex_left_disp`, `tex_right_color`, `tex_right_disp`)
are uploaded via `glTexSubImage2D` before any compute pass runs.

---

## FaceMesh Tracker Thread (`face_tracker.cpp`)

**Camera capture:** GStreamer pipeline — `v4l2src → JPEG → jpegdec → BGR → appsink`
(drop=true, max-buffers=1). Tries 1280×720 then 640×480. Focal length assumed from 70° H-FOV:
`focal_px = width / (2·tan(35°))`.

**TRT engine:** built from ONNX on first run (256 MB workspace, FP16 if available), then
serialised to a `.trt` cache file beside the ONNX and reloaded on subsequent runs.

**Detection:** Haar cascade on a 50%-scale greyscale+equalised frame, scale=1.1,
minNeighbors=4, minSize=60×60. Picks the largest detection, expands bbox by 40%, squares it,
crops and resizes to **192×192 RGB float [0,1]** for FaceMesh.

**Depth from iris size:**
```
iris_diam_px = mean of horizontal (pts 2↔4) and vertical (pts 1↔3) iris spans
Z = IRIS_DIAM_M · focal_px / iris_diam_px        (IRIS_DIAM_M = 11.7 mm)
X = (iris_cx − cam_cx) · Z / focal_px
Y = (iris_cy − cam_cy) · Z / focal_px
```
Both iris centres averaged for X/Y. IIR smooth on X, Y, Z (constant `FT_SMOOTH`).
**Calibration** (`C` key): stores current raw X/Y as reference offset; subsequent output is
the delta from that calibrated centre.

**Coordinate mapping into OAK-D space:** OAK-D faces +Z (into scene); viewer is behind it at
−Z. Face tracker x=right (tracker faces viewer) is negated to get OAK-D x. Default fallback
when tracker is inactive: `head = (0, 0, −1.05 m)`.

---

## DIBR Render Pipeline — Compute Shaders (OpenGL ES 3.1)

All image-space shaders use `layout(local_size_x=16, local_size_y=16)`. CLEAR uses `local_size_x=64`.

### UNPROJECT_CS
Reads `tex_left_disp` (R32F). Per pixel `(u, v)` with disparity `d`:
```
if d < 0.5 → write w=0 (invalid)
Z = fx · baseline / d
X = (u − cx) · Z / fx
Y = (v − cy) · Z / fy
```
Writes `vec4(X, Y, Z, 1)` to `tex_worldspace` (RGBA32F).

### CLEAR_CS
Zeroes `ssbo_depth` (uint array, one entry per output pixel) entirely on the GPU — avoids
uploading a multi-MB zero buffer from the host each frame.

### VWINDOW_DEPTH_CS
For each world point `W` in `tex_worldspace`, casts a ray from `head_pos` through `W` and
finds where it intersects the physical display plane:
```
dir = normalize(W − head_pos)
t   = dot(display_pos − head_pos, display_normal) / dot(dir, display_normal)
Q   = head_pos + t · dir                  ← intersection with display plane
u   = dot(Q − display_pos, display_right) / display_width  + 0.5
v   = dot(Q − display_pos, display_up)   / display_height + 0.5
```
Stores `atomicMax(depth[dst], floatBitsToUint(1/dist))` — packing `1/dist` as float bits
preserves the ordering so `atomicMax` picks the nearest surface.

### VWINDOW_COLOR_CS
Same ray projection as above; depth-tests against `ssbo_depth` before writing left-image
colour into `tex_output` (RGBA8).

### VWINDOW_RIGHT_CS
Fills disocclusions using the right rectified image. Per right pixel with disparity `d_r`:
- **Disocclusion filter:** skip if `d_r ≥ d_left · 0.7` (same world surface, not occluded)
  or if `d_left < 0.5` (textureless/invalid).
- Right camera origin is `+baseline` in X; otherwise same unproject + virtual-window math.
- Only writes into pixels where `depth[] == 0` (not already filled by left pass).

### JFA_INIT_CS → JFA_CS × log₂(max_dim) → JFA_GATHER_CS
Seeds: each filled pixel stores its coordinate packed as `x | (y << 16)` in an R32I texture;
holes store −1. Each JFA pass does a 9-neighbourhood lookup at the current step distance
(halves each pass: N/2 → N/4 → … → 1), propagating the nearest seed outward (ping-pong
between `tex_jfa_a` and `tex_jfa_b`). `JFA_GATHER_CS` reads each hole pixel's nearest-seed
coordinate, copies the colour from `tex_output`, writes to `tex_filled`.

### DISPLAY_VS / FS
Fullscreen quad samples `tex_filled`. A `u_scale` uniform applies NDC scaling to
letterbox/pillarbox the output to the correct aspect ratio.

---

## Key Numbers

| | |
|---|---|
| Camera resolution (native) | 640 × 480 |
| Processing scale | 1.0× / 0.5× / 0.25× (keys 1/2/3) |
| Subpixel divisor | 32 (OAK-D hardware) |
| Disparity temporal IIR α | 0.3 |
| Speckle filter max size | 150 px, max diff 16 |
| FaceMesh input | 192 × 192 RGB float |
| Iris diameter constant | 11.7 mm |
| Hole fill complexity | O(log N) passes (JFA) |
| Compute shader workgroup | 16 × 16 (image), 64 × 1 (clear) |

---

## Background Theory

### Stereo Geometry and Disparity
Two cameras separated by a known **baseline** B observe the same scene. For a point at
depth Z, the horizontal pixel shift between corresponding image points is the **disparity**
`d`. They are related by:
```
d = fx · B / Z   →   Z = fx · B / d
```
This only works after **rectification** — a homographic transform that re-projects both
images onto a common plane so that corresponding points lie on the same scanline. The
OAK-D handles rectification in hardware; the host receives already-rectified frames.

**Subpixel disparity:** hardware matchers (and the OAK-D's semi-global matcher) interpolate
between integer pixel offsets to produce finer depth resolution. The OAK-D encodes subpixel
values multiplied by 32, hence the host divides by 32 to recover pixel disparity.

**Left-right consistency check:** matching is run in both directions (left→right and
right→left). Pixels whose disparity disagreement exceeds a threshold are marked invalid.
Helps reject occlusion boundaries and textureless regions.

### Camera Intrinsics and Unprojection
A pinhole camera maps a 3D world point `(X, Y, Z)` to image pixel `(u, v)`:
```
u = fx · X/Z + cx
v = fy · Y/Z + cy
```
Inverting, given a pixel `(u, v)` and depth `Z`:
```
X = (u − cx) · Z / fx
Y = (v − cy) · Z / cy
```
`fx, fy` are focal lengths in pixels; `cx, cy` is the principal point (optical axis
intercept). These are read from the OAK-D's calibrated EEPROM.

### Depth-Image Based Rendering (DIBR) and the Virtual Window
DIBR synthesises a viewpoint that was never physically captured. The classical approach
is a **disparity-shift forward warp**: shift each pixel horizontally by a fraction of its
disparity to simulate a lateral camera offset. This is simple but only handles pure
horizontal parallax and doesn't scale correctly with viewer distance.

This system uses a **virtual window** model instead:
1. Every pixel is **unprojected** to a metric 3D world point using the intrinsics and depth.
2. A ray is cast from the viewer's head position through the world point.
3. The ray is intersected with the physical display plane.
4. The intersection coordinate, expressed in display-plane UV, gives the output pixel.

This is correct perspective projection from an arbitrary viewpoint, so lateral, vertical,
and depth-dependent parallax all emerge naturally. As the viewer moves closer, objects
appear larger; as the viewer moves sideways, occlusions reveal themselves correctly.

**Ray-plane intersection:** given ray origin `O` (head), direction `d`, plane point `P`
and plane normal `n`:
```
t = dot(P − O, n) / dot(d, n)      intersection parameter
Q = O + t · d                       intersection point in world space
```

### Forward Warping and the Depth Buffer Problem
Forward warping (splatting each source pixel to its destination) creates a many-to-one
mapping: multiple source pixels can land on the same output pixel, and some output pixels
may receive nothing (holes). A **GPU depth buffer** resolves the former — `atomicMax` on
an integer SSBO keeps the nearest surface. Holes are handled separately.

The `floatBitsToUint` trick: storing `1/dist` reinterpreted as a uint preserves the
distance ordering (both are positive, and IEEE 754 floats compare correctly as unsigned
integers), enabling `atomicMax` in compute shaders where floating-point atomics aren't
available in OpenGL ES 3.1.

### Disocclusion and Right-Eye Fill
When the virtual camera moves, regions occluded in the left camera may become visible.
The right rectified image provides data for these regions. A pixel from the right camera
is a true disocclusion (not just a textureless area) if its depth differs meaningfully
from the co-located left-camera depth (`d_right < d_left · 0.7` used here).

### Jump Flood Algorithm (JFA) for Hole Fill
JFA computes an approximate **nearest-seed Voronoi diagram** in O(log N) passes rather
than O(N) scanline search. Each pass propagates seed coordinates outward at a step
distance that halves each iteration (N/2, N/4, … 1). Each pixel checks its 9 neighbours
at the current step and adopts the seed that is closest to it. After log₂(max_dim)
passes, every pixel holds the coordinate of its nearest filled seed, from which the
colour is gathered in a final pass.

### Iris-Based Depth Estimation
The human iris has a near-constant physical diameter (~11.7 mm). Its apparent diameter in
pixels depends only on the camera focal length and the distance to the eye:
```
Z = IRIS_DIAM_M · focal_px / iris_diam_px
```
This is a form of **known-size ranging** — the same principle as estimating distance to
an object of known width from a calibrated camera. Lateral position follows directly from
the pinhole model once Z is known. The 70° H-FOV assumption provides `focal_px` when no
calibration file is available for the tracker camera.

### IIR Temporal Filtering
An **Infinite Impulse Response (IIR)** low-pass filter smooths a signal over time without
storing a fixed history window:
```
filtered[t] = α · new[t] + (1 − α) · filtered[t−1]
```
Small α (here 0.3 for disparity, a similar value for head position) gives heavy smoothing
but lags fast motion; large α tracks quickly but passes through noise. The implementation
explicitly skips the blend where `new == 0` (invalid disparity), preserving the last valid
estimate rather than decaying the depth map toward zero over occluded regions.

### TensorRT Inference
TensorRT is NVIDIA's inference optimisation library. It takes a trained model (here in
ONNX format), applies layer fusion, kernel auto-tuning, and optional FP16 quantisation,
then serialises a hardware-specific engine to disk. On subsequent runs the cached engine
loads in milliseconds rather than requiring re-optimisation. Inference is driven by
`context->executeV2(buffers)` where `buffers` is an array of GPU pointers — input and
output tensors stay on the GPU between batches.

---

## Hard Problems Encountered

### 1. Getting depthai-core to build at all

Before a single line of rendering code ran, most of a session was spent fighting the build.
The depthai-core CMake integration went through: a submodule with a missing `lz4` CMake
wrapper, a Hunter package manager gate that had to appear before `project()`, switching to
a pip-installed package, baking a virtualenv path into `CMakeLists.txt`, bumping the minimum
CMake version to 3.20, and suppressing CMP0071 warnings. `PresetType::HIGH_DENSITY` was
pulled from the API before a stable release so it had to be removed. Frames initially came
back via `getCvFrame()` which is unavailable without `DEPTHAI_OPENCV_SUPPORT` — replaced
with `getData()` + raw `cv::Mat` construction from the pointer and dimensions.

---

### 2. The face tracker went through five complete architectures

- **Lucas-Kanade optical flow** (original) — drifted without periodic re-detection.
- **TensorRT FaceMesh** — `setMaxWorkspaceSize` deprecated in TRT 8.2.1; the model download
  script had the wrong model number, wrong archive path, and used `git sparse-checkout`
  which requires a newer Git than L4T ships with.
- **Replaced with OpenCV DNN** — too slow on the Jetson.
- **Back to TRT + YuNet pre-detection** — YuNet ran full-frame detection, FaceMesh ran on
  the crop. The letterbox-to-square step was missing, so faces were squash-distorted and
  landmarks were wrong. The face presence score was treated as a binary flag; it isn't.
- **YuNet replaced with Haar cascade** — YuNet itself was too slow.

Throughout: the tracker camera was hardcoded to `/dev/video2` when it was `/dev/video0`;
the render came out upside-down; iris landmarks exist as separate named bindings in some
model variants but as indices 468–477 in the main output in others, requiring runtime
binding inspection to handle both.

---

### 3. Coordinate system chaos in the virtual window

The virtual window requires a consistent 3D space shared between OAK-D, display, and head:

- The face tracker faces the viewer, so its +X is the viewer's right — OAK-D +X is the
  scene's right (opposite). The missing negation of `hp.x` caused parallax to go backwards.
- The OAK-D faces into the scene along +Z, so the viewer is at negative Z. `hp.z` is a
  positive distance, so it also needed negation.
- The display plane was initially placed at Z=0 (the OAK-D origin), making the scene
  project hugely zoomed in. Moving to Z=−0.4 m fixed scale but then overshot. The final
  model places the display at Z=0 in world space with the OAK-D slightly behind it.

---

### 4. Right-eye disocclusion fill — three broken attempts

- **Attempt 1:** 2×2 splat on forward-warped pixels to close cracks. Accidentally filled
  holes with wrong adjacent-pixel colours. Reverted.
- **Attempt 2:** `VWINDOW_RIGHT_CS` wrote right-image colour into any empty output pixel.
  Caused ghosting — right-camera pixels landed on top of already-correct left-camera
  pixels. Fix: only write where `depth[] == 0`.
- **Attempt 3:** Still ghosted at depth discontinuities. Root cause: at the stereo baseline
  offset, the right camera sees the same surface at a slightly different disparity, so
  right pixels were filling non-disoccluded holes. Fix: disocclusion filter
  `d_right < d_left · 0.7` — if both cameras see roughly the same depth, it is the same
  surface, not an occlusion.

The right disparity construction also initially used a sign-flipped formula that placed
right-camera world points at the wrong X position.

---

### 5. StereoDepthConfig silently resetting pipeline settings

Using the runtime config channel (`XLinkIn → inputConfig`) to toggle the median filter
constructed a fresh `StereoDepthConfig` with default values, silently resetting subpixel
and left-right check to OFF on every update. Disparity quality degraded gradually after
any keypress until the process was restarted. Fix: explicitly re-assert subpixel and LR
check in every `setStereoConfig()` call.

---

### 6. Maxwell GPU OpenGL ES 3.1 format restrictions

The JFA seed texture was first declared as `rg32i` (two 32-bit ints per pixel for X and Y).
Maxwell under OpenGL ES 3.1 does not support `rg32i` as an image format — the shader
compiled without error but produced garbage at runtime. Fix: switch to `r32i` with seeds
packed as `x | (y << 16)`. Separately, `glClearBufferSubData` is absent from the L4T
`gl32.h` headers entirely, so the depth SSBO is zeroed each frame by a dedicated `CLEAR_CS`
compute shader rather than a host-side upload.

---

### 7. IIR temporal filter decaying depth into holes

The first IIR implementation applied `α·new + (1−α)·prev` unconditionally. Where the
OAK-D returned zero disparity (no stereo match), zero was blended in as if it were a valid
depth reading. Over time, valid depth estimates near persistent holes decayed toward zero,
growing the holes on every frame. Fix: detect where `new == 0` and skip the blend, preserving
the last valid estimate unchanged.

