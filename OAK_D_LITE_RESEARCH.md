# OAK-D Lite Integration Research — MECE-ClearVision

*Generated from design session, 2026-03-26*

---

## Context

This document captures research and analysis for integrating an OAK-D Lite stereo camera
into the MECE-ClearVision DIBR (Depth-Image Based Rendering) view synthesis pipeline,
specifically for a **"see-through car panel"** application:

- OAK-D Lite mounted **outside the car**, pointing away from the vehicle
- Display panel **inside the car** acts as a virtual window
- Viewer's head tracked **independently** (separate USB webcam inside car)
- Goal: real-world parallax coupled to head position — move left, see further left around obstacles

---

## Part 1: OAK-D Lite Hardware

### Core Specifications

| Property | Value |
|---|---|
| VPU | Intel Movidius Myriad X (RVC2), 4 TOPS total, 1.4 TOPS NN |
| Stereo sensors | 2× OmniVision OV7251 — **global shutter**, monochrome |
| Stereo resolution | 640×480 (VGA) at up to **200 FPS** |
| Color sensor | Sony IMX214, 13 MP (4208×3120) |
| Color resolution | 4K/30 FPS, 1080P/60 FPS |
| Stereo baseline | **75 mm** |
| USB interface | USB-C, USB 3.2 Gen1, **5 Gbps**, bus-powered |
| Power draw | up to 5 W |
| Dimensions / weight | 91×28×17.5 mm / 61 g |
| IMU | Bosch BMI270 (6-axis) |

### Field of View

| Camera | DFOV | HFOV | VFOV |
|---|---|---|---|
| OV7251 (stereo pair) | 86° | 73° | 58° |
| IMX214 (color) | 81° | 69° | 54° |

### Depth Range

| Mode | MinZ | MaxZ |
|---|---|---|
| Standard 480P | ~35 cm | ~19 m |
| Extended disparity | ~20 cm | ~10 m |
| Practical sweet spot | 40 cm | 8 m |

The OV7251 uses a **global shutter** — no rolling-shutter skew on fast lateral head movements,
which is directly relevant to head-coupled parallax.

---

## Part 2: Onboard Depth Pipeline (Myriad X Hardware)

The Myriad X runs the entire stereo stack in dedicated hardware — **zero CPU cost on the Jetson host**.

### Hardware Stages

1. **Warp engine** — stereo rectification using calibration data from on-board EEPROM.
   Runs independently of Shave processors. Replaces `cv::remap` / CUDA remap entirely.

2. **Stereo SGBM hardware block** — dedicated Semi-Global Block Matching cost-aggregation
   hardware. Dumps full cost volume to internal memory.

3. **Shave vector processors** — subpixel interpolation on the cost volume,
   producing disparity with **up to 5 fractional bits** (32 sub-steps between integer disparities).

### Disparity Output Modes

| Mode | Encoding | Range |
|---|---|---|
| Standard (default) | RAW8 | 0–95 pixels |
| Extended disparity | RAW8 | 0–190 pixels (two-pass) |
| Subpixel 3-bit | RAW16 | 0–760 |
| Subpixel 4-bit | RAW16 | 0–1520 |
| Subpixel 5-bit | RAW16 | 0–3040 |

Note: extended disparity and subpixel mode are mutually exclusive.

### Depth Output

`uint16` in millimetres. Value `0` = invalid. Optionally RGB-aligned on-device.

### Confidence Map

RAW8, 0 = maximum confidence, 255 = minimum confidence. Device auto-discards pixels
above a configurable threshold before sending downstream. Full map also available as
a separate stream for manual filtering in the warp shader.

### On-Device Post-Processing Options

- Left-Right consistency check (`setLeftRightCheck(True)`) — discards L↔R mismatch pixels
- Speckle filter — removes isolated disparity blobs
- Temporal filter — smoothing across frames
- WLS filter — typically runs on host, not on-device

### Depth Quality Benchmarks

From arXiv:2501.07421 (empirical comparison):
- OAK-D Pro: <2% error at 4 m
- ZED 2 (NVIDIA GPU on host): <0.8% error at 2 m
- OAK-D competitive at close range (<1.5 m) and under GPU-constrained conditions

---

## Part 3: Compute Offload vs. Current Jetson Pipeline

### What Moves Off the Jetson Entirely

| Task | Current Jetson cost | With OAK-D Lite |
|---|---|---|
| Stereo rectification | ~3–5 ms CPU (or CUDA remap) | **0 ms — warp engine** |
| SGBM stereo matching | **20–30 ms CPU** (dominant bottleneck) | **0 ms — hardware SGBM block** |
| CUDA BM (BM mode) | 10–15 ms, blocks Maxwell GPU | 0 ms — not needed |
| Subpixel refinement | Not available (integer only) | 5 fractional bits, free |
| L-R consistency check | CPU, performance cost | On-device, free |
| Stereo calibration management | JSON file + manual rig | EEPROM, auto-applied |
| Camera synchronisation | Software-triggered frame grab | Hardware-synced stereo pair |

### What Stays on Jetson

| Task | Notes |
|---|---|
| DIBR warp (OpenGL ES 3.1 compute shaders) | Unchanged — still `(disparity_tex, color_tex, uniforms)` |
| RGB→texture conversion + GL upload | Minor change: source is USB frame, not local stereo output |
| Head tracking | Independent USB webcam inside car — unchanged |
| Novel viewpoint computation | Host-side maths from head position |

### Quantitative Impact

- Removing CPU SGBM at 640×480: **saves 20–30 ms/frame**
- Removing CUDA BM: **saves 10–15 ms, frees Maxwell GPU entirely for DIBR warp**
- Net: pipeline goes from ~5 FPS (SGBM path) to **25–30 FPS** before other bottlenecks

The OAK-D Lite stereo hardware runs to 120 FPS @ 480P. At 30 FPS it is at 25% capacity.

---

## Part 4: Root Cause Analysis — Why the Current Image is "Drastically Warped"

Three compounding problems, all visible in the source code.

### Problem 1: `disp_amplify = 15.0` is non-physical

The disparity texture (`GL_R32F`) stores real pixel disparity values from OpenCV
(after `×1/16` de-scaling). The depth splat shader does:

```glsl
float disp = texelFetch(u_disparity, src, 0).r * u_disp_scale;  // × 15.0
float dst_xf = float(src.x) - disp * u_shift;
```

With a near object at 1 m having ~60 px disparity at 0.5× scale:
- Amplified disparity: `60 × 15 = 900`
- At `u_shift = 0.8`: shift = `900 × 0.8 = 720 pixels` — beyond image width
- For `Δu_shift = 0.1` (small head move): near object moves **90 pixels** (14% of frame)

The image tears apart. `disp_amplify` was added to make flat-looking output more dramatic,
but it masks a deeper problem (see Problem 2).

### Problem 2: `u_shift` has no connection to physical geometry

```cpp
// face_tracker.h
u_shift = 0.5f + (ema_x - ref_x) * FT_SENSITIVITY;  // FT_SENSITIVITY = 0.8
```

`ema_x` is a **normalised screen fraction** (−1 to +1) from a camera with unknown field of
view and unknown viewer distance. `FT_SENSITIVITY = 0.8` is a hand-tuned constant with no
physical derivation.

The physically correct formula for pixel shift when the viewpoint moves by offset `Δx` is:

```
pixel_shift = disparity × (Δx / B)
```

where `B` is the stereo baseline in metres and `Δx` is the head's physical lateral
displacement in metres. The current code has no way to compute `Δx` — the face tracker
never estimates how far the viewer is from its camera. Without viewer depth, lateral screen
fraction cannot be converted to physical displacement.

### Problem 3: The warp model is conceptually wrong for this use case

`u_shift ∈ [0, 1]` represents *"fractional position between the two stereo cameras."*
Left camera = 0, right camera = 1, centre = 0.5. This is geometrically correct only when
the viewer's head moves *between* the two cameras. Here the cameras are **outside the car**
and the viewer is **inside** — the viewer is not interpolating between the stereo pair at
all. No value of `u_shift` corresponds to a physically meaningful viewpoint for this setup.

### Summary of Warp Problems

| Problem | Root cause | Fix |
|---|---|---|
| Objects move too far on head movement | `disp_amplify = 15` | Remove; use physical maths |
| Scale doesn't relate to real geometry | `FT_SENSITIVITY` without physical basis | Estimate viewer depth; compute `Δx` in metres |
| Wrong model for camera placement | Interpolation-between-cameras model | Virtual window / ray projection model |
| Only horizontal parallax | `dst_y = src_y` unchanged | Add `dy` component for vertical |
| Hard seams at depth edges | Integer-pixel disparity | OAK-D subpixel removes terracing |

---

## Part 5: The Correct Geometric Model — "Virtual Window"

### Formulation

```
World coordinate system rooted at OAK-D camera:

  [External scene]       [Car body / display]        [Interior]

  OAK-D camera ──────── captures color + depth ──►
                                        Display panel = virtual window
                                        (fixed position relative to camera)
                                                          │
                                                     Viewer head H
                                                     tracked in 3D
```

For each output display pixel `P_d = (u_d, v_d)`:

1. **Map display pixel to 3D world position `W_d`**
   Using display physical dimensions, position, and orientation relative to OAK-D camera.

2. **Cast ray from head `H` through `W_d`**
   `r = normalize(W_d - H)`

3. **Intersect ray with scene** using the depth map
   The depth map gives a 3D point cloud `Q(u,v)`. Find the depth-map point along ray `r`.

4. **Project `Q` into OAK-D camera image to get source colour**
   `(u_src, v_src) = K_cam · Q / Q.z`

### Why This Model Is Correct

Parallax for a near object at `Z_near`, head displacement `Δx`:
```
pixel_shift = fx × Δx / Z_near
```

For `fx = 381` (OAK-D 480P, 73° HFOV), `Δx = 0.1 m`, `Z_near = 2 m`:
```
pixel_shift = 381 × 0.1 / 2 = 19 pixels
```

For a far object at 8 m, same head movement: `4.8 pixels` — naturally smaller.

**No `disp_amplify` needed. No `FT_SENSITIVITY` needed. Physics is correct by construction.**
No object ever shifts beyond image width regardless of disparity magnitude.

### Calibration Requirements

| Parameter | Source |
|---|---|
| OAK-D intrinsics `(fx, fy, cx, cy)` | OAK-D EEPROM — loaded automatically by DepthAI API |
| Stereo baseline | OAK-D EEPROM — 75.0 mm |
| Display physical width/height (metres) | Measured once, hard-coded |
| Display-to-camera rotation + translation | One-time calibration (ArUco/checkerboard on display) |
| Head 3D position `(hx, hy, hz)` in camera space | Head tracker — **must return metres, not fractions** |

### Updated Face Tracker Output

The face tracker must return physical displacement in metres, not `u_shift ∈ [0, 1]`:

```cpp
// Known constant: average adult face width ≈ 0.16 m
// Measured: face bbox width in pixels, tracker focal length f_t
const float FACE_WIDTH_M = 0.16f;

float Z_head = FACE_WIDTH_M * f_tracker / bbox_width_px;    // depth in metres
float dx = (face_cx_px - ref_cx_px) * Z_head / f_tracker;  // lateral, metres
float dy = (face_cy_px - ref_cy_px) * Z_head / f_tracker;  // vertical, metres

// Output: vec3(dx, dy, Z_head) in metres — passed as u_head_pos uniform
```

`FT_SENSITIVITY` is replaced entirely by real geometry. `FT_SHIFT_MIN/MAX` are replaced
by physical clamping (e.g. ±0.3 m lateral).

---

## Part 6: New GLSL Pipeline Architecture

### Shader Pass Structure

```
Pass 0: CLEAR_CS          — clear depth SSBO (existing, unchanged)
Pass 1: UNPROJECT_CS      — NEW: disparity → per-pixel world-space XYZ texture
Pass 2: VWINDOW_SPLAT_CS  — REPLACES DEPTH_SPLAT_CS: world points → display-space z-buffer
Pass 3: BACKWARD_COLOR_CS — largely unchanged: backward sample color using z-buffer
Pass 4: JFA_CS (×N)       — REPLACES HOLE_FILL_CS: Jump Flood Algorithm hole fill
Pass 5: TEMPORAL_BLEND_CS — NEW (optional): reproject previous frame for large disocclusions
Display pass              — unchanged
```

### Pass 1 — Unproject to World Space (new shader)

```glsl
// UNPROJECT_CS
layout(local_size_x = 16, local_size_y = 16) in;
uniform sampler2D u_disparity;
uniform float u_fx, u_fy, u_cx, u_cy;
uniform float u_baseline;
layout(rgba32f, binding = 0) writeonly uniform highp image2D u_worldspace;

void main() {
    ivec2 src = ivec2(gl_GlobalInvocationID.xy);
    float d = texelFetch(u_disparity, src, 0).r;   // pixels (subpixel from OAK-D)
    if (d < 0.5) { imageStore(u_worldspace, src, vec4(0)); return; }

    float Z = u_fx * u_baseline / d;               // metres
    float X = (float(src.x) - u_cx) * Z / u_fx;
    float Y = (float(src.y) - u_cy) * Z / u_fy;
    imageStore(u_worldspace, src, vec4(X, Y, Z, 1.0));
}
```

### Pass 2 — Virtual Window Forward Splat (replaces DEPTH_SPLAT_CS)

```glsl
// VWINDOW_SPLAT_CS
layout(rgba32f, binding = 0) readonly uniform highp image2D u_worldspace;
layout(std430, binding = 1) buffer DepthBuffer { uint depth[]; };
uniform vec3 u_head_pos;          // metres, in camera coordinate frame
uniform mat4 u_display_matrix;    // maps world point to display UV
uniform ivec2 u_output_size;

void main() {
    ivec2 src = ivec2(gl_GlobalInvocationID.xy);
    vec4 W = imageLoad(u_worldspace, src);
    if (W.w < 0.5) return;

    // Ray from head through display:
    // Project world point W.xyz from head's perspective onto display plane
    vec3 Q_rel = W.xyz - u_head_pos;
    vec4 d_uv = u_display_matrix * vec4(Q_rel, 1.0);
    vec2 uv = d_uv.xy / d_uv.w;                         // display UV [0,1]

    if (uv.x < 0.0 || uv.x > 1.0 || uv.y < 0.0 || uv.y > 1.0) return;

    ivec2 dst = ivec2(uv * vec2(u_output_size));
    uint depth_val = floatBitsToUint(W.z);               // Z in metres
    atomicMax(depth[dst.y * u_output_size.x + dst.x], depth_val);
}
```

### Pass 4 — Jump Flood Algorithm Hole Fill (replaces horizontal scan)

The current `HOLE_FILL_CS` does a horizontal scan (search left/right for nearest valid pixel).
This produces horizontal **streaks** in 2D holes (e.g. behind a pillar). The JFA runs
`ceil(log2(max(W, H)))` passes and finds the nearest valid pixel in all directions:

```glsl
// JFA_CS — run once per step size, halving step each pass
// Pass 0: step = max(W,H)/2
// Pass 1: step = max(W,H)/4
// ...
// Pass N: step = 1
// Total: ~10 passes for 640×480

layout(local_size_x = 16, local_size_y = 16) in;
uniform int u_step;
uniform ivec2 u_output_size;
layout(rg32i, binding = 0) uniform highp iimage2D u_seed;  // stores (src_x, src_y) of nearest valid

void main() {
    ivec2 pos = ivec2(gl_GlobalInvocationID.xy);
    ivec2 best = imageLoad(u_seed, pos).xy;
    float best_dist = (best.x >= 0) ? 0.0 : 1e9;

    for (int dy = -1; dy <= 1; dy++) {
        for (int dx = -1; dx <= 1; dx++) {
            ivec2 nb = pos + ivec2(dx, dy) * u_step;
            if (nb.x < 0 || nb.x >= u_output_size.x ||
                nb.y < 0 || nb.y >= u_output_size.y) continue;
            ivec2 seed = imageLoad(u_seed, nb).xy;
            if (seed.x < 0) continue;
            vec2 diff = vec2(pos - seed);
            float dist = dot(diff, diff);
            if (dist < best_dist) { best_dist = dist; best = seed; }
        }
    }
    imageStore(u_seed, pos, ivec4(best, 0, 0));
}
```

**Result**: every hole pixel knows the 2D-nearest valid pixel, not just the nearest horizontal
neighbour. Eliminates streaking at corners and large disocclusion regions.

### New Shader Uniforms

| Old uniform | New uniform | Change |
|---|---|---|
| `u_shift` (float 0–1) | `u_head_pos` (vec3, metres) | Physical position replaces fractional shift |
| `u_disp_scale` (15.0) | *(removed)* | Ad-hoc amplifier replaced by physics |
| — | `u_fx, u_fy, u_cx, u_cy` | Camera intrinsics for unproject |
| — | `u_baseline` | Stereo baseline in metres (0.075 for OAK-D Lite) |
| — | `u_display_matrix` | mat4: world point → display UV projection |

---

## Part 7: Additional GPU Improvements (Freed Maxwell Headroom)

### Confidence-Guided Depth Upload

OAK-D Lite provides a confidence map (RAW8, 0 = best). Before uploading disparity to GPU,
zero out pixels whose confidence exceeds a threshold:

```cpp
// On host, before glTexSubImage2D:
// conf_map: uint8 RAW8, same resolution as disparity
// disp_map: uint16 subpixel disparity
for (int i = 0; i < W*H; i++)
    if (conf_map[i] > CONF_THRESHOLD)   // e.g. 200
        disp_map[i] = 0;
```

Or do this in a compute shader (one extra pass) to keep it on GPU.

### Joint Bilateral Depth Upsampling

OAK-D stereo is 480P; color is up to 1080P from IMX214. If running the DIBR warp at
color resolution, upsample the depth map guided by the color image:

```glsl
// JBUP_CS — joint bilateral upsample
// For each high-res pixel, average low-res depth neighbours weighted by
// colour similarity to the high-res colour at that position
float weight_sum = 0.0, depth_sum = 0.0;
for each 3×3 neighborhood in low-res depth:
    float colour_dist = length(hr_colour - lr_colour_at_neighbour);
    float w = exp(-colour_dist * sigma_colour) * spatial_weight;
    depth_sum += w * lr_depth;
    weight_sum += w;
float upsampled_depth = depth_sum / weight_sum;
```

This prevents depth edges from "bleeding" into the colour texture during upsampling —
a common artifact when naively upsampling depth with bilinear filtering.

### Temporal Depth Stabilisation

Add an EWA (Exponentially Weighted Average) pass for the world-space texture:

```glsl
// Each frame: blend current world-space Z with previous frame's Z
// compensating for head motion (parallax shift between frames)
float Z_stable = alpha * Z_current + (1 - alpha) * Z_prev_reprojected;
```

Requires reprojecting the previous frame's depth into the current frame using
`Δhead_pos` between frames. Reduces temporal jitter in warp output significantly.

### Vertical Parallax

The current warp only shifts horizontally. The virtual window model naturally supports
vertical parallax too — when the viewer's head moves up, objects below the horizon appear
higher. In `VWINDOW_SPLAT_CS`, both `X` and `Y` components of the head-relative projection
are used, so vertical parallax comes for free with the new model.

---

## Part 8: DepthAI Integration (C++ API)

### Pipeline Structure

```python
# Python reference — C++ API is structurally identical

pipeline = dai.Pipeline()
camLeft  = pipeline.create(dai.node.MonoCamera)
camRight = pipeline.create(dai.node.MonoCamera)
camRGB   = pipeline.create(dai.node.ColorCamera)
stereo   = pipeline.create(dai.node.StereoDepth)

camLeft.setBoardSocket(dai.CameraBoardSocket.LEFT)
camRight.setBoardSocket(dai.CameraBoardSocket.RIGHT)
camLeft.setResolution(dai.MonoCameraProperties.SensorResolution.THE_480_P)
camRight.setResolution(dai.MonoCameraProperties.SensorResolution.THE_480_P)

stereo.setDefaultProfilePreset(dai.node.StereoDepth.PresetType.HIGH_DENSITY)
stereo.setLeftRightCheck(True)
stereo.setSubpixel(True)           # 5 fractional bits
stereo.setDepthAlign(dai.CameraBoardSocket.RGB)  # depth aligned to color FOV

camLeft.out.link(stereo.left)
camRight.out.link(stereo.right)

# XLink outputs
xout_disp  → stereo.disparity   # RAW16 subpixel
xout_depth → stereo.depth       # uint16 mm
xout_conf  → stereo.confidenceMap
xout_color → camRGB.isp         # BGR, up to 4K
```

### Jetson Nano Setup

```bash
# udev rules (required first time)
echo 'SUBSYSTEM=="usb", ATTRS{idVendor}=="03e7", MODE="0666"' | \
  sudo tee /etc/udev/rules.d/80-movidius.rules
sudo udevadm control --reload-rules && sudo udevadm trigger

# DepthAI Python (prebuilt wheel for Jetson)
python3 -m pip install depthai -U
# Use depthai 2.x on JetPack 4.x (L4T R32) — not 3.x
```

### Power Caveat

OAK-D Lite draws up to 5 W bus-powered. Jetson Nano B01 USB ports are rated 4.5 W —
marginal with other USB devices active. Use a powered USB hub or Y-cable if instability
occurs, especially with the head-tracking USB webcam also connected.

### C++ Host Thread Pattern

The DepthAI receive thread mirrors the existing `face_tracker.cpp` pattern:

```cpp
// oak_receiver.h — analogous to face_tracker.h
// Background thread: polls disparity, color, confidence queues
// Writes into shared cv::Mat with mutex protection
// Main thread reads these mats in paintGL() exactly as it currently reads
// disp_l_float and left_rgba
```

---

## Part 9: Migration Map

| Current component | Action |
|---|---|
| `calibration.cpp` | Can be retired — OAK-D EEPROM holds stereo calibration |
| SGBM/BM stereo code in `view_synthesis.cpp` | Remove entirely |
| `cv::remap` / CUDA remap calls | Remove entirely |
| `face_tracker.cpp` (`/dev/video2`) | Retain as-is for inside-car head tracking |
| `disp_amplify` uniform + key binding | Remove |
| `u_shift` uniform | Replace with `u_head_pos` vec3 |
| `DEPTH_SPLAT_CS` | Replace with `UNPROJECT_CS` + `VWINDOW_SPLAT_CS` |
| `HOLE_FILL_CS` | Replace with JFA multi-pass |
| `BACKWARD_COLOR_CS` | Minor changes — reads Z in metres from z-buffer |
| `CLEAR_CS` | Unchanged |
| Display pass | Unchanged |
| Calibration JSON | Replace with display-to-camera pose JSON |

---

## Part 10: Side-by-Side Pipeline Comparison

| Aspect | Current (Dual CSI + Jetson) | With OAK-D Lite |
|---|---|---|
| Stereo rectification | CPU `cv::remap` or CUDA, ~3–5 ms | On-device warp engine, 0 ms host |
| Stereo matching | CPU SGBM 20–30 ms (dominant bottleneck) | On-device hardware SGBM, 0 ms host |
| Disparity subpixel | Not available | 5 fractional bits (32 sub-steps) |
| Depth confidence | Not used | Available stream, improves warp quality |
| L-R consistency | CPU, costs time | On-device, free |
| Temporal stability | IIR on CPU | Stable hardware output, less noise |
| Global shutter stereo | Rolling shutter (typical CSI) | OV7251 global shutter — no skew |
| Warp model | Fractional interpolation between cameras | Physical virtual-window ray projection |
| Parallax scale | Hand-tuned (disp_amplify=15, FT_SENSITIVITY=0.8) | Derived from geometry — physically correct |
| Vertical parallax | None | Supported by new model |
| Hole fill quality | Horizontal scan — streaks | JFA — correct 2D nearest valid pixel |
| Colour texture quality | 1366×768 CSI | IMX214 up to 4K |
| Calibration workflow | JSON file + manual checkerboard rig | EEPROM auto-applied; display pose once |
| Wiring | 2× CSI ribbon + 1× USB | 1× USB-C |
| Achieved FPS (estimated) | ~5 FPS (SGBM path), ~15 FPS (CUDA BM) | **25–30 FPS** |

---

## Part 11: Remaining Challenges

1. **Disparity upsampling**: OAK-D stereo pair is 480P. Color is up to 1080P. If DIBR warp
   runs at color resolution, joint bilateral upsample of the depth map is needed.
   Some cost on Jetson GPU, but far less than SGBM.

2. **Display-to-camera calibration**: A one-time procedure to establish the physical pose
   of the display relative to the camera. Can be done with an ArUco marker on the display
   or a checkerboard fixture. Results are stored in a small JSON and loaded at startup.

3. **Large disocclusions**: When the head moves significantly, occluded background regions
   become visible with no colour data. JFA finds nearest valid pixel, but for large
   disocclusions (>32 px) temporal reprojection or background inpainting is needed.
   This is a fundamental limitation of single-viewpoint DIBR regardless of depth source.

4. **Face size → depth assumption**: Using a fixed adult face width (0.16 m) to estimate
   viewer depth from bbox size introduces ~10–15% error across real face sizes. For most
   viewers this produces visually acceptable parallax. A more robust alternative: use the
   independent head tracker's known camera calibration and bounding box tracking to maintain
   a running depth estimate via Kalman filter.

5. **Textureless regions**: Passive stereo still fails on white walls, uniform surfaces, etc.
   OAK-D's confidence map helps by flagging these pixels. Active stereo (structured light)
   would be needed for reliable depth in textureless scenes — beyond scope here.

---

## Appendix: Key Source Code Locations

| File | Key content |
|---|---|
| `src/view_synthesis.cpp:218` | `DEPTH_SPLAT_CS` — forward warp to z-buffer |
| `src/view_synthesis.cpp:254` | `COLOR_SPLAT_CS` — defined but unused in render loop |
| `src/view_synthesis.cpp:280` | `HOLE_FILL_CS` — horizontal scan hole fill |
| `src/view_synthesis.cpp:347` | `BACKWARD_COLOR_CS` — backward bilinear color sample |
| `src/view_synthesis.cpp:380` | `CLEAR_CS` — GPU z-buffer clear |
| `src/view_synthesis.cpp:1265` | `u_shift` resolution from face tracker |
| `src/view_synthesis.cpp:1271` | Pass 1: depth splat dispatch |
| `src/view_synthesis.cpp:1284` | Pass 2: backward color warp dispatch |
| `src/view_synthesis.cpp:1297` | Pass 3: hole fill dispatch |
| `src/face_tracker.h:13` | Tuning constants (`FT_SENSITIVITY`, `FT_SMOOTH`, etc.) |
| `src/face_tracker.cpp` | Haar + Lucas-Kanade tracking loop, `u_shift` output |
| `src/calibration.cpp:201` | `cv::stereoCalibrate` call |
| `src/calibration.cpp:228` | `cv::stereoRectify` call, Q matrix generation |
