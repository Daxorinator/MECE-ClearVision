# Display-to-Camera Calibration Research — MECE-ClearVision

*Documented from design session, 2026-03-27*

---

## Context

The virtual window DIBR model requires knowing the exact 6-DOF pose of the display panel
in the OAK-D camera's coordinate frame — a rotation matrix **R** and translation vector
**t** such that any point in display-local coordinates can be expressed in camera space:

```
P_camera = R · P_display + t
```

This pose is used to construct `u_display_matrix` each frame, which projects world-space
points onto the display plane as seen from the viewer's head position.

---

## Why the Naive Approach Doesn't Work

The obvious calibration approach — display a ChArUco pattern on the screen, point the
OAK-D at it, run `solvePnP` — **is not physically possible** for this installation:

- The OAK-D is mounted in the **wing mirror**, pointing outward away from the car
- The display is mounted in the **A-pillar**, facing the interior
- The camera and display point in fundamentally opposite directions and are separated
  by the car body — there is no configuration in which one can directly observe the other
- The camera cannot be "spun in place" to face inward; it is a fixed part of the vehicle

This rules out any direct line-of-sight calibration between the two devices.

---

## Indirect Calibration via Shared Reference

The standard approach when two sensors have no shared FOV is to chain transforms through
an intermediate reference target visible to both — or visible to each in separate steps
with a known relationship between them.

### Concept

```
[Outside]                         [Inside]

OAK-D ──► sees ArUco marker       marker ◄── secondary camera
           (e.g. on windscreen             (phone, or temporary webcam)
            or dashboard)                  also sees display in frame
```

1. OAK-D localises the marker → gives `T_marker_in_cam`
2. Secondary camera photographs the marker **and** the display simultaneously
   → gives `T_display_in_marker` (via geometry / `solvePnP` from the secondary view)
3. Chain: `T_display_in_cam = T_marker_in_cam · T_display_in_marker`

### Problems with this approach for this installation

- Requires a marker placement with line-of-sight to both OAK-D and a secondary camera
  simultaneously — difficult given the car body separating the two
- Errors in `T_marker_in_cam` and `T_display_in_marker` compound; even small angular
  errors (1–2°) produce centimetre-scale positional errors at display distance
- The secondary camera itself requires calibration (intrinsics), adding another source
  of error
- Practical execution is awkward: requires two calibration captures, careful marker
  placement, and a consistent mounting fixture for the secondary camera
- The result is only as good as the weakest link in the chain

### Assessment

Indirect calibration is theoretically correct but practically fragile for this geometry.
The accumulated error may not be meaningfully better than a manually-tuned estimate,
especially given the other uncertainties present in the system (see below).

---

## Current Decision: Manually Tuned Transform

Given the calibration difficulties and the uncertainty budget of the overall system,
a manually tuned display-to-camera transform is sufficient for initial development.

### Justification

The dominant sources of uncertainty in the pipeline are:

| Source | Approximate error |
|---|---|
| Head depth from iris diameter | ~5–8% (iris size variation across population) |
| Head tracker lateral position | ~5 mm at 0.5 m (pixel-level noise in tracker) |
| OAK-D disparity at 4 m | ~2% depth error (from arXiv:2501.07421 benchmark) |
| Display pose (manual tune) | ~10–20 mm translation, ~2–3° rotation |

The display pose error is not the dominant term. A manually tuned transform does not
meaningfully degrade the output quality compared to a precisely calibrated one, at
least at this stage of development.

### What to manually tune

The display pose has 6 degrees of freedom:

```
Translation (t):  tx, ty, tz   — where the display centre is in camera space (metres)
Rotation (R):     rx, ry, rz   — display orientation (Rodrigues or Euler angles)
```

In practice, the most important parameters are:

- **tz** (depth along camera Z axis): roughly the physical distance from the OAK-D to
  the display through the car body — measurable with a tape measure to ~1 cm accuracy
- **tx, ty** (lateral/vertical offset): the physical offset of the display centre from
  the camera optical axis — also measurable directly
- **ry** (yaw): how much the display is rotated left/right relative to the camera forward
  direction — typically close to 180° (they face opposite directions)
- **rx, rz** (pitch, roll): deviation from vertical — usually small for a fixed A-pillar
  display

### Tuning procedure

1. Hard-code physically measured translation (tape measure from camera to display centre
   through the car body)
2. Set rotation to nominal (display faces ~180° from camera, with small pitch/roll
   corrections for the A-pillar angle)
3. Run the pipeline with a static scene and no head movement
4. Adjust until the virtual window output is geometrically consistent with the scene
   (parallel lines in the scene appear parallel in the output, horizon is level, etc.)
5. Save to the display pose JSON (same format as the rest of the calibration data)

---

## Future Work: Proper Calibration

If sub-centimetre accuracy becomes necessary (e.g. for precise geometric alignment or
stereo matching with an additional inside camera), the most promising proper approaches
are:

### Option A: Physical measurement + photogrammetry

Measure the 3D positions of several known points (e.g. display corners) in both the
camera's coordinate frame (by placing ArUco markers on the car exterior near the camera
and triangulating) and in the real world. Use a photogrammetry tool (e.g. Meshroom,
COLMAP) to produce a consistent 3D model of the installation.

### Option B: Laser distance measurement

Use a laser rangefinder to directly measure the vector from the camera to each corner of
the display. Combined with an inclinometer/level for rotation, this gives tx/ty/tz to
~1 mm and rx/rz to ~0.5°. More practical than photogrammetry for a fixed installation.

### Option C: IMU-assisted calibration

The OAK-D Lite includes a Bosch BMI270 IMU. Combined with a separate IMU on or near the
display, the relative orientation can be established by comparing gravity vectors. Does
not help with translation, but nails the rotation component cheaply.

### Option D: Revisit indirect calibration with a proper fixture

Build a calibration fixture: a rigid bar with ArUco markers at each end, one end
positioned at the camera FOV, the other visible from inside the car looking at the
display. The rigid geometry of the bar makes the transform chain deterministic.
One-time effort, reusable if the installation is ever changed.

---

## Display Pose JSON Format

```json
{
  "display_width_m":  0.344,
  "display_height_m": 0.194,
  "R": [
    [r00, r01, r02],
    [r10, r11, r12],
    [r20, r21, r22]
  ],
  "t": [tx, ty, tz]
}
```

Loaded at startup alongside OAK-D intrinsics (from EEPROM via DepthAI API).
The combined display matrix is rebuilt each frame using `R`, `t`, and the current
head position from the face tracker.
