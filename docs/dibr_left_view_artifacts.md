# DIBR Left-View Forward Warping: Artifacts and Consequences

**Pipeline context:** `src/view_synthesis.cpp` performs DIBR using only the left
camera's colour frame and its StereoSGBM/StereoBM disparity map. The warp formula is:

```
dst_x = round(src_x + disparity * u_shift)
```

`u_shift` is in [0, 1] and is either fixed at 0.5 or driven dynamically from
`FaceTracker::shift()` based on the viewer's horizontal head position. Two GPU
compute passes (depth splat with `atomicMax`, then colour splat) synthesise the
virtual frame, followed by a horizontal hole-fill pass that searches up to 64 pixels
either side and picks the neighbour with the higher depth value.

---

## 1. Disocclusion Holes

### What they are

A disocclusion is a region of the scene that is invisible from the left camera
because a foreground object occludes it, but which becomes visible as the virtual
camera shifts right. The pipeline has no source pixels for these regions — the
left-view image never recorded what lies behind the foreground object from the
right vantage point.

### Spatial location

Holes appear on the **left edges of foreground objects** in the synthesised frame.
As the virtual camera moves rightward, background that was hidden by the right
silhouette of each foreground object is progressively revealed. The hole width at
any object boundary is:

```
gap ≈ (d_foreground − d_background) × u_shift
```

For a deep foreground object in front of a near-zero-disparity background at
`u_shift = 1.0`, the hole can be as wide as the full disparity range (`num_disparities`
pixels).

### Dependence on u_shift

The relationship is linear:

- `u_shift = 0` — no warp, no holes, synthesised image identical to left camera
- `u_shift = 0.5` — holes at half their maximum width
- `u_shift = 1.0` — holes at full width, equal to the full disparity gap at each boundary

When `u_shift` is driven by head pose, every hole in the frame changes width on
every frame. This is the primary source of temporal instability (see Section 8).

### Why single-reference maximises the problem

With only the left camera available, every disocclusion is unconditionally a hole
with no valid source pixel — it must be invented by the hole-fill algorithm. In a
bidirectional system the right camera has already seen the background that becomes
revealed; those pixels can be warped leftward to fill the hole. Fehn (2004) and
Zinger et al. (2010) both identify single-reference DIBR as producing systematically
larger and more frequent disocclusions than two-reference systems.

---

## 2. Forward Warping Cracks

### Mechanism

Because disparity values vary continuously but destination positions are quantised
to integers, consecutive source pixels can skip a destination column entirely,
leaving a one-pixel crack:

```
round((x+1) + d(x+1)·u_shift) − round(x + d(x)·u_shift) ≥ 2
```

This occurs when `Δd · u_shift > 1.0`, i.e. when the disparity changes by more
than `1/u_shift` per source pixel. On smooth fronto-parallel surfaces the gradient
is zero and cracks don't occur. On oblique surfaces (e.g. a table receding from
camera) the gradient can be large and cracks appear systematically.

The complementary problem is **collision**: two source pixels mapping to the same
destination column. The `atomicMax` SSBO resolves this by keeping the closer pixel
(higher disparity), which is correct for occlusion ordering but discards the losing
pixel's colour entirely.

### Current pipeline behaviour

Both compute shaders use `round()` independently. Pixels with `disp < 0.5` are
skipped. Cracks from oblique surfaces appear as single-pixel black columns in
`tex_output` and are generally handled by the 64-pixel hole-fill pass, except where
a crack coincides with a depth discontinuity and the fill direction logic selects
the wrong side (see Section 7).

---

## 3. Depth Ordering and Ghost Artifacts at Disparity Discontinuities

### Z-conflict at boundaries

At the boundary between foreground (high disparity) and background (low disparity),
both regions may warp to nearby destination columns. The `atomicMax` strategy
correctly resolves this when disparity values are accurate — the closer pixel wins.
In practice, StereoSGBM produces unreliable disparities at object edges because
the block-matching window straddles the boundary, producing intermediate "bleeding"
values that sort incorrectly.

### Ghost / halo artifacts

A boundary pixel carrying a slightly incorrect disparity value warps to a position
that is neither the correct foreground location nor the correct background location.
This "leaked" pixel appears as a halo or ghost stripe at the silhouette of the
virtual object:

> *"Ghosts mean that pixels at the boundary of foreground are incorrectly projected
> into the virtual view as background pixels."*
> — IEEE Watershed-based Depth Map Misalignment Correction, 2014

### Foreground dilation issue

Stereo depth maps tend to have slightly undersized foreground silhouettes (background
pixels bleed into the matching window at boundaries). When warped, a strip of
background-coloured pixels along the true foreground boundary carries foreground
magnitude disparity, placing background texture at the foreground position in the
virtual view — the reverse of ghosting.

---

## 4. Asymmetric Coverage

### Structural asymmetry

The warp formula shifts all pixels to the right for positive disparity. This means:

- The **left edge** of the synthesised frame grows an out-of-frame strip as `u_shift`
  increases — source pixels from the far left have been pushed right, and nothing
  fills from the left.
- Disocclusions always appear on the **left edge of foreground objects** (rightward
  shift reveals the background to the right of each foreground silhouette).
- The rightmost source columns warp off the right edge of the output (clipped by
  the `dst_x >= u_output_size.x` guard).

### Permanent left-view bias

Every visible surface in the synthesised view was seen by the left camera.
No surface visible only to the right camera (occluded from the left) can ever
appear correctly, regardless of `u_shift`. Even at `u_shift = 1.0` the synthesised
image is topologically the left-camera view shifted, not a genuine right-camera view.
This bias cannot be eliminated without incorporating the right camera image.

### Extrapolation beyond [0, 1]

Synthesising strictly between the two physical cameras (`0 < u_shift < 1`) is
always lower-artifact than extrapolating beyond either. If the face tracker drives
`u_shift` outside [0, 1], the warp extrapolates past the physical camera geometry,
producing a visually broken image. The `clamp` in `FaceTracker::shift()` prevents
this, but it should also be enforced at the point of use in `paintGL`.

---

## 5. Background Stretching Near Foreground Boundaries

When a foreground boundary exists at column `x`, the background pixels just to its
right barely move (low disparity), while the foreground pixels jump far right (high
disparity). After warping, a large gap separates the rightmost foreground pixel from
the nearest background pixel. On the **other side** of the foreground object, the
background texture near the edge is pulled along with the foreground shift and
appears horizontally compressed or smeared.

This is compounded by depth-map imprecision: the transition zone around a foreground
boundary spans several pixels (proportional to `block_size`), and all pixels in that
zone receive intermediate disparity values. They warp to intermediate positions, pile
up in the z-buffer (collision), and produce a visible crush artefact running along
the object boundary.

---

## 6. Why Bidirectional Warping Mitigates These Issues

### Complementary disocclusion coverage

The left and right cameras' disocclusions are geometrically complementary: the left
camera's hidden background regions are exactly what the right camera can see, and
vice versa. Warping both images toward the virtual viewpoint and blending them means
that holes from the left warp are filled by valid pixels from the right warp.

After bidirectional warping only **mutual disocclusions** (regions invisible from
both cameras) remain as true holes. These are a small fraction of the single-reference
disocclusion area. Zinger et al. (2010) demonstrated substantial PSNR improvements
over single-reference DIBR using this technique, and the MDPI 2020 asymmetric
bidirectional paper confirms the finding.

### Crack filling and depth conflict reduction

Cracks from the left warp and cracks from the right warp occur at different positions
(they depend on the sign of the disparity gradient and the warp direction), so their
union covers substantially fewer columns than either warp alone. The depth consistency
check between both warp directions can also suppress ghost pixels: a background-coloured
pixel claiming foreground depth from one direction is contradicted by the correct
foreground pixel arriving from the other direction.

### What is already available in the current pipeline

`disp_raw_r` is already computed and the right camera colour frame is already captured.
The right warp would require:
1. Upload right colour and sign-inverted right disparity to the GPU
2. A second depth-splat + colour-splat pass: `dst_x = src_x − d_right × (1 − u_shift)`
3. A modified blend pass: at each destination pixel, select between left-warped and
   right-warped pixels by comparing depth buffer values

---

## 7. Hole-Fill Algorithm: Failure Modes

### The current algorithm

The `HOLE_FILL_CS` shader searches up to 64 pixels left and right of each empty
pixel, finds the nearest non-empty pixel on each side, and **selects the side with
the higher depth value** (closer to the virtual camera).

### Failure mode 1: Foreground colour bleeding into disocclusions (critical)

A disocclusion hole lies immediately to the **left** of a foreground object. The
nearest non-empty pixel to its **right** is the foreground (high depth). The nearest
non-empty pixel to its **left** is the visible background (low depth). The algorithm
selects the higher depth — the foreground — and fills the disocclusion with foreground
texture. This is the exact opposite of the correct fill: the disocclusion reveals
background, not foreground.

> *"Directly applying certain methods would cause some foreground textures to be
> sampled to fill the holes, resulting in foreground blending."*
> — MDPI Virtual View Synthesis Based on Asymmetric Bidirectional DIBR, 2020

The correct logic for a disocclusion is to fill from the **lower-depth (background)
side**. Holes should be classified by their depth neighbourhood before selecting the
fill direction.

### Failure mode 2: Out-of-frame holes

On the left boundary of the synthesised frame the out-of-frame strip has no left
neighbour within the 64-pixel search radius. Only the right neighbour contributes,
giving the nearest visible pixel to the right of the boundary — approximately correct
in colour but lacking geometric accuracy. For large shifts the 64-pixel radius may
be insufficient, leaving unfilled black columns.

### Failure mode 3: Temporal inconsistency

The fill result for a given hole pixel depends on which neighbour is nearest and
which has higher depth, both of which change as `u_shift` varies and objects move.
A small change in `u_shift` can cause the fill direction to flip, producing a sudden
colour change that appears as a frame-to-frame flash at the hole boundary.

### Higher-quality alternatives from the literature

| Approach | Description |
|---|---|
| Background-biased fill | Classify holes as disocclusions or cracks by depth context; fill disocclusions from the low-depth (background) neighbour only |
| Foreground dilation pre-process | Morphologically dilate foreground depth values before warp to prevent boundary bleeding |
| Depth-guided exemplar inpainting | Patch holes using texture from background regions at matching depth in the warped depth map |
| Layered Depth Image (LDI) | Store multiple depth+colour layers per pixel; warp each layer independently, eliminating disocclusions entirely |
| Temporal inpainting | Use the previous frame's hole-fill as a prior to maintain temporal coherence |

---

## 8. Head-Coupled Parallax: Specific Consequences

### 8.1 Temporal occlusion popping

As `u_shift` changes smoothly, the set of disoccluded pixels changes discontinuously:
new pixels cross the occlusion boundary at specific thresholds. Each threshold
crossing produces an abrupt colour change at a foreground silhouette — the column
transitions from foreground-filled hole (wrong colour, per failure mode 1) to a
newly valid pixel (different colour, and often from a different depth layer). This
is visible as a frame-to-frame flash at object edges during head movement.

> *"Most algorithms process each frame individually while ignoring correlation between
> frames, leading to frequent flickering in virtual video, especially at object edges."*
> — EURASIP Journal on View Synthesis, 2019

The pop is most severe at large disparity discontinuities and during fast head
movements (more threshold crossings per second).

### 8.2 Hole width oscillation

As the viewer rocks left and right, `u_shift` oscillates and all disocclusion widths
oscillate in proportion. Foreground object silhouettes appear to have a "breathing"
right edge as their surrounding hole expands and contracts. This breaks the
perceptual rigidity of the 3D scene.

### 8.3 Disparity estimation latency mismatch

The disparity map is computed from frames captured slightly before the GPU warp
executes. The face tracker runs on a separate thread and its `shift()` value reflects
the viewer's current head position. For fast head movements there is a mismatch
between the stale scene geometry in the disparity map and the current warp offset:
pixels land in positions that are geometrically inconsistent with the displayed
`u_shift`. Holes appear in wrong locations and the parallax cue overstates or
understates perceived depth.

### 8.4 Depth noise amplification

Each disparity noise sample `Δd` produces a position error of `Δd × u_shift` in
the synthesised frame. At `u_shift = 0.5` a 1-unit disparity error produces 0.5 px
of positional error (largely sub-pixel, invisible). At `u_shift = 1.0` the same
error produces 1 full pixel of jitter, visible as shimmer in textureless regions
where StereoSGBM is noisiest.

### 8.5 u_shift outside [0, 1]

If `u_shift > 1.0`, pixels warp past the right camera's physical position. The warp
extrapolates beyond measured geometry, foreground objects overshoot their correct
virtual positions, and the left boundary out-of-frame strip may exceed the 64-pixel
fill radius. The `FaceTracker` class already clamps its output to [0, 1], but this
should also be enforced at the `glUniform1f` call site.

---

## 9. Consolidated Artifact Summary

| Artifact | Root cause | Scales with u_shift | Location |
|---|---|---|---|
| Disocclusion holes | No source data behind foreground objects | Linear — width ∝ u_shift | Left edge of every foreground object |
| Crack holes | Disparity gradient > 1/u_shift skips integer columns | Threshold onset | Oblique surfaces, steep depth edges |
| Ghost / halo stripe | Depth boundary noise warps background pixels to foreground position | Amplified at higher u_shift | Along foreground silhouettes |
| Foreground fill contamination | Hole-fill selects wrong (high-depth) neighbour | Always present for u_shift > 0 | Inside disocclusion holes |
| Background stretching | Foreground pixels jump right, compressing adjacent background | Proportional | Background near foreground boundary |
| Temporal occlusion popping | Discrete disocclusion threshold crossings | Per u_shift change | Foreground silhouettes (head-coupled only) |
| Disparity noise amplification | Position error = Δd × u_shift | Linear | Textureless surfaces |
| Extrapolation breakdown | Geometric extrapolation past physical camera | Only &#124;u_shift&#124; > 1 | Whole image, especially left edge |

---

## 10. Highest-Priority Improvements

1. **Fix the hole-fill direction logic** — Change the fill criterion from
   "choose the higher-depth neighbour" to "choose the lower-depth neighbour" for
   disocclusion holes. A hole bordered on the right by high-depth foreground and on
   the left by lower-depth background should always fill from the left (background)
   side. This single change eliminates failure mode 1.

2. **Clamp u_shift at the shader call site** — Enforce `u_shift = clamp(u_shift, 0, 1)`
   in `paintGL` before both `glUniform1f` calls, as a defence against any future
   face tracker path that bypasses the clamp in `FaceTracker::shift()`.

3. **Bidirectional warping** — The right camera colour frame and right disparity map
   are already available. A second depth-splat + colour-splat pass using
   `dst_x = src_x − d_right × (1 − u_shift)` would convert every single-reference
   disocclusion into a resolved pixel, leaving only mutual disocclusions as true holes.

4. **Depth boundary pre-processing** — Apply a foreground-biased morphological
   dilation to the disparity map before GPU upload to reduce ghost/halo artifacts
   at foreground silhouettes.

---

## References

- Fehn, C. (2004). *Depth-image-based rendering (DIBR), compression and transmission
  for a new approach on 3D-TV.* Proc. SPIE 5291, Stereoscopic Displays and Virtual
  Reality Systems XI.

- Zinger, S., Do, L., & de With, P.H.N. (2010). *Free-viewpoint depth image based
  rendering.* Journal of Visual Communication and Image Representation, 21(5–6),
  533–541.

- *Artifact Handling Based on Depth Image for View Synthesis* (2019). Applied Sciences
  9(9), 1834. MDPI. https://www.mdpi.com/2076-3417/9/9/1834

- *Virtual View Synthesis Based on Asymmetric Bidirectional DIBR for 3D Video and
  Free Viewpoint Video* (2020). Applied Sciences 10(5), 1562. MDPI.
  https://www.mdpi.com/2076-3417/10/5/1562

- *Watershed based depth map misalignment correction and foreground biased dilation
  for DIBR view synthesis* (2014). IEEE ICASSP 2014.
  https://ieeexplore.ieee.org/document/6638649/

- *Fast Hole Filling for View Synthesis in Free Viewpoint Video* (2020). Electronics
  9(6), 906. MDPI. https://www.mdpi.com/2079-9292/9/6/906

- *Disocclusion filling for depth-based view synthesis with adaptive utilization of
  temporal correlations* (2021). Journal of Visual Communication and Image
  Representation. https://www.sciencedirect.com/science/article/abs/pii/S1047320321000912

- *Spatio-temporal consistent DIBR using layered depth image and inpainting* (2016).
  EURASIP Journal on Image and Video Processing.
  https://jivp-eurasipjournals.springeropen.com/articles/10.1186/s13640-016-0109-6

- *View synthesis based on spatio-temporal continuity* (2019). EURASIP Journal on
  Image and Video Processing.
  https://jivp-eurasipjournals.springeropen.com/articles/10.1186/s13640-019-0485-9

- *Depth-guided disocclusion inpainting of synthesized RGB-D images.* HAL preprint.
  https://hal.science/hal-01391065/file/article_revised.pdf

- *An overview of free viewpoint depth-image-based rendering* (2010). APSIPA.
  http://www.apsipa.org/proceedings_2010/pdf/APSIPA197.pdf
