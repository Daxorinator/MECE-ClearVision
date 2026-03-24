# Stereo Baseline Analysis — A-Pillar Blind Spot Digital Window

## Application

Digital window mounted inside a car to provide vision through the A-pillar blind spot.
Target depth range: **3–5 m** behind the blind spot.

---

## Core Formula

```
Z = (f × B) / d
```

- `Z` = depth (metres)
- `f` = focal length (pixels)
- `B` = baseline (metres)
- `d` = disparity (pixels)

---

## Key Trade-offs

| Wider Baseline | Narrower Baseline |
|---|---|
| Better depth resolution at range | Smaller occlusion regions |
| Larger occlusion regions (bad for DIBR) | Easier stereo matching |
| More texture mismatch between views | Disparity collapses at range |
| Higher minimum working distance | Noise dominates at range |

For DIBR specifically: the synthesised viewpoint shift is proportional to baseline.
Larger baseline → more disoccluded regions to inpaint → more visible artefacts.

---

## Constraints for This Application

Two constraints interact:

1. **Depth accuracy** — need sufficient disparity at 3–5 m for reliable reconstruction
2. **Digital window geometry** — synthesised viewpoint shift must match the A-pillar
   obstruction width (~80–120 mm on a typical car), otherwise parallax looks wrong
   to the driver

The window geometry constraint is dominant, and it conveniently aligns with the depth constraint.

---

## Depth Accuracy at 100 mm Baseline

IMX219 CSI cameras at `proc_scale 0.5` (~640 px wide), approximate `f ≈ 800 px`:

| Distance | Disparity |
|----------|-----------|
| 1.0 m    | ~80 px    |
| 3.0 m    | ~27 px    |
| 5.0 m    | ~16 px    |

16–27 px in the 3–5 m target zone is well above the noise floor for StereoBM and SGBM.

---

## Recommendation: 80–100 mm (current baseline of 100 mm is fine)

- Matches physical A-pillar obstruction width → correct parallax for a driver seated
  ~600–700 mm from the pillar
- 16–27 px disparity in the 3–5 m zone — reliable for both BM and SGBM
- Occlusion fills at window edges remain minimal

**Avoid exceeding ~120 mm** — the synthesised viewpoint shift will overshoot the
A-pillar width and objects near the pillar will appear to jump laterally, breaking
the window illusion.

---

## Mounting Notes

- Mount cameras so their **midpoint aligns** with the driver's natural sightline to
  the obstruction
- Baseline axis must be **horizontal** — vertical offset produces vertical disparity,
  which StereoBM handles poorly and looks unnatural as a window

---

## StereoBM Tuning

With 100 mm baseline at processing resolution (`f ≈ 800 px`):

```cpp
num_disparities = 96   // covers ~1 m+; near-field pedestrians at ~1 m give ~80 px
block_size      = 11–15  // automotive textures (road, vehicles) are coarser
```

Ensure `num_disparities ≥ 96` to capture near-field objects such as pedestrians
stepping off a kerb at 1–1.5 m.
