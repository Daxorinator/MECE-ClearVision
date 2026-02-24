# MECE-ClearVision — Claude Session Notes

## Project
Stereo DIBR (Depth-Image Based Rendering) view synthesis on Jetson Nano.
Two CSI cameras → stereo rectify → disparity map → OpenGL ES compute shaders → novel viewpoint.

Target hardware: **Jetson Nano**, Maxwell GPU, L4T R32, OpenGL ES 3.1, CUDA.
Build: `cd build && cmake -DCAMERA_BACKEND_CSI=ON .. && make -j4 view_synthesis`
Run:  `./view_synthesis <calib.json>`

## Current state (commit b626b77)

### Optimisations applied to `src/view_synthesis.cpp`
1. `cv::swap` instead of `clone()` for frame grab (eliminates 12 MB memcpy/frame)
2. Rectification maps changed to `CV_32FC1` (required by `cv::cuda::remap`)
3. CUDA headers added (`cudaarithm`, `cudaimgproc`, `cudawarping`, `cudastereo`)
4. CUDA member variables added to `SynthWindow`: `gpu_map_*`, `gpu_frame_*`,
   `gpu_rect_*`, `gpu_proc_*`, `gpu_gray_*`, `gpu_disp_l`, `cuda_bm`
5. Cached GL uniform locations (`uloc_*`) — set once in `initializeGL`, used in `paintGL`
6. Constructor: detects CUDA, uploads rectification maps to GPU
7. `rebuildStereo()`: creates `cv::cuda::StereoBM` when `use_cuda && !use_sgbm`
8. `paintGL()`: full CUDA/CPU dual-path preprocessing (see below)
9. `glClearBufferSubData` was tried for SSBO clear but `gl32.h` is absent from
   L4T dev package — reverted to `glBufferSubData` + `clear_zeros` vector.

### CUDA preprocessing path in `paintGL()`
```
upload frame_l, frame_r → GPU
GPU remap (rectify) both frames
GPU resize to proc_w × proc_h  (or shallow-copy if proc_scale == 1.0)
GPU BGR→Gray both frames
if BM mode (cuda_bm valid):
    GPU StereoBM → download disp_raw_l
    if WLS: download grays → CPU right_stereo
else (SGBM mode):
    download both grays → CPU stereo->compute + right_stereo->compute
WLS filter on CPU (if enabled)
convertTo float + threshold (CPU)
download gpu_proc_l → CPU BGR→RGBA
glTexSubImage2D uploads
```

## KNOWN PERFORMANCE PROBLEM — needs debugging on Jetson

**Symptom**: 1 FPS, 95% CPU after these changes.

**Root cause hypothesis**: In **SGBM mode** (the default), the CUDA path adds
upload + download round-trips on top of the same slow CPU SGBM (20-30 ms).
Net result is *more* work than the original CPU-only path.

The CUDA preprocessing only pays off in **BM mode** (press `S`), where
`cuda_bm->compute()` replaces the 10-15 ms CPU BM entirely.

**Immediate things to check on Jetson:**

1. Check stdout — does it print `"CUDA available — GPU preprocessing enabled"`?
   If not, `use_cuda=false` and the CPU path runs but rectification maps are now
   CV_32FC1 (vs original CV_16SC2). CV_32FC1 maps work fine with CPU `cv::remap`
   so this is not the cause.

2. Press **S** to switch to BM mode. Does FPS improve significantly?
   If yes: SGBM+CUDA overhead is the problem (see fix below).
   If no:  Something else is wrong — add timing prints to isolate.

3. Press **W** to toggle WLS off. Does FPS improve?

4. Press **2** or **3** for 0.5x/0.25x scale (should already be 0.5x default).

## Fix applied (commit after b626b77)

In SGBM mode the GPU preprocessing buys nothing because we immediately
download the results for CPU SGBM. The fix is to skip CUDA remap/resize/cvtColor
in SGBM mode and only run those on GPU when `cuda_bm` is actually going to be used:

Changed `if (use_cuda)` → `if (use_cuda && cuda_bm)` in `paintGL()`.
SGBM mode now always takes the CPU path (as originally). BM mode takes
the full CUDA path (remap + resize + cvtColor + StereoBM on GPU).
GPU remap/resize/cvtColor for SGBM is not worth the upload+download overhead.

## Reference
- `src/depth_pipeline.cpp` — working CUDA pattern (remap, resize, cvtColor, StereoBM)
- The CUDA BM is created with: `cv::cuda::createStereoBM(num_disparities, block_size)`
- OpenGL ES 3.1 only (`<GLES3/gl31.h>`), `gl32.h` absent from L4T headers
- Camera backend: CSI (`nvarguscamerasrc`) on Jetson, V4L2 available as fallback
- Face tracker runs on a third USB camera (index 2), CPU DNN, background thread
