#pragma once

/*
 * camera_capture.h — Jetson Nano / V4L2 camera backend
 *
 * Drop-in replacement for the libcamera camera layer.  Each CameraCapture
 * runs a background thread that continuously reads frames from the device
 * and stores the latest one under a mutex.  Consumers grab the frame the
 * same way they did with libcamera:
 *
 *   std::lock_guard<std::mutex> lock(cap.frame_mutex);
 *   if (cap.new_frame) { cap.frame.copyTo(dst); cap.new_frame = false; }
 *
 * Backend selection — define ONE of the following before including this
 * header, or set it via CMake add_compile_definitions():
 *
 *   CAMERA_BACKEND_CSI    nvarguscamerasrc GStreamer pipeline
 *                         (Jetson CSI cameras, e.g. IMX219)
 *
 *   (default)             V4L2 device index  (USB cameras)
 */

#include <atomic>
#include <chrono>
#include <cstdio>
#include <mutex>
#include <string>
#include <thread>

#include <opencv2/core.hpp>
#include <opencv2/videoio.hpp>

/* ========================================================================
 * CameraCapture struct
 * ======================================================================== */

struct CameraCapture {
    cv::VideoCapture  cap;
    std::thread       thread;
    std::atomic<bool> running{false};

    int width{0}, height{0};

    std::mutex frame_mutex;
    cv::Mat    frame;
    bool       new_frame{false};
};

/* ========================================================================
 * Internal helpers
 * ======================================================================== */

#ifdef CAMERA_BACKEND_CSI
/*
 * Build a GStreamer pipeline string for a Jetson Nano CSI camera.
 *
 * nvarguscamerasrc captures from the NVIDIA ISP into NVMM memory;
 * nvvidconv converts to system-memory BGRx; videoconvert strips the
 * padding byte to give OpenCV a plain BGR frame.
 *
 * flip-method values: 0=none 1=ccw90 2=180 3=cw90 4=horiz 5=ul-diag
 *                     6=vert 7=ur-diag
 */
/*
 * out_w / out_h (optional): hardware-scale to this resolution before the
 * system-memory conversion.  On L4T the scale must stay in NVMM space, so
 * we insert a second nvvidconv element:
 *
 *   nvarguscamerasrc → NV12(NVMM) → nvvidconv(flip+scale) → NV12(NVMM)
 *                    → nvvidconv(format) → BGRx(system) → videoconvert → BGR
 *
 * Reduces CPU videoconvert cost proportionally (4× at proc_scale=0.5).
 * Pass 0 to skip scaling (original pipeline, always safe).
 */
static std::string csi_pipeline(int sensor_id, int width, int height,
                                 int out_w = 0, int out_h = 0,
                                 int fps = 30, int flip_method = 2)
{
    const bool do_scale = (out_w > 0 && out_h > 0
                           && (out_w != width || out_h != height));

    std::string s =
          "nvarguscamerasrc sensor-id=" + std::to_string(sensor_id)
          + " ! video/x-raw(memory:NVMM)"
              ", width="     + std::to_string(width)
          + ", height="    + std::to_string(height)
          + ", framerate=" + std::to_string(fps) + "/1"
          + ", format=NV12"
            " ! nvvidconv flip-method=0";

    if (do_scale) {
        /* Scale inside NVMM, then a second nvvidconv converts to system BGRx */
        s += " ! video/x-raw(memory:NVMM)"
             ", width="  + std::to_string(out_w)
           + ", height=" + std::to_string(out_h)
           + ", format=NV12"
             " ! nvvidconv";
    }

    /* videoflip rotates in system memory after the NVMM→BGRx conversion;
     * more reliable than nvvidconv flip-method which can be silently ignored
     * depending on GStreamer/L4T version. */
    s += " ! video/x-raw, format=BGRx"
         " ! videoflip method=" + std::to_string(flip_method)
         + " ! videoconvert"
         + " ! appsink drop=true max-buffers=1";
    return s;
}
#endif  /* CAMERA_BACKEND_CSI */

#ifndef CAMERA_BACKEND_CSI
/*
 * Build a GStreamer pipeline string for a V4L2 (USB) camera.
 *
 * Assumes the camera supports MJPG at the requested resolution, which
 * is required for 30 fps at 1080p on most USB cameras.
 *
 * flip-method values match GStreamer videoflip:
 *   0=none  1=cw90  2=180  3=ccw90  4=horiz  5=ul-diag  6=vert  7=ur-diag
 */
static std::string v4l2_pipeline(int device_idx, int width, int height,
                                  int flip_method = 2)
{
    return std::string("v4l2src device=/dev/video") + std::to_string(device_idx)
           + " ! image/jpeg"
           + ", width="  + std::to_string(width)
           + ", height=" + std::to_string(height)
           + " ! jpegdec"
           + " ! videoflip method=" + std::to_string(flip_method)
           + " ! videoconvert"
           + " ! video/x-raw, format=BGR"
           + " ! appsink drop=true max-buffers=1";
}
#endif  /* !CAMERA_BACKEND_CSI */

static void capture_thread_fn(CameraCapture *cap)
{
    while (cap->running.load()) {
        cv::Mat tmp;
        cap->cap >> tmp;
        if (tmp.empty()) {
            /* Yield briefly so a broken/slow pipeline can't spin a core to 100% */
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
            continue;
        }

        /* Swap under lock — O(1) Mat header swap, no pixel copy */
        std::lock_guard<std::mutex> lock(cap->frame_mutex);
        cv::swap(cap->frame, tmp);
        cap->new_frame = true;
    }
}

/* ========================================================================
 * Public API — matches the original libcamera interface
 * ======================================================================== */

/*
 * Open camera at camera_idx and start the capture thread.
 *
 * V4L2:  camera_idx is the /dev/videoN index.
 * CSI:   camera_idx is the MIPI sensor-id (0 or 1 on Jetson Nano).
 *
 * Returns true on success.
 */
static bool init_camera(CameraCapture *cap,
                        int camera_idx, int width, int height,
                        int out_w = 0, int out_h = 0)
{
#ifdef CAMERA_BACKEND_CSI
    const std::string pipeline = csi_pipeline(camera_idx, width, height, out_w, out_h);
    printf("[Camera %d] Opening pipeline:\n  %s\n", camera_idx, pipeline.c_str());
    cap->cap.open(pipeline, cv::CAP_GSTREAMER);
    if (!cap->cap.isOpened()) {
        printf("[Camera %d] FAILED to open CSI pipeline\n", camera_idx);
        fprintf(stderr, "[Camera %d] FAILED to open CSI pipeline:\n  %s\n",
                camera_idx, pipeline.c_str());
        return false;
    }
#else
    const std::string pipeline = v4l2_pipeline(camera_idx, width, height);
    printf("[Camera %d] Opening pipeline:\n  %s\n", camera_idx, pipeline.c_str());
    cap->cap.open(pipeline, cv::CAP_GSTREAMER);
    if (!cap->cap.isOpened()) {
        printf("[Camera %d] FAILED to open V4L2 GStreamer pipeline\n", camera_idx);
        fprintf(stderr, "[Camera %d] FAILED to open V4L2 pipeline:\n  %s\n",
                camera_idx, pipeline.c_str());
        return false;
    }
#endif

    /* Read back the actual negotiated resolution */
    cap->width  = (int)cap->cap.get(cv::CAP_PROP_FRAME_WIDTH);
    cap->height = (int)cap->cap.get(cv::CAP_PROP_FRAME_HEIGHT);
    cap->new_frame = false;
    cap->running   = true;
    cap->thread    = std::thread(capture_thread_fn, cap);

    printf("[Camera %d] %dx%d  backend: %s\n",
           camera_idx, cap->width, cap->height,
#ifdef CAMERA_BACKEND_CSI
           "CSI/GStreamer (nvarguscamerasrc)"
#else
           "V4L2/GStreamer (videoflip)"
#endif
           );
    return true;
}

static void cleanup_camera(CameraCapture *cap)
{
    cap->running = false;
    cap->cap.release();          // EOS → appsink returns NULL → cap >> tmp returns empty
    if (cap->thread.joinable())
        cap->thread.join();      // thread sees empty frame, checks running==false, exits
}
