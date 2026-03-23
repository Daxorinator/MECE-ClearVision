#include "face_tracker.h"

#include <algorithm>
#include <cmath>
#include <cstdio>

#include <opencv2/imgproc.hpp>
#include <opencv2/video/tracking.hpp>   // calcOpticalFlowPyrLK, goodFeaturesToTrack
#include <opencv2/videoio.hpp>

static inline float ft_clamp(float v, float lo, float hi)
{
    return std::max(lo, std::min(hi, v));
}

/* ========================================================================
 * Public API
 * ======================================================================== */

bool FaceTracker::start(int camera_index, const std::string &cascade_path)
{
    camera_index_ = camera_index;
    cascade_path_ = cascade_path;
    running_      = true;
    worker_       = std::thread(&FaceTracker::threadLoop, this);
    return true;
}

void FaceTracker::stop()
{
    running_ = false;
    if (worker_.joinable())
        worker_.join();
}

void FaceTracker::calibrate()
{
    std::lock_guard<std::mutex> lock(mtx_);
    ref_x_ = ema_x_;
    printf("[FaceTracker] Calibrated: ref_x = %.3f\n", ref_x_);
}

float FaceTracker::shift() const
{
    std::lock_guard<std::mutex> lock(mtx_);
    return u_shift_val_;
}

bool FaceTracker::isActive() const { return running_.load(); }

/* ========================================================================
 * Background thread
 * ======================================================================== */

void FaceTracker::threadLoop()
{
    if (!cascade_.load(cascade_path_)) {
        fprintf(stderr, "[FaceTracker] Failed to load cascade: %s\n",
                cascade_path_.c_str());
        running_ = false;
        return;
    }

    /*
     * Request 320×240 GRAY8 from GStreamer — smallest MJPEG payload to decode,
     * no further colour conversion needed.  Fall back to 640×480 if the camera
     * does not support 320×240.
     */
    const char *pipeline_fmt =
        "v4l2src device=/dev/video%d"
        " ! image/jpeg, width=%d, height=%d"
        " ! jpegdec ! videoconvert"
        " ! video/x-raw, format=GRAY8"
        " ! appsink drop=true max-buffers=1";

    cv::VideoCapture cap;
    int fw = 0, fh = 0;
    for (auto [w, h] : std::initializer_list<std::pair<int,int>>{{320,240},{640,480}}) {
        char buf[512];
        std::snprintf(buf, sizeof(buf), pipeline_fmt, camera_index_, w, h);
        cap.open(buf, cv::CAP_GSTREAMER);
        if (cap.isOpened()) { fw = w; fh = h; break; }
    }
    if (!cap.isOpened()) {
        fprintf(stderr, "[FaceTracker] Failed to open camera %d\n", camera_index_);
        running_ = false;
        return;
    }

    cv::Mat cur_gray;
    cap >> cur_gray;
    if (cur_gray.empty()) {
        fprintf(stderr, "[FaceTracker] Empty first frame from camera %d\n", camera_index_);
        running_ = false;
        return;
    }
    fw = cur_gray.cols;
    fh = cur_gray.rows;
    const float half_w = fw * 0.5f;

    /*
     * Haar detection runs on a 320×240 image.
     * If the camera already delivers 320×240 no resize is needed.
     */
    const int   det_w       = 320;
    const int   det_h       = 240;
    const bool  need_resize = (fw != det_w || fh != det_h);
    const float scale_x     = static_cast<float>(det_w) / fw;
    const float scale_y     = static_cast<float>(det_h) / fh;
    const float inv_scale_x = 1.0f / scale_x;
    const float inv_scale_y = 1.0f / scale_y;

    printf("[FaceTracker] Camera %d: %dx%d (GRAY), det %dx%d, cascade: %s\n",
           camera_index_, fw, fh, det_w, det_h, cascade_path_.c_str());

    cv::Mat prev_gray, small_gray;
    std::vector<cv::Point2f> tracked_pts;
    cv::Rect  last_bbox;          // in full-resolution pixel coords
    bool      tracker_valid = false;
    bool      force_detect  = true;
    int       frame_count   = 0;

    while (running_.load()) {
        cap >> cur_gray;
        if (cur_gray.empty())
            continue;

        /* Ensure single-channel (camera might return BGR on fallback) */
        if (cur_gray.channels() != 1)
            cv::cvtColor(cur_gray, cur_gray, cv::COLOR_BGR2GRAY);

        const bool do_detect = force_detect || (frame_count % FT_DETECT_EVERY == 0);
        force_detect = false;

        float face_x = -1.f;   // sentinel: no update this frame

        /* ----------------------------------------------------------------
         * DETECTION FRAME — run Haar cascade on downscaled image,
         * optionally restricted to an expanded ROI of the last known bbox.
         * ---------------------------------------------------------------- */
        if (do_detect) {
            if (need_resize)
                cv::resize(cur_gray, small_gray, cv::Size(det_w, det_h),
                           0, 0, cv::INTER_NEAREST);
            else
                small_gray = cur_gray;  // shallow alias — we never write to it

            /* Build detection ROI in small_gray coordinates */
            cv::Mat   det_input;
            cv::Point det_offset(0, 0);

            if (tracker_valid) {
                /* Expand last bbox by 50% each side to give the face room to
                 * move between detection frames. */
                const int mx = static_cast<int>(last_bbox.width  * scale_x * 0.5f);
                const int my = static_cast<int>(last_bbox.height * scale_y * 0.5f);
                cv::Rect roi;
                roi.x      = std::max(0, static_cast<int>(last_bbox.x * scale_x) - mx);
                roi.y      = std::max(0, static_cast<int>(last_bbox.y * scale_y) - my);
                roi.width  = std::min(det_w - roi.x,
                                      static_cast<int>(last_bbox.width  * scale_x) + 2*mx);
                roi.height = std::min(det_h - roi.y,
                                      static_cast<int>(last_bbox.height * scale_y) + 2*my);
                det_input  = small_gray(roi);
                det_offset = roi.tl();
            } else {
                det_input = small_gray;
            }

            std::vector<cv::Rect> faces;
            cascade_.detectMultiScale(det_input, faces,
                1.1, 3, 0, cv::Size(30, 30));

            if (!faces.empty()) {
                /* Largest face wins */
                const cv::Rect &best = *std::max_element(
                    faces.begin(), faces.end(),
                    [](const cv::Rect &a, const cv::Rect &b)
                    { return a.area() < b.area(); });

                /* Translate: det_input-local → small_gray → full-res */
                const int sx = best.x + det_offset.x;
                const int sy = best.y + det_offset.y;
                last_bbox = cv::Rect(
                    static_cast<int>(sx          * inv_scale_x),
                    static_cast<int>(sy          * inv_scale_y),
                    static_cast<int>(best.width  * inv_scale_x),
                    static_cast<int>(best.height * inv_scale_y))
                    & cv::Rect(0, 0, fw, fh);

                /* Re-seed LK with good features inside the fresh bbox */
                cv::goodFeaturesToTrack(cur_gray(last_bbox), tracked_pts,
                    FT_MAX_POINTS, 0.01, 5.0);
                for (auto &pt : tracked_pts) {
                    pt.x += static_cast<float>(last_bbox.x);
                    pt.y += static_cast<float>(last_bbox.y);
                }
                tracker_valid = !tracked_pts.empty();
                face_x = static_cast<float>(last_bbox.x) + last_bbox.width  * 0.5f;
            } else {
                tracker_valid = false;   // hold last u_shift
            }

        /* ----------------------------------------------------------------
         * TRACKING FRAME — propagate points with Lucas-Kanade
         * ---------------------------------------------------------------- */
        } else if (tracker_valid && !prev_gray.empty()) {
            std::vector<cv::Point2f> next_pts;
            std::vector<uchar>       status;
            std::vector<float>       err;

            cv::calcOpticalFlowPyrLK(
                prev_gray, cur_gray, tracked_pts, next_pts,
                status, err,
                cv::Size(15, 15),   // search window
                2);                 // pyramid levels

            std::vector<cv::Point2f> good;
            good.reserve(tracked_pts.size());
            for (size_t i = 0; i < status.size(); i++)
                if (status[i]) good.push_back(next_pts[i]);

            if (static_cast<int>(good.size()) < FT_MIN_POINTS) {
                /* Too many points dropped — force Haar on next frame */
                tracker_valid = false;
                force_detect  = true;
            } else {
                tracked_pts = good;
                float sum_x = 0.f;
                for (const auto &pt : good) sum_x += pt.x;
                face_x = sum_x / static_cast<float>(good.size());
            }
        }

        /* ----------------------------------------------------------------
         * Update EMA and u_shift whenever we have a valid face position.
         * ---------------------------------------------------------------- */
        if (face_x >= 0.f) {
            const float rel_x = (face_x - half_w) / half_w;   // [-1, 1]
            std::lock_guard<std::mutex> lock(mtx_);
            if (first_detection_) {
                ema_x_           = rel_x;
                first_detection_ = false;
            } else {
                ema_x_ = FT_SMOOTH * rel_x + (1.f - FT_SMOOTH) * ema_x_;
            }
            u_shift_val_ = ft_clamp(
                0.5f + (ema_x_ - ref_x_) * FT_SENSITIVITY,
                FT_SHIFT_MIN, FT_SHIFT_MAX);
        }

        cv::swap(cur_gray, prev_gray);   // O(1) — no pixel copy
        ++frame_count;
    }

    cap.release();
    printf("[FaceTracker] Thread exited\n");
}
