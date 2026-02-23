#include "face_tracker.h"

#include <algorithm>
#include <cmath>
#include <cstdio>

#include <opencv2/calib3d.hpp>
#include <opencv2/videoio.hpp>

static inline float ft_clamp(float v, float lo, float hi)
{
    return std::max(lo, std::min(hi, v));
}

/* ========================================================================
 * Construction / destruction
 * ======================================================================== */

FaceTracker::FaceTracker()
{
    /*
     * Canonical 3-D face model (mm), origin at nose tip.
     * Positive X = viewer's left, positive Y = up, positive Z = towards camera.
     * Z-values encode the physical relief of the face (eyes ~26mm behind nose
     * tip, mouth corners ~24mm behind) — this is what makes solvePnP
     * geometrically unbiased without any manual correction factors.
     *
     * Point order matches YuNet keypoint output columns 4-13:
     *   right_eye, left_eye, nose_tip, right_mouth, left_mouth
     */
    model_points = {
        cv::Point3f(-43.3f,  32.7f, -26.0f),   // right eye centre
        cv::Point3f( 43.3f,  32.7f, -26.0f),   // left eye centre
        cv::Point3f(  0.0f,   0.0f,   0.0f),   // nose tip (origin)
        cv::Point3f(-28.9f, -28.9f, -24.1f),   // right mouth corner
        cv::Point3f( 28.9f, -28.9f, -24.1f),   // left mouth corner
    };
}

FaceTracker::~FaceTracker()
{
    stop();
}

/* ========================================================================
 * Public API
 * ======================================================================== */

bool FaceTracker::start(int camera_index, const std::string &yunet_model)
{
    camera_index_ = camera_index;
    yunet_model_  = yunet_model;
    running       = true;
    worker        = std::thread(&FaceTracker::threadLoop, this);
    return true;
}

void FaceTracker::stop()
{
    running = false;
    if (worker.joinable())
        worker.join();
}

void FaceTracker::calibrate()
{
    std::lock_guard<std::mutex> lock(mtx);
    ref_yaw = ema_yaw;
    printf("[FaceTracker] Calibrated: ref_yaw = %.3f rad\n", ref_yaw);
}

float FaceTracker::shift() const
{
    std::lock_guard<std::mutex> lock(mtx);
    return u_shift_val;
}

bool FaceTracker::isActive() const
{
    return running.load();
}

/* ========================================================================
 * Background thread
 * ======================================================================== */

void FaceTracker::threadLoop()
{
    /* ---- Open webcam ---- */
    cv::VideoCapture cap(camera_index_);
    if (!cap.isOpened()) {
        fprintf(stderr, "[FaceTracker] Failed to open camera %d\n", camera_index_);
        running = false;
        return;
    }

    /* Grab one frame to learn the resolution */
    cv::Mat frame;
    cap >> frame;
    if (frame.empty()) {
        fprintf(stderr, "[FaceTracker] Empty first frame from camera %d\n",
                camera_index_);
        running = false;
        return;
    }

    const int fw = frame.cols;
    const int fh = frame.rows;
    printf("[FaceTracker] Camera %d opened: %dx%d\n", camera_index_, fw, fh);

    /*
     * Estimated pinhole intrinsics.
     * A typical webcam focal length is close to one image-width in pixels,
     * and the principal point is at the image centre.  This approximation
     * is adequate for coarse yaw extraction.
     */
    const double fx = static_cast<double>(fw);
    cam_matrix = (cv::Mat_<double>(3, 3) <<
                  fx,  0.0, fw / 2.0,
                  0.0, fx,  fh / 2.0,
                  0.0, 0.0,       1.0);
    dist_coeffs = cv::Mat::zeros(4, 1, CV_64F);

    /* ---- Create YuNet detector ---- */
    try {
        detector = cv::FaceDetectorYN::create(
            yunet_model_,
            "",                     // config: empty for ONNX
            cv::Size(fw, fh),
            0.6f,                   // score threshold
            0.3f,                   // NMS threshold
            5000,                   // top-K candidates
            FT_BACKEND,
            FT_TARGET);
    } catch (const cv::Exception &ex) {
        fprintf(stderr, "[FaceTracker] YuNet init failed: %s\n", ex.what());
        running = false;
        return;
    }

    printf("[FaceTracker] Ready — model: %s\n", yunet_model_.c_str());

    /* ---- Main capture loop ---- */
    while (running.load()) {
        cap >> frame;
        if (frame.empty())
            continue;

        /*
         * Detect faces.
         * Output Mat shape: [N, 15], type CV_32F
         * Columns per face:
         *   0-3:   bounding box (x, y, w, h)
         *   4-5:   right eye (x, y)
         *   6-7:   left eye  (x, y)
         *   8-9:   nose tip  (x, y)
         *   10-11: right mouth corner (x, y)
         *   12-13: left mouth corner  (x, y)
         *   14:    confidence score
         */
        cv::Mat faces;
        detector->detect(frame, faces);
        if (faces.rows == 0)
            continue;   // no face — hold last u_shift value

        /* Extract 5 image-plane keypoints from the highest-confidence face */
        std::vector<cv::Point2f> image_points;
        image_points.reserve(5);
        for (int i = 0; i < 5; i++) {
            float px = faces.at<float>(0, 4 + i * 2);
            float py = faces.at<float>(0, 5 + i * 2);
            image_points.emplace_back(px, py);
        }

        /* ---- solvePnP: 3-D model → 2-D landmarks ---- */
        cv::Mat rvec, tvec;
        if (!cv::solvePnP(model_points, image_points,
                          cam_matrix, dist_coeffs,
                          rvec, tvec,
                          false, cv::SOLVEPNP_ITERATIVE))
            continue;

        /* ---- Extract yaw from rotation matrix ---- */
        cv::Mat R;
        cv::Rodrigues(rvec, R);

        /*
         * Yaw angle (rotation around vertical/Y axis).
         * Using atan2(R[2][0], R[0][0]) as specified in the plan.
         * Negate FT_SENSITIVITY if the parallax direction feels inverted.
         */
        const float yaw = static_cast<float>(
            std::atan2(R.at<double>(2, 0), R.at<double>(0, 0)));

        /* ---- EMA smoothing + u_shift update (single lock) ---- */
        {
            std::lock_guard<std::mutex> lock(mtx);

            if (first_detection) {
                /* Seed EMA with first measurement to avoid startup transient */
                ema_yaw         = yaw;
                first_detection = false;
            } else {
                ema_yaw = FT_SMOOTH * yaw + (1.0f - FT_SMOOTH) * ema_yaw;
            }

            const float rel_yaw = ema_yaw - ref_yaw;
            u_shift_val = ft_clamp(0.5f + rel_yaw * FT_SENSITIVITY,
                                   0.0f, 1.0f);
        }
    }

    cap.release();
    printf("[FaceTracker] Thread exited\n");
}
