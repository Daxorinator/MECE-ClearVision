#include "face_tracker.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <vector>

#include <opencv2/dnn.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/videoio.hpp>

/* ========================================================================
 * Public API
 * ======================================================================== */

bool FaceTracker::start(int camera_index, const std::string &facemesh_onnx)
{
    camera_index_  = camera_index;
    facemesh_onnx_ = facemesh_onnx;
    running_       = true;
    worker_        = std::thread(&FaceTracker::threadLoop, this);
    return true;
}

void FaceTracker::stop()
{
    running_ = false;
    if (worker_.joinable()) worker_.join();
}

void FaceTracker::calibrate()
{
    std::lock_guard<std::mutex> lock(mtx_);
    ref_iris_x_ += head_pos_.x;
    ref_iris_y_ += head_pos_.y;
    head_pos_.x = 0.f;
    head_pos_.y = 0.f;
    printf("[FaceTracker] Calibrated  ref=(%.3f, %.3f)\n", ref_iris_x_, ref_iris_y_);
}

HeadPos FaceTracker::headPos() const
{
    std::lock_guard<std::mutex> lock(mtx_);
    return head_pos_;
}

bool FaceTracker::isActive() const { return active_.load(); }

/* ========================================================================
 * Background thread
 * ======================================================================== */

void FaceTracker::threadLoop()
{
    /* ---- Open camera ---- */
    cv::VideoCapture cap(camera_index_);
    if (!cap.isOpened()) {
        fprintf(stderr, "[FaceTracker] Cannot open camera %d\n", camera_index_);
        running_ = false;
        return;
    }
    cap.set(cv::CAP_PROP_FRAME_WIDTH,  1280);
    cap.set(cv::CAP_PROP_FRAME_HEIGHT,  720);
    int fw = (int)cap.get(cv::CAP_PROP_FRAME_WIDTH);
    int fh = (int)cap.get(cv::CAP_PROP_FRAME_HEIGHT);
    printf("[FaceTracker] Camera %d: %dx%d\n", camera_index_, fw, fh);

    /* ---- Load FaceMesh via OpenCV DNN ---- */
    cv::dnn::Net net;
    try { net = cv::dnn::readNetFromONNX(facemesh_onnx_); }
    catch (const cv::Exception &e) {
        fprintf(stderr, "[FaceTracker] Cannot load ONNX: %s\n", e.what());
        running_ = false;
        return;
    }

    /* Try CUDA backend; fall back to CPU if unavailable */
    net.setPreferableBackend(cv::dnn::DNN_BACKEND_CUDA);
    net.setPreferableTarget(cv::dnn::DNN_TARGET_CUDA);
    {
        /* Probe: if CUDA fails the first forward will throw; catch and use CPU */
        cv::Mat probe(192, 192, CV_32FC3, cv::Scalar(0.f));
        std::vector<int> nhwc = {1, 192, 192, 3};
        cv::Mat blob(nhwc, CV_32F, probe.data);
        try { net.setInput(blob.clone()); net.forward(); printf("[FaceTracker] CUDA backend OK\n"); }
        catch (...) {
            net.setPreferableBackend(cv::dnn::DNN_BACKEND_DEFAULT);
            net.setPreferableTarget(cv::dnn::DNN_TARGET_CPU);
            printf("[FaceTracker] CUDA unavailable — using CPU\n");
        }
    }

    /* ---- Detect input format (NHWC vs NCHW) once at startup ---- */
    bool use_nchw = false;
    {
        cv::Mat probe(192, 192, CV_32FC3, cv::Scalar(0.5f));
        std::vector<cv::Mat> outs;

        /* Try NHWC first (original TFLite layout) */
        std::vector<int> nhwc = {1, 192, 192, 3};
        cv::Mat nhwc_blob(nhwc, CV_32F, probe.data);
        bool nhwc_ok = false;
        try {
            net.setInput(nhwc_blob.clone());
            net.forward(outs, net.getUnconnectedOutLayersNames());
            for (auto &o : outs) if ((int)o.total() >= 478*3) { nhwc_ok = true; break; }
        } catch (...) {}

        if (!nhwc_ok) {
            /* Try NCHW */
            try {
                net.setInput(cv::dnn::blobFromImage(probe));
                net.forward(outs, net.getUnconnectedOutLayersNames());
                for (auto &o : outs) if ((int)o.total() >= 478*3) { use_nchw = true; break; }
            } catch (...) {}
        }

        if (!nhwc_ok && !use_nchw) {
            fprintf(stderr, "[FaceTracker] Model format detection failed — "
                            "check ONNX file\n");
            running_ = false;
            return;
        }
        printf("[FaceTracker] Input format: %s\n", use_nchw ? "NCHW" : "NHWC");
    }

    /* ---- Fixed square crop: full frame height centred horizontally ---- */
    const int   sq      = std::min(fw, fh);
    const float c_scale = (float)sq / 192.f;  /* crop px per model px */
    const float cx0     = (fw - sq) / 2.f;
    const float cy0     = (fh - sq) / 2.f;
    const cv::Rect sq_rect((fw-sq)/2, (fh-sq)/2, sq, sq);

    /* ---- Camera intrinsics (assumed 70° H-FOV) ---- */
    const float focal   = (float)fw / (2.f * std::tan(35.f * (float)M_PI / 180.f));
    const float IRIS_M  = 0.01170f;   /* average iris diameter, metres */

    printf("[FaceTracker] Ready — crop %dx%d  focal=%.0fpx\n", sq, sq, focal);
    active_ = true;

    cv::Mat frame, input;
    int frame_count = 0;

    while (running_.load()) {
        cap >> frame;
        if (frame.empty()) continue;

        /* Crop to square, resize to 192×192, normalise */
        cv::resize(frame(sq_rect), input, cv::Size(192, 192));
        cv::cvtColor(input, input, cv::COLOR_BGR2RGB);
        input.convertTo(input, CV_32FC3, 1.0 / 255.0);

        /* Run inference */
        std::vector<cv::Mat> outs;
        try {
            if (use_nchw) {
                net.setInput(cv::dnn::blobFromImage(input));
            } else {
                std::vector<int> nhwc = {1, 192, 192, 3};
                cv::Mat blob(nhwc, CV_32F, input.data);
                net.setInput(blob.clone());
            }
            net.forward(outs, net.getUnconnectedOutLayersNames());
        } catch (const cv::Exception &e) {
            fprintf(stderr, "[FaceTracker] Inference error: %s\n", e.what());
            ++frame_count;
            continue;
        }

        /* Find landmark tensor (>= 478×3 elements) */
        const float *lm = nullptr;
        for (auto &o : outs)
            if ((int)o.total() >= 478 * 3) { lm = o.ptr<float>(); break; }
        if (!lm) { ++frame_count; continue; }

        /* Landmarks are either normalised [0,1] or pixel [0,192] */
        float max_xy = 0.f;
        for (int i = 0; i < 10; i++)
            max_xy = std::max(max_xy,
                              std::max(std::abs(lm[i*3]), std::abs(lm[i*3+1])));
        const float sc = (max_xy < 2.f) ? 192.f : 1.f;

        auto lx = [&](int i) { return lm[i*3+0] * sc; };
        auto ly = [&](int i) { return lm[i*3+1] * sc; };

        /* Left iris (landmarks 468–472): 0=centre 1=top 2=right 3=bottom 4=left */
        float dh = std::hypot(lx(470)-lx(472), ly(470)-ly(472));
        float dv = std::hypot(lx(469)-lx(471), ly(469)-ly(471));
        float diam_crop = (dh + dv) * 0.5f;
        if (diam_crop < 2.f) { ++frame_count; continue; }   /* no face detected */

        /* Iris centre averaged across both eyes, mapped to full-frame coords */
        float cx_c   = ((lx(468) + lx(473)) * 0.5f) * c_scale + cx0;
        float cy_c   = ((ly(468) + ly(473)) * 0.5f) * c_scale + cy0;
        float diam_px = diam_crop * c_scale;

        float Z     = IRIS_M * focal / diam_px;
        float img_X = (cx_c - fw * 0.5f) * Z / focal;
        float img_Y = (cy_c - fh * 0.5f) * Z / focal;

        if (frame_count % 30 == 0)
            printf("[FaceTracker] diam=%.1fpx  Z=%.2fm  pos=(%.1f, %.1f)cm\n",
                   diam_px, Z, img_X * 100.f, img_Y * 100.f);

        {
            std::lock_guard<std::mutex> lock(mtx_);
            float nx = img_X - ref_iris_x_;
            float ny = img_Y - ref_iris_y_;
            head_pos_.x = FT_SMOOTH * nx + (1.f - FT_SMOOTH) * head_pos_.x;
            head_pos_.y = FT_SMOOTH * ny + (1.f - FT_SMOOTH) * head_pos_.y;
            head_pos_.z = FT_SMOOTH * Z  + (1.f - FT_SMOOTH) * head_pos_.z;
            head_pos_.valid = true;
        }
        ++frame_count;
    }

    cap.release();
    printf("[FaceTracker] Thread exited after %d frames\n", frame_count);
}
