#include "face_tracker.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <iterator>
#include <vector>

#include <cuda_runtime.h>
#include <NvInfer.h>
#include <NvOnnxParser.h>

#include <opencv2/imgproc.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/core.hpp>

/* ========================================================================
 * TensorRT helpers
 * ======================================================================== */

struct TRTLogger : public nvinfer1::ILogger {
    void log(Severity sev, const char *msg) noexcept override {
        if (sev <= Severity::kWARNING)
            fprintf(stderr, "[TRT] %s\n", msg);
    }
} g_trt_logger;

static nvinfer1::ICudaEngine *load_or_build_engine(const std::string &onnx_path)
{
    std::string cache_path = onnx_path + ".trt";

    {
        std::ifstream f(cache_path, std::ios::binary);
        if (f.good()) {
            std::vector<char> buf((std::istreambuf_iterator<char>(f)),
                                   std::istreambuf_iterator<char>());
            auto *runtime = nvinfer1::createInferRuntime(g_trt_logger);
            auto *engine  = runtime->deserializeCudaEngine(buf.data(), buf.size());
            delete runtime;
            if (engine) {
                printf("[FaceTracker] TRT engine loaded from cache: %s\n",
                       cache_path.c_str());
                return engine;
            }
            fprintf(stderr, "[FaceTracker] Cache corrupt — rebuilding\n");
        }
    }

    printf("[FaceTracker] Building TRT engine from %s (first run only)\n",
           onnx_path.c_str());

    auto *builder = nvinfer1::createInferBuilder(g_trt_logger);
    auto *network = builder->createNetworkV2(
        1U << static_cast<uint32_t>(
            nvinfer1::NetworkDefinitionCreationFlag::kEXPLICIT_BATCH));
    auto *parser = nvonnxparser::createParser(*network, g_trt_logger);

    if (!parser->parseFromFile(onnx_path.c_str(),
                               static_cast<int>(nvinfer1::ILogger::Severity::kWARNING))) {
        fprintf(stderr, "[FaceTracker] Failed to parse ONNX\n");
        delete parser; delete network; delete builder;
        return nullptr;
    }

    auto *config = builder->createBuilderConfig();
    config->setMaxWorkspaceSize(256u << 20);
    if (builder->platformHasFastFp16())
        config->setFlag(nvinfer1::BuilderFlag::kFP16);

    auto *plan = builder->buildSerializedNetwork(*network, *config);
    if (!plan) {
        fprintf(stderr, "[FaceTracker] Engine build failed\n");
        delete config; delete parser; delete network; delete builder;
        return nullptr;
    }

    {
        std::ofstream f(cache_path, std::ios::binary);
        f.write(static_cast<const char *>(plan->data()), plan->size());
        printf("[FaceTracker] TRT engine cached to %s\n", cache_path.c_str());
    }

    auto *runtime = nvinfer1::createInferRuntime(g_trt_logger);
    auto *engine  = runtime->deserializeCudaEngine(plan->data(), plan->size());
    delete plan; delete runtime;
    delete config; delete parser; delete network; delete builder;
    return engine;
}

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
    /* ---- Open webcam via GStreamer ---- */
    const char *pipeline_fmt =
        "v4l2src device=/dev/video%d"
        " ! image/jpeg, width=%d, height=%d"
        " ! jpegdec ! videoconvert"
        " ! video/x-raw, format=BGR"
        " ! appsink drop=true max-buffers=1";

    cv::VideoCapture cap;
    int fw = 0, fh = 0;
    for (auto [w, h] : std::initializer_list<std::pair<int,int>>{{1280,720},{640,480}}) {
        char buf[512];
        std::snprintf(buf, sizeof(buf), pipeline_fmt, camera_index_, w, h);
        cap.open(buf, cv::CAP_GSTREAMER);
        if (cap.isOpened()) { fw = w; fh = h; break; }
    }
    if (!cap.isOpened()) {
        cap.open(camera_index_);
        if (cap.isOpened()) {
            fw = (int)cap.get(cv::CAP_PROP_FRAME_WIDTH);
            fh = (int)cap.get(cv::CAP_PROP_FRAME_HEIGHT);
        }
    }
    if (!cap.isOpened()) {
        fprintf(stderr, "[FaceTracker] Failed to open camera %d\n", camera_index_);
        running_ = false;
        return;
    }
    printf("[FaceTracker] Camera %d: %dx%d\n", camera_index_, fw, fh);

    /* Focal length and image centre (assumed 70° H-FOV) */
    const float focal_px = fw / (2.0f * std::tan(35.0f * M_PI / 180.0f));
    const float cam_cx   = fw * 0.5f;
    const float cam_cy   = fh * 0.5f;
    /* Scale from 192×192 model space back to full-frame pixels */
    const float scale_x  = (float)fw / 192.0f;
    const float scale_y  = (float)fh / 192.0f;

    /* ---- Load / build TRT engine ---- */
    auto *engine = load_or_build_engine(facemesh_onnx_);
    if (!engine) { running_ = false; return; }
    auto *context = engine->createExecutionContext();

    /* ---- Inspect bindings ---- */
    const int n_bindings = engine->getNbBindings();
    int    input_idx    = -1, output_idx = -1;
    int    left_iris_idx = -1, right_iris_idx = -1;
    size_t input_bytes = 0, output_bytes = 0;
    int    n_landmarks = 0;
    bool   nhwc_input  = true;

    for (int i = 0; i < n_bindings; i++) {
        auto dims = engine->getBindingDimensions(i);
        size_t vol = 1;
        for (int d = 0; d < dims.nbDims; d++) vol *= (size_t)dims.d[d];

        if (engine->bindingIsInput(i)) {
            input_idx   = i;
            input_bytes = vol * sizeof(float);
            if (dims.nbDims == 4 && dims.d[1] == 3) nhwc_input = false;
        } else if (output_idx < 0) {
            output_idx   = i;
            output_bytes = vol * sizeof(float);
            n_landmarks  = (int)(vol / 3);
        } else if (vol == 10) {
            std::string name = engine->getBindingName(i);
            if (name.find("left")  != std::string::npos) left_iris_idx  = i;
            if (name.find("right") != std::string::npos) right_iris_idx = i;
        }

        printf("[FaceTracker] binding %d '%s' %s vol=%zu\n",
               i, engine->getBindingName(i),
               engine->bindingIsInput(i) ? "IN" : "OUT", vol);
    }
    printf("[FaceTracker] %d landmarks, input %s, iris L=%d R=%d\n",
           n_landmarks, nhwc_input ? "NHWC" : "NCHW",
           left_iris_idx, right_iris_idx);

    if (input_idx < 0 || output_idx < 0) {
        fprintf(stderr, "[FaceTracker] Unexpected bindings\n");
        delete context; delete engine; running_ = false; return;
    }

    /* ---- Allocate GPU/CPU buffers ---- */
    void *d_buffers[8] = {};
    cudaMalloc(&d_buffers[input_idx],  input_bytes);
    cudaMalloc(&d_buffers[output_idx], output_bytes);
    for (int i = 0; i < n_bindings; i++) {
        if (d_buffers[i]) continue;
        auto dims = engine->getBindingDimensions(i);
        size_t vol = 1;
        for (int d = 0; d < dims.nbDims; d++) vol *= (size_t)dims.d[d];
        cudaMalloc(&d_buffers[i], vol * sizeof(float));
    }

    std::vector<float> h_input(input_bytes / sizeof(float));
    std::vector<float> h_output(output_bytes / sizeof(float));
    std::vector<float> h_left_iris(10, 0.f), h_right_iris(10, 0.f);

    /* ---- Main loop: resize full frame → 192×192 → infer ---- */
    cv::Mat frame, resized;
    int frame_count = 0;

    while (running_.load()) {
        cap >> frame;
        if (frame.empty()) continue;

        /* Resize the full frame to 192×192 and normalise */
        cv::resize(frame, resized, cv::Size(192, 192));
        cv::cvtColor(resized, resized, cv::COLOR_BGR2RGB);
        resized.convertTo(resized, CV_32FC3, 1.0 / 255.0);

        if (nhwc_input) {
            std::memcpy(h_input.data(), resized.data, input_bytes);
        } else {
            std::vector<cv::Mat> ch(3);
            cv::split(resized, ch);
            for (int c = 0; c < 3; c++)
                std::memcpy(h_input.data() + c * 192 * 192, ch[c].data,
                            192 * 192 * sizeof(float));
        }

        cudaMemcpy(d_buffers[input_idx], h_input.data(), input_bytes,
                   cudaMemcpyHostToDevice);
        context->executeV2(d_buffers);
        cudaMemcpy(h_output.data(), d_buffers[output_idx], output_bytes,
                   cudaMemcpyDeviceToHost);
        if (left_iris_idx  >= 0)
            cudaMemcpy(h_left_iris.data(),  d_buffers[left_iris_idx],
                       10 * sizeof(float), cudaMemcpyDeviceToHost);
        if (right_iris_idx >= 0)
            cudaMemcpy(h_right_iris.data(), d_buffers[right_iris_idx],
                       10 * sizeof(float), cudaMemcpyDeviceToHost);

        const float *lm = h_output.data();

        if (frame_count == 0) {
            printf("[FaceTracker] First inference: lm[0..5] = "
                   "%.3f %.3f %.3f  %.3f %.3f %.3f\n",
                   lm[0], lm[1], lm[2], lm[3], lm[4], lm[5]);
            active_ = true;
        }

        /* Landmarks: normalised [0,1] or pixel [0,192]? */
        float lm_scale = 1.0f;
        {
            float max_xy = 0.f;
            for (int i = 0; i < std::min(n_landmarks, 10); i++)
                max_xy = std::max(max_xy, std::max(std::abs(lm[i*3]),
                                                   std::abs(lm[i*3+1])));
            if (max_xy < 2.0f) lm_scale = 192.0f;
        }

        if (n_landmarks < 478 && left_iris_idx < 0) { ++frame_count; continue; }

        const float *l_iris = (left_iris_idx  >= 0) ? h_left_iris.data()  : (lm + 468*3);
        const float *r_iris = (right_iris_idx >= 0) ? h_right_iris.data() : (lm + 473*3);
        const int stride    = (left_iris_idx  >= 0) ? 2 : 3;

        auto lix = [&](int pt, int c){ return l_iris[pt*stride+c] * lm_scale; };
        auto rix = [&](int pt, int c){ return r_iris[pt*stride+c] * lm_scale; };

        /* Iris diameter in 192-space, then scale back to real frame pixels.
         * Use per-axis scale to correct for the non-square resize. */
        float dh_192 = std::hypot(lix(2,0)-lix(4,0), lix(2,1)-lix(4,1));
        float dv_192 = std::hypot(lix(1,0)-lix(3,0), lix(1,1)-lix(3,1));
        float diam_192 = (dh_192 + dv_192) * 0.5f;
        if (diam_192 < 1.0f) { ++frame_count; continue; }

        float iris_diam_px = (dh_192 * scale_x + dv_192 * scale_y) * 0.5f;

        /* Iris centre averaged across both eyes */
        float iris_cx = ((lix(0,0) + rix(0,0)) * 0.5f) * scale_x;
        float iris_cy = ((lix(0,1) + rix(0,1)) * 0.5f) * scale_y;

        const float IRIS_DIAM_M = 0.01170f;
        float Z     = IRIS_DIAM_M * focal_px / iris_diam_px;
        float img_X = (iris_cx - cam_cx) * Z / focal_px;
        float img_Y = (iris_cy - cam_cy) * Z / focal_px;

        if (frame_count % 30 == 0)
            printf("[FaceTracker] iris=%.1fpx  Z=%.2fm  pos=(%.1f, %.1f)cm\n",
                   iris_diam_px, Z, img_X * 100.f, img_Y * 100.f);

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

    for (int i = 0; i < n_bindings; i++)
        if (d_buffers[i]) cudaFree(d_buffers[i]);
    delete context;
    delete engine;
    cap.release();
    printf("[FaceTracker] Thread exited after %d frames\n", frame_count);
}
