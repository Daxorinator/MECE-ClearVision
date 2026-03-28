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

/* ========================================================================
 * TensorRT helpers
 * ======================================================================== */

/* Minimal logger — suppresses INFO noise, prints WARN and above. */
struct TRTLogger : public nvinfer1::ILogger {
    void log(Severity sev, const char *msg) noexcept override {
        if (sev <= Severity::kWARNING)
            fprintf(stderr, "[TRT] %s\n", msg);
    }
} g_trt_logger;

/* Build a TRT engine from an ONNX file and cache it as <onnx_path>.trt.
 * On subsequent calls the cached engine is deserialized directly. */
static nvinfer1::ICudaEngine *load_or_build_engine(const std::string &onnx_path)
{
    std::string cache_path = onnx_path + ".trt";

    /* --- Try loading cached engine --- */
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
            fprintf(stderr, "[FaceTracker] Cache corrupt — rebuilding engine\n");
        }
    }

    /* --- Build from ONNX --- */
    printf("[FaceTracker] Building TRT engine from %s\n"
           "              (first run only — may take several minutes on Nano)\n",
           onnx_path.c_str());

    auto *builder = nvinfer1::createInferBuilder(g_trt_logger);
    auto *network = builder->createNetworkV2(
        1U << static_cast<uint32_t>(
            nvinfer1::NetworkDefinitionCreationFlag::kEXPLICIT_BATCH));
    auto *parser  = nvonnxparser::createParser(*network, g_trt_logger);

    if (!parser->parseFromFile(onnx_path.c_str(),
                               static_cast<int>(nvinfer1::ILogger::Severity::kWARNING))) {
        fprintf(stderr, "[FaceTracker] Failed to parse ONNX: %s\n", onnx_path.c_str());
        delete parser; delete network; delete builder;
        return nullptr;
    }

    auto *config = builder->createBuilderConfig();
    config->setMemoryPoolLimit(nvinfer1::MemoryPoolType::kWORKSPACE, 256u << 20);

    /* Use FP16 if the platform supports it (Maxwell has limited FP16 but try). */
    if (builder->platformHasFastFp16())
        config->setFlag(nvinfer1::BuilderFlag::kFP16);

    /* TRT 8: buildSerializedNetwork returns an IHostMemory plan. */
    auto *plan = builder->buildSerializedNetwork(*network, *config);
    if (!plan) {
        fprintf(stderr, "[FaceTracker] TRT engine build failed\n");
        delete config; delete parser; delete network; delete builder;
        return nullptr;
    }

    /* Cache to disk. */
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

bool FaceTracker::start(int camera_index,
                        const std::string &cascade_path,
                        const std::string &facemesh_onnx)
{
    camera_index_  = camera_index;
    cascade_path_  = cascade_path;
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
    ref_iris_x_ = head_pos_.x + ref_iris_x_;  // store absolute iris_x
    ref_iris_y_ = head_pos_.y + ref_iris_y_;
    head_pos_.x = 0.f;
    head_pos_.y = 0.f;
    calibrated_ = true;
    printf("[FaceTracker] Calibrated  ref_iris=(%.1f, %.1f)\n",
           ref_iris_x_, ref_iris_y_);
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
    /* ---- Load Haar cascade ---- */
    if (!cascade_.load(cascade_path_)) {
        fprintf(stderr, "[FaceTracker] Failed to load cascade: %s\n",
                cascade_path_.c_str());
        running_ = false;
        return;
    }

    /* ---- Open webcam via GStreamer ---- */
    const char *pipeline_fmt =
        "v4l2src device=/dev/video%d"
        " ! image/jpeg, width=%d, height=%d"
        " ! jpegdec ! videoconvert"
        " ! video/x-raw, format=BGR"
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
    printf("[FaceTracker] Camera %d: %dx%d (BGR)\n", camera_index_, fw, fh);

    /* Approximate focal length from assumed 70° horizontal FOV */
    const float focal_px = fw / (2.0f * std::tan(35.0f * M_PI / 180.0f));
    const float cam_cx   = fw * 0.5f;
    const float cam_cy   = fh * 0.5f;

    /* ---- Load / build TRT engine ---- */
    auto *engine = load_or_build_engine(facemesh_onnx_);
    if (!engine) {
        fprintf(stderr, "[FaceTracker] No TRT engine — thread exiting\n");
        running_ = false;
        return;
    }
    auto *context = engine->createExecutionContext();

    /* Inspect input and output bindings */
    const int n_bindings = engine->getNbBindings();
    int input_idx  = -1;
    int output_idx = -1;
    size_t input_bytes  = 0;
    size_t output_bytes = 0;
    int    n_landmarks  = 0;
    bool   nhwc_input   = true;   // most PINTO0309 exports use NHWC

    for (int i = 0; i < n_bindings; i++) {
        auto dims = engine->getBindingDimensions(i);
        size_t vol = 1;
        for (int d = 0; d < dims.nbDims; d++) vol *= (size_t)dims.d[d];
        printf("[FaceTracker] binding %d '%s' %s dims=[",
               i, engine->getBindingName(i),
               engine->bindingIsInput(i) ? "IN " : "OUT");
        for (int d = 0; d < dims.nbDims; d++)
            printf("%s%d", d ? "," : "", dims.d[d]);
        printf("] vol=%zu\n", vol);

        if (engine->bindingIsInput(i)) {
            input_idx   = i;
            input_bytes = vol * sizeof(float);
            /* Detect NCHW: if the binding has (B, C, H, W) with C=3 in position 1 */
            if (dims.nbDims == 4 && dims.d[1] == 3)
                nhwc_input = false;
        } else if (output_idx < 0) {
            /* Take the first output — should be the landmark tensor */
            output_idx   = i;
            output_bytes = vol * sizeof(float);
            n_landmarks  = (int)(vol / 3);
            printf("[FaceTracker] Landmark count: %d (%s iris)\n",
                   n_landmarks, n_landmarks >= 478 ? "WITH" : "WITHOUT");
        }
    }

    if (input_idx < 0 || output_idx < 0 || n_landmarks < 1) {
        fprintf(stderr, "[FaceTracker] Unexpected model bindings — aborting\n");
        delete context; delete engine;
        running_ = false;
        return;
    }

    if (n_landmarks < 478) {
        fprintf(stderr, "[FaceTracker] WARNING: model has only %d landmarks "
                "(need 478 for iris depth — check you have the *_with_attention model)\n",
                n_landmarks);
    }

    /* Allocate GPU/CPU buffers */
    void *d_buffers[8] = {};   /* device pointers, one per binding */
    cudaMalloc(&d_buffers[input_idx],  input_bytes);
    cudaMalloc(&d_buffers[output_idx], output_bytes);

    /* Allocate extra output buffers (face flag etc.) */
    for (int i = 0; i < n_bindings; i++) {
        if (!engine->bindingIsInput(i) && i != output_idx) {
            auto dims = engine->getBindingDimensions(i);
            size_t vol = 1;
            for (int d = 0; d < dims.nbDims; d++) vol *= (size_t)dims.d[d];
            cudaMalloc(&d_buffers[i], vol * sizeof(float));
        }
    }

    std::vector<float> h_input (input_bytes  / sizeof(float));
    std::vector<float> h_output(output_bytes / sizeof(float));

    /* Diagnostic: print first few landmarks on first successful inference */
    bool first_infer = true;

    /* ---- Main loop ---- */
    cv::Mat frame, face_crop_rgb;
    cv::Rect last_bbox;
    bool bbox_valid  = false;
    int  frame_count = 0;

    while (running_.load()) {
        cap >> frame;
        if (frame.empty()) continue;

        const bool do_detect = (!bbox_valid) || (frame_count % FT_DETECT_EVERY == 0);

        /* --- Face detection --- */
        if (do_detect) {
            cv::Mat gray;
            cv::cvtColor(frame, gray, cv::COLOR_BGR2GRAY);
            cv::Mat small;
            cv::resize(gray, small, cv::Size(320, 240), 0, 0, cv::INTER_NEAREST);

            std::vector<cv::Rect> faces;
            cascade_.detectMultiScale(small, faces, 1.1, 3, 0, cv::Size(30, 30));

            if (!faces.empty()) {
                /* Largest face in 320×240 space → scale to full-res */
                auto &best = *std::max_element(faces.begin(), faces.end(),
                    [](const cv::Rect &a, const cv::Rect &b){ return a.area()<b.area(); });
                float sx = (float)fw / 320.f;
                float sy = (float)fh / 240.f;
                last_bbox = cv::Rect(
                    (int)(best.x * sx), (int)(best.y * sy),
                    (int)(best.width * sx), (int)(best.height * sy))
                    & cv::Rect(0, 0, fw, fh);
                bbox_valid = true;
            } else {
                bbox_valid = false;
            }
        }

        if (!bbox_valid) { ++frame_count; continue; }

        /* --- Build square crop with 30% padding --- */
        int size = std::max(last_bbox.width, last_bbox.height);
        int pad  = size / 3;
        size += 2 * pad;
        int cx = last_bbox.x + last_bbox.width  / 2;
        int cy = last_bbox.y + last_bbox.height / 2;
        cv::Rect crop_rect(cx - size/2, cy - size/2, size, size);
        crop_rect &= cv::Rect(0, 0, fw, fh);
        if (crop_rect.width < 32 || crop_rect.height < 32) { ++frame_count; continue; }

        /* --- Preprocess to 192×192 RGB float [0,1] --- */
        cv::Mat crop = frame(crop_rect);
        cv::resize(crop, face_crop_rgb, cv::Size(192, 192), 0, 0, cv::INTER_LINEAR);
        cv::cvtColor(face_crop_rgb, face_crop_rgb, cv::COLOR_BGR2RGB);
        face_crop_rgb.convertTo(face_crop_rgb, CV_32FC3, 1.0 / 255.0);

        /* Fill input buffer (NHWC or NCHW) */
        if (nhwc_input) {
            std::memcpy(h_input.data(), face_crop_rgb.data, input_bytes);
        } else {
            /* NCHW: split channels */
            std::vector<cv::Mat> ch(3);
            cv::split(face_crop_rgb, ch);
            for (int c = 0; c < 3; c++)
                std::memcpy(h_input.data() + c * 192 * 192, ch[c].data,
                            192 * 192 * sizeof(float));
        }

        /* --- TRT inference --- */
        cudaMemcpy(d_buffers[input_idx], h_input.data(), input_bytes,
                   cudaMemcpyHostToDevice);
        context->executeV2(d_buffers);
        cudaMemcpy(h_output.data(), d_buffers[output_idx], output_bytes,
                   cudaMemcpyDeviceToHost);

        const float *lm = h_output.data();

        /* Diagnostic on first inference */
        if (first_infer) {
            printf("[FaceTracker] First inference: lm[0..5] = "
                   "%.3f %.3f %.3f  %.3f %.3f %.3f\n",
                   lm[0], lm[1], lm[2], lm[3], lm[4], lm[5]);
            first_infer = false;
            active_ = true;
        }

        /* Determine coordinate scale: if max x/y < 2, landmarks are normalised [0,1] */
        float lm_scale = 1.0f;
        {
            float max_xy = 0.f;
            int check = std::min(n_landmarks, 10);
            for (int i = 0; i < check; i++)
                max_xy = std::max(max_xy, std::max(std::abs(lm[i*3]), std::abs(lm[i*3+1])));
            if (max_xy < 2.0f)
                lm_scale = 192.0f;   /* normalised → crop pixel space */
        }

        /* Check face confidence if a second output exists */
        /* (Not strictly required — proceed regardless for now.) */

        /* --- Iris depth estimation (requires 478 landmarks) --- */
        if (n_landmarks < 478) { ++frame_count; continue; }

        /* Left iris (468–472): 468=center, 469=top, 470=right, 471=bottom, 472=left */
        auto lx = [&](int idx, int c){ return lm[idx*3+c] * lm_scale; };

        /* Diameter of left iris (average horizontal + vertical span) */
        float dh = std::hypot(lx(470,0)-lx(472,0), lx(470,1)-lx(472,1));
        float dv = std::hypot(lx(469,0)-lx(471,0), lx(469,1)-lx(471,1));
        float diam_crop = (dh + dv) * 0.5f;
        if (diam_crop < 1.0f) { ++frame_count; continue; }   /* degenerate */

        /* Scale to original image pixels */
        float crop_to_img  = (float)crop_rect.width / 192.0f;
        float iris_diam_px = diam_crop * crop_to_img;

        /* Average of left (468) and right (473) iris centres */
        float iris_cx = ((lx(468,0) + lx(473,0)) * 0.5f) * crop_to_img + crop_rect.x;
        float iris_cy = ((lx(468,1) + lx(473,1)) * 0.5f) * crop_to_img + crop_rect.y;

        /* Depth from iris diameter */
        const float IRIS_DIAM_M = 0.01170f;
        float Z = IRIS_DIAM_M * focal_px / iris_diam_px;

        /* Lateral/vertical position relative to camera centre */
        float img_X = (iris_cx - cam_cx) * Z / focal_px;
        float img_Y = (iris_cy - cam_cy) * Z / focal_px;

        /* Apply EMA smoothing and store under mutex */
        {
            std::lock_guard<std::mutex> lock(mtx_);
            float new_x = img_X - ref_iris_x_;
            float new_y = img_Y - ref_iris_y_;
            head_pos_.x = FT_SMOOTH * new_x + (1.f - FT_SMOOTH) * head_pos_.x;
            head_pos_.y = FT_SMOOTH * new_y + (1.f - FT_SMOOTH) * head_pos_.y;
            head_pos_.z = FT_SMOOTH * Z      + (1.f - FT_SMOOTH) * head_pos_.z;
            head_pos_.valid = true;
        }

        ++frame_count;
    }

    /* ---- Cleanup ---- */
    for (int i = 0; i < n_bindings; i++)
        if (d_buffers[i]) cudaFree(d_buffers[i]);
    delete context;
    delete engine;
    cap.release();
    printf("[FaceTracker] Thread exited\n");
}
