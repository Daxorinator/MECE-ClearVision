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
    config->setMaxWorkspaceSize(256u << 20);

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
    /* ---- Open webcam via GStreamer ---- */
    const char *pipeline_fmt =
        "v4l2src device=/dev/video%d"
        " ! image/jpeg, width=%d, height=%d"
        " ! jpegdec ! videoconvert"
        " ! video/x-raw, format=BGR"
        " ! appsink drop=true max-buffers=1";

    cv::VideoCapture cap;
    int fw = 0, fh = 0;
    /* Try highest resolution first — larger iris diameter = more reliable detection */
    for (auto [w, h] : std::initializer_list<std::pair<int,int>>{{640,480},{320,240}}) {
        char buf[512];
        std::snprintf(buf, sizeof(buf), pipeline_fmt, camera_index_, w, h);
        cap.open(buf, cv::CAP_GSTREAMER);
        if (cap.isOpened()) { fw = w; fh = h; break; }
    }
    /* GStreamer fallback: try plain V4L2 (no MJPEG requirement) */
    if (!cap.isOpened()) {
        cap.open(camera_index_);
        if (cap.isOpened()) {
            cap.set(cv::CAP_PROP_FRAME_WIDTH,  640);
            cap.set(cv::CAP_PROP_FRAME_HEIGHT, 480);
            fw = (int)cap.get(cv::CAP_PROP_FRAME_WIDTH);
            fh = (int)cap.get(cv::CAP_PROP_FRAME_HEIGHT);
        }
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

    /* Look for separate left/right iris outputs (vol==10) and face flag (vol==1) */
    int left_iris_idx  = -1;
    int right_iris_idx = -1;
    int face_flag_idx  = -1;
    for (int i = 0; i < n_bindings; i++) {
        if (engine->bindingIsInput(i) || i == output_idx) continue;
        auto dims = engine->getBindingDimensions(i);
        size_t vol = 1;
        for (int d = 0; d < dims.nbDims; d++) vol *= (size_t)dims.d[d];
        std::string name = engine->getBindingName(i);
        if (vol == 10) {
            if (name.find("left")  != std::string::npos) left_iris_idx  = i;
            if (name.find("right") != std::string::npos) right_iris_idx = i;
        }
        /* Face flag: scalar output that signals face presence */
        /* Face flag: prefer a binding whose name contains "score"/"flag"/"face";
         * fall back to first scalar output.  Vol==1 covers both (1,) and (1,1,1,1). */
        if (vol == 1) {
            bool named = (name.find("score") != std::string::npos ||
                          name.find("flag")  != std::string::npos ||
                          name.find("face")  != std::string::npos);
            if (named || face_flag_idx < 0)
                face_flag_idx = i;
        }
    }
    if (left_iris_idx >= 0 && right_iris_idx >= 0)
        printf("[FaceTracker] Iris outputs: left=binding%d right=binding%d\n",
               left_iris_idx, right_iris_idx);
    if (face_flag_idx >= 0) {
        printf("[FaceTracker] Face flag: binding%d ('%s')\n",
               face_flag_idx, engine->getBindingName(face_flag_idx));
    } else {
        printf("[FaceTracker] WARNING: no face flag output found — "
               "all frames will be accepted\n");
    }

    if (input_idx < 0 || output_idx < 0 || n_landmarks < 1) {
        fprintf(stderr, "[FaceTracker] Unexpected model bindings — aborting\n");
        delete context; delete engine;
        running_ = false;
        return;
    }

    if (n_landmarks < 478 && left_iris_idx < 0) {
        fprintf(stderr, "[FaceTracker] WARNING: no iris outputs found — "
                "iris depth unavailable\n");
    }

    /* Allocate GPU/CPU buffers */
    void *d_buffers[8] = {};   /* device pointers, one per binding */
    cudaMalloc(&d_buffers[input_idx],  input_bytes);
    cudaMalloc(&d_buffers[output_idx], output_bytes);

    /* Allocate all remaining output buffers (face flag, iris, etc.) */
    for (int i = 0; i < n_bindings; i++) {
        if (d_buffers[i]) continue;   /* already allocated */
        auto dims = engine->getBindingDimensions(i);
        size_t vol = 1;
        for (int d = 0; d < dims.nbDims; d++) vol *= (size_t)dims.d[d];
        cudaMalloc(&d_buffers[i], vol * sizeof(float));
    }

    std::vector<float> h_input     (input_bytes  / sizeof(float));
    std::vector<float> h_output   (output_bytes / sizeof(float));
    std::vector<float> h_left_iris (10, 0.f);
    std::vector<float> h_right_iris(10, 0.f);
    float h_face_flag = 0.f;

    /* Diagnostic: print first few landmarks on first successful inference */
    bool first_infer = true;

    /* ---- Main loop ---- */
    cv::Mat frame, face_crop_rgb;
    cv::Rect crop_rect;
    bool crop_valid  = false;
    int  frame_count = 0;

    while (running_.load()) {
        cap >> frame;
        if (frame.empty()) continue;

        /* --- Determine crop for this frame ---
         * Re-detection: use a square that spans the FULL frame width so a face
         * anywhere in the image is found — not just the centre.
         * Tracking: use the tight crop from last iris position. */
        if (!crop_valid) {
            int size = std::max(fw, fh);
            crop_rect = cv::Rect((fw - size) / 2, (fh - size) / 2, size, size);
        }

        if (crop_rect.width < 32 || crop_rect.height < 32) { ++frame_count; continue; }

        /* --- Extract crop with black padding for out-of-bounds regions.
         * Do NOT use &= to clip crop_rect — that would corrupt crop_to_img and
         * the iris coordinate mapping when the face is near a frame edge. --- */
        cv::Mat crop;
        {
            int pad_top    = std::max(0, -crop_rect.y);
            int pad_bottom = std::max(0, crop_rect.y + crop_rect.height - fh);
            int pad_left   = std::max(0, -crop_rect.x);
            int pad_right  = std::max(0, crop_rect.x + crop_rect.width  - fw);
            cv::Rect actual = crop_rect & cv::Rect(0, 0, fw, fh);
            cv::Mat  raw    = frame(actual);
            if (pad_top || pad_bottom || pad_left || pad_right)
                cv::copyMakeBorder(raw, crop, pad_top, pad_bottom,
                                   pad_left, pad_right,
                                   cv::BORDER_CONSTANT, cv::Scalar(0, 0, 0));
            else
                crop = raw;
        }

        /* --- Preprocess to 192×192 RGB float [0,1] --- */
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
        if (left_iris_idx >= 0)
            cudaMemcpy(h_left_iris.data(),  d_buffers[left_iris_idx],  10*sizeof(float),
                       cudaMemcpyDeviceToHost);
        if (right_iris_idx >= 0)
            cudaMemcpy(h_right_iris.data(), d_buffers[right_iris_idx], 10*sizeof(float),
                       cudaMemcpyDeviceToHost);
        if (face_flag_idx >= 0)
            cudaMemcpy(&h_face_flag, d_buffers[face_flag_idx], sizeof(float),
                       cudaMemcpyDeviceToHost);

        const float *lm = h_output.data();

        /* Diagnostic on first inference */
        if (first_infer) {
            printf("[FaceTracker] First inference: lm[0..5] = "
                   "%.3f %.3f %.3f  %.3f %.3f %.3f  face_flag=%.3f\n",
                   lm[0], lm[1], lm[2], lm[3], lm[4], lm[5], h_face_flag);
            first_infer = false;
            active_ = true;
        }

        /* Face flag check intentionally omitted: the binding identified as face_flag
         * (vol==1) is not a reliable presence score in this ONNX export.
         * Bad detections are handled by diam_crop < 1.0 below. */

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

        /* --- Iris depth estimation --- */
        if (n_landmarks < 478 && left_iris_idx < 0) { crop_valid = false; ++frame_count; continue; }

        /* Separate iris outputs: 5 pts × 2 coords (x,y, stride=2) in crop-pixel space.
           Packed fallback (478 lm): stride=3, left starts at 468, right at 473.
           Point order: 0=center, 1=top, 2=right, 3=bottom, 4=left */
        const float *l_iris = (left_iris_idx  >= 0) ? h_left_iris.data()  : (lm + 468*3);
        const float *r_iris = (right_iris_idx >= 0) ? h_right_iris.data() : (lm + 473*3);
        const int stride    = (left_iris_idx  >= 0) ? 2 : 3;

        auto lix = [&](int pt, int c){ return l_iris[pt*stride+c] * lm_scale; };
        auto rix = [&](int pt, int c){ return r_iris[pt*stride+c] * lm_scale; };

        /* Diameter of left iris (average horizontal + vertical span) */
        float dh = std::hypot(lix(2,0)-lix(4,0), lix(2,1)-lix(4,1));
        float dv = std::hypot(lix(1,0)-lix(3,0), lix(1,1)-lix(3,1));
        float diam_crop = (dh + dv) * 0.5f;
        if (diam_crop < 1.0f) { crop_valid = false; ++frame_count; continue; }   /* degenerate — retry from full frame */

        /* Scale to original image pixels */
        float crop_to_img  = (float)crop_rect.width / 192.0f;
        float iris_diam_px = diam_crop * crop_to_img;

        /* Average of left and right iris centres */
        float iris_cx = ((lix(0,0) + rix(0,0)) * 0.5f) * crop_to_img + crop_rect.x;
        float iris_cy = ((lix(0,1) + rix(0,1)) * 0.5f) * crop_to_img + crop_rect.y;

        /* Update crop for next frame — centred on the iris, sized ~15× iris diameter.
         * 15× captures the full face; 6× was too tight and caused FaceMesh to lose lock. */
        {
            int crop_size = std::max((int)(iris_diam_px * 15.0f), 96);
            int ix = (int)iris_cx, iy = (int)iris_cy;
            crop_rect  = cv::Rect(ix - crop_size/2, iy - crop_size/2, crop_size, crop_size);
            crop_valid = true;
        }

        /* Depth from iris diameter */
        const float IRIS_DIAM_M = 0.01170f;
        float Z = IRIS_DIAM_M * focal_px / iris_diam_px;

        if (frame_count % 60 == 0)
            printf("[FaceTracker] iris_diam=%.1fpx  Z=%.2fm  crop=%dx%d  flag=%.2f\n",
                   iris_diam_px, Z, crop_rect.width, crop_rect.height, h_face_flag);

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
