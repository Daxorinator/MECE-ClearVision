/*
 * Depth Pipeline: Real-time Stereo Disparity Map
 *
 * Consumes calibration JSON from calibration.cpp to produce a live
 * colormapped disparity map using StereoBM.  Uses cv::Mat with NEON
 * auto-vectorization (Pi 5 has no usable OpenCL runtime).
 *
 * Build:
 *   cd build && cmake .. && make depth_pipeline
 *
 * Usage:
 *   ./depth_pipeline <path-to-calibration.json>
 *   ./depth_pipeline   (auto-finds newest JSON in src/calibration_output/)
 *
 * Controls:
 *   ESC   - Exit
 *   +/-   - Increase/decrease numDisparities (by 16)
 *   [/]   - Decrease/increase blockSize (by 2, must be odd)
 *   1/2/3 - Processing scale: 1.0x, 0.5x, 0.25x
 */

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <ctime>
#include <chrono>
#include <deque>
#include <mutex>
#include <vector>
#include <map>
#include <string>
#include <memory>
#include <algorithm>

#include <sys/mman.h>
#include <dirent.h>

#include <libcamera/libcamera.h>

#include <opencv2/core.hpp>
#include <opencv2/core/ocl.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/calib3d.hpp>

#include <QApplication>
#include <QWidget>
#include <QLabel>
#include <QTimer>
#include <QKeyEvent>
#include <QVBoxLayout>
#include <QFile>
#include <QJsonDocument>
#include <QJsonObject>
#include <QJsonArray>
#include <QDir>
#include <QFileInfoList>

/* ========================================================================
 * Configuration
 * ======================================================================== */

#define LEFT_CAMERA_ID      0
#define RIGHT_CAMERA_ID     1
#define CAMERA_WIDTH        1920
#define CAMERA_HEIGHT       1080
#define DISPLAY_SCALE       0.5
#define TIMER_INTERVAL_MS   33
#define FPS_WINDOW          30
#define FPS_PRINT_INTERVAL  2.0

/* ========================================================================
 * Camera capture (reused from calibration.cpp)
 * ======================================================================== */

struct MappedBuffer {
    void   *data;
    size_t  length;
};

struct CameraCapture {
    std::shared_ptr<libcamera::Camera>                camera;
    std::unique_ptr<libcamera::CameraConfiguration>   config;
    std::unique_ptr<libcamera::FrameBufferAllocator>  allocator;
    libcamera::Stream                                *stream;
    std::vector<std::unique_ptr<libcamera::Request>>  requests;
    std::map<const libcamera::FrameBuffer *, MappedBuffer> mappings;

    int          width, height;
    unsigned int stride;

    std::mutex   frame_mutex;
    cv::Mat      frame;
    bool         new_frame;
};

static void process_request(libcamera::Request *request)
{
    if (request->status() == libcamera::Request::RequestCancelled)
        return;

    auto *cap = reinterpret_cast<CameraCapture *>(request->cookie());

    const libcamera::FrameBuffer *fb = request->findBuffer(cap->stream);
    if (fb) {
        auto it = cap->mappings.find(fb);
        if (it != cap->mappings.end()) {
            std::lock_guard<std::mutex> lock(cap->frame_mutex);
            cv::Mat tmp(cap->height, cap->width, CV_8UC3,
                        it->second.data, (size_t)cap->stride);
            tmp.copyTo(cap->frame);
            cap->new_frame = true;
        }
    }

    request->reuse(libcamera::Request::ReuseBuffers);
    cap->camera->queueRequest(request);
}

static bool init_camera(CameraCapture *cap,
                        std::shared_ptr<libcamera::CameraManager> cm,
                        int camera_idx, int width, int height)
{
    auto cameras = cm->cameras();
    if (camera_idx >= (int)cameras.size()) {
        fprintf(stderr, "Camera %d not found (%zu available)\n",
                camera_idx, cameras.size());
        return false;
    }

    cap->camera = cm->get(cameras[camera_idx]->id());
    if (!cap->camera || cap->camera->acquire()) {
        fprintf(stderr, "Failed to acquire camera %d\n", camera_idx);
        return false;
    }

    cap->config = cap->camera->generateConfiguration(
        { libcamera::StreamRole::VideoRecording });
    if (!cap->config || cap->config->empty()) {
        fprintf(stderr, "Failed to generate config for camera %d\n", camera_idx);
        return false;
    }

    libcamera::StreamConfiguration &sc = cap->config->at(0);
    sc.size        = libcamera::Size(width, height);
    sc.pixelFormat = libcamera::formats::BGR888;

    if (cap->camera->configure(cap->config.get()) != 0) {
        fprintf(stderr, "Failed to configure camera %d\n", camera_idx);
        return false;
    }

    cap->stream = sc.stream();
    cap->width  = sc.size.width;
    cap->height = sc.size.height;
    cap->stride = sc.stride;

    cap->allocator =
        std::make_unique<libcamera::FrameBufferAllocator>(cap->camera);
    if (cap->allocator->allocate(cap->stream) < 0) {
        fprintf(stderr, "Buffer allocation failed for camera %d\n", camera_idx);
        return false;
    }

    for (const auto &buf : cap->allocator->buffers(cap->stream)) {
        const auto &planes = buf->planes();
        void *mem = mmap(NULL, planes[0].length,
                         PROT_READ | PROT_WRITE, MAP_SHARED,
                         planes[0].fd.get(), 0);
        if (mem == MAP_FAILED) {
            perror("mmap");
            return false;
        }
        cap->mappings[buf.get()] = { mem, planes[0].length };
    }

    for (const auto &buf : cap->allocator->buffers(cap->stream)) {
        auto req = cap->camera->createRequest(
            reinterpret_cast<uint64_t>(cap));
        if (!req || req->addBuffer(cap->stream, buf.get()) != 0) {
            fprintf(stderr, "Request setup failed for camera %d\n", camera_idx);
            return false;
        }
        cap->requests.push_back(std::move(req));
    }

    cap->camera->requestCompleted.connect(process_request);

    if (cap->camera->start() != 0) {
        fprintf(stderr, "Failed to start camera %d\n", camera_idx);
        return false;
    }
    for (auto &req : cap->requests)
        cap->camera->queueRequest(req.get());

    cap->new_frame = false;
    printf("Camera %d: %dx%d BGR888 (stride %u)\n",
           camera_idx, cap->width, cap->height, cap->stride);
    return true;
}

static void cleanup_camera(CameraCapture *cap)
{
    if (!cap->camera)
        return;
    cap->camera->stop();
    for (auto &[fb, mb] : cap->mappings)
        munmap(mb.data, mb.length);
    cap->mappings.clear();
    cap->requests.clear();
    cap->allocator.reset();
    cap->config.reset();
    cap->camera->release();
    cap->camera.reset();
}

/* ========================================================================
 * Qt5 helpers
 * ======================================================================== */

static QPixmap mat_to_pixmap(const cv::Mat &bgr)
{
    cv::Mat rgb;
    cv::cvtColor(bgr, rgb, cv::COLOR_BGR2RGB);
    QImage img(rgb.data, rgb.cols, rgb.rows,
               (int)rgb.step, QImage::Format_RGB888);
    return QPixmap::fromImage(img.copy());
}

/* ========================================================================
 * JSON calibration loader
 * ======================================================================== */

static cv::Mat json_array_to_mat(const QJsonArray &arr)
{
    /* Handle 2D array: [[...], [...], ...] */
    if (arr.isEmpty())
        return cv::Mat();

    if (arr[0].isArray()) {
        int rows = arr.size();
        int cols = arr[0].toArray().size();
        cv::Mat m(rows, cols, CV_64F);
        for (int r = 0; r < rows; r++) {
            QJsonArray row = arr[r].toArray();
            for (int c = 0; c < cols; c++)
                m.at<double>(r, c) = row[c].toDouble();
        }
        return m;
    }

    /* Handle 1D array: [x, y, z, w] — treat as single-row matrix */
    int cols = arr.size();
    cv::Mat m(1, cols, CV_64F);
    for (int c = 0; c < cols; c++)
        m.at<double>(0, c) = arr[c].toDouble();
    return m;
}

struct CalibData {
    cv::Mat K_left, K_right;
    cv::Mat dist_left, dist_right;
    cv::Mat R, T;
    cv::Mat R1, R2, P1, P2, Q;
    cv::Size image_size;
    double baseline;
    double rms_error;
};

static bool load_calibration(const char *path, CalibData *out)
{
    QFile f(path);
    if (!f.open(QIODevice::ReadOnly)) {
        fprintf(stderr, "Cannot open calibration file: %s\n", path);
        return false;
    }

    QJsonParseError err;
    QJsonDocument doc = QJsonDocument::fromJson(f.readAll(), &err);
    f.close();

    if (doc.isNull()) {
        fprintf(stderr, "JSON parse error: %s\n",
                err.errorString().toUtf8().constData());
        return false;
    }

    QJsonObject obj = doc.object();

    out->K_left     = json_array_to_mat(obj["K_left"].toArray());
    out->K_right    = json_array_to_mat(obj["K_right"].toArray());
    out->dist_left  = json_array_to_mat(obj["dist_left"].toArray());
    out->dist_right = json_array_to_mat(obj["dist_right"].toArray());
    out->R          = json_array_to_mat(obj["R"].toArray());
    out->T          = json_array_to_mat(obj["T"].toArray());

    QJsonArray sz = obj["image_size"].toArray();
    out->image_size = cv::Size(sz[0].toInt(), sz[1].toInt());
    out->baseline   = obj["baseline"].toDouble();
    out->rms_error  = obj["rms_error"].toDouble();

    /* Validate essential matrices */
    if (out->K_left.empty() || out->K_right.empty() ||
        out->dist_left.empty() || out->dist_right.empty() ||
        out->R.empty() || out->T.empty()) {
        fprintf(stderr, "Calibration file missing required matrices\n");
        return false;
    }

    /* Load pre-saved rectification matrices from calibration.
     * Fall back to recomputing from R/T if not present in the JSON. */
    out->R1 = json_array_to_mat(obj["R1"].toArray());
    out->R2 = json_array_to_mat(obj["R2"].toArray());
    out->P1 = json_array_to_mat(obj["P1"].toArray());
    out->P2 = json_array_to_mat(obj["P2"].toArray());
    out->Q  = json_array_to_mat(obj["Q"].toArray());

    if (out->R1.empty() || out->R2.empty() ||
        out->P1.empty() || out->P2.empty() || out->Q.empty()) {
        printf("Rectification matrices not in JSON, recomputing from R/T...\n");
        cv::Rect roi_l, roi_r;
        cv::stereoRectify(out->K_left, out->dist_left,
                          out->K_right, out->dist_right,
                          out->image_size, out->R, out->T,
                          out->R1, out->R2, out->P1, out->P2, out->Q,
                          cv::CALIB_ZERO_DISPARITY, 0,
                          out->image_size, &roi_l, &roi_r);
    } else {
        printf("Rectification matrices loaded from calibration JSON\n");
    }

    printf("  T vector: [%.1f, %.1f, %.1f]\n",
           out->T.at<double>(0), out->T.at<double>(1), out->T.at<double>(2));
    printf("  P1 focal=%.1f  cx=%.1f  cy=%.1f\n",
           out->P1.at<double>(0,0), out->P1.at<double>(0,2),
           out->P1.at<double>(1,2));
    printf("  P2 focal=%.1f  cx=%.1f  cy=%.1f\n",
           out->P2.at<double>(0,0), out->P2.at<double>(0,2),
           out->P2.at<double>(1,2));

    return true;
}

/* Auto-find the newest calibration JSON in src/calibration_output/ */
static std::string find_newest_calibration(const char *dir)
{
    QDir d(dir);
    if (!d.exists())
        return "";

    QStringList filters;
    filters << "*.json";
    QFileInfoList list = d.entryInfoList(filters, QDir::Files, QDir::Name);
    if (list.isEmpty())
        return "";

    /* Filenames are timestamped, so last alphabetically = newest */
    return list.last().absoluteFilePath().toStdString();
}

/* ========================================================================
 * Main window
 * ======================================================================== */

class DepthWindow : public QWidget {
    Q_OBJECT

public:
    DepthWindow(std::shared_ptr<libcamera::CameraManager> cm,
                const CalibData &calib,
                QWidget *parent = nullptr);
    ~DepthWindow();

protected:
    void keyPressEvent(QKeyEvent *e) override;

private slots:
    void onTimer();

private:
    QLabel *view, *status_lbl;
    QTimer *timer;

    std::shared_ptr<libcamera::CameraManager> cm;
    CameraCapture left_cap, right_cap;
    bool cameras_ok;

    /* Rectification maps (cv::Mat — no OpenCL on Pi 5) */
    cv::Mat map_lx, map_ly, map_rx, map_ry;

    /* Stereo matcher */
    cv::Ptr<cv::StereoBM> stereo;
    int num_disparities;
    int block_size;

    /* Processing scale */
    double proc_scale;

    /* FPS tracking */
    std::deque<std::chrono::steady_clock::time_point> frame_times;
    std::chrono::steady_clock::time_point last_fps_print;
    double current_fps;

    /* Backend status string */
    std::string backend_status;

    /* Debug: show rectified views instead of disparity */
    bool show_rectified;

    void shutdownCameras();
    void rebuildStereo();
    void updateStatus();
};

/* ---- constructor ---- */

DepthWindow::DepthWindow(
    std::shared_ptr<libcamera::CameraManager> cm_in,
    const CalibData &calib,
    QWidget *parent)
    : QWidget(parent), cm(cm_in), cameras_ok(false),
      num_disparities(64), block_size(9),
      proc_scale(0.5), current_fps(0.0),
      show_rectified(false)
{
    setWindowTitle("Depth Pipeline");

    int dw = (int)(CAMERA_WIDTH  * DISPLAY_SCALE);
    int dh = (int)(CAMERA_HEIGHT * DISPLAY_SCALE);
    resize(dw + 20, dh + 80);

    auto *vbox = new QVBoxLayout(this);

    view = new QLabel("Initialising...");
    view->setFixedSize(dw, dh);
    view->setAlignment(Qt::AlignCenter);
    view->setStyleSheet("background: black; color: white;");
    vbox->addWidget(view);

    status_lbl = new QLabel("Initialising cameras...");
    status_lbl->setAlignment(Qt::AlignCenter);
    QFont font = status_lbl->font();
    font.setPointSize(12);
    status_lbl->setFont(font);
    vbox->addWidget(status_lbl);

    /* Report compute backend */
    if (cv::ocl::haveOpenCL()) {
        cv::ocl::setUseOpenCL(true);
        cv::ocl::Context ctx = cv::ocl::Context::getDefault();
        if (!ctx.empty()) {
            cv::ocl::Device dev = ctx.device(0);
            backend_status = "OpenCL: " + dev.name();
            printf("OpenCL enabled: %s\n", dev.name().c_str());
        } else {
            cv::ocl::setUseOpenCL(false);
            backend_status = "CPU (Cortex-A76 NEON)";
            printf("OpenCL context failed — using CPU+NEON\n");
        }
    } else {
        backend_status = "CPU (Cortex-A76 NEON)";
        printf("No OpenCL runtime — using CPU+NEON\n");
    }

    /* Precompute rectification maps */
    cv::initUndistortRectifyMap(
        calib.K_left, calib.dist_left, calib.R1, calib.P1,
        calib.image_size, CV_16SC2, map_lx, map_ly);
    cv::initUndistortRectifyMap(
        calib.K_right, calib.dist_right, calib.R2, calib.P2,
        calib.image_size, CV_16SC2, map_rx, map_ry);
    printf("Rectification maps precomputed (%dx%d)\n",
           calib.image_size.width, calib.image_size.height);

    /* Diagnostic: check map coordinate ranges */
    {
        cv::Mat map_x_float;
        /* map_lx is CV_16SC2 (interleaved x,y) — split and check */
        std::vector<cv::Mat> xy;
        cv::split(map_lx, xy);
        double mn, mx;
        cv::minMaxLoc(xy[0], &mn, &mx);
        printf("  Left map X range:  [%.0f, %.0f]  (image width=%d)\n",
               mn, mx, calib.image_size.width);
        cv::minMaxLoc(xy[1], &mn, &mx);
        printf("  Left map Y range:  [%.0f, %.0f]  (image height=%d)\n",
               mn, mx, calib.image_size.height);
    }

    /* Create stereo matcher with tuned defaults */
    rebuildStereo();

    /* Initialise cameras */
    bool ok_l = init_camera(&left_cap, cm, LEFT_CAMERA_ID,
                            CAMERA_WIDTH, CAMERA_HEIGHT);
    bool ok_r = init_camera(&right_cap, cm, RIGHT_CAMERA_ID,
                            CAMERA_WIDTH, CAMERA_HEIGHT);
    cameras_ok = ok_l && ok_r;

    if (!cameras_ok) {
        status_lbl->setText("Camera init failed - check terminal");
        return;
    }

    printf("\n============================================================\n");
    printf("DEPTH PIPELINE\n");
    printf("============================================================\n");
    printf("Resolution:      %dx%d\n", CAMERA_WIDTH, CAMERA_HEIGHT);
    printf("Baseline:        %.2f mm\n", calib.baseline);
    printf("Calib RMS error: %.4f px\n", calib.rms_error);
    printf("numDisparities:  %d\n", num_disparities);
    printf("blockSize:       %d\n", block_size);
    printf("Processing scale: %.2fx\n", proc_scale);
    printf("Backend:         %s\n", backend_status.c_str());
    printf("============================================================\n");
    printf("\nControls:\n");
    printf("  ESC   - Exit\n");
    printf("  +/-   - numDisparities +/- 16\n");
    printf("  [/]   - blockSize -/+ 2\n");
    printf("  1/2/3 - Scale: 1.0x / 0.5x / 0.25x\n");
    printf("  D     - Toggle debug view (rectified L/R)\n");
    printf("============================================================\n\n");

    last_fps_print = std::chrono::steady_clock::now();
    updateStatus();

    timer = new QTimer(this);
    connect(timer, &QTimer::timeout, this, &DepthWindow::onTimer);
    timer->start(TIMER_INTERVAL_MS);
}

/* ---- destructor ---- */

DepthWindow::~DepthWindow()
{
    shutdownCameras();
}

void DepthWindow::shutdownCameras()
{
    if (!cameras_ok)
        return;
    cameras_ok = false;
    if (timer) timer->stop();
    cleanup_camera(&left_cap);
    cleanup_camera(&right_cap);
}

/* ---- rebuild stereo matcher after parameter changes ---- */

void DepthWindow::rebuildStereo()
{
    stereo = cv::StereoBM::create(num_disparities, block_size);
    stereo->setPreFilterType(cv::StereoBM::PREFILTER_XSOBEL);
    stereo->setPreFilterCap(31);
    stereo->setUniquenessRatio(15);
    stereo->setSpeckleWindowSize(100);
    stereo->setSpeckleRange(32);
    stereo->setTextureThreshold(0);
    stereo->setMinDisparity(0);
}

/* ---- status bar ---- */

void DepthWindow::updateStatus()
{
    int proc_w = (int)(CAMERA_WIDTH  * proc_scale);
    int proc_h = (int)(CAMERA_HEIGHT * proc_scale);

    char buf[256];
    snprintf(buf, sizeof(buf),
             "FPS: %.1f  |  numDisp: %d  blockSize: %d  |  "
             "proc: %dx%d (%.0f%%)  |  %s",
             current_fps, num_disparities, block_size,
             proc_w, proc_h, proc_scale * 100.0,
             backend_status.c_str());
    status_lbl->setText(buf);
}

/* ---- key handling ---- */

void DepthWindow::keyPressEvent(QKeyEvent *e)
{
    if (e->key() == Qt::Key_Escape) {
        close();
        return;
    }

    bool changed = false;

    if (e->key() == Qt::Key_Plus || e->key() == Qt::Key_Equal) {
        num_disparities = std::min(num_disparities + 16, 512);
        changed = true;
    }
    else if (e->key() == Qt::Key_Minus) {
        num_disparities = std::max(num_disparities - 16, 16);
        changed = true;
    }
    else if (e->key() == Qt::Key_BracketRight) {
        block_size = std::min(block_size + 2, 51);
        changed = true;
    }
    else if (e->key() == Qt::Key_BracketLeft) {
        block_size = std::max(block_size - 2, 5);
        changed = true;
    }
    else if (e->key() == Qt::Key_1) {
        proc_scale = 1.0;
        changed = true;
    }
    else if (e->key() == Qt::Key_2) {
        proc_scale = 0.5;
        changed = true;
    }
    else if (e->key() == Qt::Key_3) {
        proc_scale = 0.25;
        changed = true;
    }
    else if (e->key() == Qt::Key_D) {
        show_rectified = !show_rectified;
        printf("Debug view: %s\n", show_rectified ? "rectified L/R" : "disparity");
    }

    if (changed) {
        rebuildStereo();
        printf("numDisparities=%d  blockSize=%d  scale=%.2f\n",
               num_disparities, block_size, proc_scale);
        updateStatus();
    }

    QWidget::keyPressEvent(e);
}

/* ---- periodic frame update ---- */

void DepthWindow::onTimer()
{
    if (!cameras_ok)
        return;

    /* 1. Grab L/R frames under mutex */
    cv::Mat frame_l, frame_r;
    {
        std::lock_guard<std::mutex> lock(left_cap.frame_mutex);
        if (left_cap.new_frame) {
            frame_l = left_cap.frame.clone();
            left_cap.new_frame = false;
        }
    }
    {
        std::lock_guard<std::mutex> lock(right_cap.frame_mutex);
        if (right_cap.new_frame) {
            frame_r = right_cap.frame.clone();
            right_cap.new_frame = false;
        }
    }

    if (frame_l.empty() || frame_r.empty())
        return;

    /* 2. Convert to grayscale */
    cv::Mat gray_l, gray_r;
    cv::cvtColor(frame_l, gray_l, cv::COLOR_BGR2GRAY);
    cv::cvtColor(frame_r, gray_r, cv::COLOR_BGR2GRAY);

    /* 3. Rectify at full resolution (CV_16SC2 maps = fastest remap) */
    cv::Mat rect_l, rect_r;
    cv::remap(gray_l, rect_l, map_lx, map_ly, cv::INTER_LINEAR);
    cv::remap(gray_r, rect_r, map_rx, map_ry, cv::INTER_LINEAR);

    cv::Mat output;

    if (show_rectified) {
        /* Debug view: rectified L/R side by side with epipolar lines */
        cv::Mat rect_l_color, rect_r_color;
        cv::cvtColor(rect_l, rect_l_color, cv::COLOR_GRAY2BGR);
        cv::cvtColor(rect_r, rect_r_color, cv::COLOR_GRAY2BGR);

        /* Draw horizontal epipolar lines every 40 pixels */
        for (int y = 0; y < rect_l_color.rows; y += 40) {
            cv::line(rect_l_color, cv::Point(0, y),
                     cv::Point(rect_l_color.cols, y),
                     cv::Scalar(0, 255, 0), 1);
            cv::line(rect_r_color, cv::Point(0, y),
                     cv::Point(rect_r_color.cols, y),
                     cv::Scalar(0, 255, 0), 1);
        }

        cv::putText(rect_l_color, "LEFT (rectified)",
                    cv::Point(10, 30), cv::FONT_HERSHEY_SIMPLEX,
                    1.0, cv::Scalar(0, 255, 0), 2);
        cv::putText(rect_r_color, "RIGHT (rectified)",
                    cv::Point(10, 30), cv::FONT_HERSHEY_SIMPLEX,
                    1.0, cv::Scalar(0, 255, 0), 2);

        cv::hconcat(rect_l_color, rect_r_color, output);
        /* Scale side-by-side to fit display width */
        double side_scale = DISPLAY_SCALE * 0.5;
        cv::resize(output, output, cv::Size(), side_scale, side_scale,
                   cv::INTER_LINEAR);
    } else {
        /* 4. Resize for stereo matching if scale < 1.0 */
        cv::Mat proc_l, proc_r;
        if (proc_scale < 1.0) {
            cv::resize(rect_l, proc_l, cv::Size(), proc_scale, proc_scale,
                       cv::INTER_AREA);
            cv::resize(rect_r, proc_r, cv::Size(), proc_scale, proc_scale,
                       cv::INTER_AREA);
        } else {
            proc_l = rect_l;
            proc_r = rect_r;
        }

        /* 5. Stereo matching — output is CV_16S, disparities scaled by 16 */
        cv::Mat disp_raw;
        stereo->compute(proc_l, proc_r, disp_raw);

        /* 6. Convert fixed-point to float and clamp invalid pixels */
        cv::Mat disp_float;
        disp_raw.convertTo(disp_float, CV_32F, 1.0 / 16.0);
        cv::threshold(disp_float, disp_float, 0.0, 0.0, cv::THRESH_TOZERO);

        /* 7. Normalize to 0-255 using the known disparity range */
        cv::Mat disp8;
        double max_disp = (double)num_disparities;
        disp_float.convertTo(disp8, CV_8U, 255.0 / max_disp);

        /* 8. Apply colormap and mask invalid to black */
        cv::Mat color;
        cv::applyColorMap(disp8, color, cv::COLORMAP_JET);
        color.setTo(cv::Scalar(0, 0, 0), (disp_raw <= 0));

        output = color;
        cv::resize(output, output, cv::Size(),
                   DISPLAY_SCALE / proc_scale, DISPLAY_SCALE / proc_scale,
                   cv::INTER_LINEAR);
    }

    /* FPS calculation */
    auto now = std::chrono::steady_clock::now();
    frame_times.push_back(now);
    while ((int)frame_times.size() > FPS_WINDOW)
        frame_times.pop_front();

    if (frame_times.size() >= 2) {
        double elapsed = std::chrono::duration<double>(
            frame_times.back() - frame_times.front()).count();
        current_fps = (double)(frame_times.size() - 1) / elapsed;
    }

    /* Overlay FPS text */
    char fps_text[64];
    snprintf(fps_text, sizeof(fps_text), "FPS: %.1f", current_fps);
    cv::putText(output, fps_text, cv::Point(10, 30),
                cv::FONT_HERSHEY_SIMPLEX, 0.7,
                cv::Scalar(255, 255, 255), 2);

    /* Print FPS to terminal periodically */
    double since_print = std::chrono::duration<double>(
        now - last_fps_print).count();
    if (since_print >= FPS_PRINT_INTERVAL) {
        int proc_w = (int)(CAMERA_WIDTH  * proc_scale);
        int proc_h = (int)(CAMERA_HEIGHT * proc_scale);
        printf("FPS: %.1f  |  numDisp=%d  block=%d  proc=%dx%d\n",
               current_fps, num_disparities, block_size,
               proc_w, proc_h);
        last_fps_print = now;
    }

    view->setPixmap(mat_to_pixmap(output));

    updateStatus();
}

/* ========================================================================
 * main
 * ======================================================================== */

int main(int argc, char *argv[])
{
    QApplication app(argc, argv);

    /* Determine calibration file path */
    std::string calib_path;
    if (argc >= 2) {
        calib_path = argv[1];
    } else {
        /* Try to auto-find newest JSON */
        calib_path = find_newest_calibration("src/calibration_output");
        if (calib_path.empty())
            calib_path = find_newest_calibration("calibration_output");
        if (calib_path.empty())
            calib_path = find_newest_calibration("calibration");

        if (calib_path.empty()) {
            fprintf(stderr,
                    "Usage: %s <calibration.json>\n"
                    "No calibration file found in default locations.\n",
                    argv[0]);
            return 1;
        }
        printf("Auto-selected calibration: %s\n", calib_path.c_str());
    }

    /* Load calibration */
    CalibData calib;
    if (!load_calibration(calib_path.c_str(), &calib)) {
        fprintf(stderr, "Failed to load calibration from: %s\n",
                calib_path.c_str());
        return 1;
    }

    printf("Calibration loaded: %dx%d  baseline=%.2f mm  rms=%.4f px\n",
           calib.image_size.width, calib.image_size.height,
           calib.baseline, calib.rms_error);

    /* Start camera manager */
    auto cm = std::make_shared<libcamera::CameraManager>();
    if (cm->start()) {
        fprintf(stderr, "Failed to start camera manager\n");
        return 1;
    }

    printf("Found %zu camera(s)\n", cm->cameras().size());

    if (cm->cameras().size() < 2) {
        fprintf(stderr, "Need at least 2 cameras, found %zu\n",
                cm->cameras().size());
        cm->stop();
        return 1;
    }

    {
        DepthWindow win(cm, calib);
        win.show();
        app.exec();
    }

    cm->stop();
    return 0;
}

#include "depth_pipeline.moc"
