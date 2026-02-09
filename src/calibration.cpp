/*
 * Stereo Camera Calibration
 *
 * Port of calibration.py to C++ using libcamera, OpenCV, and Qt5.
 * (C++ required by library APIs; written in procedural style.)
 *
 * Build:
 *   mkdir build && cd build && cmake .. && make
 *
 * Dependencies (Raspberry Pi OS):
 *   sudo apt install libcamera-dev libopencv-dev qtbase5-dev cmake
 *
 * Controls:
 *   C   - Capture calibration image pair
 *   Q   - Quit capture and run calibration
 *   T   - Test rectification (after calibration)
 *   ESC - Exit
 */

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <ctime>
#include <mutex>
#include <vector>
#include <map>
#include <string>
#include <memory>

#include <sys/mman.h>
#include <sys/stat.h>

#include <libcamera/libcamera.h>

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/calib3d.hpp>

#include <QApplication>
#include <QWidget>
#include <QLabel>
#include <QTimer>
#include <QKeyEvent>
#include <QVBoxLayout>
#include <QHBoxLayout>

/* ========================================================================
 * Configuration
 * ======================================================================== */

#define CHECKERBOARD_W      9
#define CHECKERBOARD_H      6
#define SQUARE_SIZE_MM      25.0
#define LEFT_CAMERA_ID      0
#define RIGHT_CAMERA_ID     1
#define CAMERA_WIDTH        1920
#define CAMERA_HEIGHT       1080
#define MIN_CALIB_IMAGES    15
#define DISPLAY_SCALE       0.5
#define DETECT_EVERY_N      4
#define TIMER_INTERVAL_MS   33

static const cv::Size BOARD_SIZE(CHECKERBOARD_W, CHECKERBOARD_H);

/* ========================================================================
 * Camera capture
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

/* Called on libcamera's internal thread when a frame is ready.
 * The CameraCapture pointer is recovered from the request cookie. */
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

    /* Allocate frame buffers */
    cap->allocator =
        std::make_unique<libcamera::FrameBufferAllocator>(cap->camera);
    if (cap->allocator->allocate(cap->stream) < 0) {
        fprintf(stderr, "Buffer allocation failed for camera %d\n", camera_idx);
        return false;
    }

    /* Memory-map all buffers */
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

    /* Create requests and attach buffers (cookie carries cap pointer) */
    for (const auto &buf : cap->allocator->buffers(cap->stream)) {
        auto req = cap->camera->createRequest(
            reinterpret_cast<uint64_t>(cap));
        if (!req || req->addBuffer(cap->stream, buf.get()) != 0) {
            fprintf(stderr, "Request setup failed for camera %d\n", camera_idx);
            return false;
        }
        cap->requests.push_back(std::move(req));
    }

    /* Connect completion callback (plain function pointer — called
       directly on the camera thread, no Object dispatch needed) */
    cap->camera->requestCompleted.connect(process_request);

    /* Start streaming and queue all requests */
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
 * Computer-vision helpers
 * ======================================================================== */

static std::vector<cv::Point3f> make_object_points(void)
{
    std::vector<cv::Point3f> pts;
    pts.reserve(CHECKERBOARD_W * CHECKERBOARD_H);
    for (int r = 0; r < CHECKERBOARD_H; r++)
        for (int c = 0; c < CHECKERBOARD_W; c++)
            pts.emplace_back((float)(c * SQUARE_SIZE_MM),
                             (float)(r * SQUARE_SIZE_MM), 0.0f);
    return pts;
}

static bool find_corners(const cv::Mat &gray,
                         std::vector<cv::Point2f> &corners)
{
    int flags = cv::CALIB_CB_ADAPTIVE_THRESH
              | cv::CALIB_CB_NORMALIZE_IMAGE
              | cv::CALIB_CB_FAST_CHECK;
    if (!cv::findChessboardCorners(gray, BOARD_SIZE, corners, flags))
        return false;

    cv::TermCriteria crit(cv::TermCriteria::EPS | cv::TermCriteria::MAX_ITER,
                          30, 0.001);
    cv::cornerSubPix(gray, corners, cv::Size(11, 11), cv::Size(-1, -1), crit);
    return true;
}

/* ========================================================================
 * Stereo calibration
 * ======================================================================== */

struct CalibrationResult {
    cv::Mat K_left, K_right;
    cv::Mat dist_left, dist_right;
    cv::Mat R, T, E, F;
    cv::Mat R1, R2, P1, P2, Q;
    cv::Rect roi_left, roi_right;
    cv::Size image_size;
    double baseline;
    double rms_error;
};

static bool calibrate_stereo(
    const std::vector<std::vector<cv::Point3f>> &obj_points_in,
    const std::vector<std::vector<cv::Point2f>> &img_left_in,
    const std::vector<std::vector<cv::Point2f>> &img_right_in,
    cv::Size image_size,
    CalibrationResult *out)
{
    printf("\n============================================================\n");
    printf("RUNNING CALIBRATION\n");
    printf("============================================================\n");

    /* Mutable copies so we can filter outlier pairs */
    auto obj_points = obj_points_in;
    auto img_left   = img_left_in;
    auto img_right  = img_right_in;

    cv::Mat K_l, dist_l, K_r, dist_r;
    std::vector<cv::Mat> rvecs_l, tvecs_l, rvecs_r, tvecs_r;

    printf("\n1. Calibrating left camera (initial)...\n");
    double rms_l = cv::calibrateCamera(obj_points, img_left, image_size,
                                       K_l, dist_l, rvecs_l, tvecs_l);
    printf("   Left RMS reprojection error: %.4f px\n", rms_l);

    printf("\n2. Calibrating right camera (initial)...\n");
    double rms_r = cv::calibrateCamera(obj_points, img_right, image_size,
                                       K_r, dist_r, rvecs_r, tvecs_r);
    printf("   Right RMS reprojection error: %.4f px\n", rms_r);

    /* --- Filter outlier image pairs by per-image reprojection error --- */
    printf("\n   Per-image reprojection errors:\n");
    const double MAX_PER_IMAGE_ERR = 1.0;
    int n = (int)obj_points.size();
    std::vector<bool> keep(n, true);
    int removed = 0;

    for (int i = 0; i < n; i++) {
        std::vector<cv::Point2f> proj_l, proj_r;
        cv::projectPoints(obj_points[i], rvecs_l[i], tvecs_l[i],
                          K_l, dist_l, proj_l);
        cv::projectPoints(obj_points[i], rvecs_r[i], tvecs_r[i],
                          K_r, dist_r, proj_r);

        double err_l = cv::norm(img_left[i], proj_l, cv::NORM_L2)
                       / sqrt((double)proj_l.size());
        double err_r = cv::norm(img_right[i], proj_r, cv::NORM_L2)
                       / sqrt((double)proj_r.size());

        printf("   Pair %2d: L=%.3f px  R=%.3f px", i, err_l, err_r);
        if (err_l > MAX_PER_IMAGE_ERR || err_r > MAX_PER_IMAGE_ERR) {
            keep[i] = false;
            removed++;
            printf("  [REMOVED]\n");
        } else {
            printf("\n");
        }
    }

    if (removed > 0) {
        printf("   Removed %d outlier pair(s), %d remaining\n",
               removed, n - removed);

        std::vector<std::vector<cv::Point3f>> obj_f;
        std::vector<std::vector<cv::Point2f>> left_f, right_f;
        for (int i = 0; i < n; i++) {
            if (keep[i]) {
                obj_f.push_back(obj_points[i]);
                left_f.push_back(img_left[i]);
                right_f.push_back(img_right[i]);
            }
        }
        obj_points = obj_f;
        img_left   = left_f;
        img_right  = right_f;

        if ((int)obj_points.size() < MIN_CALIB_IMAGES) {
            printf("   ERROR: Only %d pairs remain after filtering (need %d)!\n",
                   (int)obj_points.size(), MIN_CALIB_IMAGES);
            return false;
        }
    }

    printf("\n3. Stereo calibration (joint optimisation)...\n");
    printf("   Using SAME_FOCAL_LENGTH constraint (identical cameras)\n");
    cv::Mat R, T, E, F;
    cv::TermCriteria crit(cv::TermCriteria::EPS | cv::TermCriteria::MAX_ITER,
                          100, 1e-5);
    double rms_s = cv::stereoCalibrate(
        obj_points, img_left, img_right,
        K_l, dist_l, K_r, dist_r,
        image_size, R, T, E, F,
        cv::CALIB_SAME_FOCAL_LENGTH, crit);
    printf("   Stereo RMS error: %.4f px\n", rms_s);

    /* --- Sanity checks --- */
    double fl = K_l.at<double>(0, 0);
    double fr = K_r.at<double>(0, 0);
    double baseline = cv::norm(T);
    double tx = T.at<double>(0), ty = T.at<double>(1), tz = T.at<double>(2);

    printf("\n   Sanity checks:\n");
    printf("   Focal lengths: L=%.1f px  R=%.1f px\n", fl, fr);
    if (fabs(fl - fr) / fmax(fl, fr) > 0.05)
        printf("   WARNING: Focal lengths differ by >5%%!\n");

    printf("   T = [%.1f, %.1f, %.1f] mm  baseline=%.1f mm\n",
           tx, ty, tz, baseline);
    if (fabs(ty) / baseline > 0.15 || fabs(tz) / baseline > 0.15)
        printf("   WARNING: Large Y/Z in T vector; check camera alignment.\n");

    printf("\n4. Computing rectification transforms...\n");
    cv::Mat R1, R2, P1, P2, Q;
    cv::Rect roi_l, roi_r;
    cv::stereoRectify(K_l, dist_l, K_r, dist_r, image_size, R, T,
                      R1, R2, P1, P2, Q,
                      cv::CALIB_ZERO_DISPARITY, 0,
                      image_size, &roi_l, &roi_r);

    printf("   T vector: [%.1f, %.1f, %.1f] mm\n", tx, ty, tz);
    printf("   P1 focal=%.1f  cx=%.1f  cy=%.1f\n",
           P1.at<double>(0,0), P1.at<double>(0,2), P1.at<double>(1,2));
    printf("   P2 focal=%.1f  cx=%.1f  cy=%.1f\n",
           P2.at<double>(0,0), P2.at<double>(0,2), P2.at<double>(1,2));
    printf("   ROI L: %dx%d+%d+%d  ROI R: %dx%d+%d+%d\n",
           roi_l.width, roi_l.height, roi_l.x, roi_l.y,
           roi_r.width, roi_r.height, roi_r.x, roi_r.y);

    if (roi_l.area() == 0 || roi_r.area() == 0)
        printf("   WARNING: Empty ROI -- rectification may be unreliable.\n");

    printf("\n============================================================\n");
    printf("CALIBRATION RESULTS\n");
    printf("============================================================\n");
    printf("Baseline:              %.2f mm\n", baseline);
    printf("Left focal length:     %.1f px\n", K_l.at<double>(0, 0));
    printf("Right focal length:    %.1f px\n", K_r.at<double>(0, 0));
    printf("Left principal point:  (%.1f, %.1f)\n",
           K_l.at<double>(0, 2), K_l.at<double>(1, 2));
    printf("Right principal point: (%.1f, %.1f)\n",
           K_r.at<double>(0, 2), K_r.at<double>(1, 2));
    printf("============================================================\n");

    out->K_left     = K_l;    out->K_right    = K_r;
    out->dist_left  = dist_l; out->dist_right = dist_r;
    out->R = R; out->T = T; out->E = E; out->F = F;
    out->R1 = R1; out->R2 = R2; out->P1 = P1; out->P2 = P2; out->Q = Q;
    out->roi_left   = roi_l;  out->roi_right  = roi_r;
    out->image_size = image_size;
    out->baseline   = baseline;
    out->rms_error  = rms_s;
    return true;
}

/* ========================================================================
 * JSON output (compatible with Python json.load)
 * ======================================================================== */

static void write_mat_json(FILE *f, const char *name, const cv::Mat &m,
                           bool comma)
{
    fprintf(f, "  \"%s\": [\n", name);
    for (int r = 0; r < m.rows; r++) {
        fprintf(f, "    [");
        for (int c = 0; c < m.cols; c++) {
            fprintf(f, "%.10g", m.at<double>(r, c));
            if (c < m.cols - 1) fprintf(f, ", ");
        }
        fprintf(f, "]%s\n", (r < m.rows - 1) ? "," : "");
    }
    fprintf(f, "  ]%s\n", comma ? "," : "");
}

static void save_calibration_json(const char *path, const CalibrationResult *c)
{
    FILE *f = fopen(path, "w");
    if (!f) { perror("fopen"); return; }

    fprintf(f, "{\n");
    write_mat_json(f, "K_left",     c->K_left,     true);
    write_mat_json(f, "K_right",    c->K_right,    true);
    write_mat_json(f, "dist_left",  c->dist_left,  true);
    write_mat_json(f, "dist_right", c->dist_right, true);
    write_mat_json(f, "R",          c->R,          true);
    write_mat_json(f, "T",          c->T,          true);
    write_mat_json(f, "E",          c->E,          true);
    write_mat_json(f, "F",          c->F,          true);
    write_mat_json(f, "R1",         c->R1,         true);
    write_mat_json(f, "R2",         c->R2,         true);
    write_mat_json(f, "P1",         c->P1,         true);
    write_mat_json(f, "P2",         c->P2,         true);
    write_mat_json(f, "Q",          c->Q,          true);

    fprintf(f, "  \"roi_left\": [%d, %d, %d, %d],\n",
            c->roi_left.x, c->roi_left.y,
            c->roi_left.width, c->roi_left.height);
    fprintf(f, "  \"roi_right\": [%d, %d, %d, %d],\n",
            c->roi_right.x, c->roi_right.y,
            c->roi_right.width, c->roi_right.height);
    fprintf(f, "  \"image_size\": [%d, %d],\n",
            c->image_size.width, c->image_size.height);
    fprintf(f, "  \"baseline\": %.6f,\n", c->baseline);
    fprintf(f, "  \"rms_error\": %.6f\n", c->rms_error);
    fprintf(f, "}\n");

    fclose(f);
    printf("\nCalibration saved to: %s\n", path);
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
 * Main window
 * ======================================================================== */

class CalibrationWindow : public QWidget {
    Q_OBJECT

public:
    CalibrationWindow(std::shared_ptr<libcamera::CameraManager> cm,
                      QWidget *parent = nullptr);
    ~CalibrationWindow();

protected:
    void keyPressEvent(QKeyEvent *e) override;

private slots:
    void onTimer();

private:
    QLabel *left_view, *right_view, *status_lbl;
    QTimer *timer;

    std::shared_ptr<libcamera::CameraManager> cm;
    CameraCapture left_cap, right_cap;
    bool cameras_ok;

    enum { ST_CAPTURE, ST_RECTIFY } state;

    std::vector<cv::Point3f> objp;
    std::vector<std::vector<cv::Point3f>> obj_points;
    std::vector<std::vector<cv::Point2f>> img_pts_l, img_pts_r;
    int capture_count;

    CalibrationResult calib;
    bool calibrated;

    cv::Mat map_lx, map_ly, map_rx, map_ry;

    /* Corner detection (runs every Nth frame on downscaled images) */
    int frame_count;
    bool corners_found_l, corners_found_r;
    std::vector<cv::Point2f> corners_l, corners_r;

    /* Latest full-res frames (kept for sub-pixel refinement on capture) */
    cv::Mat last_frame_l, last_frame_r;

    void shutdownCameras();
};

/* ---- constructor ---- */

CalibrationWindow::CalibrationWindow(
    std::shared_ptr<libcamera::CameraManager> cm_in, QWidget *parent)
    : QWidget(parent), cm(cm_in), cameras_ok(false),
      state(ST_CAPTURE), capture_count(0), calibrated(false),
      frame_count(0), corners_found_l(false), corners_found_r(false)
{
    setWindowTitle("Stereo Calibration");

    int dw = (int)(CAMERA_WIDTH  * DISPLAY_SCALE);
    int dh = (int)(CAMERA_HEIGHT * DISPLAY_SCALE);
    resize(dw * 2 + 20, dh + 80);

    auto *vbox = new QVBoxLayout(this);
    auto *hbox = new QHBoxLayout();

    left_view  = new QLabel("Left");
    right_view = new QLabel("Right");
    left_view->setFixedSize(dw, dh);
    right_view->setFixedSize(dw, dh);
    left_view->setAlignment(Qt::AlignCenter);
    right_view->setAlignment(Qt::AlignCenter);
    left_view->setStyleSheet("background: black; color: white;");
    right_view->setStyleSheet("background: black; color: white;");

    hbox->addWidget(left_view);
    hbox->addWidget(right_view);
    vbox->addLayout(hbox);

    status_lbl = new QLabel("Initialising cameras...");
    status_lbl->setAlignment(Qt::AlignCenter);
    QFont font = status_lbl->font();
    font.setPointSize(12);
    status_lbl->setFont(font);
    vbox->addWidget(status_lbl);

    objp = make_object_points();

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
    printf("STEREO CAMERA CALIBRATION\n");
    printf("============================================================\n");
    printf("Checkerboard: %dx%d internal corners\n",
           CHECKERBOARD_W, CHECKERBOARD_H);
    printf("Square size:  %.1f mm\n", SQUARE_SIZE_MM);
    printf("Resolution:   %dx%d\n", CAMERA_WIDTH, CAMERA_HEIGHT);
    printf("============================================================\n\n");

    status_lbl->setText(
        QString("Captured: 0/%1  |  C = capture   Q = calibrate   ESC = exit")
            .arg(MIN_CALIB_IMAGES));

    timer = new QTimer(this);
    connect(timer, &QTimer::timeout, this, &CalibrationWindow::onTimer);
    timer->start(TIMER_INTERVAL_MS);
}

/* ---- destructor ---- */

CalibrationWindow::~CalibrationWindow()
{
    shutdownCameras();
}

void CalibrationWindow::shutdownCameras()
{
    if (!cameras_ok)
        return;
    cameras_ok = false;
    if (timer) timer->stop();
    cleanup_camera(&left_cap);
    cleanup_camera(&right_cap);
}

/* ---- key handling ---- */

void CalibrationWindow::keyPressEvent(QKeyEvent *e)
{
    if (e->key() == Qt::Key_Escape) {
        close();
        return;
    }

    if (state == ST_CAPTURE) {
        if (e->key() == Qt::Key_C) {
            if (corners_found_l && corners_found_r
                && !last_frame_l.empty() && !last_frame_r.empty()) {
                /* Scale corners from display coords to full-res and
                   refine to sub-pixel accuracy on the full-res image */
                float inv = 1.0f / (float)DISPLAY_SCALE;
                std::vector<cv::Point2f> full_l(corners_l.size());
                std::vector<cv::Point2f> full_r(corners_r.size());
                for (size_t i = 0; i < corners_l.size(); i++) {
                    full_l[i] = corners_l[i] * inv;
                    full_r[i] = corners_r[i] * inv;
                }
                cv::Mat gray_l, gray_r;
                cv::cvtColor(last_frame_l, gray_l, cv::COLOR_BGR2GRAY);
                cv::cvtColor(last_frame_r, gray_r, cv::COLOR_BGR2GRAY);
                cv::TermCriteria crit(
                    cv::TermCriteria::EPS | cv::TermCriteria::MAX_ITER,
                    30, 0.001);
                cv::cornerSubPix(gray_l, full_l,
                                 cv::Size(11, 11), cv::Size(-1, -1), crit);
                cv::cornerSubPix(gray_r, full_r,
                                 cv::Size(11, 11), cv::Size(-1, -1), crit);

                obj_points.push_back(objp);
                img_pts_l.push_back(full_l);
                img_pts_r.push_back(full_r);
                capture_count++;
                printf("Captured pair %d/%d\n",
                       capture_count, MIN_CALIB_IMAGES);
                status_lbl->setText(
                    QString("Captured: %1/%2  |  C = capture   "
                            "Q = calibrate   ESC = exit")
                        .arg(capture_count).arg(MIN_CALIB_IMAGES));
            } else {
                printf("Checkerboard not detected in both cameras\n");
            }
        }
        else if (e->key() == Qt::Key_Q) {
            if (capture_count < MIN_CALIB_IMAGES) {
                printf("Need at least %d pairs (have %d)\n",
                       MIN_CALIB_IMAGES, capture_count);
                return;
            }
            status_lbl->setText("Calibrating... (see terminal)");
            status_lbl->repaint();
            QApplication::processEvents();

            cv::Size img_sz(CAMERA_WIDTH, CAMERA_HEIGHT);
            calibrated = calibrate_stereo(obj_points, img_pts_l, img_pts_r,
                                          img_sz, &calib);
            if (calibrated) {
                /* Save to timestamped JSON */
                mkdir("calibration_output", 0755);
                time_t now = time(NULL);
                struct tm *t = localtime(&now);
                char fname[256];
                snprintf(fname, sizeof(fname),
                         "calibration_output/stereo_calibration_"
                         "%04d%02d%02d_%02d%02d%02d.json",
                         t->tm_year + 1900, t->tm_mon + 1, t->tm_mday,
                         t->tm_hour, t->tm_min, t->tm_sec);
                save_calibration_json(fname, &calib);

                status_lbl->setText(
                    "Calibration complete!  T = test rectification   "
                    "ESC = exit");
            } else {
                status_lbl->setText("Calibration failed - see terminal");
            }
        }
    }
    else if (state == ST_RECTIFY) {
        /* ESC already handled above; no other keys in rectify mode */
    }

    if (e->key() == Qt::Key_T && calibrated && state != ST_RECTIFY) {
        printf("\nSwitching to rectification test...\n");
        printf("Epipolar lines should be horizontal and aligned\n");
        state = ST_RECTIFY;

        cv::initUndistortRectifyMap(
            calib.K_left, calib.dist_left, calib.R1, calib.P1,
            calib.image_size, CV_32FC1, map_lx, map_ly);
        cv::initUndistortRectifyMap(
            calib.K_right, calib.dist_right, calib.R2, calib.P2,
            calib.image_size, CV_32FC1, map_rx, map_ry);

        status_lbl->setText("Rectification test  |  ESC = exit");
    }

    QWidget::keyPressEvent(e);
}

/* ---- periodic frame update ---- */

void CalibrationWindow::onTimer()
{
    if (!cameras_ok)
        return;

    cv::Mat frame_l, frame_r;
    {
        std::lock_guard<std::mutex> lock(left_cap.frame_mutex);
        if (left_cap.new_frame) {
            left_cap.frame.copyTo(frame_l);
            left_cap.new_frame = false;
        }
    }
    {
        std::lock_guard<std::mutex> lock(right_cap.frame_mutex);
        if (right_cap.new_frame) {
            right_cap.frame.copyTo(frame_r);
            right_cap.new_frame = false;
        }
    }

    if (frame_l.empty() || frame_r.empty())
        return;

    if (state == ST_CAPTURE) {
        /* Keep full-res frames for sub-pixel refinement on capture */
        last_frame_l = frame_l;
        last_frame_r = frame_r;

        /* Downscale first — detection and display both use the small image */
        cv::Mat small_l, small_r;
        cv::resize(frame_l, small_l, cv::Size(), DISPLAY_SCALE, DISPLAY_SCALE);
        cv::resize(frame_r, small_r, cv::Size(), DISPLAY_SCALE, DISPLAY_SCALE);

        /* Run corner detection every Nth frame on the small images */
        if (++frame_count >= DETECT_EVERY_N) {
            frame_count = 0;
            cv::Mat gray_l, gray_r;
            cv::cvtColor(small_l, gray_l, cv::COLOR_BGR2GRAY);
            cv::cvtColor(small_r, gray_r, cv::COLOR_BGR2GRAY);

            corners_found_l = find_corners(gray_l, corners_l);
            corners_found_r = find_corners(gray_r, corners_r);
        }

        /* Draw corners on small images */
        cv::drawChessboardCorners(small_l, BOARD_SIZE,
                                  corners_l, corners_found_l);
        cv::drawChessboardCorners(small_r, BOARD_SIZE,
                                  corners_r, corners_found_r);

        /* Status overlay */
        const char *sl = corners_found_l ? "READY" : "NOT FOUND";
        const char *sr = corners_found_r ? "READY" : "NOT FOUND";
        cv::Scalar cl = corners_found_l
            ? cv::Scalar(0, 255, 0) : cv::Scalar(0, 0, 255);
        cv::Scalar cr = corners_found_r
            ? cv::Scalar(0, 255, 0) : cv::Scalar(0, 0, 255);

        cv::putText(small_l, std::string("Left: ") + sl,
                    cv::Point(10, 30), cv::FONT_HERSHEY_SIMPLEX, 1, cl, 2);
        cv::putText(small_r, std::string("Right: ") + sr,
                    cv::Point(10, 30), cv::FONT_HERSHEY_SIMPLEX, 1, cr, 2);

        char buf[64];
        snprintf(buf, sizeof(buf), "Captured: %d/%d",
                 capture_count, MIN_CALIB_IMAGES);
        cv::putText(small_l, buf, cv::Point(10, 70),
                    cv::FONT_HERSHEY_SIMPLEX, 1,
                    cv::Scalar(255, 255, 255), 2);

        left_view->setPixmap(mat_to_pixmap(small_l));
        right_view->setPixmap(mat_to_pixmap(small_r));
    }
    else if (state == ST_RECTIFY) {
        /* Apply rectification */
        cv::Mat rect_l, rect_r;
        cv::remap(frame_l, rect_l, map_lx, map_ly, cv::INTER_LINEAR);
        cv::remap(frame_r, rect_r, map_rx, map_ry, cv::INTER_LINEAR);

        /* Draw epipolar lines every 50 pixels */
        for (int y = 0; y < rect_l.rows; y += 50) {
            cv::line(rect_l, cv::Point(0, y),
                     cv::Point(rect_l.cols, y), cv::Scalar(0, 255, 0), 1);
            cv::line(rect_r, cv::Point(0, y),
                     cv::Point(rect_r.cols, y), cv::Scalar(0, 255, 0), 1);
        }

        cv::Mat sc_l, sc_r;
        cv::resize(rect_l, sc_l, cv::Size(), DISPLAY_SCALE, DISPLAY_SCALE);
        cv::resize(rect_r, sc_r, cv::Size(), DISPLAY_SCALE, DISPLAY_SCALE);

        left_view->setPixmap(mat_to_pixmap(sc_l));
        right_view->setPixmap(mat_to_pixmap(sc_r));
    }
}

/* ========================================================================
 * main
 * ======================================================================== */

int main(int argc, char *argv[])
{
    QApplication app(argc, argv);

    auto cm = std::make_shared<libcamera::CameraManager>();
    if (cm->start()) {
        fprintf(stderr, "Failed to start camera manager\n");
        return 1;
    }

    printf("Found %zu camera(s)\n", cm->cameras().size());

    {
        CalibrationWindow win(cm);
        win.show();
        app.exec();
    }   /* window destroyed here, cameras cleaned up */

    cm->stop();
    return 0;
}

#include "calibration.moc"
