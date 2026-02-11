/*
 * View Synthesis: DIBR (Depth-Image Based Rendering) via GPU Compute Shaders
 *
 * Synthesises a novel viewpoint from a virtual camera positioned halfway
 * between the two physical cameras using forward warping on the Pi 5's
 * VideoCore VII (OpenGL ES 3.1 compute shaders).
 *
 * Build:
 *   cd build && cmake .. && make view_synthesis
 *
 * Usage:
 *   ./view_synthesis <path-to-calibration.json>
 *   ./view_synthesis   (auto-finds newest JSON in src/calibration_output/)
 *
 * Controls:
 *   ESC   - Exit
 *   +/-   - Increase/decrease numDisparities (by 16)
 *   [/]   - Decrease/increase blockSize (by 2, must be odd)
 *   S     - Toggle StereoBM / StereoSGBM
 *   W     - Toggle WLS disparity filter
 *   1/2/3 - Processing scale: 1.0x, 0.5x, 0.25x
 *   H     - Toggle hole filling
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
#include <opencv2/imgproc.hpp>
#include <opencv2/calib3d.hpp>
#include <opencv2/ximgproc/disparity_filter.hpp>

#include <QApplication>
#include <QOpenGLWidget>
#include <QOpenGLFunctions>
#include <QTimer>
#include <QKeyEvent>
#include <QPainter>
#include <QFile>
#include <QJsonDocument>
#include <QJsonObject>
#include <QJsonArray>
#include <QDir>
#include <QFileInfoList>
#include <QSurfaceFormat>

#include <GLES3/gl31.h>

/* ========================================================================
 * Configuration
 * ======================================================================== */

#define LEFT_CAMERA_ID      0
#define RIGHT_CAMERA_ID     1
#define CAMERA_WIDTH        1920
#define CAMERA_HEIGHT       1080
#define TIMER_INTERVAL_MS   33
#define FPS_WINDOW          30
#define FPS_PRINT_INTERVAL  2.0
#define HOLE_FILL_MAX_SEARCH 64

/* ========================================================================
 * Camera capture (identical to depth_pipeline.cpp)
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
 * JSON calibration loader (identical to depth_pipeline.cpp)
 * ======================================================================== */

static cv::Mat json_array_to_mat(const QJsonArray &arr)
{
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

    if (out->K_left.empty() || out->K_right.empty() ||
        out->dist_left.empty() || out->dist_right.empty() ||
        out->R.empty() || out->T.empty()) {
        fprintf(stderr, "Calibration file missing required matrices\n");
        return false;
    }

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

    return list.last().absoluteFilePath().toStdString();
}

/* ========================================================================
 * GLSL Compute Shaders (OpenGL ES 3.1)
 * ======================================================================== */

static const char *DEPTH_SPLAT_CS = R"(#version 310 es
layout(local_size_x = 16, local_size_y = 16) in;

uniform sampler2D u_disparity;
uniform float u_shift;
uniform ivec2 u_output_size;
layout(std430, binding = 0) buffer DepthBuffer { uint depth[]; };

void main() {
    ivec2 src = ivec2(gl_GlobalInvocationID.xy);
    if (src.x >= u_output_size.x || src.y >= u_output_size.y) return;
    float disp = texelFetch(u_disparity, src, 0).r;
    if (disp < 0.5) return;
    int dst_x = int(round(float(src.x) + disp * u_shift));
    if (dst_x < 0 || dst_x >= u_output_size.x) return;
    uint depth_val = uint(disp * 256.0);
    atomicMax(depth[src.y * u_output_size.x + dst_x], depth_val);
}
)";

static const char *COLOR_SPLAT_CS = R"(#version 310 es
layout(local_size_x = 16, local_size_y = 16) in;

uniform sampler2D u_color;
uniform sampler2D u_disparity;
uniform float u_shift;
uniform ivec2 u_output_size;
layout(std430, binding = 0) readonly buffer DepthBuffer { uint depth[]; };
layout(rgba8, binding = 1) writeonly uniform highp image2D u_output;

void main() {
    ivec2 src = ivec2(gl_GlobalInvocationID.xy);
    if (src.x >= u_output_size.x || src.y >= u_output_size.y) return;
    float disp = texelFetch(u_disparity, src, 0).r;
    if (disp < 0.5) return;
    int dst_x = int(round(float(src.x) + disp * u_shift));
    if (dst_x < 0 || dst_x >= u_output_size.x) return;
    uint my_depth = uint(disp * 256.0);
    uint stored = depth[src.y * u_output_size.x + dst_x];
    if (my_depth >= stored) {
        vec4 col = texelFetch(u_color, src, 0);
        imageStore(u_output, ivec2(dst_x, src.y), col);
    }
}
)";

static const char *HOLE_FILL_CS = R"(#version 310 es
layout(local_size_x = 16, local_size_y = 16) in;

layout(std430, binding = 0) readonly buffer DepthBuffer { uint depth[]; };
layout(rgba8, binding = 1) readonly uniform highp image2D u_input;
layout(rgba8, binding = 2) writeonly uniform highp image2D u_filled;
uniform int u_max_search;
uniform ivec2 u_output_size;

void main() {
    ivec2 pos = ivec2(gl_GlobalInvocationID.xy);
    if (pos.x >= u_output_size.x || pos.y >= u_output_size.y) return;

    int w = u_output_size.x;
    uint d = depth[pos.y * w + pos.x];
    vec4 col = imageLoad(u_input, pos);
    if (d > 0u) {
        imageStore(u_filled, pos, col);
        return;
    }
    vec4 left_col = vec4(0.0);
    vec4 right_col = vec4(0.0);
    uint left_d = 0u;
    uint right_d = 0u;
    for (int i = 1; i <= u_max_search; i++) {
        if (left_d == 0u && pos.x - i >= 0) {
            uint ld = depth[pos.y * w + pos.x - i];
            if (ld > 0u) {
                left_d = ld;
                left_col = imageLoad(u_input, ivec2(pos.x - i, pos.y));
            }
        }
        if (right_d == 0u && pos.x + i < w) {
            uint rd = depth[pos.y * w + pos.x + i];
            if (rd > 0u) {
                right_d = rd;
                right_col = imageLoad(u_input, ivec2(pos.x + i, pos.y));
            }
        }
        if (left_d > 0u && right_d > 0u) break;
    }
    if (left_d > 0u && right_d > 0u)
        col = (left_d < right_d) ? left_col : right_col;
    else if (left_d > 0u) col = left_col;
    else if (right_d > 0u) col = right_col;
    imageStore(u_filled, pos, col);
}
)";

static const char *DISPLAY_VS = R"(#version 310 es
layout(location = 0) in vec2 a_pos;
layout(location = 1) in vec2 a_uv;
out vec2 v_uv;
void main() {
    v_uv = a_uv;
    gl_Position = vec4(a_pos, 0.0, 1.0);
}
)";

static const char *DISPLAY_FS = R"(#version 310 es
precision mediump float;
in vec2 v_uv;
out vec4 frag_color;
uniform sampler2D u_texture;
void main() {
    frag_color = texture(u_texture, v_uv);
}
)";

/* ========================================================================
 * GL helpers
 * ======================================================================== */

static GLuint compile_shader(GLenum type, const char *source)
{
    GLuint s = glCreateShader(type);
    glShaderSource(s, 1, &source, nullptr);
    glCompileShader(s);
    GLint ok;
    glGetShaderiv(s, GL_COMPILE_STATUS, &ok);
    if (!ok) {
        char log[1024];
        glGetShaderInfoLog(s, sizeof(log), nullptr, log);
        fprintf(stderr, "Shader compile error:\n%s\n", log);
        glDeleteShader(s);
        return 0;
    }
    return s;
}

static GLuint create_compute_program(const char *source)
{
    GLuint cs = compile_shader(GL_COMPUTE_SHADER, source);
    if (!cs) return 0;
    GLuint prog = glCreateProgram();
    glAttachShader(prog, cs);
    glLinkProgram(prog);
    glDeleteShader(cs);
    GLint ok;
    glGetProgramiv(prog, GL_LINK_STATUS, &ok);
    if (!ok) {
        char log[1024];
        glGetProgramInfoLog(prog, sizeof(log), nullptr, log);
        fprintf(stderr, "Program link error:\n%s\n", log);
        glDeleteProgram(prog);
        return 0;
    }
    return prog;
}

static GLuint create_render_program(const char *vs_src, const char *fs_src)
{
    GLuint vs = compile_shader(GL_VERTEX_SHADER, vs_src);
    GLuint fs = compile_shader(GL_FRAGMENT_SHADER, fs_src);
    if (!vs || !fs) return 0;
    GLuint prog = glCreateProgram();
    glAttachShader(prog, vs);
    glAttachShader(prog, fs);
    glLinkProgram(prog);
    glDeleteShader(vs);
    glDeleteShader(fs);
    GLint ok;
    glGetProgramiv(prog, GL_LINK_STATUS, &ok);
    if (!ok) {
        char log[1024];
        glGetProgramInfoLog(prog, sizeof(log), nullptr, log);
        fprintf(stderr, "Program link error:\n%s\n", log);
        glDeleteProgram(prog);
        return 0;
    }
    return prog;
}

static GLuint create_texture_rgba8(int w, int h)
{
    GLuint tex;
    glGenTextures(1, &tex);
    glBindTexture(GL_TEXTURE_2D, tex);
    glTexStorage2D(GL_TEXTURE_2D, 1, GL_RGBA8, w, h);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    return tex;
}

static GLuint create_texture_r32f(int w, int h)
{
    GLuint tex;
    glGenTextures(1, &tex);
    glBindTexture(GL_TEXTURE_2D, tex);
    glTexStorage2D(GL_TEXTURE_2D, 1, GL_R32F, w, h);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    return tex;
}

static GLuint create_ssbo(int num_elements)
{
    GLuint buf;
    glGenBuffers(1, &buf);
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, buf);
    glBufferData(GL_SHADER_STORAGE_BUFFER,
                 num_elements * sizeof(GLuint), nullptr, GL_DYNAMIC_DRAW);
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0);
    return buf;
}

/* ========================================================================
 * SynthWindow — QOpenGLWidget subclass
 * ======================================================================== */

class SynthWindow : public QOpenGLWidget {
    Q_OBJECT

public:
    SynthWindow(std::shared_ptr<libcamera::CameraManager> cm,
                const CalibData &calib,
                QWidget *parent = nullptr);
    ~SynthWindow();

protected:
    void initializeGL() override;
    void paintGL() override;
    void resizeGL(int w, int h) override;
    void keyPressEvent(QKeyEvent *e) override;

private:
    std::shared_ptr<libcamera::CameraManager> cm;
    CalibData calib;
    CameraCapture left_cap, right_cap;
    bool cameras_ok;

    /* Rectification maps */
    cv::Mat map_lx, map_ly, map_rx, map_ry;

    /* Stereo matcher */
    cv::Ptr<cv::StereoMatcher> stereo;
    cv::Ptr<cv::StereoMatcher> right_stereo;
    cv::Ptr<cv::ximgproc::DisparityWLSFilter> wls_filter;
    int num_disparities;
    int block_size;
    bool use_sgbm;
    bool use_wls;
    bool use_hole_fill;

    /* Processing scale */
    double proc_scale;
    int proc_w, proc_h;

    /* GPU resources */
    GLuint prog_depth_splat, prog_color_splat, prog_hole_fill, prog_display;
    GLuint tex_left_color, tex_right_color;
    GLuint tex_left_disp, tex_right_disp;
    GLuint tex_output, tex_filled;
    GLuint ssbo_depth;
    GLuint quad_vao, quad_vbo;
    bool gl_ready;

    /* Timer */
    QTimer *timer;

    /* FPS tracking */
    std::deque<std::chrono::steady_clock::time_point> frame_times;
    std::chrono::steady_clock::time_point last_fps_print;
    double current_fps;

    /* Staging buffers to avoid per-frame allocation */
    cv::Mat rgba_buf;
    std::vector<GLuint> clear_zeros;
    std::vector<GLubyte> clear_black;

    void shutdownCameras();
    void rebuildStereo();
    void recreateTextures();
    void clearGPUBuffers();
};

/* ---- constructor ---- */

SynthWindow::SynthWindow(
    std::shared_ptr<libcamera::CameraManager> cm_in,
    const CalibData &calib_in,
    QWidget *parent)
    : QOpenGLWidget(parent), cm(cm_in), calib(calib_in), cameras_ok(false),
      num_disparities(64), block_size(9),
      use_sgbm(true), use_wls(true), use_hole_fill(true),
      proc_scale(0.5),
      prog_depth_splat(0), prog_color_splat(0), prog_hole_fill(0), prog_display(0),
      tex_left_color(0), tex_right_color(0),
      tex_left_disp(0), tex_right_disp(0),
      tex_output(0), tex_filled(0), ssbo_depth(0),
      quad_vao(0), quad_vbo(0),
      gl_ready(false), timer(nullptr), current_fps(0.0)
{
    setWindowTitle("View Synthesis (DIBR)");

    proc_w = (int)(CAMERA_WIDTH  * proc_scale);
    proc_h = (int)(CAMERA_HEIGHT * proc_scale);

    /* Precompute rectification maps */
    cv::initUndistortRectifyMap(
        calib.K_left, calib.dist_left, calib.R1, calib.P1,
        calib.image_size, CV_16SC2, map_lx, map_ly);
    cv::initUndistortRectifyMap(
        calib.K_right, calib.dist_right, calib.R2, calib.P2,
        calib.image_size, CV_16SC2, map_rx, map_ry);
    printf("Rectification maps precomputed (%dx%d)\n",
           calib.image_size.width, calib.image_size.height);

    rebuildStereo();

    /* Initialise cameras */
    bool ok_l = init_camera(&left_cap, cm, LEFT_CAMERA_ID,
                            CAMERA_WIDTH, CAMERA_HEIGHT);
    bool ok_r = init_camera(&right_cap, cm, RIGHT_CAMERA_ID,
                            CAMERA_WIDTH, CAMERA_HEIGHT);
    cameras_ok = ok_l && ok_r;

    if (!cameras_ok) {
        fprintf(stderr, "Camera init failed\n");
        return;
    }

    printf("\n============================================================\n");
    printf("VIEW SYNTHESIS (DIBR)\n");
    printf("============================================================\n");
    printf("Resolution:      %dx%d\n", CAMERA_WIDTH, CAMERA_HEIGHT);
    printf("Processing:      %dx%d (%.0f%%)\n", proc_w, proc_h, proc_scale * 100.0);
    printf("Baseline:        %.2f mm\n", calib.baseline);
    printf("Calib RMS error: %.4f px\n", calib.rms_error);
    printf("numDisparities:  %d\n", num_disparities);
    printf("blockSize:       %d\n", block_size);
    printf("Matcher:         %s\n", use_sgbm ? "StereoSGBM" : "StereoBM");
    printf("WLS filter:      %s\n", use_wls ? "ON" : "OFF");
    printf("Hole filling:    %s\n", use_hole_fill ? "ON" : "OFF");
    printf("============================================================\n");
    printf("\nControls:\n");
    printf("  ESC   - Exit\n");
    printf("  +/-   - numDisparities +/- 16\n");
    printf("  [/]   - blockSize -/+ 2\n");
    printf("  S     - Toggle StereoBM / StereoSGBM\n");
    printf("  W     - Toggle WLS disparity filter\n");
    printf("  H     - Toggle hole filling\n");
    printf("  1/2/3 - Scale: 1.0x / 0.5x / 0.25x\n");
    printf("============================================================\n\n");

    last_fps_print = std::chrono::steady_clock::now();
}

/* ---- destructor ---- */

SynthWindow::~SynthWindow()
{
    shutdownCameras();
    makeCurrent();
    if (prog_depth_splat) glDeleteProgram(prog_depth_splat);
    if (prog_color_splat) glDeleteProgram(prog_color_splat);
    if (prog_hole_fill)   glDeleteProgram(prog_hole_fill);
    if (prog_display)     glDeleteProgram(prog_display);
    GLuint textures[] = { tex_left_color, tex_right_color,
                          tex_left_disp, tex_right_disp,
                          tex_output, tex_filled };
    for (auto t : textures)
        if (t) glDeleteTextures(1, &t);
    if (ssbo_depth) glDeleteBuffers(1, &ssbo_depth);
    if (quad_vbo) glDeleteBuffers(1, &quad_vbo);
    if (quad_vao) glDeleteVertexArrays(1, &quad_vao);
    doneCurrent();
}

void SynthWindow::shutdownCameras()
{
    if (!cameras_ok)
        return;
    cameras_ok = false;
    if (timer) timer->stop();
    cleanup_camera(&left_cap);
    cleanup_camera(&right_cap);
}

/* ---- rebuild stereo matcher ---- */

void SynthWindow::rebuildStereo()
{
    if (use_sgbm) {
        int P1 = 8 * block_size * block_size;
        int P2 = 32 * block_size * block_size;
        stereo = cv::StereoSGBM::create(
            0, num_disparities, block_size,
            P1, P2, -1, 31, 10, 100, 32,
            cv::StereoSGBM::MODE_SGBM_3WAY);
    } else {
        auto bm = cv::StereoBM::create(num_disparities, block_size);
        bm->setPreFilterType(cv::StereoBM::PREFILTER_XSOBEL);
        bm->setPreFilterCap(31);
        bm->setUniquenessRatio(15);
        bm->setSpeckleWindowSize(100);
        bm->setSpeckleRange(32);
        bm->setTextureThreshold(0);
        bm->setMinDisparity(0);
        stereo = bm;
    }

    right_stereo = cv::ximgproc::createRightMatcher(stereo);
    wls_filter = cv::ximgproc::createDisparityWLSFilter(stereo);
    wls_filter->setLambda(8000.0);
    wls_filter->setSigmaColor(1.5);
}

/* ---- recreate GPU textures at current proc resolution ---- */

void SynthWindow::recreateTextures()
{
    GLuint old[] = { tex_left_color, tex_right_color,
                     tex_left_disp, tex_right_disp,
                     tex_output, tex_filled };
    for (auto t : old)
        if (t) glDeleteTextures(1, &t);
    if (ssbo_depth) glDeleteBuffers(1, &ssbo_depth);

    tex_left_color  = create_texture_rgba8(proc_w, proc_h);
    tex_right_color = create_texture_rgba8(proc_w, proc_h);
    tex_left_disp   = create_texture_r32f(proc_w, proc_h);
    tex_right_disp  = create_texture_r32f(proc_w, proc_h);
    tex_output      = create_texture_rgba8(proc_w, proc_h);
    tex_filled      = create_texture_rgba8(proc_w, proc_h);
    ssbo_depth      = create_ssbo(proc_w * proc_h);

    clear_zeros.assign(proc_w * proc_h, 0);
    clear_black.assign(proc_w * proc_h * 4, 0);

    printf("GPU textures + SSBOs created: %dx%d\n", proc_w, proc_h);
}

/* ---- clear depth buffer and output ---- */

void SynthWindow::clearGPUBuffers()
{
    /* Clear depth SSBO to 0 */
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, ssbo_depth);
    glBufferSubData(GL_SHADER_STORAGE_BUFFER, 0,
                    proc_w * proc_h * sizeof(GLuint), clear_zeros.data());
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0);

    /* Clear output to black */
    glBindTexture(GL_TEXTURE_2D, tex_output);
    glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, proc_w, proc_h,
                    GL_RGBA, GL_UNSIGNED_BYTE, clear_black.data());
}

/* ---- GL initialisation ---- */

void SynthWindow::initializeGL()
{
    printf("OpenGL version: %s\n", glGetString(GL_VERSION));
    printf("GLSL version:   %s\n", glGetString(GL_SHADING_LANGUAGE_VERSION));
    printf("Renderer:       %s\n", glGetString(GL_RENDERER));

    /* Compile compute programs */
    prog_depth_splat = create_compute_program(DEPTH_SPLAT_CS);
    prog_color_splat = create_compute_program(COLOR_SPLAT_CS);
    prog_hole_fill   = create_compute_program(HOLE_FILL_CS);
    prog_display     = create_render_program(DISPLAY_VS, DISPLAY_FS);

    if (!prog_depth_splat || !prog_color_splat || !prog_hole_fill || !prog_display) {
        fprintf(stderr, "FATAL: shader compilation failed\n");
        return;
    }

    /* Create textures */
    recreateTextures();

    /* Create fullscreen quad VAO/VBO */
    float quad[] = {
        /* pos.x  pos.y  uv.s  uv.t  (V flipped: GL bottom → UV top) */
        -1.0f, -1.0f,  0.0f, 1.0f,
         1.0f, -1.0f,  1.0f, 1.0f,
        -1.0f,  1.0f,  0.0f, 0.0f,
         1.0f,  1.0f,  1.0f, 0.0f,
    };
    glGenVertexArrays(1, &quad_vao);
    glGenBuffers(1, &quad_vbo);
    glBindVertexArray(quad_vao);
    glBindBuffer(GL_ARRAY_BUFFER, quad_vbo);
    glBufferData(GL_ARRAY_BUFFER, sizeof(quad), quad, GL_STATIC_DRAW);
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 4 * sizeof(float), (void *)0);
    glEnableVertexAttribArray(1);
    glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 4 * sizeof(float),
                          (void *)(2 * sizeof(float)));
    glBindVertexArray(0);

    gl_ready = true;
    printf("GPU pipeline initialised\n");

    /* Start frame timer */
    timer = new QTimer(this);
    connect(timer, &QTimer::timeout, this, [this]() { update(); });
    timer->start(TIMER_INTERVAL_MS);
}

/* ---- key handling ---- */

void SynthWindow::keyPressEvent(QKeyEvent *e)
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
    else if (e->key() == Qt::Key_S) {
        use_sgbm = !use_sgbm;
        changed = true;
        printf("Matcher: %s\n", use_sgbm ? "StereoSGBM" : "StereoBM");
    }
    else if (e->key() == Qt::Key_W) {
        use_wls = !use_wls;
        printf("WLS filter: %s\n", use_wls ? "ON" : "OFF");
    }
    else if (e->key() == Qt::Key_H) {
        use_hole_fill = !use_hole_fill;
        printf("Hole filling: %s\n", use_hole_fill ? "ON" : "OFF");
    }

    if (changed) {
        int new_w = (int)(CAMERA_WIDTH  * proc_scale);
        int new_h = (int)(CAMERA_HEIGHT * proc_scale);
        if (new_w != proc_w || new_h != proc_h) {
            proc_w = new_w;
            proc_h = new_h;
            if (gl_ready) {
                makeCurrent();
                recreateTextures();
                doneCurrent();
            }
        }
        rebuildStereo();
        printf("%s  numDisp=%d  block=%d  scale=%.2f  WLS=%s  hole=%s\n",
               use_sgbm ? "SGBM" : "BM",
               num_disparities, block_size, proc_scale,
               use_wls ? "ON" : "OFF",
               use_hole_fill ? "ON" : "OFF");
    }

    QOpenGLWidget::keyPressEvent(e);
}

/* ---- resize ---- */

void SynthWindow::resizeGL(int w, int h)
{
    glViewport(0, 0, w, h);
}

/* ---- main render loop ---- */

void SynthWindow::paintGL()
{
    if (!gl_ready || !cameras_ok)
        return;

    /* 1. Grab L/R frames */
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

    /* 2. Rectify at full resolution */
    cv::Mat rect_l_bgr, rect_r_bgr;
    cv::remap(frame_l, rect_l_bgr, map_lx, map_ly, cv::INTER_LINEAR);
    cv::remap(frame_r, rect_r_bgr, map_rx, map_ry, cv::INTER_LINEAR);

    /* 3. Resize for processing */
    cv::Mat proc_l_bgr, proc_r_bgr;
    if (proc_scale < 1.0) {
        cv::resize(rect_l_bgr, proc_l_bgr, cv::Size(proc_w, proc_h), 0, 0,
                   cv::INTER_AREA);
        cv::resize(rect_r_bgr, proc_r_bgr, cv::Size(proc_w, proc_h), 0, 0,
                   cv::INTER_AREA);
    } else {
        proc_l_bgr = rect_l_bgr;
        proc_r_bgr = rect_r_bgr;
    }

    /* 4. Grayscale for stereo matching */
    cv::Mat gray_l, gray_r;
    cv::cvtColor(proc_l_bgr, gray_l, cv::COLOR_BGR2GRAY);
    cv::cvtColor(proc_r_bgr, gray_r, cv::COLOR_BGR2GRAY);

    /* 5. Stereo match — left disparity */
    cv::Mat disp_raw_l;
    stereo->compute(gray_l, gray_r, disp_raw_l);

    /* 6. Right disparity (always needed for DIBR) */
    cv::Mat disp_raw_r;
    right_stereo->compute(gray_r, gray_l, disp_raw_r);

    /* 7. Optional WLS filter on left disparity */
    if (use_wls && wls_filter) {
        cv::Mat disp_filtered;
        wls_filter->filter(disp_raw_l, gray_l, disp_filtered, disp_raw_r);
        disp_raw_l = disp_filtered;
    }

    /* 8. Convert left disparity to float */
    cv::Mat disp_l_float;
    disp_raw_l.convertTo(disp_l_float, CV_32F, 1.0 / 16.0);
    cv::threshold(disp_l_float, disp_l_float, 0.0, 0.0, cv::THRESH_TOZERO);

    /* 9. Convert left BGR → RGBA for GPU upload */
    cv::Mat left_rgba;
    cv::cvtColor(proc_l_bgr, left_rgba, cv::COLOR_BGR2RGBA);

    /* 10. Upload left color + disparity to GPU */
    glBindTexture(GL_TEXTURE_2D, tex_left_color);
    glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, proc_w, proc_h,
                    GL_RGBA, GL_UNSIGNED_BYTE, left_rgba.data);

    glBindTexture(GL_TEXTURE_2D, tex_left_disp);
    glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, proc_w, proc_h,
                    GL_RED, GL_FLOAT, disp_l_float.data);

    /* 11. Clear depth buffer and output */
    clearGPUBuffers();

    GLuint groups_x = (proc_w + 15) / 16;
    GLuint groups_y = (proc_h + 15) / 16;

    /* ---- Pass 1: Depth splat (left view → z-buffer) ---- */
    glUseProgram(prog_depth_splat);
    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_2D, tex_left_disp);
    glUniform1i(glGetUniformLocation(prog_depth_splat, "u_disparity"), 0);
    glUniform1f(glGetUniformLocation(prog_depth_splat, "u_shift"), 0.5f);
    glUniform2i(glGetUniformLocation(prog_depth_splat, "u_output_size"), proc_w, proc_h);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, ssbo_depth);
    glDispatchCompute(groups_x, groups_y, 1);

    glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);

    /* ---- Pass 2: Color splat (left view, z-tested) ---- */
    glUseProgram(prog_color_splat);
    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_2D, tex_left_color);
    glUniform1i(glGetUniformLocation(prog_color_splat, "u_color"), 0);
    glActiveTexture(GL_TEXTURE1);
    glBindTexture(GL_TEXTURE_2D, tex_left_disp);
    glUniform1i(glGetUniformLocation(prog_color_splat, "u_disparity"), 1);
    glUniform1f(glGetUniformLocation(prog_color_splat, "u_shift"), 0.5f);
    glUniform2i(glGetUniformLocation(prog_color_splat, "u_output_size"), proc_w, proc_h);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, ssbo_depth);
    glBindImageTexture(1, tex_output, 0, GL_FALSE, 0, GL_WRITE_ONLY, GL_RGBA8);
    glDispatchCompute(groups_x, groups_y, 1);

    glMemoryBarrier(GL_SHADER_IMAGE_ACCESS_BARRIER_BIT);

    /* ---- Pass 3: Hole fill ---- */
    GLuint display_tex;
    if (use_hole_fill) {
        glUseProgram(prog_hole_fill);
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, ssbo_depth);
        glBindImageTexture(1, tex_output, 0, GL_FALSE, 0, GL_READ_ONLY, GL_RGBA8);
        glBindImageTexture(2, tex_filled, 0, GL_FALSE, 0, GL_WRITE_ONLY, GL_RGBA8);
        glUniform1i(glGetUniformLocation(prog_hole_fill, "u_max_search"),
                    HOLE_FILL_MAX_SEARCH);
        glUniform2i(glGetUniformLocation(prog_hole_fill, "u_output_size"), proc_w, proc_h);
        glDispatchCompute(groups_x, groups_y, 1);
        glMemoryBarrier(GL_SHADER_IMAGE_ACCESS_BARRIER_BIT);
        display_tex = tex_filled;
    } else {
        display_tex = tex_output;
    }

    /* ---- Render fullscreen quad ---- */
    glViewport(0, 0, width(), height());
    glClear(GL_COLOR_BUFFER_BIT);
    glUseProgram(prog_display);
    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_2D, display_tex);
    glUniform1i(glGetUniformLocation(prog_display, "u_texture"), 0);
    glBindVertexArray(quad_vao);
    glDrawArrays(GL_TRIANGLE_STRIP, 0, 4);
    glBindVertexArray(0);

    /* ---- FPS overlay using QPainter ---- */
    auto now = std::chrono::steady_clock::now();
    frame_times.push_back(now);
    while ((int)frame_times.size() > FPS_WINDOW)
        frame_times.pop_front();

    if (frame_times.size() >= 2) {
        double elapsed = std::chrono::duration<double>(
            frame_times.back() - frame_times.front()).count();
        current_fps = (double)(frame_times.size() - 1) / elapsed;
    }

    QPainter painter(this);
    painter.setPen(Qt::white);
    QFont font("monospace", 14);
    font.setBold(true);
    painter.setFont(font);

    char info[256];
    snprintf(info, sizeof(info),
             "FPS: %.1f | %s nDisp=%d blk=%d WLS=%s Hole=%s | %dx%d",
             current_fps,
             use_sgbm ? "SGBM" : "BM",
             num_disparities, block_size,
             use_wls ? "ON" : "OFF",
             use_hole_fill ? "ON" : "OFF",
             proc_w, proc_h);

    /* Draw text shadow for readability */
    painter.setPen(Qt::black);
    painter.drawText(12, 32, info);
    painter.setPen(Qt::white);
    painter.drawText(10, 30, info);
    painter.end();

    /* Print FPS to terminal periodically */
    double since_print = std::chrono::duration<double>(
        now - last_fps_print).count();
    if (since_print >= FPS_PRINT_INTERVAL) {
        printf("FPS: %.1f  |  %s  numDisp=%d  block=%d  WLS=%s  hole=%s  proc=%dx%d\n",
               current_fps,
               use_sgbm ? "SGBM" : "BM",
               num_disparities, block_size,
               use_wls ? "ON" : "OFF",
               use_hole_fill ? "ON" : "OFF",
               proc_w, proc_h);
        last_fps_print = now;
    }
}

/* ========================================================================
 * main
 * ======================================================================== */

int main(int argc, char *argv[])
{
    /* Request OpenGL ES 3.1 context */
    QSurfaceFormat fmt;
    fmt.setRenderableType(QSurfaceFormat::OpenGLES);
    fmt.setVersion(3, 1);
    fmt.setSwapBehavior(QSurfaceFormat::DoubleBuffer);
    QSurfaceFormat::setDefaultFormat(fmt);

    QApplication app(argc, argv);

    /* Determine calibration file path */
    std::string calib_path;
    if (argc >= 2) {
        calib_path = argv[1];
    } else {
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
        SynthWindow win(cm, calib);
        win.resize(960, 540);
        win.show();
        app.exec();
    }

    cm->stop();
    return 0;
}

#include "view_synthesis.moc"
