/*
 * View Synthesis: DIBR (Depth-Image Based Rendering) via GPU Compute Shaders
 *
 * Synthesises a novel viewpoint from a virtual camera positioned halfway
 * between the two physical cameras using forward warping via
 * OpenGL ES 3.1 compute shaders.
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
 *   +/-   - Increase/decrease numDisparities (steps of 16 for BM; 64/128/256 for SGBM)
 *   [/]   - Decrease/increase blockSize (by 2, must be odd)
 *   S     - Toggle StereoBM / StereoSGBM (libSGM CUDA when available)
 *   W     - Toggle WLS disparity filter
 *   1/2/3 - Processing scale: 1.0x, 0.5x, 0.25x
 *   H     - Toggle hole filling
 *   F     - Toggle face-tracking shift
 *   C     - Recalibrate face tracker (look straight ahead first)
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

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/calib3d.hpp>
#include <opencv2/ximgproc/disparity_filter.hpp>
#include <opencv2/cudaarithm.hpp>
#include <opencv2/cudaimgproc.hpp>
#include <opencv2/cudawarping.hpp>
#include <opencv2/cudastereo.hpp>

#include <libsgm.h>

#include "face_tracker.h"

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

#define LEFT_CAMERA_ID      1
#define RIGHT_CAMERA_ID     0
#define CAMERA_WIDTH        1920
#define CAMERA_HEIGHT       1080
#define FPS_WINDOW          30
#define FPS_PRINT_INTERVAL  2.0
#define HOLE_FILL_MAX_SEARCH 16
#define FACE_CAM_INDEX      2   // V4L2 index of the USB viewer-facing webcam
                                // CSI stereo cameras occupy /dev/video0 and /dev/video1
                                // on Jetson Nano; USB webcam is typically /dev/video2

/* ========================================================================
 * Camera capture
 * ======================================================================== */

#include "camera_capture.h"

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

    float dst_xf = float(src.x) + disp * u_shift;
    int dst_x0 = int(floor(dst_xf));
    int dst_x1 = dst_x0 + 1;

    uint depth_val = floatBitsToUint(disp);

    if (dst_x0 >= 0 && dst_x0 < u_output_size.x)
        atomicMax(depth[src.y * u_output_size.x + dst_x0], depth_val);

    bool write_second = true;
    if (src.x + 1 < u_output_size.x) {
        float disp_neighbour = texelFetch(u_disparity, ivec2(src.x + 1, src.y), 0).r;
        if (abs(disp - disp_neighbour) > 2.0)
            write_second = false;
    }

    if (write_second && dst_x1 >= 0 && dst_x1 < u_output_size.x)
        atomicMax(depth[src.y * u_output_size.x + dst_x1], depth_val);
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
    uint my_depth = floatBitsToUint(disp);
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

static const char *BACKWARD_COLOR_CS = R"(#version 310 es
layout(local_size_x = 16, local_size_y = 16) in;

uniform sampler2D u_colour;
uniform ivec2 u_output_size;
uniform float u_shift;

layout(std430, binding = 0) readonly buffer DepthBuffer { uint depth[]; };
layout(rgba8, binding = 1) writeonly uniform highp image2D u_output;

void main() {
    ivec2 dst = ivec2(gl_GlobalInvocationID.xy);
    if (dst.x >= u_output_size.x || dst.y >= u_output_size.y) return;

    uint stored = depth[dst.y * u_output_size.x + dst.x];
    if (stored == 0u) return;

    float warped_disp = uintBitsToFloat(stored);
    float src_xf = float(dst.x) - warped_disp * u_shift;
    float src_yf = float(dst.y);

    if (src_xf < 0.0 || src_xf >= float(u_output_size.x) - 1.0) return;

    vec2 uv = vec2((src_xf + 0.5) / float(u_output_size.x),
                   (src_yf + 0.5) / float(u_output_size.y));

    vec4 col = texture(u_colour, uv);
    imageStore(u_output, dst, col);
}
)";

static const char *COMPOSITE_CS = R"(#version 310 es
layout(local_size_x = 16, local_size_y = 16) in;

uniform ivec2 u_output_size;
uniform float u_shift;
layout(rgba8, binding = 0) readonly uniform highp image2D u_left;
layout(rgba8, binding = 1) readonly uniform highp image2D u_right;
layout(rgba8, binding = 2) writeonly uniform highp image2D u_out;
layout(std430, binding = 0) buffer LeftDepth { uint left_depth[]; };
layout(std430, binding = 1) readonly buffer RightDepth { uint right_depth[]; };

void main() {
    ivec2 pos = ivec2(gl_GlobalInvocationID.xy);
    if (pos.x >= u_output_size.x || pos.y >= u_output_size.y) return;

    int idx = pos.y * u_output_size.x + pos.x;
    uint ld = left_depth[idx];
    uint rd = right_depth[idx];

    // Weight each view by proximity of the virtual camera to each physical camera.
    // At u_shift=0.5 both are equally weighted; at u_shift=0 left dominates; at u_shift=1 right dominates.
    float w_left  = 1.0 - u_shift;
    float w_right = u_shift;

    vec4 col;
    if (ld > 0u && rd > 0u) {
        vec4 left_col  = imageLoad(u_left,  pos);
        vec4 right_col = imageLoad(u_right, pos);
        col = left_col * w_left + right_col * w_right;
    } else if (ld > 0u) {
        col = imageLoad(u_left, pos);
    } else if (rd > 0u) {
        col = imageLoad(u_right, pos);
    } else {
        col = vec4(0.0);  // true hole — hole fill pass handles this
    }

    imageStore(u_out, pos, col);
    // Write composite depth for hole-fill pass
    left_depth[idx] = (ld > 0u) ? ld : rd;
}
)";

/* Zero the depth SSBO entirely on the GPU — avoids uploading a 4 MB
 * CPU zero-vector every frame via glBufferSubData. */
static const char *CLEAR_CS = R"(#version 310 es
layout(local_size_x = 64) in;
layout(std430, binding = 0) buffer DepthBuf { uint depth[]; };
uniform int u_total;
void main() {
    uint idx = gl_GlobalInvocationID.x;
    if (idx < uint(u_total)) depth[idx] = 0u;
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
    SynthWindow(const CalibData &calib, QWidget *parent = nullptr);
    ~SynthWindow();

protected:
    void initializeGL() override;
    void paintGL() override;
    void resizeGL(int w, int h) override;
    void keyPressEvent(QKeyEvent *e) override;

private:
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
    GLuint prog_clear{0};
    GLuint prog_backward_color{0};
    GLuint prog_composite{0};
    GLint  uloc_clear_total{-1};
    GLuint tex_left_color, tex_right_color;
    GLuint tex_left_disp, tex_right_disp;
    GLuint tex_output, tex_filled;
    GLuint ssbo_depth;
    GLuint ssbo_depth_r{0};
    GLuint tex_right_col_rect{0};
    GLuint tex_right_disp_gl{0};
    GLuint tex_output_r{0};
    GLuint quad_vao, quad_vbo;
    bool gl_ready;
    bool diag_frames_logged{false};   // one-shot: first frame received
    bool diag_disp_logged{false};     // one-shot: first non-zero disparity

    /* Timer — singleShot chain drives the render loop (no fixed-interval cap) */
    bool render_running{false};

    /* Face tracking */
    FaceTracker *face_tracker          = nullptr;
    bool         face_tracking_enabled = false;

    /* FPS tracking */
    std::deque<std::chrono::steady_clock::time_point> frame_times;
    std::chrono::steady_clock::time_point last_fps_print;
    double current_fps;

    /* Staging buffers to avoid per-frame allocation */
    cv::Mat rgba_buf;
    std::vector<GLubyte> clear_black;
    cv::Mat m_disp_l_float;
    cv::Mat m_disp_r_float;
    cv::Mat m_left_rgba;
    cv::Mat m_right_rgba;

    /* CUDA acceleration */
    bool use_cuda{false};
    cv::cuda::GpuMat gpu_map_lx,   gpu_map_ly,   gpu_map_rx,   gpu_map_ry;   // full-res maps
    cv::cuda::GpuMat gpu_map_lx_s, gpu_map_ly_s, gpu_map_rx_s, gpu_map_ry_s; // proc_scale maps (full-res source)
    cv::cuda::GpuMat gpu_map_lx_c, gpu_map_ly_c, gpu_map_rx_c, gpu_map_ry_c; // camera-space maps (proc_scale source)
    cv::cuda::GpuMat gpu_frame_l,  gpu_frame_r;
    cv::cuda::GpuMat gpu_proc_l,   gpu_proc_r;
    cv::cuda::GpuMat gpu_gray_l,   gpu_gray_r;
    cv::cuda::GpuMat gpu_disp_l;
    cv::cuda::GpuMat gpu_rgba_l;
    cv::Ptr<cv::cuda::StereoBM> cuda_bm;
    std::unique_ptr<sgm::StereoSGM> sgm_cuda;
    cv::cuda::GpuMat              gpu_disp_sgm;    // CV_16U, libSGM left output
    cv::cuda::GpuMat              gpu_disp_float;
    cv::cuda::GpuMat              disp_filtered_gpu;
    bool                          disp_filtered_init{false};

    /* Cached GL uniform locations (set once in initializeGL) */
    GLint uloc_ds_disparity{-1}, uloc_ds_shift{-1}, uloc_ds_size{-1};
    GLint uloc_cs_color{-1}, uloc_cs_disp{-1}, uloc_cs_shift{-1}, uloc_cs_size{-1};
    GLint uloc_hf_maxsearch{-1}, uloc_hf_size{-1};
    GLint uloc_disp_texture{-1};
    GLint uloc_bc_colour{-1}, uloc_bc_size{-1}, uloc_bc_shift{-1};
    GLint uloc_comp_size{-1}, uloc_comp_shift{-1};

    void shutdownCameras();
    void rebuildStereo();
    void recreateTextures();
    void clearGPUBuffers();
};

/* ---- constructor ---- */

SynthWindow::SynthWindow(const CalibData &calib_in, QWidget *parent)
    : QOpenGLWidget(parent), calib(calib_in), cameras_ok(false),
      num_disparities(64), block_size(9),
      use_sgbm(true), use_wls(false), use_hole_fill(true),
      proc_scale(0.5),
      prog_depth_splat(0), prog_color_splat(0), prog_hole_fill(0), prog_display(0), prog_backward_color(0), prog_composite(0),
      tex_left_color(0), tex_right_color(0),
      tex_left_disp(0), tex_right_disp(0),
      tex_output(0), tex_filled(0), ssbo_depth(0), ssbo_depth_r(0),
      tex_right_col_rect(0), tex_right_disp_gl(0), tex_output_r(0),
      quad_vao(0), quad_vbo(0),
      gl_ready(false), current_fps(0.0)
{
    setWindowTitle("View Synthesis (DIBR)");

    proc_w = (int)(CAMERA_WIDTH  * proc_scale);
    proc_h = (int)(CAMERA_HEIGHT * proc_scale);

    /* Precompute rectification maps */
    cv::initUndistortRectifyMap(
        calib.K_left, calib.dist_left, calib.R1, calib.P1,
        calib.image_size, CV_32FC1, map_lx, map_ly);
    cv::initUndistortRectifyMap(
        calib.K_right, calib.dist_right, calib.R2, calib.P2,
        calib.image_size, CV_32FC1, map_rx, map_ry);
    printf("Rectification maps precomputed (%dx%d)\n",
           calib.image_size.width, calib.image_size.height);

    if (cv::cuda::getCudaEnabledDeviceCount() > 0) {
        use_cuda = true;
        gpu_map_lx.upload(map_lx);  gpu_map_ly.upload(map_ly);
        gpu_map_rx.upload(map_rx);  gpu_map_ry.upload(map_ry);
        printf("CUDA available — GPU preprocessing enabled\n");
    } else {
        printf("No CUDA — CPU preprocessing fallback\n");
    }

    rebuildStereo();

    /* Initialise cameras — request hardware downscale to proc resolution via
     * a second nvvidconv stage inside the GStreamer pipeline (CSI backend only).
     * This reduces the per-frame CPU/bus payload from ~12 MB to ~3 MB. */
    bool ok_l = init_camera(&left_cap, LEFT_CAMERA_ID,
                            CAMERA_WIDTH, CAMERA_HEIGHT, proc_w, proc_h);
    bool ok_r = init_camera(&right_cap, RIGHT_CAMERA_ID,
                            CAMERA_WIDTH, CAMERA_HEIGHT, proc_w, proc_h);
    cameras_ok = ok_l && ok_r;

    if (!cameras_ok) {
        printf("[DIAG] CAMERA INIT FAILED  ok_l=%d ok_r=%d\n", ok_l, ok_r);
        fprintf(stderr, "[DIAG] CAMERA INIT FAILED  ok_l=%d ok_r=%d\n", ok_l, ok_r);
        return;
    }
    printf("[DIAG] Cameras OK  L:%dx%d  R:%dx%d\n",
           left_cap.width, left_cap.height,
           right_cap.width, right_cap.height);

    printf("\n============================================================\n");
    printf("VIEW SYNTHESIS (DIBR)\n");
    printf("============================================================\n");
    printf("Resolution:      %dx%d\n", CAMERA_WIDTH, CAMERA_HEIGHT);
    printf("Processing:      %dx%d (%.0f%%)\n", proc_w, proc_h, proc_scale * 100.0);
    printf("Baseline:        %.2f mm\n", calib.baseline);
    printf("Calib RMS error: %.4f px\n", calib.rms_error);
    printf("numDisparities:  %d\n", num_disparities);
    printf("blockSize:       %d\n", block_size);
    printf("Matcher:         %s\n", use_sgbm ? "SGBM (libSGM CUDA / CPU fallback)" : "StereoBM");
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
    printf("  F     - Toggle face-tracking shift\n");
    printf("  C     - Recalibrate face tracker (look straight ahead first)\n");
    printf("============================================================\n\n");

    /* ---- Face tracker (USB webcam, background thread) ---- */
    {
        // Look for the YuNet ONNX model alongside the binary or in the
        // source tree; fall back gracefully if it is not present.
        // 2021sep is the original model from when FaceDetectorYN was added (OpenCV 4.5.4).
        // 2022mar/2023mar use newer ONNX opsets that crash OpenCV 4.5.x (Jetson L4T).
        const char *candidates[] = {
            "face_detection_yunet_2021sep.onnx",
            "models/face_detection_yunet_2021sep.onnx",
            "src/models/face_detection_yunet_2021sep.onnx",
        };
        std::string yunet_path;
        for (const char *c : candidates) {
            if (FILE *f = std::fopen(c, "rb")) {
                std::fclose(f);
                yunet_path = c;
                break;
            }
        }

        if (yunet_path.empty()) {
            printf("[FaceTracker] Model not found — face tracking disabled.\n");
            printf("  Download: face_detection_yunet_2021sep.onnx\n");
            printf("  (see: opencv_zoo/models/face_detection_yunet)\n");
        } else {
            face_tracker = new FaceTracker();
            if (face_tracker->start(FACE_CAM_INDEX, yunet_path)) {
                face_tracking_enabled = true;
                printf("[FaceTracker] Started on camera %d  model: %s\n",
                       FACE_CAM_INDEX, yunet_path.c_str());
                printf("[FaceTracker] Press C to calibrate, F to toggle\n");
            } else {
                printf("[FaceTracker] Failed to start\n");
                delete face_tracker;
                face_tracker = nullptr;
            }
        }
    }

    last_fps_print = std::chrono::steady_clock::now();
}

/* ---- destructor ---- */

SynthWindow::~SynthWindow()
{
    if (face_tracker) {
        face_tracker->stop();
        delete face_tracker;
        face_tracker = nullptr;
    }
    shutdownCameras();
    makeCurrent();
    if (prog_depth_splat)    glDeleteProgram(prog_depth_splat);
    if (prog_color_splat)    glDeleteProgram(prog_color_splat);
    if (prog_hole_fill)      glDeleteProgram(prog_hole_fill);
    if (prog_display)        glDeleteProgram(prog_display);
    if (prog_clear)          glDeleteProgram(prog_clear);
    if (prog_backward_color) glDeleteProgram(prog_backward_color);
    if (prog_composite)      glDeleteProgram(prog_composite);
    GLuint textures[] = { tex_left_color, tex_right_color,
                          tex_left_disp, tex_right_disp,
                          tex_output, tex_filled,
                          tex_right_col_rect, tex_right_disp_gl, tex_output_r };
    for (auto t : textures)
        if (t) glDeleteTextures(1, &t);
    if (ssbo_depth)   glDeleteBuffers(1, &ssbo_depth);
    if (ssbo_depth_r) glDeleteBuffers(1, &ssbo_depth_r);
    if (quad_vbo) glDeleteBuffers(1, &quad_vbo);
    if (quad_vao) glDeleteVertexArrays(1, &quad_vao);
    doneCurrent();
}

void SynthWindow::shutdownCameras()
{
    if (!cameras_ok)
        return;
    cameras_ok = false;
    render_running = false;
    cleanup_camera(&left_cap);
    cleanup_camera(&right_cap);
}

/* ---- rebuild stereo matcher ---- */

void SynthWindow::rebuildStereo()
{
    cuda_bm.reset();
    sgm_cuda.reset();
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
        if (use_cuda) {
            int cuda_ndisp = std::min(num_disparities, 256);
            cuda_bm = cv::cuda::createStereoBM(cuda_ndisp, block_size);
            cuda_bm->setPreFilterType(cv::StereoBM::PREFILTER_XSOBEL);
            cuda_bm->setPreFilterCap(31);
            cuda_bm->setUniquenessRatio(15);
            cuda_bm->setTextureThreshold(0);
            cuda_bm->setMinDisparity(0);
        }
        stereo = bm;
    }

    right_stereo = cv::ximgproc::createRightMatcher(stereo);
    wls_filter = cv::ximgproc::createDisparityWLSFilter(stereo);
    wls_filter->setLambda(8000.0);
    wls_filter->setSigmaColor(1.5);

    // libSGM CUDA SGM — replaces CPU SGBM when CUDA is available
    if (use_sgbm && use_cuda) {
        // libSGM disparity_size must be 64, 128, or 256
        const int disp_size = (num_disparities <= 64)  ? 64  :
                              (num_disparities <= 128) ? 128 : 256;

        // Pre-allocate grayscale + output buffers so pitches are stable
        gpu_gray_l.create(proc_h, proc_w, CV_8U);
        gpu_gray_r.create(proc_h, proc_w, CV_8U);
        gpu_disp_sgm.create(proc_h, proc_w, CV_16U);

        // P1/P2 must be calibrated to Census Transform cost scale, NOT the
        // SSD/SAD scale used by OpenCV StereoSGBM.  libSGM uses a 9×7 Census
        // window whose Hamming distance maxes out at 63.  The SGBM formula
        // (8/32 × block_size²) gives P1=648, P2=2592 with block_size=9 —
        // ~10× above max census cost — which locks disparity constant along
        // each scanline and causes severe horizontal streaking.
        // libSGM defaults (P1=10, P2=120) are appropriate for census costs.
        const int P1 = 10;
        const int P2 = 120;
        // SCAN_4PATH (H + V + 2 diagonal) halves aggregation work vs 8-path,
        // recovering FPS on Maxwell while still suppressing most streaking.
        sgm::StereoSGM::Parameters sgm_params(P1, P2);
        sgm_params.path_type = sgm::PathType::SCAN_4PATH;

        // src_pitch: gpu_gray step in pixels (CV_8U → step == bytes == pixels)
        // dst_pitch: gpu_disp_sgm step in uint16_t units
        sgm_cuda = std::make_unique<sgm::StereoSGM>(
            proc_w, proc_h, disp_size,
            8, 16,
            static_cast<int>(gpu_gray_l.step),
            static_cast<int>(gpu_disp_sgm.step / sizeof(uint16_t)),
            sgm::EXECUTE_INOUT_CUDA2CUDA,
            sgm_params);

        printf("[SynthWindow] libSGM CUDA: disp_size=%d P1=%d P2=%d 4-path (%dx%d)\n",
               disp_size, P1, P2, proc_w, proc_h);
    }
}

/* ---- recreate GPU textures at current proc resolution ---- */

void SynthWindow::recreateTextures()
{
    GLuint old[] = { tex_left_color, tex_right_color,
                     tex_left_disp, tex_right_disp,
                     tex_output, tex_filled,
                     tex_right_col_rect, tex_output_r };
    for (auto t : old)
        if (t) glDeleteTextures(1, &t);
    if (ssbo_depth)   glDeleteBuffers(1, &ssbo_depth);
    if (ssbo_depth_r) glDeleteBuffers(1, &ssbo_depth_r);

    tex_left_color     = create_texture_rgba8(proc_w, proc_h);
    tex_right_color    = create_texture_rgba8(proc_w, proc_h);
    tex_left_disp      = create_texture_r32f(proc_w, proc_h);
    tex_right_disp     = create_texture_r32f(proc_w, proc_h);
    tex_output         = create_texture_rgba8(proc_w, proc_h);
    tex_filled         = create_texture_rgba8(proc_w, proc_h);
    ssbo_depth         = create_ssbo(proc_w * proc_h);
    tex_right_col_rect = create_texture_rgba8(proc_w, proc_h);
    tex_right_disp_gl  = create_texture_r32f(proc_w, proc_h);
    tex_output_r       = create_texture_rgba8(proc_w, proc_h);
    ssbo_depth_r       = create_ssbo(proc_w * proc_h);

    clear_black.assign(proc_w * proc_h * 4, 0);
    m_disp_l_float.create(proc_h, proc_w, CV_32F);
    m_left_rgba.create(proc_h, proc_w, CV_8UC4);
    m_right_rgba.create(proc_h, proc_w, CV_8UC4);

    /* Pre-scale rectification maps to proc_scale so remap outputs directly
     * at processing resolution without a separate GPU resize pass */
    if (use_cuda) {
        cv::Mat slx, sly, srx, sry;
        cv::resize(map_lx, slx, cv::Size(proc_w, proc_h));
        cv::resize(map_ly, sly, cv::Size(proc_w, proc_h));
        cv::resize(map_rx, srx, cv::Size(proc_w, proc_h));
        cv::resize(map_ry, sry, cv::Size(proc_w, proc_h));
        gpu_map_lx_s.upload(slx);  gpu_map_ly_s.upload(sly);
        gpu_map_rx_s.upload(srx);  gpu_map_ry_s.upload(sry);

        /* Camera-space maps: same proc_scale output but source coords
         * are in proc_scale space (when camera hardware-downscales to proc_w×proc_h).
         * Multiply pre-scaled map values by proc_scale to address the smaller source. */
        gpu_map_lx_c.upload(slx * proc_scale);
        gpu_map_ly_c.upload(sly * proc_scale);
        gpu_map_rx_c.upload(srx * proc_scale);
        gpu_map_ry_c.upload(sry * proc_scale);
    }

    printf("GPU textures + SSBOs created: %dx%d\n", proc_w, proc_h);
}

/* ---- clear depth buffer and output ---- */

void SynthWindow::clearGPUBuffers()
{
    /* Clear depth SSBO to 0 on GPU — avoids uploading a ~4 MB zero-vector
     * every frame via glBufferSubData. */
    glUseProgram(prog_clear);
    glUniform1i(uloc_clear_total, proc_w * proc_h);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, ssbo_depth);
    glDispatchCompute((proc_w * proc_h + 63) / 64, 1, 1);
    glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);

    /* Clear output texture to black */
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
    prog_depth_splat    = create_compute_program(DEPTH_SPLAT_CS);
    prog_color_splat    = create_compute_program(COLOR_SPLAT_CS);
    prog_hole_fill      = create_compute_program(HOLE_FILL_CS);
    prog_display        = create_render_program(DISPLAY_VS, DISPLAY_FS);
    prog_clear          = create_compute_program(CLEAR_CS);
    prog_backward_color = create_compute_program(BACKWARD_COLOR_CS);
    prog_composite      = create_compute_program(COMPOSITE_CS);

    if (!prog_depth_splat || !prog_color_splat || !prog_hole_fill || !prog_display || !prog_clear || !prog_backward_color || !prog_composite) {
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
    uloc_ds_disparity = glGetUniformLocation(prog_depth_splat, "u_disparity");
    uloc_ds_shift     = glGetUniformLocation(prog_depth_splat, "u_shift");
    uloc_ds_size      = glGetUniformLocation(prog_depth_splat, "u_output_size");
    uloc_cs_color     = glGetUniformLocation(prog_color_splat, "u_color");
    uloc_cs_disp      = glGetUniformLocation(prog_color_splat, "u_disparity");
    uloc_cs_shift     = glGetUniformLocation(prog_color_splat, "u_shift");
    uloc_cs_size      = glGetUniformLocation(prog_color_splat, "u_output_size");
    uloc_hf_maxsearch = glGetUniformLocation(prog_hole_fill,   "u_max_search");
    uloc_hf_size      = glGetUniformLocation(prog_hole_fill,   "u_output_size");
    uloc_disp_texture = glGetUniformLocation(prog_display,        "u_texture");
    uloc_clear_total  = glGetUniformLocation(prog_clear,          "u_total");
    uloc_bc_colour    = glGetUniformLocation(prog_backward_color, "u_colour");
    uloc_bc_size      = glGetUniformLocation(prog_backward_color, "u_output_size");
    uloc_bc_shift     = glGetUniformLocation(prog_backward_color, "u_shift");
    uloc_comp_size    = glGetUniformLocation(prog_composite,      "u_output_size");
    uloc_comp_shift   = glGetUniformLocation(prog_composite,      "u_shift");
    printf("GPU pipeline initialised\n");

    /* Kick off the self-sustaining render loop (no fixed-interval cap) */
    render_running = true;
    QTimer::singleShot(0, this, [this](){ update(); });
}

/* ---- key handling ---- */

void SynthWindow::keyPressEvent(QKeyEvent *e)
{
    if (e->key() == Qt::Key_Escape) {
        close();
        return;
    }

    bool changed = false;

    // Valid libSGM disparity sizes in order
    static constexpr int SGM_DISP_SIZES[] = {64, 128, 256};

    if (e->key() == Qt::Key_Plus || e->key() == Qt::Key_Equal) {
        if (use_sgbm) {
            // Step up through {64, 128, 256}
            for (int d : SGM_DISP_SIZES)
                if (d > num_disparities) { num_disparities = d; break; }
        } else {
            num_disparities = std::min(num_disparities + 16,
                                       use_cuda ? 256 : 512);
        }
        changed = true;
    }
    else if (e->key() == Qt::Key_Minus) {
        if (use_sgbm) {
            // Step down through {256, 128, 64}
            for (int i = 2; i >= 0; --i)
                if (SGM_DISP_SIZES[i] < num_disparities) { num_disparities = SGM_DISP_SIZES[i]; break; }
        } else {
            num_disparities = std::max(num_disparities - 16, 16);
        }
        changed = true;
    }
    else if (e->key() == Qt::Key_BracketRight) {
        block_size = std::min(block_size + 2, 51);
        changed = true;
    }
    else if (e->key() == Qt::Key_BracketLeft) {
        block_size = std::max(block_size - 2, 3);
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
        if (use_sgbm) {
            // Snap num_disparities to nearest of {64, 128, 256}
            int best = 64;
            for (int d : SGM_DISP_SIZES)
                if (std::abs(d - num_disparities) < std::abs(best - num_disparities))
                    best = d;
            num_disparities = best;
        }
        changed = true;
        printf("Matcher: %s\n", use_sgbm ? "SGBM (libSGM CUDA / CPU fallback)" : "StereoBM");
    }
    else if (e->key() == Qt::Key_W) {
        use_wls = !use_wls;
        printf("WLS filter: %s\n", use_wls ? "ON" : "OFF");
    }
    else if (e->key() == Qt::Key_H) {
        use_hole_fill = !use_hole_fill;
        printf("Hole filling: %s\n", use_hole_fill ? "ON" : "OFF");
    }
    else if (e->key() == Qt::Key_F) {
        if (face_tracker) {
            face_tracking_enabled = !face_tracking_enabled;
            printf("Face tracking: %s\n", face_tracking_enabled ? "ON" : "OFF");
        } else {
            printf("Face tracking not available (model not loaded)\n");
        }
    }
    else if (e->key() == Qt::Key_C) {
        if (face_tracker && face_tracking_enabled) {
            face_tracker->calibrate();
            printf("Face tracker recalibrated\n");
        } else {
            printf("Face tracking not active — cannot calibrate\n");
        }
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
    if (!gl_ready) { printf("[DIAG] paintGL: gl_ready=false\n"); return; }
    if (!cameras_ok) { printf("[DIAG] paintGL: cameras_ok=false\n"); return; }

    /* 1. Grab L/R frames — cv::swap avoids clone() memcpy */
    cv::Mat frame_l, frame_r;
    {
        std::lock_guard<std::mutex> lock(left_cap.frame_mutex);
        if (left_cap.new_frame) {
            cv::swap(frame_l, left_cap.frame);
            left_cap.new_frame = false;
        }
    }
    {
        std::lock_guard<std::mutex> lock(right_cap.frame_mutex);
        if (right_cap.new_frame) {
            cv::swap(frame_r, right_cap.frame);
            right_cap.new_frame = false;
        }
    }

    if (frame_l.empty() || frame_r.empty())
        return;

    if (!diag_frames_logged) {
        printf("[DIAG] First frames: L=%dx%d type=%d  R=%dx%d type=%d  proc=%dx%d\n",
               frame_l.cols, frame_l.rows, frame_l.type(),
               frame_r.cols, frame_r.rows, frame_r.type(),
               proc_w, proc_h);
        printf("[DIAG] use_cuda=%d  cuda_bm=%s  sgm_cuda=%s  use_sgbm=%d\n",
               use_cuda, cuda_bm ? "valid" : "null",
               sgm_cuda ? "valid" : "null", use_sgbm);
        diag_frames_logged = true;
    }

    /* Preprocessing — CUDA path when GPU BM or libSGM is available;
     * CPU SGBM fallback only when use_cuda=false */
    cv::Mat &disp_l_float = m_disp_l_float;
    cv::Mat &left_rgba    = m_left_rgba;
    if (use_cuda && (cuda_bm || sgm_cuda)) {
        /* 2. Upload frames to GPU.
         * If camera hardware-downscaled to proc_w×proc_h, the upload is 4× smaller
         * and we use camera-space maps (source coords in proc_scale space).
         * If camera still outputs full-res (e.g. after runtime scale change),
         * fall back to pre-scaled maps (source coords in full-res space). */
        bool cam_at_proc = (frame_l.cols == proc_w && frame_l.rows == proc_h);
        gpu_frame_l.upload(frame_l);
        gpu_frame_r.upload(frame_r);

        /* 3. Remap + rectify directly to proc_scale in one pass */
        const cv::cuda::GpuMat &map_lx_use = cam_at_proc ? gpu_map_lx_c : gpu_map_lx_s;
        const cv::cuda::GpuMat &map_ly_use = cam_at_proc ? gpu_map_ly_c : gpu_map_ly_s;
        const cv::cuda::GpuMat &map_rx_use = cam_at_proc ? gpu_map_rx_c : gpu_map_rx_s;
        const cv::cuda::GpuMat &map_ry_use = cam_at_proc ? gpu_map_ry_c : gpu_map_ry_s;
        cv::cuda::remap(gpu_frame_l, gpu_proc_l, map_lx_use, map_ly_use, cv::INTER_LINEAR);
        cv::cuda::remap(gpu_frame_r, gpu_proc_r, map_rx_use, map_ry_use, cv::INTER_LINEAR);

        /* 4. BGR→Gray */
        cv::cuda::cvtColor(gpu_proc_l, gpu_gray_l, cv::COLOR_BGR2GRAY);
        cv::cuda::cvtColor(gpu_proc_r, gpu_gray_r, cv::COLOR_BGR2GRAY);

        /* 5. Stereo match — left and right disparity */
        if (sgm_cuda) {
            /* --- libSGM CUDA path --- */
            sgm_cuda->execute(gpu_gray_l.data, gpu_gray_r.data, gpu_disp_sgm.data);

            const auto invalid_val = static_cast<uint16_t>(sgm_cuda->get_invalid_disparity());

            cv::Mat disp_u16;
            gpu_disp_sgm.download(disp_u16);
            cv::Mat invalid_mask = (disp_u16 == invalid_val);
            disp_u16.convertTo(disp_l_float, CV_32F);
            disp_l_float.setTo(0.0f, invalid_mask);

            /* Right disparity via CPU right matcher (correct search direction) */
            cv::Mat gray_l_host, gray_r_host;
            gpu_gray_l.download(gray_l_host);
            gpu_gray_r.download(gray_r_host);
            cv::Mat disp_raw_r;
            right_stereo->compute(gray_r_host, gray_l_host, disp_raw_r);
            disp_raw_r.convertTo(m_disp_r_float, CV_32F, -1.0 / 16.0);
            cv::threshold(m_disp_r_float, m_disp_r_float, 0.0, 0.0, cv::THRESH_TOZERO);
        } else {
            /* --- cuda_bm path --- */
            cuda_bm->compute(gpu_gray_l, gpu_gray_r, gpu_disp_l, cv::cuda::Stream::Null());

            cv::Mat disp_raw_l;
            gpu_disp_l.download(disp_raw_l);
            disp_raw_l.convertTo(disp_l_float, CV_32F, 1.0 / 16.0);
            cv::threshold(disp_l_float, disp_l_float, 0.0, 0.0, cv::THRESH_TOZERO);

            /* Right disparity via CPU right matcher (correct search direction) */
            cv::Mat gray_l_host, gray_r_host;
            gpu_gray_l.download(gray_l_host);
            gpu_gray_r.download(gray_r_host);
            cv::Mat disp_raw_r;
            right_stereo->compute(gray_r_host, gray_l_host, disp_raw_r);
            disp_raw_r.convertTo(m_disp_r_float, CV_32F, -1.0 / 16.0);
            cv::threshold(m_disp_r_float, m_disp_r_float, 0.0, 0.0, cv::THRESH_TOZERO);
        }

        if (!diag_disp_logged) {
            double dmin, dmax;
            cv::minMaxLoc(disp_l_float, &dmin, &dmax);
            cv::Mat mask = disp_l_float > 0.5f;
            int nonzero = cv::countNonZero(mask);
            printf("[DIAG] Disparity: min=%.2f max=%.2f nonzero=%d/%d  "
                   "gray_l=%dx%d\n",
                   dmin, dmax, nonzero, proc_w * proc_h,
                   gpu_gray_l.cols, gpu_gray_l.rows);
            diag_disp_logged = true;
        }

        /* 9. BGR→RGBA on GPU, download directly (avoids CPU cvtColor round-trip) */
        cv::cuda::cvtColor(gpu_proc_l, gpu_rgba_l, cv::COLOR_BGR2RGBA);
        gpu_rgba_l.download(left_rgba);

        /* Also download right rectified frame as RGBA for two-view DIBR */
        cv::cuda::GpuMat gpu_rgba_r;
        cv::cuda::cvtColor(gpu_proc_r, gpu_rgba_r, cv::COLOR_BGR2RGBA);
        gpu_rgba_r.download(m_right_rgba);

    } else {
        /* ---- CPU path ---- */
        /* 2. Rectify at full resolution */
        cv::Mat rect_l_bgr, rect_r_bgr;
        cv::remap(frame_l, rect_l_bgr, map_lx, map_ly, cv::INTER_LINEAR);
        cv::remap(frame_r, rect_r_bgr, map_rx, map_ry, cv::INTER_LINEAR);

        /* 3. Resize for processing */
        cv::Mat proc_l_bgr, proc_r_bgr;
        if (proc_scale < 1.0) {
            cv::resize(rect_l_bgr, proc_l_bgr, cv::Size(proc_w, proc_h), 0, 0, cv::INTER_AREA);
            cv::resize(rect_r_bgr, proc_r_bgr, cv::Size(proc_w, proc_h), 0, 0, cv::INTER_AREA);
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

        /* 6. Right disparity (for WLS) */
        cv::Mat disp_raw_r;
        right_stereo->compute(gray_r, gray_l, disp_raw_r);

        /* 7. WLS filter */
        if (use_wls && wls_filter) {
            cv::Mat disp_filtered;
            wls_filter->filter(disp_raw_l, gray_l, disp_filtered, disp_raw_r);
            disp_raw_l = disp_filtered;
        }

        /* 8. Convert left and right disparity to float */
        disp_raw_l.convertTo(disp_l_float, CV_32F, 1.0 / 16.0);
        cv::threshold(disp_l_float, disp_l_float, 0.0, 0.0, cv::THRESH_TOZERO);

        disp_raw_r.convertTo(m_disp_r_float, CV_32F, -1.0 / 16.0);
        cv::threshold(m_disp_r_float, m_disp_r_float, 0.0, 0.0, cv::THRESH_TOZERO);

        /* 9. Convert left/right BGR → RGBA for GPU upload */
        cv::cvtColor(proc_l_bgr, left_rgba, cv::COLOR_BGR2RGBA);
        cv::cvtColor(proc_r_bgr, m_right_rgba, cv::COLOR_BGR2RGBA);
    }

    /* IIR temporal disparity filter — smooths frame-to-frame flicker */
    if (use_cuda) {
        const float alpha = 0.2f;
        gpu_disp_float.upload(disp_l_float);
        if (!disp_filtered_init || disp_filtered_gpu.size() != gpu_disp_float.size()) {
            gpu_disp_float.copyTo(disp_filtered_gpu);
            disp_filtered_init = true;
        } else {
            cv::cuda::addWeighted(gpu_disp_float, alpha, disp_filtered_gpu, 1.0 - alpha, 0.0, disp_filtered_gpu);
        }
        disp_filtered_gpu.download(disp_l_float);
    }

    /* 10. Upload left color + disparity to GL */
    glBindTexture(GL_TEXTURE_2D, tex_left_color);
    glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, proc_w, proc_h,
                    GL_RGBA, GL_UNSIGNED_BYTE, left_rgba.data);

    glBindTexture(GL_TEXTURE_2D, tex_left_disp);
    glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, proc_w, proc_h,
                    GL_RED, GL_FLOAT, disp_l_float.data);

    /* Upload right rectified colour and right disparity to GL */
    glBindTexture(GL_TEXTURE_2D, tex_right_col_rect);
    glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, proc_w, proc_h,
                    GL_RGBA, GL_UNSIGNED_BYTE, m_right_rgba.data);

    glBindTexture(GL_TEXTURE_2D, tex_right_disp_gl);
    glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, proc_w, proc_h,
                    GL_RED, GL_FLOAT, m_disp_r_float.data);

    /* 11. Clear depth buffer and output */
    clearGPUBuffers();

    GLuint groups_x = (proc_w + 15) / 16;
    GLuint groups_y = (proc_h + 15) / 16;

    /* ---- Resolve virtual camera shift ---- */
    const float u_shift = (face_tracking_enabled && face_tracker &&
                           face_tracker->isActive())
                          ? face_tracker->shift()
                          : 0.5f;

    /* ---- Pass 1: Depth splat (left view → z-buffer) ---- */
    glUseProgram(prog_depth_splat);
    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_2D, tex_left_disp);
    glUniform1i(uloc_ds_disparity, 0);
    glUniform1f(uloc_ds_shift, u_shift);
    glUniform2i(uloc_ds_size, proc_w, proc_h);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, ssbo_depth);
    glDispatchCompute(groups_x, groups_y, 1);

    glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);

    /* ---- Pass 2: Backward colour warp (destination-driven, bilinear source) ---- */
    glUseProgram(prog_backward_color);
    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_2D, tex_left_color);
    glUniform1i(uloc_bc_colour, 0);
    glUniform2i(uloc_bc_size, proc_w, proc_h);
    glUniform1f(uloc_bc_shift, u_shift);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, ssbo_depth);
    glBindImageTexture(1, tex_output, 0, GL_FALSE, 0, GL_WRITE_ONLY, GL_RGBA8);
    glDispatchCompute(groups_x, groups_y, 1);

    glMemoryBarrier(GL_SHADER_IMAGE_ACCESS_BARRIER_BIT);

    /* ---- Right-camera DIBR passes ---- */
    const float right_shift = -(1.0f - u_shift);

    /* Clear right depth SSBO */
    glUseProgram(prog_clear);
    glUniform1i(uloc_clear_total, proc_w * proc_h);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, ssbo_depth_r);
    glDispatchCompute((proc_w * proc_h + 63) / 64, 1, 1);
    glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);

    /* Clear right output texture to black */
    glBindTexture(GL_TEXTURE_2D, tex_output_r);
    glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, proc_w, proc_h,
                    GL_RGBA, GL_UNSIGNED_BYTE, clear_black.data());

    /* Right depth splat — use actual right disparity with right_shift */
    glUseProgram(prog_depth_splat);
    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_2D, tex_right_disp_gl);
    glUniform1i(uloc_ds_disparity, 0);
    glUniform1f(uloc_ds_shift, right_shift);
    glUniform2i(uloc_ds_size, proc_w, proc_h);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, ssbo_depth_r);
    glDispatchCompute(groups_x, groups_y, 1);
    glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);

    /* Right backward colour warp */
    glUseProgram(prog_backward_color);
    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_2D, tex_right_col_rect);
    glUniform1i(uloc_bc_colour, 0);
    glUniform2i(uloc_bc_size, proc_w, proc_h);
    glUniform1f(uloc_bc_shift, right_shift);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, ssbo_depth_r);
    glBindImageTexture(1, tex_output_r, 0, GL_FALSE, 0, GL_WRITE_ONLY, GL_RGBA8);
    glDispatchCompute(groups_x, groups_y, 1);
    glMemoryBarrier(GL_SHADER_IMAGE_ACCESS_BARRIER_BIT);

    /* Composite: merge left + right into tex_filled, update left SSBO with composite depth.
     * We write to tex_filled here (not tex_output) to avoid same-texture read-write hazard.
     * After this pass: tex_filled holds the composite colour; ssbo_depth holds composite depth. */
    glUseProgram(prog_composite);
    glUniform2i(uloc_comp_size, proc_w, proc_h);
    glUniform1f(uloc_comp_shift, u_shift);
    glBindImageTexture(0, tex_output,   0, GL_FALSE, 0, GL_READ_ONLY,  GL_RGBA8);
    glBindImageTexture(1, tex_output_r, 0, GL_FALSE, 0, GL_READ_ONLY,  GL_RGBA8);
    glBindImageTexture(2, tex_filled,   0, GL_FALSE, 0, GL_WRITE_ONLY, GL_RGBA8);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, ssbo_depth);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, ssbo_depth_r);
    glDispatchCompute(groups_x, groups_y, 1);
    glMemoryBarrier(GL_SHADER_IMAGE_ACCESS_BARRIER_BIT | GL_SHADER_STORAGE_BARRIER_BIT);

    /* ---- Pass 3: Hole fill ---- */
    /* After composite: tex_filled = composite colour, ssbo_depth = composite depth.
     * Hole fill reads from tex_filled and writes to tex_output. Display from tex_output. */
    GLuint display_tex;
    if (use_hole_fill) {
        glUseProgram(prog_hole_fill);
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, ssbo_depth);
        glBindImageTexture(1, tex_filled, 0, GL_FALSE, 0, GL_READ_ONLY,  GL_RGBA8);
        glBindImageTexture(2, tex_output, 0, GL_FALSE, 0, GL_WRITE_ONLY, GL_RGBA8);
        glUniform1i(uloc_hf_maxsearch, HOLE_FILL_MAX_SEARCH);
        glUniform2i(uloc_hf_size, proc_w, proc_h);
        glDispatchCompute(groups_x, groups_y, 1);
        glMemoryBarrier(GL_SHADER_IMAGE_ACCESS_BARRIER_BIT);
        display_tex = tex_output;
    } else {
        display_tex = tex_filled;
    }

    /* ---- Render fullscreen quad ---- */
    glViewport(0, 0, width(), height());
    glClear(GL_COLOR_BUFFER_BIT);
    glUseProgram(prog_display);
    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_2D, display_tex);
    glUniform1i(uloc_disp_texture, 0);
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

    char info[320];
    snprintf(info, sizeof(info),
             "FPS: %.1f | %s nDisp=%d blk=%d WLS=%s Hole=%s | %dx%d | Track:%s shift=%.2f",
             current_fps,
             use_sgbm ? "SGBM" : "BM",
             num_disparities, block_size,
             use_wls ? "ON" : "OFF",
             use_hole_fill ? "ON" : "OFF",
             proc_w, proc_h,
             (face_tracking_enabled && face_tracker && face_tracker->isActive())
                 ? "ON" : "OFF",
             u_shift);

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
        printf("FPS: %.1f  |  %s  numDisp=%d  block=%d  WLS=%s  hole=%s  proc=%dx%d  track=%s  shift=%.2f\n",
               current_fps,
               use_sgbm ? "SGBM" : "BM",
               num_disparities, block_size,
               use_wls ? "ON" : "OFF",
               use_hole_fill ? "ON" : "OFF",
               proc_w, proc_h,
               (face_tracking_enabled && face_tracker && face_tracker->isActive())
                   ? "ON" : "OFF",
               u_shift);
        last_fps_print = now;
    }

    /* Re-arm the render loop — singleShot(0) posts to the event queue so
     * other Qt events (key presses etc.) are processed between frames. */
    if (render_running)
        QTimer::singleShot(0, this, [this](){ update(); });
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

    SynthWindow win(calib);
    win.resize(960, 540);
    win.show();
    return app.exec();
}

#include "view_synthesis.moc"
