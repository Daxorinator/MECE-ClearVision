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
 *   ./view_synthesis
 *
 * Controls:
 *   ESC   - Exit
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
#include <opencv2/cudaarithm.hpp>
#include <opencv2/cudaimgproc.hpp>
#include <opencv2/cudawarping.hpp>

#include "face_tracker.h"
#include "oak_receiver.h"

#include <QApplication>
#include <QOpenGLWidget>
#include <QOpenGLFunctions>
#include <QTimer>
#include <QKeyEvent>
#include <QPainter>
#include <QSurfaceFormat>

#include <GLES3/gl31.h>

/* ========================================================================
 * Configuration
 * ======================================================================== */

#define CAMERA_WIDTH        1920
#define CAMERA_HEIGHT       1080
#define FPS_WINDOW          24
#define FPS_PRINT_INTERVAL  2.0
#define HOLE_FILL_MAX_SEARCH 16
#define FACE_CAM_INDEX      2   // V4L2 index of the USB viewer-facing webcam

/* ========================================================================
 * GLSL Compute Shaders (OpenGL ES 3.1)
 * ======================================================================== */

static const char *DEPTH_SPLAT_CS = R"(#version 310 es
layout(local_size_x = 16, local_size_y = 16) in;

uniform sampler2D u_disparity;
uniform float u_shift;
uniform float u_disp_scale;
uniform ivec2 u_output_size;
layout(std430, binding = 0) buffer DepthBuffer { uint depth[]; };

void main() {
    ivec2 src = ivec2(gl_GlobalInvocationID.xy);
    if (src.x >= u_output_size.x || src.y >= u_output_size.y) return;
    float disp = texelFetch(u_disparity, src, 0).r * u_disp_scale;
    if (disp < 0.5) return;

    float dst_xf = float(src.x) - disp * u_shift;
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
    int dst_x = int(round(float(src.x) - disp * u_shift));
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
    float src_xf = float(dst.x) + warped_disp * u_shift;
    float src_yf = float(dst.y);

    if (src_xf < 0.0 || src_xf >= float(u_output_size.x) - 1.0) return;

    vec2 uv = vec2((src_xf + 0.5) / float(u_output_size.x),
                   (src_yf + 0.5) / float(u_output_size.y));

    vec4 col = texture(u_colour, uv);
    imageStore(u_output, dst, col);
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
    SynthWindow(QWidget *parent = nullptr);
    ~SynthWindow();

protected:
    void initializeGL() override;
    void paintGL() override;
    void resizeGL(int w, int h) override;
    void keyPressEvent(QKeyEvent *e) override;

private:
    OAKReceiver oak_receiver;

    bool use_hole_fill;

    /* Processing scale */
    double proc_scale;
    int proc_w, proc_h;

    /* GPU resources */
    GLuint prog_depth_splat, prog_color_splat, prog_hole_fill, prog_display;
    GLuint prog_clear{0};
    GLuint prog_backward_color{0};
    GLint  uloc_clear_total{-1};
    GLuint tex_left_color, tex_right_color;
    GLuint tex_left_disp, tex_right_disp;
    GLuint tex_output, tex_filled;
    GLuint ssbo_depth;
    GLuint quad_vao, quad_vbo;
    bool gl_ready;
    bool diag_frames_logged{false};   // one-shot: first frame received


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
    cv::Mat m_left_rgba;

    /* CUDA acceleration */
    bool use_cuda{false};
    cv::cuda::GpuMat gpu_disp_float;
    cv::cuda::GpuMat disp_filtered_gpu;
    bool             disp_filtered_init{false};

    /* Cached GL uniform locations (set once in initializeGL) */
    GLint uloc_ds_disparity{-1}, uloc_ds_shift{-1}, uloc_ds_size{-1}, uloc_ds_scale{-1};
    GLint uloc_cs_color{-1}, uloc_cs_disp{-1}, uloc_cs_shift{-1}, uloc_cs_size{-1};
    GLint uloc_hf_maxsearch{-1}, uloc_hf_size{-1};
    GLint uloc_disp_texture{-1};
    GLint uloc_bc_colour{-1}, uloc_bc_size{-1}, uloc_bc_shift{-1};

    void recreateTextures();
    void clearGPUBuffers();
};

/* ---- constructor ---- */

SynthWindow::SynthWindow(QWidget *parent)
    : QOpenGLWidget(parent),
      use_hole_fill(true),
      proc_scale(0.5),
      prog_depth_splat(0), prog_color_splat(0), prog_hole_fill(0), prog_display(0), prog_backward_color(0),
      tex_left_color(0), tex_right_color(0),
      tex_left_disp(0), tex_right_disp(0),
      tex_output(0), tex_filled(0), ssbo_depth(0),
      quad_vao(0), quad_vbo(0),
      gl_ready(false), current_fps(0.0)
{
    setWindowTitle("View Synthesis (DIBR)");

    proc_w = (int)(CAMERA_WIDTH  * proc_scale);
    proc_h = (int)(CAMERA_HEIGHT * proc_scale);

    if (cv::cuda::getCudaEnabledDeviceCount() > 0) {
        use_cuda = true;
        printf("CUDA available — GPU temporal filter enabled\n");
    } else {
        printf("No CUDA — CPU temporal filter fallback\n");
    }

    oak_receiver.start();

    printf("\n============================================================\n");
    printf("VIEW SYNTHESIS (DIBR)\n");
    printf("============================================================\n");
    printf("Resolution:      %dx%d\n", CAMERA_WIDTH, CAMERA_HEIGHT);
    printf("Processing:      %dx%d (%.0f%%)\n", proc_w, proc_h, proc_scale * 100.0);
    printf("Hole filling:    %s\n", use_hole_fill ? "ON" : "OFF");
    printf("============================================================\n");
    printf("\nControls:\n");
    printf("  ESC   - Exit\n");
    printf("  H     - Toggle hole filling\n");
    printf("  1/2/3 - Scale: 1.0x / 0.5x / 0.25x\n");
    printf("  F     - Toggle face-tracking shift\n");
    printf("  C     - Recalibrate face tracker (look straight ahead first)\n");
    printf("============================================================\n\n");

    /* ---- Face tracker (USB webcam, background thread) ---- */
    {
        // Look for the Haar frontal-face cascade alongside the binary or in
        // standard OpenCV system data directories.
        const char *cascade_candidates[] = {
            "haarcascade_frontalface_alt2.xml",
            "/usr/share/opencv4/haarcascades/haarcascade_frontalface_alt2.xml",
            "/usr/local/share/opencv4/haarcascades/haarcascade_frontalface_alt2.xml",
            "/usr/share/opencv/haarcascades/haarcascade_frontalface_alt2.xml",
        };
        std::string cascade_path;
        for (const char *c : cascade_candidates) {
            if (FILE *f = std::fopen(c, "rb")) {
                std::fclose(f);
                cascade_path = c;
                break;
            }
        }

        if (cascade_path.empty()) {
            printf("[FaceTracker] Haar cascade not found — face tracking disabled.\n");
            printf("  Expected: haarcascade_frontalface_alt2.xml\n");
            printf("  Install: sudo apt-get install opencv-data\n");
        } else {
            face_tracker = new FaceTracker();
            if (face_tracker->start(FACE_CAM_INDEX, cascade_path)) {
                face_tracking_enabled = true;
                printf("[FaceTracker] Started on camera %d  cascade: %s\n",
                       FACE_CAM_INDEX, cascade_path.c_str());
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
    oak_receiver.stop();
    makeCurrent();
    if (prog_depth_splat) glDeleteProgram(prog_depth_splat);
    if (prog_color_splat) glDeleteProgram(prog_color_splat);
    if (prog_hole_fill)   glDeleteProgram(prog_hole_fill);
    if (prog_display)        glDeleteProgram(prog_display);
    if (prog_backward_color) glDeleteProgram(prog_backward_color);
    if (prog_clear)       glDeleteProgram(prog_clear);
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

    clear_black.assign(proc_w * proc_h * 4, 0);
    m_disp_l_float.create(proc_h, proc_w, CV_32F);
    m_left_rgba.create(proc_h, proc_w, CV_8UC4);

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

    if (!prog_depth_splat || !prog_color_splat || !prog_hole_fill || !prog_display || !prog_clear || !prog_backward_color) {
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
    uloc_ds_scale     = glGetUniformLocation(prog_depth_splat, "u_disp_scale");
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

    if (e->key() == Qt::Key_1) {
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
        printf("scale=%.2f  hole=%s\n",
               proc_scale,
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

    /* 1. Grab frame from OAK-D Lite */
    OAKFrame oak_frame;
    if (!oak_receiver.getFrame(oak_frame)) { update(); return; }
    cv::Mat &frame_l = oak_frame.color;
    // Convert disparity from CV_16U subpixel (divide by 32) to CV_32F
    cv::Mat disp_l_float;
    oak_frame.disparity.convertTo(disp_l_float, CV_32F, 1.0f / 32.0f);

    if (!diag_frames_logged) {
        printf("[DIAG] First frame: color=%dx%d type=%d  disp=%dx%d type=%d  proc=%dx%d\n",
               frame_l.cols, frame_l.rows, frame_l.type(),
               oak_frame.disparity.cols, oak_frame.disparity.rows, oak_frame.disparity.type(),
               proc_w, proc_h);
        printf("[DIAG] use_cuda=%d\n", use_cuda);
        diag_frames_logged = true;
    }

    /* Downscale OAK-D frames (1920×1080) to processing resolution */
    cv::Mat frame_proc;
    cv::resize(frame_l, frame_proc, cv::Size(proc_w, proc_h), 0, 0, cv::INTER_LINEAR);
    cv::resize(disp_l_float, disp_l_float, cv::Size(proc_w, proc_h), 0, 0, cv::INTER_AREA);

    /* 2. Convert left BGR → RGBA for GPU upload */
    cv::Mat &left_rgba = m_left_rgba;
    cv::cvtColor(frame_proc, left_rgba, cv::COLOR_BGR2RGBA);

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

    /* 3. Upload left color + disparity to GL */
    glBindTexture(GL_TEXTURE_2D, tex_left_color);
    glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, proc_w, proc_h,
                    GL_RGBA, GL_UNSIGNED_BYTE, left_rgba.data);

    glBindTexture(GL_TEXTURE_2D, tex_left_disp);
    glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, proc_w, proc_h,
                    GL_RED, GL_FLOAT, disp_l_float.data);

    /* 4. Clear depth buffer and output */
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
    glUniform1f(uloc_ds_scale, 1.0f);
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

    /* ---- Pass 3: Hole fill ---- */
    GLuint display_tex;
    if (use_hole_fill) {
        glUseProgram(prog_hole_fill);
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, ssbo_depth);
        glBindImageTexture(1, tex_output, 0, GL_FALSE, 0, GL_READ_ONLY, GL_RGBA8);
        glBindImageTexture(2, tex_filled, 0, GL_FALSE, 0, GL_WRITE_ONLY, GL_RGBA8);
        glUniform1i(uloc_hf_maxsearch, HOLE_FILL_MAX_SEARCH);
        glUniform2i(uloc_hf_size, proc_w, proc_h);
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
             "FPS: %.1f | Hole=%s | %dx%d | Track:%s shift=%.2f",
             current_fps,
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
        double dmin, dmax;
        cv::minMaxLoc(disp_l_float, &dmin, &dmax);
        int nonzero = cv::countNonZero(disp_l_float > 0.5f);
        printf("FPS: %.1f  |  hole=%s  proc=%dx%d  track=%s  shift=%.2f  disp=[%.1f..%.1f] valid=%d/%d\n",
               current_fps,
               use_hole_fill ? "ON" : "OFF",
               proc_w, proc_h,
               (face_tracking_enabled && face_tracker && face_tracker->isActive())
                   ? "ON" : "OFF",
               u_shift,
               dmin, dmax, nonzero, proc_w * proc_h);
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

    SynthWindow win;
    win.resize(960, 540);
    win.show();
    return app.exec();
}

#include "view_synthesis.moc"
