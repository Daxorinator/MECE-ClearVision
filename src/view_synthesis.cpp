/*
 * View Synthesis: DIBR (Depth-Image Based Rendering) via GPU Compute Shaders
 *
 * Synthesises a novel viewpoint from a virtual camera positioned at the
 * viewer's head position, projecting through a virtual window model.
 * Uses OAK-D Lite hardware disparity → 3D unproject → virtual window splat
 * → JFA hole fill.
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
#include <opencv2/calib3d.hpp>
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

#define CAMERA_WIDTH        640
#define CAMERA_HEIGHT       480
#define FPS_WINDOW          24
#define FPS_PRINT_INTERVAL  2.0
#define FACE_CAM_INDEX      0

/* ========================================================================
 * GLSL Compute Shaders (OpenGL ES 3.1)
 * ======================================================================== */

/* Unprojects disparity texture to world-space 3D points.
 * Output rgba32f: (X, Y, Z, 1) in metres; w=0 if invalid. */
static const char *UNPROJECT_CS = R"(#version 310 es
precision highp float;
layout(local_size_x = 16, local_size_y = 16) in;

uniform sampler2D u_disparity;
uniform float u_fx, u_fy, u_cx, u_cy;
uniform float u_baseline;
uniform ivec2 u_size;
layout(rgba32f, binding = 0) writeonly uniform highp image2D u_worldspace;

void main() {
    ivec2 pos = ivec2(gl_GlobalInvocationID.xy);
    if (pos.x >= u_size.x || pos.y >= u_size.y) return;
    float d = texelFetch(u_disparity, pos, 0).r;
    if (d < 0.5) { imageStore(u_worldspace, pos, vec4(0.0)); return; }
    float Z = u_fx * u_baseline / d;
    float X = (float(pos.x) - u_cx) * Z / u_fx;
    float Y = (float(pos.y) - u_cy) * Z / u_fy;
    imageStore(u_worldspace, pos, vec4(X, Y, Z, 1.0));
}
)";

/* Splats world-space points into the depth SSBO via virtual-window ray projection.
 * For each world point W, casts a ray from the viewer's head through W and finds
 * where it intersects the physical display plane.  Maps intersection UV to the
 * output texture (out_w × out_h) and stores 1/dist for atomicMax depth test. */
static const char *VWINDOW_DEPTH_CS = R"(#version 310 es
precision highp float;
layout(local_size_x = 16, local_size_y = 16) in;

layout(rgba32f, binding = 0) readonly uniform highp image2D u_worldspace;
uniform vec3  u_head_pos;
uniform vec3  u_display_pos;
uniform vec3  u_display_normal;   /* points from display toward scene */
uniform vec3  u_display_right;    /* display +X direction in world space */
uniform vec3  u_display_up;       /* display +Y direction in world space */
uniform vec2  u_display_size_m;   /* (width_m, height_m) */
uniform ivec2 u_src_size;         /* source texture dims (proc_w × proc_h) */
uniform ivec2 u_out_size;         /* output texture dims (out_w × out_h) */
layout(std430, binding = 0) buffer DepthBuffer { uint depth[]; };

void main() {
    ivec2 src = ivec2(gl_GlobalInvocationID.xy);
    if (src.x >= u_src_size.x || src.y >= u_src_size.y) return;
    vec4 wp = imageLoad(u_worldspace, src);
    if (wp.w < 0.5) return;
    vec3 W   = wp.xyz;
    vec3 dir = normalize(W - u_head_pos);
    float denom = dot(dir, u_display_normal);
    if (abs(denom) < 1e-6) return;
    float t = dot(u_display_pos - u_head_pos, u_display_normal) / denom;
    if (t <= 0.0) return;
    vec3 Q     = u_head_pos + t * dir;
    vec3 Q_rel = Q - u_display_pos;
    float u_coord = dot(Q_rel, u_display_right) / u_display_size_m.x + 0.5;
    float v_coord = dot(Q_rel, u_display_up)    / u_display_size_m.y + 0.5;
    if (u_coord < 0.0 || u_coord >= 1.0 || v_coord < 0.0 || v_coord >= 1.0) return;
    int dst_x = int(u_coord * float(u_out_size.x));
    int dst_y = int(v_coord * float(u_out_size.y));
    if (dst_x < 0 || dst_x >= u_out_size.x || dst_y < 0 || dst_y >= u_out_size.y) return;
    float dist = length(W - u_head_pos);
    atomicMax(depth[dst_y * u_out_size.x + dst_x], floatBitsToUint(1.0 / dist));
}
)";

/* Writes colour for each world point that wins its depth bucket.
 * Uses virtual-window ray projection: same intersection math as VWINDOW_DEPTH_CS. */
static const char *VWINDOW_COLOR_CS = R"(#version 310 es
precision highp float;
layout(local_size_x = 16, local_size_y = 16) in;

layout(rgba32f, binding = 0) readonly uniform highp image2D u_worldspace;
uniform sampler2D u_colour;
uniform vec3  u_head_pos;
uniform vec3  u_display_pos;
uniform vec3  u_display_normal;
uniform vec3  u_display_right;
uniform vec3  u_display_up;
uniform vec2  u_display_size_m;
uniform ivec2 u_src_size;
uniform ivec2 u_out_size;
layout(std430, binding = 0) readonly buffer DepthBuffer { uint depth[]; };
layout(rgba8, binding = 1) writeonly uniform highp image2D u_output;

void main() {
    ivec2 src = ivec2(gl_GlobalInvocationID.xy);
    if (src.x >= u_src_size.x || src.y >= u_src_size.y) return;
    vec4 wp = imageLoad(u_worldspace, src);
    if (wp.w < 0.5) return;
    vec3 W   = wp.xyz;
    vec3 dir = normalize(W - u_head_pos);
    float denom = dot(dir, u_display_normal);
    if (abs(denom) < 1e-6) return;
    float t = dot(u_display_pos - u_head_pos, u_display_normal) / denom;
    if (t <= 0.0) return;
    vec3 Q     = u_head_pos + t * dir;
    vec3 Q_rel = Q - u_display_pos;
    float u_coord = dot(Q_rel, u_display_right) / u_display_size_m.x + 0.5;
    float v_coord = dot(Q_rel, u_display_up)    / u_display_size_m.y + 0.5;
    if (u_coord < 0.0 || u_coord >= 1.0 || v_coord < 0.0 || v_coord >= 1.0) return;
    int dst_x = int(u_coord * float(u_out_size.x));
    int dst_y = int(v_coord * float(u_out_size.y));
    if (dst_x < 0 || dst_x >= u_out_size.x || dst_y < 0 || dst_y >= u_out_size.y) return;
    float dist     = length(W - u_head_pos);
    uint  my_depth = floatBitsToUint(1.0 / dist);
    if (my_depth < depth[dst_y * u_out_size.x + dst_x]) return;
    vec2 uv = (vec2(src) + 0.5) / vec2(u_src_size);
    imageStore(u_output, ivec2(dst_x, dst_y), texture(u_colour, uv));
}
)";

/* Initialises JFA seed image from depth SSBO.
 * Valid pixels store packed (x | y<<16); holes store -1. */
static const char *JFA_INIT_CS = R"(#version 310 es
precision highp float;
layout(local_size_x = 16, local_size_y = 16) in;

uniform ivec2 u_size;
layout(std430, binding = 0) readonly buffer DepthBuffer { uint depth[]; };
layout(r32i, binding = 0) writeonly uniform highp iimage2D u_seed;

void main() {
    ivec2 pos = ivec2(gl_GlobalInvocationID.xy);
    if (pos.x >= u_size.x || pos.y >= u_size.y) return;
    if (depth[pos.y * u_size.x + pos.x] > 0u)
        imageStore(u_seed, pos, ivec4(pos.x | (pos.y << 16), 0, 0, 0));
    else
        imageStore(u_seed, pos, ivec4(-1, 0, 0, 0));
}
)";

/* One JFA pass — 9-neighbourhood lookup at current step distance.
 * Seed packed as (x | y<<16) in r32i; -1 = no seed. */
static const char *JFA_CS = R"(#version 310 es
precision highp float;
layout(local_size_x = 16, local_size_y = 16) in;

uniform int u_step;
uniform ivec2 u_size;
layout(r32i, binding = 0) readonly  uniform highp iimage2D u_seed_in;
layout(r32i, binding = 1) writeonly uniform highp iimage2D u_seed_out;

ivec2 unpack(int p) { return ivec2(p & 0xFFFF, (p >> 16) & 0xFFFF); }

void main() {
    ivec2 pos = ivec2(gl_GlobalInvocationID.xy);
    if (pos.x >= u_size.x || pos.y >= u_size.y) return;
    int best_p = imageLoad(u_seed_in, pos).r;
    ivec2 best = (best_p >= 0) ? unpack(best_p) : ivec2(-1);
    float best_dist = (best_p >= 0) ? length(vec2(pos - best)) : 1e30;
    for (int dy = -1; dy <= 1; dy++) {
        for (int dx = -1; dx <= 1; dx++) {
            if (dx == 0 && dy == 0) continue;
            ivec2 nb = pos + ivec2(dx, dy) * u_step;
            if (nb.x < 0 || nb.x >= u_size.x || nb.y < 0 || nb.y >= u_size.y) continue;
            int p = imageLoad(u_seed_in, nb).r;
            if (p < 0) continue;
            ivec2 seed = unpack(p);
            float d = length(vec2(pos - seed));
            if (d < best_dist) { best_dist = d; best_p = p; best = seed; }
        }
    }
    imageStore(u_seed_out, pos, ivec4(best_p, 0, 0, 0));
}
)";

/* Copies colour from nearest valid seed into the filled output texture. */
static const char *JFA_GATHER_CS = R"(#version 310 es
precision highp float;
layout(local_size_x = 16, local_size_y = 16) in;

uniform ivec2 u_size;
layout(r32i, binding = 0) readonly  uniform highp iimage2D u_seed;
layout(rgba8, binding = 1) readonly  uniform highp image2D  u_src;
layout(rgba8, binding = 2) writeonly uniform highp image2D  u_out;

void main() {
    ivec2 pos = ivec2(gl_GlobalInvocationID.xy);
    if (pos.x >= u_size.x || pos.y >= u_size.y) return;
    int p = imageLoad(u_seed, pos).r;
    vec4 col = (p >= 0) ? imageLoad(u_src, ivec2(p & 0xFFFF, (p >> 16) & 0xFFFF)) : vec4(0.0);
    imageStore(u_out, pos, col);
}
)";

/* Right-eye disocclusion fill.
 * Iterates over right-camera pixels using a CPU-reprojected right disparity map.
 * Disocclusion filter ensures only true disocclusions (not textureless areas) are filled.
 * Uses virtual-window ray projection to find display UV; writes only into holes. */
static const char *VWINDOW_RIGHT_CS = R"(#version 310 es
precision highp float;
layout(local_size_x = 16, local_size_y = 16) in;

uniform sampler2D u_right_colour;
uniform sampler2D u_disparity;      /* right-camera disparity (CPU-reprojected) */
uniform sampler2D u_left_disp;      /* original left-camera disparity */
uniform vec3  u_head_pos;
uniform float u_fx, u_fy, u_cx, u_cy;
uniform float u_baseline;
uniform vec3  u_display_pos;
uniform vec3  u_display_normal;
uniform vec3  u_display_right;
uniform vec3  u_display_up;
uniform vec2  u_display_size_m;
uniform ivec2 u_src_size;
uniform ivec2 u_out_size;
layout(std430, binding = 0) buffer DepthBuffer { uint depth[]; };
layout(rgba8, binding = 1) writeonly uniform highp image2D u_output;

void main() {
    ivec2 src = ivec2(gl_GlobalInvocationID.xy);
    if (src.x >= u_src_size.x || src.y >= u_src_size.y) return;
    float d = texelFetch(u_disparity, src, 0).r;
    if (d < 0.5) return;
    /* Disocclusion filter: skip textureless areas (d_left≈0) and same-world-point
     * pixels where both cameras see the same surface depth. */
    float d_left = texelFetch(u_left_disp, src, 0).r;
    if (d_left < 0.5 || d >= d_left * 0.7) return;
    float Z  = u_fx * u_baseline / d;
    /* Right camera origin is +baseline in X relative to left camera. */
    float Xw = (float(src.x) - u_cx) * Z / u_fx + u_baseline;
    float Yw = (float(src.y) - u_cy) * Z / u_fy;
    vec3 W   = vec3(Xw, Yw, Z);
    vec3 dir = normalize(W - u_head_pos);
    float denom = dot(dir, u_display_normal);
    if (abs(denom) < 1e-6) return;
    float t = dot(u_display_pos - u_head_pos, u_display_normal) / denom;
    if (t <= 0.0) return;
    vec3 Q     = u_head_pos + t * dir;
    vec3 Q_rel = Q - u_display_pos;
    float u_coord = dot(Q_rel, u_display_right) / u_display_size_m.x + 0.5;
    float v_coord = dot(Q_rel, u_display_up)    / u_display_size_m.y + 0.5;
    if (u_coord < 0.0 || u_coord >= 1.0 || v_coord < 0.0 || v_coord >= 1.0) return;
    int dst_x = int(u_coord * float(u_out_size.x));
    int dst_y = int(v_coord * float(u_out_size.y));
    if (dst_x < 0 || dst_x >= u_out_size.x || dst_y < 0 || dst_y >= u_out_size.y) return;
    if (depth[dst_y * u_out_size.x + dst_x] != 0u) return;
    float dist = length(W - u_head_pos);
    atomicMax(depth[dst_y * u_out_size.x + dst_x], floatBitsToUint(1.0 / dist));
    vec2 uv = (vec2(src) + 0.5) / vec2(u_src_size);
    imageStore(u_output, ivec2(dst_x, dst_y), texture(u_right_colour, uv));
}
)";

/* Zeroes the depth SSBO on the GPU — avoids uploading a multi-MB zero vector. */
static const char *CLEAR_CS = R"(#version 310 es
layout(local_size_x = 64) in;
layout(std430, binding = 0) buffer DepthBuf { uint depth[]; };
uniform int u_total;
void main() {
    uint idx = gl_GlobalInvocationID.x;
    if (idx < uint(u_total)) depth[idx] = 0u;
}
)";

static const char *DISPLAY_VS = R"(#version 310 es
layout(location = 0) in vec2 a_pos;
layout(location = 1) in vec2 a_uv;
uniform vec2 u_scale;   /* NDC scale to letterbox/pillarbox output to correct aspect */
out vec2 v_uv;
void main() { v_uv = a_uv; gl_Position = vec4(a_pos * u_scale, 0.0, 1.0); }
)";

static const char *DISPLAY_FS = R"(#version 310 es
precision mediump float;
in vec2 v_uv;
out vec4 frag_color;
uniform sampler2D u_texture;
void main() { frag_color = texture(u_texture, v_uv); }
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

static GLuint create_texture_rgba32f(int w, int h)
{
    GLuint tex;
    glGenTextures(1, &tex);
    glBindTexture(GL_TEXTURE_2D, tex);
    glTexStorage2D(GL_TEXTURE_2D, 1, GL_RGBA32F, w, h);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    return tex;
}

static GLuint create_texture_r32i(int w, int h)
{
    GLuint tex;
    glGenTextures(1, &tex);
    glBindTexture(GL_TEXTURE_2D, tex);
    glTexStorage2D(GL_TEXTURE_2D, 1, GL_R32I, w, h);
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

    bool use_hole_fill{true};

    double proc_scale{1.0};
    int proc_w, proc_h;

    /* Physical display geometry — bench-test defaults.
     * pos/normal/right/up in OAK-D world space (metres).
     * display is 0.8 m in front of OAK-D; viewer (head) is at z≈0 (OAK-D position).
     * normal points from display toward viewer (−Z in OAK-D frame). */
    struct DisplayConfig {
        float pos[3]     = {0.f, 0.f, 0.4f};
        float normal[3]  = {0.f, 0.f,-1.f};   /* toward viewer */
        float right_v[3] = {1.f, 0.f, 0.f};
        float up_v[3]    = {0.f, 1.f, 0.f};   /* +Y (cam-space down) so dst_y=0 = screen top */
        float width_m    = 0.133f;             /* 480/1920 × 532 mm */
        float height_m   = 0.300f;
    } display_config;

    int out_w{480};
    int out_h{1080};

    /* GL programs */
    GLuint prog_unproject{0};
    GLuint prog_vwindow_depth{0};
    GLuint prog_vwindow_color{0};
    GLuint prog_vwindow_right{0};
    GLuint prog_jfa_init{0};
    GLuint prog_jfa{0};
    GLuint prog_jfa_gather{0};
    GLuint prog_clear{0};
    GLuint prog_display{0};

    /* GL textures */
    GLuint tex_left_color{0};   /* rgba8,   proc_w × proc_h — left colour input */
    GLuint tex_right_color{0};  /* rgba8,   proc_w × proc_h — right rect (gray→rgba) */
    GLuint tex_left_disp{0};    /* r32f,    proc_w × proc_h — disparity input */
    GLuint tex_right_disp{0};   /* r32f,    proc_w × proc_h — right-camera disparity */
    GLuint tex_worldspace{0};   /* rgba32f, proc_w × proc_h — unprojected 3D */
    GLuint tex_output{0};       /* rgba8,   out_w  × out_h  — virtual-window splat */
    GLuint tex_filled{0};       /* rgba8,   out_w  × out_h  — JFA filled */
    GLuint tex_jfa_a{0};        /* r32i,    out_w  × out_h  — JFA ping */
    GLuint tex_jfa_b{0};        /* r32i,    out_w  × out_h  — JFA pong */

    GLuint ssbo_depth{0};
    GLuint quad_vao{0}, quad_vbo{0};
    bool gl_ready{false};
    bool diag_frames_logged{false};

    bool render_running{false};

    /* Face tracking */
    FaceTracker *face_tracker{nullptr};
    bool face_tracking_enabled{false};

    /* FPS tracking */
    std::deque<std::chrono::steady_clock::time_point> frame_times;
    std::chrono::steady_clock::time_point last_fps_print;
    double current_fps{0.0};

    /* Staging buffers */
    std::vector<GLubyte> clear_black;
    cv::Mat m_left_rgba;
    cv::Mat m_right_rgba;

    /* Temporal IIR filter state */
    bool use_cuda{false};
    cv::cuda::GpuMat gpu_disp_float;
    cv::cuda::GpuMat disp_filtered_gpu;
    cv::cuda::GpuMat disp_invalid_mask_gpu;  /* scratch for zero-decay fix */
    cv::Mat          disp_filtered_cpu;       /* CPU fallback state */
    bool disp_filtered_init{false};

    /* Cached uniform locations */
    GLint uloc_clear_total{-1};
    GLint uloc_disp_texture{-1};
    GLint uloc_disp_scale{-1};      /* DISPLAY_VS: NDC scale for aspect-correct letterbox */

    GLint uloc_unproj_disp{-1}, uloc_unproj_fx{-1}, uloc_unproj_fy{-1};
    GLint uloc_unproj_cx{-1},   uloc_unproj_cy{-1}, uloc_unproj_baseline{-1};
    GLint uloc_unproj_size{-1};

    /* VWINDOW_DEPTH_CS */
    GLint uloc_vwd_head{-1};
    GLint uloc_vwd_disp_pos{-1}, uloc_vwd_disp_normal{-1};
    GLint uloc_vwd_disp_right{-1}, uloc_vwd_disp_up{-1};
    GLint uloc_vwd_disp_size{-1};
    GLint uloc_vwd_src_size{-1}, uloc_vwd_out_size{-1};

    /* VWINDOW_COLOR_CS */
    GLint uloc_vwc_colour{-1}, uloc_vwc_head{-1};
    GLint uloc_vwc_disp_pos{-1}, uloc_vwc_disp_normal{-1};
    GLint uloc_vwc_disp_right{-1}, uloc_vwc_disp_up{-1};
    GLint uloc_vwc_disp_size{-1};
    GLint uloc_vwc_src_size{-1}, uloc_vwc_out_size{-1};

    /* VWINDOW_RIGHT_CS */
    GLint uloc_vwr_right{-1}, uloc_vwr_disp{-1}, uloc_vwr_ldisp{-1}, uloc_vwr_head{-1};
    GLint uloc_vwr_fx{-1},    uloc_vwr_fy{-1},   uloc_vwr_cx{-1};
    GLint uloc_vwr_cy{-1},    uloc_vwr_baseline{-1};
    GLint uloc_vwr_disp_pos{-1}, uloc_vwr_disp_normal{-1};
    GLint uloc_vwr_disp_right{-1}, uloc_vwr_disp_up{-1};
    GLint uloc_vwr_disp_size{-1};
    GLint uloc_vwr_src_size{-1}, uloc_vwr_out_size{-1};

    GLint uloc_jfai_size{-1};
    GLint uloc_jfa_step{-1}, uloc_jfa_size{-1};
    GLint uloc_jfag_size{-1};

    void recreateTextures();
    void clearGPUBuffers();
};

/* ---- constructor ---- */

SynthWindow::SynthWindow(QWidget *parent) : QOpenGLWidget(parent)
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

    oak_receiver.want_color      = false;  /* use mono stereo images for correct coord frame */
    oak_receiver.want_left_rect  = true;
    oak_receiver.want_right_rect = true;
    oak_receiver.start();

    printf("\n============================================================\n");
    printf("VIEW SYNTHESIS (DIBR)\n");
    printf("============================================================\n");
    printf("Resolution:      %dx%d\n", CAMERA_WIDTH, CAMERA_HEIGHT);
    printf("Processing:      %dx%d (%.0f%%)\n", proc_w, proc_h, proc_scale * 100.0);
    printf("Output (display):%dx%d\n", out_w, out_h);
    printf("Display size:    %.0f × %.0f mm @ (0,0,%.2fm)\n",
           display_config.width_m * 1000.f, display_config.height_m * 1000.f,
           display_config.pos[2]);
    printf("Hole filling:    %s\n", use_hole_fill ? "ON (JFA)" : "OFF");
    printf("============================================================\n");
    printf("\nControls:\n");
    printf("  ESC   - Exit\n");
    printf("  H     - Toggle hole filling\n");
    printf("  1/2/3 - Scale: 1.0x / 0.5x / 0.25x\n");
    printf("  F     - Toggle face-tracking shift\n");
    printf("  C     - Recalibrate face tracker (look straight ahead first)\n");
    printf("============================================================\n\n");

    /* Face tracker — search for FaceMesh ONNX model */
    const char *facemesh_candidates[] = {
        "face_landmark_with_attention.onnx",
        "../src/models/face_landmark_with_attention.onnx",
        "src/models/face_landmark_with_attention.onnx",
    };
    std::string facemesh_path;
    for (const char *c : facemesh_candidates) {
        if (FILE *f = std::fopen(c, "rb")) { std::fclose(f); facemesh_path = c; break; }
    }

    if (facemesh_path.empty()) {
        printf("[FaceTracker] FaceMesh ONNX model not found — face tracking disabled.\n");
        printf("  Run:  bash scripts/download_facemesh.sh\n");
        printf("  Then place face_landmark_with_attention.onnx in src/models/\n");
    } else {
        face_tracker = new FaceTracker();
        if (face_tracker->start(FACE_CAM_INDEX, facemesh_path)) {
            face_tracking_enabled = true;
            printf("[FaceTracker] Started on camera %d  model: %s\n",
                   FACE_CAM_INDEX, facemesh_path.c_str());
        } else {
            printf("[FaceTracker] Failed to start\n");
            delete face_tracker;
            face_tracker = nullptr;
        }
    }

    last_fps_print = std::chrono::steady_clock::now();
}

/* ---- destructor ---- */

SynthWindow::~SynthWindow()
{
    if (face_tracker) { face_tracker->stop(); delete face_tracker; face_tracker = nullptr; }
    oak_receiver.stop();
    makeCurrent();
    GLuint progs[] = { prog_unproject, prog_vwindow_depth, prog_vwindow_color,
                       prog_vwindow_right,
                       prog_jfa_init, prog_jfa, prog_jfa_gather,
                       prog_clear, prog_display };
    for (auto p : progs) if (p) glDeleteProgram(p);
    GLuint textures[] = { tex_left_color, tex_right_color, tex_left_disp, tex_right_disp,
                          tex_worldspace, tex_output, tex_filled, tex_jfa_a, tex_jfa_b };
    for (auto t : textures) if (t) glDeleteTextures(1, &t);
    if (ssbo_depth) glDeleteBuffers(1, &ssbo_depth);
    if (quad_vbo)   glDeleteBuffers(1, &quad_vbo);
    if (quad_vao)   glDeleteVertexArrays(1, &quad_vao);
    doneCurrent();
}

/* ---- recreate GPU textures at current proc resolution ---- */

void SynthWindow::recreateTextures()
{
    GLuint old_tex[] = { tex_left_color, tex_right_color, tex_left_disp, tex_right_disp,
                         tex_worldspace, tex_output, tex_filled, tex_jfa_a, tex_jfa_b };
    for (auto t : old_tex) if (t) glDeleteTextures(1, &t);
    if (ssbo_depth) glDeleteBuffers(1, &ssbo_depth);

    /* Source / processing textures — sized proc_w × proc_h */
    tex_left_color  = create_texture_rgba8  (proc_w, proc_h);
    tex_right_color = create_texture_rgba8  (proc_w, proc_h);
    tex_left_disp   = create_texture_r32f   (proc_w, proc_h);
    tex_right_disp  = create_texture_r32f   (proc_w, proc_h);
    tex_worldspace  = create_texture_rgba32f(proc_w, proc_h);

    /* Output / virtual-window textures — sized out_w × out_h (display strip) */
    tex_output = create_texture_rgba8(out_w, out_h);
    tex_filled = create_texture_rgba8(out_w, out_h);
    tex_jfa_a  = create_texture_r32i (out_w, out_h);
    tex_jfa_b  = create_texture_r32i (out_w, out_h);
    ssbo_depth = create_ssbo(out_w * out_h);

    clear_black.assign(out_w * out_h * 4, 0);
    m_left_rgba.create(proc_h, proc_w, CV_8UC4);
    m_right_rgba.create(proc_h, proc_w, CV_8UC4);

    printf("GPU textures created: src %dx%d  out %dx%d\n", proc_w, proc_h, out_w, out_h);
}

/* ---- clear depth SSBO and output texture ---- */

void SynthWindow::clearGPUBuffers()
{
    glUseProgram(prog_clear);
    glUniform1i(uloc_clear_total, out_w * out_h);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, ssbo_depth);
    glDispatchCompute((out_w * out_h + 63) / 64, 1, 1);
    glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);

    glBindTexture(GL_TEXTURE_2D, tex_output);
    glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, out_w, out_h,
                    GL_RGBA, GL_UNSIGNED_BYTE, clear_black.data());
}

/* ---- GL initialisation ---- */

void SynthWindow::initializeGL()
{
    printf("OpenGL version: %s\n", glGetString(GL_VERSION));
    printf("GLSL version:   %s\n", glGetString(GL_SHADING_LANGUAGE_VERSION));
    printf("Renderer:       %s\n", glGetString(GL_RENDERER));

    prog_unproject     = create_compute_program(UNPROJECT_CS);
    prog_vwindow_depth = create_compute_program(VWINDOW_DEPTH_CS);
    prog_vwindow_color = create_compute_program(VWINDOW_COLOR_CS);
    prog_vwindow_right = create_compute_program(VWINDOW_RIGHT_CS);
    prog_jfa_init      = create_compute_program(JFA_INIT_CS);
    prog_jfa           = create_compute_program(JFA_CS);
    prog_jfa_gather    = create_compute_program(JFA_GATHER_CS);
    prog_clear         = create_compute_program(CLEAR_CS);
    prog_display       = create_render_program(DISPLAY_VS, DISPLAY_FS);

    if (!prog_unproject || !prog_vwindow_depth || !prog_vwindow_color ||
        !prog_vwindow_right ||
        !prog_jfa_init  || !prog_jfa           || !prog_jfa_gather    ||
        !prog_clear     || !prog_display) {
        fprintf(stderr, "FATAL: shader compilation failed\n");
        return;
    }

    recreateTextures();

    /* Fullscreen quad */
    float quad[] = {
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

    /* Cache uniform locations */
    uloc_clear_total    = glGetUniformLocation(prog_clear,   "u_total");
    uloc_disp_texture   = glGetUniformLocation(prog_display, "u_texture");
    uloc_disp_scale     = glGetUniformLocation(prog_display, "u_scale");

    uloc_unproj_disp     = glGetUniformLocation(prog_unproject, "u_disparity");
    uloc_unproj_fx       = glGetUniformLocation(prog_unproject, "u_fx");
    uloc_unproj_fy       = glGetUniformLocation(prog_unproject, "u_fy");
    uloc_unproj_cx       = glGetUniformLocation(prog_unproject, "u_cx");
    uloc_unproj_cy       = glGetUniformLocation(prog_unproject, "u_cy");
    uloc_unproj_baseline = glGetUniformLocation(prog_unproject, "u_baseline");
    uloc_unproj_size     = glGetUniformLocation(prog_unproject, "u_size");

    uloc_vwd_head         = glGetUniformLocation(prog_vwindow_depth, "u_head_pos");
    uloc_vwd_disp_pos     = glGetUniformLocation(prog_vwindow_depth, "u_display_pos");
    uloc_vwd_disp_normal  = glGetUniformLocation(prog_vwindow_depth, "u_display_normal");
    uloc_vwd_disp_right   = glGetUniformLocation(prog_vwindow_depth, "u_display_right");
    uloc_vwd_disp_up      = glGetUniformLocation(prog_vwindow_depth, "u_display_up");
    uloc_vwd_disp_size    = glGetUniformLocation(prog_vwindow_depth, "u_display_size_m");
    uloc_vwd_src_size     = glGetUniformLocation(prog_vwindow_depth, "u_src_size");
    uloc_vwd_out_size     = glGetUniformLocation(prog_vwindow_depth, "u_out_size");

    uloc_vwc_colour       = glGetUniformLocation(prog_vwindow_color, "u_colour");
    uloc_vwc_head         = glGetUniformLocation(prog_vwindow_color, "u_head_pos");
    uloc_vwc_disp_pos     = glGetUniformLocation(prog_vwindow_color, "u_display_pos");
    uloc_vwc_disp_normal  = glGetUniformLocation(prog_vwindow_color, "u_display_normal");
    uloc_vwc_disp_right   = glGetUniformLocation(prog_vwindow_color, "u_display_right");
    uloc_vwc_disp_up      = glGetUniformLocation(prog_vwindow_color, "u_display_up");
    uloc_vwc_disp_size    = glGetUniformLocation(prog_vwindow_color, "u_display_size_m");
    uloc_vwc_src_size     = glGetUniformLocation(prog_vwindow_color, "u_src_size");
    uloc_vwc_out_size     = glGetUniformLocation(prog_vwindow_color, "u_out_size");

    uloc_vwr_right        = glGetUniformLocation(prog_vwindow_right, "u_right_colour");
    uloc_vwr_disp         = glGetUniformLocation(prog_vwindow_right, "u_disparity");
    uloc_vwr_ldisp        = glGetUniformLocation(prog_vwindow_right, "u_left_disp");
    uloc_vwr_head         = glGetUniformLocation(prog_vwindow_right, "u_head_pos");
    uloc_vwr_fx           = glGetUniformLocation(prog_vwindow_right, "u_fx");
    uloc_vwr_fy           = glGetUniformLocation(prog_vwindow_right, "u_fy");
    uloc_vwr_cx           = glGetUniformLocation(prog_vwindow_right, "u_cx");
    uloc_vwr_cy           = glGetUniformLocation(prog_vwindow_right, "u_cy");
    uloc_vwr_baseline     = glGetUniformLocation(prog_vwindow_right, "u_baseline");
    uloc_vwr_disp_pos     = glGetUniformLocation(prog_vwindow_right, "u_display_pos");
    uloc_vwr_disp_normal  = glGetUniformLocation(prog_vwindow_right, "u_display_normal");
    uloc_vwr_disp_right   = glGetUniformLocation(prog_vwindow_right, "u_display_right");
    uloc_vwr_disp_up      = glGetUniformLocation(prog_vwindow_right, "u_display_up");
    uloc_vwr_disp_size    = glGetUniformLocation(prog_vwindow_right, "u_display_size_m");
    uloc_vwr_src_size     = glGetUniformLocation(prog_vwindow_right, "u_src_size");
    uloc_vwr_out_size     = glGetUniformLocation(prog_vwindow_right, "u_out_size");

    uloc_jfai_size = glGetUniformLocation(prog_jfa_init,   "u_size");
    uloc_jfa_step  = glGetUniformLocation(prog_jfa,        "u_step");
    uloc_jfa_size  = glGetUniformLocation(prog_jfa,        "u_size");
    uloc_jfag_size = glGetUniformLocation(prog_jfa_gather, "u_size");

    gl_ready = true;
    printf("GPU pipeline initialised\n");

    render_running = true;
    QTimer::singleShot(0, this, [this](){ update(); });
}

/* ---- key handling ---- */

void SynthWindow::keyPressEvent(QKeyEvent *e)
{
    if (e->key() == Qt::Key_Escape) { close(); return; }

    bool changed = false;
    if      (e->key() == Qt::Key_1) { proc_scale = 1.0;  changed = true; }
    else if (e->key() == Qt::Key_2) { proc_scale = 0.5;  changed = true; }
    else if (e->key() == Qt::Key_3) { proc_scale = 0.25; changed = true; }
    else if (e->key() == Qt::Key_H) {
        use_hole_fill = !use_hole_fill;
        printf("Hole filling: %s\n", use_hole_fill ? "ON (JFA)" : "OFF");
    }
    else if (e->key() == Qt::Key_F) {
        if (face_tracker) {
            face_tracking_enabled = !face_tracking_enabled;
            printf("Face tracking: %s\n", face_tracking_enabled ? "ON" : "OFF");
        } else {
            printf("Face tracking not available\n");
        }
    }
    else if (e->key() == Qt::Key_C) {
        if (face_tracker && face_tracking_enabled) {
            face_tracker->calibrate();
            printf("Face tracker recalibrated\n");
        }
    }

    if (changed) {
        int new_w = (int)(CAMERA_WIDTH  * proc_scale);
        int new_h = (int)(CAMERA_HEIGHT * proc_scale);
        if (new_w != proc_w || new_h != proc_h) {
            proc_w = new_w; proc_h = new_h;
            if (gl_ready) { makeCurrent(); recreateTextures(); doneCurrent(); }
        }
        printf("scale=%.2f  hole=%s\n", proc_scale, use_hole_fill ? "ON" : "OFF");
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
    if (!gl_ready) return;

    OAKFrame oak_frame;
    if (!oak_receiver.getFrame(oak_frame)) { update(); return; }
    if (oak_frame.left_rect.empty()) { update(); return; }

    /* Speckle filter on raw CV_16U disparity — removes isolated false-match clusters
     * before temporal filtering.  filterSpeckles requires CV_16SC1; OAK-D subpixel
     * values are always positive so the reinterpret is safe. */
    cv::Mat disp_s16;
    oak_frame.disparity.convertTo(disp_s16, CV_16SC1);
    cv::filterSpeckles(disp_s16, 0, 150, 16);

    /* Convert disparity CV_16S subpixel (÷32) → CV_32F pixel disparity */
    cv::Mat disp_l_float;
    disp_s16.convertTo(disp_l_float, CV_32F, 1.0f / 32.0f);

    if (!diag_frames_logged) {
        printf("[DIAG] First frame: left=%dx%d  disp=%dx%d type=%d  proc=%dx%d  cuda=%d\n",
               oak_frame.left_rect.cols, oak_frame.left_rect.rows,
               oak_frame.disparity.cols, oak_frame.disparity.rows,
               oak_frame.disparity.type(), proc_w, proc_h, use_cuda);
        diag_frames_logged = true;
    }

    /* Grayscale rectified left → RGBA, resize to processing resolution */
    {
        cv::Mat rgba_full;
        cv::cvtColor(oak_frame.left_rect, rgba_full, cv::COLOR_GRAY2RGBA);
        cv::resize(rgba_full, m_left_rgba, cv::Size(proc_w, proc_h), 0, 0, cv::INTER_LINEAR);
    }
    cv::resize(disp_l_float, disp_l_float, cv::Size(proc_w, proc_h), 0, 0, cv::INTER_AREA);

    /* IIR temporal filter on disparity.
     * Critical: do NOT blend invalid (zero) new pixels into the running state —
     * that would decay valid depth values toward zero, growing holes over time.
     * Where new disparity is zero, preserve the previous filtered value unchanged. */
    const float alpha = 0.3f;
    if (use_cuda) {
        gpu_disp_float.upload(disp_l_float);
        if (!disp_filtered_init || disp_filtered_gpu.size() != gpu_disp_float.size()) {
            gpu_disp_float.copyTo(disp_filtered_gpu);
            disp_filtered_init = true;
        } else {
            cv::cuda::GpuMat blended;
            cv::cuda::addWeighted(gpu_disp_float, alpha, disp_filtered_gpu, 1.0f - alpha,
                                  0.0f, blended);
            /* Restore previous state where new disparity is zero (invalid). */
            cv::cuda::threshold(gpu_disp_float, disp_invalid_mask_gpu,
                                0.5, 255.0, cv::THRESH_BINARY_INV);
            disp_invalid_mask_gpu.convertTo(disp_invalid_mask_gpu, CV_8U);
            disp_filtered_gpu.copyTo(blended, disp_invalid_mask_gpu);
            blended.copyTo(disp_filtered_gpu);
        }
        disp_filtered_gpu.download(disp_l_float);
    } else {
        if (!disp_filtered_init || disp_filtered_cpu.size() != disp_l_float.size()) {
            disp_l_float.copyTo(disp_filtered_cpu);
            disp_filtered_init = true;
        } else {
            cv::Mat blended;
            cv::addWeighted(disp_l_float, alpha, disp_filtered_cpu, 1.0f - alpha,
                            0.0f, blended);
            cv::Mat invalid = (disp_l_float == 0.0f);
            disp_filtered_cpu.copyTo(blended, invalid);
            blended.copyTo(disp_filtered_cpu);
        }
        disp_filtered_cpu.copyTo(disp_l_float);
    }

    /* Upload colour + disparity textures */
    glBindTexture(GL_TEXTURE_2D, tex_left_color);
    glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, proc_w, proc_h,
                    GL_RGBA, GL_UNSIGNED_BYTE, m_left_rgba.data);

    glBindTexture(GL_TEXTURE_2D, tex_left_disp);
    glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, proc_w, proc_h,
                    GL_RED, GL_FLOAT, disp_l_float.data);

    /* Build right-camera disparity by reprojecting left disparity.
     * For each left pixel (xl, y) with disparity d, the same world point appears
     * in the right camera at xr = xl - d.  Take max disparity (nearest object wins)
     * when multiple left pixels map to the same right pixel.
     * Then fill disocclusion holes (d=0) by scanning right-to-left: the background
     * depth that fills each hole is visible just to the right of the hole. */
    cv::Mat right_disp_float = cv::Mat::zeros(proc_h, proc_w, CV_32F);
    for (int y = 0; y < proc_h; y++) {
        const float* src = disp_l_float.ptr<float>(y);
        float*       dst = right_disp_float.ptr<float>(y);
        for (int x = 0; x < proc_w; x++) {
            float d = src[x];
            if (d < 0.5f) continue;
            int xr = (int)(x - d + 0.5f);
            if (xr >= 0 && xr < proc_w && d > dst[xr])
                dst[xr] = d;
        }
    }
    for (int y = 0; y < proc_h; y++) {
        float* row = right_disp_float.ptr<float>(y);
        float fill = 0.0f;
        for (int x = proc_w - 1; x >= 0; x--) {
            if (row[x] > 0.5f) fill = row[x];
            else if (fill > 0.5f) row[x] = fill;
        }
    }
    glBindTexture(GL_TEXTURE_2D, tex_right_disp);
    glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, proc_w, proc_h,
                    GL_RED, GL_FLOAT, right_disp_float.data);

    /* Upload right rectified image (grayscale → RGBA, scaled to proc res) */
    if (!oak_frame.right_rect.empty()) {
        cv::Mat right_resized;
        cv::resize(oak_frame.right_rect, right_resized,
                   cv::Size(proc_w, proc_h), 0, 0, cv::INTER_LINEAR);
        cv::cvtColor(right_resized, m_right_rgba, cv::COLOR_GRAY2RGBA);
    } else {
        m_right_rgba.setTo(0);
    }
    glBindTexture(GL_TEXTURE_2D, tex_right_color);
    glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, proc_w, proc_h,
                    GL_RGBA, GL_UNSIGNED_BYTE, m_right_rgba.data);

    clearGPUBuffers();

    /* Intrinsics scaled to processing resolution */
    float scale_x = (float)proc_w / CAMERA_WIDTH;
    float scale_y = (float)proc_h / CAMERA_HEIGHT;
    float vfx = oak_receiver.fx * scale_x;
    float vfy = oak_receiver.fy * scale_y;
    float vcx = oak_receiver.cx * scale_x;
    float vcy = oak_receiver.cy * scale_y;

    /* Head position from iris-depth face tracker.
     * x/y are relative to calibration reference; z is absolute depth in metres.
     * Negate x: face-tracker camera faces viewer (x=right from cam = viewer's left),
     * OAK-D faces scene (x=right from cam = viewer's right). */
    float head_x = 0.0f, head_y = 0.0f, head_z = 0.0f;
    if (face_tracking_enabled && face_tracker && face_tracker->isActive()) {
        HeadPos hp = face_tracker->headPos();
        if (hp.valid) {
            head_x = -hp.x;
            head_y =  hp.y;
            head_z =  0.0f;  /* z offset from display plane deferred to Phase 6 */
        }
    }

    /* Source dispatch groups (proc_w × proc_h) — used for unproject + splat passes */
    GLuint groups_x = (proc_w + 15) / 16;
    GLuint groups_y = (proc_h + 15) / 16;
    /* Output dispatch groups (out_w × out_h) — used for JFA passes */
    GLuint out_groups_x = (out_w + 15) / 16;
    GLuint out_groups_y = (out_h + 15) / 16;

    /* ---- Pass 1: Unproject disparity → world-space texture ---- */
    glUseProgram(prog_unproject);
    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_2D, tex_left_disp);
    glUniform1i(uloc_unproj_disp,     0);
    glUniform1f(uloc_unproj_fx,       vfx);
    glUniform1f(uloc_unproj_fy,       vfy);
    glUniform1f(uloc_unproj_cx,       vcx);
    glUniform1f(uloc_unproj_cy,       vcy);
    glUniform1f(uloc_unproj_baseline, oak_receiver.baseline_m);
    glUniform2i(uloc_unproj_size,     proc_w, proc_h);
    glBindImageTexture(0, tex_worldspace, 0, GL_FALSE, 0, GL_WRITE_ONLY, GL_RGBA32F);
    glDispatchCompute(groups_x, groups_y, 1);
    glMemoryBarrier(GL_SHADER_IMAGE_ACCESS_BARRIER_BIT);

    /* ---- Pass 2: Virtual-window depth splat → SSBO ---- */
    glUseProgram(prog_vwindow_depth);
    glUniform3f (uloc_vwd_head,        head_x, head_y, head_z);
    glUniform3fv(uloc_vwd_disp_pos,    1, display_config.pos);
    glUniform3fv(uloc_vwd_disp_normal, 1, display_config.normal);
    glUniform3fv(uloc_vwd_disp_right,  1, display_config.right_v);
    glUniform3fv(uloc_vwd_disp_up,     1, display_config.up_v);
    glUniform2f (uloc_vwd_disp_size,   display_config.width_m, display_config.height_m);
    glUniform2i (uloc_vwd_src_size,    proc_w, proc_h);
    glUniform2i (uloc_vwd_out_size,    out_w, out_h);
    glBindImageTexture(0, tex_worldspace, 0, GL_FALSE, 0, GL_READ_ONLY, GL_RGBA32F);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, ssbo_depth);
    glDispatchCompute(groups_x, groups_y, 1);
    glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);

    /* ---- Pass 3: Virtual-window colour splat → tex_output ---- */
    glUseProgram(prog_vwindow_color);
    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_2D, tex_left_color);
    glUniform1i (uloc_vwc_colour,       0);
    glUniform3f (uloc_vwc_head,         head_x, head_y, head_z);
    glUniform3fv(uloc_vwc_disp_pos,     1, display_config.pos);
    glUniform3fv(uloc_vwc_disp_normal,  1, display_config.normal);
    glUniform3fv(uloc_vwc_disp_right,   1, display_config.right_v);
    glUniform3fv(uloc_vwc_disp_up,      1, display_config.up_v);
    glUniform2f (uloc_vwc_disp_size,    display_config.width_m, display_config.height_m);
    glUniform2i (uloc_vwc_src_size,     proc_w, proc_h);
    glUniform2i (uloc_vwc_out_size,     out_w, out_h);
    glBindImageTexture(0, tex_worldspace, 0, GL_FALSE, 0, GL_READ_ONLY,  GL_RGBA32F);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, ssbo_depth);
    glBindImageTexture(1, tex_output,     0, GL_FALSE, 0, GL_WRITE_ONLY, GL_RGBA8);
    glDispatchCompute(groups_x, groups_y, 1);
    glMemoryBarrier(GL_SHADER_IMAGE_ACCESS_BARRIER_BIT);

    /* ---- Pass 3b: Right-eye disocclusion fill ---- */
    glUseProgram(prog_vwindow_right);
    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_2D, tex_right_color);
    glUniform1i (uloc_vwr_right,        0);
    glActiveTexture(GL_TEXTURE1);
    glBindTexture(GL_TEXTURE_2D, tex_right_disp);
    glUniform1i (uloc_vwr_disp,         1);
    glActiveTexture(GL_TEXTURE2);
    glBindTexture(GL_TEXTURE_2D, tex_left_disp);
    glUniform1i (uloc_vwr_ldisp,        2);
    glUniform3f (uloc_vwr_head,         head_x, head_y, head_z);
    glUniform1f (uloc_vwr_fx,           vfx);
    glUniform1f (uloc_vwr_fy,           vfy);
    glUniform1f (uloc_vwr_cx,           vcx);
    glUniform1f (uloc_vwr_cy,           vcy);
    glUniform1f (uloc_vwr_baseline,     oak_receiver.baseline_m);
    glUniform3fv(uloc_vwr_disp_pos,     1, display_config.pos);
    glUniform3fv(uloc_vwr_disp_normal,  1, display_config.normal);
    glUniform3fv(uloc_vwr_disp_right,   1, display_config.right_v);
    glUniform3fv(uloc_vwr_disp_up,      1, display_config.up_v);
    glUniform2f (uloc_vwr_disp_size,    display_config.width_m, display_config.height_m);
    glUniform2i (uloc_vwr_src_size,     proc_w, proc_h);
    glUniform2i (uloc_vwr_out_size,     out_w, out_h);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, ssbo_depth);
    glBindImageTexture(1, tex_output, 0, GL_FALSE, 0, GL_WRITE_ONLY, GL_RGBA8);
    glDispatchCompute(groups_x, groups_y, 1);
    glMemoryBarrier(GL_SHADER_IMAGE_ACCESS_BARRIER_BIT | GL_SHADER_STORAGE_BARRIER_BIT);

    /* ---- Pass 4: JFA hole fill (operates on out_w × out_h) ---- */
    GLuint display_tex = tex_output;
    if (use_hole_fill) {
        /* Init seed image from depth SSBO */
        glUseProgram(prog_jfa_init);
        glUniform2i(uloc_jfai_size, out_w, out_h);
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, ssbo_depth);
        glBindImageTexture(0, tex_jfa_a, 0, GL_FALSE, 0, GL_WRITE_ONLY, GL_R32I);
        glDispatchCompute(out_groups_x, out_groups_y, 1);
        glMemoryBarrier(GL_SHADER_IMAGE_ACCESS_BARRIER_BIT);

        /* JFA passes — step halves each iteration */
        GLuint cur  = tex_jfa_a;
        GLuint next = tex_jfa_b;
        int max_dim = std::max(out_w, out_h);
        int step = 1;
        while (step < max_dim) step <<= 1;
        step >>= 1;

        while (step >= 1) {
            glUseProgram(prog_jfa);
            glUniform1i(uloc_jfa_step, step);
            glUniform2i(uloc_jfa_size, out_w, out_h);
            glBindImageTexture(0, cur,  0, GL_FALSE, 0, GL_READ_ONLY,  GL_R32I);
            glBindImageTexture(1, next, 0, GL_FALSE, 0, GL_WRITE_ONLY, GL_R32I);
            glDispatchCompute(out_groups_x, out_groups_y, 1);
            glMemoryBarrier(GL_SHADER_IMAGE_ACCESS_BARRIER_BIT);
            std::swap(cur, next);
            step >>= 1;
        }

        /* Gather: copy nearest-seed colour into tex_filled */
        glUseProgram(prog_jfa_gather);
        glUniform2i(uloc_jfag_size, out_w, out_h);
        glBindImageTexture(0, cur,        0, GL_FALSE, 0, GL_READ_ONLY,  GL_R32I);
        glBindImageTexture(1, tex_output, 0, GL_FALSE, 0, GL_READ_ONLY,  GL_RGBA8);
        glBindImageTexture(2, tex_filled, 0, GL_FALSE, 0, GL_WRITE_ONLY, GL_RGBA8);
        glDispatchCompute(out_groups_x, out_groups_y, 1);
        glMemoryBarrier(GL_SHADER_IMAGE_ACCESS_BARRIER_BIT);

        display_tex = tex_filled;
    }

    /* ---- Render output texture letterboxed in the Qt window.
     * Output is out_w × out_h (e.g. 480×1080, A-pillar strip aspect).
     * Scale the unit quad so it fills the correct fraction of the window
     * without stretching — pillarbox or letterbox as needed. ---- */
    {
        float tex_aspect = (float)out_w / (float)out_h;
        float win_aspect = (float)width() / (float)height();
        float sx, sy;
        if (tex_aspect < win_aspect) {
            /* Pillarbox: texture narrower than window — fill height, pad sides */
            sx = tex_aspect / win_aspect;
            sy = 1.0f;
        } else {
            /* Letterbox: texture wider than window — fill width, pad top/bottom */
            sx = 1.0f;
            sy = win_aspect / tex_aspect;
        }
        glViewport(0, 0, width(), height());
        glClear(GL_COLOR_BUFFER_BIT);
        glUseProgram(prog_display);
        glActiveTexture(GL_TEXTURE0);
        glBindTexture(GL_TEXTURE_2D, display_tex);
        glUniform1i(uloc_disp_texture, 0);
        glUniform2f(uloc_disp_scale, sx, sy);
        glBindVertexArray(quad_vao);
        glDrawArrays(GL_TRIANGLE_STRIP, 0, 4);
        glBindVertexArray(0);
    }

    /* ---- FPS overlay ---- */
    auto now = std::chrono::steady_clock::now();
    frame_times.push_back(now);
    while ((int)frame_times.size() > FPS_WINDOW) frame_times.pop_front();
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
             "FPS: %.1f | Hole=%s | %dx%d | Track:%s hx=%.1fcm",
             current_fps,
             use_hole_fill ? "JFA" : "OFF",
             proc_w, proc_h,
             (face_tracking_enabled && face_tracker && face_tracker->isActive()) ? "ON" : "OFF",
             head_x * 100.0f);
    painter.setPen(Qt::black); painter.drawText(12, 32, info);
    painter.setPen(Qt::white); painter.drawText(10, 30, info);
    painter.end();

    double since_print = std::chrono::duration<double>(now - last_fps_print).count();
    if (since_print >= FPS_PRINT_INTERVAL) {
        printf("FPS: %.1f  hole=%s  proc=%dx%d  track=%s  head=(%.1f,%.1f,%.1f)cm\n",
               current_fps, use_hole_fill ? "JFA" : "OFF", proc_w, proc_h,
               (face_tracking_enabled && face_tracker && face_tracker->isActive()) ? "ON" : "OFF",
               head_x * 100.0f, head_y * 100.0f, head_z * 100.0f);
        last_fps_print = now;
    }

    if (render_running)
        QTimer::singleShot(0, this, [this](){ update(); });
}

/* ========================================================================
 * main
 * ======================================================================== */

int main(int argc, char *argv[])
{
    QSurfaceFormat fmt;
    fmt.setRenderableType(QSurfaceFormat::OpenGLES);
    fmt.setVersion(3, 1);
    fmt.setSwapBehavior(QSurfaceFormat::DoubleBuffer);
    QSurfaceFormat::setDefaultFormat(fmt);

    QApplication app(argc, argv);
    SynthWindow win;
    win.resize(1920, 1080);
    win.show();
    return app.exec();
}

#include "view_synthesis.moc"
