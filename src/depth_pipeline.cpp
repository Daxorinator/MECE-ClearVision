/*
 * Depth Pipeline: OAK-D Lite live disparity visualiser
 *
 * Opens the OAK-D Lite via USB, receives hardware-computed disparity
 * (subpixel 5-bit, divide by 32 for pixel units) and renders a
 * colormapped depth image in a Qt window.
 *
 * Build:  cd build && cmake .. && make -j4 depth_pipeline
 * Run:    ./depth_pipeline
 *
 * Controls:
 *   ESC   - Exit
 *   1/2/3 - Display scale: 1.0x, 0.5x, 0.25x
 */

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <chrono>
#include <deque>
#include <string>

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>

#include "oak_receiver.h"

#include <QApplication>
#include <QWidget>
#include <QLabel>
#include <QTimer>
#include <QKeyEvent>
#include <QVBoxLayout>

/* ========================================================================
 * Configuration
 * ======================================================================== */

#define TIMER_INTERVAL_MS   33
#define FPS_WINDOW          30
#define FPS_PRINT_INTERVAL  2.0

/* ========================================================================
 * Qt5 helpers
 * ======================================================================== */

static QPixmap mat_to_pixmap(const cv::Mat &bgr)
{
    cv::Mat rgb;
    cv::cvtColor(bgr, rgb, cv::COLOR_BGR2RGB);
    QImage img(rgb.data, rgb.cols, rgb.rows,
               (int)rgb.step, QImage::Format_RGB888);
    return QPixmap::fromImage(img);  // fromImage copies internally; no extra .copy() needed
}

/* ========================================================================
 * Main window
 * ======================================================================== */

class DepthWindow : public QWidget {
    Q_OBJECT

public:
    DepthWindow(QWidget *parent = nullptr);
    ~DepthWindow();

protected:
    void keyPressEvent(QKeyEvent *e) override;

private slots:
    void onTimer();

private:
    QLabel *view, *status_lbl;
    QTimer *timer;

    OAKReceiver oak;

    double disp_scale{0.5};   // display scale relative to 1920×1080

    /* FPS tracking */
    std::deque<std::chrono::steady_clock::time_point> frame_times;
    std::chrono::steady_clock::time_point last_fps_print;
    double current_fps{0.0};

    void updateStatus();
};

/* ---- constructor ---- */

DepthWindow::DepthWindow(QWidget *parent)
    : QWidget(parent)
{
    setWindowTitle("Depth Pipeline (OAK-D Lite)");

    // Disparity-only mode: no color stream, no depth alignment.
    // Disparity comes back at native 640×480 (6.5× less USB bandwidth).
    oak.want_color      = false;
    oak.want_confidence = false;

    int dw = (int)(640 * disp_scale);
    int dh = (int)(480 * disp_scale);
    resize(dw + 20, dh + 80);

    auto *vbox = new QVBoxLayout(this);

    view = new QLabel("Waiting for OAK-D Lite...");
    view->setFixedSize(dw, dh);
    view->setAlignment(Qt::AlignCenter);
    view->setStyleSheet("background: black; color: white;");
    vbox->addWidget(view);

    status_lbl = new QLabel("Initialising...");
    status_lbl->setAlignment(Qt::AlignCenter);
    QFont font = status_lbl->font();
    font.setPointSize(12);
    status_lbl->setFont(font);
    vbox->addWidget(status_lbl);

    if (!oak.start()) {
        status_lbl->setText("Failed to open OAK-D Lite — check USB connection");
        fprintf(stderr, "Failed to open OAK-D Lite\n");
        return;
    }

    printf("\n============================================================\n");
    printf("DEPTH PIPELINE (OAK-D Lite)\n");
    printf("============================================================\n");
    printf("fx=%.1f  fy=%.1f  cx=%.1f  cy=%.1f  baseline=%.1f mm\n",
           oak.fx, oak.fy, oak.cx, oak.cy, oak.baseline_m * 1000.0f);
    printf("Display scale: %.2fx\n", disp_scale);
    printf("Controls: ESC=exit  1/2/3=scale\n");
    printf("============================================================\n\n");

    last_fps_print = std::chrono::steady_clock::now();
    updateStatus();

    timer = new QTimer(this);
    connect(timer, &QTimer::timeout, this, &DepthWindow::onTimer);
    timer->start(0);  // 0ms — fires each idle event loop iteration, no artificial cap
}

DepthWindow::~DepthWindow()
{
    if (timer) timer->stop();
    oak.stop();
}

void DepthWindow::updateStatus()
{
    char buf[256];
    snprintf(buf, sizeof(buf), "FPS: %.1f  |  scale: %.0f%%", current_fps, disp_scale * 100.0);
    status_lbl->setText(buf);
}

void DepthWindow::keyPressEvent(QKeyEvent *e)
{
    if (e->key() == Qt::Key_Escape) {
        close();
        return;
    }

    bool changed = false;
    if (e->key() == Qt::Key_1) { disp_scale = 1.0; changed = true; }
    else if (e->key() == Qt::Key_2) { disp_scale = 0.5;  changed = true; }
    else if (e->key() == Qt::Key_3) { disp_scale = 0.25; changed = true; }

    if (changed) {
        int dw = (int)(1920 * disp_scale);
        int dh = (int)(1080 * disp_scale);
        view->setFixedSize(dw, dh);
        resize(dw + 20, dh + 80);
        printf("Display scale: %.2fx  (%dx%d)\n", disp_scale, dw, dh);
        updateStatus();
    }

    QWidget::keyPressEvent(e);
}

void DepthWindow::onTimer()
{
    OAKFrame f;
    if (!oak.getFrame(f))
        return;

    /* Resize to display resolution first — all subsequent ops run on smaller image */
    int dw = (int)(f.disparity.cols * disp_scale);
    int dh = (int)(f.disparity.rows * disp_scale);
    cv::Mat disp_small;
    if (disp_scale != 1.0)
        cv::resize(f.disparity, disp_small, cv::Size(dw, dh), 0, 0, cv::INTER_NEAREST);
    else
        disp_small = f.disparity;

    /* Convert subpixel CV_16U → CV_32F, auto-normalise to full colour range */
    cv::Mat disp_float;
    disp_small.convertTo(disp_float, CV_32F, 1.0f / 32.0f);

    cv::Mat disp8;
    cv::normalize(disp_float, disp8, 0, 255, cv::NORM_MINMAX, CV_8U, disp_small > 0);

    cv::Mat color;
    cv::applyColorMap(disp8, color, cv::COLORMAP_JET);
    color.setTo(cv::Scalar(0, 0, 0), (disp_small == 0));

    /* FPS */
    auto now = std::chrono::steady_clock::now();
    frame_times.push_back(now);
    while ((int)frame_times.size() > FPS_WINDOW) frame_times.pop_front();
    if (frame_times.size() >= 2) {
        double elapsed = std::chrono::duration<double>(
            frame_times.back() - frame_times.front()).count();
        current_fps = (double)(frame_times.size() - 1) / elapsed;
    }

    /* FPS overlay */
    char fps_text[64];
    snprintf(fps_text, sizeof(fps_text), "FPS: %.1f", current_fps);
    cv::putText(color, fps_text, cv::Point(10, 30),
                cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(255, 255, 255), 2);

    double since_print = std::chrono::duration<double>(now - last_fps_print).count();
    if (since_print >= FPS_PRINT_INTERVAL) {
        double dmin, dmax;
        cv::minMaxLoc(disp_float, &dmin, &dmax, nullptr, nullptr, disp_small > 0);
        printf("FPS: %.1f  disp=[%.1f..%.1f] px  display=%dx%d\n",
               current_fps, dmin, dmax, dw, dh);
        last_fps_print = now;
    }

    view->setPixmap(mat_to_pixmap(color));
    updateStatus();
}

/* ========================================================================
 * main
 * ======================================================================== */

int main(int argc, char *argv[])
{
    QApplication app(argc, argv);

    DepthWindow win;
    win.show();
    return app.exec();
}

#include "depth_pipeline.moc"
