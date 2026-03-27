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
    return QPixmap::fromImage(img.copy());
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

    int dw = (int)(1920 * disp_scale);
    int dh = (int)(1080 * disp_scale);
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
    timer->start(TIMER_INTERVAL_MS);
}

DepthWindow::~DepthWindow()
{
    if (timer) timer->stop();
    oak.stop();
}

void DepthWindow::updateStatus()
{
    int dw = (int)(1920 * disp_scale);
    int dh = (int)(1080 * disp_scale);
    char buf[256];
    snprintf(buf, sizeof(buf),
             "FPS: %.1f  |  display: %dx%d (%.0f%%)",
             current_fps, dw, dh, disp_scale * 100.0);
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

    /* Convert CV_16U subpixel disparity (0–3040) → CV_32F pixel units
     * OAK-D subpixel = 5-bit, divide by 32.0 for actual pixel disparity */
    cv::Mat disp_float;
    f.disparity.convertTo(disp_float, CV_32F, 1.0f / 32.0f);

    /* Colourmap: normalise to [0, max_disp], apply JET, mask invalid pixels */
    const float max_disp = 95.0f;  // OAK-D stereo max ~95 px at 480P/75mm baseline
    cv::Mat disp8;
    disp_float.convertTo(disp8, CV_8U, 255.0f / max_disp);
    cv::Mat color;
    cv::applyColorMap(disp8, color, cv::COLORMAP_JET);
    color.setTo(cv::Scalar(0, 0, 0), (f.disparity == 0));

    /* Scale for display */
    int dw = (int)(color.cols * disp_scale);
    int dh = (int)(color.rows * disp_scale);
    if (disp_scale != 1.0)
        cv::resize(color, color, cv::Size(dw, dh), 0, 0, cv::INTER_LINEAR);

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
        cv::minMaxLoc(disp_float, &dmin, &dmax);
        printf("FPS: %.1f  disp=[%.1f..%.1f] px\n", current_fps, dmin, dmax);
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
