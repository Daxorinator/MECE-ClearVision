/*
 * Depth Pipeline: OAK-D Lite live disparity visualiser
 *
 * Opens the OAK-D Lite via USB, receives hardware-computed disparity
 * and renders a colormapped depth image at native 640×480.
 *
 * Build:  cd build && cmake .. && make -j4 depth_pipeline
 * Run:    ./depth_pipeline
 *
 * Controls:
 *   ESC   - Exit
 *   M     - Toggle hardware median 7×7 filter
 *   +/-   - Increase/decrease confidence threshold (lower = stricter)
 */

#include <cstdio>
#include <algorithm>
#include <chrono>
#include <deque>

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/calib3d.hpp>

#include "oak_receiver.h"

#include <QApplication>
#include <QWidget>
#include <QLabel>
#include <QTimer>
#include <QKeyEvent>
#include <QVBoxLayout>

#define FPS_WINDOW         30
#define FPS_PRINT_INTERVAL 2.0

static QPixmap mat_to_pixmap(const cv::Mat &bgr)
{
    cv::Mat rgb;
    cv::cvtColor(bgr, rgb, cv::COLOR_BGR2RGB);
    QImage img(rgb.data, rgb.cols, rgb.rows, (int)rgb.step, QImage::Format_RGB888);
    return QPixmap::fromImage(img);
}

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
    QTimer *timer{nullptr};
    OAKReceiver oak;
    std::deque<std::chrono::steady_clock::time_point> frame_times;
    std::chrono::steady_clock::time_point last_fps_print;
    double current_fps{0.0};

    bool median_on{true};
    int  confidence{200};
};

DepthWindow::DepthWindow(QWidget *parent) : QWidget(parent)
{
    setWindowTitle("Depth Pipeline (OAK-D Lite)");

    oak.want_color      = false;
    oak.want_confidence = false;

    auto *vbox = new QVBoxLayout(this);
    vbox->setContentsMargins(0, 0, 0, 0);
    vbox->setSpacing(0);

    view = new QLabel("Waiting for OAK-D Lite...");
    view->setFixedSize(640, 480);
    view->setAlignment(Qt::AlignCenter);
    view->setStyleSheet("background: black; color: white;");
    vbox->addWidget(view);

    status_lbl = new QLabel("Initialising...");
    status_lbl->setAlignment(Qt::AlignCenter);
    vbox->addWidget(status_lbl);

    adjustSize();

    if (!oak.start()) {
        status_lbl->setText("Failed to open OAK-D Lite — check USB connection");
        fprintf(stderr, "Failed to open OAK-D Lite\n");
        return;
    }

    printf("OAK-D Lite opened  fx=%.1f fy=%.1f cx=%.1f cy=%.1f baseline=%.1fmm\n",
           oak.fx, oak.fy, oak.cx, oak.cy, oak.baseline_m * 1000.0f);
    printf("Controls: M=median  +/-=confidence(%d)  ESC=exit\n", confidence);

    last_fps_print = std::chrono::steady_clock::now();

    timer = new QTimer(this);
    connect(timer, &QTimer::timeout, this, &DepthWindow::onTimer);
    timer->start(0);
}

DepthWindow::~DepthWindow()
{
    if (timer) timer->stop();
    oak.stop();
}

void DepthWindow::keyPressEvent(QKeyEvent *e)
{
    if (e->key() == Qt::Key_Escape) {
        close();
    } else if (e->key() == Qt::Key_M) {
        median_on = !median_on;
        oak.setStereoConfig(median_on, confidence);
        printf("Median 7x7: %s\n", median_on ? "ON" : "OFF");
    } else if (e->key() == Qt::Key_Plus || e->key() == Qt::Key_Equal) {
        confidence = std::min(confidence + 10, 255);
        oak.setStereoConfig(median_on, confidence);
        printf("Confidence threshold: %d (less strict)\n", confidence);
    } else if (e->key() == Qt::Key_Minus) {
        confidence = std::max(confidence - 10, 0);
        oak.setStereoConfig(median_on, confidence);
        printf("Confidence threshold: %d (more strict)\n", confidence);
    } else {
        QWidget::keyPressEvent(e);
    }
}

void DepthWindow::onTimer()
{
    OAKFrame f;
    if (!oak.getFrame(f)) return;

    // Speckle filter — removes isolated false-match clusters before display.
    // maxSpeckleSize=100: components < 100px are zeroed.
    // maxDiff=16: 16 raw units = 0.5px at 1/32 subpixel scale.
    cv::filterSpeckles(f.disparity, 0, 100, 16);

    cv::Mat disp_float;
    f.disparity.convertTo(disp_float, CV_32F, 1.0f / 32.0f);

    cv::Mat disp8;
    cv::normalize(disp_float, disp8, 0, 255, cv::NORM_MINMAX, CV_8U, f.disparity > 0);

    cv::Mat color;
    cv::applyColorMap(disp8, color, cv::COLORMAP_JET);
    color.setTo(cv::Scalar(0, 0, 0), (f.disparity == 0));

    auto now = std::chrono::steady_clock::now();
    frame_times.push_back(now);
    while ((int)frame_times.size() > FPS_WINDOW) frame_times.pop_front();
    if (frame_times.size() >= 2) {
        double elapsed = std::chrono::duration<double>(
            frame_times.back() - frame_times.front()).count();
        current_fps = (double)(frame_times.size() - 1) / elapsed;
    }

    char fps_text[32];
    snprintf(fps_text, sizeof(fps_text), "FPS: %.1f", current_fps);
    cv::putText(color, fps_text, cv::Point(8, 24),
                cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(255, 255, 255), 2);

    double since_print = std::chrono::duration<double>(now - last_fps_print).count();
    if (since_print >= FPS_PRINT_INTERVAL) {
        double dmin, dmax;
        cv::minMaxLoc(disp_float, &dmin, &dmax, nullptr, nullptr, f.disparity > 0);
        printf("FPS: %.1f  disp=[%.1f..%.1f] px\n", current_fps, dmin, dmax);
        last_fps_print = now;
    }

    status_lbl->setText(QString("FPS: %1  |  Median: %2  |  Confidence: %3")
        .arg(current_fps, 0, 'f', 1)
        .arg(median_on ? "7x7" : "OFF")
        .arg(confidence));
    view->setPixmap(mat_to_pixmap(color));
}

int main(int argc, char *argv[])
{
    QApplication app(argc, argv);
    DepthWindow win;
    win.show();
    return app.exec();
}

#include "depth_pipeline.moc"
