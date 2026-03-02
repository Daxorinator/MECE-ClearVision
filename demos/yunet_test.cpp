/*
 * yunet_test.cpp — YuNet face detection + solvePnP head pose demo
 *
 * C++ equivalent of face_landmarker_demo.py.
 * Uses OpenCV FaceDetectorYN (YuNet) for 5-keypoint detection and solvePnP
 * for 3D head pose, then visualises the face-forward direction vector in the
 * same style as the Python MediaPipe demo.
 *
 * Usage:
 *   ./yunet_test [model.onnx] [camera_index]
 *
 * Defaults: face_detection_yunet_2021sep.onnx, camera 2.
 * On Jetson Nano: cameras 0/1 are CSI stereo cameras (need GStreamer pipeline),
 * camera 2 is the USB face webcam — use default or pass 2 explicitly.
 *
 * Controls:
 *   q — quit
 */

#include <opencv2/calib3d.hpp>
#include <opencv2/core.hpp>
#include <opencv2/dnn.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/objdetect.hpp>

#include <chrono>
#include <cmath>
#include <cstdio>
#include <string>
#include <vector>

/* ---- Canonical 3-D face model (mm), origin at nose tip ----
 *
 * Positive X = viewer's right, positive Y = up, positive Z = toward camera.
 * Z values encode physical relief of the face — eyes ~26 mm behind nose tip,
 * mouth corners ~24 mm behind.
 *
 * Point order matches YuNet keypoint output columns 4-13:
 *   right_eye, left_eye, nose_tip, right_mouth, left_mouth
 */
static const std::vector<cv::Point3f> MODEL_POINTS = {
    cv::Point3f(-43.3f,  32.7f, -26.0f),   // right eye centre
    cv::Point3f( 43.3f,  32.7f, -26.0f),   // left eye centre
    cv::Point3f(  0.0f,   0.0f,   0.0f),   // nose tip  (model origin)
    cv::Point3f(-28.9f, -28.9f, -24.1f),   // right mouth corner
    cv::Point3f( 28.9f, -28.9f, -24.1f),   // left mouth corner
};

/* Colours */
static const cv::Scalar COL_BOX    (255,   0,   0);   // blue
static const cv::Scalar COL_KPT    (  0, 255,   0);   // green
static const cv::Scalar COL_ARROW  (  0, 255,   0);   // green
static const cv::Scalar COL_NOSE   (  0,   0, 255);   // red
static const cv::Scalar COL_TEXT   (255, 255, 255);   // white
static const cv::Scalar COL_FPS    (255, 255,   0);   // yellow


/* ---- draw_pose_vector -----------------------------------------------
 * Mirrors draw_pose_vector() in face_landmarker_demo.py.
 *
 * Projects nose tip and a point 80 mm along the face-forward axis (+Z in
 * model space) using the solvePnP result, then draws an arrow between them.
 *
 * direction_cam: face normal vector in camera space (used for depth colour).
 */
static void draw_pose_vector(cv::Mat &frame,
                             const cv::Mat &rvec,
                             const cv::Mat &tvec,
                             const cv::Mat &cam_matrix,
                             const cv::Mat &dist_coeffs,
                             const cv::Vec3d &direction_cam)
{
    /* Project nose tip (origin) and point 80 mm in front of it */
    std::vector<cv::Point3f> axis_pts = {
        cv::Point3f(0.0f, 0.0f,  0.0f),   // nose tip
        cv::Point3f(0.0f, 0.0f, 80.0f),   // 80 mm forward (toward camera)
    };
    std::vector<cv::Point2f> proj;
    cv::projectPoints(axis_pts, rvec, tvec, cam_matrix, dist_coeffs, proj);

    cv::Point nose_pt(static_cast<int>(proj[0].x), static_cast<int>(proj[0].y));
    cv::Point fwd_pt (static_cast<int>(proj[1].x), static_cast<int>(proj[1].y));

    /* Arrow — nose tip → forward point */
    cv::arrowedLine(frame, nose_pt, fwd_pt, COL_ARROW, 3, cv::LINE_AA, 0, 0.3);

    /* Red dot at nose tip */
    cv::circle(frame, nose_pt, 6, COL_NOSE, -1);

    /* Depth colour ring: blue = facing camera (dz > 0), red = facing away */
    double dz = direction_cam[2];
    int b = static_cast<int>(std::max(0.0,  dz) * 255);
    int r = static_cast<int>(std::max(0.0, -dz) * 255);
    cv::circle(frame, nose_pt, 10, cv::Scalar(b, 0, r), 2);

    /* Direction text, matching Python demo format */
    char buf[64];
    std::snprintf(buf, sizeof(buf), "Dir: (%.2f, %.2f, %.2f)",
                  direction_cam[0], direction_cam[1], direction_cam[2]);
    cv::putText(frame, buf, cv::Point(10, 30),
                cv::FONT_HERSHEY_SIMPLEX, 0.6, COL_TEXT, 2);
}


int main(int argc, char **argv)
{
    const std::string model_path = (argc > 1)
        ? argv[1] : "face_detection_yunet_2021sep.onnx";
    const int cam_idx = (argc > 2) ? std::stoi(argv[2]) : 2;  // 2 = USB face webcam on Jetson

    /* ---- Open camera ---- */
    cv::VideoCapture cap(cam_idx);
    if (!cap.isOpened()) {
        std::fprintf(stderr, "Error: Cannot open camera %d\n", cam_idx);
        return 1;
    }
    cap.set(cv::CAP_PROP_FRAME_WIDTH,  640);
    cap.set(cv::CAP_PROP_FRAME_HEIGHT, 480);

    cv::Mat frame;
    cap >> frame;
    if (frame.empty()) {
        std::fprintf(stderr, "Error: Empty first frame\n");
        return 1;
    }

    const int fw = frame.cols;
    const int fh = frame.rows;
    std::printf("Camera %d opened: %dx%d\n", cam_idx, fw, fh);

    /* ---- Estimated pinhole intrinsics (same heuristic as face_tracker) ---- */
    const double focal = static_cast<double>(fw);
    cv::Mat cam_matrix = (cv::Mat_<double>(3, 3) <<
        focal, 0.0,   fw / 2.0,
        0.0,   focal, fh / 2.0,
        0.0,   0.0,   1.0);
    cv::Mat dist_coeffs = cv::Mat::zeros(4, 1, CV_64F);

    /* ---- Create YuNet detector ---- */
    cv::Ptr<cv::FaceDetectorYN> detector;
    try {
        detector = cv::FaceDetectorYN::create(
            model_path, "",
            cv::Size(fw, fh),
            0.6f,   /* score threshold */
            0.3f,   /* NMS threshold   */
            5000,   /* top-K           */
            cv::dnn::DNN_BACKEND_DEFAULT,
            cv::dnn::DNN_TARGET_CPU);
    } catch (const cv::Exception &ex) {
        std::fprintf(stderr, "YuNet init failed: %s\n", ex.what());
        return 1;
    }
    std::printf("YuNet ready — model: %s\n", model_path.c_str());
    std::printf("Press 'q' to quit.\n");

    /* ---- FPS tracking ---- */
    auto fps_start  = std::chrono::steady_clock::now();
    int  fps_count  = 0;
    double fps      = 0.0;

    while (true) {
        cap >> frame;
        if (frame.empty())
            break;

        /* Mirror effect — matches face_landmarker_demo.py cv.flip(frame, 1) */
        cv::flip(frame, frame, 1);

        /* ---- Detect ---- */
        cv::Mat faces;
        detector->detect(frame, faces);

        if (faces.rows > 0) {
            /*
             * faces row layout (CV_32F, 15 cols per face):
             *   0-3:   bbox (x, y, w, h)
             *   4-5:   right_eye  (x, y)
             *   6-7:   left_eye   (x, y)
             *   8-9:   nose_tip   (x, y)
             *   10-11: right_mouth (x, y)
             *   12-13: left_mouth  (x, y)
             *   14:    confidence
             */
            const float bx    = faces.at<float>(0,  0);
            const float by    = faces.at<float>(0,  1);
            const float bw    = faces.at<float>(0,  2);
            const float bh    = faces.at<float>(0,  3);
            const float score = faces.at<float>(0, 14);

            /* Bounding box */
            cv::rectangle(frame,
                          cv::Rect(static_cast<int>(bx), static_cast<int>(by),
                                   static_cast<int>(bw), static_cast<int>(bh)),
                          COL_BOX, 2);

            /* Keypoints */
            std::vector<cv::Point2f> image_pts;
            image_pts.reserve(5);
            for (int i = 0; i < 5; ++i) {
                float px = faces.at<float>(0, 4 + i * 2);
                float py = faces.at<float>(0, 5 + i * 2);
                image_pts.emplace_back(px, py);
                cv::circle(frame, cv::Point(static_cast<int>(px),
                                            static_cast<int>(py)),
                           4, COL_KPT, -1);
            }

            /* ---- solvePnP: 3-D model → 2-D landmarks ---- */
            cv::Mat rvec, tvec;
            if (cv::solvePnP(MODEL_POINTS, image_pts,
                             cam_matrix, dist_coeffs,
                             rvec, tvec,
                             false, cv::SOLVEPNP_EPNP)) {

                /* Face-forward direction in camera space.
                 * Model +Z = toward camera, so the face normal = R * [0, 0, 1]
                 * = third column of the rotation matrix.                        */
                cv::Mat R;
                cv::Rodrigues(rvec, R);
                cv::Vec3d dir(R.at<double>(0, 2),
                              R.at<double>(1, 2),
                              R.at<double>(2, 2));

                draw_pose_vector(frame, rvec, tvec,
                                 cam_matrix, dist_coeffs, dir);

                /* Score overlay */
                char buf[32];
                std::snprintf(buf, sizeof(buf), "Score: %.2f", score);
                cv::putText(frame, buf, cv::Point(10, 80),
                            cv::FONT_HERSHEY_SIMPLEX, 0.5,
                            cv::Scalar(200, 200, 200), 1);
            }
        }

        /* ---- FPS ---- */
        ++fps_count;
        auto now = std::chrono::steady_clock::now();
        double elapsed = std::chrono::duration<double>(now - fps_start).count();
        if (elapsed >= 1.0) {
            fps       = fps_count / elapsed;
            fps_count = 0;
            fps_start = now;
        }
        char fps_buf[32];
        std::snprintf(fps_buf, sizeof(fps_buf), "FPS: %.1f", fps);
        cv::putText(frame, fps_buf, cv::Point(10, 60),
                    cv::FONT_HERSHEY_SIMPLEX, 0.6, COL_FPS, 2);

        /* ---- Show ---- */
        cv::imshow("YuNet Head Pose", frame);
        if ((cv::waitKey(1) & 0xFF) == 'q')
            break;
    }

    cap.release();
    cv::destroyAllWindows();
    return 0;
}
