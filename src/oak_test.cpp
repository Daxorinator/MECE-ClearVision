//
// oak_test.cpp — minimal OAK-D Lite connectivity test
//
// Opens the device, configures disparity + color output, grabs one frame of
// each, prints dimensions and camera intrinsics from EEPROM, then exits.
//
// Build:  cd build && cmake .. && make -j4 oak_test
// Run:    ./oak_test
//

#include <cstdio>
#include <depthai/depthai.hpp>

int main()
{
    dai::Pipeline pipeline;

    // --- Mono cameras ---
    auto camLeft  = pipeline.create<dai::node::MonoCamera>();
    auto camRight = pipeline.create<dai::node::MonoCamera>();
    camLeft->setBoardSocket(dai::CameraBoardSocket::LEFT);
    camRight->setBoardSocket(dai::CameraBoardSocket::RIGHT);
    camLeft->setResolution(dai::MonoCameraProperties::SensorResolution::THE_480_P);
    camRight->setResolution(dai::MonoCameraProperties::SensorResolution::THE_480_P);

    // --- Color camera ---
    auto camRGB = pipeline.create<dai::node::ColorCamera>();
    camRGB->setResolution(dai::ColorCameraProperties::SensorResolution::THE_1080_P);
    camRGB->setInterleaved(false);
    camRGB->setColorOrder(dai::ColorCameraProperties::ColorOrder::BGR);

    // --- Stereo depth ---
    auto stereo = pipeline.create<dai::node::StereoDepth>();
    stereo->setDefaultProfilePreset(dai::node::StereoDepth::PresetType::HIGH_DENSITY);
    stereo->setLeftRightCheck(true);
    stereo->setSubpixel(true);   // 5-bit subpixel (RAW16, range 0-3040)
    stereo->setDepthAlign(dai::CameraBoardSocket::RGB);
    camLeft->out.link(stereo->left);
    camRight->out.link(stereo->right);

    // --- XLink outputs ---
    auto xoutDisp  = pipeline.create<dai::node::XLinkOut>();
    auto xoutColor = pipeline.create<dai::node::XLinkOut>();
    xoutDisp->setStreamName("disparity");
    xoutColor->setStreamName("color");
    stereo->disparity.link(xoutDisp->input);
    camRGB->isp.link(xoutColor->input);

    // --- Open device ---
    printf("Opening OAK-D Lite...\n");
    dai::Device device(pipeline);
    printf("Device opened: %s\n", device.getMxId().c_str());

    // --- Read intrinsics from EEPROM ---
    auto calib = device.readCalibration();
    auto intrinsics = calib.getCameraIntrinsics(dai::CameraBoardSocket::RGB, 1920, 1080);
    printf("RGB intrinsics (1920x1080):\n");
    printf("  fx=%.2f  fy=%.2f  cx=%.2f  cy=%.2f\n",
           intrinsics[0][0], intrinsics[1][1],
           intrinsics[0][2], intrinsics[1][2]);

    float baseline_cm = calib.getBaselineDistance(
        dai::CameraBoardSocket::LEFT, dai::CameraBoardSocket::RIGHT);
    printf("Stereo baseline: %.1f mm\n", baseline_cm * 10.0f);

    // --- Grab one disparity frame ---
    auto dispQueue  = device.getOutputQueue("disparity",  1, false);
    auto colorQueue = device.getOutputQueue("color",      1, false);

    printf("Waiting for disparity frame...\n");
    auto dispFrame = dispQueue->get<dai::ImgFrame>();
    printf("Disparity frame: %ux%u  type=%d\n",
           dispFrame->getWidth(), dispFrame->getHeight(),
           (int)dispFrame->getType());

    printf("Waiting for color frame...\n");
    auto colorFrame = colorQueue->get<dai::ImgFrame>();
    printf("Color frame:     %ux%u  type=%d\n",
           colorFrame->getWidth(), colorFrame->getHeight(),
           (int)colorFrame->getType());

    printf("OK — OAK-D Lite is working.\n");
    return 0;
}
