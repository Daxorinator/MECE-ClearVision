#pragma once
#include <opencv2/core/cuda.hpp>

// Invert a left-camera disparity map into an approximate right-camera disparity map.
// For each source pixel (x, y) with disparity d, writes d to destination (x - round(d), y)
// using atomicMax on float bit patterns so the foreground (highest disparity) wins.
// dst must be CV_32F and the same size as src; it is zeroed before the kernel runs.
void invert_disparity_cuda(const cv::cuda::GpuMat &src, cv::cuda::GpuMat &dst,
                           cudaStream_t stream = 0);
