#include "disp_invert.h"
#include <cuda_runtime.h>

// Each thread processes one source pixel (x, y).
// If disparity d >= 0.5, it atomically maximises (by bit-reinterpretation)
// the destination pixel at (x - round(d), y).
// Because positive IEEE 754 floats are ordered identically to their uint32
// bit patterns, atomicMax on __float_as_uint correctly selects the largest
// disparity (nearest / foreground surface) at each destination pixel.
__global__ void invert_disparity_kernel(const float *src, int src_step_floats,
                                        unsigned int *dst, int dst_step_uints,
                                        int cols, int rows)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= cols || y >= rows) return;

    float d = src[y * src_step_floats + x];
    if (d < 0.5f) return;

    int rx = x - __float2int_rn(d);   // round to nearest integer
    if (rx < 0 || rx >= cols) return;

    atomicMax(&dst[y * dst_step_uints + rx], __float_as_uint(d));
}

void invert_disparity_cuda(const cv::cuda::GpuMat &src, cv::cuda::GpuMat &dst)
{
    dst.create(src.size(), CV_32F);
    dst.setTo(cv::Scalar(0.0f));

    const int cols = src.cols;
    const int rows = src.rows;

    const int src_step_floats = static_cast<int>(src.step  / sizeof(float));
    const int dst_step_uints  = static_cast<int>(dst.step  / sizeof(unsigned int));

    dim3 block(16, 16);
    dim3 grid((cols + 15) / 16, (rows + 15) / 16);

    invert_disparity_kernel<<<grid, block>>>(
        src.ptr<float>(),
        src_step_floats,
        reinterpret_cast<unsigned int *>(dst.data),
        dst_step_uints,
        cols, rows);
}
