#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <thrust/device_vector.h>
#include <iostream>
#include <assert.h>
#include <stdio.h>
using namespace std;

namespace {

template <typename scalar_t>
__global__ void quant_forward_cuda_kernel(
    torch::PackedTensorAccessor<scalar_t,1,torch::RestrictPtrTraits,size_t> x,
    torch::PackedTensorAccessor<scalar_t,1,torch::RestrictPtrTraits,size_t> y,
    size_t x_size,
    size_t y_size,
    int normal_size,
    torch::PackedTensorAccessor<scalar_t,1,torch::RestrictPtrTraits,size_t> z,
    torch::PackedTensorAccessor<scalar_t,1,torch::RestrictPtrTraits,size_t> tensor_idx)
{
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= x_size) return;

    float x_v = x[idx];
    float sub_min = 1e10f;
    float z_min = 0.0f;

    // 情況 1: normal_size < 0 → 原本行為（linear scan 全部）
    if (normal_size < 0) {
        for (int i = 0; i < y_size; i++) {
            float y_v = y[i];
            float sub_v = fabsf(x_v - y_v);
            if (sub_v <= sub_min) {
                sub_min = sub_v;
                z_min = y_v;
            }
        }
    }
    // 情況 2: normal_size >= 0 → 分別處理 normal / outlier
    else {
        if (fabsf(x_v) <= 32.0f) {
            // Normal: linear scan 前 normal_size 個
            for (int i = 0; i < normal_size; i++) {
                float y_v = y[i];
                float sub_v = fabsf(x_v - y_v);
                if (sub_v <= sub_min) {
                    sub_min = sub_v;
                    z_min = y_v;
                }
            }
        } else {
            // Outlier: binary search 剩下的
            int lo = normal_size;
            int hi = y_size;
            while (lo < hi) {
                int mid = (lo + hi) / 2;
                if (y[mid] < x_v) {
                    lo = mid + 1;
                } else {
                    hi = mid;
                }
            }
            if (lo == normal_size) {
                z_min = y[normal_size];
            } else if (lo >= (int)y_size) {
                z_min = y[y_size - 1];
            } else {
                float diff_hi = fabsf(y[lo] - x_v);
                float diff_lo = fabsf(y[lo - 1] - x_v);
                z_min = (diff_hi <= diff_lo) ? y[lo] : y[lo - 1];
            }
        }
    }

    z[idx] = z_min;
}

} // namespace

std::tuple<torch::Tensor, torch::Tensor> quant_forward_cuda(
    torch::Tensor x,
    torch::Tensor y,
    int normal_size)
{
    const int threads = 1024;
    const dim3 blocks((x.size(0) + threads - 1) / threads);
    auto z   = torch::zeros_like(x);
    auto idx = torch::zeros_like(x);

    AT_DISPATCH_FLOATING_TYPES(x.type(), "quant_forward_cuda", ([&] {
        quant_forward_cuda_kernel<scalar_t><<<blocks, threads>>>(
            x.packed_accessor<scalar_t,1,torch::RestrictPtrTraits,size_t>(),
            y.packed_accessor<scalar_t,1,torch::RestrictPtrTraits,size_t>(),
            x.size(0),
            y.size(0),
            normal_size,
            z.packed_accessor<scalar_t,1,torch::RestrictPtrTraits,size_t>(),
            idx.packed_accessor<scalar_t,1,torch::RestrictPtrTraits,size_t>());
    }));

    return std::make_tuple(z, idx);
}