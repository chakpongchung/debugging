//
// Created by cpchung on 9/25/24.
//


#ifndef FLASH_CUH
#define FLASH_CUH

#include <torch/types.h>
#include <cuda.h>
#include <cuda_runtime.h>
torch::Tensor forward(torch::Tensor Q, torch::Tensor K, torch::Tensor V);

__global__
void forward_kernel(const float* Q, const float* K, const float* V, const int N, const int d,
                    const int Tc, const int Tr, const int Bc, const int Br, const float softmax_scale,
                    float* l, float *m, float* O);
#endif //FLASH_CUH
