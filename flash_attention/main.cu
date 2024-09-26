#include <algorithm>
#include <cfloat>
#include <iostream>

#include <stdint.h>
#include <vector_types.h>

#include <torch/torch.h>
#include <torch/all.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>


#include "flash.cuh"

int main() {
    torch::Tensor tensor = torch::rand({2, 3});
    std::cout << tensor << std::endl;

    // Use small model params, otherwise slower than manual attention. See caveats in README.
    int batch_size = 16;
    int n_head = 12;
    int seq_len = 64;
    int head_embd = 64;
   torch::Tensor   q = torch::rand({batch_size, n_head, seq_len, head_embd});
    torch::Tensor  k = torch::rand({batch_size, n_head, seq_len, head_embd});
   torch::Tensor   v = torch::rand({batch_size, n_head, seq_len, head_embd});

    forward(q,k,v);


    if (cudaDeviceSynchronize() != cudaSuccess) {
        fprintf(stderr, "Cuda call failed\n");
    }
    return 0;
}
