#include <cuda_runtime.h>
#include <stdio.h>

#include <torch/types.h>
#include <cuda.h>
#include <cuda_runtime.h>

#include "flash.cuh"


#define N 4         // Size of the matrix (N x N)
#define TILE_SIZE 2 // Size of the tile

// CUDA kernel for tiled matrix multiplication
__global__ void tiledMatrixMul(const float *A, const float *B, float *C, int n)
{
    // Shared memory for tiles of A and B
    __shared__ float tileA[TILE_SIZE][TILE_SIZE];
    __shared__ float tileB[TILE_SIZE][TILE_SIZE];

    // Calculate row and column index of the element
    int row = blockIdx.y * TILE_SIZE + threadIdx.y;
    int col = blockIdx.x * TILE_SIZE + threadIdx.x;

    float sum = 0.0f;

    // Loop over the tiles required to compute C(row, col)
    for (int t = 0; t < n / TILE_SIZE; ++t)
    {
        // Load tiles into shared memory
        tileA[threadIdx.y][threadIdx.x] = A[row * n + t * TILE_SIZE + threadIdx.x];
        tileB[threadIdx.y][threadIdx.x] = B[(t * TILE_SIZE + threadIdx.y) * n + col];

        __syncthreads(); // Synchronize to make sure tiles are loaded

        // Compute partial product for this tile
        for (int k = 0; k < TILE_SIZE; ++k)
        {
            sum += tileA[threadIdx.y][k] * tileB[k][threadIdx.x];
        }

        __syncthreads(); // Synchronize to make sure computation is done before loading new tiles
    }

    // Store the computed value in the output matrix
    C[row * n + col] = sum;
}

__global__ void tile_matrix_multiply(float *A, float *B, float *C, int width)
{

    __shared__ float shareA[2][2];
    __shared__ float shareB[2][2];
    int bx = blockIdx.x;
    int by = blockIdx.y;
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int row = by * 2 + ty;
    int col = bx * 2 + tx;
    printf("row: %d, col: %d, bx: %d, by: %d, tx: %d, ty: %d\n",row, col,bx,by,tx,ty);
    float temp = 0;
    for (int i = 0; i < width / 2; ++i)
    {

        shareA[ty][tx] = A[row * width + (i * 2 + tx)];
        shareB[ty][tx] = B[(i * 2 + ty) * width + col];
        __syncthreads();

        for (int k = 0; k < 2; ++k)
        {
            temp += shareA[ty][k] * shareB[k][tx];
            __syncthreads();
        }
    }
    C[row * width + col] = temp;
}

// Helper function to initialize matrices
void initializeMatrices(float *A, float *B, int n)
{
    for (int i = 0; i < n * n; ++i)
    {
        // A[i] = static_cast<float>(rand()) / RAND_MAX;
        // B[i] = static_cast<float>(rand()) / RAND_MAX;

        A[i] = i;
        B[i] = i;

        // 0,1,2,3
        // 4,5,6,7
        // 8,9,10,11
        // 12,13,14,15
    }
}

int main()
{
    int size = N * N * sizeof(float);

    // Host memory allocation
    float *h_A, *h_B, *h_C;
    h_A = (float *)malloc(size);

    h_B = (float *)malloc(size);
    h_C = (float *)malloc(size);

    initializeMatrices(h_A, h_B, N);

    // Device memory allocation
    float *d_A, *d_B, *d_C;
    cudaMalloc((void **)&d_A, size);
    cudaMalloc((void **)&d_B, size);
    cudaMalloc((void **)&d_C, size);

    // Copy matrices from host to device
    cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice);

    // Define block and grid sizes
    dim3 threadsPerBlock(TILE_SIZE, TILE_SIZE);
    dim3 blocksPerGrid(N / TILE_SIZE, N / TILE_SIZE);

    // Launch the tiled matrix multiplication kernel
    // tiledMatrixMul<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, N);
    tile_matrix_multiply<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, N);

    // Copy the result back to the host
    cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost);

    for (int i = 0; i < N; i++)
    {
        for (int j = 0; j < N; j++)
        {
            printf("%d, ", (int)h_C[i * N + j]);
        }

        printf("\n");
    }
    // Cleanup
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    free(h_A);
    free(h_B);
    free(h_C);


    return 0;
}
