// nvcc -arch=sm_90 kogge_stone_scan.cu -o kogge_stone_scan


#include <iostream>
#include <cuda_runtime.h>

#define BLOCK_SIZE 8

__global__ void Kogge_Stone_scan_kernel(float *X, float *Y, unsigned int N) {
    __shared__ float XY[BLOCK_SIZE];
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (i < N) {
        XY[threadIdx.x] = X[i];
    } else {
        XY[threadIdx.x] = 0.0f;
    }

    for (unsigned int stride = 1; stride < blockDim.x; stride *= 2) {
        __syncthreads();
        float temp = XY[threadIdx.x];
        if (threadIdx.x >= stride) {
            temp += XY[threadIdx.x - stride];
        }
        __syncthreads();
        // This __syncthreads() call ensures that all threads have read the value of XY[threadIdx.x]
        // before any thread writes to it. This is necessary because XY[threadIdx.x] might be read
        // by other threads in the statement `temp += XY[threadIdx.x - stride]`.
        if (threadIdx.x >= stride) {
            XY[threadIdx.x] = temp;
        }
    }
    
    if (i < N) {
        Y[i] = XY[threadIdx.x];
    }
}

int main() {
    unsigned int N = 16;
    float h_X[N], h_Y[N];
    float *d_X, *d_Y;

    for (unsigned int i = 0; i < N; i++) {
        h_X[i] = static_cast<float>(i + 1);
    }

    cudaMalloc((void**)&d_X, N * sizeof(float));
    cudaMalloc((void**)&d_Y, N * sizeof(float));

    cudaMemcpy(d_X, h_X, N * sizeof(float), cudaMemcpyHostToDevice);

    dim3 blockSize(BLOCK_SIZE);
    dim3 gridSize(N / BLOCK_SIZE + 1);
    Kogge_Stone_scan_kernel<<<gridSize, blockSize>>>(d_X, d_Y, N);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA Error: %s\n", cudaGetErrorString(err));
    }

    cudaMemcpy(h_Y, d_Y, N * sizeof(float), cudaMemcpyDeviceToHost);

    std::cout << "Input: ";
    for (unsigned int i = 0; i < N; i++) {
        std::cout << h_X[i] << " ";
    }
    std::cout << "\nOutput: ";
    for (unsigned int i = 0; i < N; i++) {
        std::cout << h_Y[i] << " ";
    }
    std::cout << std::endl;

    cudaFree(d_X);
    cudaFree(d_Y);

    return 0;
}
