#include <iostream>
#include <cuda_runtime.h>

#define SECTION_SIZE 10

__global__ void Brent_Kung_scan_kernel(float *X, float *Y, unsigned int N) {
    __shared__ float XY[SECTION_SIZE];
    unsigned int i = 2*blockIdx.x*blockDim.x + threadIdx.x;
    if(i < N) XY[threadIdx.x] = X[i];
    if(i + blockDim.x < N) XY[threadIdx.x + blockDim.x] = X[i + blockDim.x];

    // Upsweep phase: builds binary tree from bottom to top
    // stride doubles each time: 1 -> 2 -> 4 -> 8 -> ... -> n/2
    for (int stride = 1; stride < SECTION_SIZE; stride *= 2) {
        __syncthreads();

        unsigned int index = (threadIdx.x + 1)*2*stride - 1;
        if(index < SECTION_SIZE) {
            XY[index] += XY[index - stride];
        }
    }

    // Downsweep phase: distributes values down the tree
    // starts from n//4 (not n/2) because:
    // - last element at stride n/2 already has correct value
    // - need to start from middle level of tree
    // stride halves each time: n/4 -> n/8 -> n/16 -> ... -> 1
    for (int stride = SECTION_SIZE/4; stride > 0; stride /= 2) {
        __syncthreads();

        unsigned int index = (threadIdx.x + 1)*stride*2 - 1;
        if(index + stride < SECTION_SIZE) {
            XY[index + stride] += XY[index];
        }
    }

    __syncthreads();
    if (i < N) Y[i] = XY[threadIdx.x];
    if (i + blockDim.x < N) Y[i + blockDim.x] = XY[threadIdx.x + blockDim.x];
}

int main() {
    unsigned int N = SECTION_SIZE;
    float h_X[N], h_Y[N];
    float *d_X, *d_Y;

    for (unsigned int i = 0; i < N; i++) {
        h_X[i] = static_cast<float>(i + 1);
    }

    cudaMalloc((void**)&d_X, N * sizeof(float));
    cudaMalloc((void**)&d_Y, N * sizeof(float));

    cudaMemcpy(d_X, h_X, N * sizeof(float), cudaMemcpyHostToDevice);

    dim3 blockSize(SECTION_SIZE/2);
    dim3 gridSize(1);

    std::cout << "\n=== Brent-Kung Scan Kernel ===\n";
    
    Brent_Kung_scan_kernel<<<gridSize, blockSize>>>(d_X, d_Y, N);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA Error: %s\n", cudaGetErrorString(err));
        return 1;
    }

    cudaMemcpy(h_Y, d_Y, N * sizeof(float), cudaMemcpyDeviceToHost);

    std::cout << "Input:  ";
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
