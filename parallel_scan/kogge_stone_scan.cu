// nvcc -arch=sm_90 kogge_stone_scan.cu -o kogge_stone_scan


#include <iostream>
#include <cuda_runtime.h>

#define SECTION_SIZE 16

__global__ void Kogge_Stone_scan_kernel(float *X, float *Y, unsigned int N) {
    __shared__ float XY[SECTION_SIZE];
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

__global__ void Kogge_Stone_scan_kernel_double_buffer(float *X, float *Y, unsigned int N) {
    // Use double buffer to remove the second __syncthreads() call in the original Kogge-Stone scan kernel
    __shared__ float XY[SECTION_SIZE];
    __shared__ float XY_next[SECTION_SIZE];
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (i < N) {
        XY[threadIdx.x] = X[i];
        XY_next[threadIdx.x] = X[i];
    } else {
        XY[threadIdx.x] = 0.0f;
        XY_next[threadIdx.x] = 0.0f;
    }

    float *input_buffer = XY;
    float *output_buffer = XY_next;

    for (unsigned int stride = 1; stride < blockDim.x; stride *= 2) {
        __syncthreads();

        float *temp = input_buffer;
        input_buffer = output_buffer;
        output_buffer = temp;

        if (threadIdx.x >= stride) {
            output_buffer[threadIdx.x] = input_buffer[threadIdx.x - stride] + input_buffer[threadIdx.x];
        } else {
            output_buffer[threadIdx.x] = input_buffer[threadIdx.x];
        }
    }
    
    if (i < N) {
        Y[i] = output_buffer[threadIdx.x];
    }
}

int main() {
    unsigned int N = 16;
    // Note: N must be smaller than SECTION_SIZE

    float h_X[N], h_Y[N];
    float *d_X, *d_Y;

    // Initialize input array
    for (unsigned int i = 0; i < N; i++) {
        h_X[i] = static_cast<float>(i + 1);
    }

    // Allocate device memory
    cudaMalloc((void**)&d_X, N * sizeof(float));
    cudaMalloc((void**)&d_Y, N * sizeof(float));

    // Copy input data to device
    cudaMemcpy(d_X, h_X, N * sizeof(float), cudaMemcpyHostToDevice);

    dim3 blockSize(SECTION_SIZE);
    dim3 gridSize(N / SECTION_SIZE + 1);

    void (*kernels[])(float*, float*, unsigned int) = {
        Kogge_Stone_scan_kernel,
        Kogge_Stone_scan_kernel_double_buffer
    };
    const char* kernel_names[] = {
        "Original Kogge-Stone",
        "Double Buffer Kogge-Stone"
    };

    // Run both kernels
    for (int k = 0; k < 2; k++) {
        std::cout << "\n=== " << kernel_names[k] << " Kernel ===\n";
        
        kernels[k]<<<gridSize, blockSize>>>(d_X, d_Y, N);
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
    }

    // Free device memory
    cudaFree(d_X);
    cudaFree(d_Y);

    return 0;
}
