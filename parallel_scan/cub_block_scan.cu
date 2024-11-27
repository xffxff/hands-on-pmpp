#include <cub/block/block_scan.cuh>
#include <iostream>

__global__ void BlockScanExclusiveSumKernel(int *d_input, int *d_output, int num_elements) {
    __shared__ typename cub::BlockScan<int, 128>::TempStorage temp_storage;

    int thread_idx = threadIdx.x;
    int block_offset = blockIdx.x * blockDim.x;

    int input = 0;
    if (block_offset + thread_idx < num_elements) {
        input = d_input[block_offset + thread_idx];
    }

    int output = 0;
    cub::BlockScan<int, 128>(temp_storage).InclusiveSum(input, output);

    if (block_offset + thread_idx < num_elements) {
        d_output[block_offset + thread_idx] = output;
    }
}

int main() {
    const int num_elements = 256;
    const int block_size = 128;
    const int num_blocks = (num_elements + block_size - 1) / block_size;

    int h_input[num_elements];
    int h_output[num_elements];

    for (int i = 0; i < num_elements; ++i) {
        h_input[i] = 1;
    }

    int *d_input, *d_output;
    cudaMalloc(&d_input, num_elements * sizeof(int));
    cudaMalloc(&d_output, num_elements * sizeof(int));

    cudaMemcpy(d_input, h_input, num_elements * sizeof(int), cudaMemcpyHostToDevice);

    BlockScanExclusiveSumKernel<<<num_blocks, block_size>>>(d_input, d_output, num_elements);
    cudaDeviceSynchronize();

    cudaMemcpy(h_output, d_output, num_elements * sizeof(int), cudaMemcpyDeviceToHost);

    std::cout << "Input: ";
    for (int i = 0; i < num_elements; ++i) {
        std::cout << h_input[i] << " ";
    }
    std::cout << "\nOutput: ";
    for (int i = 0; i < num_elements; ++i) {
        std::cout << h_output[i] << " ";
    }
    std::cout << std::endl;

    cudaFree(d_input);
    cudaFree(d_output);

    return 0;
}