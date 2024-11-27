#include <cub/block/block_scan.cuh>
#include <iostream>

__global__ void BlockScanExclusiveSumKernel(int *d_input, int *d_output, int num_elements) {
    __shared__ typename cub::BlockScan<int, 128>::TempStorage temp_storage;

    int thread_idx = threadIdx.x;

    int thread_data[2];
    for (int i = 0; i < 2; i++) {
        thread_data[i] = d_input[thread_idx * 2 + i];
    }

    cub::BlockScan<int, 128>(temp_storage).InclusiveSum(thread_data, thread_data);

    for (int i = 0; i < 2; i++) {
        d_output[thread_idx * 2 + i] = thread_data[i];
    }
}

int main() {
    const int num_elements = 256;
    const int block_size = 128;
    const int num_blocks = 1;

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