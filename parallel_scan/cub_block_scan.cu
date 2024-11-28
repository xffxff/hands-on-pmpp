#include <cub/block/block_scan.cuh>
#include <cub/cub.cuh>
#include <iostream>


__global__ void BlockScanExclusiveSumKernel(int *d_input, int *d_output, int num_elements) {
    __shared__ typename cub::BlockScan<int, 128>::TempStorage temp_storage;

    using BlockLoad = cub::BlockLoad<int, 128, 2>;
    __shared__ typename BlockLoad::TempStorage temp_storage_load;

    using BlockStore = cub::BlockStore<int, 128, 2>;
    __shared__ typename BlockStore::TempStorage temp_storage_store;

    int thread_data[2];
    BlockLoad(temp_storage_load).Load(d_input, thread_data);

    cub::BlockScan<int, 128>(temp_storage).InclusiveSum(thread_data, thread_data);

    BlockStore(temp_storage_store).Store(d_output, thread_data);
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