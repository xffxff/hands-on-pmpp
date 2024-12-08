#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

__global__ void causal_conv1d_kernel(
                                    float* input, 
                                    float* output, 
                                    float* weights, 
                                    int kernel_size, 
                                    int seq_len, 
                                    int input_batch_stride,
                                    int input_channel_stride,
                                    int output_batch_stride,
                                    int output_channel_stride,
                                    int weights_channel_stride
                                    ) {
    int batch_id = blockIdx.x;
    int channel_id = blockIdx.y;
    int tid = threadIdx.x;
    
    int elements_per_thread = (seq_len + blockDim.x - 1) / blockDim.x;
    int start = tid * elements_per_thread;
    int end = min(start + elements_per_thread, seq_len);
    
    for(int i = start; i < end; i++) {
        float sum = 0.0f;
        for(int k = 0; k < kernel_size; k++) {
            // kernel_size = 3
            // i = 0, k = 0, input[-2] * w[0]
            // i = 0, k = 1, input[-1] * w[1]
            // i = 0, k = 2, input[0] * w[2]
            int offset =  i + k - kernel_size + 1;
            float input_val = offset >= 0 ? input[batch_id * input_batch_stride + channel_id * input_channel_stride + offset] : 0.0f;
            sum += input_val * weights[channel_id * weights_channel_stride + k];
        }
        output[batch_id * output_batch_stride + channel_id * output_channel_stride + i] = sum;
    }
}

void causal_conv1d_cuda(
    torch::Tensor input,
    torch::Tensor output,
    torch::Tensor weights) {
    
    const int batch_size = input.size(0);
    const int channels = input.size(1);
    const int seq_len = input.size(2);
    const int kernel_size = weights.size(1);
    
    dim3 blocks(batch_size, channels);
    int threads = 128;  
    
    causal_conv1d_kernel<<<blocks, threads>>>(
        input.data_ptr<float>(),
        output.data_ptr<float>(),
        weights.data_ptr<float>(),
        kernel_size,
        seq_len,
        input.stride(0),
        input.stride(1),
        output.stride(0),
        output.stride(1),
        weights.stride(0)
    );
}
