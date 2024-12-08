#include <torch/extension.h>
#include <vector>

// CUDA forward declarations
void causal_conv1d_cuda(
    torch::Tensor input,
    torch::Tensor output,
    torch::Tensor weights);

// Python-visible function
torch::Tensor causal_conv1d(
    torch::Tensor input,
    torch::Tensor weights) {
    
    TORCH_CHECK(input.dim() == 3, "Input must be 3D tensor (batch, channels, seq_len)");
    TORCH_CHECK(weights.dim() == 2, "Weights must be 2D tensor (channels, kernel_size)");
    TORCH_CHECK(weights.size(0) == input.size(1), "Weights channels must match input channels");
    TORCH_CHECK(input.is_cuda(), "Input must be a CUDA tensor");
    TORCH_CHECK(weights.is_cuda(), "Weights must be a CUDA tensor");
    TORCH_CHECK(input.scalar_type() == torch::kFloat32, "Input must be float32");
    TORCH_CHECK(weights.scalar_type() == torch::kFloat32, "Weights must be float32");
    
    auto output = torch::zeros_like(input);
    
    causal_conv1d_cuda(input, output, weights);
    
    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("causal_conv1d", &causal_conv1d, "Causal 1D convolution");
} 