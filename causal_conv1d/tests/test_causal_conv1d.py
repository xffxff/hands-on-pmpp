import pytest
import torch
from causal_conv1d_interface import causal_conv1d_fn, causal_conv1d_fn_ref

def test_causal_conv1d_cuda():
    # Skip if CUDA is not available
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    
    # Test parameters
    batch_size = 2
    channels = 3
    seq_len = 10
    kernel_size = 4
    
    # Create input tensors on GPU
    input_tensor = torch.randn(batch_size, channels, seq_len, device='cuda', dtype=torch.float32)
    weights = torch.randn(channels, kernel_size, device='cuda', dtype=torch.float32)
    print(input_tensor)
    print(weights)
    
    # Run the convolution
    output = causal_conv1d_fn(input_tensor, weights)
    output_ref = causal_conv1d_fn_ref(input_tensor, weights)
    print(output)
    print(output_ref)