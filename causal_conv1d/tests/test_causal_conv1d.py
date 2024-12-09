import pytest
import torch
from causal_conv1d_interface import causal_conv1d_fn, causal_conv1d_fn_ref


@pytest.mark.parametrize("channels", [1024, 2048, 4096])
@pytest.mark.parametrize("seq_len", [2**i for i in range(0, 12)])
@pytest.mark.parametrize("kernel_size", [2, 3, 4])
def test_causal_conv1d_cuda(channels, seq_len, kernel_size):
    # Skip if CUDA is not available
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    
    batch_size = 2
    
    input_tensor = torch.randn(batch_size, channels, seq_len, device='cuda', dtype=torch.float32)
    weights = torch.randn(channels, kernel_size, device='cuda', dtype=torch.float32)
    
    output = causal_conv1d_fn(input_tensor, weights)
    output_ref = causal_conv1d_fn_ref(input_tensor, weights)

    torch.testing.assert_close(output, output_ref)