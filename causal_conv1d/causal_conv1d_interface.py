import torch
import causal_conv1d_cuda

def causal_conv1d_fn(input: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    """
    Applies causal 1D convolution.
    
    Args:
        input: Input tensor of shape (batch_size, channels, seq_len)
        weights: Weights tensor of shape (kernel_size,)
    
    Returns:
        Output tensor of shape (batch_size, channels, seq_len)
    """
    return causal_conv1d_cuda.causal_conv1d(input, weights) 


def causal_conv1d_fn_ref(input: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    """
    input: (batch_size, dim, seq_len)
    weights: (dim, kernel_size)
    """
    seq_len = input.size(2)
    dim, kernel_size = weights.shape
    return torch.nn.functional.conv1d(input, weights.unsqueeze(1), padding=kernel_size - 1, groups=dim)[..., :seq_len]