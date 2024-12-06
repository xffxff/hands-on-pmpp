import causal_conv1d_cuda


def causal_conv1d_fwd(x, weight, bias=None, conv_states=None, query_start_loc=None, 
                     cache_indices=None, has_initial_state=None, silu_activation=False, 
                     pad_slot_id=-1):
    """
    Causal 1D convolution forward pass.
    
    Args:
        x (Tensor): Input tensor of shape (batch, dim, seqlen) or (dim, seqlen) for variable length
        weight (Tensor): Weight tensor of shape (dim, width)
        bias (Tensor, optional): Bias tensor of shape (dim,)
        conv_states (Tensor, optional): Convolution states tensor
        query_start_loc (Tensor, optional): Query start locations for variable length sequences
        cache_indices (Tensor, optional): Cache indices tensor
        has_initial_state (Tensor, optional): Boolean tensor indicating if initial state exists
        silu_activation (bool): Whether to apply SiLU activation
        pad_slot_id (int): Padding slot ID
        
    Returns:
        Tensor: Output tensor of the same shape as input
    """
    causal_conv1d_cuda.causal_conv1d_fwd(x, weight, bias, conv_states, 
                                               query_start_loc, cache_indices,
                                               has_initial_state, silu_activation, 
                                               pad_slot_id) 
    return x