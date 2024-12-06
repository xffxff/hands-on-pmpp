import torch
from causal_conv1d import causal_conv1d_fwd

dim = 64
seq_len = 1025

x = (
    torch.randn(1, dim, seq_len).to(torch.bfloat16).cuda()
)

weight = (
    torch.randn(dim, 4).to(torch.bfloat16).cuda()
)

output = causal_conv1d_fwd(x, weight, None, silu_activation=True)

print(output)