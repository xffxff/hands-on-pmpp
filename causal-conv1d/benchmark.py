import triton
from torch import nn
import torch

from causal_conv1d import causal_conv1d_fwd


@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=['size'],  # Argument names to use as an x-axis for the plot.
        x_vals=[2**i for i in range(9,13)],  # Different possible values for `x_name`.
        x_log=True,  # x axis is logarithmic.
        line_arg='provider',  # Argument name whose value corresponds to a different line in the plot.
        line_vals=['torch', 'causal'],  # Possible values for `line_arg`.
        line_names=['Conv1d', 'CausalConv1d'],  # Label name for the lines.
        styles=[('blue', '-'), ('green', '-')],  # Line styles.
        ylabel='ms',  # Label name for the y-axis.
        plot_name='conv1d-performance',  # Name for the plot. Used also as a file name for saving the plot.
        args={},  # Values for function arguments not in `x_names` and `y_name`.
    ))
def benchmark(size, provider):
    conv_dim = 4096
    d_conv = 4
    conv_bias = False

    conv1d = nn.Conv1d(
        in_channels=conv_dim,
        out_channels=conv_dim,
        bias=conv_bias,
        kernel_size=d_conv,
        groups=conv_dim,
        padding=d_conv - 1,
    ).cuda()

    x = torch.randn(1, conv_dim, size, device='cuda')

    quantiles = [0.5, 0.2, 0.8]
    if provider == 'torch':
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: conv1d(x), quantiles=quantiles)
    if provider == 'causal':
        ms, min_ms, max_ms = triton.testing.do_bench(
            lambda: causal_conv1d_fwd(x, conv1d.weight.squeeze(1), conv1d.bias, silu_activation=False),
            quantiles=quantiles
        )
    # gbps = lambda ms: 3 * x.numel() * x.element_size() * 1e-9 / (ms * 1e-3)
    # return gbps(ms), gbps(max_ms), gbps(min_ms)
    return ms, min_ms, max_ms

benchmark.run(show_plots=False, print_data=True)