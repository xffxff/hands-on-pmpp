import triton
import torch

from causal_conv1d_interface import causal_conv1d_fn
from causal_conv1d_interface import causal_conv1d_fn_ref


@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=['size'],
        x_vals=[2**i for i in range(9,13)],
        x_log=True,
        line_arg='provider',
        line_vals=['torch', 'causal'],
        line_names=['Conv1d', 'CausalConv1d'],
        styles=[('blue', '-'), ('green', '-')],
        ylabel='ms',
        plot_name='conv1d-performance',
        args={},
    ))
def benchmark(size, provider):
    conv_dim = 4096
    d_conv = 4

    x = torch.randn(2, conv_dim, size, device='cuda')
    weights = torch.randn(conv_dim, d_conv, device='cuda')

    quantiles = [0.5, 0.2, 0.8]
    if provider == 'torch':
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: causal_conv1d_fn_ref(x, weights), quantiles=quantiles)
    if provider == 'causal':
        ms, min_ms, max_ms = triton.testing.do_bench(
            lambda: causal_conv1d_fn(x, weights),
            quantiles=quantiles
        )
    return ms, min_ms, max_ms

benchmark.run(show_plots=False, print_data=True)