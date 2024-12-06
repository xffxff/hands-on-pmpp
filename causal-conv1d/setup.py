from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='causal_conv1d_cuda',
    ext_modules=[
        CUDAExtension(
            name='causal_conv1d_cuda',
            sources=[
                'csrc/causal_conv1d.cu',
                'csrc/causal_conv1d_binding.cpp'
            ],
            extra_compile_args={
                'cxx': ['-O3'],
                'nvcc': [
                    '-O3',
                    '--use_fast_math',
                    '-gencode=arch=compute_90,code=sm_90',
                    '--ptxas-options=-v',
                    '--extended-lambda',
                    '--expt-relaxed-constexpr',
                ]
            },
            include_dirs=['csrc']
        )
    ],
    cmdclass={
        'build_ext': BuildExtension
    }
) 