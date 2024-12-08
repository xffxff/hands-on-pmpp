from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='causal_conv1d',
    ext_modules=[
        CUDAExtension(
            name='causal_conv1d_cuda',
            sources=['causal_conv1d_binding.cpp', 'causal_conv1d.cu'],
            extra_compile_args={
                'cxx': [], 
                'nvcc': [
                    '-gencode', 'arch=compute_90,code=sm_90',
                ]
            }
        ),
    ],
    cmdclass={
        'build_ext': BuildExtension
    }
) 