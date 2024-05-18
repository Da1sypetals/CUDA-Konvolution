import os
os.environ['CUDA_HOME'] = '/mnt/public/lib/cuda/cuda-12.1'


from setuptools import find_packages, setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension



setup(
    name='cuLegKan',
    packages=find_packages(),
    version='0.0.0',
    author='Yuxue Yang',
    ext_modules=[
        CUDAExtension(
            'legendre_ops', # operator name
            ['./cpp/legendre.cpp',
             './cpp/legendre_cuda.cu',]
        ),
        CUDAExtension(
            'legendre_2d_ops', # operator name
            ['./legendre_2d_cpp/legendre_2d.cpp',
             './legendre_2d_cpp/legendre_2d_cuda.cu',]
        ),
        CUDAExtension(
            'rightway_legendre_ops', # operator name
            ['./right_way_cpp/legendre.cpp',
             './right_way_cpp/legendre_cuda.cu',]
        ),
    ],
    cmdclass={
        'build_ext': BuildExtension
    }
)