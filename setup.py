from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='custom_flash_attn',
    ext_modules=[
        CUDAExtension(
            name='custom_flash_attn',
            sources=['flash_attn.cu',
#                     'flash_attn_v1.cuh',
#                     'flash_attn_v2.cuh',
#                     'flash_attn_v3.cuh',
            ],
        ),
    ],
    cmdclass={'build_ext': BuildExtension},
)
