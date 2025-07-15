from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='custom_flash_attn',
    ext_modules=[
        CUDAExtension(
            name='custom_flash_attn',
            sources=['csrc/flash_attn.cu',
            ],
        ),
    ],
    cmdclass={'build_ext': BuildExtension},
)
