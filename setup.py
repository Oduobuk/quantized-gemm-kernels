import os
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CppExtension

extra_compile_args = ["-O3"]
if os.name != "nt":
    extra_compile_args += ["-std=c++17", "-Wall"]

setup(
    name="quantized_gemm_cpu_ext",
    packages=["quantized_gemm"],
    ext_modules=[
        CppExtension(
            name="quantized_gemm_cpu_ext",
            sources=["csrc/quantized_gemm_cpu.cpp"],
            extra_compile_args=extra_compile_args,
        )
    ],
    cmdclass={"build_ext": BuildExtension},
)
