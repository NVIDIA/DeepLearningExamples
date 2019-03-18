# Troubleshooting

Here is a compilation if common issues that you might face
while compiling / running this code:

## Compilation errors when compiling the library
If you encounter build errors like the following:
```
/usr/include/c++/6/type_traits:1558:8: note: provided for ‘template<class _From, class _To> struct std::is_convertible’
     struct is_convertible
        ^~~~~~~~~~~~~~
/usr/include/c++/6/tuple:502:1: error: body of constexpr function ‘static constexpr bool std::_TC<<anonymous>, _Elements>::_NonNestedTuple() [with _SrcTuple = std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor>&&; bool <anonymous> = true; _Elements = {at::Tensor, at::Tensor, at::Tensor, at::Tensor}]’ not a return-statement
     }
 ^
error: command '/usr/local/cuda/bin/nvcc' failed with exit status 1
```
check your CUDA version and your `gcc` version.
```
nvcc --version
gcc --version
```
If you are using CUDA 9.0 and gcc 6.4.0, then refer to https://github.com/facebookresearch/maskrcnn-benchmark/issues/25,
which has a summary of the solution. Basically, CUDA 9.0 is not compatible with gcc 6.4.0.

## ImportError: No module named maskrcnn_benchmark.config when running webcam.py

This means that `maskrcnn-benchmark` has not been properly installed.
Refer to https://github.com/facebookresearch/maskrcnn-benchmark/issues/22 for a few possible issues.
Note that we now support Python 2 as well.


## ImportError: Undefined symbol: __cudaPopCallConfiguration error when import _C

This probably means that the inconsistent version of NVCC compile and your conda CUDAToolKit package. This is firstly mentioned in https://github.com/facebookresearch/maskrcnn-benchmark/issues/45 . All you need to do is:

```
# Check the NVCC compile version(e.g.)
/usr/cuda-9.2/bin/nvcc --version
# Check the CUDAToolKit version(e.g.)
~/anaconda3/bin/conda list | grep cuda

# If you need to update your CUDAToolKit
~/anaconda3/bin/conda install -c anaconda cudatoolkit==9.2
```

Both of them should have the **same** version. For example, if NVCC==9.2 and CUDAToolKit==9.2, this will be fine while when NVCC==9.2 but CUDAToolKit==9, it fails.


## Segmentation fault (core dumped) when running the library
This probably means that you have compiled the library using GCC < 4.9, which is ABI incompatible with PyTorch.
Indeed, during installation, you probably saw a message like
```
Your compiler (g++ 4.8) may be ABI-incompatible with PyTorch!
Please use a compiler that is ABI-compatible with GCC 4.9 and above.
See https://gcc.gnu.org/onlinedocs/libstdc++/manual/abi.html.

See https://gist.github.com/goldsborough/d466f43e8ffc948ff92de7486c5216d6
for instructions on how to install GCC 4.9 or higher.
```
Follow the instructions on https://gist.github.com/goldsborough/d466f43e8ffc948ff92de7486c5216d6
to install GCC 4.9 or higher, and try recompiling `maskrcnn-benchmark` again, after cleaning the
`build` folder with
```
rm -rf build
```


