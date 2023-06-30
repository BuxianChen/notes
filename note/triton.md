# 安装

这里特别说明一下离线安装

```python
git clone https://github.com/openai/triton.git
git clone https://github.com/google/googletest.git

```

做如下修改:

- unittest 中的 GIT_REPOSITORY 改为 /local/path/to/googletest
- python/setup.py 中分别下载 pybind11, conda-cuda-nvcc, llvm 相关文件, 注意conda-cuda-nvcc的标签可以在[这里](https://anaconda.org/conda-forge/cuda-nvcc/files)查看, 然后将相关的 `url` 改为: `file:///path/to/pybind11-or-llvm-or-conda-cuda-nvcc`

安装

```bash
cd triton/python
pip install cmake # build-time dependency
pip install -e .
```

但失败了, 环境是

- python 3.8
- system driver: 525.85.12 (support cuda 12.0)
- system cuda: 11.4
- conda cuda nvcc: 12.0.76


colab 上安装(似乎 `triton.language.device_print` 是新特性, 源码安装才有)

```
!mkdir third_party
%cd third_party
!git clone https://github.com/openai/triton.git
!pip install cmake
%cd triton/python
!pip install -v .
```


```
!mkdir third_party
%cd third_party
!git clone https://github.com/openai/triton.git
!wget https://github.com/pybind/pybind11/archive/refs/tags/v2.10.1.tar.gz
!wget https://github.com/ptillet/triton-llvm-releases/releases/download/llvm-17.0.0-c5dede880d17/llvm+mlir-17.0.0-x86_64-linux-gnu-ubuntu-18.04-release.tar.xz
!wget https://conda.anaconda.org/nvidia/label/cuda-12.0.0/linux-64/cuda-nvcc-12.0.76-0.tar.bz2
!git clone https://github.com/google/googletest.git
!pip install cmake
%cd triton/python
!pip install -v .
```

这个成功了, 环境是:

- python 3.10
- system driver: 525.85.12 (support cuda 12.0)
- system cuda: 11.8
- conda cuda nvcc: 12.0.76


# 语法

不带 `autotune`

```python
@triton.jit
def add_kernel(
    x_ptr,  # *Pointer* to first input vector.
    y_ptr,  # *Pointer* to second input vector.
    output_ptr,  # *Pointer* to output vector.
    n_elements,  # Size of the vector.
    # Mete
    BLOCK_SIZE: tl.constexpr,  # Number of elements each program should process.
                 # NOTE: `constexpr` so it can be used as a shape value.
):
    pass

N = 2068
x = torch.rand(N)
output = torch.empty_like(x)
assert x.is_cuda and y.is_cuda and output.is_cuda
n_elements = output.numel()

grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)
add_kernel[grid](x, y, output, n_elements, BLOCK_SIZE=1024)
print(output)
```

带 `autotune`

```python
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=5, num_warps=2),
        triton.Config({'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=5, num_warps=2),
    ],
    key=['M', 'N', 'K'],  # 这些普通参数发生变化时触发autotune
)
@triton.jit
def matmul_kernel(
    # Pointers to matrices
    a_ptr, b_ptr, c_ptr,
    # Matrix dimensions
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    # Meta-parameters
    BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
):

torch.manual_seed(0)
a = torch.randn((512, 512), device='cuda', dtype=torch.float16)
b = torch.randn((512, 512), device='cuda', dtype=torch.float16)
triton_output = matmul(a, b)
assert a.shape[1] == b.shape[0], "Incompatible dimensions"
assert a.is_contiguous(), "Matrix A must be contiguous"
assert b.is_contiguous(), "Matrix B must be contiguous"
M, K = a.shape
K, N = b.shape
# Allocates output.
c = torch.empty((M, N), device=a.device, dtype=a.dtype)
# 1D launch kernel where each block gets its own program. 注意 grid 中使用了被注解为 tl.constexpr 的参数
grid = lambda META: (triton.cdiv(M, META['BLOCK_SIZE_M']) * triton.cdiv(N, META['BLOCK_SIZE_N']),)
matmul_kernel[grid](
    a, b, c,
    M, N, K,
    a.stride(0), a.stride(1),
    b.stride(0), b.stride(1),
    c.stride(0), c.stride(1),
)
print(c)
```

# 示例: 矩阵乘

```python
# 原本的
pid_m = first_pid_m + (pid % group_size_m)
pid_n = (pid % num_pid_in_group) // group_size_m

# 修改后的
# pid_m = first_pid_m + (pid % num_pid_in_group) % group_size_m
# pid_n = (pid % num_pid_in_group) // group_size_m
```