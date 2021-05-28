---
title: "Constant Memory"
teaching: 0
exercises: 0
questions:
- "Question"
objectives:
- "Understanding when to use constant memory"
keypoints:
- ""
---

# Constant Memory

Constant memory is a read only cache which content can be broadcasted to multiple threads in a block.
It is allocated by the host using the `__constant__` identifier, and it must be a global variable, i.e. it must be declared in a scope that contains the kernel, not inside the kernel.
Although constant memory is declared on the host, it is not accessible by the host itself.

~~~
extern "C" {
#define BLOCKS 2

__constant__ factors[BLOCKS];

__global__ void sum_and_multiply(const float * A, const float * B, float * C, const int size)
{
    int item = (blockIdx.x * blockDim.x) + threadIdx.x;
    C[item] = (A[item] + B[item]) * factors[blockIdx.x];
}
}
~~~
{: .language-c}

And now the Python code to execute it.

~~~
# size of the vectors
size = 2048

# allocating and populating the vectors
a_gpu = cupy.random.rand(size, dtype=cupy.float32)
b_gpu = cupy.random.rand(size, dtype=cupy.float32)
c_gpu = cupy.zeros(size, dtype=cupy.float32)

# CUDA code
cuda_code = r'''
extern "C"
__constant__ float factors[2];

extern "C"
__global__ void sum_and_multiply(const float * A, const float * B, float * C, const int size)
{
    int item = (blockIdx.x * blockDim.x) + threadIdx.x;
    C[item] = (A[item] + B[item]) * factors[blockIdx.x];
}
'''

# Compile and access the code
module = cupy.RawModule(code=cuda_code)
sum_and_multiply = module.get_function("sum_and_multiply")
# Allocate and copy constant memory
factors_ptr = module.get_global("factors")
factors_gpu = cupy.ndarray(2, cupy.float32, factors_ptr)
factors_gpu[...] = cupy.random.random(2, dtype=cupy.float32)

sum_and_multiply((2, 1, 1), (size // 2, 1, 1), (a_gpu, b_gpu, c_gpu, size))
~~~
{: .language-python}

As you can see the code is not very general, it still uses constants, but it is a working example of how to use constant memory.

{% include links.md %}