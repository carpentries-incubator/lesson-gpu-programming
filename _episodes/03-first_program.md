---
title: "Your First GPU Kernel"
teaching: 0
exercises: 0
questions:
- "How can I parallelize a Python application on a GPU?"
- "How to write a GPU program?"
- "What is CUDA?"
objectives:
- "Recognize possible data parallelism in Python code"
- "Understand the structure of a CUDA program"
- "Execute a CUDA program from Python"
keypoints:
- ""
---

# Summing Two Vectors

We start by introducing a program that, given two input vectors of the same size, returns a third vector containing the sum of the corresponding elements of the two input vectors.

~~~python
def vector_add(A, B, C, size):
    for item in range(0, size):
        C[item] = A[item] + B[item]
    
    return C
~~~

One of the characteristics of this program is that each iteration of the *for* loop is independent from the other iterations.
In other words, we can not just reorder the iterations and still produce the same output, but also compute part of the iterations on one device and part of the iterations on another device, and still end up with the same result.

The CUDA-C language is a GPU programming language and API developed by NVIDIA.
It is mostly equivalent to C/C++, with some special keywords and built-in variables and functions.

We begin our introduction to CUDA by writing a small kernel, i.e. a GPU program, that computes the same function that we just described in Python.

~~~c++
extern "C"
__global__ void vector_add(const float * A, const float * B, float * C, const int size)
{
    int item = threadIdx.x;
    C[item] = A[item] + B[item];
}
~~~

Before delving deeper into the meaning of all lines of code, let us try to execute the code on a GPU.
To compile the code and manage the GPU in Python we are going to use the interface provided by CuPy.

~~~python
# size of the arrays
size = 1024

# allocating and populating the arrays
a_gpu = cupy.random.rand(size, dtype=cupy.float32)
b_gpu = cupy.random.rand(size, dtype=cupy.float32)
c_gpu = cupy.zeros(size, dtype=cupy.float32)

# CUDA vector_add
vector_add_gpu = cupy.RawKernel(r'''
extern "C"
__global__ void vector_add(const float * A, const float * B, float * C, const int size)
{
    int item = threadIdx.x;
    C[item] = A[item] + B[item];
}
''', "vector_add")

vector_add_gpu((1, 1, 1), (size, 1, 1), (a_gpu, b_gpu, c_gpu, size))
~~~

And to be sure that the CUDA code does exactly what we want, we can execute our sequential Python code and compare the results.

~~~python
a_cpu = a_gpu
b_cpu = b_gpu
c_cpu = numpy.zeros(size, dtype=numpy.float32)

vector_add(a_cpu, b_cpu, c_cpu, size)

# test
if numpy.allclose(c_cpu, c_gpu):
    print("Correct results!")
~~~

We can now move back to the CUDA code and analyze it line by line to highlight the differences between CUDA-C and normal C.

~~~c++
__global__ void vector_add(const float * A, const float * B, float * C, const int size)
~~~

This is the definition of our *vector_add* function.
The "*\_\_global\_\_*" keyword is specific to CUDA, and all the definitions of our kernels will be preceded by this keyword.
What the keyword means is that the defined function will run on the GPU, but can be called from the host, and in some cases also from the GPU itself.
Therefore the use of the word global, to specify the execution scope of the function.

Other execution space specifiers in CUDA-C are "*\_\_host\_\_*", and "*\_\_device\_\_*".
Functions annotated with the "*\_\_host\_\_*" specifier will run on the host, and be only callable from the host, while functions annotated with the "*\_\_device\_\_*" specifier will run on the GPU, but can only be called from the GPU itself.

~~~c++
int item = threadIdx.x;
C[item] = A[item] + B[item];
~~~

This is the part of the code in which we do the actual work.
As you may see, it looks similar to the innermost loop of our *vector_add* Python function, with the main difference being in how the value of the *item* variable is evaluated.

In fact, while in Python the content of *item* is the result of the *range* function, in CUDA we are reading a special variable, i.e. *threadIdx*, containing a triplet that indicates the id of a thread inside a three-dimensional CUDA block.
In this particular case we are working on a one dimensional array, and therefore only interested in the first dimension, that is stored in the *x* field of this variable.

We know enough now to pause for a moment and do a little exercise.
Assume that in our *vector_add* kernel we change the following line:

~~~c
int item = threadIdx.x;
~~~

With this other line of code:

~~~c
int item = 1;
~~~

Which of the following options is the correct answer?

1) Nothing changes
2) Only the thread with id 1 is working
3) Only C[1] is written
4) All elements of C are zero

The correct answer is number 3.


**TODO**: explain the CUDA hierarchy of grid, block, thread.

**TODO**: introduce blockIdx, gridDim, blockDim.

In the following code, fill in the blank to work with arrays that are larger than the largest CUDA block.

~~~c
extern "C"
__global__ void vector_add(const float * A, const float * B, float * C, const int size)
{
    int item = ______________;
    C[item] = A[item] + B[item];
}
~~~

The correct answer is *(blockIdx.x * blockDim.x) + threadIdx.x*.

{% include links.md %}

