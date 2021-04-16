---
title: "Your First GPU Kernel"
teaching: 45
exercises: 25
questions:
- "How can I parallelize a Python application on a GPU?"
- "How to write a GPU program?"
- "What is CUDA?"
objectives:
- "Recognize possible data parallelism in Python code"
- "Understand the structure of a CUDA program"
- "Execute a CUDA program in Python using CuPy"
keypoints:
- ""
---

# Summing Two Vectors in Python

We start by introducing a program that, given two input vectors of the same size, returns a third vector containing the sum of the corresponding elements of the two input vectors.

~~~
def vector_add(A, B, C, size):
    for item in range(0, size):
        C[item] = A[item] + B[item]
    
    return C
~~~
{: .language-python}

One of the characteristics of this program is that each iteration of the `for` loop is independent from the other iterations.
In other words, we could reorder the iterations and still produce the same output, or even compute each iteration in parallel or on a different device, and still come up with the same output.
These are the kind of programs that we would call *naturally parallel*, and they are perfect candidates for being executed on a GPU.

# Summing Two Vectors in CUDA

The CUDA-C language is a GPU programming language and API developed by NVIDIA.
It is mostly equivalent to C/C++, with some special keywords, built-in variables, and functions.

We begin our introduction to CUDA by writing a small kernel, i.e. a GPU program, that computes the same function that we just described in Python.

~~~
extern "C"
__global__ void vector_add(const float * A, const float * B, float * C, const int size)
{
    int item = threadIdx.x;
    C[item] = A[item] + B[item];
}
~~~
{: .language-c}

# Running Code on the GPU with CuPy

Before delving deeper into the meaning of all lines of code, let us try to execute the code on a GPU.
To compile the code and manage the GPU in Python we are going to use the interface provided by CuPy.

~~~
# size of the vectors
size = 1024

# allocating and populating the vectors
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
{: .language-python}

And to be sure that the CUDA code does exactly what we want, we can execute our sequential Python code and compare the results.

~~~
a_cpu = a_gpu
b_cpu = b_gpu
c_cpu = numpy.zeros(size, dtype=numpy.float32)

vector_add(a_cpu, b_cpu, c_cpu, size)

# test
if numpy.allclose(c_cpu, c_gpu):
    print("Correct results!")
~~~
{: .language-python}

# Understanding the CUDA Code

We can now move back to the CUDA code and analyze it line by line to highlight the differences between CUDA-C and normal C.

~~~
__global__ void vector_add(const float * A, const float * B, float * C, const int size)
~~~
{: .language-c}

This is the definition of our `vector_add` function.
The `__global__` keyword is specific to CUDA, and all the definitions of our kernels will be preceded by this keyword.
What the keyword means is that the defined function will run on the GPU, but can be called from the host, and in some cases also from the GPU itself.
Therefore the use of the word global, to specify the execution scope of the function.

Other execution space specifiers in CUDA-C are `__host__`, and `__device__`.
Functions annotated with the `__host__` specifier will run on the host, and be only callable from the host, while functions annotated with the `__device__` specifier will run on the GPU, but can only be called from the GPU itself.

~~~
int item = threadIdx.x;
C[item] = A[item] + B[item];
~~~
{: .language-c}

This is the part of the code in which we do the actual work.
As you may see, it looks similar to the innermost loop of our `vector_add` Python function, with the main difference being in how the value of the `item` variable is evaluated.

In fact, while in Python the content of `item` is the result of the `range` function, in CUDA we are reading a special variable, i.e. `threadIdx`, containing a triplet that indicates the id of a thread inside a three-dimensional CUDA block.
In this particular case we are working on a one dimensional vector, and therefore only interested in the first dimension, that is stored in the `x` field of this variable.

> ## Challenge
>
> We know enough now to pause for a moment and do a little exercise.
> Assume that in our `vector_add` kernel we change the following line:
>
> ~~~
> int item = threadIdx.x;
> ~~~
> {: .language-c}
> 
> With this other line of code:
>
> ~~~
> int item = 1;
> ~~~
> {: .language-c}
>
> Which of the following options is the correct answer?
>
> 1) Nothing changes
>
> 2) Only the first thread is working
>
> 3) Only `C[1]` is written
>
> 4) All elements of `C` are zero
>
> > ## Solution
> > The correct answer is number 3.
> {: .solution}
{: .challenge}

# Computing Hierarchy in CUDA

In the previous example we had a small vector of size 1024, and each of the 1024 threads we generated was working on one of the element.

What would happen if we changed the size of the vector to a larger number, such as 2048?
We modify the value of the variable size and try again.

~~~
# size of the vectors
size = 2048

# allocating and populating the vectors
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
{: .language-python}

This is how the output should look like when running the code in a Jupyter Notebook:

~~~
---------------------------------------------------------------------------

CUDADriverError                           Traceback (most recent call last)

<ipython-input-4-a26bc8acad2f> in <module>()
     19 ''', "vector_add")
     20 
---> 21 vector_add_gpu((1, 1, 1), (size, 1, 1), (a_gpu, b_gpu, c_gpu, size))
     22 
     23 print(c_gpu)

cupy/core/raw.pyx in cupy.core.raw.RawKernel.__call__()

cupy/cuda/function.pyx in cupy.cuda.function.Function.__call__()

cupy/cuda/function.pyx in cupy.cuda.function._launch()

cupy_backends/cuda/api/driver.pyx in cupy_backends.cuda.api.driver.launchKernel()

cupy_backends/cuda/api/driver.pyx in cupy_backends.cuda.api.driver.check_status()

CUDADriverError: CUDA_ERROR_INVALID_VALUE: invalid argument
~~~
{: .output}

The reason for this error is that most GPUs will not allow us to execute a block composed of more than 1024 threads.
If we look at the parameters of our functions we see that the first two parameters are two triplets.

~~~
vector_add_gpu((1, 1, 1), (size, 1, 1), (a_gpu, b_gpu, c_gpu, size))
~~~
{: .language-python}

The first triplet specifies the size of the CUDA **grid**, while the second triplet specifies the size of the CUDA **block**.
The grid is a three-dimensional structure in the CUDA programming model and it represent the organization of a whole kernel execution.
A grid is made of one or more independent blocks, and in the case of previous code we have a grid composed by a single block `(1, 1, 1)`.
The size of this block is specified by the second triplet, in our case `(size, 1, 1)`.
While blocks are independent of each other, the thread composing a block are not completely independent, they share resources and can also communicate with each other.

To go back to our example, we can modify che grid specification from `(1, 1, 1)` to `(2, 1, 1)`, and the block specification from `(size, 1, 1)` to `(size // 2, 1, 1)`.
If we run the code again, we should again get the correct output.

We already introduced the special variable `threadIdx` when introducing the CUDA code, and we said it contains a triplet specifying the coordinates of a thread in a thread block.
CUDA has other variables that are important to understand the coordinates of each thread and block in the overall structure of the computation.

These special variables are `blockDim`, `blockIdx`, and `gridDim`, and they are all triplets.
The triplet contained in `blockDim` represents the size of the calling thread's block in three dimensions.
While the content of `threadIdx` is different for each thread in the same block, the content of `blockDim` is the same because the size of the block is the same for all threads.
The coordinates of a block in the computational grid are contained in `blockIdx`, therefore the content of this variable will be the same for all threads in the same block, but different for threads in different blocks.
Finally, `gridDim` contains the size of the grid in three dimensions, and it is again the same for all threads.

> ## Challenge
>
> Given the following snippet of code:
>
> ~~~
> size = 512
> vector_add_gpu((4, 1, 1), (size, 1, 1), (a_gpu, b_gpu, c_gpu, size))
> ~~~
> {: .language-python}
> 
> What is the content of the `blockDim` and `gridDim` variables?
>
> > ## Solution
> > The content of `blockDim` is `(512, 1, 1)` and the content of `gridDim` is `(4, 1, 1)` for all threads.
> {: .solution}
{: .challenge}

What happens if we run the code that we just fixed to work on an vector of 2048 elements, and compare the results with our CPU version?

~~~
# size of the vectors
size = 2048

# allocating and populating the vectors
a_gpu = cupy.random.rand(size, dtype=cupy.float32)
b_gpu = cupy.random.rand(size, dtype=cupy.float32)
c_gpu = cupy.zeros(size, dtype=cupy.float32)
a_cpu = a_gpu
b_cpu = b_gpu
c_cpu = numpy.zeros(size, dtype=numpy.float32)

# CPU code
def vector_add(A, B, C, size):
    for item in range(0, size):
        C[item] = A[item] + B[item]
    
    return C

# CUDA vector_add
vector_add_gpu = cupy.RawKernel(r'''
extern "C"
__global__ void vector_add(const float * A, const float * B, float * C, const int size)
{
    int item = threadIdx.x;
    C[item] = A[item] + B[item];
}
''', "vector_add")

# execute the code
vector_add_gpu((2, 1, 1), (size // 2, 1, 1), (a_gpu, b_gpu, c_gpu, size))
vector_add(a_cpu, b_cpu, c_cpu, size)

# test
if numpy.allclose(c_cpu, c_gpu):
    print("Correct results!")
else:
    print("Wrong results!")
~~~
{: .language-python}

~~~
Wrong results!
~~~
{: .output}

The results are wrong!
In fact, while we increased the number of threads we launch, we did not modify the kernel code to compute the correct results using the new builtin variables we just introduced.

> ## Challenge
>
> In the following code, fill in the blank to work with vectors that are larger than the largest CUDA block (i.e. 1024).
>
> ~~~
> extern "C"
> __global__ void vector_add(const float * A, const float * B, float * C, const int size)
> {
>    int item = ______________;
>    C[item] = A[item] + B[item];
> }
> ~~~
> {: .language-c}
>
> > ## Solution
> > The correct answer is `(blockIdx.x * blockDim.x) + threadIdx.x`.
> >
> > ~~~
> > extern "C"
> > __global__ void vector_add(const float * A, const float * B, float * C, const int size)
> > {
> >    int item = (blockIdx.x * blockDim.x) + threadIdx.x;
> >    C[item] = A[item] + B[item];
> > }
> > ~~~
> > {: .language-c}
> {: .solution}
{: .challenge}

# Vectors of Arbitrary Size

So far we have worked with a number of threads that is the same as the elements in the vector.
However, in a real world scenario we may have to process vectors of arbitrary size, and to do this we need to modify both the kernel and the way it is launched.

> ## Challenge
>
> We modified the `vector_add` kernel to include a check for the size of the vector, so that we only compute elements that are inside the vector boundaries.
> However the code is not correct as it is written now.
> Can you reorder the lines of the source code to make it work?
>
> ~~~
> extern "C"
> __global__ void vector_add(const float * A, const float * B, float * C, const int size)
> {
>    if ( item < size )
>    {
>       int item = (blockIdx.x * blockDim.x) + threadIdx.x;
>    }
>    C[item] = A[item] + B[item];
> }
> ~~~
> {: .language-c}
>
> > ## Solution
> > The correct way to modify the `vector_add` to work on vectors of arbitrary size is to first compute the coordinates of each thread, and then perform the sum only on elements that are within the vector boundaries.
> >
> > ~~~
> > extern "C"
> > __global__ void vector_add(const float * A, const float * B, float * C, const int size)
> > {
> >    int item = (blockIdx.x * blockDim.x) + threadIdx.x;
> >    if ( item < size )
> >    {
> >       C[item] = A[item] + B[item];
> >    }
> > }
> > ~~~
> > {: .language-c}
> {: .solution}
{: .challenge}

To test our changes we can modify the `size` of the vectors from 2048 to 10000, and execute the code again.

~~~
---------------------------------------------------------------------------

CUDADriverError                           Traceback (most recent call last)

<ipython-input-20-00d938215d28> in <module>()
     31 
     32 # Execute the code
---> 33 vector_add_gpu((2, 1, 1), (size // 2, 1, 1), (a_gpu, b_gpu, c_gpu, size))
     34 vector_add(a_cpu, b_cpu, c_cpu, size)
     35 

cupy/core/raw.pyx in cupy.core.raw.RawKernel.__call__()

cupy/cuda/function.pyx in cupy.cuda.function.Function.__call__()

cupy/cuda/function.pyx in cupy.cuda.function._launch()

cupy/cuda/driver.pyx in cupy.cuda.driver.launchKernel()

cupy/cuda/driver.pyx in cupy.cuda.driver.check_status()

CUDADriverError: CUDA_ERROR_INVALID_VALUE: invalid argument
~~~
{: .output}

This error is telling us that CUDA cannot launch a block with `size // 2` threads, because the maximum amount of threads in a kernel is 1024 and we are requesting 5000 threads.

What we need to do is to make grid and block more flexible, so that they can adapt to vectors of arbitrary size.
To do that, we can replace the line in which we call `vector_add_gpu` with the following code.

~~~
grid_size = (int(math.ceil(size / 1024)), 1, 1)
block_size = (1024, 1, 1)
vector_add_gpu(grid_size, block_size, (a_gpu, b_gpu, c_gpu, size))
~~~
{: .language-python}

With these changes we always have blocks of 1024 threads, but we adapt the number of blocks so that we always have enough to threads to compute all elements in the vector.
We also need to add an `import math` in our Python code to be able to call the `math.ceil()` function.

We can now execute the code again.

~~~
Correct results!
~~~
{: .output}

{% include links.md %}
