---
title: "Shared Memory and Synchronization"
teaching: 30
exercises: 25
questions:
- "Question"
objectives:
- "Learn how to share data between threads"
- "Learn how to synchronize threads"
keypoints:
- ""
---

So far we looked at how to use CUDA to accelerate the computation, but a common pattern in all the examples we encountered so far is that threads worked in isolation.
While having different threads perform the same operation on different data is a good pattern for working with GPUs, there are cases in which threads need to communicate.
This communication may be necessary because of the way the algorithm we are trying to implement works, or it may derive from a performance goal we are trying to achieve.

# Shared Memory

Shared memory is a CUDA memory space that is shared by all threads in a thread block.
In this case *shared* means that all threads in a thread block can write and read to block-allocated shared memory, and all changes to this memory will be eventually available to all threads in the block.

To allocate an array in shared memory we need to preface the definition with the identifier `__shared__`.

> ## Challenge: use of shared memory
>
> Modify the following code to allocate the `temp` array in shared memory.
>
> ~~~
> extern "C"
> __global__ void vector_add(const float * A, const float * B, float * C, const int size)
> {
>   int item = (blockIdx.x * blockDim.x) + threadIdx.x;
>   float temp[3];
>
>   if ( item < size )
>   {
>       temp[0] = A[item];
>       temp[1] = B[item];
>       temp[2] = temp[0] + temp[1];
>       C[item] = temp[2];
>   }
> }
> ~~~
> {: .language-c}
> > ## Solution
> > 
> > To use shared memory for the `temp` array add the identifier `__shared__` to its definition, like in the following code.
> >
> > ~~~
> > extern "C"
> > __global__ void vector_add(const float * A, const float * B, float * C, const int size)
> > {
> >   int item = (blockIdx.x * blockDim.x) + threadIdx.x;
> >   __shared__ float temp[3];
> >
> >   if ( item < size )
> >   {
> >       temp[0] = A[item];
> >       temp[1] = B[item];
> >       temp[2] = temp[0] + temp[1];
> >       C[item] = temp[2];
> >   }
> > }
> > ~~~
> > {: .language-c}
> {: .solution}
{: .challenge}

While syntactically correct, the previous example is functionally wrong.
The reason is that the `temp` array is not anymore private to the thread allocating it, but it is now shared by the whole thread block.

> ## Challenge: what is the result of the previous code block?
>
> The previous code example is functionally wrong. Can you guess what the result of its execution will be?
>
> > ## Solution
> >
> > The result is non deterministic, and definitely not the same as the previous versions of `vector_add`.
> > Threads will overwrite each other temporary values,and there will be no guarantee on which value is visible by each thread.
> {: .solution}
{: .challenge}

To fix the previous kernel we should allocate enough shared memory for each thread to store three values, so that each thread has its own section of the shared memory array to work with.

To allocate enough memory we need to replace the constant 3 in `__shared__ float temp[3];` with something else.
If we know that each thread block has 1024 threads, we can write something like the following:

~~~
__shared__ float temp[3 * 1024];
~~~
{: .language-c}

But we know by experience that having constants in the code is not a scalable and maintainable solution.
The problem is that we need to have a constant value if we want to declare a shared memory array, because the compiler needs to know how much memory to allocate.

A solution to this problem is to declare our array as a pointer, such as:

~~~
extern __shared__ float temp[];
~~~
{: .language-c}

And then use CuPy to instruct the compiler about how much shared memory, in bytes, each thread block needs:

~~~
# execute the code
vector_add_gpu((2, 1, 1), (size // 2, 1, 1), (a_gpu, b_gpu, c_gpu, size), shared_mem=((size // 2) * 3 * cupy.dtype(cupy.float32).itemsize))
~~~
{: .language-python}

So before compiling and executing the kernel, we need to set `attributes.max_dynamic_shared_size_bytes` with the number of bytes necessary.
As you may notice, we had to retrieve the size in bytes of the data type `cupy.float32`, and this is done with `cupy.dtype(cupy.float32).itemsize`.

After these changes, the body of the kernel needs to be modified to use the right indices:

~~~
extern "C"
__global__ void vector_add(const float * A, const float * B, float * C, const int size)
{
  int item = (blockIdx.x * blockDim.x) + threadIdx.x;
  int offset = threadIdx.x * 3;
  extern __shared__ float temp[];

  if ( item < size )
  {
      temp[offset + 0] = A[item];
      temp[offset + 1] = B[item];
      temp[offset + 2] = temp[offset + 0] + temp[offset + 1];
      C[item] = temp[offset + 2];
  }
}
~~~
{: .language-c}

And for completeness, we present the full Python code.

~~~
import math
import numpy
import cupy

# vector size
size = 2048

# GPU memory allocation
a_gpu = cupy.random.rand(size, dtype=cupy.float32)
b_gpu = cupy.random.rand(size, dtype=cupy.float32)
c_gpu = cupy.zeros(size, dtype=cupy.float32)
gpu_args = (a_gpu, b_gpu, c_gpu, size)

# CPU memory allocation
a_cpu = cupy.asnumpy(a_gpu)
b_cpu = cupy.asnumpy(b_gpu)
c_cpu = numpy.zeros(size, dtype=numpy.float32)

# CUDA code
vector_add_cuda_code = r'''
extern "C"
__global__ void vector_add(const float * A, const float * B, float * C, const int size)
{
  int item = (blockIdx.x * blockDim.x) + threadIdx.x;
  int offset = threadIdx.x * 3;
  extern __shared__ float temp[];

  if ( item < size )
  {
      temp[offset + 0] = A[item];
      temp[offset + 1] = B[item];
      temp[offset + 2] = temp[offset + 0] + temp[offset + 1];
      C[item] = temp[offset + 2];
  }
}
'''

# compile and execute code
vector_add_gpu = cupy.RawKernel(vector_add_cuda_code, "vector_add")
threads_per_block = 32
grid_size = (int(math.ceil(size / threads_per_block)), 1, 1)
block_size = (threads_per_block, 1, 1)
vector_add_gpu(grid_size, block_size, gpu_args, shared_mem=(threads_per_block * 3 * cupy.dtype(cupy.float32).itemsize))

# execute Python code and compare results
vector_add(a_cpu, b_cpu, c_cpu, size)
numpy.allclose(c_cpu, c_gpu)
~~~
{: .language-python}

The code is now correct, although it is still not very useful.
We are definitely using shared memory, and we are using it the correct way, but there is no performance gain we achieved by doing so.
In practice, we are making our code slower, not faster, because shared memory is slower than registers.

Let us, therefore, work on an example where using shared memory is actually useful.
We start again with some Python code.

~~~
def histogram(input_array, output_array):
    for item in input_array:
        output_array[item] = output_array[item] + 1
    return output_array
~~~
{: .language-python}

The `histogram` function, as the name suggests, computes the histogram of an array of integers, i.e. counts how many instances of each integer are in `input_array`, and writes the count in `output_array`.
We can now generate some data and run the code.

~~~
input_array = numpy.random.randint(256, size=2048, dtype=numpy.int32)
output_array = numpy.zeros(256, dtype=numpy.int32)
output_array = histogram(input_array, output_array)
~~~
{: .language-python}

Everything as expected.
We can now write equivalent code in CUDA.

~~~
__global__ void histogram(const int * input, int * output)
{
    int item = (blockIdx.x * blockDim.x) + threadIdx.x;

    output[input[item]] = output[input[item]] + 1;
}
~~~
{: .language-c}

> ## Challenge: error in the histogram
>
> If you look at the CUDA `histogram` code, there is a logical error that prevents it to produce the right result.
> Can you spot it?
>
> ~~~
> __global__ void histogram(const int * input, int * output)
> {
>     int item = (blockIdx.x * blockDim.x) + threadIdx.x;
> 
>     output[input[item]] = output[input[item]] + 1;
> }
> ~~~
> {: .language-c}
>
> > ## Solution
> >
> > The GPU is a highly parallel device, executing multiple threads at the same time.
> > In the previous code different threads could be updating the same output item at the same time, producing wrong results.
> {: .solution}
{: .challenge}

To solve this problem, we need to use a function from the CUDA library named `atomicAdd`.
This function ensures that the increment of `output_array` happens in an atomic way, so that there are no conflicts in case multiple threads want to update the same item at the same time.

~~~
__global__ void histogram(const int * input, int * output)
{
    int item = (blockIdx.x * blockDim.x) + threadIdx.x;

    atomicAdd(&(output[input[item]]), 1);
}
~~~
{: .language-c}

And the full Python code snippet.

~~~
import math
import numpy
import cupy

# input size
size = 2048

# allocate memory on CPU and GPU
input_gpu = cupy.random.randint(256, size=size, dtype=cupy.int32)
input_cpu = cupy.asnumpy(input_gpu)
output_gpu = cupy.zeros(256, dtype=cupy.int32)
output_cpu = cupy.asnumpy(output_gpu)

# CUDA code
histogram_cuda_code = r'''
extern "C"
__global__ void histogram(const int * input, int * output)
{
    int item = (blockIdx.x * blockDim.x) + threadIdx.x;

    atomicAdd(&(output[input[item]]), 1);
}
'''

# compile and setup CUDA code
histogram_gpu = cupy.RawKernel(histogram_cuda_code, "histogram")
threads_per_block = 256
grid_size = (int(math.ceil(size / threads_per_block)), 1, 1)
block_size = (threads_per_block, 1, 1)

# execute code on CPU and GPU
histogram_gpu(grid_size, block_size, (input_gpu, output_gpu))
histogram(input_cpu, output_cpu)

# compare results
numpy.allclose(output_cpu, output_gpu)
~~~
{: .language-python}

The CUDA code is now correct, and computes the same result as the Python code.
However, we are accumulating the results directly in global memory, and the more conflicts we have in global memory, the lower performance our `histogram` will have.
Moreover, the access pattern to the output array is very irregular, being dependent on the content of the input array.
The best performance is obtained on the GPU when consecutive threads access consecutive addresses in memory, and this is not the case in our code.

As you may expect, we can improve performance by using shared memory.

> ## Challenge: use shared memory to speed up the histogram
>
> Implement a new version of the CUDA `histogram` function that uses shared memory to reduce conflicts in global memory. 
>
> ~~~
> __global__ void histogram(const int * input, int * output)
> {
>     int item = (blockIdx.x * blockDim.x) + threadIdx.x;
> 
>     atomicAdd(&(output[input[item]]), 1);
> }
> ~~~
> {: .language-c}
>
> Hint: for this exercise, you can safely assume that the size of `output` is the same as the number of threads in a block.
> 
> Hint: `atomicAdd` can be used on both global and shared memory.
>
> > ## Solution
> >
> > The following code shows one of the possible solutions.
> > 
> > ~~~
> > __global__ void histogram(const int * input, int * output)
> > {
> >     int item = (blockIdx.x * blockDim.x) + threadIdx.x;
> >     extern __shared__ int temp_histogram[];
> > 
> >     atomicAdd(&(temp_histogram[input[item]]), 1);
> >     atomicAdd(&(output[threadIdx.x]), temp_histogram[threadIdx.x]);
> > }
> > ~~~
> > {: .language-c}
> >
> > The idea behind this solution is to reduce the expensive conflicts in global memory by having a temporary histogram in shared memory.
> > After a block has finished processing its fraction of the input array, and the local histogram is populated, threads collaborate to update the global histogram.
> > Not only this solution potentially reduces the conflicts in global memory, it also produces a better access pattern because threads read adjacent items of the `input` array, and write to adjacent elements of the `output` array.
> >
> {: .solution}
{: .challenge}

# Thread Synchronization

There is still one potentially big issue in the `histogram` code we just wrote, and the issue is that shared memory is not coherent without explicit synchronization.
The problem lies in the following two lines of code:

~~~
atomicAdd(&(temp_histogram[input[item]]), 1);
atomicAdd(&(output[threadIdx.x]), temp_histogram[threadIdx.x]);
~~~
{: .language-c}

In the first line each thread updates one arbitrary position in shared memory, depending on the value of the input, while in the second line each thread reads the element in shared memory corresponding to its thread ID.
However, the changes to shared memory are not automatically available to all other threads, and therefore the final result may not be correct.

To solve this issue, we need to explicitly synchronize all threads in a block, so that memory operations are also finalized and visible to all.
To synchronize threads in a block, we use the `__syncthreads()` CUDA function.
Moreover, shared memory is not initialized, and the programmer needs to take care of that too.

~~~
__global__ void histogram(const int * input, int * output)
{
    int item = (blockIdx.x * blockDim.x) + threadIdx.x;
    extern __shared__ int temp_histogram[];
 
    // Initialize shared memory and synchronize
    temp_histogram[threadId.x] = 0;
    __syncthreads();

    // Compute shared memory histogram and synchronize
    atomicAdd(&(temp_histogram[input[item]]), 1);
    __syncthreads();

    // Update global histogram
    atomicAdd(&(output[threadIdx.x]), temp_histogram[threadIdx.x]);
}
~~~
{: .language-c}

And the full Python code snippet.

~~~
import math
import numpy
import cupy

# input size
size = 2048

# allocate memory on CPU and GPU
input_gpu = cupy.random.randint(256, size=size, dtype=cupy.int32)
input_cpu = cupy.asnumpy(input_gpu)
output_gpu = cupy.zeros(256, dtype=cupy.int32)
output_cpu = cupy.asnumpy(output_gpu)

# CUDA code
histogram_cuda_code = r'''
extern "C"
__global__ void histogram(const int * input, int * output)
{
    int item = (blockIdx.x * blockDim.x) + threadIdx.x;
    extern __shared__ int temp_histogram[];
 
    // Initialize shared memory and synchronize
    temp_histogram[threadIdx.x] = 0;
    __syncthreads();

    // Compute shared memory histogram and synchronize
    atomicAdd(&(temp_histogram[input[item]]), 1);
    __syncthreads();

    // Update global histogram
    atomicAdd(&(output[threadIdx.x]), temp_histogram[threadIdx.x]);
}
'''

# compile and setup CUDA code
histogram_gpu = cupy.RawKernel(histogram_cuda_code, "histogram")
threads_per_block = 256
grid_size = (int(math.ceil(size / threads_per_block)), 1, 1)
block_size = (threads_per_block, 1, 1)

# execute code on CPU and GPU
histogram_gpu(grid_size, block_size, (input_gpu, output_gpu), shared_mem=(threads_per_block * cupy.dtype(cupy.int32).itemsize))
histogram(input_cpu, output_cpu)

# compare results
numpy.allclose(output_cpu, output_gpu)
~~~
{: .language-python}

{% include links.md %}
