---
title: "Shared Memory and Synchronization"
teaching: 0
exercises: 0
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
> Modify the following code to use shared memory for the `temp` array.
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
> The previous code example is functionally wrong. Do you know what the result of its execution will be?
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

```
__shared__ float temp[3 * 1024];
```
{: .language-c}

But we know by experience that having constants in the code is not a scalable and maintainable solution.
The problem is that we need to have a constant value if we want to declare a shared memory array, because the compiler needs to know how much memory to allocate.

A solution to this problem is to declare our array as a pointer, such as:

```
__shared__ float * temp;
```
{: .language-c}

And then use CuPy to instruct the compiler about how much shared memory each thread block needs:

```
# execute the code
vector_add_gpu.attributes.max_dynamic_shared_size_bytes = (size // 2) * 3 * cupy.dtype(cupy.float32).itemsize
vector_add_gpu((2, 1, 1), (size // 2, 1, 1), (a_gpu, b_gpu, c_gpu, size))
```
{: .language-python}

So before compiling and executing the kernel, we need to set `attributes.max_dynamic_shared_size_bytes` with the number of bytes necessary.
As you may notice, we had to retrieve the size in bytes of the data type `cupy.float32`, and this is done with `cupy.dtype(cupy.float32).itemsize`.

After these changes, the body of the kernel needs to be modified to use the right indices:

```
temp[item * 3] = A[item];
temp[(item * 3) + 1] = B[item];
temp[(item * 3) + 2] = temp[item * 3] + temp[(item * 3) + 1];
C[item] = temp[(item * 3) + 2];
```
{: .language-c}

The code is now correct, although it is still not very useful.
We are definitely using shared memory, and we are using it the correct way, but there is no performance gain we achieved by doing so.
In practice, we are making our code slower, not faster, because shared memory is slower than registers.

Let us, therefore, work on an example where using shared memory is actually useful.
We start again with some Python code.

```
def histogram(input_array, output_array):
    for item in input_array:
        output_array[item] = output_array[item] + 1
    return output_array
```
{: .language-python}

The `histogram` function, as the name suggests, computes the histogram of an array of integers, i.e. counts how many instances of each integer are in `input_array`, and writes the count in `output_array`.
We can now generate some data and run the code.

```
input_array = numpy.random.randint(256, size=2048, dtype=numpy.int32)
output_array = numpy.zeros(256, dtype=numpy.int32)
output_array = histogram(input_array, output_array)
```
{: .language-python}

Everything as expected.
We can now write equivalent code in CUDA.

```
__global__ void histogram(const int * input, int * output)
{
    int item = (blockIdx.x * blockDim.x) + threadIdx.x;

    output[input[item]] = input[item] + 1;
}
```
{: .language-c}

> ## Challenge: error in the histogram
>
> If you look at the CUDA `histogram` code, there is a logical error that prevents the code to produce the right result.
> Can you spot it?
>
> > ## Solution
> >
> > The GPU is a highly parallel device, executing multiple threads at the same time.
> > In the previous code different threads could be updating the same output item at the same time, producing wrong results.
> {: .solution}
{: .challenge}

To solve this problem, we need to use a function from the CUDA library named `atomicAdd`.
This function ensures that the increment of `output_array` happens in an atomic way, so that there are no conflicts in case multiple threads want to update the same item at the same time.

```
__global__ void histogram(const int * input, int * output)
{
    int item = (blockIdx.x * blockDim.x) + threadIdx.x;

    atomicAdd(&(output[input[item]]), 1);
}
```
{: .language-c}



# Thread Synchronization

{% include links.md %}