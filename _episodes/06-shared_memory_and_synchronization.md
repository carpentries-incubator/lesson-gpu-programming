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

> ## Challenge: use shared memory
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
The reason is that the `temp` array is not anymore private to the thread allocating it, but it is now shared by the whole thread block, and so threads overwrite each other results making the final result wrong.

# Thread Synchronization

{% include links.md %}