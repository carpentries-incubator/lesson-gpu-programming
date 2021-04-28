---
title: "Registers, Global and Local Memory"
teaching: 0
exercises: 0
questions:
- "Question"
objectives:
- "Understanding the difference between registers and device memory"
- "Understanding the difference between local and global memory"
keypoints:
- "Registers can be used to save memory operations"
---

In this episode we are going to look at three different memories: registers, global memory, and local memory.

# Registers

Registers are on-chip fast memories.

Did we use registers in the previous episode?
Yes we did, the variable `item` is stored in a register; all scalar variables defined in the CUDA code are stored in registers.

~~~
extern "C"
__global__ void vector_add(const float * A, const float * B, float * C, const int size)
{
   int item = (blockIdx.x * blockDim.x) + threadIdx.x;
   
   if ( item < size )
   {
      C[item] = A[item] + B[item];
   }
}
~~~
{: .language-c}

Registers are local to a thread, and each thread has its own registers; values in registers cannot be accessed by other threads, even from the same block, and are not available for the host.
Registers are not permanent, therefore data stored in registers is only available during the execution of a thread.

We certainly did use more than one register in the previous code; most certainly the content of the vectors, and the temporary result of `A[item] + B[item]` are first stored in registers.

We could also make this explicit by rewriting our code in a slightly different, but equivalent, way.

~~~
extern "C"
__global__ void vector_add(const float * A, const float * B, float * C, const int size)
{
   int item = (blockIdx.x * blockDim.x) + threadIdx.x;
   float temp_a, temp_b, temp_c;

   if ( item < size )
   {
       temp_a = A[item];
       temp_b = B[item];
       temp_c = temp_a + temp_b;
       C[item] = temp_c;
   }
}
~~~
{: .language-c}

In our case this is not necessary, we are just doing unnecessary work that the compiler should do. 

However, explicit register usage can be important for reusing items already loaded from memory.
We will look at some examples of using registers for memory reuse in future episodes.

Small arrays, which size is known at compile time, will also be allocated in registers by the compiler.
We can rewrite the previous example to work with an array of registers.

~~~
extern "C"
__global__ void vector_add(const float * A, const float * B, float * C, const int size)
{
   int item = (blockIdx.x * blockDim.x) + threadIdx.x;
   float temp[3];

   if ( item < size )
   {
       temp[0] = A[item];
       temp[1] = B[item];
       temp[2] = temp_a + temp_b;
       C[item] = temp[2];
   }
}
~~~
{: .language-c}

Once again, this is not something that we would normally do, and it is just an example to learn how to work with arrays of registers.

**TODO** add exercise with registers; maybe something like "how many registers do you think we are using in this code?" or similar.

# Global Memory

Accessible by the host and all threads on the GPU.
Only way to exchange data between CPU and GPU.

> ## Challenge: identify when global memory is used
>
> Observe the code of `vector_add` and identify where global memory is used.
>
> ~~~
> extern "C"
> __global__ void vector_add(const float * A, const float * B, float * C, const int size)
> {
>    int item = (blockIdx.x * blockDim.x) + threadIdx.x;
>    
>    if ( item < size )
>    {
>       C[item] = A[item] + B[item];
>    }
> }
> ~~~
> {: .language-c}
> > ## Solution
> > The vectors `A`, `B`, and `C` are in global memory.
> {: .solution}
{: .challenge}

Memory allocated on the host, and passed to the kernel as a function parameter, is allocated in global memory.
In CUDA there is no particular keyword to specify for global memory allocation.

# Local Memory

Only accessible by the thread allocating it.
All threads allocate their own local memory, but cannot see the content of the memory of the other threads.

It is not a fast memory, it has the same throughput and latency of global memory, but it is much larger than registers.
It is automatically used by the CUDA compiler to store spilled registers, i.e. variables that cannot be kept in registers because there is not enough space.

# Constant Memory

{% include links.md %}