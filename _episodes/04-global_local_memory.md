---
title: "Global and Local Memory"
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

# Global Memory

Accessible by the host and all threads on the GPU.
Only way to exchange data between CPU and GPU.

# Local Memory

Only accessible by the thread allocating it.
All threads allocate their own local memory, but cannot see the content of the memory of the other threads.

{% include links.md %}