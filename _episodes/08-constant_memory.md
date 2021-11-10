---
title: "Constant Memory"
teaching: 20
exercises: 20
questions:
- "Question"
objectives:
- "Understanding when to use constant memory"
keypoints:
- ""
---

# Constant Memory

Constant memory is a read-only cache which content can be broadcasted to multiple threads in a block.
A variable allocated in constant memory needs to be declared in CUDA by using the special `__constant__` identifier, and it must be a global variable, i.e. it must be declared in the scope that contains the kernel, not inside the kernel itself.
If all of this sounds complex do not worry, we are going to see how this works with an example.

~~~
extern "C" {
#define BLOCKS 2

__constant__ float factors[BLOCKS];

__global__ void sum_and_multiply(const float * A, const float * B, float * C, const int size)
{
    int item = (blockIdx.x * blockDim.x) + threadIdx.x;
    C[item] = (A[item] + B[item]) * factors[blockIdx.x];
}
}
~~~
{: .language-c}

In the previous code snippet we implemented a kernel that, given two vectors `A` and `B`, stores their element-wise sum in a third vector, `C`, scaled by a certain factor; this factor is the same for all threads in the same thread block.
Because these factors are shared, i.e. all threads in the same thread block use the same factor for scaling their sums, it is a good idea to use constant memory for the `factors` array.
In fact you can see that the definition of `factors` is preceded by the `__constant__` keyword, and said definition is in the global scope.
It is important to note that the size of the constant array needs to be known at compile time, therefore the use of the `define` preprocessor statement.
On the kernel side there is no need to do more, the `factors` vector can be normally accessed inside the code as any other vector, and because it is a global variable it does not need to be passed to the kernel as a function argument.

The initialization of constant memory happens on the host side, and we show how this is done in the next code snippet.

~~~
# compile the code
module = cupy.RawModule(code=cuda_code)
# allocate and copy constant memory
factors_ptr = module.get_global("factors")
factors_gpu = cupy.ndarray(2, cupy.float32, factors_ptr)
factors_gpu[...] = cupy.random.random(2, dtype=cupy.float32)
~~~
{: .language-python}

From the previous code it is clear that dealing with constant memory is a slightly more verbose affair than usual.
First, we need to compile the code, that in this case is contained in a Python string named `cuda_code`.
This is necessary because constant memory is defined in the CUDA code, so we need CUDA to allocate the necessary memory, and then provide us with a pointer to this memory.
By calling the method `get_global` we ask the CUDA subsystem to provide us with the location of a global object, in this case the array `factors`.
We can then create our own CuPy array and point that to the object returned by `get_global`, so that we can use it in Python as we would normally do.
Note that we use the constant `2` for the size of the array, the same number we are using in the CUDA code; it is important that we use the same number or we may end up accessing memory that is outside the bound of the array.
Lastly, we initialize the array with some random floating point numbers.

> ## Challenge: print the content of constant memory
>
> What should be the output of the following line of code?
>
> ~~~
> print(factors_gpu)
> ~~~
> {: .language-python}
> > ## Solution
> > 
> > In our case the output of this line of code is two floating point numbers, e.g. `[0.11390183 0.2585096 ]`.
> > However, we are not really accessing the content of the GPU's constant memory from the host, we are simply accessing the host-side copy of the data maintained by CuPy.
> {: .solution}
{: .challenge}

We can now combine all the code together and execute it.

~~~
# size of the vectors
size = 2048

# allocating and populating the vectors
a_gpu = cupy.random.rand(size, dtype=cupy.float32)
b_gpu = cupy.random.rand(size, dtype=cupy.float32)
c_gpu = cupy.zeros(size, dtype=cupy.float32)
# prepare arguments
args = (a_gpu, b_gpu, c_gpu, size)

# CUDA code
cuda_code = r'''
extern "C" {
#define BLOCKS 2

__constant__ float factors[BLOCKS];

__global__ void sum_and_multiply(const float * A, const float * B, float * C, const int size)
{
    int item = (blockIdx.x * blockDim.x) + threadIdx.x;
    C[item] = (A[item] + B[item]) * factors[blockIdx.x];
}
}
'''

# compile and access the code
module = cupy.RawModule(code=cuda_code)
sum_and_multiply = module.get_function("sum_and_multiply")
# allocate and copy constant memory
factors_ptr = module.get_global("factors")
factors_gpu = cupy.ndarray(2, cupy.float32, factors_ptr)
factors_gpu[...] = cupy.random.random(2, dtype=cupy.float32)

sum_and_multiply((2, 1, 1), (size // 2, 1, 1), args)
~~~
{: .language-python}

As you can see the code is not very general, it uses constants and works only with two blocks, but it is a working example of how to use constant memory.

> ## Challenge: generalize the previous code
>
> Have a look again at the code using constant memory, and make it general enough to be able to run on input of arbitrary size.
> Experiment with some different input sizes.
>
> > ## Solution
> >
> > One of the possible solutions is the following one. 
> >
> > ~~~
> > # size of the vectors
> > size = 10**6
> > 
> > # allocating and populating the vectors
> > a_gpu = cupy.random.rand(size, dtype=cupy.float32)
> > b_gpu = cupy.random.rand(size, dtype=cupy.float32)
> > c_gpu = cupy.zeros(size, dtype=cupy.float32)
> > # prepare arguments
> > args = (a_gpu, b_gpu, c_gpu, size)
> > 
> > # CUDA code
> > cuda_code = r'''
> > extern "C" {
> > __constant__ float factors[BLOCKS];
> > 
> > __global__ void sum_and_multiply(const float * A, const float * B, float * C, const int size)
> > {
> >     int item = (blockIdx.x * blockDim.x) + threadIdx.x;
> >     if ( item < size )
> >     {
> >         C[item] = (A[item] + B[item]) * factors[blockIdx.x];
> >     }
> > }
> > }
> > '''
> >
> > # compute the number of blocks and replace "BLOCKS" in the CUDA code
> > threads_per_block = 1024
> > num_blocks = int(math.ceil(size / threads_per_block))
> > cuda_code = cuda_code.replace("BLOCKS", f"{num_blocks}") 
> >
> > # compile and access the code
> > module = cupy.RawModule(code=cuda_code)
> > sum_and_multiply = module.get_function("sum_and_multiply")
> > # allocate and copy constant memory
> > factors_ptr = module.get_global("factors")
> > factors_gpu = cupy.ndarray(num_blocks, cupy.float32, factors_ptr)
> > factors_gpu[...] = cupy.random.random(num_blocks, dtype=cupy.float32)
> > 
> > sum_and_multiply((num_blocks, 1, 1), (threads_per_block, 1, 1), args)
> > ~~~
> > {: .language-python}
> >
> {: .solution}
{: .challenge}

{% include links.md %}