---
title: "Concurrent access to the GPU"
teaching: 20
exercises: 20
---

:::::::::::::::::::::::::::::::::::::: questions
- "Is it possible to concurrently execute more than one kernel on a single GPU?"
::::::::::::::::::::::::::::::::::::::

:::::::::::::::::::::::::::::::::::::: objectives
- "Understand how to use CUDA streams and events"
::::::::::::::::::::::::::::::::::::::

# Concurrently execute two kernels on the same GPU

So far we only focused on completing one operation at the time on the GPU, writing and executing a single CUDA kernel each time.
However the GPU has enough resources to perform more than one task at the same time.

Let us assume that, for our program, we need to compute both a list of prime numbers, and a histogram, two kernels that we developed in this same lesson.
We could write both kernels in CUDA, and then execute them on the GPU, as shown in the following code.

~~~python
import numpy as np
import cupy
import math
from cupyx.profiler import benchmark

upper_bound = 100_000
histogram_size = 2**25

# GPU code
check_prime_gpu_code = r'''
extern "C"
__global__ void all_primes_to(int size, int * const all_prime_numbers)
{
    int number = (blockIdx.x * blockDim.x) + threadIdx.x;
    int result = 1;

    if ( number < size )
    {
        for ( int factor = 2; factor <= number / 2; factor++ )
        {
            if ( number % factor == 0 )
            {
                result = 0;
                break;
            }
        }

        all_prime_numbers[number] = result;
    }
}
'''
histogram_cuda_code = r'''
extern "C"
__global__ void histogram(const int * input, int * output)
{
    int item = (blockIdx.x * blockDim.x) + threadIdx.x;
    __shared__ int temp_histogram[256];
 
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

# Allocate memory
all_primes_gpu = cupy.zeros(upper_bound, dtype=cupy.int32)
input_gpu = cupy.random.randint(256, size=histogram_size, dtype=cupy.int32)
output_gpu = cupy.zeros(256, dtype=cupy.int32)

# Compile and setup the grid
all_primes_to_gpu = cupy.RawKernel(check_prime_gpu_code, "all_primes_to")
grid_size_primes = (int(math.ceil(upper_bound / 1024)), 1, 1)
block_size_primes = (1024, 1, 1)
histogram_gpu = cupy.RawKernel(histogram_cuda_code, "histogram")
threads_per_block_hist = 256
grid_size_hist = (int(math.ceil(histogram_size / threads_per_block_hist)), 1, 1)
block_size_hist = (threads_per_block_hist, 1, 1)

# Execute the kernels
all_primes_to_gpu(grid_size_primes, block_size_primes, (upper_bound, all_primes_gpu))
histogram_gpu(grid_size_hist, block_size_hist, (input_gpu, output_gpu))

# Save results
output_one = all_primes_gpu
output_two = output_gpu
~~~

In the previous code, after allocating memory and compiling, we execute and measure the performance of one kernel, and when we are done we do the same for the other kernel.

This is technically correct, but there is no reason why one kernel should be executed before the other, because there is no dependency between these two operations.

Therefore, while this is fine in our example, in a real application we may want to run the two kernels concurrently on the GPU to reduce the total execution time.
To achieve this in CUDA we need to use CUDA *streams*.

A stream is a sequence of GPU operations that is executed in order, and so far we have been implicitly using the defaul stream.
This is the reason why all the operations we issued, such as data transfers and kernel invocations, are performed in the order we specify them in the Python code, and not in any other.

Have you wondered why after requesting data transfers to and from the GPU, we do not stop to check if they are complete before performing operations on such data?
The reason is that within a stream all operations are carried out in order, so the kernel calls in our code are always performed after the data transfer from host to device is complete, and so on.

If we want to create new CUDA streams, we can do it this way using CuPy.

~~~python
stream_one = cupy.cuda.Stream()
stream_two = cupy.cuda.Stream()
~~~

We can then execute the kernels in different streams by using the Python `with` statement.

~~~python
with stream_one:
    all_primes_to_gpu(grid_size_primes, block_size_primes, (upper_bound, all_primes_gpu))
with stream_two:
    histogram_gpu(grid_size_hist, block_size_hist, (input_gpu, output_gpu))
~~~

Using the `with` statement we implicitly execute the CUDA operations in the code block using that stream.
The result of doing this is that the second kernel, i.e. `histogram_gpu`, does not need to wait for `all_primes_to_gpu` to finish before being executed.

# Stream synchronization

If we need to wait for all operations on a certain stream to finish, we can call the `synchronize` method.
Continuing with the previous example, in the following Python snippet we wait for the execution of `all_primes_to_gpu` on `stream_one` to finish.

~~~python
stream_one.synchronize()
~~~

This synchronization primitive is useful when we need to be sure that all operations on a stream are finished, before continuing.
It is, however, a bit coarse grained.
Imagine to have a stream with a whole sequence of operations enqueued, and another stream with one data dependency on one of these operations.
If we use `synchronize`, we wait until all operations of said stream are completed before executing the other stream, thus negating the whole reason of using streams in the first place.

A possible solution is to insert a CUDA *event* at a certain position in the stream, and then wait specifically for that event.
Events are created in Python in the following way.

~~~python
interesting_event = cupy.cuda.Event()
~~~

And can then be added to a stream by using the `record` method.
In the following example we will create two streams: in the first we will execute `histogram_gpu` twice, while in the second one we will execute `all_primes_to_gpu` with the condition that the kernel in the second stream is executed only after the first histogram is computed.

~~~python
stream_one = cupy.cuda.Stream()
stream_two = cupy.cuda.Stream()
sync_point = cupy.cuda.Event()

with stream_one:
    all_primes_to_gpu(grid_size_primes, block_size_primes, (upper_bound, all_primes_gpu))
    sync_point.record(stream=stream_one)
    all_primes_to_gpu(grid_size_primes, block_size_primes, (upper_bound, all_primes_gpu))
with stream_two:
    stream_two.wait_event(sync_point)
    histogram_gpu(grid_size_hist, block_size_hist, (input_gpu, output_gpu))
~~~

With streams and events, it is possible to build up arbitrary execution graphs for complex operations on the GPU.

# Measure execution time using streams and events

We can also use streams and events to measure the execution time of kernels, without having to use the CuPy `benchmark` function.
In the following example, we go back to the `vector_add` code and add code that uses events on the default stream to measure the execution time.

~~~python
import math
import cupy
import numpy as np

# size of the vectors
size = 100_000_000

# allocating and populating the vectors
a_gpu = cupy.random.rand(size, dtype=cupy.float32)
b_gpu = cupy.random.rand(size, dtype=cupy.float32)
c_gpu = cupy.zeros(size, dtype=cupy.float32)
a_cpu = cupy.asnumpy(a_gpu)
b_cpu = cupy.asnumpy(b_gpu)
c_cpu = np.zeros(size, dtype=np.float32)

# CPU code
def vector_add(A, B, C, size):
    for item in range(0, size):
        C[item] = A[item] + B[item]

# CUDA vector_add
vector_add_gpu = cupy.RawKernel(r'''
extern "C"
__global__ void vector_add(const float * A, const float * B, float * C, const int size)
{
   int item = (blockIdx.x * blockDim.x) + threadIdx.x;
   if ( item < size )
   {
      C[item] = A[item] + B[item];
   }
}
''', "vector_add")

# execute the code and measure time
threads_per_block = 1024
grid_size = (int(math.ceil(size / threads_per_block)), 1, 1)
block_size = (threads_per_block, 1, 1)
%timeit -n 1 -r 1 vector_add(a_cpu, b_cpu, c_cpu, size)
gpu_times = []
for _ in range(0, 10):
    start_gpu = cupy.cuda.Event()
    end_gpu = cupy.cuda.Event()
    start_gpu.record()
    vector_add_gpu(grid_size, block_size, (a_gpu, b_gpu, c_gpu, size))
    end_gpu.record()
    end_gpu.synchronize()
    gpu_times.append(cupy.cuda.get_elapsed_time(start_gpu, end_gpu))
gpu_avg_time = np.average(gpu_times)
print(f"{gpu_avg_time:.6f} s")

# test
if np.allclose(c_cpu, c_gpu):
    print("Correct results!")
else:
    print("Wrong results!")
~~~

:::::::::::::::::::::::::::::::::::::: keypoints
- ""
::::::::::::::::::::::::::::::::::::::
