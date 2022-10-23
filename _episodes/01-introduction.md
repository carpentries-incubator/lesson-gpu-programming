---
title: "Introduction"
teaching: 15
exercises: 0
questions:
- "What is a Graphics Processing Unit?"
- "Can a GPU be used for anything else than graphics?"
- "Are GPUs faster than CPUs?"
objectives:
- "Understand the differences between CPU and GPU"
- "See the possible performance benefits of GPU acceleration"
keypoints:
- "CPUs and GPUs are both useful and each has its own place in our toolbox"
- "In the context of GPU programming, we often refer to the GPU as the *device* and the CPU as the *host*"
- "Using GPUs to accelerate computation can provide large performance gains"
- "Using the GPU with Python is not particularly difficult"
---

# Graphics Processing Unit

The Graphics Processing Unit (**GPU**) is one of the components of a computer's video card, together with specialized memory and different Input/Output (I/O) units.
In the context of the video card, the GPU fulfills a role similar to the one that the Central Processing Unit (**CPU**) has in a general purpose computing system: it processes input data to generate some kind of output.
In the traditional context of video cards, GPUs process data in order to render images on an output device, such as a screen.
However, modern GPUs are general purpose computing devices that can be used to perform any kind of computation, and this is what we are going to do in this lesson.

# Parallel by Design

But what is the reason to use GPUs to perform general purpose computation, when computers already have fast CPUs that are able to perform any kind of computation?
One way to answer this question is to go back to the roots of what a GPU is designed to do: render images.

An image can be seen as a matrix of points called **pixels** (a portmanteau of the words *picture* and *element*), with each pixel representing the color the image should have in that particular point, and the traditional task performed by video cards is to produce the images a user will see on the screen.
A single 4K UHD image contains more than 8 million pixels.
For a GPU  to render a continuous stream of 25 4K frames (images) per second, enough so that users not experience delay in a videogame, movie, or any other video output, it must process over 200 million pixels per second.
So GPUs are designed to render multiple pixels at the same time, and they are designed to do it efficiently.
This design principle results in the GPU being, from a hardware point of view, very different from a CPU.

The CPU is a very general purpose device, good at different tasks, being them parallel or sequential in nature; it is also designed for interaction with the user, so it has to be responsive and guarantee minimal latency.
In practice, we want our CPU to be available whenever we sent it a new task.
The result is a device where most of the silicon is used for memory caches and control-flow logic, not just compute units.

By contrast, most of the silicon on a GPU is actually used for compute units.
The GPU does not need an overly complicated cache hierarchy, nor it does need complex control logic, because the overall goal is not to minimize the latency of any given thread, but to maximize the throughput of the whole computation.
With many compute units available, the GPU can run massively parallel programs, programs in which thousands of threads are executed at the same time, while thousands more are ready for execution to hide the cost of memory operations.

A high-level introduction on the differences between CPU and GPU can also be found in the following YouTube video.

[![Graphics Processing Unit (GPU)](http://img.youtube.com/vi/bZdxcHEM-uc/0.jpg)](https://www.youtube.com/watch?v=bZdxcHEM-uc "Graphics Processing Unit (GPU)")

# Speed Benefits

So, GPUs are massively parallel devices that can execute thousands of threads at the same time.
But what does it mean in practice for the user? Why anyone would need to use a GPU to compute something that can be easily computed on a CPU?
We can try to answer this question with an example.

Suppose we want to sort a large array in Python.
Using NumPy, we first need to create an array of random single precision floating point numbers.

~~~
import numpy as np
size = 4096 * 4096
input = np.random.random(size).astype(np.float32)
~~~
{: .language-python}

We then time the execution of the NumPy `sort()` function, to see how long sorting this array takes on the CPU.

~~~
%timeit -n 1 -r 1 output = np.sort(input)
~~~
{: .language-python}

While the timing of this operation will differ depending on the system on which you run the code, these are the results for one experiment running on a Jupyter notebook on Google Colab.

~~~
1.83 s ± 0 ns per loop (mean ± std. dev. of 1 run, 1 loop each)
~~~
{: .output}

We now perform the same sorting operation, but this time we will be using CuPy to execute the `sort()` function on the GPU.
CuPy is an open-source library, compatible with NumPy, for GPU computing in Python.

~~~
from cupyx.profiler import benchmark
import cupy as cp
input_gpu = cp.asarray(input)
execution_gpu = benchmark(cp.sort, (input_gpu,), n_repeat=10)
gpu_avg_time = np.average(execution_gpu.gpu_times)
print(f"{gpu_avg_time:.6f} s")
~~~
{: .language-python}

We also report the output, obtained on the same notebook on Google Colab; as always note that your result will vary based on the environment and GPU you are using.

~~~
0.008949 s
~~~
{: .output}

Notice that the first thing we need to do, is to copy the input data to the GPU. The distinction between data on the GPU and that on the *host* is a very important one that we will get back to later.

> ## Host vs. Device
> From now on we may also call the GPU the *device*, while we refer to other computations as taking place on the *host*. We'll also talk about *host memory* and *device memory*, but much more on memory in later episodes!
{: .callout}


Sorting an array using CuPy, and therefore using the GPU, is clearly much faster than using NumPy, but can we quantify how much faster?
Having recorded the average execution time of both operations, we can then compute the speedup of using CuPy over NumPy.
The speedup is defined as the ratio between the sequential (NumPy in our case) and parallel (CuPy in our case) execution times; beware that both execution times need to be in the same unit, this is why we had to convert the GPU execution time from milliseconds to seconds.

~~~
speedup = 1.83 / 0.008949
print(speedup)
~~~
{: .language-python}

With the result of the previous operation being the following.

~~~
204.49212202480723
~~~
{: .output}

We can therefore say that just by using the GPU with CuPy to sort an array of size `4096 * 4096` we achieved a performance improvement of 204 times.

{% include links.md %}
