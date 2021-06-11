---
title: "Introduction"
teaching: 20
exercises: 0
questions:
- "What is a Graphics Processing Unit?"
- "Can a GPU be used for anything else than graphics?"
- "Are GPUs useful for the development of scientific software?"
objectives:
- "Learn how a GPU works"
- "Understand the differences between CPU and GPU"
keypoints:
- ""
---

# Graphics Processing Unit

The Graphics Processing Unit (**GPU**) is one of the components of a computer's video card, together with specialized memory and different Input/Output (I/O) units.
In the context of the video card, the GPU fulfills a role similar to the one that the Central Processing Unit (CPU) has in a general purpose computing system: it processes input data data to generate some kind of output.
In the traditional context of video cards, GPUs process data in order to render images on an output device, such as a screen or monitor.
However, current GPUs are general purpose computing devices able to perform any kind of computation.

# Parallel by Design

But what is the reason to use GPUs to perform general purpose computation, when computers already have fast CPUs that are able to perform any kind of computation?
One way to answer this question is to go back to the roots of what a GPU is designed to efficiently do.

An image can be seen as a matrix of points called **pixels** (a portmanteau of the words *picture* and *element*), with each pixel representing the color the image should have in that particular point, and the traditional task performed by video cards is to produce the images a user will see on the screen.
So GPUs are designed with this particular task in mind: render multiple pixels at the same time.

A single 4K UHD image contains more than 8 million pixels
If a GPU has to generate a fluid stream of 25 4K frames (images) per second, enough for a user to not experience delays in a videogame, movie, or any other video output, it needs to process over 200 million pixels per second.
So GPUs are not only designed to render multiple pixels at the same time, they are designed to do it efficiently.

This design principle results in the GPU being, from a hardware point of view, a very different device than a CPU.
The CPU is a very general purpose device, good at different tasks, being them parallel or sequential in nature; it is also designed for interaction with the user, so it has to be responsive and guarantee minimal latency.
The result is a device where most of the silicon is used for memory caches and control-flow logic, not just compute units.
By contrast, most of the silicon on a GPU is actually used for compute units.

The GPU does not need an overly complicated cache hierarchy, nor it does need complex control logic, because the overall goal is not to maximize the latency of any given thread, but to maximize the throughput of the whole computation.
With many compute units available, the GPU can run massively parallel programs, programs in which thousands of threads are executed at the same time, while thousands more are ready for execution to hide the latency of memory operations.

# Speed Benefits

So, GPUs are massively parallel devices that can execute thousands of threads at the same time, providing large performance benefits for their users, but what does this mean in practice?

We begin with an example: sorting a large array in Python.
First we need to create an array of random single precision floating point numbers.

~~~
import numpy as np
size = 4096 * 4096
input = np.random.random(size)
~~~
{: .language-python}

We then time the execution of the Numpy `sort()` function, to see how long sorting this array takes on the CPU.

~~~
%timeit output = np.sort(input)
~~~
{: .language-python}

While the timing of this operation will differ depending on the system on which you run the code, these are the results for one experiment running on a Jupyter Notebook on Google Colab.

~~~
1 loop, best of 5: 1.84 s per loop
~~~
{: .output}

We now perform the same sorting operation, but this time we will be using Cupy to execute the `sort()` on the GPU.

~~~
import cupy as cp
input_gpu = cp.asarray(input)
%timeit output_gpu = cp.sort(input_gpu)
~~~
{: .language-python}

We also report the output, obtained on the same Notebook on Google Colab.

~~~
100 loops, best of 5: 6.83 ms per loop
~~~
{: .output}

And we finish by computing how much faster the GPU code is compared with the CPU one.
We do this by dividing the execution time of the sequential execution by the time of the parallel execution, both in seconds.

~~~
1.84 / 0.00683
~~~
{: .language-python}

With the result being the following.

~~~
269.39970717423137
~~~
{: .output}

Using the GPU to sort the array we obtained an increase in performance of 269 times.

{% include links.md %}