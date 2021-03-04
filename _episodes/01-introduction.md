---
title: "Introduction"
teaching: 0
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

A Graphics Processing Unit (**GPU**) is one of the components of a computer's video card, together with specialized memory and different Input/Output (I/O) units.
The role of the GPU in the context of the video card is similar to the role that the Central Processing Unit (CPU) has in a general computing system: the GPU processes data from memory to generate some output values.
While in the context of graphics the most common form of output for a GPU is images, modern GPUs are general computing devices capable of performing general computations.

Development driven by computer graphics in entertainment and videogames, multi-billion consumer markets.

# Parallel by Design

An image can also be seen as a matrix of points called **pixels** (a portmanteau of the words *picture* and *element*), each representing the color the image has in that point.
If we think at 4K UHD, a single image contains more than 8 million pixels.
For a GPU to generate a continuous stream of 25 4K frames (images) per second, it needs to process 200 million pixels per second.

Structural comparison of CPU and GPU.
Show how the GPU is optimized for throughput while the CPU is optimized for latency.

# General Purpose Programming on GPUs

SIMT processing.

Programming models.

# GPUs and Supercomputers

Performance of CPUs and GPUs since 2010.

In June 2020 edition of the TOP500 list, 6 out of the top 10 supercomputers have GPUs.

{% include links.md %}

# Speed benefits

Let's start with a very simple example: sorting a long array. We will use the [Cupy](https://cupy.dev/) library to sort on a GPU and `Numpy` to sort on a CPU.
We will time the operations using the `timeit` command from the iPython shell.

~~~python
%import numpy as np
%length = 4096*4096
%rnd = np.random.random((length,))
%timeit sorted = np.sort(rnd)
~~~
{: .source}

Which returned on my CPU:
~~~
1.61 s ± 313 µs per loop (mean ± std. dev. of 7 runs, 1 loop each)
~~~
{: .output}

Okay, let's try to do the same calculation on a GPU.

~~~python
%import cupy as cp
%rnd_cp = cp.asarray(rnd)
%timeit sorted_cp = cp.ndarray.sort(rnd_cp)
~~~
{: .source}

Which returned,  on my GPU:
~~~
12.1 ms ± 1.19 µs per loop (mean ± std. dev. of 7 runs, 100 loops each)
~~~
{: .output}

So that is tremendous speedup of a factor 133, with very little work, provided you have a GPU at your disposal and the Cupy library installed.
