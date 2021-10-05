---
title: "Closure Day 1"
teaching: 10
exercises: 0
questions:
- "How do I organise the computations on my GPU in an efficient manner?"
objectives:
- "Understand the building blocks of the CUDA programming model, i.e. threads, blocks and grids"
keypoints:
- "There is a large amount of freedom in distributing your computations over the GPU, but a lot of configurations will render your GPU mostly idle."
---

# Threads, blocks and grids

On Day 1, computations were done on a GPU using the CuPy and Numpy interfaces, i.e. from a high abstraction level. How the compute power of the GPU was applied remained 'under the hood', with one exception, where we used a single GPU thread to perform a computation. On Day 2 it will be shown that we have full control on how are computations are distributed over the GPU. The concepts that define that control are shown in this graph. It is meant to sink in overnight, but we will not discuss all of its aspects now. That will be done on the second day.

![Threads, blocks and grids](./SlideDeck-PRACE_December_2020_slide_25_gedraaid.png)
{% include links.md %}
