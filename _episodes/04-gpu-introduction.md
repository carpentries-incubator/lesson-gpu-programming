---
title: "Another Look at the GPU"
teaching: 10
exercises: 0
questions:
- "How does a GPU work?"
objectives:
- "Understand how the GPU is organized."
- "Understand the building blocks of the CUDA programming model."
keypoints:
- ""
---

# Threads, blocks and grids

On Day 1, computations were done on a GPU using the CuPy and Numpy interfaces, i.e. from a high abstraction level. How the compute power of the GPU was applied remained 'under the hood', with one exception, where we used a single GPU thread to perform a computation. On Day 2 it will be shown that we have full control on how are computations are distributed over the GPU. The concepts that define that control are shown in this graph. It is meant to sink in overnight, but we will not discuss all of its aspects now. That will be done on the second day.

![Threads, blocks and grids](../fig/SlideDeck-PRACE_December_2020_slide_25_gedraaid.png)

{% include links.md %}
