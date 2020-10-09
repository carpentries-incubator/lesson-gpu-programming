---
title: "Introduction"
teaching: 0
exercises: 0
questions:
- "What is a Graphics Processing Unit?"
- "Can a GPU be used for anything else than graphics?"
- "Are GPUs useful for scientific research?"
objectives:
- "Learn how a GPU works"
- "Understand the differences between a CPU and a GPU"
keypoints:
- ""
---

# Graphics Processing Unit

A Graphics Processing Unit (**GPU**) is one of the components of a computer's video card, together with specialized memory and different Input/Output (I/O) units.
The role of the GPU in the context of the video card is similar to the role that the Central Processing Unit (CPU) has in a general computing system: the GPU processes data from memory to generate some output values.
While in the context of graphics the most common form of output for a GPU is images, modern GPUs are general computing devices capable of performing general computations.

# Parallel by Design

An image can also be seen as a matrix of points called **pixels** (a portmanteau of the words *picture* and *element*), each representing the color the image has in that point.
If we think at 4K UHD, a single image contains more than 8 million pixels.
For a GPU to generate a continuous stream of 25 4K frames (images) per second, it needs to process 200 million pixels per second.

# General Purpose Programming on GPUs

SIMT processing.
Programming models.

# GPUs and Supercomputers

In 2020 GPUs are in most supercomputers.

{% include links.md %}

