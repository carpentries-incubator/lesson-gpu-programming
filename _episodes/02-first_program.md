---
title: "Your First GPU Program"
teaching: 0
exercises: 0
questions:
- "How can I parallelize a Python application on GPUs?"
- "How to write a GPU program?"
- "What is CUDA?"
objectives:
- "Recognize possible data parallelism in Python code"
- "Understand the structure of a CUDA program"
- "Execute a CUDA program from Python"
keypoints:
- ""
---

# Summing Two Vectors

We start by introducing a program that, given two input vectors of the same size, returns a third vector containing the sum of the corresponding elements of the two input vectors.

~~~python
def vector_add(A, B, C, size):
    for item in range(0, size):
        C[item] = A[item] + B[item]
    
    return C
~~~

One of the characteristics of this program is that each iteration of the *for* loop is independent from the other iterations.
In other words, we could not only reorder the iterations and still produce the same output, but also compute part of the iterations on one device and part of the iterations on another device, and still end up with the same result.

{% include links.md %}

