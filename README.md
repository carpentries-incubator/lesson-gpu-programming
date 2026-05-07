# GPU Programming

A [Carpentries Incubator](https://github.com/carpentries-incubator) lesson on programming Graphics Processing Units (GPUs) in Python.

The rendered lesson is available at: https://carpentries-incubator.github.io/lesson-gpu-programming/

## Lesson Content

The lesson covers the following topics:

- **Introduction**: overview of GPU architecture and when to use a GPU
- **CuPy**: GPU-accelerated NumPy-like arrays
- **PyTorch**: tensor operations and `@torch.compile` for general-purpose GPU computing
- **Numba**: JIT-compiling Python functions for the GPU
- **CUDA fundamentals**: writing and launching GPU kernels using CUDA and Python
- **Memory hierarchy**: global, local, shared, and constant memory
- **Streams and events**: concurrent kernel execution and fine-grained synchronization

## Prerequisites

Learners are expected to have:

- Basic Python programming skills
- Familiarity with NumPy arrays

No prior GPU or parallel programming experience is required.

## Teaching this lesson?

This material is open-source and freely available. If you are planning to use it in your teaching, please get in touch at [training@esciencecenter.nl](mailto:training@esciencecenter.nl). We would love to help you prepare and receive feedback based on your experience.

## Contributing

Contributions are welcome! Please read [CONTRIBUTING.md](CONTRIBUTING.md) before opening a pull request or issue.

## Current maintainer(s)

- Alessio Sclocco [@isazi](https://github.com/isazi)

