---
title: "CuPy"

teaching: 0

exercises: 0

questions:

- "How can I copy my data to the GPU?"
- "How can I do a calculation on a GPU?"
- "How can I copy the result back to my computer?"

objectives:
- "Be able to indicate if an array, represented by a variable in an iPython shell, is stored in host or device memory."
- "Be able to copy the contents of this array from host to device memory and vice versa."
- "Be able to select the appropriate function to either convolve an image using either CPU or GPU compute power."
- "Be able to quickly estimate the speed benefits for a simple calculation by moving it from the CPU to the GPU."

keypoints:
- ""
---

# CuPy - principles

[CuPy](https://docs.cupy.dev) is a GPU array backend that implements a subset of the NumPy interface, through Cuda, the GPU programming language designed by NVIDIA, the largest producer of GPUs in the world. This makes it a very convenient tool to use the compute power of GPUs for people that have some experience with Numpy.

# Generate some data and do a computation on the host (=CPU)

Start by generating some data on the host (i.e. your laptop, desktop or CPU cluster node) from an **iPython shell or a Jupyter notebook**. An artificial "image", for instance, will do. For this example, we will quickly generate a large image from scratch. The image will be all zeros, except for isolated pixels with value one, on a regular grid. We will convolve it with a Gaussian and inspect the result. We will also record the time to do this convolution on the CPU. Enter these commands in either an iPython shell or a cell in a Jupyter notebook running or your computer or a notebook from Google Colab.

~~~python
import numpy as np
#Construct a subimage with all zeros and one in the middle.
primary_unit = np.zeros((16,16))
primary_unit[8,8]=1
#Now duplicate this subimage many times to construct a large image.
deltas = np.tile(primary_unit, (128, 128))
print(deltas.shape)
~~~
{: .source}

This should show that you have indeed built a large image.
~~~
Out[7]: (2048, 2048)
~~~
{: .output}

To get some feeling for what you have just constucted, display a corner of this large image.

~~~python
import pylab as pyl
#Display the image and you can seize the opportunity to zoom in using the menu in the window that will appear.
pyl.imshow(deltas[0:32, 0;32])
pyl.show()
~~~
{: .source}

You should see four times primary_unit concatenated.

The computation we want to perform is convolution (think of it as blurring), both on the host (CPU) and device (GPU) and compare the results and runtimes.
To do so, we need to convolve the input with some blurring a function. We will use a Gaussian, because it is very common. Let us construct the Gaussian and display it. Remember that at this point we are still doing everything in the normal way, no use of a GPU yet.

~~~python
x, y = np.meshgrid(np.linspace(-2,2,15), np.linspace(-2,2,15))
dst = np.sqrt(x*x+y*y)
sigma = 1
muu = 0.000
gauss = np.exp(-((dst-muu)**2/(2.0 * sigma**2)))
pyl.imshow(gauss)
pyl.show()
~~~
{: .source}

This should show you a symmetrical two-dimensional Gaussian. Now we are ready to do the convolution using the CPU of our machine. We do not have to write this convolution function ourselves, it is very conveniently provided by Scipy. Let us also record the time to perform this convolution and inspect the top left corner of the convolved image.

~~~python
from scipy.signal import convolve2d as convolve2d_cpu
convolved_image_using_CPU = convolve2d_cpu(deltas, gauss)
%timeit convolve2d_cpu(deltas, gauss)
pyl.imshow(convolved_image_using_CPU[0:32, 0;32])
pyl.show()
~~~
{: .source}

Obviously, the time to perform this convolution will depend very much on the power of your CPU, but I expect you to find it will take a couple of seconds.
~~~
1 loop, best of 5: 2.52 s per loop
~~~
{: .output}

When you display corner of the image, you can see that the "ones" surrounded by zeros have actually been blurred by a Gaussian, so we end up with a regular grid of Gaussians.

# Copy the input image and convolving function to the GPU and convolve using the power of the GPU

This is part of a course on GPU programming, so let's use the GPU. Although there is a physical connection - i.e. a cable - between the CPU and the GPU, they do not share the same memory space. This means that an array created from e.g. an iPython shell using Numpy is physically loaded into the RAM of the CPU. It is not yet present in GPU memory, which means that we need to copy our data, the input image and the convolving function to the GPU first. We have "deltas" and "gauss" in the host's RAM now and we need to copy it to GPU memory using the CuPy library.

~~~python
import cupy as cp
deltas_gpu = cp.asarray(deltas)
gauss_gpu = cp.asarray(gauss)
~~~
{: .source}

Now it is time to do the convolution on the GPU. Scipy does not provide for functions that use the GPU, so we need to import the convolution function from another library, called "cupyx". cupyx.scipy contains a subset of all Scipy routines. You will see that the GPU convolution function from the "cupyx" library looks very much like the convolution function from Scipy we used previously. In general, Numpy and CuPy look very similar as well as the Scipy and "cupyx" libraries, as intended by the authors of those two libraries. Let us again record the time to do the job, so we can compare with the time it took on the CPU to perform the convolution.

~~~python
from cupyx.scipy.signal import convolve2d as convolve2d_gpu
convolved_image_using_GPU = convolve2d_gpu(deltas_gpu, gauss_gpu)
%timeit convolve2d_gpu(deltas_gpu, gauss_gpu)
~~~
{: .source}

Similar to above, the time to perform this convolution will depend very much on the power of your GPU, but I expect you to find it will take about 10ms. This is what I got on a TITAN X (Pascal) GPU:
~~~
1000 loops, best of 5: 20.2 ms per loop
~~~
{: .output}

This is a lot faster than on the CPU, I found a speedup factor of 125. Impressive!

> ## Challenge: Try to convolve the Numpy array deltas with the Numpy array gauss directly on the GPU, so without using CuPy arrays.
> If we succeed, this should save us the time and effort of transferring deltas and gauss to the GPU.
>
> > ## Solution
> > We can again use the GPU convolution function from the cupyx library: convolve2d_gpu and use deltas and gauss as input.
> > ~~~python
> > convolve2d_gpu(deltas, gauss)
> > ~~~
> > 
> > However, this gives a long error message with this last line:
> > ~~~
> > TypeError: Unsupported type <class 'numpy.ndarray'>
> > ~~~
> > {: .output}
> >
> > It is unfortunately not possible to access Numpy arrays from the GPU directly. Numpy arrays exist in the 
> > Random Access Memory (RAM) of the host and not in GPU memory. These types of memory are not united, but transfers are possible.
> >
> {: .solution}
{: .challenge}

# Compare the results. Copy the convolved image from the device back to the host

Let us first check that we have the exact same output. 

~~~python
np.allclose(convolved_image_using_GPU, convolved_image_using_CPU)
~~~
{: .source}

If this returns
~~~
array(True)
~~~
{: .output}

we have the same output from our convolution on the CPU and the GPU and we should be satisfied.

> ## Challenge: Compute the CPU vs GPU speedup while taking into account the transfers of data to the GPU and back.
> You should now find a lower speedup from taking the overhead of the transfer of arrays into account.
> Hint: To copy a CuPy array back to the host (CPU), use cp.asnumpy().
>
> > ## Solution
> > For timing, it is most convenient to define a function that completes all the steps.
> > ~~~python
> > def transfer_compute_transferback():
> >     deltas_gpu = cp.asarray(deltas)
> >     gauss_gpu = cp.asarray(gauss)
> >     convolved_image_using_GPU = convolve2d_gpu(deltas_gpu, gauss_gpu)
> >     convolved_image_using_GPU_copied_to_host = cp.asnumpy(convolved_image_using_GPU)
> >    
> > %timeit transfer_compute_transferback()
> > ~~~
> > ~~~
> > 10 loops, best of 5: 35.1 ms per loop
> > ~~~
> > {: .output}
> >
> > This means that our speedup has decreased from 2520 ms/20.2 ms = 125 to 2520 ms/35.1 ms = 72. This is still a significant 
> > speedup of our computations and adequately takes account of additional data transfers.
> {: .solution}
{: .challenge}

# A shortcut: performing Numpy routines on the GPU.

We saw above that we cannot execute routines from the "cupyx" library directly on Numpy arrays. First, a transfer of data needs to take place, from the host to the device (GPU). Vice versa, if we try to execute a regular Scipy routine (i.e. designed for the CPU) on a CuPy array, we will also encounter an error. Try the following:

~~~python
convolve2d_cpu(deltas_gpu, gauss_gpu)
~~~
{: .source}

This results in 
~~~
......
......
......
TypeError: Implicit conversion to a NumPy array is not allowed. Please use `.get()` to construct a NumPy array explicitly.
~~~
{: .output}

So Scipy routines cannot have CuPy arrays as input. We can, however, execute a simpler command that does not require Scipy. Instead of 2D convolution, we can do 1D convolution. For that we can use a Numpy routine instead of a Scipy routine. The "convolve" routine from Numpy performs linear (1D) convolution. To generate some input for a linear convolution, we can flatten our image from 2D to 1D (using ravel()), but we also need a 1D kernel. For the latter we will take the diagonal elements of our 2D Gaussian kernel. Try the following three instructions for linear convolution on the CPU:

~~~python
deltas_1d = deltas.ravel()
gauss_1d = gauss.diagonal()
%timeit np.convolve(deltas_1d, gauss_1d)
~~~
{: .source}

You could arrive at something similar to this timing result:
~~~
104 ms ± 32.9 µs per loop (mean ± std. dev. of 7 runs, 10 loops each)
~~~
{: .output}

We have performed a regular linear convolution using our CPU. Now let us try something bold. We will transfer the 1D arrays to the GPU and use the Numpy (!) routine to do the convolution. Again, we have to issue three commands:

~~~python
deltas_1d_gpu = cp.asarray(deltas_1d)
gauss_1d_gpu = cp.asarray(gauss_1d)
%timeit np.convolve(deltas_1d_gpu, gauss_1d_gpu)
~~~
{: .source}

You may be surprised that we can issue these commands without error. Contrary to Scipy routines, Numpy accepts CuPy arrays, i.e. arrays that exist in GPU memory, as input. [Here](https://docs.cupy.dev/en/v8.2.0/reference/interoperability.html#numpy) you can find some background on why Numpy routines can handle CuPy arrays. 

Also, remember the "np.allclose" command above? With a Numpy and a CuPy array as input. That worked for the same reason.

The linear convolution is actually performed on the GPU, which is shown by a nice speedup:

~~~
731 µs ± 106 µs per loop (mean ± std. dev. of 7 runs, 1 loop each)
~~~
{: .output}

So this implies a speedup of a factor 104/0.731 = 142. Impressive.

# Using Numba to execute Python code on the GPU

[Numba](http://numba.pydata.org/) is a Python library that "translates Python functions to optimized machine code at runtime using the industry-standard LLVM compiler library". You might want to try it to speed up your code on a CPU. However, Numba also [compiles a subset of Python code into CUDA kernels](https://numba.pydata.org/numba-doc/latest/cuda/overview.html) which is what we will use here. So the idea is that we can do what we are used to, i.e. write Python code and still benefit from the speed that GPUs offer us.

The code we want to run is straightforward. Let us compute all prime numbers between 1 and 10000 on the CPU and see if we can speed it up.
This is code that you can find on many websites. Small variations are possible, but it will look something like this:

~~~python
def find_all_primes_cpu(upper):
    all_prime_numbers=[]
    for num in range(2, upper):
        # all prime numbers are greater than 1
        for i in range(2, num):
            if (num % i) == 0:
                break
        else:
            all_prime_numbers.append(num)
    return all_prime_numbers
~~~
{: .source}

Calling "find_all_primes_cpu(10000)" will return all prime numbers between 1 and 10000 as a list. Let us time it:

~~~python
%timeit find_all_primes_cpu(10000) 
~~~
{: .source}

You will probably find that "find_all_primes_cpu" takes several hundreds of milliseconds to complete:

~~~
378 ms ± 45.6 ms per loop (mean ± std. dev. of 7 runs, 1 loop each)
~~~
{: .output}

As a quick sidestep, add Numba's JIT (Just in Time compilation) decorator to the "find_all_primes_cpu" function. You can either add it to the function definition or to the call, so either in this way:

~~~python
@jit(nopython=True)
def find_all_primes_cpu(upper):
    all_prime_numbers=[]
    ....
    ....
~~~
{: .source}

or in this way:

~~~python
%timeit jit(nopython=True)(find_all_primes_cpu)(upper_limit)
~~~
{: .source}

which can give you a timing result similar to this:

~~~
165 ms ± 19 ms per loop (mean ± std. dev. of 7 runs, 1 loop each)
~~~
{: .output}

So twice as fast, by using a simple decorator. The speedup is much larger for upper = 100000, but that takes a little too much waiting time for this course. Despite the "jit(nopython=True)" decorator the computation is still performed on the CPU. Let us move the computation to the GPU. There are a number of ways to achieve this, one of them is the usage of the "jit(device=True)" decorator, but it depends very much on the nature of the computation. Let us write our first GPU kernel which checks if a number is a prime, using the cuda.jit decorator, so different from the jit decorator for CPU computations. It is essentially the inner loop of "find_all_primes_cpu":

~~~python
from numba import cuda

@cuda.jit
def check_prime_gpu_kernel(num, result):
   # all prime numbers are greater than 1
   result[0] =  0
   for i in range(2, num):
       if (num % i) == 0:
           break
   else:
       result[0] = num
~~~
{: .source}

A number of things are worth noting. CUDA kernels do not return anything, so you have to supply for an array to be modified. All arguments have to be arrays, if you work with scalars, make them arrays of length one. This is the case here, because we check if a single number is a prime or not. Let us see if this works:

~~~python
result = np.zeros((1), np.int32)
check_prime_gpu_kernel[1, 1](11, result)
print(result[0])
check_prime_gpu_kernel[1, 1](12, result)
print(result[0])

~~~
{: .source}

This should return "11", because that is a prime and "0" because 12 is not a prime:

~~~
11
0
~~~
{: .output}

Note the extra arguments in square brackets - [1, 1] - that are added to the call of "check_prime_gpu_kernel". These indicate the number of "threads per block" and the number of "blocks per grid". These concepts will be explained in a later session. We will both set them to 1 for now.

> ## Challenge: Write a function find_all_primes_cpu_and_gpu that uses check_prime_gpu_kernel and the outer loop similar to find_all_primes_cpu. 
> # How long does it take to find all primes up to 10000?
>
> > ## Solution
> > ~~~python
> > @cuda.jit
> > def find_all_primes_cpu_and_gpu(upper):
> >     all_prime_numbers=[]
> >     for num in range(2, upper):
> >         result = np.zeros((1), np.int32)
> >         # Calculate the number of thread blocks in the grid
> >         check_prime_gpu_kernel[1,1](num, result)
> >         if result[0]>0:
> >             all_prime_numbers.append(num)
> >     return all_prime_numbers
> >    
> > %timeit find_all_primes_cpu_and_gpu(10000)
> > ~~~
> > ~~~
> > 6.62 s ± 152 ms per loop (mean ± std. dev. of 7 runs, 1 loop each)
> > ~~~
> > {: .output}
> >
> > Wow, that is slow! So much slower than find_all_primes_cpu. Clearly, we have not given the GPU enough work to do, the overhead is a lot larger than the 
> > workload.
> {: .solution}
{: .challenge}


Let us give the GPU a work load large enough to compensate for the overhead of data transfers to and from the GPU. For this example of computing primes we can best use the "vectorize" decorator for a new "check_prime_gpu" function that takes an array as input instead of "upper" in order to increase the work load. This is the array we have to use as input for our new "check_prime_gpu" function, instead of upper, a single integer:

~~~python
np.arange(2, 10000, dtype=np.int32)
~~~
{: .source}

So that input to the new "check_prime_gpu" function is simply the array of numbers we need to check for primes. "check_prime_gpu" looks similar to "check_prime_gpu_kernel", but it is not a kernel, so it can return values:

~~~python
@nb.vectorize(['int32(int32)'], target='cuda')
def check_prime_gpu(num):
   for i in range(2, num):
       if (num % i) == 0:
           return 0
   else:
       return num
~~~
{: .source}

where we have added the "vectorize" decorator from Numba. The argument of "check_prime_gpu" seems to be defined as a scalar (single integer in this case), but the "vectorize" decorator will allow us to use an array as input. That array should consist of 4B (byte) of 32b (bit) integers, indicated by "(int32)". The return array will also consist of 32b integers, with zeros for the non-primes. The nonzero values are the primes. 

Let us run it and record the elapsed time:

~~~python
%timeit check_prime_gpu(np.arange(2, upper_limit, dtype=np.int32))
~~~
{: .source}

which should show you a significant speedup:

~~~
3.25 ms ± 138 µs per loop (mean ± std. dev. of 7 runs, 100 loops each)
~~~
{: .output}

This amounts to an accelleration of our code of a factor 165/3.25 = 50.8 compared to the "jit(nopython=True)" decorated code on the CPU.

{% include links.md %}

