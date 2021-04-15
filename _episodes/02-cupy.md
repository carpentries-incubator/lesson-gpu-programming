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

[CuPy](https://docs.cupy.dev) is a GPU array backend that implements a subset of NumPy interface, through Cuda, the GPU programming language designed by NVIDIA, the largest producer of GPUs in the world. This makes it a very convenient tool to use the compute power of GPUs for people that have some experience with Numpy.

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
from scipy.signal import convolve2d
convolved_image_using_CPU = convolve2d(deltas, gauss)
%timeit convolve2d(deltas, gauss)
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

Now it is time to do the convolution on the GPU. Scipy does not provide for functions that use the GPU, so we need to import the convolution function from another library, called "cupyx". cupyx.scipy contains a subset of all Scipy routines. You will see that the GPU convolution function from the "cupyx" library looks very much like the convolution function from Scipy we used previously. In general, Numpy and Cupy look very similar as well as the Scipy and "cupyx" libraries, as intended by the authors of those two libraries. Let us again record the time to do the job, so we can compare with the time it took on the CPU to perform the convolution.

~~~python
from cupyx.scipy import signal
convolved_image_using_GPU = signal.convolve2d(deltas_gpu, gauss_gpu)
%timeit signal.convolve2d(deltas_gpu, gauss_gpu)
~~~
{: .source}

Similar to above, the time to perform this convolution will depend very much on the power of your GPU, but I expect you to find it will take about 10ms. This is what I got on a TITAN X (Pascal) GPU:
~~~
1000 loops, best of 5: 20.2 ms per loop
~~~
{: .output}

This is a lot faster than on the CPU, I found a speedup factor of 125. Impressive!

# Compare the results. Copy the convolved image from the device back to the host

Let us first check that we have the exact same output. To do so, we first need to copy the output from the device (GPU) to the host (CPU):

~~~python
convolved_image_using_GPU_copied_to_host = cp.asnumpy(convolved_image_using_GPU)
np.allclose(convolved_image_using_GPU_copied_to_host, convolved_image_using_CPU)
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
>
> > ## Solution
> > For timing, it is most convenient to define a function that completes all the steps.
> > ~~~python
> > def transfer_compute_transferback():
> >     deltas_gpu = cp.asarray(deltas)
> >     gauss_gpu = cp.asarray(gauss)
> >     convolved_image_using_GPU = convolve2d(deltas_gpu, gauss_gpu)
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


{% include links.md %}

