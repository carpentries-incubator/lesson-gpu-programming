---
title: "Accelerate your Python code with Numba"

teaching: 45

exercises: 15

questions:
- "How can I run my own Python functions on the GPU?"

objectives:
- "Learn how to use Numba decorators to improve the performance of your Python code."
- "Run your first application on the GPU."

keypoints:
- "Numba can be used to run your own Python functions on the GPU."
- "Functions may need to be changed to run correctly on a GPU."
---

# Using Numba to execute Python code on the GPU

[Numba](http://numba.pydata.org/) is a Python library that "translates Python functions to optimized machine code at runtime using the industry-standard LLVM compiler library". You might want to try it to speed up your code on a CPU. However, Numba [can also translate a subset of the Python language into CUDA](https://numba.pydata.org/numba-doc/latest/cuda/overview.html), which is what we will be using here. So the idea is that we can do what we are used to, i.e. write Python code and still benefit from the speed that GPUs offer us.

We want to compute all [prime numbers](https://en.wikipedia.org/wiki/Prime_number) - i.e. numbers that have only 1 or themselves as exact divisors - between 1 and 10000 on the CPU and see if we can speed it up, by deploying a similar algorithm on a GPU. 
This is code that you can find on many websites. Small variations are possible, but it will look something like this:

~~~
def find_all_primes_cpu(upper):
    all_prime_numbers = []
    for num in range(0, upper):
        prime = True
        for i in range(2, (num // 2) + 1):
            if (num % i) == 0:
                prime = False
                break
        if prime:
            all_prime_numbers.append(num)
    return all_prime_numbers
~~~
{: .language-python}

Calling `find_all_primes_cpu(10_000)` will return all prime numbers between 1 and 10000 as a list. Let us time it:

~~~
%timeit -n 10 -r 1 find_all_primes_cpu(10_000)
~~~
{: .language-python}

You will probably find that `find_all_primes_cpu` takes several hundreds of milliseconds to complete:

~~~
176 ms ± 0 ns per loop (mean ± std. dev. of 1 run, 10 loops each)
~~~
{: .output}

As a quick sidestep, add Numba's JIT (Just in Time compilation) decorator to the `find_all_primes_cpu` function. You can either add it to the function definition or to the call, so either in this way:

~~~
from numba import jit

@jit(nopython=True)
def find_all_primes_cpu(upper):
    all_prime_numbers = []
    for num in range(0, upper):
        prime = True
        for i in range(2, (num // 2) + 1):
            if (num % i) == 0:
                prime = False
                break
        if prime:
            all_prime_numbers.append(num)
    return all_prime_numbers

%timeit -n 10 -r 1 find_all_primes_cpu(10_000)
~~~
{: .language-python}

or in this way:

~~~
from numba import jit

upper = 10_000
%timeit -n 10 -r 1 jit(nopython=True)(find_all_primes_cpu)(upper)
~~~
{: .language-python}

which can give you a timing result similar to this:

~~~
69.5 ms ± 0 ns per loop (mean ± std. dev. of 1 run, 10 loops each)
~~~
{: .output}

So twice as fast, by using a simple decorator. The speedup is much larger for `upper = 100_000`, but that takes a little too much waiting time for this course.
Despite the `jit(nopython=True)` decorator the computation is still performed on the CPU.
Let us move the computation to the GPU.
There are a number of ways to achieve this, one of them is the usage of the `jit(device=True)` decorator, but it depends very much on the nature of the computation.
Let us write our first GPU kernel which checks if a number is a prime, using the `cuda.jit` decorator, so different from the `jit` decorator for CPU computations.
It is essentially the inner loop of `find_all_primes_cpu`:

~~~
from numba import cuda

@cuda.jit
def check_prime_gpu_kernel(num, result):
   result[0] =  num
   for i in range(2, (num // 2) + 1):
       if (num % i) == 0:
           result[0] = 0
           break
~~~
{: .language-python}

A number of things are worth noting. CUDA kernels do not return anything, so you have to supply for an array to be modified. All arguments have to be arrays, if you work with scalars, make them arrays of length one. This is the case here, because we check if a single number is a prime or not. Let us see if this works:

~~~
import numpy as np

result = np.zeros((1), np.int32)
check_prime_gpu_kernel[1, 1](11, result)
print(result[0])
check_prime_gpu_kernel[1, 1](12, result)
print(result[0])
~~~
{: .language-python}

If we have not made any mistake, the first call should return "11", because 11 is a prime number, while the second call should return "0" because 12 is not a prime:

~~~
11
0
~~~
{: .output}

Note the extra arguments in square brackets - `[1, 1]` - that are added to the call of `check_prime_gpu_kernel`: these indicate the number of threads we want to run on the GPU.
While this is an important argument, we will explain it later and for now we can keep using `1`.

> ## Challenge: compute prime numbers
>
> Write a new function `find_all_primes_cpu_and_gpu` that uses `check_prime_gpu_kernel` instead of the inner loop of `find_all_primes_cpu`.
> How long does this new function take to find all primes up to 10000?
>
> > ## Solution
> >
> > One possible implementation of this function is the following one.
> >
> > ~~~
> > def find_all_primes_cpu_and_gpu(upper):
> >     all_prime_numbers = []
> >     for num in range(0, upper):
> >         result = np.zeros((1), np.int32)
> >         check_prime_gpu_kernel[1,1](num, result)
> >         if result[0] > 0:
> >             all_prime_numbers.append(num)
> >     return all_prime_numbers
> >    
> > %timeit -n 10 -r 1 find_all_primes_cpu_and_gpu(10_000)
> > ~~~
> > {: .language-python}
> > ~~~
> > 6.21 s ± 0 ns per loop (mean ± std. dev. of 1 run, 10 loops each)
> > ~~~
> > {: .output}
> >
> > As you may have noticed, `find_all_primes_cpu_and_gpu` is much slower than the original `find_all_primes_cpu`.
> > The reason is that the overhead of calling the GPU, and transferring data to and from it, for each number of the sequence is too large.
> > To be efficient the GPU needs enough work to keep all of its cores busy.
> {: .solution}
{: .challenge}


Let us give the GPU a work load large enough to compensate for the overhead of data transfers to and from the GPU. For this example of computing primes we can best use the `vectorize` decorator for a new `check_prime_gpu` function that takes an array as input instead of `upper` in order to increase the work load. This is the array we have to use as input for our new `check_prime_gpu` function, instead of upper, a single integer:

~~~
np.arange(0, 10_000, dtype=np.int32)
~~~
{: .language-python}

So that input to the new `check_prime_gpu` function is simply the array of numbers we need to check for primes. `check_prime_gpu` looks similar to `check_prime_gpu_kernel`, but it is not a kernel, so it can return values:

~~~
import numba as nb

@nb.vectorize(['int32(int32)'], target='cuda')
def check_prime_gpu(num):
    for i in range(2, (num // 2) + 1):
       if (num % i) == 0:
           return 0
    return num
~~~
{: .language-python}

where we have added the `vectorize` decorator from Numba. The argument of `check_prime_gpu` seems to be defined as a scalar (single integer in this case), but the `vectorize` decorator will allow us to use an array as input. That array should consist of 4B (bytes) or 32b (bit) integers, indicated by `(int32)`. The return array will also consist of 32b integers, with zeros for the non-primes. The nonzero values are the primes. 

Let us run it and record the elapsed time:

~~~
%timeit -n 10 -r 1 check_prime_gpu(numbers)
~~~
{: .language-python}

which should show you a significant speedup:

~~~
5.9 ms ± 0 ns per loop (mean ± std. dev. of 1 run, 10 loops each)
~~~
{: .output}

This amounts to a speedup of our code of a factor 11 compared to the `jit(nopython=True)` decorated code on the CPU.

{% include links.md %}
