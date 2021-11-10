---
title: "Accelerate your own Python code with Numba"

teaching: 30

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

We want to compute all prime numbers - i.e. numbers that have only 1 or itself as divisors without a remainder - between 1 and 10000 on the CPU and see if we can speed it up, by deploying a similar algorithm on a GPU. 
This is code that you can find on many websites. Small variations are possible, but it will look something like this:

~~~
def find_all_primes_cpu(upper):
    all_prime_numbers = []
    for num in range(2, upper):
        for i in range(2, (num // 2) + 1):
            if (num % i) == 0:
                break
        else:
            all_prime_numbers.append(num)
    return all_prime_numbers
~~~
{: .language-python}

Calling `find_all_primes_cpu(10000)` will return all prime numbers between 1 and 10000 as a list. Let us time it:

~~~
%timeit find_all_primes_cpu(10000)
~~~
{: .language-python}

You will probably find that `find_all_primes_cpu` takes several hundreds of milliseconds to complete:

~~~
378 ms ± 45.6 ms per loop (mean ± std. dev. of 7 runs, 1 loop each)
~~~
{: .output}

As a quick sidestep, add Numba's JIT (Just in Time compilation) decorator to the `find_all_primes_cpu` function. You can either add it to the function definition or to the call, so either in this way:

~~~
from numba import jit

@jit(nopython=True)
def find_all_primes_cpu(upper):
    all_prime_numbers = []
    for num in range(2, upper):
        for i in range(2, (num // 2) + 1):
            if (num % i) == 0:
                break
        else:
            all_prime_numbers.append(num)
    return all_prime_numbers
~~~
{: .language-python}

or in this way:

~~~
from numba import jit

%timeit jit(nopython=True)(find_all_primes_cpu)(upper_limit)
~~~
{: .language-python}

which can give you a timing result similar to this:

~~~
165 ms ± 19 ms per loop (mean ± std. dev. of 7 runs, 1 loop each)
~~~
{: .output}

So twice as fast, by using a simple decorator. The speedup is much larger for `upper = 100000`, but that takes a little too much waiting time for this course.
Despite the `jit(nopython=True)` decorator the computation is still performed on the CPU.
Let us move the computation to the GPU.
There are a number of ways to achieve this, one of them is the usage of the `jit(device=True)` decorator, but it depends very much on the nature of the computation.
Let us write our first GPU kernel which checks if a number is a prime, using the `cuda.jit` decorator, so different from the `jit` decorator for CPU computations.
It is essentially the inner loop of `find_all_primes_cpu`:

~~~
from numba import cuda

@cuda.jit
def check_prime_gpu_kernel(num, result):
   result[0] =  0
   for i in range(2, (num // 2) + 1):
       if (num % i) == 0:
           break
   else:
       result[0] = num
~~~
{: .language-python}

A number of things are worth noting. CUDA kernels do not return anything, so you have to supply for an array to be modified. All arguments have to be arrays, if you work with scalars, make them arrays of length one. This is the case here, because we check if a single number is a prime or not. Let us see if this works:

~~~
result = np.zeros((1), np.int32)
check_prime_gpu_kernel[1, 1](11, result)
print(result[0])
check_prime_gpu_kernel[1, 1](12, result)
print(result[0])
~~~
{: .language-python}

This should return "11", because that is a prime and "0" because 12 is not a prime:

~~~
11
0
~~~
{: .output}

Note the extra arguments in square brackets - `[1, 1]` - that are added to the call of `check_prime_gpu_kernel`. These indicate the number of "threads per block" and the number of "blocks per grid". These concepts will be explained in a later session. We will both set them to 1 for now.

> ## Challenge: compute prime numbers
> Write a function `find_all_primes_cpu_and_gpu` that uses `check_prime_gpu_kernel` and the outer loop similar to `find_all_primes_cpu`.
> How long does it take to find all primes up to 10000?
>
> > ## Solution
> > ~~~
> > def find_all_primes_cpu_and_gpu(upper):
> >     all_prime_numbers = []
> >     for num in range(2, upper):
> >         result = np.zeros((1), np.int32)
> >         check_prime_gpu_kernel[1,1](num, result)
> >         if result[0] > 0:
> >             all_prime_numbers.append(num)
> >     return all_prime_numbers
> >    
> > %timeit find_all_primes_cpu_and_gpu(10000)
> > ~~~
> > {: .language-python}
> > ~~~
> > 6.62 s ± 152 ms per loop (mean ± std. dev. of 7 runs, 1 loop each)
> > ~~~
> > {: .output}
> >
> > Wow, that is slow! So much slower than find_all_primes_cpu. Clearly, we have not given the GPU enough work to do, the overhead is a lot larger than the 
> > workload.
> {: .solution}
{: .challenge}


Let us give the GPU a work load large enough to compensate for the overhead of data transfers to and from the GPU. For this example of computing primes we can best use the `vectorize` decorator for a new `check_prime_gpu` function that takes an array as input instead of `upper` in order to increase the work load. This is the array we have to use as input for our new `check_prime_gpu` function, instead of upper, a single integer:

~~~
np.arange(2, 10000, dtype=np.int32)
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
   else:
       return num
~~~
{: .language-python}

where we have added the `vectorize` decorator from Numba. The argument of `check_prime_gpu` seems to be defined as a scalar (single integer in this case), but the `vectorize` decorator will allow us to use an array as input. That array should consist of 4B (bytes) or 32b (bit) integers, indicated by `(int32)`. The return array will also consist of 32b integers, with zeros for the non-primes. The nonzero values are the primes. 

Let us run it and record the elapsed time:

~~~
%timeit check_prime_gpu(np.arange(2, 10000, dtype=np.int32))
~~~
{: .language-python}

which should show you a significant speedup:

~~~
3.25 ms ± 138 µs per loop (mean ± std. dev. of 7 runs, 100 loops each)
~~~
{: .output}

This amounts to an acceleration of our code of a factor 165/3.25 = 50.8 compared to the `jit(nopython=True)` decorated code on the CPU.

{% include links.md %}
