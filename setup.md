---
title: Setup
---

# Programming environment

The GPU programming lesson can be taught using [Jupyter Notebook](https://jupyter.org/), a programming environment that runs in a web browser.
For this to work you we need a reasonably up-to-date browser.
The current versions of the Chrome, Safari and Firefox browsers are all [supported](https://jupyter-notebook.readthedocs.io/en/stable/notebook.html#browser-compatibility) (some older browsers, including Internet Explorer version 9 and below, are not).

In case you do not have any GPU available on your laptop, a good alternative is to use [Google Colab](https://colab.research.google.com).

## Local setup

To setup locally, depending on how you installed, there are two alternatives:
- you installed Python normally using your OS's package manager or app store,
- you installed the conda distribution of Python.

In case you don't have Python installed, we recommend you start with a variant of the conda distribution: [mambaforge](https://mamba.readthedocs.io/en/latest/installation.html).  `mambaforge` by default sets the `conda-forge` channel as the default, and provides the alternative package manager `mamba`.  `mamba` is a lot more performant compared to `conda`.

Whichever case it is for you, the first step is to create an isolated environment for the workshop, this way you won't interfere with your existing setup.  You can install all the dependencies for the workshop within this environment.  In the Python ecosystem, this kind of isolated environments are known as *virtual environments*. 

### Using `pip`

To create a virtual environment using `pip`, you need to install `virtualenv` package using your OS's package manager (it may have alternate names like `python-virtualenv` or `python3-virtualenv`).  After you have this setup, you can follow the steps below:
~~~bash
python -m virtualenv --prompt gpu-workshop venv
source venv/bin/activate
pip install -U pip  # it is good to update pip to the latest version
pip install cupy-cuda11x numba jupyterlab
~~~

### Using `conda` or `mamba`

`conda` or `mamba` have support for virtual environments built-in.  You can create a new virtual environment with
~~~bash
mamba create -n gpu-workshop
mamba activate gpu-workshop
mamba install cupy numba jupyterlab
~~~
If you are using `conda`, you can simply replace `mamba` with `conda` in the commands above.

### Starting a Jupyter server

Now you can start your Jupyter server as shown below, which will open a tab with Jupyter in your default browser:
~~~bash
jupyter-lab
~~~

If you do not want Jupyter to open a tab in your browser automatically, you can use the alternative below:
~~~bash
jupyter-lab --no-browser
~~~

This will print out a url in your terminal, which you can then open in the browser of your choice.

{% include links.md %}
