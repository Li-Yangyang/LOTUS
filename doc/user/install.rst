.. _install:

Installation
============

.. note:: ``LOTUS`` requires Python 3.7.*

This code uses ``Theano`` as backend tool to realize MCMC sampling via
``PyMC3`` but ``Theano`` has been deprecated after version 1.0.5 and has been
transferred into ``Aesara`` for ``PyMC3>=3.10.0``. In our future version, we will
update this code with up-to-date ``PyMC3`` but now we choose to use previous version,
which need to freeze ``python==3.7``. Therefore before installing this code,
it is recommended that create an independent environment via anaconda:

.. code-block:: bash

      conda create -n lotus python=3.7

and activate it:

.. code-block:: bash

      conda activate lotus


Using Pip
---------

The recommended method of installing *LOTUS* is with `pip
<https://pip.pypa.io>`_:

.. code-block:: bash

      python -m pip install lotus-nlte==0.1.1rc3

.. _source:

From Source
-----------

*LOTUS* can be downloaded and installed `from GitHub source code
<https://github.com/Li-Yangyang/LOTUS>`_ by running:

.. code-block:: bash

      git clone https://github.com/Li-Yangyang/LOTUS
      cd LOTUS
      python -m pip install -e .

The following dependencies are required to install it successfully:

- `numpy <https://numpy.org>`_>=1.16.4
- `pandas <https://pandas.pydata.org/>`_ ==0.24.2
- `scipy <https://scipy.org/>`_>=1.5.0
- `scikit-learn <https://scikit-learn.org/stable/>`_ ==0.23.2
- `sympy <https://www.sympy.org/en/index.html>`_>=1.6.2
- `pymc3 <https://docs.pymc.io>`_ ==3.7
- `theano <https://pypi.org/project/Theano/1.0.4/>`_ ==1.0.4
- `astropy <https://www.astropy.org/>`_>=3.2.1
- `h5py <https://www.h5py.org/>`_>=2.10.0
- `joblib <https://joblib.readthedocs.io/en/latest/>`_ ==1.0.1
- `numdifftools <https://github.com/pbrod/numdifftools>`_ ==0.9.39
- `matplotlib <https://matplotlib.org/>`_>=3.1.3 (for plotting)
- `tqdm <https://tqdm.github.io/>`_
- `corner <https://corner.readthedocs.io/en/latest/>`_
- `requests <https://docs.python-requests.org/en/latest/>`_

The rest will be installed as well but are used for future more ways of
interpolation:

- `torch <https://pytorch.org/>`_
- `gpytorch <https://gpytorch.ai/>`_
- `rbf <https://pypi.org/project/rbf/>`_

You can install them by:

.. code-block:: bash

      python -m pip install "lotus-nlte[advanced-interp]"
