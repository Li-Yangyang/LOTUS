.. LOTUS documentation master file, created by
   sphinx-quickstart on Wed Nov 17 14:33:30 2021.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

LOTUS
=====
*LOTUS* (non-LTE Optimization Tool Utilized for the derivation of atmospheric
Stellar parameters) is a python package for the derivation of stellar parameters,
such :math:`T_{\mathrm{eff}}`, :math:`\mathrm{log\mathit g}`, :math:`\mathrm{[Fe/H]}`
and :math:`\xi_{t}` via *Equivalent Width (EW)* method with the assumption of
**1D Non Local Thermodynamic Equilibrium**. It mainly applies on the spectroscopic
data from high resolution spectral survey. It can provide extremely accurate
measurement of stellar parameters compared with non-spectroscipic analysis from
benchmark stars as: :math:`\Delta T_{\mathrm{eff}} \lesssim 50 K`, :math:`\Delta \mathrm{log\mathit g} \lesssim 0.1`.
*LOTUS* provides features:

* Fast optimizer for obtaining stellar parameters based on
  `Differential Evolution <https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.differential_evolution.html>`_ algorithm.
* Well constrained uncertainty of derived stellar parameters from slice-sampling MCMC
  from `PyMC3 <https://docs.pymc.io>`_.
* Interpolation of *Curve of Growth* from theoretical EW grid under the assumptions of
  LTE and Non-LTE.
* Visualization of excitation and ionization balance when at the optimal combination
  of stellar parameters.

*LOTUS* is being developed in `GitHub Repository <https://github.com/Li-Yangyang/LOTUS>`_,
so if you catch any issues during running it please `open an issue <https://github.com/Li-Yangyang/LOTUS/issues>`_ here.

Contents
--------

.. toctree::
   :maxdepth: 1
   :caption: User Guide

   user/install
   user/api

.. toctree::
   :maxdepth: 1
   :caption: Tutorials

   tutorials/quickstart.ipynb
   tutorials/examples.ipynb
