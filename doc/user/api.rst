.. _api:

API documentation
=================

GCOG
------

.. autoclass:: lotus_nlte.gcogs.SingleGCOG
   :inherited-members:

.. autoclass:: lotus_nlte.gcogs.MultiGCOG
   :inherited-members:
   
.. autoclass:: lotus_nlte.gcogs.PolyMultiGCOG
   :inherited-members:
   :noindex: lotus_nlte.interpolation.MultivariatePolynomialInterpolation
   
.. autofunction:: lotus_nlte.gcogs.utils.get_keys_and_atmos_pars
   
.. autofunction:: lotus_nlte.gcogs.utils.find_closest_model
   
Interpolation
-------------

.. autoclass:: lotus_nlte.interpolation.MultivariatePolynomialInterpolation
   :inherited-members:
   
Optimization
------------

.. autoclass:: lotus_nlte.optimize.StellarOptimization
   :inherited-members:

.. autoclass:: lotus_nlte.optimize.DiffEvoStellarOptimization
   :inherited-members:
   
.. autoclass:: lotus_nlte.optimize.ShgoStellarOptimization
   :inherited-members:
   
Sampling
--------
.. autofunction:: lotus_nlte.sampling.slicesampling
   
Plot
----

.. autofunction:: lotus_nlte.plot.plot_optimized_equilibrium

.. autofunction:: lotus_nlte.plot.plot_results_brute
