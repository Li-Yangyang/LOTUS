#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 30 19:41:37 2020

@author: yangyangli
"""
import numpy as np
import pymc3 as pm
#import pymc3_ext as pmx
from .theano_op.likelihood import LogLikeWithGrad
from .theano_op.predict import GenerateMets
import theano.tensor as tt

def slicesampling(log_likelihood, mgcog, priors, priors_stderr, bounds, ndraws=1000, ntunes=200, chains=4):
    """
    Slice sampling wrapper

    Parameters
    ----------
    log_likelihood : lotus_nlte.optimize.StellarOptimization.log_likelihood
        calculate log likelihood function
    mgcog : lotus_nlte.gcogs.MultiGCOG
        Instance of MultiGCOG
    priors : list or ndarray
        Priors of Teff, logg, vt
    priors_stderr : list or ndarray
        Estimated priors standard deviation
    bounds : list of tuple
        boundary for parameters
    ndraws : int, optional
        Number of draws. The default is 1000.
    ntunes : int, optional
        Number of tunning(burning). The default is 200.
    chains : int, optional
        Number of chains. The default is 4.

    Returns
    -------
    trace : MultiTrace
        Contains all samples up to the current iteration


    """
    logl = LogLikeWithGrad(log_likelihood)
    _generate_mets = GenerateMets(mgcog)
    #ndraws = 1000  # number of draws from the distribution
    #ntunes = 200   # number of "burn-in points" (which we'll discard)

    with pm.Model() as opmodel:
        # uniform priors on m and c
        if np.isnan(priors_stderr[0]) or (2*priors_stderr[0]>(bounds[0][1]-bounds[0][0])):
            priors_teff = 25
        else:
            priors_teff = priors_stderr[0]
        if np.isnan(priors_stderr[1]) or (2*priors_stderr[1]>(bounds[1][1]-bounds[1][0])):
            priors_logg = 0.05
        else:
            priors_logg = priors_stderr[1]
        if np.isnan(priors_stderr[2]) or (2*priors_stderr[2]>(bounds[2][1]-bounds[2][0])):
            priors_vt = 0.25
        else:
            priors_vt = priors_stderr[2]
        teff = pm.Bound(pm.Normal, lower=bounds[0][0], upper=bounds[0][1])('Teff', mu=priors[0], sigma=priors_teff)
        logg = pm.Bound(pm.Normal, lower=bounds[1][0], upper=bounds[1][1])('logg', mu=priors[1], sigma=priors_logg)
        vt = pm.Bound(pm.Normal, lower=bounds[2][0], upper=bounds[2][1])('vt', mu=priors[2], sigma=priors_vt)
        log_f = pm.Uniform('log_f' ,lower=-10.0, upper=1.0)

        # convert m and c to a tensor vector
        theta = tt.as_tensor_variable([teff, logg, vt, log_f])

        #feh
        pm.Deterministic("feh", _generate_mets(theta[:3]))

        # use a DensityDist
        pm.DensityDist('likelihood', lambda v: logl(v), observed={'v': theta})

        #start = pm.find_MAP(method="powell", vars=[teff, logg])
        #start = pm.find_MAP(start=start, vars=[teff, logg, vt, log_f])
        #start = pmx.optimize(start=start)
        #trace = pm.sample(ndraws, tune=nburn, discard_tuned_samples=True)

    with opmodel:
        step = pm.Slice()
        trace = pm.sample(ndraws, step=step, tune=ntunes, chains=chains)

    return trace
