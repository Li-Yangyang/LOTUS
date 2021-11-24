#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 18 15:29:27 2021

@author: yangyangli
"""
import numpy as np
import pandas as pd
from sympy.abc import e
from sympy import Array

from functools import partial
import itertools
import multiprocessing as mp

from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LinearRegression

from ..utils import generate_ranges

class MultivariatePolynomialInterpolation:
    """
    Multivariate polynomial interpolator

    Parameters
    ----------
    X: list or ndarray, (N,4)
       [Teff, logg, vt, EW]
    Y: list or ndarray, (N,1)
       [Fe/H]
    degree: int
       degree of polynomial 
        
    """

    def __init__(self, X, Y, degree):
        self.X = X
        self.Y = Y
        self.degree = degree

    def fit(self, **kargs):
        """
        Compute interpolation.

        Parameters
        ----------
        **kargs : dict
            args into sklearn PolynomicalFeatures and LinearRegression

        Returns
        -------
        model : sklearn.pipeline.Pipeline
            interpolated model

        """
        #X = self.hyper_surface[:,[0,1,3,4]]
        #Y = self.hyper_surface[:,2]
        model = make_pipeline(PolynomialFeatures(degree=self.degree, **kargs), LinearRegression(**kargs))
        model.fit(self.X, self.Y)
        return model

def full_mul_poly(powers, coeffs, intercept, s):
    """
    Generate complete polynomial as a function of equivalent width
    """
    r0 = s**powers
    r1 = np.prod(r0, axis=1)
    r2 = coeffs*r1
    r = np.sum(r2)
    r = r + intercept
    return r

def solve_poly(a, feh, th_ew):
    coeffs = a.as_poly().all_coeffs()
    coeffs[-1] = coeffs[-1]-feh
    r = np.roots(coeffs)
    real = r.real[abs(r.imag)<1e-5]
    if len(real) > 0:
        result = r.real[abs(r.imag)<1e-5] # where I chose 1-e5 as a threshold
        return result.flat[np.abs(result - th_ew).argmin()]
    else:
        return np.nan

def worker(params, ewlibpath, idx, oneline_model, cal):
    hdf = pd.HDFStore(ewlibpath, 'r')
    Teff, logg, feh, vt = params
    hname = 'blue/rezzeddine/share/grid2/{0:d}K/logg{1:s}/z{2:s}/vt{3:s}'.format(int(Teff),
             format(logg, ".2f"), format(feh, ".2f"), format(vt, ".2f"))
    try:
        single_df = hdf[hname]
    except KeyError:
        return False*np.ones(6)
    result = [Teff, round(logg,1), feh, vt]
    target = single_df.iloc[idx]
    #print(target)
    #for oneline_model in oneline_models:
    s = Array([Teff, logg, vt, e])
    a = full_mul_poly(oneline_model[0].powers_, oneline_model[1].coef_, oneline_model[1].intercept_, s)
    th_ew = round(np.float64(target[cal+'_EW']), 2)
    ewdiff = round(solve_poly(a,feh,th_ew)-th_ew, 2)
    result = np.append(result, [th_ew, ewdiff])
    result = list(result)
    hdf.close()
    return result

def ewdiff(line, stellar_type, ewlib_path, oneline_model, cal, nprocessor=4):
    """
    Function to get equivalent width difference between interpolation and theoratical calculation per line and per stellar type
    Return pandas.Dataframe
    ------
    line: string, taking the following format: 'wavelength(AA)_excitationpotential(ev)_ion(FeI of FeII)'
    stellar_type: string, taking the following format 'spectral type/stellar size description/metalicity description'. Spectral-
                  type options are F,G,K,M; Stellar size description options are dwarf, subgiants and giants; Metalicity description
                  are metal_rich, metal_poor, very_metal_poor
    ewlib_path: path of theoretical results
    poly_path: parent path of multivariate polynomial interpolator (general curve of growth)
    """
    Teff_range, logg_range, feh_range = generate_ranges(stellar_type)

    t_step = 50
    g_step = 0.1
    f_step = 0.5

    paramlist = list(itertools.product(np.arange(Teff_range[0], Teff_range[1]+t_step, t_step),
                                   np.arange(logg_range[0], logg_range[1]+g_step, g_step),
                                   np.arange(feh_range[0], feh_range[1]+f_step, f_step),
                                   np.arange(0.5, 3.5, 0.5)))

    np.random.seed(121)
    random_i = np.random.choice(np.shape(paramlist)[0], 4000)
    argsort = np.argsort(random_i)
    paramlist = np.array(paramlist)[random_i[argsort]]

    hdf = pd.HDFStore(ewlib_path, 'r')

    #inner function to obtain the first idx of line in the atmosphere model file
    def get_first_idx():
        wl = float(line.split("_")[0])
        ep = float(line.split("_")[1])
        ele = line.split("_")[2]
        for Teff in np.arange(Teff_range[0], Teff_range[1]+t_step, t_step):
            for logg in np.arange(logg_range[0], logg_range[1]+g_step, g_step):
                for feh in np.arange(feh_range[0], feh_range[1]+f_step, f_step):
                    for vt in np.arange(0.5,3.5, 0.5):
                        hname = 'blue/rezzeddine/share/grid2/{0:d}K/logg{1:s}/z{2:s}/vt{3:s}'.format(int(Teff),
                                 format(logg, ".2f"), format(feh, ".2f"), format(vt, ".2f"))
                        try:
                            single_df = hdf[hname]
                        except KeyError:
                            continue
                        idx = np.where((single_df.element == ele) & \
                            np.isclose(single_df.th_wavelength, wl, atol=0.025) &\
                            np.isclose(single_df.th_EP, ep, atol=0.02))[0]
                        if len(idx) > 1:
                            idx = idx[0]
                        hdf.close()
                        return idx

    idx = get_first_idx()

    #multi-loop over the ew libary to obtain lines in all atmosphere model
    results = []
    p = mp.Pool(nprocessor)
    for result in p.map(partial(worker, ewlibpath=ewlib_path, idx=idx, oneline_model=oneline_model, cal=cal),
                        paramlist):
        ##print(result)
        try:
            if not sum(result):
                #print('terminating')
                p.terminate()
                continue
        except ValueError:
           print(result)
        results.append(result)

    p.close()
    p.join()

    hdf.close()
    column_names = ["Teff", "logg", "feh", "vt", "EW_"+cal, "delta_"+cal]
    final_df = pd.DataFrame(results, columns=column_names)
    print("Complete {0:s} calculation of delta EW for line {1:s} with stellar type {2:s}!".format(cal, line, stellar_type))
    del results
    return final_df
