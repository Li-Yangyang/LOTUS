#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 18 15:31:41 2021

@author: yangyangli
"""

from rbf.interpolate import KNearestRBFInterpolant

class RBFRegressionInterpolation:
    """
    RBF regression interpolator (Not fully implemented yet)

    Parameters
    ----------
    X: list or ndarray, (N,4)
        [Teff, logg, vt, EW]
    Y: list or ndarray, (N,1)
       [Fe/H]
    kernel: str
        kernel type
    k: neareset number of grid points considering into interpolatin
    model: KNearestRBFInterpolant instance
        if None, you need to fit first;
        if not None, you can test your model with test data
        
    """
    def __init__(self, X, Y, kernel, k):
        self.X = X
        self.Y = Y
        self.kernel = kernel
        self.k = k
        self.model = None

    def fit(self, **kargs):
        self.model = KNearestRBFInterpolant(self.X, self.Y, k=self.k, phi=self.kernel, order=0)
        return self.model

    def test(self, test_x, test_y):
        return self.model(test_x) - test_y