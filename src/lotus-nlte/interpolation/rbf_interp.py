#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 18 15:31:41 2021

@author: yangyangli
"""

from rbf.interpolate import KNearestRBFInterpolant

class RBFRegressionInterpolation:
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