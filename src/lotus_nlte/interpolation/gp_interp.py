#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 13 21:53:54 2020

@author: yangyangli
"""
import numpy as np

import torch
import gpytorch
from ..utils import generate_ranges

#import time

class GPRegressionModel(gpytorch.models.ExactGP):
    
    def __init__(self, train_x, train_y, likelihood, bounds):
        super(GPRegressionModel, self).__init__(train_x, train_y, likelihood)

        # SKI requires a grid size hyperparameter. This util can help with that
        grid_size = gpytorch.utils.grid.choose_grid_size(train_x)

        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(
            gpytorch.kernels.GridInterpolationKernel(
                gpytorch.kernels.RBFKernel(), grid_size=grid_size, num_dims=4, grid_bounds=bounds
            )
        )

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

class GPInterpolation:
    """
    Gaussian Process regression interpolator (Not fully implemented yet)

    Parameters
    ----------
    x: list or ndarray, (N,4)
        [Teff, logg, vt, EW]
    y: list or ndarray, (N,1)
       [Fe/H]
    stellar_type: str
        The stellar type of your star, like:
                {spectral type, e.g. F, G, K}/{giant or subgiant or dwarf}/{metal_rich or metal_poor or very_metal_poor}
        or
        the estimation of your atmospheric parameters in such form:
                {T_low}_{T_high}/{logg_low}_{logg_high}/{[Fe/H]_low}_{[Fe/H]_high}
    training_iter: int
        number of training iterations
        
    """
    import torch
    def __init__(self, x, y, stellar_type, training_iter=50):
        self.train_x = torch.from_numpy(x).to(torch.float)
        self.train_y = torch.from_numpy(y).to(torch.float)
        self.bounds = list(generate_ranges(stellar_type)[:2])
        self.bounds.append([0.5, 3.0])
        self.bounds.append([max(0, x[:,3].min()-10), x[:,3].max()+10])
        self.training_iter = 50
        self._model = None
        self._likelihood = None

        idx = np.random.choice(range(np.shape(x)[0]), int(np.floor(np.shape(x)[0]*0.3)))
        self.test_x = torch.from_numpy(x[idx,:]).to(torch.float)
        self.test_y = torch.from_numpy(y[idx]).to(torch.float)
        self.th_met = y[idx]

    def train(self):
        likelihood = gpytorch.likelihoods.GaussianLikelihood()
        model = GPRegressionModel(self.train_x, self.train_y, likelihood, self.bounds)

        #train the model
        model.train()
        likelihood.train()

        # Use the adam optimizer
        optimizer = torch.optim.Adam([
            {'params': model.parameters()},  # Includes GaussianLikelihood parameters
            ], lr=0.1)

        # "Loss" for GPs - the marginal log likelihood
        mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)


        for i in range(self.training_iter):
            optimizer.zero_grad()
            output = model(self.train_x)
            loss = -mll(output, self.train_y)
            loss.backward()
            #print('Iter %d/%d - Loss: %.3f' % (i + 1, training_iterations, loss.item()))
            optimizer.step()
        self._model = model
        self._likelihood = likelihood

    def test(self):
        self._model.eval()
        self._likelihood.eval()

        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            prediction = self._likelihood(self._model(self.test_x))
            mean = prediction.mean
            var = prediction.variance

        delta_mean = mean.tolist()-self.th_met
        return np.array([delta_mean.tolist(), np.sqrt(var.tolist()).tolist()])

    def predict(self, data):
        self._model.eval()
        self._likelihood.eval()

        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            #start_time = time.time()
            prediction = self._likelihood(self._model(data))
            mean = prediction.mean
            # get covariance matrix
            var = prediction.variance
            #fast_time_with_cache = time.time() - start_time

        #print('Time to compute mean + variances (cache): {:.2f}s'.format(fast_time_with_cache))

        return [mean.tolist()[0], np.sqrt(var.tolist()).tolist()[0]]
