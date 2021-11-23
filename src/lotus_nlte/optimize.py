#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 18 20:37:47 2020

@author: yangyangli
"""
import numpy as np
import pandas as pd
from numdifftools import Jacobian, Hessian
from scipy.optimize import differential_evolution, shgo
#from gcog import GCOG, MultiGCOG

from sympy import Array
from sympy.abc import e

from .utils import slope_measure
from .interpolation.multipoly_interp import full_mul_poly, solve_poly


class StellarOptimization:
    """
    Base class for stellar parameter opimization
    
    Parameters:
        mgcog: lotus_nlte.gcog.MultiGCOG
        physicaltol: torelence for stopping the iteration of optimization
    
    """

    def __init__(self, mgcog, physicaltol=1e-5):
        self.mgcog = mgcog
        if len(np.where(np.array(mgcog.obs_ele) =="FeII")[0]) < 2:
            self.limited_feii = True
        else:
            self.limited_feii = False
        if isinstance(physicaltol, float):
            self.physicaltol = physicaltol * np.ones(3)
        else:
            self.physicaltol = physicaltol
        self._xk_prev = None
        self._yk_prev = None
        self._ykerr_prev = None
        self.abunds = None
        self._sk_prev = np.empty(3)
        self.callback = self._callback

    def checkintols(self, xk):
        """
        Check if proposal match with the convergence condition

        Parameters
        ----------
        xk : list
            proposal, [Teff, logg, vt]

        Returns
        -------
        bool
            Whether to stop the optimization

        """
        generated_met = []
        if self.mgcog.interp_method != "SKIGP":
            for i in range(len(self.mgcog.models)):
                result = self.generate_met(self.mgcog.models[i], xk[0],
                                             xk[1], xk[2],
                                             self.mgcog.obs_ew[i])
                generated_met.append(result[0])

            if self.limited_feii:
                Achi1, AREW1 = self.obs_calculation(generated_met, self.mgcog.obs_ele,
                                        self.mgcog.obs_ew, self.mgcog.obs_wavelength,
                                        self.mgcog.obs_ep)
                if self._xk_prev is not None:
                    if (np.abs(Achi1[0])<=self.physicaltol[0] and np.abs(AREW1[0])<=self.physicaltol[1]):
                        return True

                self._xk_prev = xk
                self._yk_prev = np.array(generated_met) + 7.46
                self._sk_prev = np.array([Achi1, AREW1])
                return False

            generated_met_err = None

        if self.mgcog.interp_method == "SKIGP":
            generated_met_err = []
            pred_xs = []
            for i in range(len(self.mgcog.obs_ew)):
                pred_x = torch.from_numpy(np.array([[xk[0],
                                        xk[1], xk[2],
                                        self.mgcog.obs_ew[i]]])).to(torch.float)
                pred_xs.append(pred_x)

            self.mgcog.models.eval()
            self.mgcog.likelihoods.eval()
            with torch.no_grad(), gpytorch.settings.fast_pred_var():
                predictions = self.mgcog.likelihoods(*self.mgcog.models(*pred_xs))

            for p in predictions:
                generated_met.append(p.mean.tolist()[0])
                generated_met_err.append(p.variance.tolist()[0]**0.5)


        Achi1, AREW1, dFe = self.obs_calculation(generated_met, self.mgcog.obs_ele,
                                         self.mgcog.obs_ew, self.mgcog.obs_wavelength,
                                         self.mgcog.obs_ep, generated_met_err)
        target_f = (Achi1[0]/Achi1[1]**0.5)**2+(AREW1[0]/Achi1[1]**0.5)**2 +(dFe[0]/dFe[1])**2.
        if self._xk_prev is not None:
            if (np.abs(Achi1[0])<=self.physicaltol[0] and np.abs(AREW1[0])<=self.physicaltol[1] and np.abs(dFe[0])<=self.physicaltol[2]) or target_f<=min(self.physicaltol):
                return True

        self._xk_prev = xk
        self._yk_prev = np.array(generated_met) + 7.46
        if self.mgcog.interp_method == "SKIGP":
            self._ykerr_prev = np.array(generated_met_err)
        self._sk_prev = np.array([Achi1, AREW1, dFe])
        return False

    def _callback(self, xk):
        flag = self.checkintols(xk)
        return flag

    def _bounds(self):
        if not self.limited_feii:
            self.bounds = [(4000, 6850), (0.5, 5.0), (0.5, 3.0)]
        else:
            self.bounds = [(4000, 6850), (0.0, 5.0), (0.5, 3.0), (-3.5, 0.5)]

    def generate_met(self, model, teff, logg, vt, ew):
        """
        Generate metalicity for single line given the interpolated model and proposed
        stellar paramters and its observed EW.

        Parameters
        ----------
        model : sklearn.pipeline.Pipeline
            interpolated model
        teff : int or float
            Teff
        logg : int or float
            logg
        vt : int or float
            micro-turbulence velocity
        ew : int or float
            observed EW

        Returns
        -------
        predict_y : number
            Predicted metalicity given by Teff, logg, vt and EW

        """
        if self.mgcog.interp_method == "[2-5]":
            #This function is for those with enough feii lines
            if np.ndim(teff) == 0 and np.ndim(logg) == 0 and np.ndim(vt) == 0 and np.ndim(ew) == 0:
                teff, logg, vt, ew = [teff], [logg], [vt], [ew]

            predict_x0, predict_x1, predict_x2, predict_x3 = np.meshgrid(teff, logg, vt, ew)
            predict_x = np.concatenate((predict_x0.reshape(-1, 1),
                            predict_x1.reshape(-1, 1),
                            predict_x2.reshape(-1, 1),
                            predict_x3.reshape(-1, 1)),
                           axis=1)
            #predict_x_ = poly.fit_transform(predict_x)
            predict_y = model.predict(predict_x)

        if "RBF" in self.mgcog.interp_method:
            predcit_x = np.array([[teff, logg, vt, ew]])
            predict_y = model(predcit_x)

        return predict_y
        #    predict_x = torch.from_numpy(np.array([[teff, logg, vt, ew]])).to(torch.float)
        #    predict_ys = model.predict(predict_x)
        #    return predict_ys

    def generate_ew(self, model, teff, logg, vt, feh, ew, **kargs):
        #This function is for those with not enough feii lines (no. of feii < 2)
        s = Array([teff, logg, vt, e])
        a = full_mul_poly(model[0].powers_, model[1].coef_, model[1].intercept_, s)
        predict_ew = solve_poly(a, feh, ew)
        return predict_ew

    def generate_ewerror(self, line, ewdiff_df):
        mean = ewdiff_df["delta_"+self.mgcog.cal].mean()
        return mean

    def obs_calculation(self, gen_met, obs_ele, obs_ew, obs_wavelength, obs_ep, gen_met_err=None):
        """
        Calculate observed slope between excitation potentials and abundances, 
        the slope between reduced equivalent widths and abundances and the difference 
        between abundances of FeI and FeII

        Parameters
        ----------
        gen_met : list
            Predicted metalicity for every lines
        obs_ele : ndarray
            Species for the observed lines
        obs_ew : ndarray
            EWs for the observed lines
        obs_wavelength : ndarray
            Wavelength for the observed lines
        obs_ep : ndarray
            Excitation potential for the observed lines
        gen_met_err : ndarray, optional
            Not implemented yet. The default is None.

        Returns
        -------
        List
            [the slope between excitation potentials and abundances, 
            the slope between reduced equivalent widths and abundances,
            the difference between abundances of FeI and FeII]

        """

        idx_fei = np.where(np.array(obs_ele) =="FeI")
        idx_feii = np.where(np.array(obs_ele) =="FeII")

        abunds = np.array(gen_met) + 7.46
        REWs = np.log10(1e-3*np.array(obs_ew)/np.array(obs_wavelength))
        chis = np.array(obs_ep)
        if gen_met_err == None:
            abunds_err = None
        else:
            abunds_err = np.array(gen_met_err)[idx_fei]
        popt_Achi1, pcov_Achi1 = slope_measure(chis[idx_fei], abunds[idx_fei], abunds_err)
        popt_AREW1, pcov_AREW1 = slope_measure(REWs[idx_fei], abunds[idx_fei], abunds_err)
        #popt_Achi2, pcov_Achi2 = slope_measure(chis[idx_feii], abunds[idx_feii])
        #popt_AREW2, pcov_AREW2 = slope_measure(REWs[idx_feii], abunds[idx_feii])
        if not self.limited_feii :
            dFe = np.mean(abunds[idx_fei]) - np.mean(abunds[idx_feii])
            deltadFe = np.sqrt((np.std(abunds[idx_fei])/np.shape(idx_fei)[1]**0.5)**2 + (np.std(abunds[idx_feii])/np.shape(idx_feii)[1]**0.5)**2)

            return [popt_Achi1[0], pcov_Achi1[0][0]], [popt_AREW1[0], pcov_AREW1[0][0]], [dFe, deltadFe]

        else:
            return [popt_Achi1[0], pcov_Achi1[0][0]], [popt_AREW1[0], pcov_AREW1[0][0]]

    def minimisation_function(self, stellar_parameters):
        """
        Calculate objective function given proposed stellar parameters

        Parameters
        ----------
        stellar_parameters : list
            [Teff, logg, vt]

        Returns
        -------
        Number
            objective function

        """
        generated_met = []
        if self.mgcog.interp_method != "SKIGP":
            for i in range(len(self.mgcog.models)):
                result = self.generate_met(self.mgcog.models[i], stellar_parameters[0],
                                        stellar_parameters[1], stellar_parameters[2],
                                        self.mgcog.obs_ew[i])

                generated_met.append(result[0])

            if self.limited_feii:
                generated_ew = []
                generated_ewerror = []
                for i in range(len(self.mgcog.models)):
                    line = str(round(self.mgcog.obs_wavelength[i], 2)) \
                        +"_"+ str(round(self.mgcog.obs_ep[i], 2)) +"_" + self.mgcog.obs_ele[i]
                    ewdiff_file = self.mgcog.ewdiffpath + self.mgcog.stellar_type + "/" + line + ".csv"
                    ewdiff_df = pd.read_csv(ewdiff_file)
                    generated_ew.append(self.generate_ew(self.mgcog.models[i], stellar_parameters[0],
                                         stellar_parameters[1], stellar_parameters[2],
                                         stellar_parameters[3], self.mgcog.obs_ew[i]))
                    generated_ewerror.append(self.generate_ewerror(line, ewdiff_df))
                Achi1, AREW1 = self.obs_calculation(generated_met, self.mgcog.obs_ele, self.mgcog.obs_ew, self.mgcog.obs_wavelength, self.mgcog.obs_ep)
                generated_ew = np.array(generated_ew)
                generated_ewerror = np.array(generated_ewerror)

                return (Achi1[0]/Achi1[1]**0.5)**2+(AREW1[0]/Achi1[1]**0.5)**2+np.sum((generated_ew/generated_ewerror)**2)

            Achi1, AREW1, dFe = self.obs_calculation(generated_met, self.mgcog.obs_ele, self.mgcog.obs_ew, self.mgcog.obs_wavelength, self.mgcog.obs_ep)
            return (Achi1[0]/Achi1[1]**0.5)**2+(AREW1[0]/Achi1[1]**0.5)**2+(dFe[0]/dFe[1])**2.

        if self.mgcog.interp_method == "SKIGP":
            generated_met_err = []
            pred_xs = []
            for i in range(len(self.mgcog.obs_ew)):
                pred_x = torch.from_numpy(np.array([[stellar_parameters[0],
                                        stellar_parameters[1], stellar_parameters[2],
                                        self.mgcog.obs_ew[i]]])).to(torch.float)
                pred_xs.append(pred_x)

            self.mgcog.models.eval()
            self.mgcog.likelihoods.eval()
            with torch.no_grad(), gpytorch.settings.fast_pred_var():
                predictions = self.mgcog.likelihoods(*self.mgcog.models(*pred_xs))

            for p in predictions:
                generated_met.append(p.mean.tolist()[0])
                generated_met_err.append(p.variance.tolist()[0]**0.5)

            Achi1, AREW1, dFe = self.obs_calculation(generated_met, self.mgcog.obs_ele,
                                                     self.mgcog.obs_ew, self.mgcog.obs_wavelength,
                                                     self.mgcog.obs_ep, generated_met_err)
            return (Achi1[0]/Achi1[1]**0.5)**2+(AREW1[0]/Achi1[1]**0.5)**2+(dFe[0]/dFe[1])**2.


    def _minimisation_function_wrapper(self, stellar_parameters):
            return self.minimisation_function(stellar_parameters)

    def fun_der(self, stellar_parameters):
        """
        Calculate first derivatives

        Parameters
        ----------
        stellar_parameters : list
            stellar parameter at the minimization of function

        Returns
        -------
        ndarray
            Jacobian function

        """
        return Jacobian(lambda x: self.minimisation_function(x))(stellar_parameters).ravel()

    def fun_hess(self, stellar_parameters, **kargs):
        """
        Calculate second derivatives

        Parameters
        ----------
        stellar_parameters : list
            stellar parameter at the minimization of function
        **kargs : dict
            args feed to Hessian matrix functon

        Returns
        -------
        ndarray
            Hessian matrix

        """
        return Hessian(lambda x: self.minimisation_function(x), **kargs)(stellar_parameters)

    def uncertainty(self, result):
        """
        Uncertainty estimation according given the Hessian Matrix

        Parameters
        ----------
        result : dict
            result containing optimization result


        """
        from numpy import linalg
        if self.mgcog.interp_method != "SKIGP":
            if "RBF" in self.mgcog.interp_method:
                base_step = np.array([50, 0.1, 0.5])
                stderr = np.sqrt(np.diag(linalg.inv(self.fun_hess(result.x, base_step=base_step))))
            else:
                stderr = np.sqrt(np.diag(linalg.inv(self.fun_hess(result.x))))
        if self.mgcog.interp_method == "SKIGP":
            r = np.diff(self.bounds).reshape((3,))
            step_nom = np.log1p(np.abs(result.x)/r).clip(min=1.0)
            base_step = np.array([50, 0.1, 0.5])
            try:
                stderr = np.sqrt(np.diag(linalg.inv(self.fun_hess(result.x, base_step=base_step,
                                                        num_steps=3,step_ratio=1, step_nom=step_nom))))
            except RuntimeError:
                try:
                    stderr = np.sqrt(np.diag(linalg.inv(self.fun_hess(result.x, method='forward',
                                                            base_step=base_step, num_steps=3,
                                                              step_ratio=1, step_nom=step_nom))))
                except RuntimeError:
                    try:
                        stderr = np.sqrt(np.diag(linalg.inv(self.fun_hess(result.x, method='backward',
                                                            base_step=base_step, num_steps=3,
                                                              step_ratio=1, step_nom=step_nom))))
                    except RuntimeError:
                        stderr = np.ones(3) * np.nan
        result.stderrs = np.ones(len(result.x))
        for i,p in enumerate(result.x):
            result.stderrs[i] = stderr[i]
        if self.mgcog.interp_method == "SKIGP":
            result.stderrs = np.where(np.isnan(result.stderrs), np.inf, result.stderrs)

    def optimize(self, method_func, callback=None, **kargs):
        """
        Optimization wrapper

        Parameters
        ----------
        method_func : func
            scipy.optimize.differential_evolution or scipy.optimization.shgo
        callback : func, None
            Callback function. The default is None.
        **kargs : dict
            args feed into method_func

        Returns
        -------
        func
            optimization progress

        """
        cb = callback
        return method_func(self.minimisation_function, callback=cb, **kargs)

    def _set_up_meshgrids(self, steps=[50, 0.1, 0.1]):
        """
        Setup grids given the stellar type and calculate objective function at these
        grid points

        Parameters
        ----------
        steps : list, optional
            steps for Teff, logg, vt. The default is [50, 0.1, 0.1].

        Returns
        -------
        list
            Grid points in the shape of (N,4)

        """
        from multiprocessing import Pool
        Teff_v = np.arange(self.bounds[0][0], self.bounds[0][1], steps[0])
        logg_v = np.arange(self.bounds[1][0], self.bounds[1][1], steps[1])
        vt_v = np.arange(self.bounds[2][0], self.bounds[2][1], steps[2])
        if self.limited_feii:
            feh_v = np.arange(self.bounds[2][0], self.bounds[2][1], 0.1)
            Teff_grid, logg_grid, vt_grid, feh_grid  =  np.meshgrid(Teff_v, logg_v, vt_v, feh_v, indexing="ij")
            positions = np.vstack([Teff_grid.ravel(), logg_grid.ravel(), vt_grid.ravel(), feh_grid.ravel()]).T

            with Pool() as p :
                f_grid = np.array(list(p.map(self._minimisation_function_wrapper,positions)))

            new_f_grid = f_grid.reshape(Teff_grid.shape)

            return [Teff_grid, logg_grid, vt_grid, feh_grid, new_f_grid]

        Teff_grid, logg_grid, vt_grid  =  np.meshgrid(Teff_v, logg_v, vt_v, indexing="ij")
        positions = np.vstack([Teff_grid.ravel(), logg_grid.ravel(), vt_grid.ravel()]).T

        with Pool() as p :
            f_grid = np.array(list(p.map(self._minimisation_function_wrapper,positions)))

        new_f_grid = f_grid.reshape(Teff_grid.shape)

        return [Teff_grid, logg_grid, vt_grid, new_f_grid]

    def log_likelihood(self,theta):
        """
        Calculate loglikehood function at the proposed stellar parameters

        Parameters
        ----------
        theta : list or ndarray
            [Teff, logg, vt]

        Returns
        -------
        Number
            loglikehood function

        """
        stellar_parameters = theta[:3]
        log_f = theta[-1]#.detach().numpy()
        generated_met = []
        if self.mgcog.interp_method != "SKIGP":
            for i in range(len(self.mgcog.models)):
                generated_met.append(self.generate_met(self.mgcog.models[i], stellar_parameters[0],
                                         stellar_parameters[1], stellar_parameters[2],
                                         self.mgcog.obs_ew[i])[0])

            Achi1, AREW1, dFe = self.obs_calculation(generated_met, self.mgcog.obs_ele,
                                                 self.mgcog.obs_ew, self.mgcog.obs_wavelength,
                                                 self.mgcog.obs_ep)

        if self.mgcog.interp_method == "SKIGP":
            generated_met_err = []
            pred_xs = []
            for i in range(len(self.mgcog.obs_ew)):
                pred_x = torch.from_numpy(np.array([[stellar_parameters[0],
                                        stellar_parameters[1], stellar_parameters[2],
                                        self.mgcog.obs_ew[i]]])).to(torch.float)
                pred_xs.append(pred_x)

            self.mgcog.models.eval()
            self.mgcog.likelihoods.eval()
            with torch.no_grad(), gpytorch.settings.fast_pred_var():
                predictions = self.mgcog.likelihoods(*self.mgcog.models(*pred_xs))

            for p in predictions:
                generated_met.append(p.mean.tolist()[0])
                generated_met_err.append(p.variance.tolist()[0]**0.5)

            Achi1, AREW1, dFe = self.obs_calculation(generated_met, self.mgcog.obs_ele,
                                                 self.mgcog.obs_ew, self.mgcog.obs_wavelength,
                                                 self.mgcog.obs_ep, generated_met_err)
        #To remove the explosion of gradient when doing MCMC
        if np.isinf(Achi1[1]):
            Achi1[1] = 0
        if np.isinf(AREW1[1]):
            AREW1[1] = 0
        sigma2_Achi1 = Achi1[1] + Achi1[0] **2 * np.exp(2 * log_f)
        sigma2_AREW1 = AREW1[1] + AREW1[0] **2 * np.exp(2 * log_f)
        sigma2_dFe = dFe[1] + dFe[0] **2 * np.exp(2 * log_f)
        sigma2s = np.array([sigma2_Achi1, sigma2_AREW1, sigma2_dFe])
        models = np.array([Achi1[0], AREW1[0], dFe[0]])

        return -0.5 * np.sum(models ** 2 / sigma2s + np.log(sigma2s))

class DiffEvoStellarOptimization(StellarOptimization):
    """
    Optimizer using Differential Evolution Algorithm, wrapped from 
    scipy.optimize.differential_evolution
    
    Parameters:
        bounds: list of tuples, None
        if None, use the boundary of our grid
    
    """
    def __init__(self, mgcog, bounds=None, physicaltol=1e-5):
        super().__init__(mgcog, physicaltol)
        #self.model = model
        if bounds == None:
            self._bounds()
        else:
            self.bounds = bounds
        self.callback = self._callback

    def _callback(self, xk, convergence):
        flag = self.checkintols(xk)
        flag2 = convergence >= 1
        return flag or flag2

    def optimize(self, method_func=differential_evolution, **kargs):
        if self.limited_feii:
            results  = {"ScipyOptimizeResult": method_func(self.minimisation_function,
                           callback=self.callback, bounds=self.bounds, **kargs),
                    "dAdchi": self._sk_prev[0], "dAdREW":self._sk_prev[1]}
            results['var_names'] = ['$T_{eff}$', 'logg', '$V_{mic}$', '[Fe/H]']
        else:
            results  = {"ScipyOptimizeResult": method_func(self.minimisation_function,
                           callback=self.callback, bounds=self.bounds, **kargs),
                    "dAdchi": self._sk_prev[0], "dAdREW":self._sk_prev[1],
                    "deltaFe": self._sk_prev[2]}
            results['var_names'] = ['$T_{eff}$', 'logg', '$V_{mic}$']
        self.uncertainty(results["ScipyOptimizeResult"])
        self.abunds = self._yk_prev
        idx_fei = np.where(np.array(self.mgcog.obs_ele) =="FeI")
        results['stellarpars'] = {"Teff": [results['ScipyOptimizeResult'].x[0], results['ScipyOptimizeResult'].stderrs[0]],
              "logg": [results['ScipyOptimizeResult'].x[1], results['ScipyOptimizeResult'].stderrs[1]],
              "feh":[np.mean(np.array(self.abunds)[idx_fei]-7.46), np.std(np.array(self.abunds)[idx_fei]-7.46)],
              "Vmic":[results['ScipyOptimizeResult'].x[2], results['ScipyOptimizeResult'].stderrs[2]]}
        return results

class ShgoStellarOptimization(StellarOptimization):
    """
    Optimizer using SHG optimization, wrapped from 
    scipy.optimize.shgo
    
    Parameters:
        bounds: list of tuples, None
        if None, use the boundary of our grid
    
    """
    def __init__(self, mgcog, bounds=None, physicaltol=1e-5):
        super().__init__(mgcog, physicaltol)
        #self.model = model
        if bounds == None:
            self._bounds()
        else:
            self.bounds = bounds
        self.callback = self._callback


    def _callback(self, xk):
        flag = self.checkintols(xk)
        return flag

    def optimize(self, method_func=shgo, **kargs):
        if self.limited_feii:
            results  = {"ScipyOptimizeResult": method_func(self.minimisation_function,
                           callback=self.callback, bounds=self.bounds, **kargs),
                    "dAdchi": self._sk_prev[0], "dAdREW":self._sk_prev[1]}
            results['var_names'] = ['$T_{eff}$', 'logg', '$V_{mic}$', '[Fe/H]']
        elif self.mgcog.interp_method != "SKIGP":
            results  = {"ScipyOptimizeResult": method_func(self.minimisation_function,
                           callback=self.callback, bounds=self.bounds, **kargs),
                    "dAdchi": self._sk_prev[0], "dAdREW":self._sk_prev[1],
                    "deltaFe": self._sk_prev[2]}
            results['var_names'] = ['$T_{eff}$', 'logg', '$V_{mic}$']

        elif self.mgcog.interp_method == "SKIGP":
            from utils import check_on_the_edge
            from scipy import optimize
            results  = {"ScipyOptimizeResult": method_func(self.minimisation_function,
                           callback=self.callback, bounds=self.bounds, **kargs),
                    "dAdchi": self._sk_prev[0], "dAdREW":self._sk_prev[1],
                    "deltaFe": self._sk_prev[2]}
            final_x, final_fun = check_on_the_edge(results["ScipyOptimizeResult"].xl, results["ScipyOptimizeResult"].funl, self.bounds)
            results["ScipyOptimizeResult"].x = final_x
            results["ScipyOptimizeResult"].fun = final_fun
            results['var_names'] = ['$T_{eff}$', 'logg', '$V_{mic}$']
            self.uncertainty(results["ScipyOptimizeResult"])

            print(results["ScipyOptimizeResult"])
            local_bounds = tuple(np.transpose([[max(results["ScipyOptimizeResult"].x[i]-results["ScipyOptimizeResult"].stderrs[i], self.bounds[i][0]),
                                            min(results["ScipyOptimizeResult"].x[i]+results["ScipyOptimizeResult"].stderrs[i], self.bounds[i][1])] for i in range(3)]).tolist())
            try:
                res_local = optimize.least_squares(self.minimisation_function, results["ScipyOptimizeResult"].x,
                              bounds=local_bounds, loss='soft_l1', f_scale=0.1)
                print(res_local)
                results["ScipyOptimizeResult"].x = res_local.x
                results["ScipyOptimizeResult"].fun = res_local.fun
                self.uncertainty(results["ScipyOptimizeResult"])
            except Exception as e:
                import traceback
                import logging
                print(e)
                logging.error(traceback.format_exc())
        self.uncertainty(results["ScipyOptimizeResult"])
        self.abunds = self._yk_prev
        idx_fei = np.where(np.array(self.mgcog.obs_ele) =="FeI")
        results['stellarpars'] = {"Teff": [results['ScipyOptimizeResult'].x[0], results['ScipyOptimizeResult'].stderrs[0]],
              "logg": [results['ScipyOptimizeResult'].x[1], results['ScipyOptimizeResult'].stderrs[1]],
              "feh":[np.mean(np.array(self.abunds)[idx_fei]-7.46), np.std(np.array(self.abunds)[idx_fei]-7.46)],
              "Vmic":[results['ScipyOptimizeResult'].x[2], results['ScipyOptimizeResult'].stderrs[2]]}
        return results
